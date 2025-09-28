#!/usr/bin/env python3
"""
Convert Qwen3ForCausalLM to Qwen3ForSequenceClassification

This script converts a Qwen3 Causal Language Model to a Sequence Classification model
by extracting the "yes" and "no" token weights from the lm_head and creating a binary
classifier based on their difference.

Based on the methodology described in example/seq-cls.md:
1. Load Qwen3ForCausalLM and extract lm_head weight matrix
2. Extract rows for "yes" and "no" tokens
3. Subtract "no" row from "yes" row (equivalent to 2-classes + softmax)
4. Load model using Qwen3ForSequenceClassification with num_labels=1
5. Replace the score layer with the computed vector

Copyright 2025, MIT License
"""

import argparse
import os
import torch
from pathlib import Path
from transformers import (
    AutoTokenizer,
    Qwen2ForCausalLM,  # Note: Qwen3 uses Qwen2 architecture
    Qwen2ForSequenceClassification,
    AutoConfig,
)
from huggingface_hub import HfApi


def convert_qwen3_to_sequence_classifier(
    model_name: str,
    output_dir: str,
    yes_token: str = "yes",
    no_token: str = "no",
    push_to_hub: bool = False,
    hub_repo_id: str = None,
    verify_conversion: bool = True,
) -> None:
    """
    Convert a Qwen3ForCausalLM model to Qwen3ForSequenceClassification.
    
    Args:
        model_name: Name or path of the source Qwen3 model
        output_dir: Directory to save the converted model
        yes_token: Token representing positive class (default: "yes")
        no_token: Token representing negative class (default: "no")
        push_to_hub: Whether to push the converted model to HuggingFace Hub
        hub_repo_id: Repository ID for HuggingFace Hub upload
        verify_conversion: Whether to verify the conversion with test inputs
    """
    
    print(f"🚀 Converting {model_name} to Sequence Classification model")
    print(f"📁 Output directory: {output_dir}")
    
    # --- Step 1: Load the Causal LM and extract lm_head weights ---
    print(f"\n1️⃣ Loading Causal LM: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    causal_lm = Qwen2ForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        device_map="auto"
    )
    
    # The lm_head is the final linear layer that maps hidden states to vocabulary logits
    lm_head_weights = causal_lm.lm_head.weight
    print(f"   ✅ lm_head weight shape: {lm_head_weights.shape}")  # (vocab_size, hidden_size)
    
    # --- Step 2: Get the token IDs for "yes" and "no" ---
    print(f"\n2️⃣ Finding token IDs for '{yes_token}' and '{no_token}'")
    yes_token_id = tokenizer.convert_tokens_to_ids(yes_token)
    no_token_id = tokenizer.convert_tokens_to_ids(no_token)
    
    if yes_token_id is None or no_token_id is None:
        raise ValueError(f"Could not find token IDs for '{yes_token}' or '{no_token}' in tokenizer")
    
    print(f"   ✅ ID for '{yes_token}': {yes_token_id}")
    print(f"   ✅ ID for '{no_token}': {no_token_id}")
    
    # --- Step 3: Create the classifier vector ---
    print(f"\n3️⃣ Creating classifier vector from lm_head weights")
    # Extract the specific rows (weight vectors) for our target tokens
    yes_vector = lm_head_weights[yes_token_id]
    no_vector = lm_head_weights[no_token_id]
    
    # The new classifier is the difference between the 'yes' and 'no' vectors
    # This is equivalent to computing logit_yes - logit_no for binary classification
    classifier_vector = yes_vector - no_vector
    print(f"   ✅ Shape of classifier vector: {classifier_vector.shape}")
    
    # --- Step 4: Load the model as a Sequence Classifier ---
    print(f"\n4️⃣ Loading Sequence Classification model with num_labels=1")
    # num_labels=1 is key for binary classification represented by a single logit
    seq_cls_model = Qwen2ForSequenceClassification.from_pretrained(
        model_name,
        num_labels=1,
        ignore_mismatched_sizes=True,
        torch_dtype=torch.float16,
        device_map="auto"
    )
    
    # --- Step 5: Replace the classifier's weights ---
    print(f"\n5️⃣ Replacing the classifier weights")
    # The classification head in Qwen is named 'score'. It's a torch.nn.Linear layer.
    # Its weight matrix has shape (num_labels, hidden_size), which is (1, hidden_size) here.
    with torch.no_grad():
        # We need to add a dimension to our vector to match the (1, hidden_size) shape
        seq_cls_model.score.weight.copy_(classifier_vector.unsqueeze(0))
        # Zero out the bias for a clean transfer
        if seq_cls_model.score.bias is not None:
            seq_cls_model.score.bias.zero_()
    
    print("   ✅ Classifier head replaced successfully")
    
    # Update model configuration
    seq_cls_model.config.id2label = {0: "negative", 1: "positive"}
    seq_cls_model.config.label2id = {"negative": 0, "positive": 1}
    seq_cls_model.config.problem_type = "single_label_classification"
    
    # --- Verification: Test the conversion ---
    if verify_conversion:
        print(f"\n🔍 VERIFICATION")
        test_texts = [
            "This is a great example of relevant content.",
            "This text is completely unrelated to the query.",
            "The document perfectly answers the question.",
        ]
        
        for i, text in enumerate(test_texts):
            print(f"\n   Test {i+1}: '{text[:50]}...'")
            inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
            
            # Move inputs to same device as models
            inputs = {k: v.to(seq_cls_model.device) for k, v in inputs.items()}
            
            with torch.no_grad():
                # A. Get logits from the original Causal LM
                outputs_causal = causal_lm(**inputs)
                last_token_logits = outputs_causal.logits[0, -1, :]
                manual_logit_diff = last_token_logits[yes_token_id] - last_token_logits[no_token_id]
                
                # B. Get the single logit from our new Sequence Classification model
                outputs_seq_cls = seq_cls_model(**inputs)
                model_logit = outputs_seq_cls.logits.squeeze()  # Shape is (1,), squeeze to scalar
                
                # Compute probabilities
                classification_prob = torch.sigmoid(model_logit)
                
                print(f"      Manual logit difference: {manual_logit_diff.item():.4f}")
                print(f"      Seq classifier output:   {model_logit.item():.4f}")
                print(f"      Classification prob:     {classification_prob.item():.4f}")
                print(f"      ✅ Match: {torch.allclose(manual_logit_diff, model_logit, atol=1e-3)}")
    
    # --- Save the converted model ---
    print(f"\n💾 Saving converted model to {output_dir}")
    os.makedirs(output_dir, exist_ok=True)
    
    # Save model and tokenizer
    seq_cls_model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    
    # Update tokenizer_config.json with reranker template
    tokenizer_config_path = os.path.join(output_dir, "tokenizer_config.json")
    if os.path.exists(tokenizer_config_path):
        import json
        with open(tokenizer_config_path, "r") as f:
            tokenizer_config = json.load(f)
        
        # Add the official Qwen3 reranker template
        reranker_template = (
            '<|im_start|>system\n'
            'Judge whether the Document meets the requirements based on the Query and the Instruct provided. '
            'Note that the answer can only be "yes" or "no".<|im_end|>\n'
            '<|im_start|>user\n'
            '<Instruct>: {instruction}\n'
            '<Query>: {query}\n'
            '<Document>: {document}<|im_end|>\n'
            '<|im_start|>assistant\n'
            '<think>\n\n</think>\n\n'
        )
        
        tokenizer_config["reranker_template"] = reranker_template
        
        with open(tokenizer_config_path, "w") as f:
            json.dump(tokenizer_config, f, indent=2)
        
        print("   ✅ Added reranker_template to tokenizer_config.json")
    
    # Save conversion metadata
    metadata = {
        "source_model": model_name,
        "conversion_type": "Qwen3ForCausalLM -> Qwen3ForSequenceClassification",
        "yes_token": yes_token,
        "no_token": no_token,
        "yes_token_id": yes_token_id,
        "no_token_id": no_token_id,
        "num_labels": 1,
        "problem_type": "single_label_classification",
        "template_format": "qwen3_reranker",
        "template_added": True
    }
    
    import json
    with open(os.path.join(output_dir, "conversion_metadata.json"), "w") as f:
        json.dump(metadata, f, indent=2)
    
    print("   ✅ Model saved successfully")
    
    # --- Push to Hub if requested ---
    if push_to_hub and hub_repo_id:
        print(f"\n🚀 Pushing to HuggingFace Hub: {hub_repo_id}")
        try:
            api = HfApi()
            api.create_repo(repo_id=hub_repo_id, exist_ok=True)
            api.upload_folder(
                repo_id=hub_repo_id,
                folder_path=output_dir,
                commit_message=f"Convert {model_name} to sequence classification"
            )
            print("   ✅ Successfully pushed to Hub")
        except Exception as e:
            print(f"   ❌ Failed to push to Hub: {e}")
    
    # Clean up memory
    del causal_lm
    del seq_cls_model
    torch.cuda.empty_cache() if torch.cuda.is_available() else None
    
    print(f"\n🎉 Conversion completed successfully!")
    print(f"📁 Converted model saved at: {output_dir}")
    if push_to_hub and hub_repo_id:
        print(f"🔗 Hub URL: https://huggingface.co/{hub_repo_id}")


def main():
    parser = argparse.ArgumentParser(
        description="Convert Qwen3ForCausalLM to Qwen3ForSequenceClassification",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic conversion
  python convert_qwen3_to_seq_cls.py --model Qwen/Qwen3-Reranker-0.6B --output ./qwen3-0.6b-seq-cls

  # Convert with custom tokens
  python convert_qwen3_to_seq_cls.py --model Qwen/Qwen3-Reranker-4B --output ./qwen3-4b-seq-cls --yes-token "relevant" --no-token "irrelevant"

  # Convert and push to Hub
  python convert_qwen3_to_seq_cls.py --model Qwen/Qwen3-Reranker-0.6B --output ./qwen3-0.6b-seq-cls --push-to-hub --hub-repo-id "your-username/qwen3-0.6b-seq-cls"
        """
    )
    
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="Name or path of the source Qwen3 model (e.g., 'Qwen/Qwen3-Reranker-0.6B')"
    )
    
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Output directory to save the converted model"
    )
    
    parser.add_argument(
        "--yes-token",
        type=str,
        default="yes",
        help="Token representing positive class (default: 'yes')"
    )
    
    parser.add_argument(
        "--no-token",
        type=str,
        default="no",
        help="Token representing negative class (default: 'no')"
    )
    
    parser.add_argument(
        "--push-to-hub",
        action="store_true",
        help="Push the converted model to HuggingFace Hub"
    )
    
    parser.add_argument(
        "--hub-repo-id",
        type=str,
        help="Repository ID for HuggingFace Hub upload (required if --push-to-hub is used)"
    )
    
    parser.add_argument(
        "--no-verify",
        action="store_true",
        help="Skip verification of the conversion"
    )
    
    args = parser.parse_args()
    
    if args.push_to_hub and not args.hub_repo_id:
        parser.error("--hub-repo-id is required when --push-to-hub is used")
    
    convert_qwen3_to_sequence_classifier(
        model_name=args.model,
        output_dir=args.output,
        yes_token=args.yes_token,
        no_token=args.no_token,
        push_to_hub=args.push_to_hub,
        hub_repo_id=args.hub_repo_id,
        verify_conversion=not args.no_verify,
    )


if __name__ == "__main__":
    main()