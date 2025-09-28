"""
Simple Qwen3 Sequence Classification Converter

A streamlined script to convert Qwen3ForCausalLM to Qwen3ForSequenceClassification
following the 5-step methodology from seq-cls.md.

Copyright 2025, MIT License
"""

import torch
from transformers import (
    AutoTokenizer,
    Qwen2ForCausalLM,
    Qwen2ForSequenceClassification,
)


@torch.no_grad()
def convert_qwen3_to_seq_classifier(
    model_name: str,
    yes_token: str = "yes",
    no_token: str = "no",
    num_labels: int = 1,
) -> Qwen2ForSequenceClassification:
    """
    Convert Qwen3ForCausalLM to Qwen3ForSequenceClassification.
    
    Follows the 5-step process:
    1. Load Qwen3ForCausalLM and extract lm_head weights
    2. Get token IDs for "yes" and "no" tokens
    3. Subtract "no" vector from "yes" vector
    4. Load Qwen3ForSequenceClassification with num_labels=1
    5. Replace score layer with computed vector
    
    Args:
        model_name: HuggingFace model name or local path
        yes_token: Token for positive class (default: "yes")
        no_token: Token for negative class (default: "no")
        num_labels: Number of labels for classification (default: 1 for binary)
    
    Returns:
        Converted Qwen2ForSequenceClassification model
    """
    
    print(f"Converting {model_name} to sequence classifier...")
    
    # Step 1: Load Causal LM and extract lm_head weights
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    causal_lm = Qwen2ForCausalLM.from_pretrained(model_name)
    lm_head_weights = causal_lm.lm_head.weight
    
    # Step 2: Get token IDs for yes/no tokens
    yes_token_id = tokenizer.convert_tokens_to_ids(yes_token)
    no_token_id = tokenizer.convert_tokens_to_ids(no_token)
    
    if yes_token_id is None or no_token_id is None:
        raise ValueError(f"Could not find token IDs for '{yes_token}' or '{no_token}'")
    
    print(f"Token IDs - {yes_token}: {yes_token_id}, {no_token}: {no_token_id}")
    
    # Step 3: Create classifier vector (yes - no)
    yes_vector = lm_head_weights[yes_token_id]
    no_vector = lm_head_weights[no_token_id]
    classifier_vector = yes_vector - no_vector
    
    # Step 4: Load as Sequence Classification model
    seq_cls_model = Qwen2ForSequenceClassification.from_pretrained(
        model_name,
        num_labels=num_labels,
        ignore_mismatched_sizes=True
    )
    
    # Step 5: Replace score layer weights
    seq_cls_model.score.weight.copy_(classifier_vector.unsqueeze(0))
    if seq_cls_model.score.bias is not None:
        seq_cls_model.score.bias.zero_()
    
    # Update config
    seq_cls_model.config.id2label = {0: "negative", 1: "positive"}
    seq_cls_model.config.label2id = {"negative": 0, "positive": 1}
    
    # Clean up memory
    del causal_lm
    
    print("✅ Conversion completed!")
    return seq_cls_model


def verify_conversion(
    original_model_name: str,
    converted_model: Qwen2ForSequenceClassification,
    test_text: str = "This is a test document for verification.",
    yes_token: str = "yes",
    no_token: str = "no",
) -> bool:
    """
    Verify that the conversion was successful by comparing outputs.
    
    Args:
        original_model_name: Name of the original causal LM
        converted_model: The converted sequence classification model
        test_text: Text to use for verification
        yes_token: Positive token used in conversion
        no_token: Negative token used in conversion
    
    Returns:
        True if verification passes, False otherwise
    """
    
    print(f"Verifying conversion with test text: '{test_text[:50]}...'")
    
    tokenizer = AutoTokenizer.from_pretrained(original_model_name)
    causal_lm = Qwen2ForCausalLM.from_pretrained(original_model_name)
    
    yes_token_id = tokenizer.convert_tokens_to_ids(yes_token)
    no_token_id = tokenizer.convert_tokens_to_ids(no_token)
    
    inputs = tokenizer(test_text, return_tensors="pt", truncation=True)
    
    with torch.no_grad():
        # Get logits from original causal LM
        causal_outputs = causal_lm(**inputs)
        last_token_logits = causal_outputs.logits[0, -1, :]
        manual_diff = last_token_logits[yes_token_id] - last_token_logits[no_token_id]
        
        # Get logits from converted model
        seq_cls_outputs = converted_model(**inputs)
        model_logit = seq_cls_outputs.logits.squeeze()
        
        # Check if they match
        matches = torch.allclose(manual_diff, model_logit, atol=1e-3)
        
        print(f"Manual difference: {manual_diff.item():.4f}")
        print(f"Model output: {model_logit.item():.4f}")
        print(f"Match: {matches}")
        
        del causal_lm
        return matches


def batch_convert_qwen3_models(
    model_names: list[str],
    output_base_dir: str = "./converted_models",
    yes_token: str = "yes",
    no_token: str = "no",
) -> dict[str, str]:
    """
    Convert multiple Qwen3 models to sequence classifiers.
    
    Args:
        model_names: List of model names to convert
        output_base_dir: Base directory for saving converted models
        yes_token: Positive token
        no_token: Negative token
    
    Returns:
        Dictionary mapping original model names to output paths
    """
    
    import os
    results = {}
    
    for model_name in model_names:
        print(f"\n{'='*60}")
        print(f"Converting: {model_name}")
        print(f"{'='*60}")
        
        try:
            # Convert model
            converted_model = convert_qwen3_to_seq_classifier(
                model_name=model_name,
                yes_token=yes_token,
                no_token=no_token
            )
            
            # Verify conversion
            if verify_conversion(model_name, converted_model, yes_token=yes_token, no_token=no_token):
                print("✅ Verification passed!")
            else:
                print("❌ Verification failed!")
                continue
            
            # Save model
            model_dir_name = model_name.replace("/", "_").replace("-", "_")
            output_dir = os.path.join(output_base_dir, f"{model_dir_name}_seq_cls")
            os.makedirs(output_dir, exist_ok=True)
            
            converted_model.save_pretrained(output_dir)
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            tokenizer.save_pretrained(output_dir)
            
            results[model_name] = output_dir
            print(f"✅ Saved to: {output_dir}")
            
        except Exception as e:
            print(f"❌ Failed to convert {model_name}: {e}")
            results[model_name] = None
    
    return results


# Example usage and test functions
def test_qwen3_conversion():
    """Test the conversion with a small Qwen3 model."""
    
    model_name = "Qwen/Qwen3-Reranker-0.6B"
    
    print("Testing Qwen3 conversion...")
    
    # Convert model
    converted_model = convert_qwen3_to_seq_classifier(
        model_name=model_name,
        yes_token="yes",
        no_token="no"
    )
    
    # Verify conversion
    test_texts = [
        "This document is highly relevant to the query.",
        "This text has nothing to do with the question.",
        "The answer can be found in this passage.",
    ]
    
    for text in test_texts:
        verify_conversion(model_name, converted_model, text)
        print()
    
    print("Test completed!")


if __name__ == "__main__":
    # Example: Convert single model
    test_qwen3_conversion()
    
    # Example: Batch convert multiple models
    # models_to_convert = [
    #     "Qwen/Qwen3-Reranker-0.6B",
    #     "Qwen/Qwen3-Reranker-4B",
    #     "Qwen/Qwen3-Reranker-8B",
    # ]
    # 
    # results = batch_convert_qwen3_models(models_to_convert)
    # print("\nConversion Results:")
    # for model, path in results.items():
    #     status = "✅ Success" if path else "❌ Failed"
    #     print(f"{model}: {status}")
    #     if path:
    #         print(f"  Saved to: {path}")