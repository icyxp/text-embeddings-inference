# Qwen3 Model Conversion Scripts

This directory contains scripts to convert Qwen3ForCausalLM models to Qwen3ForSequenceClassification models for reranking tasks.

## Overview

The conversion follows a 5-step methodology as described in `seq-cls.md`:

1. **Load Qwen3ForCausalLM** and extract lm_head weight matrix
2. **Extract token vectors** for "yes" and "no" tokens (or custom tokens)
3. **Compute classifier vector** by subtracting "no" vector from "yes" vector
4. **Load Qwen3ForSequenceClassification** with num_labels=1
5. **Replace score layer** with the computed classifier vector

## ⚠️ Important: Template Formatting

**Critical**: Qwen3 reranker models require specific template formatting to work correctly. The input must be formatted exactly as:

```
<|im_start|>system
Judge whether the Document meets the requirements based on the Query and the Instruct provided. Note that the answer can only be "yes" or "no".<|im_end|>
<|im_start|>user
<Instruct>: {instruction}
<Query>: {query}
<Document>: {document}<|im_end|>
<|im_start|>assistant
<think>

</think>

```

Our conversion scripts automatically add this template to `tokenizer_config.json` as `reranker_template`, making it compatible with text-embeddings-inference.

## Scripts

### 1. `convert_qwen3_to_seq_cls.py` - Full-featured Converter

A comprehensive script with command-line interface, verification, and Hub upload capabilities.

#### Features:
- ✅ Command-line interface with argparse
- ✅ Automatic verification of conversion
- ✅ HuggingFace Hub upload support
- ✅ Custom token support
- ✅ Detailed logging and error handling
- ✅ Metadata saving
- ✅ **Automatic template injection** into tokenizer_config.json

#### Usage:

```bash
# Basic conversion
python convert_qwen3_to_seq_cls.py \
    --model Qwen/Qwen3-Reranker-0.6B \
    --output ./qwen3-0.6b-seq-cls

# Convert with custom tokens
python convert_qwen3_to_seq_cls.py \
    --model Qwen/Qwen3-Reranker-4B \
    --output ./qwen3-4b-seq-cls \
    --yes-token "relevant" \
    --no-token "irrelevant"

# Convert and push to Hub
python convert_qwen3_to_seq_cls.py \
    --model Qwen/Qwen3-Reranker-0.6B \
    --output ./qwen3-0.6b-seq-cls \
    --push-to-hub \
    --hub-repo-id "your-username/qwen3-0.6b-seq-cls"
```

#### Arguments:
- `--model`: Source Qwen3 model name or path (required)
- `--output`: Output directory for converted model (required)
- `--yes-token`: Token for positive class (default: "yes")
- `--no-token`: Token for negative class (default: "no")
- `--push-to-hub`: Push converted model to HuggingFace Hub
- `--hub-repo-id`: Repository ID for Hub upload
- `--no-verify`: Skip conversion verification

### 2. `qwen3_seq_cls_converter.py` - Lightweight Converter

A streamlined script for programmatic use and batch conversions.

#### Features:
- ✅ Simple function-based API
- ✅ Batch conversion support
- ✅ Built-in verification
- ✅ Memory efficient
- ✅ Easy to integrate into other scripts

### 3. `qwen3_template_utils.py` - Template Utilities

Essential utilities for proper Qwen3 reranker template formatting.

#### Features:
- ✅ Official Qwen3 reranker template formatting
- ✅ Batch processing support
- ✅ CrossEncoder compatibility
- ✅ Wrapper class for easy usage
- ✅ **Ensures identical results to official implementation**

#### Usage:

```python
from qwen3_seq_cls_converter import convert_qwen3_to_seq_classifier, verify_conversion

# Convert single model
converted_model = convert_qwen3_to_seq_classifier(
    model_name="Qwen/Qwen3-Reranker-0.6B",
    yes_token="yes",
    no_token="no"
)

# Verify conversion
is_valid = verify_conversion(
    original_model_name="Qwen/Qwen3-Reranker-0.6B",
    converted_model=converted_model,
    test_text="This is a test document."
)

# Save converted model
converted_model.save_pretrained("./converted_model")
```

#### Batch Conversion:

```python
from qwen3_seq_cls_converter import batch_convert_qwen3_models

models = [
    "Qwen/Qwen3-Reranker-0.6B",
    "Qwen/Qwen3-Reranker-4B",
    "Qwen/Qwen3-Reranker-8B",
]

results = batch_convert_qwen3_models(
    model_names=models,
    output_base_dir="./converted_models"
)
```

## Requirements

```bash
pip install torch transformers huggingface_hub
```

## Model Compatibility

These scripts are designed for Qwen3 reranker models, which use the Qwen2 architecture internally. Compatible models include:

- `Qwen/Qwen3-Reranker-0.6B`
- `Qwen/Qwen3-Reranker-4B`
- `Qwen/Qwen3-Reranker-8B`
- Custom fine-tuned Qwen3 models

## Token Configuration

The scripts support custom tokens for different use cases:

### Default (Qwen3 Rerankers):
- `yes_token="yes"`
- `no_token="no"`

### Alternative Configurations:
- Relevance: `yes_token="relevant"`, `no_token="irrelevant"`
- Binary: `yes_token="1"`, `no_token="0"`
- Custom: Any single tokens in the vocabulary

## Verification Process

Both scripts include verification to ensure the conversion is correct:

1. **Input Processing**: Same tokenization for both models
2. **Logit Comparison**: Compare `logit_yes - logit_no` from original vs. converted model
3. **Tolerance Check**: Verify outputs match within acceptable tolerance (1e-3)
4. **Multiple Test Cases**: Test with different input texts

## Output Structure

Converted models are saved with the following structure:

```
output_directory/
├── config.json                 # Model configuration
├── pytorch_model.bin          # Model weights
├── tokenizer.json             # Tokenizer
├── tokenizer_config.json      # Tokenizer configuration
├── special_tokens_map.json    # Special tokens
├── vocab.txt                  # Vocabulary
└── conversion_metadata.json   # Conversion details (full script only)
```

## Integration with text-embeddings-inference

The converted models are compatible with text-embeddings-inference and will automatically use the template system:

```bash
text-embeddings-router \
    --model-id /path/to/converted/model \
    --auto-truncate \
    --max-client-batch-size 200
```

The system will:
1. Detect the sequence classification model
2. Apply appropriate templates from `tokenizer_config.json`
3. Use the converted score layer for binary classification

## Troubleshooting

### Common Issues:

1. **Token Not Found**: Ensure `yes_token` and `no_token` exist in the model's vocabulary
2. **Memory Issues**: Use smaller batch sizes or enable gradient checkpointing
3. **Device Mismatch**: Ensure all tensors are on the same device (CPU/GPU)
4. **Hub Upload Fails**: Check authentication and repository permissions

### Debug Tips:

1. **Enable Verification**: Always run with verification enabled first
2. **Check Token IDs**: Print token IDs to verify they're correct
3. **Test Small Inputs**: Start with short test texts
4. **Monitor Memory**: Use `torch.cuda.empty_cache()` between conversions

## Examples

### Example 1: Basic Conversion

```bash
# Convert Qwen3-Reranker-0.6B to sequence classifier
python convert_qwen3_to_seq_cls.py \
    --model Qwen/Qwen3-Reranker-0.6B \
    --output ./qwen3-0.6b-seq-cls
```

### Example 2: Using Template Utilities (Recommended)

```python
from qwen3_template_utils import Qwen3RerankerWrapper

# Initialize wrapper (handles template formatting automatically)
wrapper = Qwen3RerankerWrapper("./qwen3-0.6b-seq-cls")

# Single prediction
score = wrapper.predict(
    "Which planet is known as the Red Planet?",
    "Mars, known for its reddish appearance, is often referred to as the Red Planet."
)
print(f"Relevance score: {score:.4f}")

# Batch ranking
query = "Best Python practices"
documents = [
    "Follow PEP 8 style guidelines and write clean code.",
    "The weather is nice today.",
    "Use virtual environments and write unit tests.",
]

rankings = wrapper.rank_documents(query, documents, return_scores=True)
for i, (idx, score) in enumerate(rankings):
    print(f"{i+1}. Score: {score:.4f} - {documents[idx]}")
```

### Example 2: Custom Tokens

```python
# Use custom relevance tokens
python convert_qwen3_to_seq_cls.py \
    --model Qwen/Qwen3-Reranker-4B \
    --output ./qwen3-4b-relevance \
    --yes-token "relevant" \
    --no-token "irrelevant"
```

### Example 3: Programmatic Usage

```python
from qwen3_seq_cls_converter import convert_qwen3_to_seq_classifier

# Convert and use immediately
model = convert_qwen3_to_seq_classifier("Qwen/Qwen3-Reranker-0.6B")

# Test the converted model
from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-Reranker-0.6B")

text = "This document is relevant to the query."
inputs = tokenizer(text, return_tensors="pt")
outputs = model(**inputs)
probability = torch.sigmoid(outputs.logits).item()

print(f"Relevance probability: {probability:.4f}")
```

## License

MIT License - See individual script headers for details.
