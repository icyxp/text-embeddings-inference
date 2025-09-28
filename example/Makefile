# Makefile for Qwen3 conversion scripts
# Usage: make <target>

.PHONY: help test convert-small convert-medium convert-large example clean install

# Default model for testing
MODEL ?= Qwen/Qwen3-Reranker-0.6B
OUTPUT_DIR ?= ./converted_models

help:
	@echo "Qwen3 Model Conversion Makefile"
	@echo "================================"
	@echo ""
	@echo "Available targets:"
	@echo "  help           - Show this help message"
	@echo "  install        - Install required dependencies"
	@echo "  test           - Run conversion tests"
	@echo "  test-quick     - Run quick conversion tests"
	@echo "  convert-small  - Convert Qwen3-Reranker-0.6B"
	@echo "  convert-medium - Convert Qwen3-Reranker-4B"
	@echo "  convert-large  - Convert Qwen3-Reranker-8B"
	@echo "  convert-custom - Convert custom model (set MODEL variable)"
	@echo "  example        - Run usage examples"
	@echo "  clean          - Clean up generated files"
	@echo ""
	@echo "Variables:"
	@echo "  MODEL          - Model to convert (default: $(MODEL))"
	@echo "  OUTPUT_DIR     - Output directory (default: $(OUTPUT_DIR))"
	@echo ""
	@echo "Examples:"
	@echo "  make test"
	@echo "  make convert-small"
	@echo "  make convert-custom MODEL=Qwen/Qwen3-Reranker-4B"
	@echo "  make convert-custom MODEL=./local/model OUTPUT_DIR=./my_output"

install:
	@echo "📦 Installing required dependencies..."
	pip install torch transformers huggingface_hub
	@echo "✅ Dependencies installed!"

test:
	@echo "🧪 Running conversion tests..."
	python test_qwen3_conversion.py --model $(MODEL)

test-quick:
	@echo "⚡ Running quick conversion tests..."
	python test_qwen3_conversion.py --model $(MODEL) --quick

convert-small:
	@echo "🔄 Converting Qwen3-Reranker-0.6B..."
	python convert_qwen3_to_seq_cls.py \
		--model Qwen/Qwen3-Reranker-0.6B \
		--output $(OUTPUT_DIR)/qwen3-0.6b-seq-cls

convert-medium:
	@echo "🔄 Converting Qwen3-Reranker-4B..."
	python convert_qwen3_to_seq_cls.py \
		--model Qwen/Qwen3-Reranker-4B \
		--output $(OUTPUT_DIR)/qwen3-4b-seq-cls

convert-large:
	@echo "🔄 Converting Qwen3-Reranker-8B..."
	python convert_qwen3_to_seq_cls.py \
		--model Qwen/Qwen3-Reranker-8B \
		--output $(OUTPUT_DIR)/qwen3-8b-seq-cls

convert-custom:
	@echo "🔄 Converting $(MODEL)..."
	@mkdir -p $(OUTPUT_DIR)
	python convert_qwen3_to_seq_cls.py \
		--model $(MODEL) \
		--output $(OUTPUT_DIR)/$(shell echo $(MODEL) | sed 's/\//_/g' | sed 's/-/_/g')-seq-cls

convert-batch:
	@echo "🔄 Converting multiple models..."
	python -c "from qwen3_seq_cls_converter import batch_convert_qwen3_models; \
		models = ['Qwen/Qwen3-Reranker-0.6B', 'Qwen/Qwen3-Reranker-4B']; \
		batch_convert_qwen3_models(models, '$(OUTPUT_DIR)')"

example:
	@echo "🚀 Running usage examples..."
	python example_usage.py

verify:
	@echo "🔍 Verifying converted models..."
	@if [ -d "$(OUTPUT_DIR)" ]; then \
		echo "Found converted models in $(OUTPUT_DIR):"; \
		ls -la $(OUTPUT_DIR)/; \
	else \
		echo "No converted models found. Run 'make convert-small' first."; \
	fi

clean:
	@echo "🧹 Cleaning up generated files..."
	rm -rf $(OUTPUT_DIR)
	rm -rf __pycache__
	rm -rf *.pyc
	rm -rf .pytest_cache
	@echo "✅ Cleanup completed!"

# Development targets
lint:
	@echo "🔍 Running code linting..."
	python -m flake8 *.py --max-line-length=100 --ignore=E203,W503

format:
	@echo "🎨 Formatting code..."
	python -m black *.py --line-length=100

check-deps:
	@echo "🔍 Checking dependencies..."
	@python -c "import torch; print(f'✅ PyTorch: {torch.__version__}')" 2>/dev/null || echo "❌ PyTorch not found"
	@python -c "import transformers; print(f'✅ Transformers: {transformers.__version__}')" 2>/dev/null || echo "❌ Transformers not found"
	@python -c "import huggingface_hub; print(f'✅ HuggingFace Hub: {huggingface_hub.__version__}')" 2>/dev/null || echo "❌ HuggingFace Hub not found"

# Quick start target
quickstart: install test-quick convert-small example
	@echo ""
	@echo "🎉 Quickstart completed!"
	@echo "✅ Dependencies installed"
	@echo "✅ Tests passed"
	@echo "✅ Model converted"
	@echo "✅ Examples run"
	@echo ""
	@echo "Your converted model is ready at: $(OUTPUT_DIR)/qwen3-0.6b-seq-cls"