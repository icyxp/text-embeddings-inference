#!/usr/bin/env python3
"""
Example usage of Qwen3 sequence classification conversion

This script demonstrates how to use the converted Qwen3 models for reranking tasks.
"""

import torch
from transformers import AutoTokenizer
from qwen3_seq_cls_converter import convert_qwen3_to_seq_classifier


def format_qwen3_reranker_input(instruction, query, document):
    """
    Format input for Qwen3 reranker using the official template.
    This matches the format used in the original Qwen3-Reranker models.
    """
    prefix = '<|im_start|>system\nJudge whether the Document meets the requirements based on the Query and the Instruct provided. Note that the answer can only be "yes" or "no".<|im_end|>\n<|im_start|>user\n'
    suffix = "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
    
    if instruction is None:
        instruction = "Given a web search query, retrieve relevant passages that answer the query"
    
    return f"{prefix}<Instruct>: {instruction}\n<Query>: {query}\n<Document>: {document}{suffix}"


def example_reranking_task():
    """
    Example of using a converted Qwen3 model for document reranking.
    Uses the official Qwen3 reranker template format.
    """
    
    print("🔄 Converting Qwen3 model for reranking...")
    
    # Convert the model
    model_name = "Qwen/Qwen3-Reranker-0.6B"
    model = convert_qwen3_to_seq_classifier(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    # Example query and documents
    query = "What is machine learning?"
    instruction = "Given a web search query, retrieve relevant passages that answer the query"
    
    documents = [
        "Machine learning is a subset of artificial intelligence that enables computers to learn and make decisions from data without being explicitly programmed.",
        "The weather today is sunny with temperatures reaching 25 degrees Celsius.",
        "Machine learning algorithms include supervised learning, unsupervised learning, and reinforcement learning techniques.",
        "Cooking pasta requires boiling water, adding salt, and cooking for 8-12 minutes depending on the type.",
        "Deep learning is a specialized branch of machine learning that uses neural networks with multiple layers to model complex patterns.",
    ]
    
    print(f"📝 Query: {query}")
    print(f"📚 Ranking {len(documents)} documents...")
    
    # Score each document using proper template
    scores = []
    for i, doc in enumerate(documents):
        # Use the official Qwen3 reranker template format
        input_text = format_qwen3_reranker_input(instruction, query, doc)
        
        # Tokenize and get model prediction
        inputs = tokenizer(input_text, return_tensors="pt", truncation=True, max_length=8192)
        
        with torch.no_grad():
            outputs = model(**inputs)
            score = torch.sigmoid(outputs.logits).item()  # Convert to probability
        
        scores.append((i, score, doc))
        print(f"   Doc {i+1}: {score:.4f} - {doc[:80]}...")
    
    # Sort by score (descending)
    scores.sort(key=lambda x: x[1], reverse=True)
    
    print(f"\n🏆 Ranking Results:")
    for rank, (doc_idx, score, doc) in enumerate(scores, 1):
        print(f"   {rank}. Score: {score:.4f} - {doc[:100]}...")
    
    return scores


def example_binary_classification():
    """
    Example of using a converted Qwen3 model for binary relevance classification.
    Uses the official Qwen3 reranker template format.
    """
    
    print("\n" + "="*60)
    print("🎯 Binary Classification Example")
    print("="*60)
    
    # Convert the model
    model_name = "Qwen/Qwen3-Reranker-0.6B"
    model = convert_qwen3_to_seq_classifier(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    # Example query-document pairs
    examples = [
        {
            "query": "How to install Python?",
            "document": "To install Python, visit python.org, download the installer for your operating system, and follow the installation instructions.",
            "expected": "relevant"
        },
        {
            "query": "How to install Python?",
            "document": "The best pizza toppings include pepperoni, mushrooms, and extra cheese for a delicious meal.",
            "expected": "irrelevant"
        },
        {
            "query": "What is the capital of Japan?",
            "document": "Tokyo is the capital and largest city of Japan, serving as the country's political and economic center.",
            "expected": "relevant"
        },
        {
            "query": "What is the capital of Japan?",
            "document": "Machine learning algorithms can be trained on large datasets to improve their performance over time.",
            "expected": "irrelevant"
        },
    ]
    
    instruction = "Given a web search query, retrieve relevant passages that answer the query"
    print("🔍 Classifying query-document pairs...")
    
    correct_predictions = 0
    total_predictions = len(examples)
    
    for i, example in enumerate(examples, 1):
        # Use the official Qwen3 reranker template format
        input_text = format_qwen3_reranker_input(instruction, example['query'], example['document'])
        
        # Get prediction
        inputs = tokenizer(input_text, return_tensors="pt", truncation=True, max_length=8192)
        
        with torch.no_grad():
            outputs = model(**inputs)
            probability = torch.sigmoid(outputs.logits).item()
        
        # Convert to binary prediction (threshold = 0.5)
        predicted = "relevant" if probability > 0.5 else "irrelevant"
        is_correct = predicted == example['expected']
        
        if is_correct:
            correct_predictions += 1
        
        status = "✅" if is_correct else "❌"
        print(f"\n   Example {i}: {status}")
        print(f"   Query: {example['query']}")
        print(f"   Document: {example['document'][:100]}...")
        print(f"   Expected: {example['expected']}")
        print(f"   Predicted: {predicted} (prob: {probability:.4f})")
    
    accuracy = correct_predictions / total_predictions
    print(f"\n📊 Accuracy: {correct_predictions}/{total_predictions} ({accuracy:.2%})")
    
    return accuracy


def example_batch_processing():
    """
    Example of batch processing multiple query-document pairs efficiently.
    Uses the official Qwen3 reranker template format.
    """
    
    print("\n" + "="*60)
    print("⚡ Batch Processing Example")
    print("="*60)
    
    # Convert the model
    model_name = "Qwen/Qwen3-Reranker-0.6B"
    model = convert_qwen3_to_seq_classifier(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    # Example batch data
    query = "Best practices for Python programming"
    instruction = "Given a web search query, retrieve relevant passages that answer the query"
    documents = [
        "Follow PEP 8 style guidelines, use meaningful variable names, write docstrings, and implement proper error handling.",
        "The history of the Roman Empire spans several centuries and includes many significant events and figures.",
        "Use virtual environments, write unit tests, keep functions small and focused, and avoid global variables when possible.",
        "Chocolate chip cookies require flour, sugar, butter, eggs, and chocolate chips mixed together and baked.",
        "Code reviews, version control with Git, and continuous integration are essential for collaborative Python development.",
    ]
    
    print(f"📝 Query: {query}")
    print(f"📦 Processing batch of {len(documents)} documents...")
    
    # Prepare batch inputs using proper template
    input_texts = [format_qwen3_reranker_input(instruction, query, doc) for doc in documents]
    
    # Tokenize batch
    inputs = tokenizer(
        input_texts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=8192
    )
    
    # Get batch predictions
    with torch.no_grad():
        outputs = model(**inputs)
        probabilities = torch.sigmoid(outputs.logits).squeeze()
    
    # Process results
    results = list(zip(documents, probabilities.tolist()))
    results.sort(key=lambda x: x[1], reverse=True)  # Sort by relevance score
    
    print(f"\n🏆 Batch Results (sorted by relevance):")
    for i, (doc, prob) in enumerate(results, 1):
        relevance = "🟢 Relevant" if prob > 0.5 else "🔴 Irrelevant"
        print(f"   {i}. {prob:.4f} {relevance}")
        print(f"      {doc[:100]}...")
    
    return results


def main():
    """
    Run all examples to demonstrate the converted model capabilities.
    """
    
    print("🚀 Qwen3 Sequence Classification Examples")
    print("="*60)
    
    try:
        # Example 1: Document reranking
        print("📋 Example 1: Document Reranking")
        reranking_results = example_reranking_task()
        
        # Example 2: Binary classification
        binary_accuracy = example_binary_classification()
        
        # Example 3: Batch processing
        batch_results = example_batch_processing()
        
        # Summary
        print(f"\n" + "="*60)
        print("📊 SUMMARY")
        print("="*60)
        print(f"✅ Document reranking: {len(reranking_results)} documents ranked")
        print(f"✅ Binary classification accuracy: {binary_accuracy:.2%}")
        print(f"✅ Batch processing: {len(batch_results)} documents processed")
        print(f"\n🎉 All examples completed successfully!")
        
    except Exception as e:
        print(f"❌ Error running examples: {e}")
        print(f"💡 Make sure you have the required dependencies installed:")
        print(f"   pip install torch transformers")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())