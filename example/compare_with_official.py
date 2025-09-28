#!/usr/bin/env python3
"""
Compare converted model with official implementation

This script compares the output of our converted Qwen3 sequence classification model
with the official Qwen3 reranker implementation to ensure they produce identical results.
"""

import torch
import numpy as np
from typing import List, Tuple
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModelForSequenceClassification

# Import our utilities
from qwen3_template_utils import format_qwen3_reranker_input, Qwen3RerankerWrapper
from qwen3_seq_cls_converter import convert_qwen3_to_seq_classifier


def compare_original_vs_converted(
    model_name: str = "Qwen/Qwen3-Reranker-0.6B",
    test_cases: List[Tuple[str, str]] = None,
    tolerance: float = 1e-4
) -> bool:
    """
    Compare original causal LM with converted sequence classification model.
    
    Args:
        model_name: Name of the original model
        test_cases: List of (query, document) pairs to test
        tolerance: Numerical tolerance for comparison
        
    Returns:
        True if all comparisons pass, False otherwise
    """
    
    if test_cases is None:
        test_cases = [
            ("Which planet is known as the Red Planet?", 
             "Venus is often called Earth's twin because of its similar size and proximity."),
            ("Which planet is known as the Red Planet?", 
             "Mars, known for its reddish appearance, is often referred to as the Red Planet."),
            ("Which planet is known as the Red Planet?", 
             "Jupiter, the largest planet in our solar system, has a prominent red spot."),
            ("Which planet is known as the Red Planet?", 
             "Saturn, famous for its rings, is sometimes mistaken for the Red Planet."),
            ("How to install Python?",
             "To install Python, visit python.org and download the installer for your OS."),
            ("How to install Python?",
             "The weather today is sunny with a chance of rain in the afternoon."),
        ]
    
    print(f"🔍 Comparing original vs converted model: {model_name}")
    print(f"📊 Testing {len(test_cases)} cases with tolerance {tolerance}")
    
    # Load original model (causal LM)
    print("📥 Loading original causal LM...")
    tokenizer = AutoTokenizer.from_pretrained(model_name, padding_side='left')
    original_model = AutoModelForCausalLM.from_pretrained(model_name).eval()
    
    yes_token_id = tokenizer.convert_tokens_to_ids("yes")
    no_token_id = tokenizer.convert_tokens_to_ids("no")
    
    # Convert to sequence classification
    print("🔄 Converting to sequence classification...")
    converted_model = convert_qwen3_to_seq_classifier(model_name)
    
    # Test each case
    all_passed = True
    results = []
    
    instruction = "Given a web search query, retrieve relevant passages that answer the query"
    
    for i, (query, document) in enumerate(test_cases, 1):
        print(f"\n📝 Test {i}/{len(test_cases)}")
        print(f"   Query: {query[:60]}...")
        print(f"   Document: {document[:60]}...")
        
        # Format input using official template
        formatted_input = format_qwen3_reranker_input(instruction, query, document)
        inputs = tokenizer(formatted_input, return_tensors="pt", truncation=True, max_length=8192)
        
        with torch.no_grad():
            # Original model prediction
            original_outputs = original_model(**inputs)
            last_token_logits = original_outputs.logits[0, -1, :]
            
            # Method 1: Manual logit difference
            manual_logit_diff = last_token_logits[yes_token_id] - last_token_logits[no_token_id]
            
            # Method 2: Softmax probability
            yes_no_logits = torch.stack([last_token_logits[no_token_id], last_token_logits[yes_token_id]])
            original_prob = torch.softmax(yes_no_logits, dim=0)[1]
            
            # Converted model prediction
            converted_outputs = converted_model(**inputs)
            converted_logit = converted_outputs.logits.squeeze()
            converted_prob = torch.sigmoid(converted_logit)
        
        # Compare results
        logit_match = torch.allclose(manual_logit_diff, converted_logit, atol=tolerance)
        prob_match = torch.allclose(original_prob, converted_prob, atol=tolerance)
        
        test_passed = logit_match and prob_match
        all_passed = all_passed and test_passed
        
        results.append({
            'query': query,
            'document': document,
            'original_logit_diff': manual_logit_diff.item(),
            'converted_logit': converted_logit.item(),
            'original_prob': original_prob.item(),
            'converted_prob': converted_prob.item(),
            'logit_diff': abs(manual_logit_diff.item() - converted_logit.item()),
            'prob_diff': abs(original_prob.item() - converted_prob.item()),
            'logit_match': logit_match,
            'prob_match': prob_match,
            'passed': test_passed
        })
        
        status = "✅ PASS" if test_passed else "❌ FAIL"
        print(f"   Original logit diff: {manual_logit_diff.item():.6f}")
        print(f"   Converted logit:     {converted_logit.item():.6f}")
        print(f"   Original prob:       {original_prob.item():.6f}")
        print(f"   Converted prob:      {converted_prob.item():.6f}")
        print(f"   Logit difference:    {abs(manual_logit_diff.item() - converted_logit.item()):.8f}")
        print(f"   Prob difference:     {abs(original_prob.item() - converted_prob.item()):.8f}")
        print(f"   Result: {status}")
    
    # Summary
    print(f"\n{'='*60}")
    print(f"📊 COMPARISON SUMMARY")
    print(f"{'='*60}")
    
    passed_count = sum(1 for r in results if r['passed'])
    print(f"✅ Passed: {passed_count}/{len(test_cases)}")
    print(f"❌ Failed: {len(test_cases) - passed_count}/{len(test_cases)}")
    
    if all_passed:
        print(f"🎉 ALL TESTS PASSED! Conversion is mathematically equivalent.")
        
        # Show statistics
        logit_diffs = [r['logit_diff'] for r in results]
        prob_diffs = [r['prob_diff'] for r in results]
        
        print(f"\n📈 Error Statistics:")
        print(f"   Max logit difference:  {max(logit_diffs):.8f}")
        print(f"   Mean logit difference: {np.mean(logit_diffs):.8f}")
        print(f"   Max prob difference:   {max(prob_diffs):.8f}")
        print(f"   Mean prob difference:  {np.mean(prob_diffs):.8f}")
        
    else:
        print(f"⚠️  SOME TESTS FAILED! Check conversion logic.")
        
        # Show failed cases
        failed_cases = [r for r in results if not r['passed']]
        print(f"\n❌ Failed cases:")
        for case in failed_cases:
            print(f"   Query: {case['query'][:50]}...")
            print(f"   Logit diff: {case['logit_diff']:.8f}")
            print(f"   Prob diff:  {case['prob_diff']:.8f}")
    
    # Clean up
    del original_model
    del converted_model
    torch.cuda.empty_cache() if torch.cuda.is_available() else None
    
    return all_passed


def compare_with_official_format():
    """
    Compare our implementation with the exact format from the official example.
    """
    
    print(f"\n{'='*60}")
    print(f"🎯 OFFICIAL FORMAT COMPARISON")
    print(f"{'='*60}")
    
    # Test data from the official example
    queries = [
        "Which planet is known as the Red Planet?",
        "Which planet is known as the Red Planet?",
        "Which planet is known as the Red Planet?",
        "Which planet is known as the Red Planet?",
    ]
    
    documents = [
        "Venus is often called Earth's twin because of its similar size and proximity.",
        "Mars, known for its reddish appearance, is often referred to as the Red Planet.",
        "Jupiter, the largest planet in our solar system, has a prominent red spot.",
        "Saturn, famous for its rings, is sometimes mistaken for the Red Planet.",
    ]
    
    # Expected scores from official example
    expected_scores = [0.04272603616118431, 0.9991921782493591, 0.40642625093460083, 0.9718492031097412]
    
    model_name = "Qwen/Qwen3-Reranker-0.6B"
    
    # Test with our wrapper
    print("🔄 Testing with our Qwen3RerankerWrapper...")
    wrapper = Qwen3RerankerWrapper(model_name, model_type="causal_lm")
    
    instruction = "Given a web search query, retrieve relevant passages that answer the query"
    our_scores = wrapper.predict(queries, documents, instruction)
    
    print(f"\n📊 Score Comparison:")
    print(f"{'Index':<5} {'Expected':<12} {'Our Score':<12} {'Difference':<12} {'Match':<8}")
    print(f"{'-'*60}")
    
    all_match = True
    tolerance = 1e-6
    
    for i, (expected, actual) in enumerate(zip(expected_scores, our_scores)):
        diff = abs(expected - actual)
        match = diff < tolerance
        all_match = all_match and match
        
        status = "✅" if match else "❌"
        print(f"{i:<5} {expected:<12.8f} {actual:<12.8f} {diff:<12.8f} {status:<8}")
    
    print(f"\n🏁 Overall Match: {'✅ YES' if all_match else '❌ NO'}")
    
    if all_match:
        print(f"🎉 Perfect match with official implementation!")
    else:
        print(f"⚠️  Scores don't match exactly. Check template formatting.")
    
    return all_match


def main():
    """
    Run comprehensive comparison tests.
    """
    
    print("🚀 Qwen3 Model Conversion Comparison")
    print("="*60)
    
    try:
        # Test 1: Compare original vs converted
        print("🧪 Test 1: Original vs Converted Model")
        conversion_passed = compare_original_vs_converted()
        
        # Test 2: Compare with official format
        print("\n🧪 Test 2: Official Format Comparison")
        format_passed = compare_with_official_format()
        
        # Final summary
        print(f"\n{'='*60}")
        print(f"🏁 FINAL RESULTS")
        print(f"{'='*60}")
        
        print(f"✅ Conversion Test: {'PASSED' if conversion_passed else 'FAILED'}")
        print(f"✅ Format Test:     {'PASSED' if format_passed else 'FAILED'}")
        
        overall_passed = conversion_passed and format_passed
        
        if overall_passed:
            print(f"\n🎉 ALL TESTS PASSED!")
            print(f"✅ Your conversion is mathematically equivalent to the original")
            print(f"✅ Template formatting matches official implementation")
            print(f"✅ Ready for production use!")
        else:
            print(f"\n⚠️  SOME TESTS FAILED!")
            print(f"❌ Please review the conversion logic and template formatting")
        
        return 0 if overall_passed else 1
        
    except Exception as e:
        print(f"❌ Error during comparison: {e}")
        return 1


if __name__ == "__main__":
    exit(main())