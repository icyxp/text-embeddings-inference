#!/usr/bin/env python3
"""
Test script for Qwen3 model conversion

This script tests the conversion of Qwen3ForCausalLM to Qwen3ForSequenceClassification
and verifies that the conversion works correctly.

Usage:
    python test_qwen3_conversion.py [--model MODEL_NAME] [--quick]
"""

import argparse
import torch
import tempfile
import os
from transformers import AutoTokenizer, Qwen2ForCausalLM, Qwen2ForSequenceClassification

# Import our conversion functions
try:
    from qwen3_seq_cls_converter import convert_qwen3_to_seq_classifier, verify_conversion
except ImportError:
    print("❌ Could not import conversion functions. Make sure qwen3_seq_cls_converter.py is in the same directory.")
    exit(1)


def test_conversion_accuracy(model_name: str, num_test_cases: int = 5):
    """
    Test the accuracy of the conversion by comparing outputs.
    
    Args:
        model_name: Name of the model to test
        num_test_cases: Number of test cases to run
    """
    
    print(f"🧪 Testing conversion accuracy for {model_name}")
    print(f"📊 Running {num_test_cases} test cases...")
    
    # Test texts covering different scenarios
    test_texts = [
        "This document perfectly answers the user's question about machine learning.",
        "The weather today is sunny with a chance of rain in the afternoon.",
        "Python is a popular programming language used for data science and web development.",
        "The capital of France is Paris, which is located in the northern part of the country.",
        "This text is completely unrelated to any specific query or topic.",
        "The search results show relevant information about the requested topic.",
        "Error 404: Page not found. Please check the URL and try again.",
        "Machine learning algorithms can be used to solve complex problems in various domains.",
        "The user manual provides detailed instructions for setting up the software.",
        "Lorem ipsum dolor sit amet, consectetur adipiscing elit, sed do eiusmod tempor.",
    ]
    
    # Select test cases
    selected_tests = test_texts[:num_test_cases]
    
    try:
        # Convert the model
        print("🔄 Converting model...")
        converted_model = convert_qwen3_to_seq_classifier(model_name)
        
        # Load original model for comparison
        print("📥 Loading original model for comparison...")
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        original_model = Qwen2ForCausalLM.from_pretrained(model_name)
        
        yes_token_id = tokenizer.convert_tokens_to_ids("yes")
        no_token_id = tokenizer.convert_tokens_to_ids("no")
        
        print(f"🎯 Token IDs - yes: {yes_token_id}, no: {no_token_id}")
        
        # Test each case
        all_passed = True
        results = []
        
        for i, test_text in enumerate(selected_tests, 1):
            print(f"\n📝 Test {i}/{num_test_cases}: '{test_text[:60]}...'")
            
            inputs = tokenizer(test_text, return_tensors="pt", truncation=True, max_length=512)
            
            with torch.no_grad():
                # Original model prediction
                original_outputs = original_model(**inputs)
                last_token_logits = original_outputs.logits[0, -1, :]
                manual_diff = last_token_logits[yes_token_id] - last_token_logits[no_token_id]
                
                # Converted model prediction
                converted_outputs = converted_model(**inputs)
                model_logit = converted_outputs.logits.squeeze()
                
                # Calculate probabilities
                manual_prob = torch.softmax(torch.stack([last_token_logits[no_token_id], last_token_logits[yes_token_id]]), dim=0)[1]
                converted_prob = torch.sigmoid(model_logit)
                
                # Check if they match
                logit_match = torch.allclose(manual_diff, model_logit, atol=1e-3)
                prob_match = torch.allclose(manual_prob, converted_prob, atol=1e-3)
                
                test_passed = logit_match and prob_match
                all_passed = all_passed and test_passed
                
                results.append({
                    'text': test_text[:60] + '...',
                    'manual_logit': manual_diff.item(),
                    'converted_logit': model_logit.item(),
                    'manual_prob': manual_prob.item(),
                    'converted_prob': converted_prob.item(),
                    'logit_match': logit_match,
                    'prob_match': prob_match,
                    'passed': test_passed
                })
                
                status = "✅ PASS" if test_passed else "❌ FAIL"
                print(f"   Manual logit:    {manual_diff.item():.6f}")
                print(f"   Converted logit: {model_logit.item():.6f}")
                print(f"   Manual prob:     {manual_prob.item():.6f}")
                print(f"   Converted prob:  {converted_prob.item():.6f}")
                print(f"   Result: {status}")
        
        # Summary
        print(f"\n{'='*60}")
        print(f"📊 TEST SUMMARY")
        print(f"{'='*60}")
        
        passed_count = sum(1 for r in results if r['passed'])
        print(f"✅ Passed: {passed_count}/{num_test_cases}")
        print(f"❌ Failed: {num_test_cases - passed_count}/{num_test_cases}")
        
        if all_passed:
            print(f"🎉 ALL TESTS PASSED! Conversion is working correctly.")
        else:
            print(f"⚠️  Some tests failed. Please check the conversion logic.")
            
            # Show failed cases
            failed_cases = [r for r in results if not r['passed']]
            if failed_cases:
                print(f"\n❌ Failed test cases:")
                for case in failed_cases:
                    print(f"   - {case['text']}")
                    print(f"     Logit diff: {abs(case['manual_logit'] - case['converted_logit']):.6f}")
                    print(f"     Prob diff:  {abs(case['manual_prob'] - case['converted_prob']):.6f}")
        
        # Clean up
        del original_model
        del converted_model
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
        
        return all_passed
        
    except Exception as e:
        print(f"❌ Test failed with error: {e}")
        return False


def test_model_saving_and_loading(model_name: str):
    """
    Test that the converted model can be saved and loaded correctly.
    
    Args:
        model_name: Name of the model to test
    """
    
    print(f"💾 Testing model saving and loading for {model_name}")
    
    try:
        # Convert model
        converted_model = convert_qwen3_to_seq_classifier(model_name)
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        
        # Save to temporary directory
        with tempfile.TemporaryDirectory() as temp_dir:
            print(f"📁 Saving to temporary directory: {temp_dir}")
            
            # Save model and tokenizer
            converted_model.save_pretrained(temp_dir)
            tokenizer.save_pretrained(temp_dir)
            
            # Load the saved model
            print("📥 Loading saved model...")
            loaded_model = Qwen2ForSequenceClassification.from_pretrained(temp_dir)
            loaded_tokenizer = AutoTokenizer.from_pretrained(temp_dir)
            
            # Test that loaded model works
            test_text = "This is a test document for verification."
            
            # Original converted model
            inputs = tokenizer(test_text, return_tensors="pt")
            with torch.no_grad():
                original_output = converted_model(**inputs).logits.squeeze()
            
            # Loaded model
            inputs = loaded_tokenizer(test_text, return_tensors="pt")
            with torch.no_grad():
                loaded_output = loaded_model(**inputs).logits.squeeze()
            
            # Check if outputs match
            outputs_match = torch.allclose(original_output, loaded_output, atol=1e-6)
            
            if outputs_match:
                print("✅ Save/load test PASSED! Model can be saved and loaded correctly.")
                return True
            else:
                print("❌ Save/load test FAILED! Outputs don't match after loading.")
                print(f"   Original: {original_output.item():.6f}")
                print(f"   Loaded:   {loaded_output.item():.6f}")
                return False
                
    except Exception as e:
        print(f"❌ Save/load test failed with error: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(description="Test Qwen3 model conversion")
    parser.add_argument(
        "--model",
        type=str,
        default="Qwen/Qwen3-Reranker-0.6B",
        help="Model to test (default: Qwen/Qwen3-Reranker-0.6B)"
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Run quick test with fewer test cases"
    )
    parser.add_argument(
        "--skip-save-test",
        action="store_true",
        help="Skip the save/load test"
    )
    
    args = parser.parse_args()
    
    print(f"🚀 Starting Qwen3 conversion tests")
    print(f"📦 Model: {args.model}")
    print(f"⚡ Quick mode: {args.quick}")
    print(f"{'='*60}")
    
    # Determine number of test cases
    num_test_cases = 3 if args.quick else 5
    
    # Run accuracy test
    accuracy_passed = test_conversion_accuracy(args.model, num_test_cases)
    
    # Run save/load test
    save_load_passed = True
    if not args.skip_save_test:
        print(f"\n{'='*60}")
        save_load_passed = test_model_saving_and_loading(args.model)
    
    # Final summary
    print(f"\n{'='*60}")
    print(f"🏁 FINAL RESULTS")
    print(f"{'='*60}")
    
    print(f"✅ Accuracy Test: {'PASSED' if accuracy_passed else 'FAILED'}")
    if not args.skip_save_test:
        print(f"✅ Save/Load Test: {'PASSED' if save_load_passed else 'FAILED'}")
    
    overall_passed = accuracy_passed and save_load_passed
    
    if overall_passed:
        print(f"\n🎉 ALL TESTS PASSED! The conversion is working correctly.")
        print(f"✅ You can safely use the conversion scripts for {args.model}")
    else:
        print(f"\n⚠️  SOME TESTS FAILED! Please check the conversion logic.")
        print(f"❌ Do not use the conversion until issues are resolved.")
    
    return 0 if overall_passed else 1


if __name__ == "__main__":
    exit(main())