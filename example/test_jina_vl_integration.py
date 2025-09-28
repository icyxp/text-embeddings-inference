#!/usr/bin/env python3
"""
Jina VL Integration Test

This script tests the basic functionality of the jina-embeddings-v4-vllm-retrieval
model integration with text-embeddings-router.

Usage:
    python test_jina_vl_integration.py [--base-url http://localhost:3000]
"""

import argparse
import requests
import json
import base64
import io
from PIL import Image
import sys


def create_test_image(text: str = "Test Image") -> str:
    """Create a simple test image"""
    img = Image.new('RGB', (100, 100), color='lightblue')
    
    # Convert to base64
    buffer = io.BytesIO()
    img.save(buffer, format='JPEG')
    encoded_string = base64.b64encode(buffer.getvalue()).decode('utf-8')
    return f"data:image/jpeg;base64,{encoded_string}"


def test_service_health(base_url: str) -> bool:
    """Test if the service is running"""
    try:
        response = requests.get(f"{base_url}/health", timeout=5)
        return response.status_code == 200
    except:
        return False


def test_model_info(base_url: str) -> dict:
    """Test model info endpoint"""
    try:
        response = requests.get(f"{base_url}/info", timeout=10)
        if response.status_code == 200:
            return response.json()
        else:
            return {"error": f"Status {response.status_code}: {response.text}"}
    except Exception as e:
        return {"error": str(e)}


def test_text_embedding(base_url: str) -> dict:
    """Test text-only embedding"""
    payload = {
        "inputs": {
            "text": "This is a test document for embedding",
            "input_type": "passage"
        },
        "normalize": True
    }
    
    try:
        response = requests.post(
            f"{base_url}/embed_multimodal",
            json=payload,
            headers={"Content-Type": "application/json"},
            timeout=30
        )
        
        if response.status_code == 200:
            embedding = response.json()[0]
            return {
                "success": True,
                "embedding_size": len(embedding),
                "sample_values": embedding[:5],
                "response_time": response.elapsed.total_seconds()
            }
        else:
            return {
                "success": False,
                "error": f"Status {response.status_code}: {response.text}"
            }
    except Exception as e:
        return {"success": False, "error": str(e)}


def test_multimodal_embedding(base_url: str) -> dict:
    """Test multimodal embedding with image"""
    test_image = create_test_image()
    
    payload = {
        "inputs": {
            "text": "This is a test document with an image",
            "image": test_image,
            "input_type": "passage"
        },
        "normalize": True
    }
    
    try:
        response = requests.post(
            f"{base_url}/embed_multimodal",
            json=payload,
            headers={"Content-Type": "application/json"},
            timeout=30
        )
        
        if response.status_code == 200:
            embedding = response.json()[0]
            return {
                "success": True,
                "embedding_size": len(embedding),
                "sample_values": embedding[:5],
                "response_time": response.elapsed.total_seconds()
            }
        else:
            return {
                "success": False,
                "error": f"Status {response.status_code}: {response.text}"
            }
    except Exception as e:
        return {"success": False, "error": str(e)}


def test_batch_embedding(base_url: str) -> dict:
    """Test batch embedding"""
    test_image = create_test_image()
    
    payload = {
        "inputs": [
            {
                "text": "First test document",
                "input_type": "passage"
            },
            {
                "text": "Second test document with image",
                "image": test_image,
                "input_type": "passage"
            },
            {
                "text": "Third test document",
                "input_type": "query"
            }
        ],
        "normalize": True
    }
    
    try:
        response = requests.post(
            f"{base_url}/embed_multimodal",
            json=payload,
            headers={"Content-Type": "application/json"},
            timeout=60
        )
        
        if response.status_code == 200:
            embeddings = response.json()
            return {
                "success": True,
                "batch_size": len(embeddings),
                "embedding_sizes": [len(emb) for emb in embeddings],
                "response_time": response.elapsed.total_seconds()
            }
        else:
            return {
                "success": False,
                "error": f"Status {response.status_code}: {response.text}"
            }
    except Exception as e:
        return {"success": False, "error": str(e)}


def test_similarity_computation(base_url: str) -> dict:
    """Test similarity between embeddings"""
    # Get embeddings for similar texts
    similar_texts = [
        {"text": "Machine learning is a subset of AI", "input_type": "passage"},
        {"text": "AI includes machine learning techniques", "input_type": "passage"},
        {"text": "The weather is sunny today", "input_type": "passage"}
    ]
    
    embeddings = []
    
    try:
        for text_data in similar_texts:
            response = requests.post(
                f"{base_url}/embed_multimodal",
                json={"inputs": text_data, "normalize": True},
                headers={"Content-Type": "application/json"},
                timeout=30
            )
            
            if response.status_code == 200:
                embeddings.append(response.json()[0])
            else:
                return {
                    "success": False,
                    "error": f"Failed to get embedding: {response.status_code}"
                }
        
        # Compute similarities
        def cosine_similarity(a, b):
            dot_product = sum(x * y for x, y in zip(a, b))
            magnitude_a = sum(x * x for x in a) ** 0.5
            magnitude_b = sum(x * x for x in b) ** 0.5
            return dot_product / (magnitude_a * magnitude_b)
        
        sim_1_2 = cosine_similarity(embeddings[0], embeddings[1])  # Similar texts
        sim_1_3 = cosine_similarity(embeddings[0], embeddings[2])  # Different texts
        
        return {
            "success": True,
            "similarity_similar_texts": sim_1_2,
            "similarity_different_texts": sim_1_3,
            "similarity_difference": sim_1_2 - sim_1_3
        }
        
    except Exception as e:
        return {"success": False, "error": str(e)}


def run_integration_tests(base_url: str) -> dict:
    """Run all integration tests"""
    print("🧪 Running Jina VL Integration Tests")
    print("=" * 40)
    
    results = {}
    
    # Test 1: Service Health
    print("1️⃣ Testing service health...")
    health_ok = test_service_health(base_url)
    results["health"] = {"success": health_ok}
    
    if not health_ok:
        print("❌ Service is not running!")
        print("Please start the service with:")
        print(f"text-embeddings-router --model-id /data/jina-embeddings-v4-vllm-retrieval --json-output --payload-limit 2000000 --auto-truncate --max-client-batch-size 200")
        return results
    
    print("✅ Service is healthy")
    
    # Test 2: Model Info
    print("\n2️⃣ Testing model info...")
    info_result = test_model_info(base_url)
    results["model_info"] = info_result
    
    if "error" in info_result:
        print(f"❌ Model info failed: {info_result['error']}")
    else:
        print(f"✅ Model: {info_result.get('model_id', 'Unknown')}")
        print(f"   Type: {info_result.get('model_type', 'Unknown')}")
    
    # Test 3: Text Embedding
    print("\n3️⃣ Testing text embedding...")
    text_result = test_text_embedding(base_url)
    results["text_embedding"] = text_result
    
    if text_result["success"]:
        print(f"✅ Text embedding successful")
        print(f"   Embedding size: {text_result['embedding_size']}")
        print(f"   Response time: {text_result['response_time']:.3f}s")
    else:
        print(f"❌ Text embedding failed: {text_result['error']}")
    
    # Test 4: Multimodal Embedding
    print("\n4️⃣ Testing multimodal embedding...")
    mm_result = test_multimodal_embedding(base_url)
    results["multimodal_embedding"] = mm_result
    
    if mm_result["success"]:
        print(f"✅ Multimodal embedding successful")
        print(f"   Embedding size: {mm_result['embedding_size']}")
        print(f"   Response time: {mm_result['response_time']:.3f}s")
    else:
        print(f"❌ Multimodal embedding failed: {mm_result['error']}")
    
    # Test 5: Batch Embedding
    print("\n5️⃣ Testing batch embedding...")
    batch_result = test_batch_embedding(base_url)
    results["batch_embedding"] = batch_result
    
    if batch_result["success"]:
        print(f"✅ Batch embedding successful")
        print(f"   Batch size: {batch_result['batch_size']}")
        print(f"   Response time: {batch_result['response_time']:.3f}s")
    else:
        print(f"❌ Batch embedding failed: {batch_result['error']}")
    
    # Test 6: Similarity Computation
    print("\n6️⃣ Testing similarity computation...")
    sim_result = test_similarity_computation(base_url)
    results["similarity"] = sim_result
    
    if sim_result["success"]:
        print(f"✅ Similarity computation successful")
        print(f"   Similar texts similarity: {sim_result['similarity_similar_texts']:.4f}")
        print(f"   Different texts similarity: {sim_result['similarity_different_texts']:.4f}")
        print(f"   Difference: {sim_result['similarity_difference']:.4f}")
    else:
        print(f"❌ Similarity computation failed: {sim_result['error']}")
    
    return results


def print_summary(results: dict):
    """Print test summary"""
    print("\n" + "=" * 40)
    print("📊 TEST SUMMARY")
    print("=" * 40)
    
    total_tests = 0
    passed_tests = 0
    
    for test_name, result in results.items():
        total_tests += 1
        if result.get("success", False):
            passed_tests += 1
            status = "✅ PASS"
        else:
            status = "❌ FAIL"
        
        print(f"{status} {test_name.replace('_', ' ').title()}")
    
    print(f"\nResults: {passed_tests}/{total_tests} tests passed")
    
    if passed_tests == total_tests:
        print("🎉 All tests passed! The integration is working correctly.")
        return True
    else:
        print("⚠️ Some tests failed. Please check the service configuration.")
        return False


def main():
    parser = argparse.ArgumentParser(description="Test Jina VL integration")
    parser.add_argument("--base-url", default="http://localhost:3000",
                       help="Base URL of the service")
    parser.add_argument("--output", help="Output file for results (JSON)")
    
    args = parser.parse_args()
    
    # Run tests
    results = run_integration_tests(args.base_url)
    
    # Print summary
    success = print_summary(results)
    
    # Save results if requested
    if args.output:
        with open(args.output, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\n💾 Results saved to {args.output}")
    
    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())