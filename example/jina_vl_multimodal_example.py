#!/usr/bin/env python3
"""
Jina VL Multimodal Embedding Example

This script demonstrates how to use the jina-embeddings-v4-vllm-retrieval model
for multimodal embeddings with text-embeddings-router.

Usage:
    python jina_vl_multimodal_example.py
"""

import requests
import json
import base64
from PIL import Image
import io
from typing import List, Dict, Any, Optional


class JinaVLClient:
    """Client for Jina VL multimodal embedding service"""
    
    def __init__(self, base_url: str = "http://localhost:3000"):
        self.base_url = base_url.rstrip('/')
        
    def encode_image_to_base64(self, image_path: str) -> str:
        """Encode image file to base64 string"""
        with open(image_path, "rb") as image_file:
            encoded_string = base64.b64encode(image_file.read()).decode('utf-8')
            return f"data:image/jpeg;base64,{encoded_string}"
    
    def create_image_from_text(self, text: str, size: tuple = (400, 300)) -> str:
        """Create a simple image with text for demonstration"""
        from PIL import Image, ImageDraw, ImageFont
        
        # Create a simple image with text
        img = Image.new('RGB', size, color='white')
        draw = ImageDraw.Draw(img)
        
        try:
            # Try to use a default font
            font = ImageFont.load_default()
        except:
            font = None
        
        # Draw text on image
        draw.text((10, 10), text, fill='black', font=font)
        
        # Convert to base64
        buffer = io.BytesIO()
        img.save(buffer, format='JPEG')
        encoded_string = base64.b64encode(buffer.getvalue()).decode('utf-8')
        return f"data:image/jpeg;base64,{encoded_string}"
    
    def embed_text(self, text: str, input_type: str = "passage") -> List[float]:
        """Embed text only"""
        payload = {
            "inputs": {
                "text": text,
                "input_type": input_type
            }
        }
        
        response = requests.post(
            f"{self.base_url}/embed_multimodal",
            json=payload,
            headers={"Content-Type": "application/json"}
        )
        
        if response.status_code == 200:
            return response.json()[0]
        else:
            raise Exception(f"Error: {response.status_code} - {response.text}")
    
    def embed_text_with_image(
        self, 
        text: str, 
        image_base64: str, 
        input_type: str = "passage"
    ) -> List[float]:
        """Embed text with image"""
        payload = {
            "inputs": {
                "text": text,
                "image": image_base64,
                "input_type": input_type
            }
        }
        
        response = requests.post(
            f"{self.base_url}/embed_multimodal",
            json=payload,
            headers={"Content-Type": "application/json"}
        )
        
        if response.status_code == 200:
            return response.json()[0]
        else:
            raise Exception(f"Error: {response.status_code} - {response.text}")
    
    def embed_batch(self, inputs: List[Dict[str, Any]]) -> List[List[float]]:
        """Embed batch of multimodal inputs"""
        payload = {"inputs": inputs}
        
        response = requests.post(
            f"{self.base_url}/embed_multimodal",
            json=payload,
            headers={"Content-Type": "application/json"}
        )
        
        if response.status_code == 200:
            return response.json()
        else:
            raise Exception(f"Error: {response.status_code} - {response.text}")
    
    def compute_similarity(self, embedding1: List[float], embedding2: List[float]) -> float:
        """Compute cosine similarity between two embeddings"""
        import math
        
        # Compute dot product
        dot_product = sum(a * b for a, b in zip(embedding1, embedding2))
        
        # Compute magnitudes
        magnitude1 = math.sqrt(sum(a * a for a in embedding1))
        magnitude2 = math.sqrt(sum(b * b for b in embedding2))
        
        # Compute cosine similarity
        if magnitude1 == 0 or magnitude2 == 0:
            return 0.0
        
        return dot_product / (magnitude1 * magnitude2)


def example_text_only_embedding():
    """Example 1: Text-only embedding"""
    print("🔤 Example 1: Text-only Embedding")
    print("=" * 50)
    
    client = JinaVLClient()
    
    # Text inputs
    query = "What is machine learning?"
    passages = [
        "Machine learning is a subset of artificial intelligence that enables computers to learn from data.",
        "The weather today is sunny with a chance of rain.",
        "Deep learning uses neural networks with multiple layers to model complex patterns.",
    ]
    
    try:
        # Embed query
        query_embedding = client.embed_text(query, input_type="query")
        print(f"Query: {query}")
        print(f"Query embedding shape: {len(query_embedding)}")
        
        # Embed passages
        similarities = []
        for i, passage in enumerate(passages):
            passage_embedding = client.embed_text(passage, input_type="passage")
            similarity = client.compute_similarity(query_embedding, passage_embedding)
            similarities.append((i, similarity, passage))
            print(f"Passage {i+1}: {passage[:50]}...")
            print(f"  Similarity: {similarity:.4f}")
        
        # Sort by similarity
        similarities.sort(key=lambda x: x[1], reverse=True)
        print(f"\n🏆 Most relevant passage:")
        print(f"  {similarities[0][2]}")
        print(f"  Similarity: {similarities[0][1]:.4f}")
        
    except Exception as e:
        print(f"❌ Error: {e}")


def example_multimodal_embedding():
    """Example 2: Multimodal embedding with images"""
    print("\n🖼️ Example 2: Multimodal Embedding")
    print("=" * 50)
    
    client = JinaVLClient()
    
    # Create sample images with text
    image1 = client.create_image_from_text("A cat sitting on a chair")
    image2 = client.create_image_from_text("A dog running in the park")
    image3 = client.create_image_from_text("A bird flying in the sky")
    
    query = "Show me a pet animal"
    
    try:
        # Embed query (text only)
        query_embedding = client.embed_text(query, input_type="query")
        print(f"Query: {query}")
        
        # Embed multimodal inputs
        multimodal_inputs = [
            {"text": "A cute cat", "image": image1, "input_type": "passage"},
            {"text": "A playful dog", "image": image2, "input_type": "passage"},
            {"text": "A flying bird", "image": image3, "input_type": "passage"},
        ]
        
        similarities = []
        for i, input_data in enumerate(multimodal_inputs):
            embedding = client.embed_text_with_image(
                input_data["text"], 
                input_data["image"], 
                input_data["input_type"]
            )
            similarity = client.compute_similarity(query_embedding, embedding)
            similarities.append((i, similarity, input_data["text"]))
            print(f"Input {i+1}: {input_data['text']}")
            print(f"  Similarity: {similarity:.4f}")
        
        # Sort by similarity
        similarities.sort(key=lambda x: x[1], reverse=True)
        print(f"\n🏆 Most relevant multimodal input:")
        print(f"  {similarities[0][2]}")
        print(f"  Similarity: {similarities[0][1]:.4f}")
        
    except Exception as e:
        print(f"❌ Error: {e}")


def example_batch_processing():
    """Example 3: Batch processing"""
    print("\n📦 Example 3: Batch Processing")
    print("=" * 50)
    
    client = JinaVLClient()
    
    # Create sample image
    sample_image = client.create_image_from_text("Sample product image")
    
    # Batch inputs (mix of text-only and multimodal)
    batch_inputs = [
        {"text": "Product description for item A", "input_type": "passage"},
        {"text": "Product description for item B", "image": sample_image, "input_type": "passage"},
        {"text": "User review of the product", "input_type": "passage"},
        {"text": "Technical specifications", "image": sample_image, "input_type": "passage"},
    ]
    
    try:
        # Process batch
        embeddings = client.embed_batch(batch_inputs)
        
        print(f"Processed {len(embeddings)} inputs in batch")
        for i, embedding in enumerate(embeddings):
            has_image = "image" in batch_inputs[i]
            print(f"Input {i+1}: {batch_inputs[i]['text'][:30]}...")
            print(f"  Type: {'Multimodal' if has_image else 'Text-only'}")
            print(f"  Embedding shape: {len(embedding)}")
        
        # Compute similarities within batch
        print(f"\n🔗 Pairwise similarities:")
        for i in range(len(embeddings)):
            for j in range(i+1, len(embeddings)):
                similarity = client.compute_similarity(embeddings[i], embeddings[j])
                print(f"  Input {i+1} ↔ Input {j+1}: {similarity:.4f}")
        
    except Exception as e:
        print(f"❌ Error: {e}")


def example_retrieval_task():
    """Example 4: Document retrieval with multimodal content"""
    print("\n🔍 Example 4: Multimodal Document Retrieval")
    print("=" * 50)
    
    client = JinaVLClient()
    
    # Create document images
    doc_images = [
        client.create_image_from_text("Chart showing sales data"),
        client.create_image_from_text("Photo of a conference room"),
        client.create_image_from_text("Diagram of system architecture"),
    ]
    
    # Documents with text and images
    documents = [
        {
            "text": "Q3 sales report showing 15% growth in revenue",
            "image": doc_images[0],
            "input_type": "passage"
        },
        {
            "text": "Meeting notes from the quarterly review session",
            "image": doc_images[1],
            "input_type": "passage"
        },
        {
            "text": "Technical documentation for the new system design",
            "image": doc_images[2],
            "input_type": "passage"
        },
    ]
    
    queries = [
        "Show me financial performance data",
        "Find information about meetings",
        "I need technical documentation",
    ]
    
    try:
        print("📚 Documents:")
        for i, doc in enumerate(documents):
            print(f"  {i+1}. {doc['text']}")
        
        print(f"\n🔍 Queries and Results:")
        
        for query in queries:
            print(f"\nQuery: '{query}'")
            
            # Embed query
            query_embedding = client.embed_text(query, input_type="query")
            
            # Embed documents and compute similarities
            similarities = []
            for i, doc in enumerate(documents):
                doc_embedding = client.embed_text_with_image(
                    doc["text"], 
                    doc["image"], 
                    doc["input_type"]
                )
                similarity = client.compute_similarity(query_embedding, doc_embedding)
                similarities.append((i, similarity, doc["text"]))
            
            # Sort by similarity
            similarities.sort(key=lambda x: x[1], reverse=True)
            
            print(f"  🏆 Best match: Document {similarities[0][0]+1}")
            print(f"     Text: {similarities[0][2]}")
            print(f"     Similarity: {similarities[0][1]:.4f}")
            
            print(f"  📊 All similarities:")
            for idx, sim, text in similarities:
                print(f"     Doc {idx+1}: {sim:.4f}")
        
    except Exception as e:
        print(f"❌ Error: {e}")


def check_service_health():
    """Check if the service is running"""
    try:
        response = requests.get("http://localhost:3000/health", timeout=5)
        return response.status_code == 200
    except:
        return False


def main():
    """Run all examples"""
    print("🚀 Jina VL Multimodal Embedding Examples")
    print("=" * 60)
    
    # Check service health
    if not check_service_health():
        print("❌ Service is not running!")
        print("Please start the service with:")
        print("text-embeddings-router --model-id /data/jina-embeddings-v4-vllm-retrieval --json-output --payload-limit 2000000 --auto-truncate --max-client-batch-size 200")
        return
    
    print("✅ Service is running!")
    
    # Run examples
    try:
        example_text_only_embedding()
        example_multimodal_embedding()
        example_batch_processing()
        example_retrieval_task()
        
        print(f"\n🎉 All examples completed successfully!")
        
    except KeyboardInterrupt:
        print(f"\n⏹️ Examples interrupted by user")
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")


if __name__ == "__main__":
    main()