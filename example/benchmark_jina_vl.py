#!/usr/bin/env python3
"""
Jina VL Performance Benchmark

This script benchmarks the performance of the jina-embeddings-v4-vllm-retrieval model
with different input types and batch sizes.

Usage:
    python benchmark_jina_vl.py [--base-url http://localhost:3000] [--max-batch-size 32]
"""

import argparse
import time
import statistics
import requests
import json
import base64
from typing import List, Dict, Any, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed
import io
from PIL import Image


class JinaVLBenchmark:
    """Benchmark suite for Jina VL multimodal embedding service"""
    
    def __init__(self, base_url: str = "http://localhost:3000"):
        self.base_url = base_url.rstrip('/')
        self.results = {}
        
    def create_sample_image(self, text: str, size: Tuple[int, int] = (200, 150)) -> str:
        """Create a sample image with text"""
        img = Image.new('RGB', size, color='lightblue')
        
        # Convert to base64
        buffer = io.BytesIO()
        img.save(buffer, format='JPEG', quality=85)
        encoded_string = base64.b64encode(buffer.getvalue()).decode('utf-8')
        return f"data:image/jpeg;base64,{encoded_string}"
    
    def generate_test_data(self, count: int, include_images: bool = False) -> List[Dict[str, Any]]:
        """Generate test data"""
        data = []
        
        for i in range(count):
            item = {
                "text": f"This is test document number {i+1} with some sample content for embedding generation.",
                "input_type": "passage" if i % 2 == 0 else "query"
            }
            
            if include_images and i % 3 == 0:  # Add image to every 3rd item
                item["image"] = self.create_sample_image(f"Image {i+1}")
            
            data.append(item)
        
        return data
    
    def embed_single(self, input_data: Dict[str, Any]) -> Tuple[float, int]:
        """Embed single input and return (latency, embedding_size)"""
        start_time = time.time()
        
        response = requests.post(
            f"{self.base_url}/embed_multimodal",
            json={"inputs": input_data, "normalize": True},
            headers={"Content-Type": "application/json"}
        )
        
        latency = time.time() - start_time
        
        if response.status_code == 200:
            embedding = response.json()[0]
            return latency, len(embedding)
        else:
            raise Exception(f"Error: {response.status_code} - {response.text}")
    
    def embed_batch(self, inputs: List[Dict[str, Any]]) -> Tuple[float, int, int]:
        """Embed batch and return (latency, batch_size, embedding_size)"""
        start_time = time.time()
        
        response = requests.post(
            f"{self.base_url}/embed_multimodal",
            json={"inputs": inputs, "normalize": True},
            headers={"Content-Type": "application/json"}
        )
        
        latency = time.time() - start_time
        
        if response.status_code == 200:
            embeddings = response.json()
            return latency, len(embeddings), len(embeddings[0]) if embeddings else 0
        else:
            raise Exception(f"Error: {response.status_code} - {response.text}")
    
    def benchmark_single_requests(self, num_requests: int = 50) -> Dict[str, Any]:
        """Benchmark single requests"""
        print(f"🔄 Benchmarking {num_requests} single requests...")
        
        # Test text-only
        text_latencies = []
        for i in range(num_requests):
            data = {"text": f"Sample text {i}", "input_type": "passage"}
            latency, emb_size = self.embed_single(data)
            text_latencies.append(latency)
        
        # Test multimodal
        multimodal_latencies = []
        sample_image = self.create_sample_image("Sample")
        for i in range(min(num_requests, 20)):  # Fewer multimodal tests
            data = {
                "text": f"Sample text with image {i}", 
                "image": sample_image,
                "input_type": "passage"
            }
            latency, emb_size = self.embed_single(data)
            multimodal_latencies.append(latency)
        
        return {
            "text_only": {
                "count": len(text_latencies),
                "mean_latency": statistics.mean(text_latencies),
                "median_latency": statistics.median(text_latencies),
                "min_latency": min(text_latencies),
                "max_latency": max(text_latencies),
                "std_latency": statistics.stdev(text_latencies) if len(text_latencies) > 1 else 0,
                "throughput": len(text_latencies) / sum(text_latencies),
            },
            "multimodal": {
                "count": len(multimodal_latencies),
                "mean_latency": statistics.mean(multimodal_latencies),
                "median_latency": statistics.median(multimodal_latencies),
                "min_latency": min(multimodal_latencies),
                "max_latency": max(multimodal_latencies),
                "std_latency": statistics.stdev(multimodal_latencies) if len(multimodal_latencies) > 1 else 0,
                "throughput": len(multimodal_latencies) / sum(multimodal_latencies),
            },
            "embedding_size": emb_size
        }
    
    def benchmark_batch_sizes(self, batch_sizes: List[int] = [1, 2, 4, 8, 16, 32]) -> Dict[str, Any]:
        """Benchmark different batch sizes"""
        print(f"📦 Benchmarking batch sizes: {batch_sizes}")
        
        results = {}
        
        for batch_size in batch_sizes:
            print(f"  Testing batch size {batch_size}...")
            
            # Text-only batches
            text_data = self.generate_test_data(batch_size, include_images=False)
            text_latencies = []
            
            for _ in range(5):  # 5 runs per batch size
                try:
                    latency, actual_batch_size, emb_size = self.embed_batch(text_data)
                    text_latencies.append(latency)
                except Exception as e:
                    print(f"    Error with text batch size {batch_size}: {e}")
                    break
            
            # Multimodal batches (smaller sizes)
            multimodal_latencies = []
            if batch_size <= 8:  # Limit multimodal batch size
                multimodal_data = self.generate_test_data(batch_size, include_images=True)
                
                for _ in range(3):  # Fewer runs for multimodal
                    try:
                        latency, actual_batch_size, emb_size = self.embed_batch(multimodal_data)
                        multimodal_latencies.append(latency)
                    except Exception as e:
                        print(f"    Error with multimodal batch size {batch_size}: {e}")
                        break
            
            if text_latencies:
                results[batch_size] = {
                    "text_only": {
                        "mean_latency": statistics.mean(text_latencies),
                        "throughput": batch_size / statistics.mean(text_latencies),
                        "items_per_second": batch_size / statistics.mean(text_latencies),
                    }
                }
                
                if multimodal_latencies:
                    results[batch_size]["multimodal"] = {
                        "mean_latency": statistics.mean(multimodal_latencies),
                        "throughput": batch_size / statistics.mean(multimodal_latencies),
                        "items_per_second": batch_size / statistics.mean(multimodal_latencies),
                    }
        
        return results
    
    def benchmark_concurrent_requests(self, num_threads: int = 4, requests_per_thread: int = 10) -> Dict[str, Any]:
        """Benchmark concurrent requests"""
        print(f"🔀 Benchmarking {num_threads} concurrent threads with {requests_per_thread} requests each...")
        
        def worker_thread(thread_id: int) -> List[float]:
            latencies = []
            for i in range(requests_per_thread):
                data = {"text": f"Thread {thread_id} request {i}", "input_type": "passage"}
                try:
                    latency, _ = self.embed_single(data)
                    latencies.append(latency)
                except Exception as e:
                    print(f"    Thread {thread_id} error: {e}")
            return latencies
        
        start_time = time.time()
        all_latencies = []
        
        with ThreadPoolExecutor(max_workers=num_threads) as executor:
            futures = [executor.submit(worker_thread, i) for i in range(num_threads)]
            
            for future in as_completed(futures):
                thread_latencies = future.result()
                all_latencies.extend(thread_latencies)
        
        total_time = time.time() - start_time
        total_requests = len(all_latencies)
        
        return {
            "total_requests": total_requests,
            "total_time": total_time,
            "overall_throughput": total_requests / total_time,
            "mean_latency": statistics.mean(all_latencies),
            "median_latency": statistics.median(all_latencies),
            "p95_latency": sorted(all_latencies)[int(0.95 * len(all_latencies))],
            "p99_latency": sorted(all_latencies)[int(0.99 * len(all_latencies))],
        }
    
    def benchmark_image_sizes(self) -> Dict[str, Any]:
        """Benchmark different image sizes"""
        print("🖼️ Benchmarking different image sizes...")
        
        image_sizes = [
            (100, 100),
            (200, 200),
            (400, 400),
            (800, 600),
        ]
        
        results = {}
        
        for width, height in image_sizes:
            print(f"  Testing image size {width}x{height}...")
            
            image = self.create_sample_image(f"Test {width}x{height}", (width, height))
            data = {
                "text": f"Image with size {width}x{height}",
                "image": image,
                "input_type": "passage"
            }
            
            latencies = []
            for _ in range(5):
                try:
                    latency, emb_size = self.embed_single(data)
                    latencies.append(latency)
                except Exception as e:
                    print(f"    Error with size {width}x{height}: {e}")
                    break
            
            if latencies:
                # Estimate image size in bytes
                image_data = image.split(',')[1]  # Remove data URL prefix
                image_bytes = len(base64.b64decode(image_data))
                
                results[f"{width}x{height}"] = {
                    "mean_latency": statistics.mean(latencies),
                    "image_size_bytes": image_bytes,
                    "image_size_kb": image_bytes / 1024,
                }
        
        return results
    
    def run_full_benchmark(self, max_batch_size: int = 32) -> Dict[str, Any]:
        """Run complete benchmark suite"""
        print("🚀 Starting Jina VL Performance Benchmark")
        print("=" * 50)
        
        # Check service health
        try:
            response = requests.get(f"{self.base_url}/health", timeout=5)
            if response.status_code != 200:
                raise Exception("Service unhealthy")
            print("✅ Service is healthy")
        except Exception as e:
            print(f"❌ Service check failed: {e}")
            return {}
        
        results = {}
        
        # Single request benchmark
        try:
            results["single_requests"] = self.benchmark_single_requests()
        except Exception as e:
            print(f"❌ Single request benchmark failed: {e}")
        
        # Batch size benchmark
        try:
            batch_sizes = [1, 2, 4, 8, 16, min(32, max_batch_size)]
            results["batch_sizes"] = self.benchmark_batch_sizes(batch_sizes)
        except Exception as e:
            print(f"❌ Batch size benchmark failed: {e}")
        
        # Concurrent requests benchmark
        try:
            results["concurrent_requests"] = self.benchmark_concurrent_requests()
        except Exception as e:
            print(f"❌ Concurrent requests benchmark failed: {e}")
        
        # Image size benchmark
        try:
            results["image_sizes"] = self.benchmark_image_sizes()
        except Exception as e:
            print(f"❌ Image size benchmark failed: {e}")
        
        return results
    
    def print_results(self, results: Dict[str, Any]):
        """Print benchmark results in a readable format"""
        print("\n" + "=" * 60)
        print("📊 BENCHMARK RESULTS")
        print("=" * 60)
        
        # Single requests
        if "single_requests" in results:
            single = results["single_requests"]
            print(f"\n🔤 Single Request Performance:")
            print(f"  Text-only:")
            print(f"    Mean latency: {single['text_only']['mean_latency']:.3f}s")
            print(f"    Throughput: {single['text_only']['throughput']:.1f} req/s")
            print(f"  Multimodal:")
            print(f"    Mean latency: {single['multimodal']['mean_latency']:.3f}s")
            print(f"    Throughput: {single['multimodal']['throughput']:.1f} req/s")
            print(f"  Embedding size: {single['embedding_size']} dimensions")
        
        # Batch sizes
        if "batch_sizes" in results:
            batch = results["batch_sizes"]
            print(f"\n📦 Batch Performance:")
            print(f"  {'Batch Size':<12} {'Text Latency':<15} {'Text Throughput':<18} {'MM Latency':<15} {'MM Throughput':<15}")
            print(f"  {'-'*12} {'-'*15} {'-'*18} {'-'*15} {'-'*15}")
            
            for batch_size, data in batch.items():
                text_lat = f"{data['text_only']['mean_latency']:.3f}s"
                text_thr = f"{data['text_only']['items_per_second']:.1f} it/s"
                
                if "multimodal" in data:
                    mm_lat = f"{data['multimodal']['mean_latency']:.3f}s"
                    mm_thr = f"{data['multimodal']['items_per_second']:.1f} it/s"
                else:
                    mm_lat = "N/A"
                    mm_thr = "N/A"
                
                print(f"  {batch_size:<12} {text_lat:<15} {text_thr:<18} {mm_lat:<15} {mm_thr:<15}")
        
        # Concurrent requests
        if "concurrent_requests" in results:
            conc = results["concurrent_requests"]
            print(f"\n🔀 Concurrent Performance:")
            print(f"  Total requests: {conc['total_requests']}")
            print(f"  Overall throughput: {conc['overall_throughput']:.1f} req/s")
            print(f"  Mean latency: {conc['mean_latency']:.3f}s")
            print(f"  P95 latency: {conc['p95_latency']:.3f}s")
            print(f"  P99 latency: {conc['p99_latency']:.3f}s")
        
        # Image sizes
        if "image_sizes" in results:
            img = results["image_sizes"]
            print(f"\n🖼️ Image Size Performance:")
            print(f"  {'Size':<10} {'Latency':<12} {'Image KB':<12}")
            print(f"  {'-'*10} {'-'*12} {'-'*12}")
            
            for size, data in img.items():
                lat = f"{data['mean_latency']:.3f}s"
                kb = f"{data['image_size_kb']:.1f} KB"
                print(f"  {size:<10} {lat:<12} {kb:<12}")
        
        print(f"\n🎯 Summary:")
        if "single_requests" in results:
            text_thr = results["single_requests"]["text_only"]["throughput"]
            mm_thr = results["single_requests"]["multimodal"]["throughput"]
            print(f"  - Text-only throughput: {text_thr:.1f} req/s")
            print(f"  - Multimodal throughput: {mm_thr:.1f} req/s")
        
        if "batch_sizes" in results:
            best_batch = max(results["batch_sizes"].items(), 
                           key=lambda x: x[1]["text_only"]["items_per_second"])
            print(f"  - Best batch size: {best_batch[0]} ({best_batch[1]['text_only']['items_per_second']:.1f} it/s)")
        
        print(f"\n💡 Recommendations:")
        print(f"  - Use batch processing for better throughput")
        print(f"  - Optimize image sizes (200x200 to 400x400 recommended)")
        print(f"  - Consider concurrent requests for high-load scenarios")
        print(f"  - Monitor GPU memory usage with large batches")


def main():
    parser = argparse.ArgumentParser(description="Benchmark Jina VL multimodal embedding service")
    parser.add_argument("--base-url", default="http://localhost:3000", 
                       help="Base URL of the service")
    parser.add_argument("--max-batch-size", type=int, default=32,
                       help="Maximum batch size to test")
    parser.add_argument("--output", help="Output file for results (JSON)")
    
    args = parser.parse_args()
    
    benchmark = JinaVLBenchmark(args.base_url)
    results = benchmark.run_full_benchmark(args.max_batch_size)
    
    if results:
        benchmark.print_results(results)
        
        if args.output:
            with open(args.output, 'w') as f:
                json.dump(results, f, indent=2)
            print(f"\n💾 Results saved to {args.output}")
    else:
        print("❌ Benchmark failed to run")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())