#!/bin/bash

# Jina VL Multimodal Embedding cURL Examples
# 
# This script demonstrates how to use the jina-embeddings-v4-vllm-retrieval model
# with text-embeddings-router using cURL commands.
#
# Prerequisites:
# 1. Start the service:
#    text-embeddings-router --model-id /data/jina-embeddings-v4-vllm-retrieval \
#                          --json-output --payload-limit 2000000 \
#                          --auto-truncate --max-client-batch-size 200
#
# 2. Install jq for JSON processing: sudo apt-get install jq

set -e

BASE_URL="http://localhost:3000"

echo "🚀 Jina VL Multimodal Embedding cURL Examples"
echo "=============================================="

# Function to check service health
check_health() {
    echo "🔍 Checking service health..."
    if curl -s -f "$BASE_URL/health" > /dev/null; then
        echo "✅ Service is running!"
        return 0
    else
        echo "❌ Service is not running!"
        echo "Please start the service with:"
        echo "text-embeddings-router --model-id /data/jina-embeddings-v4-vllm-retrieval --json-output --payload-limit 2000000 --auto-truncate --max-client-batch-size 200"
        return 1
    fi
}

# Function to create a simple base64 encoded image
create_sample_image() {
    # Create a simple 1x1 pixel PNG image in base64
    echo "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNkYPhfDwAChAI9jU77zgAAAABJRU5ErkJggg=="
}

# Example 1: Text-only embedding
example_text_only() {
    echo ""
    echo "📝 Example 1: Text-only Embedding"
    echo "================================="
    
    echo "Embedding a simple text..."
    
    curl -X POST "$BASE_URL/embed_multimodal" \
         -H "Content-Type: application/json" \
         -d '{
           "inputs": {
             "text": "This is a sample text for embedding",
             "input_type": "passage"
           },
           "normalize": true
         }' | jq -r '.[0][:5]' | head -5
    
    echo "✅ Text embedding completed (showing first 5 dimensions)"
}

# Example 2: Query embedding
example_query_embedding() {
    echo ""
    echo "🔍 Example 2: Query Embedding"
    echo "============================="
    
    echo "Embedding a query..."
    
    curl -X POST "$BASE_URL/embed_multimodal" \
         -H "Content-Type: application/json" \
         -d '{
           "inputs": {
             "text": "What is machine learning?",
             "input_type": "query"
           },
           "normalize": true
         }' | jq -r 'length'
    
    echo "✅ Query embedding completed"
}

# Example 3: Multimodal embedding (text + image)
example_multimodal() {
    echo ""
    echo "🖼️ Example 3: Multimodal Embedding (Text + Image)"
    echo "================================================="
    
    echo "Embedding text with image..."
    
    SAMPLE_IMAGE=$(create_sample_image)
    
    curl -X POST "$BASE_URL/embed_multimodal" \
         -H "Content-Type: application/json" \
         -d "{
           \"inputs\": {
             \"text\": \"A sample image with descriptive text\",
             \"image\": \"data:image/png;base64,$SAMPLE_IMAGE\",
             \"input_type\": \"passage\"
           },
           \"normalize\": true
         }" | jq -r 'length'
    
    echo "✅ Multimodal embedding completed"
}

# Example 4: Batch processing
example_batch() {
    echo ""
    echo "📦 Example 4: Batch Processing"
    echo "=============================="
    
    echo "Processing batch of mixed inputs..."
    
    SAMPLE_IMAGE=$(create_sample_image)
    
    curl -X POST "$BASE_URL/embed_multimodal" \
         -H "Content-Type: application/json" \
         -d "{
           \"inputs\": [
             {
               \"text\": \"First text document\",
               \"input_type\": \"passage\"
             },
             {
               \"text\": \"Second document with image\",
               \"image\": \"data:image/png;base64,$SAMPLE_IMAGE\",
               \"input_type\": \"passage\"
             },
             {
               \"text\": \"Third text-only document\",
               \"input_type\": \"passage\"
             }
           ],
           \"normalize\": true
         }" | jq -r 'length'
    
    echo "✅ Batch processing completed"
}

# Example 5: Similarity computation
example_similarity() {
    echo ""
    echo "🔗 Example 5: Similarity Computation"
    echo "===================================="
    
    echo "Computing similarity between query and passages..."
    
    # Get query embedding
    QUERY_EMBEDDING=$(curl -s -X POST "$BASE_URL/embed_multimodal" \
         -H "Content-Type: application/json" \
         -d '{
           "inputs": {
             "text": "machine learning algorithms",
             "input_type": "query"
           },
           "normalize": true
         }')
    
    # Get passage embeddings
    PASSAGE1_EMBEDDING=$(curl -s -X POST "$BASE_URL/embed_multimodal" \
         -H "Content-Type: application/json" \
         -d '{
           "inputs": {
             "text": "Machine learning is a subset of artificial intelligence",
             "input_type": "passage"
           },
           "normalize": true
         }')
    
    PASSAGE2_EMBEDDING=$(curl -s -X POST "$BASE_URL/embed_multimodal" \
         -H "Content-Type: application/json" \
         -d '{
           "inputs": {
             "text": "The weather is sunny today",
             "input_type": "passage"
           },
           "normalize": true
         }')
    
    echo "Query: 'machine learning algorithms'"
    echo "Passage 1: 'Machine learning is a subset of artificial intelligence'"
    echo "Passage 2: 'The weather is sunny today'"
    echo ""
    echo "Embeddings obtained. Use the /similarity endpoint to compute similarities."
    echo "✅ Similarity example completed"
}

# Example 6: Model info
example_model_info() {
    echo ""
    echo "ℹ️ Example 6: Model Information"
    echo "==============================="
    
    echo "Getting model information..."
    
    curl -s "$BASE_URL/info" | jq '{
        model_id: .model_id,
        model_type: .model_type,
        max_input_length: .max_input_length,
        max_client_batch_size: .max_client_batch_size
    }'
    
    echo "✅ Model info retrieved"
}

# Example 7: Health check
example_health_check() {
    echo ""
    echo "🏥 Example 7: Health Check"
    echo "=========================="
    
    echo "Checking service health..."
    
    if curl -s -f "$BASE_URL/health" > /dev/null; then
        echo "✅ Service is healthy"
    else
        echo "❌ Service is unhealthy"
    fi
}

# Main execution
main() {
    # Check if service is running
    if ! check_health; then
        exit 1
    fi
    
    # Run all examples
    example_text_only
    example_query_embedding
    example_multimodal
    example_batch
    example_similarity
    example_model_info
    example_health_check
    
    echo ""
    echo "🎉 All cURL examples completed!"
    echo ""
    echo "💡 Tips:"
    echo "  - Use 'input_type': 'query' for search queries"
    echo "  - Use 'input_type': 'passage' for documents to be searched"
    echo "  - Include base64-encoded images in the 'image' field for multimodal inputs"
    echo "  - Set 'normalize': true for better similarity computation"
    echo "  - Use batch processing for better performance with multiple inputs"
}

# Run main function
main "$@"