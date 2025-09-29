import torch
import torch.nn.functional as F
from typing import List, Optional, Union, Dict, Any
from PIL import Image
import base64
import io
from transformers import AutoTokenizer, AutoConfig, AutoModel, AutoProcessor
from text_embeddings_server.models.model import Model
from text_embeddings_server.models.types import Batch, Embedding


class VisionPooler:
    """
    Vision-aware pooler for multimodal embeddings
    Based on Jina-embed-v4 VisionPooler implementation
    """
    
    def __init__(self, config):
        self.config = config
        self.vision_start_token_id = getattr(config, 'vision_start_token_id', 151652)
        self.vision_end_token_id = getattr(config, 'vision_end_token_id', 151653)
        self.hidden_size = getattr(config, 'hidden_size', 2048)
    
    def pool_vision_tokens(self, hidden_states: torch.Tensor, token_ids: torch.Tensor) -> torch.Tensor:
        """
        Pool vision tokens from hidden states based on vision start/end tokens
        This implementation matches the vLLM VisionPooler behavior
        """
        # Find vision token positions
        vision_start_positions = (token_ids == self.vision_start_token_id).nonzero(as_tuple=True)[0]
        vision_end_positions = (token_ids == self.vision_end_token_id).nonzero(as_tuple=True)[0]
        
        if len(vision_start_positions) > 0 and len(vision_end_positions) > 0:
            # Use the last vision token range (matching vLLM behavior)
            start_pos = vision_start_positions[-1].item()
            end_pos = vision_end_positions[-1].item()
            
            # Extract vision embeddings and pool them (mean pooling)
            # This matches the vLLM implementation: mean pooling between start and end tokens
            vision_embeddings = hidden_states[start_pos:end_pos + 1]
            pooled_embedding = vision_embeddings.mean(dim=0, dtype=torch.float32)
        else:
            # For text-only inputs, use last token pooling (matching Jina behavior)
            # This is different from simple mean pooling
            pooled_embedding = hidden_states[-1].to(torch.float32)
        
        return pooled_embedding


class JinaVLModel(Model):
    """
    Jina Vision-Language Embedding Model
    
    This model supports multimodal embeddings for both text and images
    using the Qwen2_5_VLForConditionalGeneration architecture.
    Based on Jina-embed-v4 implementation without vLLM dependency.
    """
    
    def __init__(
        self,
        model_path: str,
        device: torch.device,
        dtype: torch.dtype,
        **kwargs
    ):
        self.device = device
        self.dtype = dtype
        self.model_path = model_path
        
        # Load tokenizer, config, and model
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        self.config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
        
        # Load the model
        self.model = AutoModel.from_pretrained(
            model_path,
            torch_dtype=dtype,
            device_map=device,
            trust_remote_code=True,
            **kwargs
        )
        self.model.eval()
        
        # Load processor for multimodal inputs
        try:
            self.processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
        except Exception:
            # Fallback if processor is not available
            self.processor = None
        
        # Vision token IDs
        self.vision_start_token_id = getattr(self.config, 'vision_start_token_id', 151652)
        self.vision_end_token_id = getattr(self.config, 'vision_end_token_id', 151653)
        self.image_token_id = getattr(self.config, 'image_token_id', 151655)
        
        # Initialize vision pooler
        self.vision_pooler = VisionPooler(self.config)
        
        # Set model properties
        self.max_input_length = getattr(self.config, 'max_position_embeddings', 131072)
        self.hidden_size = getattr(self.config, 'hidden_size', 2048)
    
    @property
    def batch_type(self) -> type:
        return Batch
    
    @property
    def max_batch_size(self) -> Optional[int]:
        return 32  # Reasonable default for multimodal models
    
    def embed(self, batch: Batch) -> List[Embedding]:
        """
        Generate embeddings for a batch of inputs (text and/or images)
        """
        embeddings = []
        
        with torch.no_grad():
            for i, input_ids in enumerate(batch.input_ids):
                # Convert input_ids to tensor if needed
                if not isinstance(input_ids, torch.Tensor):
                    input_ids = torch.tensor(input_ids, device=self.device)
                else:
                    input_ids = input_ids.to(self.device)
                
                # Prepare inputs
                model_inputs = {"input_ids": input_ids.unsqueeze(0)}
                
                # Add attention mask
                attention_mask = torch.ones_like(input_ids)
                model_inputs["attention_mask"] = attention_mask.unsqueeze(0)
                
                # Handle images if present
                if hasattr(batch, 'images') and batch.images and i < len(batch.images):
                    image_data = batch.images[i]
                    if image_data:
                        image = self._process_image(image_data)
                        if image is not None and self.processor is not None:
                            # Use processor to handle multimodal inputs
                            text = self.tokenizer.decode(input_ids, skip_special_tokens=False)
                            processed = self.processor(
                                text=text,
                                images=image,
                                return_tensors="pt",
                                padding=True
                            )
                            # Update model inputs with processed data
                            for key, value in processed.items():
                                if isinstance(value, torch.Tensor):
                                    model_inputs[key] = value.to(self.device)
                
                # Forward pass
                try:
                    outputs = self.model(**model_inputs, output_hidden_states=True)
                    hidden_states = outputs.hidden_states[-1]  # Last layer
                    
                    # Extract embedding for this sequence
                    sequence_hidden = hidden_states[0]  # Remove batch dimension
                    
                    # Always use vision-aware pooling for Jina VL model
                    # This matches the vLLM implementation
                    embedding = self.vision_pooler.pool_vision_tokens(
                        sequence_hidden, input_ids
                    )
                    
                    # Normalize embedding (important for consistency with vLLM)
                    embedding = F.normalize(embedding, p=2, dim=-1)
                    embeddings.append(Embedding(values=embedding.cpu().tolist()))
                    
                except Exception as e:
                    # Fallback: create zero embedding
                    print(f"Error processing input {i}: {e}")
                    zero_embedding = torch.zeros(self.hidden_size, dtype=torch.float32)
                    embeddings.append(Embedding(values=zero_embedding.tolist()))
        
        return embeddings
    
    def _process_image(self, image_data: Union[str, Image.Image]) -> Optional[Image.Image]:
        """Process image data into PIL Image format"""
        try:
            if isinstance(image_data, str):
                # Base64 encoded image
                return self._decode_base64_image(image_data)
            elif isinstance(image_data, Image.Image):
                return image_data
            else:
                return None
        except Exception as e:
            print(f"Error processing image: {e}")
            return None
    
    def _decode_base64_image(self, base64_string: str) -> Image.Image:
        """Decode base64 string to PIL Image"""
        try:
            # Remove data URL prefix if present
            if base64_string.startswith('data:image'):
                base64_string = base64_string.split(',')[1]
            
            # Decode base64
            image_data = base64.b64decode(base64_string)
            image = Image.open(io.BytesIO(image_data))
            
            # Convert to RGB if necessary
            if image.mode != 'RGB':
                image = image.convert('RGB')
            
            return image
            
        except Exception as e:
            raise ValueError(f"Failed to decode base64 image: {str(e)}")
    
    def predict(self, batch: Batch) -> torch.Tensor:
        """
        JinaVL is an embedding model, not a classification model.
        This method is not supported.
        """
        raise NotImplementedError("JinaVL model does not support classification/prediction tasks")
    
    def encode_text(self, text: str) -> str:
        """
        Encode text with proper formatting for embedding
        """
        # Add query/passage prefix if not already present
        if not (text.startswith("Query:") or text.startswith("Passage:")):
            # Default to passage for general text
            text = f"Passage: {text}"
        
        return text
    
    def encode_image_text(self, text: str, has_image: bool = True) -> str:
        """
        Encode text with image placeholders for multimodal input
        """
        if has_image:
            # Format for multimodal input with image
            return f"<|im_start|>user\n<|vision_start|><|image_pad|><|vision_end|>{text}<|im_end|>\n"
        else:
            return self.encode_text(text)


def get_model(
    model_path: str,
    device: torch.device,
    dtype: torch.dtype,
    **kwargs
) -> JinaVLModel:
    """Factory function to create JinaVL model instance"""
    return JinaVLModel(
        model_path=model_path,
        device=device,
        dtype=dtype,
        **kwargs
    )