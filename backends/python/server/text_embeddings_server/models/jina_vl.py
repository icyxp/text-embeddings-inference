import torch
import torch.nn.functional as F
from typing import List, Optional, Union, Dict, Any
from PIL import Image
import base64
import io
from transformers import AutoTokenizer, AutoConfig
from text_embeddings_server.models.model import Model
from text_embeddings_server.models.types import Batch, Embedding


class JinaVLModel(Model):
    """
    Jina Vision-Language Embedding Model
    
    This model supports multimodal embeddings for both text and images
    using the Qwen2_5_VLForConditionalGeneration architecture.
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
        
        # Load tokenizer and config
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        self.config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
        
        # Vision token IDs
        self.vision_start_token_id = getattr(self.config, 'vision_start_token_id', 151652)
        self.vision_end_token_id = getattr(self.config, 'vision_end_token_id', 151653)
        self.image_token_id = getattr(self.config, 'image_token_id', 151655)
        
        # Initialize the model using vLLM
        self._init_vllm_model()
        
        # Set model properties
        self.max_input_length = getattr(self.config, 'max_position_embeddings', 131072)
        self.hidden_size = getattr(self.config, 'hidden_size', 2048)
        
    def _init_vllm_model(self):
        """Initialize the vLLM model for embedding inference"""
        try:
            from vllm import LLM
            from vllm.config import PoolerConfig
            
            # Initialize vLLM model with embedding configuration
            self.model = LLM(
                model=self.model_path,
                task="embed",
                override_pooler_config=PoolerConfig(
                    pooling_type="VISION",  # Use vision-aware pooling
                    normalize=True,
                    softmax=False
                ),
                dtype=str(self.dtype).split('.')[-1],  # Convert torch.float16 to "float16"
                trust_remote_code=True,
                enforce_eager=True,  # Disable CUDA graphs for embedding tasks
                max_model_len=self.max_input_length,
            )
            
        except ImportError:
            raise ImportError(
                "vLLM is required for JinaVL model. Please install it with: "
                "pip install vllm"
            )
    
    @property
    def batch_type(self) -> type:
        return Batch
    
    @property
    def max_batch_size(self) -> Optional[int]:
        return None  # Let vLLM handle batching
    
    def embed(self, batch: Batch) -> List[Embedding]:
        """
        Generate embeddings for a batch of inputs (text and/or images)
        """
        from vllm.inputs.data import TextPrompt
        
        # Prepare prompts for vLLM
        prompts = []
        
        for i, input_ids in enumerate(batch.input_ids):
            # Decode input_ids back to text for vLLM processing
            text = self.tokenizer.decode(input_ids, skip_special_tokens=False)
            
            # Check if this input contains images
            multimodal_data = {}
            if hasattr(batch, 'images') and batch.images and i < len(batch.images):
                image_data = batch.images[i]
                if image_data:
                    # Handle different image input formats
                    if isinstance(image_data, str):
                        # Base64 encoded image
                        image = self._decode_base64_image(image_data)
                    elif isinstance(image_data, Image.Image):
                        image = image_data
                    else:
                        # Assume it's already in the correct format
                        image = image_data
                    
                    multimodal_data["image"] = image
            
            # Create TextPrompt with multimodal data
            if multimodal_data:
                prompt = TextPrompt(prompt=text, multi_modal_data=multimodal_data)
            else:
                prompt = TextPrompt(prompt=text)
            
            prompts.append(prompt)
        
        # Generate embeddings using vLLM
        try:
            outputs = self.model.encode(prompts)
            embeddings = self._extract_embeddings(outputs)
            
            return [
                Embedding(values=embedding.tolist())
                for embedding in embeddings
            ]
            
        except Exception as e:
            raise RuntimeError(f"Error generating embeddings: {str(e)}")
    
    def _extract_embeddings(self, outputs) -> List[torch.Tensor]:
        """
        Extract and process embeddings from vLLM outputs
        """
        embeddings = []
        
        for output in outputs:
            # Check if this is a vision-text multimodal input
            if (hasattr(output, 'prompt_token_ids') and 
                self.vision_start_token_id in output.prompt_token_ids):
                
                # Extract vision token embeddings
                token_ids = torch.tensor(output.prompt_token_ids)
                vision_start_positions = (token_ids == self.vision_start_token_id).nonzero(as_tuple=True)[0]
                vision_end_positions = (token_ids == self.vision_end_token_id).nonzero(as_tuple=True)[0]
                
                if len(vision_start_positions) > 0 and len(vision_end_positions) > 0:
                    # Use the last vision token range
                    start_pos = vision_start_positions[-1].item()
                    end_pos = vision_end_positions[-1].item()
                    
                    # Extract vision embeddings
                    vision_embeddings = output.outputs.data[start_pos:end_pos + 1]
                    
                    # Pool vision embeddings (mean pooling)
                    pooled_embedding = vision_embeddings.mean(dim=0, dtype=torch.float32)
                else:
                    # Fallback to all tokens if vision tokens not found
                    pooled_embedding = output.outputs.data.mean(dim=0, dtype=torch.float32)
            else:
                # Text-only input - use all tokens
                pooled_embedding = output.outputs.data.mean(dim=0, dtype=torch.float32)
            
            # Normalize embedding
            normalized_embedding = F.normalize(pooled_embedding, p=2, dim=-1)
            embeddings.append(normalized_embedding)
        
        return embeddings
    
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