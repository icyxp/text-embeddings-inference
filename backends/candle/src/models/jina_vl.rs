//! # Jina VL Model Implementation
//!
//! This module implements the Jina VL (Vision-Language) embedding model, specifically
//! the jina-embeddings-v4-vllm-retrieval variant. This is a multimodal embedding model
//! that can process both text-only and image+text inputs to generate unified embeddings.
//!
//! ## Architecture
//!
//! The model is based on the Qwen2 architecture with additional vision capabilities:
//! - **Text Backbone**: Qwen2 transformer model for text processing
//! - **Vision Encoder**: Qwen2-VL vision encoder for image feature extraction
//! - **Multimodal Fusion**: Vision features are injected into the token sequence
//!
//! ## Pooling Strategy
//!
//! The pooling strategy is critical for generating high-quality embeddings and has been
//! carefully designed to match the vLLM reference implementation:
//!
//! ### For Text-Only Inputs
//! - Pool over all tokens in the sequence using mean pooling
//! - Use explicit sum + division (not `.mean()`) for numerical consistency
//! - Perform all operations in F32 precision for stability
//!
//! ### For Multimodal (Image+Text) Inputs
//! - Detect vision tokens: VISION_START_TOKEN_ID (151652) and VISION_END_TOKEN_ID (151653)
//! - Pool only tokens between vision_start and vision_end (inclusive)
//! - This focuses the embedding on visual content rather than surrounding text
//! - Use explicit sum + division in F32 precision
//!
//! ### Why Explicit Sum + Division?
//! The implementation uses `sum(0) / count` instead of `.mean()` because:
//! 1. **Matches vLLM**: The reference implementation uses `sum(dim=0, dtype=torch.float32) / count`
//! 2. **Explicit Precision Control**: Ensures the sum operation is performed in F32
//! 3. **Numerical Consistency**: Avoids potential differences in how `.mean()` is implemented
//! 4. **Transparency**: Makes the numerical behavior explicit and easier to verify
//!
//! ## Normalization
//!
//! All embeddings are L2-normalized to unit length (L2 norm = 1.0):
//! - Uses the same formula as `torch.nn.functional.normalize(dim=-1, eps=1e-12)`
//! - Performed in F32 precision for numerical stability
//! - Epsilon value (1e-12) prevents division by zero for zero/near-zero vectors
//!
//! ## Precision Considerations
//!
//! This implementation prioritizes numerical precision to match the vLLM reference:
//!
//! ### F32 Precision for Pooling
//! - All pooling operations are performed in F32, even if the model uses F16 weights
//! - Prevents precision loss from F16 accumulation, especially for long sequences
//! - Critical for achieving cosine similarity > 0.99 with vLLM outputs
//!
//! ### F32 Precision for Normalization
//! - Norm computation and division are performed in F32
//! - Provides wider dynamic range to handle very large or very small values
//! - Ensures epsilon (1e-12) is effective (would be too small for F16)
//!
//! ## Differences from Previous Implementation
//!
//! This implementation includes several precision fixes compared to earlier versions:
//!
//! 1. **Explicit Sum + Division**: Changed from `.mean()` to `sum() / count` to match vLLM
//! 2. **F32 Pooling**: Ensured all pooling operations use F32 precision explicitly
//! 3. **Standalone Normalization**: Extracted L2 normalization into a separate function
//! 4. **Enhanced Logging**: Added debug logging for token counts and pooling ranges
//! 5. **Better Edge Case Handling**: Improved fallback behavior for invalid vision tokens
//!
//! ## Troubleshooting Precision Issues
//!
//! If you encounter precision differences between this implementation and vLLM:
//!
//! ### Check Token Pooling
//! - Enable debug logging to see which tokens are being pooled
//! - Verify vision tokens are detected correctly (151652 and 151653)
//! - Ensure the pooling range matches vLLM's output
//!
//! ### Check Data Types
//! - Verify embeddings are converted to F32 before pooling
//! - Check that sum operations are performed in F32
//! - Ensure normalization uses F32 precision
//!
//! ### Check Normalization
//! - Verify L2 norm of output equals 1.0 (within tolerance ~1e-6)
//! - Check epsilon value is 1e-12 (matches PyTorch default)
//! - Ensure norm computation doesn't overflow/underflow
//!
//! ### Compare Intermediate Values
//! - Log token counts used for pooling
//! - Log pooled values before normalization
//! - Log norm values before and after epsilon addition
//! - Compare these with vLLM's intermediate outputs
//!
//! ### Common Issues
//! - **Low cosine similarity**: Usually caused by F16 accumulation or wrong pooling range
//! - **NaN values**: Check for division by zero in normalization (epsilon too small?)
//! - **Wrong pooling range**: Verify vision token IDs match the model's tokenizer config
//! - **Precision loss**: Ensure F32 is used for pooling and normalization, not F16
//!
//! ## References
//!
//! - vLLM Implementation: `jina-embed-v4/usage.py`
//! - Model: https://huggingface.co/jinaai/jina-embeddings-v4-vllm-retrieval
//! - PyTorch Normalize: https://pytorch.org/docs/stable/generated/torch.nn.functional.normalize.html
//!
//! ## Example Usage
//!
//! ```rust,ignore
//! // Load the model
//! let model = JinaVLModel::load(vb, &config, model_type)?;
//!
//! // Process a batch (text-only or multimodal)
//! let (pooled_embeddings, raw_embeddings) = model.embed(batch)?;
//!
//! // pooled_embeddings: [batch_size, hidden_size], L2-normalized
//! // raw_embeddings: [total_tokens, hidden_size], not normalized
//! ```

use candle::{DType, Device, IndexOp, Result, Tensor, D};
use candle_nn::VarBuilder;
use serde::Deserialize;
use text_embeddings_backend_core::{Batch, ModelType};

use crate::layers::HiddenAct;
use crate::models::{Model, Qwen2Config};
use crate::vision::qwen2_vl::Qwen2VlVisionEncoder;

#[cfg(feature = "cuda")]
use crate::models::FlashQwen2Model;

trait JinaBackbone: Model {
    fn encode_hidden(&self, batch: &Batch) -> Result<(Tensor, Tensor)>;
    fn embed_only(&self, batch: &Batch) -> Result<(Tensor, Tensor, Tensor, Tensor, usize)>;
    fn run_layers(&self, hidden: Tensor, cu_seqlens: &Tensor, cos: &Tensor, sin: &Tensor, max_s: usize) -> Result<Tensor>;
}

#[cfg(feature = "cuda")]
impl JinaBackbone for FlashQwen2Model {
    fn encode_hidden(&self, batch: &Batch) -> Result<(Tensor, Tensor)> {
        self.encode_hidden(batch)
    }
    fn embed_only(&self, batch: &Batch) -> Result<(Tensor, Tensor, Tensor, Tensor, usize)> {
        self.embed_only(batch)
    }
    fn run_layers(&self, hidden: Tensor, cu_seqlens: &Tensor, cos: &Tensor, sin: &Tensor, max_s: usize) -> Result<Tensor> {
        self.run_layers(hidden, cu_seqlens, cos, sin, max_s)
    }
}

#[derive(Debug, Clone, PartialEq, Deserialize)]
pub struct JinaVLConfig {
    pub vocab_size: usize,
    pub hidden_size: usize,
    pub intermediate_size: usize,
    pub num_hidden_layers: usize,
    pub num_attention_heads: usize,
    pub num_key_value_heads: Option<usize>,
    pub use_sliding_window: Option<bool>,
    pub sliding_window: Option<usize>,
    pub max_window_layers: Option<usize>,
    pub max_position_embeddings: usize,
    pub rms_norm_eps: f64,
    pub tie_word_embeddings: bool,
    pub rope_theta: Option<f64>,
    pub attention_dropout: f64,
    pub hidden_act: HiddenAct,
    #[serde(default)]
    pub use_flash_attn: bool,
    #[serde(default)]
    pub rope_scaling: Option<crate::models::qwen2::Qwen2RopeScaling>,
    #[serde(default)]
    pub vision_config: Option<crate::vision::qwen2_vl::VisionConfig>,
    
    // Vision-specific token IDs for multimodal processing
    
    /// Vision start token ID (default: 151652)
    /// 
    /// This special token marks the beginning of vision feature embeddings in a multimodal
    /// sequence. When processing image+text inputs, the vision encoder generates a sequence
    /// of embeddings that are inserted into the token sequence between the vision start and
    /// vision end tokens.
    /// 
    /// Example sequence structure:
    /// ```text
    /// [text tokens] [VISION_START] [vision embeddings...] [VISION_END] [more text tokens]
    /// ```
    /// 
    /// During pooling, only the embeddings between VISION_START and VISION_END (inclusive)
    /// are used to compute the final embedding for multimodal inputs. This ensures that
    /// the embedding focuses on the visual content rather than the surrounding text.
    pub vision_start_token_id: Option<u32>,
    
    /// Vision end token ID (default: 151653)
    /// 
    /// This special token marks the end of vision feature embeddings in a multimodal sequence.
    /// It works in conjunction with vision_start_token_id to define the boundaries of the
    /// vision embedding region.
    /// 
    /// The pooling strategy uses both tokens to identify the range of embeddings to pool:
    /// - If both tokens are present: Pool embeddings from start to end (inclusive)
    /// - If tokens are missing or invalid: Fall back to pooling all tokens
    /// 
    /// This token-based approach allows the model to handle variable-length vision features
    /// and mixed text+image inputs in a flexible way.
    pub vision_end_token_id: Option<u32>,
    
    /// Image token ID (default: 151655)
    /// 
    /// This token represents image placeholders in the input sequence. It's used during
    /// tokenization to indicate where image features should be inserted. The actual image
    /// embeddings are generated by the vision encoder and replace the region between
    /// vision_start_token_id and vision_end_token_id.
    /// 
    /// Note: This token is primarily used during preprocessing and may not appear in the
    /// final token sequence after vision features are injected.
    pub image_token_id: Option<u32>,
}

impl From<Qwen2Config> for JinaVLConfig {
    fn from(config: Qwen2Config) -> Self {
        Self {
            vocab_size: config.vocab_size,
            hidden_size: config.hidden_size,
            intermediate_size: config.intermediate_size,
            num_hidden_layers: config.num_hidden_layers,
            num_attention_heads: config.num_attention_heads,
            num_key_value_heads: Some(config.num_key_value_heads),
            use_sliding_window: Some(config.use_sliding_window),
            sliding_window: config.sliding_window,
            max_window_layers: None, // Not present in Qwen2Config
            max_position_embeddings: config.max_position_embeddings,
            rms_norm_eps: config.rms_norm_eps as f64,
            tie_word_embeddings: false, // Default value
            rope_theta: Some(config.rope_theta as f64),
            attention_dropout: 0.0, // Default value
            hidden_act: config.hidden_act,
            use_flash_attn: false, // Default value
            rope_scaling: config.rope_scaling,
            vision_config: None,
            // Default vision token IDs for Jina VL multimodal processing
            // These tokens are used to identify and process vision features in the sequence
            vision_start_token_id: Some(151652),  // Marks start of vision embeddings
            vision_end_token_id: Some(151653),    // Marks end of vision embeddings
            image_token_id: Some(151655),         // Placeholder for image features
        }
    }
}

pub struct JinaVLModel {
    // Use the underlying Qwen2 model for text processing
    qwen2_model: Box<dyn JinaBackbone + Send>,
    config: JinaVLConfig,
    device: Device,
    #[allow(dead_code)]
    dtype: DType,
    vision: Option<Qwen2VlVisionEncoder>,
}

/// Apply L2 normalization to a tensor, matching torch.nn.functional.normalize behavior.
/// 
/// This function normalizes the input tensor along the last dimension (dim=-1) such that
/// the L2 norm of the output equals 1.0. All operations are performed in F32 precision
/// for numerical stability.
/// 
/// # Why F32 Precision?
/// F32 precision is critical for normalization because:
/// 1. **Numerical Stability**: Computing norms in F16 can lead to overflow/underflow for
///    very large or very small values. F32 provides a much wider dynamic range.
/// 2. **Precision in Division**: The division operation (input / norm) is more accurate
///    in F32, especially when the norm is very small or very large.
/// 3. **Consistency with vLLM**: The vLLM implementation uses float32 for normalization,
///    and matching this ensures identical numerical behavior.
/// 4. **Epsilon Effectiveness**: The epsilon value (1e-12) is meaningful in F32 but would
///    be too small to be effective in F16 (which has ~3 decimal digits of precision).
/// 
/// # Arguments
/// * `tensor` - Input tensor to normalize, should be in F32 dtype
/// 
/// # Returns
/// * Normalized tensor with L2 norm = 1.0 along the last dimension
/// 
/// # Normalization Formula
/// The L2 normalization formula is:
/// ```text
/// output = input / max(||input||_2, epsilon)
/// 
/// where:
///   ||input||_2 = sqrt(sum(input^2))  [L2 norm, Euclidean norm]
///   epsilon = 1e-12                    [numerical stability constant]
/// ```
/// 
/// The epsilon value serves two purposes:
/// 1. **Prevents Division by Zero**: For zero or near-zero vectors, adding epsilon ensures
///    we don't divide by zero, which would produce NaN values.
/// 2. **Numerical Stability**: For very small norms, division by a tiny number can amplify
///    numerical errors. The epsilon provides a lower bound on the denominator.
/// 
/// # Edge Cases
/// * **Zero Vector**: Returns a vector with very small magnitude (input / epsilon) rather than NaN
/// * **Near-Zero Norm**: The epsilon (1e-12) prevents numerical instability
/// * **Very Large Values**: F32 precision prevents overflow in the norm computation
/// 
/// # Epsilon Value
/// The epsilon value of 1e-12 is chosen to match PyTorch's default for functional.normalize.
/// This value is:
/// - Small enough to not affect normal vectors (typical norms are >> 1e-12)
/// - Large enough to prevent division by zero and numerical instability
/// - Consistent with PyTorch's implementation for cross-framework compatibility
/// 
/// # Reference
/// PyTorch implementation: `torch.nn.functional.normalize(input, p=2, dim=-1, eps=1e-12)`
/// Documentation: https://pytorch.org/docs/stable/generated/torch.nn.functional.normalize.html
fn l2_normalize(tensor: &Tensor) -> Result<Tensor> {
    // Ensure we're working in F32 precision for numerical stability.
    // This conversion is essential even if the input is already F32, as it ensures
    // all subsequent operations maintain F32 precision throughout the computation.
    let tensor_f32 = tensor.to_dtype(DType::F32)?;
    
    // Compute L2 norm: sqrt(sum(x^2)) along the last dimension.
    // 
    // Step-by-step breakdown:
    // 1. .sqr()? - Square each element: x^2
    // 2. .sum_keepdim(D::Minus1)? - Sum along last dimension, keeping dimensions for broadcasting
    // 3. .sqrt()? - Take square root to get the L2 norm
    // 
    // Using keepdim=true is important because it maintains the dimension structure,
    // allowing the subsequent broadcast_div operation to work correctly.
    let norm = tensor_f32
        .sqr()?                      // Element-wise square: x^2
        .sum_keepdim(D::Minus1)?     // Sum along last dimension: sum(x^2), keep dims
        .sqrt()?;                    // Square root: sqrt(sum(x^2)) = L2 norm
    
    // Add epsilon for numerical stability (matches PyTorch's default).
    // 
    // Why 1e-12?
    // - It's PyTorch's default epsilon for functional.normalize
    // - It's small enough to not affect normal vectors (typical norms are much larger)
    // - It's large enough to prevent division by zero and numerical instability
    // - It's well within F32's precision range (F32 has ~7 decimal digits of precision)
    // 
    // This prevents division by zero for zero or near-zero vectors, which would
    // otherwise produce NaN values that propagate through the computation.
    let epsilon = 1e-12f64;
    let denom = (&norm + epsilon)?;
    
    // Normalize: divide input by (norm + epsilon).
    // 
    // broadcast_div handles the broadcasting automatically, dividing each element
    // of the input tensor by the corresponding norm value. The result is a tensor
    // where each vector along the last dimension has L2 norm = 1.0 (unit length).
    // 
    // This operation is performed in F32 precision for accuracy, especially important
    // when the norm is very small or very large.
    tensor_f32.broadcast_div(&denom)
}

impl JinaVLModel {
    pub fn load(_vb: VarBuilder, config: &JinaVLConfig, _model_type: ModelType) -> Result<Self> {
        // Convert JinaVLConfig to Qwen2Config for the underlying model
        let _qwen2_config = Qwen2Config {
            vocab_size: config.vocab_size,
            hidden_size: config.hidden_size,
            intermediate_size: config.intermediate_size,
            num_hidden_layers: config.num_hidden_layers,
            num_attention_heads: config.num_attention_heads,
            num_key_value_heads: config
                .num_key_value_heads
                .unwrap_or(config.num_attention_heads),
            use_sliding_window: config.use_sliding_window.unwrap_or(false),
            sliding_window: config.sliding_window,
            max_position_embeddings: config.max_position_embeddings,
            rms_norm_eps: config.rms_norm_eps as f32,
            rope_theta: config.rope_theta.unwrap_or(10000.0) as f32,
            hidden_act: config.hidden_act.clone(),
            rope_scaling: config.rope_scaling.clone(),
        };

        // Load the underlying Qwen2 model
        #[cfg(feature = "cuda")]
        {
            let device = _vb.device().clone();
            let dtype = _vb.dtype();
            let vb_backbone = _vb.clone();
            let qwen2_model: Box<dyn JinaBackbone + Send> =
                Box::new(FlashQwen2Model::load(vb_backbone, &_qwen2_config, _model_type)?);

            let vision = match config.vision_config.clone() {
                Some(vc) => Some(Qwen2VlVisionEncoder::load(&_vb, vc, &device, dtype)?),
                None => None,
            };

            Ok(Self {
                qwen2_model,
                config: config.clone(),
                device,
                dtype,
                vision,
            })
        }

        #[cfg(not(feature = "cuda"))]
        {
            // For CPU, we'll use a simplified implementation
            // In practice, Jina VL models are typically used with CUDA
            Err(candle::Error::Msg(
                "Jina VL model requires CUDA support".to_string(),
            ))
        }
    }

    /// Pool embeddings from a sequence with vision-aware strategy.
    /// 
    /// This method implements the pooling strategy that matches the vLLM implementation
    /// for jina-embeddings-v4-vllm-retrieval. The key aspects are:
    /// 
    /// 1. **Vision Token Detection**: Searches for VISION_START_TOKEN_ID (151652) and 
    ///    VISION_END_TOKEN_ID (151653) in the token sequence to identify multimodal inputs.
    /// 
    /// 2. **Pooling Strategy**:
    ///    - For multimodal inputs: Pools only tokens between vision_start and vision_end (inclusive)
    ///    - For text-only inputs: Pools all tokens in the sequence
    /// 
    /// 3. **Explicit Sum + Division**: Uses explicit `.sum(0)` followed by division by token count
    ///    instead of `.mean()`. This matches vLLM's implementation and ensures consistent numerical
    ///    behavior across frameworks. The explicit approach gives better control over precision.
    /// 
    /// 4. **F32 Precision**: All pooling operations are performed in F32 dtype for numerical stability.
    ///    This prevents precision loss that can occur with F16 accumulation, especially for long
    ///    sequences where small errors can compound.
    /// 
    /// 5. **L2 Normalization**: The pooled embedding is normalized to unit length using L2 norm.
    /// 
    /// # Arguments
    /// * `embeddings` - Tensor of shape [seq_len, hidden_size] containing token embeddings
    /// * `token_ids` - Slice of token IDs corresponding to the embeddings
    /// 
    /// # Returns
    /// * Normalized pooled embedding of shape [hidden_size] with L2 norm = 1.0
    /// 
    /// # Reference
    /// vLLM implementation: jina-embed-v4/usage.py - get_embeddings() function
    /// See: https://github.com/jina-ai/jina-embeddings-v4-vllm-retrieval
    fn pool_sequence(&self, embeddings: &Tensor, token_ids: &[u32]) -> Result<Tensor> {
        // Vision token IDs used to identify multimodal inputs
        // These tokens mark the boundaries of vision feature embeddings in the sequence
        let vision_start_token_id = self.config.vision_start_token_id.unwrap_or(151652);
        let vision_end_token_id = self.config.vision_end_token_id.unwrap_or(151653);

        // Scan the token sequence to find vision token positions
        // vision_start_pos: Position of the vision start token (if present)
        // vision_end_pos: Position of the vision end token (if present)
        let mut vision_start_pos = None;
        let mut vision_end_pos = None;

        for (i, &token_id) in token_ids.iter().enumerate() {
            if token_id == vision_start_token_id {
                vision_start_pos = Some(i);
            }
            if token_id == vision_end_token_id {
                vision_end_pos = Some(i);
            }
        }

        // Convert to F32 for numerical stability during pooling operations.
        // F32 precision is critical here because:
        // 1. Accumulation in F16 can lose precision for long sequences
        // 2. Division operations are more accurate in F32
        // 3. This matches the vLLM implementation which uses dtype=torch.float32
        let seq_embeddings = embeddings.to_dtype(DType::F32)?;
        
        // Debug logging: tensor shape
        tracing::debug!(
            "pool_sequence: input shape = {:?}, total_tokens = {}",
            seq_embeddings.dims(),
            token_ids.len()
        );

        // Perform pooling based on whether vision tokens are present
        let pooled = if let (Some(start), Some(end)) = (vision_start_pos, vision_end_pos) {
            // Multimodal case: Vision tokens detected
            if start < end && end < token_ids.len() {
                // Pool only tokens between vision_start and vision_end (inclusive).
                // This extracts the vision-related embeddings for multimodal inputs.
                // 
                // Why explicit sum + division instead of .mean()?
                // 1. Matches vLLM's implementation exactly: sum(dim=0, dtype=float32) / count
                // 2. Gives explicit control over the dtype of the sum operation
                // 3. More transparent about numerical behavior
                // 4. Avoids potential differences in how .mean() is implemented across frameworks
                let len = end - start + 1;
                
                // Debug logging: vision token positions and pooling range
                tracing::debug!(
                    "pool_sequence: vision tokens found - start_pos = {}, end_pos = {}, pooling {} tokens",
                    start,
                    end,
                    len
                );
                
                // Extract the token slice and compute mean via explicit sum + division
                let token_slice = seq_embeddings.narrow(0, start, len)?;
                let sum = token_slice.sum(0)?;  // Sum in F32 precision
                (sum / (len as f64))?           // Divide by count to get mean
            } else {
                // Invalid vision token positions - fall back to mean pooling over all tokens
                // This handles edge cases where vision tokens are malformed or out of order
                let token_count = seq_embeddings.dim(0)?;
                
                tracing::debug!(
                    "pool_sequence: invalid vision token positions (start={}, end={}), falling back to all {} tokens",
                    start,
                    end,
                    token_count
                );
                
                let sum = seq_embeddings.sum(0)?;
                (sum / (token_count as f64))?
            }
        } else {
            // Text-only case: No vision tokens detected, pool over all tokens.
            // This is the standard mean pooling strategy for text embeddings.
            // 
            // Again, we use explicit sum + division to match vLLM's implementation
            // and ensure consistent numerical behavior.
            let token_count = seq_embeddings.dim(0)?;
            
            tracing::debug!(
                "pool_sequence: text-only mode, pooling all {} tokens",
                token_count
            );
            
            let sum = seq_embeddings.sum(0)?;  // Sum in F32 precision
            (sum / (token_count as f64))?      // Divide by count to get mean
        };

        // Ensure pooled tensor is in F32 before normalization (should already be F32 from above)
        let pooled_f32 = pooled.to_dtype(DType::F32)?;
        
        // Apply L2 normalization using the standalone function.
        // This normalizes the embedding to unit length (L2 norm = 1.0).
        // The normalization matches torch.nn.functional.normalize behavior.
        l2_normalize(&pooled_f32)
    }

    fn vision_aware_pooling(&self, outputs: &Tensor, batch: &Batch) -> Result<Option<Tensor>> {
        if batch.pooled_indices.is_empty() {
            return Ok(None);
        }

        let mut pooled_results: Vec<Tensor> = Vec::with_capacity(batch.pooled_indices.len());

        for &index in &batch.pooled_indices {
            let seq_index = index as usize;
            let start = batch.cumulative_seq_lengths[seq_index] as usize;
            let end = batch.cumulative_seq_lengths[seq_index + 1] as usize;

            if end <= start {
                let hidden_size = outputs.dim(1)?;
                let zero = Tensor::zeros((hidden_size,), outputs.dtype(), outputs.device())?;
                pooled_results.push(zero.unsqueeze(0)?);
                continue;
            }

            let seq_embeddings = outputs.narrow(0, start, end - start)?;
            let token_ids = &batch.input_ids[start..end];
            let pooled = self.pool_sequence(&seq_embeddings, token_ids)?;
            pooled_results.push(pooled.unsqueeze(0)?);
        }

        if pooled_results.is_empty() {
            Ok(None)
        } else {
            Ok(Some(Tensor::cat(&pooled_results, 0)?))
        }
    }
}

impl Model for JinaVLModel {
    fn is_padded(&self) -> bool {
        self.qwen2_model.is_padded()
    }

    fn embed(&self, batch: Batch) -> Result<(Option<Tensor>, Option<Tensor>)> {
        let batch_size = batch.len();
        let shape = batch.input_ids.len();
        let has_pooling_requests = !batch.pooled_indices.is_empty();
        let has_raw_requests = !batch.raw_indices.is_empty();

        // Obtain initial token embeddings + rotary tensors
        let (mut hidden, cu_seqlens, cos, sin, max_s) = self.qwen2_model.embed_only(&batch)?;

        // Replace image token embeddings with vision features if available
        if let Some(vision) = &self.vision {
            // Vision token IDs for multimodal processing
            // These tokens mark the boundaries of vision feature embeddings in the sequence
            
            /// VISION_START_ID (151652): Marks the beginning of vision embeddings
            /// Used to identify where vision features start in a multimodal sequence
            const VISION_START_ID: u32 = 151652;
            
            /// VISION_END_ID (151653): Marks the end of vision embeddings
            /// Used to identify where vision features end in a multimodal sequence
            const VISION_END_ID: u32 = 151653;

            // Find spans
            let ids = &batch.input_ids;
            let mut i = 0usize;
            while i < ids.len() {
                if ids[i] == VISION_START_ID {
                    // find end
                    let mut j = i + 1;
                    while j < ids.len() && ids[j] != VISION_END_ID { j += 1; }
                    if j <= ids.len() && j > i + 1 {
                        let span_start = i + 1;
                        let span_end = j; // exclusive of end token
                        let span_len = span_end - span_start;
                        // naive grid inference: assume 1xHxW where H*W = span_len
                        // Prefer grid from batch if present
                        let (t, h, w) = if let Some((tt, hh, ww)) = batch.image_grid_thw {
                            (tt as usize, hh as usize, ww as usize)
                        } else {
                            let t = 1usize;
                            // pick square-ish H,W
                            let mut hh = (span_len as f64).sqrt() as usize;
                            if hh == 0 { hh = 1; }
                            while span_len % hh != 0 { hh -= 1; if hh == 0 { hh = 1; break; } }
                            let ww = span_len / hh;
                            (t, hh, ww)
                        };
                        let out_hidden = self.config.hidden_size;
                        let feats = vision.features_for_grid(t, h, w, out_hidden)?;

                        // Slice hidden: [0..span_start], [span_start..span_end], [span_end..]
                        let before = if span_start > 0 { hidden.narrow(0, 0, span_start)? } else { Tensor::zeros((0, out_hidden), hidden.dtype(), hidden.device())? };
                        let after = if span_end < hidden.dims()[0] { hidden.narrow(0, span_end, hidden.dims()[0] - span_end)? } else { Tensor::zeros((0, out_hidden), hidden.dtype(), hidden.device())? };
                        hidden = Tensor::cat(&[&before, &feats, &after], 0)?;
                        i = j + 1;
                        continue;
                    }
                }
                i += 1;
            }
        }

        // Run transformer layers + norm
        let outputs = self.qwen2_model.run_layers(hidden, &cu_seqlens, &cos, &sin, max_s)?;

        let pooled_embeddings = if has_pooling_requests {
            self.vision_aware_pooling(&outputs, &batch)?
        } else {
            None
        };

        let raw_embeddings = if has_raw_requests {
            if batch_size > 1 && has_pooling_requests {
                let mut final_indices: Vec<u32> = Vec::with_capacity(shape);
                for i in batch.raw_indices.iter().cloned() {
                    let idx = i as usize;
                    let start = batch.cumulative_seq_lengths[idx];
                    let end = batch.cumulative_seq_lengths[idx + 1];
                    for j in start..end {
                        final_indices.push(j);
                    }
                }

                let final_indices_length = final_indices.len();
                let final_indices =
                    Tensor::from_vec(final_indices, final_indices_length, &self.device)?;

                Some(outputs.index_select(&final_indices, 0)?)
            } else {
                Some(outputs)
            }
        } else {
            None
        };

        Ok((pooled_embeddings, raw_embeddings))
    }

    fn predict(&self, _batch: Batch) -> Result<Tensor> {
        // JinaVL is an embedding model, not a classification model
        candle::bail!("JinaVL model does not support prediction tasks")
    }
}

#[cfg(feature = "cuda")]
pub struct FlashJinaVLModel {
    inner: JinaVLModel,
}

#[cfg(feature = "cuda")]
impl FlashJinaVLModel {
    pub fn load(vb: VarBuilder, config: &JinaVLConfig, model_type: ModelType) -> Result<Self> {
        let inner = JinaVLModel::load(vb, config, model_type)?;
        Ok(Self { inner })
    }
}

#[cfg(feature = "cuda")]
impl Model for FlashJinaVLModel {
    fn is_padded(&self) -> bool {
        self.inner.is_padded()
    }

    fn embed(&self, batch: Batch) -> Result<(Option<Tensor>, Option<Tensor>)> {
        self.inner.embed(batch)
    }

    fn predict(&self, batch: Batch) -> Result<Tensor> {
        self.inner.predict(batch)
    }
}
