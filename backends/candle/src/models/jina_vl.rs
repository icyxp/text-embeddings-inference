use candle::{DType, Device, IndexOp, Result, Tensor, D};
use candle_nn::VarBuilder;
use serde::Deserialize;
use text_embeddings_backend_core::{Batch, ModelType};

use crate::models::{Model, Qwen2Config};
use crate::layers::HiddenAct;

#[cfg(feature = "cuda")]
use crate::models::FlashQwen2Model;

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
    // Vision-specific configurations
    pub vision_start_token_id: Option<u32>,
    pub vision_end_token_id: Option<u32>,
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
            // Default vision token IDs for Jina VL
            vision_start_token_id: Some(151652),
            vision_end_token_id: Some(151653),
            image_token_id: Some(151655),
        }
    }
}

pub struct JinaVLModel {
    // Use the underlying Qwen2 model for text processing
    qwen2_model: Box<dyn Model>,
    config: JinaVLConfig,
    device: Device,
    dtype: DType,
}

impl JinaVLModel {
    pub fn load(
        vb: VarBuilder,
        config: &JinaVLConfig,
        model_type: ModelType,
    ) -> Result<Self> {
        // Convert JinaVLConfig to Qwen2Config for the underlying model
        let qwen2_config = Qwen2Config {
            vocab_size: config.vocab_size,
            hidden_size: config.hidden_size,
            intermediate_size: config.intermediate_size,
            num_hidden_layers: config.num_hidden_layers,
            num_attention_heads: config.num_attention_heads,
            num_key_value_heads: config.num_key_value_heads.unwrap_or(config.num_attention_heads),
            use_sliding_window: config.use_sliding_window.unwrap_or(false),
            sliding_window: config.sliding_window,
            max_position_embeddings: config.max_position_embeddings,
            rms_norm_eps: config.rms_norm_eps as f32,
            rope_theta: config.rope_theta.unwrap_or(10000.0) as f32,
            hidden_act: config.hidden_act.clone(),
        };

        // Load the underlying Qwen2 model
        #[cfg(feature = "cuda")]
        let qwen2_model = {
            Box::new(FlashQwen2Model::load(vb, &qwen2_config, model_type)?) as Box<dyn Model>
        };

        #[cfg(not(feature = "cuda"))]
        let qwen2_model = {
            // For CPU, we'll use a simplified implementation
            // In practice, Jina VL models are typically used with CUDA
            return Err(candle::Error::Msg("Jina VL model requires CUDA support".to_string()));
        };

        Ok(Self {
            qwen2_model,
            config: config.clone(),
            device: vb.device().clone(),
            dtype: vb.dtype(),
        })
    }

    fn vision_aware_pooling(&self, embeddings: &Tensor, input_ids: &[u32]) -> Result<Tensor> {
        let vision_start_token_id = self.config.vision_start_token_id.unwrap_or(151652);
        let vision_end_token_id = self.config.vision_end_token_id.unwrap_or(151653);

        // Find vision token positions
        let mut vision_start_pos = None;
        let mut vision_end_pos = None;

        for (i, &token_id) in input_ids.iter().enumerate() {
            if token_id == vision_start_token_id {
                vision_start_pos = Some(i);
            }
            if token_id == vision_end_token_id {
                vision_end_pos = Some(i);
            }
        }

        let pooled = if let (Some(start), Some(end)) = (vision_start_pos, vision_end_pos) {
            if start < end && end < input_ids.len() {
                // Extract vision tokens and perform mean pooling
                let vision_embeddings = embeddings.i((start..=end))?;
                vision_embeddings.mean(0)?
            } else {
                // Fallback to last token if vision tokens are malformed
                let seq_len = embeddings.dim(0)?;
                embeddings.i(seq_len - 1)?
            }
        } else {
            // For text-only inputs, use last token pooling
            let seq_len = embeddings.dim(0)?;
            embeddings.i(seq_len - 1)?
        };

        // Normalize the embedding (important for consistency with vLLM)
        let norm = pooled.sqr()?.sum_keepdim(D::Minus1)?.sqrt()?;
        let normalized = pooled.broadcast_div(&norm)?;

        Ok(normalized)
    }
}

impl Model for JinaVLModel {
    fn is_padded(&self) -> bool {
        self.qwen2_model.is_padded()
    }

    fn embed(&self, batch: Batch) -> Result<(Option<Tensor>, Option<Tensor>)> {
        // Get embeddings from the underlying Qwen2 model
        let (embeddings, attention_bias) = self.qwen2_model.embed(batch)?;

        if let Some(embeddings) = embeddings {
            let mut pooled_embeddings = Vec::new();

            // Process each sequence in the batch
            let batch_size = embeddings.dim(0)?;
            for i in 0..batch_size {
                let sequence_embeddings = embeddings.i(i)?;
                
                // Get the input_ids for this sequence
                let input_ids = &batch.input_ids[i];
                
                // Apply vision-aware pooling
                let pooled = self.vision_aware_pooling(&sequence_embeddings, input_ids)?;
                pooled_embeddings.push(pooled);
            }

            // Stack the pooled embeddings
            let result = Tensor::stack(&pooled_embeddings, 0)?;
            Ok((Some(result), attention_bias))
        } else {
            Ok((None, attention_bias))
        }
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
    pub fn load(
        vb: VarBuilder,
        config: &JinaVLConfig,
        model_type: ModelType,
    ) -> Result<Self> {
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