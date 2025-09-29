use candle::{DType, Device, IndexOp, Result, Tensor, D};
use candle_nn::VarBuilder;
use serde::Deserialize;
use text_embeddings_backend_core::{Batch, ModelType};

use crate::layers::HiddenAct;
use crate::models::{Model, Qwen2Config};

#[cfg(feature = "cuda")]
use crate::models::FlashQwen2Model;

trait JinaBackbone: Model {
    fn encode_hidden(&self, batch: &Batch) -> Result<(Tensor, Tensor)>;
}

#[cfg(feature = "cuda")]
impl JinaBackbone for FlashQwen2Model {
    fn encode_hidden(&self, batch: &Batch) -> Result<(Tensor, Tensor)> {
        self.encode_hidden(batch)
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
    qwen2_model: Box<dyn JinaBackbone + Send>,
    config: JinaVLConfig,
    device: Device,
    #[allow(dead_code)]
    dtype: DType,
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
        };

        // Load the underlying Qwen2 model
        #[cfg(feature = "cuda")]
        {
            let device = _vb.device().clone();
            let dtype = _vb.dtype();
            let qwen2_model: Box<dyn JinaBackbone + Send> =
                Box::new(FlashQwen2Model::load(_vb, &_qwen2_config, _model_type)?);

            Ok(Self {
                qwen2_model,
                config: config.clone(),
                device,
                dtype,
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

    fn pool_sequence(&self, embeddings: &Tensor, token_ids: &[u32]) -> Result<Tensor> {
        let vision_start_token_id = self.config.vision_start_token_id.unwrap_or(151652);
        let vision_end_token_id = self.config.vision_end_token_id.unwrap_or(151653);

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

        let pooled = if let (Some(start), Some(end)) = (vision_start_pos, vision_end_pos) {
            if start < end && end < token_ids.len() {
                let len = end - start + 1;
                embeddings.narrow(0, start, len)?.mean(0)?
            } else {
                let seq_len = embeddings.dim(0)?;
                embeddings.i(seq_len - 1)?
            }
        } else {
            embeddings.mean(0)?
        };

        let pooled = pooled.to_dtype(DType::F32)?;

        let norm = pooled.sqr()?.sum_keepdim(D::Minus1)?.sqrt()?;
        let denom = (&norm + 1e-12f64)?;
        pooled.broadcast_div(&denom)
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

        let (outputs, _) = self.qwen2_model.encode_hidden(&batch)?;

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
