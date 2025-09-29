use candle::{DType, Device, Result, Tensor};
use candle_nn::VarBuilder;
use crate::layers::{LayerNorm, Linear};
use serde::Deserialize;
use candle::D;

#[derive(Debug, Clone, PartialEq, Deserialize, Default)]
pub struct VisionConfig {
    pub hidden_size: Option<usize>,
    pub out_hidden_size: Option<usize>,
    pub patch_size: Option<usize>,
    pub spatial_merge_size: Option<usize>,
    pub window_size: Option<usize>,
    pub depth: Option<usize>,
    pub num_heads: Option<usize>,
}

pub struct Qwen2VlVisionEncoder {
    pub config: VisionConfig,
    pub device: Device,
    pub dtype: DType,
    pos_table: Option<Tensor>,
    out_proj: Option<Linear>,
    blocks: Vec<VisionBlock>,
    vision_hidden: usize,
}

struct VisionBlock {
    norm1: LayerNorm,
    qkv: Linear,
    proj: Linear,
    norm2: LayerNorm,
    fc1: Linear,
    fc2: Linear,
    num_heads: usize,
}

impl VisionBlock {
    fn load(vb: &VarBuilder, hidden: usize, num_heads: usize) -> Result<Self> {
        // LayerNorms
        let norm1 = LayerNorm::load(vb.pp("norm1"), hidden, 1e-5)?;
        let norm2 = LayerNorm::load(vb.pp("norm2"), hidden, 1e-5)?;
        // Attention qkv and proj
        let qkv = Linear::new(
            vb.pp("attn.qkv").get((3 * hidden, hidden), "weight")?,
            vb.pp("attn.qkv").get(3 * hidden, "bias").ok(),
            None,
        );
        let proj = Linear::new(
            vb.pp("attn.proj").get((hidden, hidden), "weight")?,
            vb.pp("attn.proj").get(hidden, "bias").ok(),
            None,
        );
        // MLP
        let mlp_hidden = (hidden * 4).max(1);
        let fc1 = Linear::new(
            vb.pp("mlp.fc1").get((mlp_hidden, hidden), "weight")?,
            vb.pp("mlp.fc1").get(mlp_hidden, "bias").ok(),
            None,
        );
        let fc2 = Linear::new(
            vb.pp("mlp.fc2").get((hidden, mlp_hidden), "weight")?,
            vb.pp("mlp.fc2").get(hidden, "bias").ok(),
            None,
        );

        Ok(Self { norm1, qkv, proj, norm2, fc1, fc2, num_heads })
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        // x: [n, hidden]
        let n = x.dim(0)?;
        let hidden = x.dim(1)?;
        let head_dim = hidden / self.num_heads;
        // Attn
        let x1 = self.norm1.forward(x, None)?;
        let qkv = self.qkv.forward(&x1)?; // [n, 3*hidden]
        let q = qkv.narrow(1, 0, hidden)?;
        let k = qkv.narrow(1, hidden, hidden)?;
        let v = qkv.narrow(1, 2 * hidden, hidden)?;
        let q = q.reshape((n, self.num_heads, head_dim))?;
        let k = k.reshape((n, self.num_heads, head_dim))?;
        let v = v.reshape((n, self.num_heads, head_dim))?;
        // Compute attention per head: attn = softmax(q k^T / sqrt(d)) v
        let scale = (1.0 / (head_dim as f64).sqrt()) as f32;
        let mut heads_out: Vec<Tensor> = Vec::with_capacity(self.num_heads);
        for h in 0..self.num_heads {
            let qh = q.narrow(1, h, 1)?.squeeze(1)?; // [n, head_dim]
            let kh = k.narrow(1, h, 1)?.squeeze(1)?; // [n, head_dim]
            let vh = v.narrow(1, h, 1)?.squeeze(1)?; // [n, head_dim]
            // scores: [n, n]
            let mut scores = qh.matmul(&kh.t()?)?.to_dtype(DType::F32)?;
            let scale_t = Tensor::new(scale, qh.device())?;
            scores = (scores * scale_t)?;
            // softmax over last dim
            let max = scores.max(D::Minus1)?;
            scores = (scores.broadcast_sub(&max)?).exp()?;
            let sums = scores.sum_keepdim(D::Minus1)?;
            let probs = scores.broadcast_div(&sums)?;
            let out = probs.matmul(&vh)?; // [n, head_dim]
            heads_out.push(out);
        }
        let attn_out = Tensor::cat(&heads_out.iter().collect::<Vec<_>>(), 1)?; // [n, hidden]
        let attn_out = self.proj.forward(&attn_out)?;
        let x2 = x.add(&attn_out)?;

        // MLP
        let x3 = self.norm2.forward(&x2, None)?;
        let x3 = self.fc1.forward(&x3)?;
        let x3 = x3.gelu()?;
        let x3 = self.fc2.forward(&x3)?;
        x2.add(&x3)
    }
}

impl Qwen2VlVisionEncoder {
    pub fn load(vb: &VarBuilder, config: VisionConfig, device: &Device, dtype: DType) -> Result<Self> {
        // Attempt to load a simple output projection if present.
        let out_hidden = config.out_hidden_size.unwrap_or(config.hidden_size.unwrap_or(2048));
        let hidden = config.hidden_size.unwrap_or(out_hidden);
        let try_linear = |key: &str| -> Result<Option<Linear>> {
            if vb.contains_tensor(&format!("{key}.weight")) {
                let w = vb.pp(key).get((out_hidden, out_hidden), "weight")?;
                let b = vb.pp(key).get(out_hidden, "bias").ok();
                return Ok(Some(Linear::new(w, b, None)));
            }
            Ok(None)
        };
        let mut out_proj = None;
        for key in [
            "vision.proj",
            "vision.projector",
            "visual.projector",
            "projector",
        ] {
            if let Some(lin) = try_linear(key)? { out_proj = Some(lin); break; }
        }
        // Load a minimal set of blocks if available
        let depth = config.depth.unwrap_or(0);
        let mut blocks = Vec::new();
        for i in 0..depth {
            let vb_blk = vb.pp(format!("vision.blocks.{i}"));
            match VisionBlock::load(&vb_blk, hidden, config.num_heads.unwrap_or(8)) {
                Ok(b) => blocks.push(b),
                Err(_) => break,
            }
        }
        Ok(Self { config, device: device.clone(), dtype, pos_table: None, out_proj, blocks, vision_hidden: hidden })
    }

    /// Produce deterministic positional features for a T x H x W grid.
    /// This creates a sin/cos positional encoding over the flattened index
    /// and expands/interleaves to `out_hidden_size`.
    pub fn features_for_grid(&self, t: usize, h: usize, w: usize, out_hidden_size: usize) -> Result<Tensor> {
        let n = t * h * w;
        // If we have a learned table, slice the first n rows.
        if let Some(ref table) = self.pos_table {
            let total = table.dim(0)?;
            let take = n.min(total);
            let head = table.narrow(D::Minus2, 0, take)?;
            if take < n {
                // pad with zeros to reach n
                let pad = Tensor::zeros((n - take, table.dim(1)?), self.dtype, &self.device)?;
                return Tensor::cat(&[&head, &pad], D::Minus2);
            }
            return Ok(head.clone());
        }

        // Axis-aware sinusoidal features for T/H/W segments
        let idx: Vec<usize> = (0..n).collect();
        let h_u = h as usize;
        let w_u = w as usize;
        let mut tpos = Vec::with_capacity(n);
        let mut hpos = Vec::with_capacity(n);
        let mut wpos = Vec::with_capacity(n);
        for k in idx.into_iter() {
            let ti = k / (h_u * w_u);
            let hi = (k / w_u) % h_u;
            let wi = k % w_u;
            tpos.push(ti as f32);
            hpos.push(hi as f32);
            wpos.push(wi as f32);
        }
        let tpos = Tensor::from_vec(tpos, (n, 1), &self.device)?;
        let hpos = Tensor::from_vec(hpos, (n, 1), &self.device)?;
        let wpos = Tensor::from_vec(wpos, (n, 1), &self.device)?;

        // Split feature dims equally across T/H/W with cos+sin per axis
        let axis_half = (out_hidden_size / 6).max(1); // per-axis cos half
        let base: f32 = 10_000.0;
        let inv_build = |dim: usize| -> Result<Tensor> {
            let inv: Vec<_> = (0..dim)
                .map(|i| 1f32 / base.powf(i as f32 / dim as f32))
                .collect();
            Tensor::from_vec(inv, (1, dim), &self.device)
        };
        let inv_t = inv_build(axis_half)?;
        let inv_h = inv_build(axis_half)?;
        let inv_w = inv_build(axis_half)?;

        let fe_t = tpos.matmul(&inv_t)?; // [n, axis_half]
        let fe_h = hpos.matmul(&inv_h)?;
        let fe_w = wpos.matmul(&inv_w)?;
        let cos_t = fe_t.cos()?.to_dtype(self.dtype)?;
        let sin_t = fe_t.sin()?.to_dtype(self.dtype)?;
        let cos_h = fe_h.cos()?.to_dtype(self.dtype)?;
        let sin_h = fe_h.sin()?.to_dtype(self.dtype)?;
        let cos_w = fe_w.cos()?.to_dtype(self.dtype)?;
        let sin_w = fe_w.sin()?.to_dtype(self.dtype)?;

        let feat_t = Tensor::cat(&[&cos_t, &sin_t], D::Minus1)?; // [n, 2*axis_half]
        let feat_h = Tensor::cat(&[&cos_h, &sin_h], D::Minus1)?;
        let feat_w = Tensor::cat(&[&cos_w, &sin_w], D::Minus1)?;
        let mut feat = Tensor::cat(&[&feat_t, &feat_h, &feat_w], D::Minus1)?; // [n, 6*axis_half]
        let current = 6 * axis_half;
        if current > out_hidden_size {
            feat = feat.narrow(D::Minus1, 0, out_hidden_size)?;
        } else if current < out_hidden_size {
            let pad = Tensor::zeros((n, out_hidden_size - current), self.dtype, &self.device)?;
            feat = Tensor::cat(&[&feat, &pad], D::Minus1)?;
        }
        // If vision blocks exist, map to hidden dim, run blocks, then map to out_hidden
        let mut feat = feat;
        if !self.blocks.is_empty() {
            // adjust to vision hidden size
            let d = feat.dim(1)?;
            let hidden = self.vision_hidden;
            if d != hidden {
                if d > hidden { feat = feat.narrow(D::Minus1, 0, hidden)?; }
                else {
                    let pad = Tensor::zeros((n, hidden - d), self.dtype, &self.device)?;
                    feat = Tensor::cat(&[&feat, &pad], D::Minus1)?;
                }
            }
            for b in &self.blocks {
                feat = b.forward(&feat)?;
            }
            // After blocks, map to out_hidden via out_proj or pad/trunc
            if let Some(ref proj) = self.out_proj {
                feat = proj.forward(&feat)?;
            } else {
                let d2 = feat.dim(1)?;
                if d2 != out_hidden_size {
                    if d2 > out_hidden_size { feat = feat.narrow(D::Minus1, 0, out_hidden_size)?; }
                    else {
                        let pad = Tensor::zeros((n, out_hidden_size - d2), self.dtype, &self.device)?;
                        feat = Tensor::cat(&[&feat, &pad], D::Minus1)?;
                    }
                }
            }
        } else {
            // Optional learned projection
            if let Some(ref proj) = self.out_proj { feat = proj.forward(&feat)?; }
        }
        Ok(feat)
    }
}
