//! Candle-based MiniLM embedder. Loads the embedded safetensors (mmap'd from
//! disk after first-run materialization), runs masked mean pooling + L2
//! normalization — numerically identical to sentence-transformers.

use crate::assets::{self, EMBED_DIM};
use anyhow::Result;
use candle_core::{Device, Tensor};
use candle_nn::VarBuilder;
use candle_transformers::models::bert::BertModel;
use std::path::Path;
use tokenizers::{PaddingParams, PaddingStrategy, Tokenizer};

pub struct Embedder {
    model: BertModel,
    tokenizer: Tokenizer,
    device: Device,
}

impl Embedder {
    /// Load from a materialized cache dir (weights + tokenizer on disk).
    pub fn load(cache_dir: &Path) -> Result<Self> {
        let (weights_path, tok_path) = assets::materialize(cache_dir)?;

        let tokenizer = Tokenizer::from_file(&tok_path)
            .map_err(|e| anyhow::anyhow!("loading tokenizer: {e}"))?;
        let mut config = assets::bert_config()?;
        config.hidden_act = candle_transformers::models::bert::HiddenAct::GeluApproximate;

        // mmap keeps RSS low; pages are shared and read-only.
        let vb = unsafe {
            VarBuilder::from_mmaped_safetensors(&[&weights_path], assets::DTYPE, &Device::Cpu)?
        };
        let start = std::time::Instant::now();
        let model = BertModel::load(vb, &config)?;
        tracing::debug!(elapsed = ?start.elapsed(), "BERT model loaded");

        Ok(Self {
            model,
            tokenizer,
            device: Device::Cpu,
        })
    }

    /// Configure batch padding so batches encode correctly.
    fn prepare_tokenizer(&mut self) -> Result<()> {
        if let Some(pp) = self.tokenizer.get_padding_mut() {
            pp.strategy = PaddingStrategy::BatchLongest;
        } else {
            self.tokenizer.with_padding(Some(PaddingParams {
                strategy: PaddingStrategy::BatchLongest,
                ..Default::default()
            }));
        }
        self.tokenizer
            .with_truncation(Some(tokenizers::TruncationParams {
                max_length: 256,
                ..Default::default()
            }))
            .map_err(|e| anyhow::anyhow!("setting truncation: {e}"))?;
        Ok(())
    }

    /// Embed a batch of texts -> matrix [n, 384], L2-normalized rows.
    pub fn embed_batch(&mut self, texts: &[String]) -> Result<Vec<Vec<f32>>> {
        if texts.is_empty() {
            return Ok(Vec::new());
        }
        self.prepare_tokenizer()?;

        let encodings = self
            .tokenizer
            .encode_batch(texts.to_vec(), true)
            .map_err(|e| anyhow::anyhow!("tokenization failed: {e}"))?;

        let token_ids: Vec<Tensor> = encodings
            .iter()
            .map(|enc| Tensor::new(enc.get_ids().to_vec(), &self.device))
            .collect::<candle_core::Result<_>>()?;
        let attention_mask: Vec<Tensor> = encodings
            .iter()
            .map(|enc| Tensor::new(enc.get_attention_mask().to_vec(), &self.device))
            .collect::<candle_core::Result<_>>()?;

        let token_ids = Tensor::stack(&token_ids, 0)?;
        let attention_mask = Tensor::stack(&attention_mask, 0)?;
        let token_type_ids = token_ids.zeros_like()?;

        // [batch, seq, hidden]
        let embeddings = self
            .model
            .forward(&token_ids, &token_type_ids, Some(&attention_mask))?;

        // Masked mean pool over sequence dim.
        let mask = attention_mask.to_dtype(assets::DTYPE)?.unsqueeze(2)?;
        let sum_mask = mask.sum(1)?;
        let pooled = embeddings.broadcast_mul(&mask)?.sum(1)?;
        let pooled = pooled.broadcast_div(&sum_mask)?;

        // L2 normalize each row -> cosine similarity == dot product.
        let norm = pooled.sqr()?.sum_keepdim(1)?.sqrt()?;
        let normalized = pooled.broadcast_div(&norm)?;

        let (n, dim) = normalized.dims2()?;
        debug_assert_eq!(dim, EMBED_DIM);
        let flat = normalized.flatten_all()?.to_vec1::<f32>()?;
        Ok(flat.chunks(dim).take(n).map(|c| c.to_vec()).collect())
    }

    /// Convenience single-text embedding (query path).
    pub fn embed_one(&mut self, text: &str) -> Result<Vec<f32>> {
        Ok(self.embed_batch(&[text.to_string()])?.remove(0))
    }
}
