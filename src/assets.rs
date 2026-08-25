//! Embedded model assets: MiniLM-L6-v2 weights, tokenizer, and config are
//! compiled into the binary so a release build needs zero network access.

use anyhow::{Context, Result};
use candle_core::DType;
use std::path::Path;

/// Raw safetensors bytes (~90 MB) — mmap'd from memory by the embedder.
pub static MODEL_SAFETENSORS: &[u8] = include_bytes!("../assets/model.safetensors");
/// HuggingFace tokenizer.json (~710 KB).
pub static TOKENIZER_JSON: &str = include_str!("../assets/tokenizer.json");
/// BERT config.json (hidden sizes, activation, etc.).
pub static CONFIG_JSON: &str = include_str!("../assets/config.json");

/// Dimension of all-MiniLM-L6-v2 embeddings.
pub const EMBED_DIM: usize = 384;

/// Parse the embedded config.json into a candle `Config`.
///
/// We avoid depending on `candle_transformers::models::bert::Config` directly
/// here so callers get one deserialization path; the struct is re-exported for
/// the embedder.
pub fn bert_config() -> Result<candle_transformers::models::bert::Config> {
    let cfg: candle_transformers::models::bert::Config =
        serde_json::from_str(CONFIG_JSON).context("parsing embedded model config")?;
    Ok(cfg)
}

/// Write the embedded assets to disk once (first run) so later runs can mmap
/// them instead of holding 90 MB of anonymous memory. Returns (weights_path,
/// tokenizer_path).
pub fn materialize(cache_dir: &Path) -> Result<(std::path::PathBuf, std::path::PathBuf)> {
    std::fs::create_dir_all(cache_dir)
        .with_context(|| format!("creating cache dir {}", cache_dir.display()))?;
    let weights = cache_dir.join("model.safetensors");
    let tok = cache_dir.join("tokenizer.json");

    if !weights.exists() {
        // Write to a temp file then rename so a crash mid-write never leaves a
        // truncated model that "exists".
        let tmp = weights.with_extension("tmp");
        std::fs::write(&tmp, MODEL_SAFETENSORS)
            .with_context(|| format!("writing {}", tmp.display()))?;
        std::fs::rename(&tmp, &weights).context("renaming model into place")?;
    }
    if !tok.exists() {
        std::fs::write(&tok, TOKENIZER_JSON)
            .with_context(|| format!("writing {}", tok.display()))?;
    }
    Ok((weights, tok))
}

/// Dtype used for inference. f32 keeps numerics identical to
/// sentence-transformers on CPU; f16 would halve RAM but lose precision.
pub const DTYPE: DType = DType::F32;
