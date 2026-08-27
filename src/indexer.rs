//! Incremental indexing: walk a directory (or take explicit files), hash
//! contents, skip unchanged docs, embed + store changed ones.

use crate::chunker;
use crate::embedder::Embedder;
use crate::store::{ChunkKey, Store};
use anyhow::{Context, Result};
use std::collections::HashMap;
use std::path::{Path, PathBuf};
use walkdir::WalkDir;

pub const DEFAULT_EXTENSIONS: &[&str] = &[
    "md",
    "markdown",
    "txt",
    "rst",
    "json",
    "yaml",
    "yml",
    "toml",
    "csv",
    "tsv",
    "html",
    "htm",
    "xml",
    "log",
    "rs",
    "py",
    "js",
    "jsx",
    "ts",
    "tsx",
    "go",
    "c",
    "h",
    "cpp",
    "hpp",
    "java",
    "rb",
    "sh",
    "bash",
    "zsh",
    "sql",
    "proto",
    "graphql",
    "dockerfile",
    "makefile",
    "ini",
    "cfg",
    "conf",
    "env",
];

/// Skip dirs that are never useful in a knowledge corpus.
fn is_ignored_dir(name: &str) -> bool {
    // All dot-directories are config/metadata by convention (.git, .obsidian,
    // .vscode, ...) — skip them wholesale rather than enumerating known names.
    if name.starts_with('.') {
        return true;
    }
    matches!(
        name,
        "node_modules" | "target" | "dist" | "build" | "venv" | "__pycache__" | "vendor"
    )
}

fn ext_of(path: &Path) -> Option<String> {
    // Support extension-less well-known names too.
    let name = path.file_name()?.to_str()?.to_lowercase();
    if matches!(name.as_str(), "dockerfile" | "makefile") {
        return Some(name);
    }
    path.extension()
        .and_then(|e| e.to_str())
        .map(|e| e.to_lowercase())
}

/// Discover candidate files under `root`.
pub fn discover_files(root: &Path, extra_exts: &[String]) -> Result<Vec<PathBuf>> {
    let mut allowed: std::collections::HashSet<String> =
        DEFAULT_EXTENSIONS.iter().map(|s| s.to_string()).collect();
    for e in extra_exts {
        allowed.insert(e.trim_start_matches('.').to_lowercase());
    }

    let mut out = Vec::new();
    for entry in WalkDir::new(root)
        .follow_links(false)
        .into_iter()
        .filter_entry(|e| {
            e.file_type().is_file()
                || e.file_name()
                    .to_str()
                    .map(|n| !is_ignored_dir(n))
                    .unwrap_or(true)
        })
    {
        let entry = entry.with_context(|| format!("walking {}", root.display()))?;
        if !entry.file_type().is_file() {
            continue;
        }
        let path = entry.path();
        // Size guard: skip anything absurd (> 8 MB) — likely data dumps.
        let meta = entry.metadata()?;
        if meta.len() > 8 * 1024 * 1024 {
            tracing::warn!(path = %path.display(), size = meta.len(), "skipping large file");
            continue;
        }
        if let Some(ext) = ext_of(path) {
            if allowed.contains(&ext) {
                out.push(path.to_path_buf());
            }
        }
    }
    out.sort();
    Ok(out)
}

pub struct IndexReport {
    pub indexed: Vec<String>,
    pub skipped_unchanged: usize,
    pub removed_missing: usize,
    pub failed: Vec<String>,
}

impl IndexReport {
    pub fn summary(&self) -> String {
        format!(
            "indexed {}, unchanged {}, pruned {}{}",
            self.indexed.len(),
            self.skipped_unchanged,
            self.removed_missing,
            if self.failed.is_empty() {
                String::new()
            } else {
                format!(", FAILED {}", self.failed.join(", "))
            }
        )
    }
}

fn hash_bytes(bytes: &[u8]) -> u64 {
    // FNV-1a 64-bit: fast, stable across runs/platforms, plenty for
    // change detection (not a security hash).
    let mut hash: u64 = 0xcbf29ce484222325;
    for &b in bytes {
        hash ^= b as u64;
        hash = hash.wrapping_mul(0x100000001b3);
    }
    hash
}

struct FileFacts {
    hash: u64,
    mtime_secs: i64,
    size: u64,
}

fn facts(path: &Path) -> Result<FileFacts> {
    let bytes = std::fs::read(path).with_context(|| format!("reading {}", path.display()))?;
    let meta = std::fs::metadata(path)?;
    Ok(FileFacts {
        hash: hash_bytes(&bytes),
        mtime_secs: meta
            .modified()?
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_secs() as i64)
            .unwrap_or(0),
        size: meta.len(),
    })
}

/// Index one file into the store + tantivy sidecar.
pub fn index_one(
    path: &Path,
    store: &Store,
    tantivy_idx: &crate::search::TantivyIndex,
    embedder: &mut Embedder,
) -> Result<usize> {
    let text = read_text(path)?;
    let chunks = chunker::chunk_text(&text);
    if chunks.is_empty() {
        anyhow::bail!("no indexable content");
    }
    let f = facts(path)?;
    let key = path.to_string_lossy().to_string();
    let n = store.upsert_document(&key, f.hash, f.mtime_secs, f.size, &chunks, embedder)?;
    tantivy_idx.rebuild_from(store)?;
    Ok(n)
}

/// Full incremental pass over a directory.
pub fn index_directory(
    dir: &Path,
    extra_exts: &[String],
    store: &Store,
    tantivy_idx: &crate::search::TantivyIndex,
    embedder: &mut Embedder,
) -> Result<IndexReport> {
    if !store.schema_matches()? {
        store.clear()?;
    }

    let files = discover_files(dir, extra_exts)?;
    let known: HashMap<String, crate::store::DocumentMeta> = store.list_documents()?;
    let mut report = IndexReport {
        indexed: Vec::new(),
        skipped_unchanged: 0,
        removed_missing: 0,
        failed: Vec::new(),
    };

    // Batch documents into groups of ~50 so redb does one fsync per batch
    // instead of one per document. Each doc is embedded up-front (CPU-bound,
    // no txn held); the write transaction only does the fast I/O.
    const BATCH: usize = 50;
    let mut docs_meta: HashMap<String, (u64, i64, u64, Vec<ChunkKey>)> = HashMap::new();
    let mut all_chunks: Vec<String> = Vec::new();
    let mut all_vectors: Vec<Vec<f32>> = Vec::new();
    let mut cursor: u64 = store.next_chunk_key()?;

    // Helper: flush current batch with one fsync, clearing accumulators.
    macro_rules! flush_batch {
        ($store:expr, $meta:expr, $chunks:expr, $vecs:expr) => {{
            if !$meta.is_empty() {
                let c = std::mem::take($chunks);
                let v = std::mem::take($vecs);
                $store.upsert_batch($meta, &c, &v)?;
                $meta.clear();
            }
        }};
    }

    for path in &files {
        let key = path.to_string_lossy().to_string();
        let f = match facts(path) {
            Ok(f) => f,
            Err(e) => {
                flush_batch!(store, &mut docs_meta, &mut all_chunks, &mut all_vectors);
                report.failed.push(format!("{} ({e})", path.display()));
                continue;
            }
        };
        if let Some(meta) = known.get(&key) {
            if meta.hash == f.hash && meta.mtime_secs == f.mtime_secs {
                report.skipped_unchanged += 1;
                continue;
            }
        }
        let text = match read_text(path) {
            Ok(t) => t,
            Err(e) => {
                flush_batch!(store, &mut docs_meta, &mut all_chunks, &mut all_vectors);
                report.failed.push(format!("{} ({e})", path.display()));
                continue;
            }
        };
        let chunks = chunker::chunk_text(&text);
        if chunks.is_empty() {
            flush_batch!(store, &mut docs_meta, &mut all_chunks, &mut all_vectors);
            report
                .failed
                .push(format!("{} (no content)", path.display()));
            continue;
        }
        // Embed outside the txn.
        let vectors = embedder.embed_batch(&chunks)?;
        let chunk_keys: Vec<ChunkKey> = (0..chunks.len()).map(|i| cursor + i as u64).collect();
        cursor += chunks.len() as u64;
        docs_meta.insert(key, (f.hash, f.mtime_secs, f.size, chunk_keys));
        all_chunks.extend(chunks);
        all_vectors.extend(vectors);
        report.indexed.push(path.to_string_lossy().to_string());

        // Flush when we've accumulated ~BATCH docs.
        if docs_meta.len() >= BATCH {
            flush_batch!(store, &mut docs_meta, &mut all_chunks, &mut all_vectors);
        }
    }
    // Flush remainder.
    flush_batch!(store, &mut docs_meta, &mut all_chunks, &mut all_vectors);

    // Prune docs that no longer exist on disk.
    let known_paths: std::collections::HashSet<String> = files
        .iter()
        .map(|p| p.to_string_lossy().to_string())
        .collect();
    for k in known.keys() {
        if !known_paths.contains(k) {
            if store.delete_document(k)? {
                report.removed_missing += 1;
            }
        }
    }

    // Rebuild the BM25 sidecar once for the full batch.
    tantivy_idx.rebuild_from(store)?;
    Ok(report)
}

/// Read a file as UTF-8 text, tolerating a BOM and replacing invalid bytes.
pub fn read_text(path: &Path) -> Result<String> {
    let bytes = std::fs::read(path).with_context(|| format!("reading {}", path.display()))?;
    let bytes = if bytes.starts_with(&[0xEF, 0xBB, 0xBF]) {
        &bytes[3..]
    } else {
        &bytes[..]
    };
    Ok(String::from_utf8_lossy(bytes).to_string())
}
