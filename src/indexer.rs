//! Incremental indexing: walk a directory (or take explicit files), hash
//! contents, skip unchanged docs, embed + store changed ones.
//!
//! Pruning is scoped to the directory root passed to `index_directory`:
//! documents outside the current walk (e.g. indexed earlier from a different
//! root) are never removed. Pass `--force` (CLI) or re-index the union tree
//! when you intentionally want a full replacement.

use crate::chunker;
use crate::embedder::Embedder;
use crate::store::{ChunkKey, Store};
use anyhow::{Context, Result};
use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::time::Instant;
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
    if name.starts_with('.') {
        return true;
    }
    matches!(
        name,
        "node_modules" | "target" | "dist" | "build" | "venv" | "__pycache__" | "vendor"
    )
}

fn ext_of(path: &Path) -> Option<String> {
    let name = path.file_name()?.to_str()?.to_lowercase();
    if matches!(name.as_str(), "dockerfile" | "makefile") {
        return Some(name);
    }
    path.extension()
        .and_then(|e| e.to_str())
        .map(|e| e.to_lowercase())
}

/// Discover candidate files under `root`, following symlinks.
///
/// Symlinked files and directories are included (walkdir reports link cycles
/// as errors, which we log and skip rather than abort on). Per-entry errors —
/// broken symlinks, permission failures — are logged and skipped.
pub fn discover_files(root: &Path, extra_exts: &[String]) -> Result<Vec<PathBuf>> {
    let mut allowed: std::collections::HashSet<String> =
        DEFAULT_EXTENSIONS.iter().map(|s| s.to_string()).collect();
    for e in extra_exts {
        allowed.insert(e.trim_start_matches('.').to_lowercase());
    }

    let mut out = Vec::new();
    let mut walk_errors = 0usize;
    for entry in WalkDir::new(root)
        .follow_links(true)
        .into_iter()
        .filter_entry(|e| {
            e.file_type().is_file()
                || e.file_name()
                    .to_str()
                    .map(|n| !is_ignored_dir(n))
                    .unwrap_or(true)
        })
    {
        let entry = match entry {
            Ok(e) => e,
            Err(err) => {
                walk_errors += 1;
                if let Some(path) = err.path() {
                    tracing::warn!("skipping unreadable path {}: {err}", path.display());
                } else {
                    tracing::warn!("walk error: {err}");
                }
                continue;
            }
        };
        if !entry.file_type().is_file() {
            continue;
        }
        let path = entry.path();
        let meta = match entry.metadata() {
            // With follow_links(true) this is the link target's metadata: a
            // broken symlink surfaces here as an error.
            Ok(m) => m,
            Err(err) => {
                walk_errors += 1;
                tracing::warn!("skipping {}: {err}", path.display());
                continue;
            }
        };
        if meta.is_symlink() {
            tracing::warn!(
                path = %path.display(),
                "skipping symlink pointing at a symlink (possible cycle)"
            );
            continue;
        }
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
    if walk_errors > 0 {
        tracing::warn!("{walk_errors} entries were skipped due to walk errors");
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
                format!(", failed {}", self.failed.len())
            }
        )
    }
}

fn hash_bytes(bytes: &[u8]) -> u64 {
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

/// Read a file as text, auto-detecting encoding via chardet.
fn read_text(path: &Path) -> Result<String> {
    let bytes = std::fs::read(path).with_context(|| format!("reading {}", path.display()))?;
    if bytes.starts_with(b"\xef\xbb\xbf") {
        // Strip BOM and try UTF-8.
        let sans_bom = &bytes[3..];
        return String::from_utf8(sans_bom.to_vec())
            .with_context(|| format!("decoding {} as utf-8", path.display()));
    }
    if let Ok(s) = String::from_utf8(bytes.clone()) {
        return Ok(s);
    }
    // Fall back to lossy: replace invalid UTF-8 with U+FFFD.
    Ok(String::from_utf8_lossy(&bytes).to_string())
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
///
/// Embeds all changed documents in batches so candle's rayon thread pool can
/// parallelize across the corpus rather than being invoked once per document
/// with tiny batches. Progress is logged per sub-batch so long CPU runs are
/// visibly alive.
///
/// Only documents under `dir` are eligible for pruning: a doc in the store
/// survives this call unless its path sits inside `dir` and is no longer
/// discovered. Indexing a different root therefore never deletes another
/// root's corpus.
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

    let t_walk = Instant::now();
    let files = discover_files(dir, extra_exts)?;
    let walk_s = t_walk.elapsed().as_secs_f64();
    if files.is_empty() {
        tracing::warn!(
            "no indexable files found under {} — nothing was indexed; \
             check the path, file extensions (-e), and that sources are \
             real files rather than dangling links",
            dir.display()
        );
    }
    let known: HashMap<String, crate::store::DocumentMeta> = store.list_documents()?;
    let mut report = IndexReport {
        indexed: Vec::new(),
        skipped_unchanged: 0,
        removed_missing: 0,
        failed: Vec::new(),
    };

    // Collect all chunks from changed docs, then embed everything in one
    // giant batch for better CPU utilization.
    let mut docs_meta: HashMap<String, (u64, i64, u64, Vec<ChunkKey>)> = HashMap::new();
    let mut all_chunks: Vec<String> = Vec::new();
    let mut cursor: u64 = store.next_chunk_key()?;

    for path in &files {
        let key = path.to_string_lossy().to_string();
        let f = match facts(path) {
            Ok(f) => f,
            Err(e) => {
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
                report.failed.push(format!("{} ({e})", path.display()));
                continue;
            }
        };
        let chunks = chunker::chunk_text(&text);
        if chunks.is_empty() {
            report
                .failed
                .push(format!("{} (no content)", path.display()));
            continue;
        }
        let chunk_keys: Vec<ChunkKey> = (0..chunks.len()).map(|i| cursor + i as u64).collect();
        cursor += chunks.len() as u64;
        docs_meta.insert(key.clone(), (f.hash, f.mtime_secs, f.size, chunk_keys));
        all_chunks.extend(chunks);
        report.indexed.push(path.to_string_lossy().to_string());
    }

    // Single large embed batch for all chunks across all changed documents.
    // Embed in sub-batches of 128 to bound memory while still letting
    // candle's rayon pool parallelize within each batch (much better than
    // per-document batches of ~16). Log progress per window: a silent
    // 7-minute stretch is indistinguishable from a hang.
    let t_embed = Instant::now();
    if !all_chunks.is_empty() {
        tracing::info!(
            "embedding {} chunks across {} documents",
            all_chunks.len(),
            docs_meta.len()
        );
        let mut all_vectors: Vec<Vec<f32>> = Vec::with_capacity(all_chunks.len());
        let total = all_chunks.len();
        for (i, chunk_window) in all_chunks.chunks(128).enumerate() {
            let batch_vecs = embedder.embed_batch(chunk_window)?;
            all_vectors.extend(batch_vecs);
            let done = (i + 1) * chunk_window.len();
            tracing::info!(
                "embedded {}/{} chunks ({}%)",
                done,
                total,
                done * 100 / total
            );
        }
        tracing::info!("embedding done in {:.1}s", t_embed.elapsed().as_secs_f64());
        store.upsert_batch(&docs_meta, &all_chunks, &all_vectors)?;
    }

    // Prune docs that no longer exist on disk — scoped to this index root.
    // Docs indexed from other roots (e.g. a previous `quillrag index
    // ~/other-notes`) must survive this call. A doc is only prunable when
    // its stored path sits inside the directory walked here.
    let known_paths: std::collections::HashSet<String> = files
        .iter()
        .map(|p| p.to_string_lossy().to_string())
        .collect();
    for k in known.keys() {
        if under_root(Path::new(k), dir) && !known_paths.contains(k) && store.delete_document(k)? {
            report.removed_missing += 1;
        }
    }

    // Rebuild the BM25 sidecar once for the full corpus.
    tantivy_idx.rebuild_from(store)?;

    tracing::debug!(
        "index pass over {} took walk {:.2}s, embed {:.2}s",
        dir.display(),
        walk_s,
        t_embed.elapsed().as_secs_f64()
    );

    Ok(report)
}

/// True when `path` equals `root` or lives underneath it (lexically, by
/// components — no filesystem access, works on already-normalized walk keys).
fn under_root(path: &Path, root: &Path) -> bool {
    path.starts_with(root)
}
