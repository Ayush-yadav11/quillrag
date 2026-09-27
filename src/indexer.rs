//! Incremental indexing: walk a directory (or take explicit files), hash
//! contents, skip unchanged docs, embed + store changed ones.
//!
//! Pruning is scoped to the directory root passed to `index_directory`:
//! documents outside the current walk (e.g. indexed earlier from a different
//! root) are never removed. `--force` re-embeds only the selected sources.

use crate::chunker;
use crate::embedder::Embedder;
use crate::extract::{self, Extractor, Section, DEFAULT_MAX_FILE_BYTES};
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
/// Symlinked files and directories are included. Any walk error aborts
/// discovery before the index changes, including broken links and cycles.
pub fn discover_files(root: &Path, extra_exts: &[String]) -> Result<Vec<PathBuf>> {
    discover(root, extra_exts, &|_| DEFAULT_MAX_FILE_BYTES)
}

/// [`discover_files`] with a size limit per extension.
fn discover(
    root: &Path,
    extra_exts: &[String],
    max_bytes: &dyn Fn(&str) -> u64,
) -> Result<Vec<PathBuf>> {
    let mut allowed: std::collections::HashSet<String> =
        DEFAULT_EXTENSIONS.iter().map(|s| s.to_string()).collect();
    for e in extra_exts {
        allowed.insert(e.trim_start_matches('.').to_lowercase());
    }

    let mut out = Vec::new();
    for entry in WalkDir::new(root)
        .follow_links(true)
        .into_iter()
        .filter_entry(|e| {
            e.depth() == 0
                || e.file_type().is_file()
                || e.file_name()
                    .to_str()
                    .map(|n| !is_ignored_dir(n))
                    .unwrap_or(true)
        })
    {
        let entry =
            entry.with_context(|| format!("walking {}; index unchanged", root.display()))?;
        if !entry.file_type().is_file() {
            continue;
        }
        let path = entry.path();
        let Some(ext) = ext_of(path).filter(|e| allowed.contains(e)) else {
            continue;
        };
        let meta = entry
            .metadata()
            .with_context(|| format!("reading metadata {}", path.display()))?;
        if meta.len() > max_bytes(&ext) {
            tracing::warn!(path = %path.display(), size = meta.len(), "skipping large file");
            continue;
        }
        out.push(path.to_path_buf());
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

fn mtime_secs(meta: &std::fs::Metadata) -> Result<i64> {
    Ok(meta
        .modified()?
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_secs() as i64)
        .unwrap_or(0))
}

fn facts(path: &Path) -> Result<FileFacts> {
    let bytes = std::fs::read(path).with_context(|| format!("reading {}", path.display()))?;
    let meta = std::fs::metadata(path)?;
    Ok(FileFacts {
        hash: hash_bytes(&bytes),
        mtime_secs: mtime_secs(&meta)?,
        size: meta.len(),
    })
}

/// Same size and mtime as when indexed: skip without reading the file.
/// Content hashing only runs for files whose metadata changed, which keeps
/// re-scans of large PDF folders cheap.
fn unchanged_by_stat(path: &Path, known: &crate::store::DocumentMeta) -> bool {
    std::fs::metadata(path)
        .ok()
        .and_then(|m| Some(m.len() == known.size && mtime_secs(&m).ok()? == known.mtime_secs))
        .unwrap_or(false)
}

/// Index one file into the store + tantivy sidecar.
pub fn index_one(
    path: &Path,
    store: &Store,
    tantivy_idx: &crate::search::TantivyIndex,
    embedder: &mut Embedder,
) -> Result<usize> {
    index_one_with(path, store, tantivy_idx, embedder, &IndexHooks::default())
}

/// [`index_one`] with extractor hooks, so binary formats can be indexed.
pub fn index_one_with(
    path: &Path,
    store: &Store,
    tantivy_idx: &crate::search::TantivyIndex,
    embedder: &mut Embedder,
    hooks: &IndexHooks<'_>,
) -> Result<usize> {
    let absolute = absolute_source_path(path)?;
    let path = absolute.as_path();
    let sections = hooks.extract(path)?;
    let (chunks, pages) = chunker::chunk_sections(&sections);
    if chunks.is_empty() {
        anyhow::bail!("no indexable content");
    }
    let f = facts(path)?;
    let key = path.to_string_lossy().to_string();
    let vectors = embedder.embed_batch(&chunks)?;
    let first = store.next_chunk_key()?;
    let keys = (0..chunks.len()).map(|i| first + i as u64).collect();
    let docs_meta = HashMap::from([(key, (f.hash, f.mtime_secs, f.size, keys))]);
    let n = store.upsert_batch_with_pages(&docs_meta, &chunks, &pages, &vectors)?;
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
    index_directory_with_options(
        dir,
        extra_exts,
        store,
        tantivy_idx,
        embedder,
        IndexOptions::default(),
    )
}

#[derive(Default, Clone, Copy)]
pub struct IndexOptions {
    pub no_prune: bool,
    pub force: bool,
    /// Re-extract unchanged files that were recorded as having no text,
    /// e.g. after an extractor gained a capability such as OCR.
    pub retry_empty: bool,
}

/// Progress events emitted during an index pass, for UIs that want more than
/// the stderr log.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum IndexProgress {
    /// Discovery finished with `files` candidate files.
    Discovered { files: usize },
    /// Change scan: `done` of `total` files hashed and (if changed) extracted.
    Scanned { done: usize, total: usize },
    /// `done` of `total` changed chunks embedded.
    Embedding { done: usize, total: usize },
    /// The pass committed; the BM25 sidecar is rebuilt.
    Finished,
}

/// Optional extension points for an index pass. `IndexHooks::default()`
/// reproduces the plain-text CLI behavior.
#[derive(Default, Clone, Copy)]
pub struct IndexHooks<'a> {
    /// Extractors for non-text formats; their extensions are added to
    /// discovery automatically.
    pub extractors: &'a [&'a dyn Extractor],
    /// Called synchronously on the indexing thread; keep it cheap.
    pub progress: Option<&'a (dyn Fn(IndexProgress) + Sync)>,
}

impl IndexHooks<'_> {
    fn emit(&self, event: IndexProgress) {
        if let Some(f) = self.progress {
            f(event);
        }
    }

    fn extract(&self, path: &Path) -> Result<Vec<Section>> {
        match ext_of(path).and_then(|e| extract::find(self.extractors, &e)) {
            // The extractor's own message ("password-protected PDF") is what
            // the user needs; the path is added by the caller's report.
            Some(x) => x.extract(path),
            None => Ok(vec![Section::text(extract::read_text(path)?)]),
        }
    }

    fn max_file_bytes(&self, ext: &str) -> u64 {
        extract::find(self.extractors, ext)
            .map(|x| x.max_file_bytes())
            .unwrap_or(DEFAULT_MAX_FILE_BYTES)
    }

    fn extensions(&self) -> impl Iterator<Item = String> + '_ {
        self.extractors
            .iter()
            .flat_map(|x| x.extensions().iter().map(|e| e.to_lowercase()))
    }
}

pub fn index_directory_with_options(
    dir: &Path,
    extra_exts: &[String],
    store: &Store,
    tantivy_idx: &crate::search::TantivyIndex,
    embedder: &mut Embedder,
    options: IndexOptions,
) -> Result<IndexReport> {
    index_directory_with(
        dir,
        extra_exts,
        store,
        tantivy_idx,
        embedder,
        options,
        &IndexHooks::default(),
    )
}

/// Full incremental pass with extractor and progress hooks.
pub fn index_directory_with(
    dir: &Path,
    extra_exts: &[String],
    store: &Store,
    tantivy_idx: &crate::search::TantivyIndex,
    embedder: &mut Embedder,
    options: IndexOptions,
    hooks: &IndexHooks<'_>,
) -> Result<IndexReport> {
    let dir = absolute_source_path(dir)?;
    let dir = dir.as_path();
    anyhow::ensure!(dir.is_dir(), "not a directory: {}", dir.display());
    let t_walk = Instant::now();
    let exts: Vec<String> = extra_exts
        .iter()
        .cloned()
        .chain(hooks.extensions())
        .collect();
    let files = discover(dir, &exts, &|ext| hooks.max_file_bytes(ext))?;
    hooks.emit(IndexProgress::Discovered { files: files.len() });
    anyhow::ensure!(
        store.schema_matches()?,
        "incompatible store schema; rebuild into a new data directory"
    );
    let walk_s = t_walk.elapsed().as_secs_f64();
    if files.is_empty() {
        tracing::warn!(
            "no indexable files found under {}; nothing was indexed; \
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
    let mut all_pages: Vec<Option<u32>> = Vec::new();
    let mut cursor: u64 = store.next_chunk_key()?;

    let total_files = files.len();
    for (i, path) in files.iter().enumerate() {
        if i > 0 {
            hooks.emit(IndexProgress::Scanned {
                done: i,
                total: total_files,
            });
        }
        let key = path.to_string_lossy().to_string();
        let may_skip = |meta: &crate::store::DocumentMeta| {
            !options.force && !(options.retry_empty && meta.chunk_keys.is_empty())
        };
        if let Some(meta) = known.get(&key) {
            if may_skip(meta) && unchanged_by_stat(path, meta) {
                report.skipped_unchanged += 1;
                continue;
            }
        }
        let f = match facts(path) {
            Ok(f) => f,
            Err(e) => {
                report.failed.push(format!("{} ({e})", path.display()));
                continue;
            }
        };
        if let Some(meta) = known.get(&key) {
            if may_skip(meta) && meta.hash == f.hash && meta.mtime_secs == f.mtime_secs {
                report.skipped_unchanged += 1;
                continue;
            }
        }
        // Files without text are stored as empty documents (no chunks) so
        // the next pass skips them while unchanged. Other errors may be
        // transient (locked file, flaky drive) and are retried next pass.
        let remember_empty = |docs_meta: &mut HashMap<_, _>| {
            docs_meta.insert(key.clone(), (f.hash, f.mtime_secs, f.size, Vec::new()));
        };
        let sections = match hooks.extract(path) {
            Ok(s) => s,
            Err(e) => {
                report.failed.push(format!("{} ({e:#})", path.display()));
                if e.is::<extract::NoText>() {
                    remember_empty(&mut docs_meta);
                }
                continue;
            }
        };
        let (chunks, pages) = chunker::chunk_sections(&sections);
        if chunks.is_empty() {
            report
                .failed
                .push(format!("{} (no content)", path.display()));
            remember_empty(&mut docs_meta);
            continue;
        }
        let chunk_keys: Vec<ChunkKey> = (0..chunks.len()).map(|i| cursor + i as u64).collect();
        cursor += chunks.len() as u64;
        docs_meta.insert(key.clone(), (f.hash, f.mtime_secs, f.size, chunk_keys));
        all_chunks.extend(chunks);
        all_pages.extend(pages);
        report.indexed.push(path.to_string_lossy().to_string());
    }

    hooks.emit(IndexProgress::Scanned {
        done: total_files,
        total: total_files,
    });

    // Single large embed batch for all chunks across all changed documents.
    // Embed in sub-batches of 128 to bound memory while still letting
    // candle's rayon pool parallelize within each batch (much better than
    // per-document batches of ~16). Log progress per window: a silent
    // 7-minute stretch is indistinguishable from a hang.
    let t_embed = Instant::now();
    let mut all_vectors: Vec<Vec<f32>> = Vec::with_capacity(all_chunks.len());
    if !all_chunks.is_empty() {
        tracing::info!(
            "embedding {} chunks across {} documents",
            all_chunks.len(),
            docs_meta.len()
        );
        let total = all_chunks.len();
        for chunk_window in all_chunks.chunks(128) {
            let batch_vecs = embedder.embed_batch(chunk_window)?;
            all_vectors.extend(batch_vecs);
            let done = all_vectors.len();
            hooks.emit(IndexProgress::Embedding { done, total });
            tracing::info!(
                "embedded {}/{} chunks ({}%)",
                done,
                total,
                done * 100 / total
            );
        }
        tracing::info!("embedding done in {:.1}s", t_embed.elapsed().as_secs_f64());
    }
    // Also runs with no chunks at all, to record files without text.
    if !docs_meta.is_empty() {
        store.upsert_batch_with_pages(&docs_meta, &all_chunks, &all_pages, &all_vectors)?;
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
        if !options.no_prune
            && under_root(Path::new(k), dir)
            && !known_paths.contains(k)
            && !Path::new(k).try_exists().unwrap_or(true)
            && store.delete_document(k)?
        {
            report.removed_missing += 1;
        }
    }

    // Rebuild the BM25 sidecar once for the full corpus.
    tantivy_idx.rebuild_from(store)?;
    hooks.emit(IndexProgress::Finished);

    tracing::debug!(
        "index pass over {} took walk {:.2}s, embed {:.2}s",
        dir.display(),
        walk_s,
        t_embed.elapsed().as_secs_f64()
    );

    Ok(report)
}

fn absolute_source_path(path: &Path) -> Result<PathBuf> {
    let absolute = std::path::absolute(path)?;
    if absolute
        .components()
        .any(|c| c == std::path::Component::ParentDir)
    {
        return absolute
            .canonicalize()
            .context("resolving parent-directory components");
    }
    Ok(absolute)
}

/// Preserve legacy relative keys because their original cwd is unknown.
fn under_root(path: &Path, root: &Path) -> bool {
    path.is_absolute()
        && !path
            .components()
            .any(|c| c == std::path::Component::ParentDir)
        && path.starts_with(root)
}
