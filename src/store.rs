//! Persistence: a single redb database holding chunk text, raw f32 vectors,
//! and document metadata. One file, atomic commits, crash-safe.

use anyhow::{Context, Result};
use redb::{Database, ReadableDatabase, ReadableTable, ReadableTableMetadata, TableDefinition};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::Arc;

/// Bump when on-disk layout changes; mismatch forces full re-index.
pub const SCHEMA_VERSION: u32 = 1;

pub type ChunkKey = u64;

/// chunk_key -> serialized ChunkRow
pub static CHUNKS: TableDefinition<ChunkKey, &str> = TableDefinition::new("chunks");
/// chunk_key -> raw little-endian f32 bytes (384 * 4 = 1536 B)
pub static VECS: TableDefinition<ChunkKey, &[u8]> = TableDefinition::new("vecs");
/// doc_path -> serialized DocumentMeta
pub static DOCS: TableDefinition<&str, &str> = TableDefinition::new("docs");
/// small string-keyed counters ("schema_version", ...)
pub static META: TableDefinition<&str, u64> = TableDefinition::new("meta");

#[derive(Serialize, Deserialize, Clone)]
pub struct ChunkRow {
    pub path: String,
    pub ordinal: usize,
    pub text: String,
}

#[derive(Serialize, Deserialize, Clone)]
pub struct DocumentMeta {
    pub hash: u64,
    pub mtime_secs: i64,
    pub size: u64,
    /// chunk_keys[i] holds the vector+text for chunk ordinal i.
    pub chunk_keys: Vec<ChunkKey>,
}

/// One search hit.
#[derive(Clone, Debug, serde::Serialize)]
pub struct Hit {
    /// Path of the source document (as passed to rag_index).
    pub path: String,
    /// 0-based chunk position within the document.
    pub chunk_index: usize,
    /// Fused relevance score (higher is better).
    pub score: f32,
    /// The chunk text.
    pub text: String,
}

#[derive(Debug, Clone, serde::Serialize)]
pub struct StoreStats {
    pub documents: u64,
    pub chunks: u64,
    pub total_source_bytes: u64,
    pub by_extension: HashMap<String, u64>,
}

pub struct Store {
    db: Arc<Database>,
}

/// Remove a doc's metadata row, returning its chunk keys (owned). Free
/// function so the AccessGuard borrow ends at the function boundary instead
/// of leaking through `?`-in-scrutinee temporaries.
fn take_doc_meta(table: &mut redb::Table<'_, &str, &str>, path: &str) -> Result<Vec<ChunkKey>> {
    let guard = table
        .remove(path)
        .map_err(|e| anyhow::anyhow!("redb remove: {e}"))?;
    if let Some(guard) = guard {
        let json = guard.value().to_string();
        let meta: DocumentMeta = serde_json::from_str(&json)?;
        Ok(meta.chunk_keys)
    } else {
        Ok(Vec::new())
    }
}

impl Store {
    pub fn open(data_dir: &std::path::Path) -> Result<Self> {
        std::fs::create_dir_all(data_dir)
            .with_context(|| format!("creating data dir {}", data_dir.display()))?;
        let db = Database::create(data_dir.join("index.redb")).context("opening index.redb")?;
        let wf = db.begin_write()?;
        {
            let _ = wf.open_table(CHUNKS)?;
            let _ = wf.open_table(VECS)?;
            let _ = wf.open_table(DOCS)?;
            let mut meta = wf.open_table(META)?;
            if meta.get("schema_version")?.is_none() {
                meta.insert("schema_version", SCHEMA_VERSION as u64)?;
            }
        }
        wf.commit()?;
        Ok(Self { db: Arc::new(db) })
    }

    pub fn schema_matches(&self) -> Result<bool> {
        let rx = self.db.begin_read()?;
        let table = rx.open_table(META)?;
        Ok(table
            .get("schema_version")?
            .map(|v| v.value() == SCHEMA_VERSION as u64)
            .unwrap_or(false))
    }

    pub fn get_chunk_row(&self, key: ChunkKey) -> Result<Option<ChunkRow>> {
        let rx = self.db.begin_read()?;
        let chunks = rx.open_table(CHUNKS)?;
        if let Some(bytes) = chunks.get(key)? {
            return Ok(Some(serde_json::from_str(bytes.value())?));
        }
        Ok(None)
    }

    /// Dense scan: (chunk_key, cosine score) sorted desc. Cosine == dot
    /// because every vector is L2-normalized at write time.
    pub fn dense_scan(&self, query: &[f32]) -> Result<Vec<(ChunkKey, f32)>> {
        let rx = self.db.begin_read()?;
        let vecs = rx.open_table(VECS)?;
        let mut out = Vec::with_capacity(vecs.len()? as usize);
        for row in vecs.iter()? {
            let (key, bytes) = row?;
            out.push((key.value(), dot_f32(bytes.value(), query)));
        }
        out.sort_by(|a, b| b.1.total_cmp(&a.1));
        Ok(out)
    }

    /// Insert/replace one document's chunks atomically. Embedding (CPU-bound)
    /// happens before the transaction opens.
    pub fn upsert_document(
        &self,
        path: &str,
        hash: u64,
        mtime_secs: i64,
        size: u64,
        chunks: &[String],
        embedder: &mut crate::embedder::Embedder,
    ) -> Result<usize> {
        let vectors = embedder.embed_batch(chunks)?;
        let next_key = self.next_chunk_key()?;

        // Pre-compute DocumentMeta chunk_keys.
        let chunk_keys: Vec<ChunkKey> = (0..chunks.len()).map(|i| next_key + i as u64).collect();

        let mut docs_meta: HashMap<String, (u64, i64, u64, Vec<ChunkKey>)> = HashMap::new();
        docs_meta.insert(path.to_string(), (hash, mtime_secs, size, chunk_keys));

        self.upsert_batch(&docs_meta, chunks, &vectors)
    }

    /// Insert/replace multiple documents' chunks in a single write
    /// transaction (one fsync for the whole batch). Each entry in
    /// `docs_meta` maps a doc path to (hash, mtime_secs, size, chunk_keys),
    /// with chunk_keys already assigned in ascending order starting from
    /// next_chunk_key().
    ///
    /// chunks/vectors must be aligned 1:1 with the concatenation of all
    /// documents' chunk_keys in the order they appear in docs_meta.
    pub fn upsert_batch(
        &self,
        docs_meta: &HashMap<String, (u64, i64, u64, Vec<ChunkKey>)>,
        chunks: &[String],
        vectors: &[Vec<f32>],
    ) -> Result<usize> {
        let db = self.db.clone();
        let wf = db.begin_write()?;
        {
            // Collect all chunk_keys so we can clean up removed docs in
            // the same transaction.
            let mut all_old_keys: Vec<ChunkKey> = Vec::new();
            for path in docs_meta.keys() {
                let old_keys = {
                    let mut docs = wf.open_table(DOCS)?;
                    take_doc_meta(&mut docs, path)?
                };
                all_old_keys.extend(old_keys);
            }
            if !all_old_keys.is_empty() {
                let mut chunks_t = wf.open_table(CHUNKS)?;
                let mut vecs_t = wf.open_table(VECS)?;
                for k in all_old_keys {
                    chunks_t.remove(k)?;
                    vecs_t.remove(k)?;
                }
            }

            // Write all new chunk/vec rows.
            let mut chunks_t = wf.open_table(CHUNKS)?;
            let mut vecs_t = wf.open_table(VECS)?;
            let mut idx = 0usize;
            for (path, (_hash, _mtime, _size, chunk_keys)) in docs_meta {
                for (i, key) in chunk_keys.iter().enumerate() {
                    let text = chunks.get(idx).map(|s| s.as_str()).unwrap_or("");
                    let vec = vectors.get(idx).cloned().unwrap_or_default();
                    idx += 1;
                    let row = ChunkRow {
                        path: path.clone(),
                        ordinal: i,
                        text: text.to_string(),
                    };
                    let json = serde_json::to_string(&row)?;
                    chunks_t.insert(*key, json.as_str())?;
                    let mut byte_buf = Vec::with_capacity(vec.len() * 4);
                    for v in &vec {
                        byte_buf.extend_from_slice(&v.to_le_bytes());
                    }
                    vecs_t.insert(*key, byte_buf.as_slice())?;
                }
            }

            // Write doc metadata.
            let mut docs = wf.open_table(DOCS)?;
            for (path, (hash, mtime_secs, size, chunk_keys)) in docs_meta {
                let meta = DocumentMeta {
                    hash: *hash,
                    mtime_secs: *mtime_secs,
                    size: *size,
                    chunk_keys: chunk_keys.clone(),
                };
                let json = serde_json::to_string(&meta)?;
                docs.insert(path.as_str(), json.as_str())?;
            }
        }
        wf.commit()?;
        Ok(chunks.len())
    }

    pub fn next_chunk_key(&self) -> Result<u64> {
        let rx = self.db.begin_read()?;
        let vecs = rx.open_table(VECS)?;
        let next = vecs.last()?.map(|(k, _)| k.value() + 1).unwrap_or(0);
        Ok(next)
    }

    /// Remove a document and its chunks. Returns true if it existed.
    pub fn delete_document(&self, path: &str) -> Result<bool> {
        let db = self.db.clone();
        let wf = db.begin_write()?;
        let removed_keys: Option<Vec<ChunkKey>> = {
            let mut docs = wf.open_table(DOCS)?;
            let keys = take_doc_meta(&mut docs, path)?;
            if keys.is_empty() {
                None
            } else {
                Some(keys)
            }
        };
        let existed = removed_keys.is_some();
        if let Some(keys) = removed_keys {
            let mut chunks_t = wf.open_table(CHUNKS)?;
            let mut vecs_t = wf.open_table(VECS)?;
            for k in keys {
                chunks_t.remove(k)?;
                vecs_t.remove(k)?;
            }
        }
        wf.commit()?;
        Ok(existed)
    }

    /// Wipe all content (keeps schema_version).
    pub fn clear(&self) -> Result<()> {
        let db = self.db.clone();
        let wf = db.begin_write()?;
        {
            wf.delete_table(CHUNKS)?;
            wf.delete_table(VECS)?;
            wf.delete_table(DOCS)?;
            let _ = wf.open_table(CHUNKS)?;
            let _ = wf.open_table(VECS)?;
            let _ = wf.open_table(DOCS)?;
            let mut meta = wf.open_table(META)?;
            meta.insert("schema_version", SCHEMA_VERSION as u64)?;
        }
        wf.commit()?;
        Ok(())
    }

    pub fn stats(&self) -> Result<StoreStats> {
        let rx = self.db.begin_read()?;
        let docs = rx.open_table(DOCS)?;
        let chunks = rx.open_table(CHUNKS)?;

        let mut by_ext: HashMap<String, u64> = HashMap::new();
        let mut total_bytes = 0u64;
        for row in docs.iter()? {
            let (path, bytes) = row?;
            let ext = std::path::Path::new(path.value())
                .extension()
                .and_then(|e| e.to_str())
                .unwrap_or("noext")
                .to_lowercase();
            *by_ext.entry(ext).or_default() += 1;
            let meta: DocumentMeta = serde_json::from_str(bytes.value())?;
            total_bytes += meta.size;
        }
        Ok(StoreStats {
            documents: docs.len()?,
            chunks: chunks.len()?,
            total_source_bytes: total_bytes,
            by_extension: by_ext,
        })
    }

    pub fn list_documents(&self) -> Result<HashMap<String, DocumentMeta>> {
        let rx = self.db.begin_read()?;
        let docs = rx.open_table(DOCS)?;
        let mut out = HashMap::new();
        for row in docs.iter()? {
            let (path, bytes) = row?;
            let meta: DocumentMeta = serde_json::from_str(bytes.value())?;
            out.insert(path.value().to_string(), meta);
        }
        Ok(out)
    }

    /// All chunk rows sorted by (path, ordinal) — feeds tantivy indexing.
    pub fn iter_chunks(&self) -> Result<Vec<ChunkRow>> {
        let rx = self.db.begin_read()?;
        let chunks = rx.open_table(CHUNKS)?;
        let mut rows = Vec::with_capacity(chunks.len()? as usize);
        for r in chunks.iter()? {
            let (_, v) = r?;
            rows.push(serde_json::from_str::<ChunkRow>(v.value())?);
        }
        rows.sort_by(|a, b| (&a.path, a.ordinal).cmp(&(&b.path, b.ordinal)));
        Ok(rows)
    }

    /// Fetch a chunk row by document path + ordinal (search hydration path).
    pub fn get_chunk_by_ordinal(
        &self,
        path: &str,
        ordinal: usize,
    ) -> Result<Option<(String, usize, String)>> {
        let rx = self.db.begin_read()?;
        let docs = rx.open_table(DOCS)?;
        if let Some(bytes) = docs.get(path)? {
            let meta: DocumentMeta = serde_json::from_str(bytes.value())?;
            drop(docs);
            if let Some(key) = meta.chunk_keys.get(ordinal) {
                if let Some(row) = self.get_chunk_row(*key)? {
                    return Ok(Some((row.path, row.ordinal, row.text)));
                }
            }
        }
        Ok(None)
    }
}

fn dot_f32(bytes: &[u8], query: &[f32]) -> f32 {
    bytes
        .chunks_exact(4)
        .zip(query.iter())
        .map(|(b, q)| f32::from_le_bytes([b[0], b[1], b[2], b[3]]) * q)
        .sum()
}
