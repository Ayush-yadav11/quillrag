//! Hybrid retrieval: BM25 via an embedded tantivy sidecar + dense cosine from
//! the vector store, fused with Reciprocal Rank Fusion.

use crate::store::{Hit, Store};
use anyhow::{Context, Result};
use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};
use tantivy::collector::TopDocs;
use tantivy::query::QueryParser;
use tantivy::schema::{NumericOptions, Schema, TantivyDocument, Value, TEXT};
use tantivy::{Index, IndexReader, IndexWriter, ReloadPolicy};

/// Where the sidecar index lives (inside the data dir).
const SIDECAR_DIR: &str = "bm25";

/// u64 stored + fast field.
fn ordinal_options() -> NumericOptions {
    NumericOptions::default()
        .set_fast()
        .set_indexed()
        .set_stored()
}

pub struct TantivyIndex {
    dir: PathBuf,
    index: Index,
    reader: IndexReader,
    /// tantivy's IndexWriter is `commit(&mut self)` and not Clone, so it
    /// lives behind a mutex for shared access.
    writer: Arc<Mutex<IndexWriter>>,
    field_path: tantivy::schema::Field,
    field_ordinal: tantivy::schema::Field,
    field_text: tantivy::schema::Field,
}

impl TantivyIndex {
    pub fn open(data_dir: &Path) -> Result<Self> {
        let dir = data_dir.join(SIDECAR_DIR);
        std::fs::create_dir_all(&dir).with_context(|| format!("creating {}", dir.display()))?;

        let mut sb = Schema::builder();
        let f_path = sb.add_text_field("path", TEXT | tantivy::schema::STORED);
        let f_ord = sb.add_u64_field("ordinal", ordinal_options());
        let f_text = sb.add_text_field("text", TEXT);
        let schema = sb.build();

        let index = match Index::open_in_dir(&dir) {
            Ok(idx) => {
                // Schema drift (version upgrade): wipe and recreate.
                if idx.schema() != schema {
                    std::fs::remove_dir_all(&dir)?;
                    std::fs::create_dir_all(&dir)?;
                    Index::create_in_dir(&dir, schema)?
                } else {
                    idx
                }
            }
            Err(_) => {
                // Fresh dir or unreadable segment -> start a clean index.
                let _ = std::fs::remove_dir_all(&dir);
                std::fs::create_dir_all(&dir)?;
                Index::create_in_dir(&dir, schema)?
            }
        };

        let writer = Arc::new(Mutex::new(index.writer(15_000_000)?)); // 15 MB heap
        let reader: IndexReader = index
            .reader_builder()
            .reload_policy(ReloadPolicy::Manual)
            .try_into()?;

        Ok(Self {
            dir,
            reader,
            writer,
            index,
            field_path: f_path,
            field_ordinal: f_ord,
            field_text: f_text,
        })
    }

    /// Rebuild the whole BM25 index from the store's chunks. At pocket-scale
    /// (hundreds to low-thousands of chunks) a full rebuild per indexing pass
    /// is simpler than incremental commit tracking and takes milliseconds.
    pub fn rebuild_from(&self, store: &Store) -> Result<()> {
        let mut writer = self.writer.lock().expect("tantivy writer mutex poisoned");
        writer.delete_all_documents()?;
        for row in store.iter_chunks()? {
            let mut doc = TantivyDocument::default();
            doc.add_text(self.field_path, &row.path);
            doc.add_u64(self.field_ordinal, row.ordinal as u64);
            doc.add_text(self.field_text, &row.text);
            writer.add_document(doc)?;
        }
        writer.commit()?;
        self.reader.reload()?;
        Ok(())
    }

    /// BM25 keyword search. Returns (path, ordinal) ranked by score.
    pub fn search_bm25(&self, query: &str, limit: usize) -> Result<Vec<(String, usize)>> {
        if query.trim().is_empty() {
            return Ok(Vec::new());
        }
        // Sanitize first: characters like '-' are Lucene operators (`-term`
        // excludes!), so a query for "json-rpc" would otherwise exclude every
        // document mentioning rpc. We trade exotic query syntax for
        // predictable behavior — exact tokens (E0382, MINLML6V2) survive.
        //
        // Stopwords are dropped too: the query parser ORs terms, so "notes on
        // the migration" would otherwise match every document containing
        // "the". A stopword-only query leaves ranking to the dense side.
        let sanitized: String = query
            .split(|c: char| !c.is_alphanumeric())
            .filter(|s| !s.is_empty() && !is_stopword(s))
            .collect::<Vec<_>>()
            .join(" ");
        if sanitized.is_empty() {
            return Ok(Vec::new());
        }

        let searcher = self.reader.searcher();
        let parser = QueryParser::for_index(&self.index, vec![self.field_text, self.field_path]);
        let q = parser
            .parse_query(&sanitized)
            .context("building bm25 query")?;
        let top = searcher.search(&q, &TopDocs::with_limit(limit).order_by_score())?;
        let mut out = Vec::with_capacity(top.len());
        for (_score, addr) in top {
            let doc: TantivyDocument = searcher.doc(addr)?;
            let path = doc
                .get_first(self.field_path)
                .and_then(|v| v.as_str())
                .unwrap_or_default()
                .to_string();
            let ordinal = doc
                .get_first(self.field_ordinal)
                .and_then(|v| v.as_u64())
                .unwrap_or(0) as usize;
            out.push((path, ordinal));
        }
        Ok(out)
    }

    #[allow(dead_code)]
    pub fn dir_path(&self) -> &Path {
        &self.dir
    }
}

/// Common English function words that carry no topical signal for BM25.
const STOPWORDS: &[&str] = &[
    "a", "about", "an", "and", "are", "as", "at", "be", "by", "can", "do", "does", "for", "from",
    "how", "i", "in", "into", "is", "it", "its", "me", "my", "of", "on", "or", "our", "that",
    "the", "their", "this", "to", "was", "we", "what", "when", "where", "which", "who", "why",
    "will", "with", "you", "your",
];

fn is_stopword(token: &str) -> bool {
    token.len() <= 5 && STOPWORDS.contains(&token.to_lowercase().as_str())
}

/// Knobs for [`hybrid_search_with`].
#[derive(Debug, Clone, Copy)]
pub struct SearchOptions {
    /// Dense candidates below this cosine similarity are dropped before
    /// fusion, so unrelated documents stop filling the result list. Keyword
    /// (BM25) matches are kept regardless. `f32::NEG_INFINITY` disables it.
    pub min_similarity: f32,
}

impl Default for SearchOptions {
    fn default() -> Self {
        Self {
            min_similarity: f32::NEG_INFINITY,
        }
    }
}

/// Reciprocal Rank Fusion over dense + BM25 rankings.
///
/// RRF(d) = Σ 1/(k + rank_i(d)); k=60 is the standard constant.
pub fn rrf_fuse(
    dense: Vec<(String, usize)>,
    sparse: Vec<(String, usize)>,
    top_k: usize,
) -> Vec<(String, usize)> {
    const K: f32 = 60.0;
    let mut scores: HashMap<(String, usize), f32> = HashMap::new();
    for (rank, key) in dense.iter().enumerate() {
        *scores.entry(key.clone()).or_default() += 1.0 / (K + rank as f32);
    }
    for (rank, key) in sparse.iter().enumerate() {
        *scores.entry(key.clone()).or_default() += 1.0 / (K + rank as f32);
    }
    let mut fused: Vec<((String, usize), f32)> = scores.into_iter().collect();
    fused.sort_by(|a, b| b.1.total_cmp(&a.1));
    fused.truncate(top_k);
    fused.into_iter().map(|(k, _)| k).collect()
}

/// Full hybrid search pipeline: embed query -> dense scan -> BM25 -> fuse ->
/// hydrate chunk rows from the store.
pub fn hybrid_search(
    query: &str,
    top_k: usize,
    store: &Store,
    tantivy_idx: &TantivyIndex,
    embedder: &mut crate::embedder::Embedder,
) -> Result<Vec<Hit>> {
    hybrid_search_with(
        query,
        top_k,
        store,
        tantivy_idx,
        embedder,
        SearchOptions::default(),
    )
}

/// [`hybrid_search`] with a relevance floor for dense candidates.
pub fn hybrid_search_with(
    query: &str,
    top_k: usize,
    store: &Store,
    tantivy_idx: &TantivyIndex,
    embedder: &mut crate::embedder::Embedder,
    options: SearchOptions,
) -> Result<Vec<Hit>> {
    // 1. Dense candidates: fetch 4x top_k so fusion has overlap to work with.
    let qvec = embedder.embed_one(query)?;
    let dense_keys = store.dense_scan(&qvec)?;
    let mut dense: Vec<(String, usize)> = Vec::with_capacity(top_k * 4);
    for (key, _) in dense_keys
        .into_iter()
        .take_while(|(_, score)| *score >= options.min_similarity)
        .take(top_k * 4)
    {
        if let Some(row) = store.get_chunk_row(key)? {
            dense.push((row.path, row.ordinal));
        }
    }

    // 2. Sparse candidates.
    let sparse = tantivy_idx.search_bm25(query, top_k * 4)?;

    // 3. Fuse ranks.
    let fused = rrf_fuse(dense, sparse, top_k);

    // 4. Hydrate text; score = normalized RRF rank for display.
    let n = fused.len() as f32;
    let mut hits = Vec::with_capacity(fused.len());
    for (i, (path, ordinal)) in fused.iter().enumerate() {
        if let Some(row) = store.get_chunk_row_by_ordinal(path, *ordinal)? {
            hits.push(Hit {
                path: row.path,
                chunk_index: row.ordinal,
                score: ((n - i as f32) / n * 100.0).round() / 100.0,
                text: row.text,
                page: row.page,
            });
        }
    }
    Ok(hits)
}
