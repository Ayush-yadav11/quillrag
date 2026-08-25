//! MCP tool layer: four tools over stdio — rag_search, rag_index, rag_status,
//! rag_clear. The embedder is lazy so the initialize handshake completes in
//! milliseconds before any model loading happens.

use crate::embedder::Embedder;
use crate::indexer;
use crate::search::{self, TantivyIndex};
use crate::store::Store;
use anyhow::Result;
use rmcp::{
    handler::server::wrapper::Parameters, model::*, schemars, tool, tool_handler, tool_router,
    ErrorData as McpError, ServerHandler,
};
use std::path::PathBuf;
use std::sync::{Arc, Mutex};

/// Shared engine handles. One Store / one TantivyIndex per process — redb
/// file-locks its database and tantivy allows a single writer, so every tool
/// funnels through these Arcs instead of reopening files.
pub struct Engine {
    pub data_dir: PathBuf,
    pub store: Arc<Store>,
    pub bm25: Arc<TantivyIndex>,
}

#[derive(Debug, serde::Deserialize, schemars::JsonSchema)]
pub struct SearchArgs {
    /// The natural-language question or keywords to search for.
    pub query: String,
    /// Max results to return (default 5, capped at 25).
    #[serde(default)]
    #[schemars(description = "Max results to return (default 5, max 25)")]
    pub top_k: Option<u32>,
}

#[derive(Debug, serde::Deserialize, schemars::JsonSchema)]
pub struct IndexArgs {
    /// Absolute path to a directory of documents (or a single file) to index.
    pub path: String,
    /// Extra file extensions to include beyond the defaults, e.g. ["log2"].
    #[serde(default)]
    #[schemars(description = "Extra file extensions to include beyond the defaults")]
    pub extensions: Option<Vec<String>>,
}

#[derive(Clone)]
pub struct PocketRag {
    engine: Arc<Engine>,
    /// Lazily-initialized MiniLM. First search/index pays ~1-2 s once; the
    /// initialize handshake never touches this.
    embedder: Arc<Mutex<Option<Embedder>>>,
}

#[tool_router]
impl PocketRag {
    pub fn new(data_dir: PathBuf) -> Result<Self> {
        let store = Arc::new(Store::open(&data_dir)?);
        let bm25 = Arc::new(TantivyIndex::open(&data_dir)?);
        Ok(Self {
            engine: Arc::new(Engine {
                data_dir,
                store,
                bm25,
            }),
            embedder: Arc::new(Mutex::new(None)),
        })
    }

    /// Lock and (if needed) load the embedder. Blocking by design: callers
    /// hand the guard to spawn_blocking so inference never stalls the runtime.
    fn embedder(&self) -> Result<Arc<Mutex<Option<Embedder>>>> {
        let arc = self.embedder.clone();
        {
            let mut guard = arc.lock().expect("embedder mutex poisoned");
            if guard.is_none() {
                *guard = Some(
                    Embedder::load(&self.engine.data_dir.join("model")).inspect_err(|e| {
                        tracing::error!(error = %e, "failed to load embedding model");
                    })?,
                );
                tracing::info!("embedding model loaded");
            }
        }
        Ok(arc)
    }

    #[tool(
        description = "Search the local knowledge base with hybrid semantic + keyword retrieval. Returns ranked chunks with source paths."
    )]
    async fn rag_search(
        &self,
        Parameters(args): Parameters<SearchArgs>,
    ) -> Result<CallToolResult, McpError> {
        let top_k = args.top_k.unwrap_or(5).clamp(1, 25) as usize;
        let embedder_arc = self.embedder().map_err(|e| internal(e.to_string()))?;
        let store = self.engine.store.clone();
        let bm25 = self.engine.bm25.clone();
        let query = args.query;

        let result = tokio::task::spawn_blocking(move || -> Result<Vec<crate::store::Hit>> {
            let mut guard = embedder_arc.lock().expect("embedder mutex poisoned");
            let embedder = guard.as_mut().expect("embedder loaded above");
            search::hybrid_search(&query, top_k, &store, &bm25, embedder)
        })
        .await;

        match result {
            Ok(Ok(hits)) => {
                if hits.is_empty() {
                    return Ok(tool_text(
                        "No results. The index may be empty — try rag_index first.",
                    ));
                }
                let mut out = String::new();
                for (i, hit) in hits.iter().enumerate() {
                    out.push_str(&format!(
                        "[{}] {}#chunk{} (score {:.2})\n{}\n\n",
                        i + 1,
                        hit.path,
                        hit.chunk_index,
                        hit.score,
                        hit.text
                    ));
                }
                Ok(tool_text(out.trim_end()))
            }
            Ok(Err(e)) => Err(internal(format!("search failed: {e:#}"))),
            Err(e) => Err(internal(format!("search task panicked: {e}"))),
        }
    }

    #[tool(
        description = "Index documents into the local knowledge base. Pass a directory for an incremental walk (skips unchanged files, prunes deleted ones) or a single file. Safe to re-run."
    )]
    async fn rag_index(
        &self,
        Parameters(args): Parameters<IndexArgs>,
    ) -> Result<CallToolResult, McpError> {
        let path = PathBuf::from(expand_home(&args.path));
        let extra_exts = args.extensions.unwrap_or_default();
        let embedder_arc = self.embedder().map_err(|e| internal(e.to_string()))?;
        let store = self.engine.store.clone();
        let bm25 = self.engine.bm25.clone();

        let outcome = tokio::task::spawn_blocking(move || -> Result<String> {
            let mut guard = embedder_arc.lock().expect("embedder mutex poisoned");
            let embedder = guard.as_mut().expect("embedder loaded above");

            if path.is_dir() {
                let report = indexer::index_directory(&path, &extra_exts, &store, &bm25, embedder)?;
                Ok(report.summary())
            } else if path.is_file() {
                let n = indexer::index_one(&path, &store, &bm25, embedder)?;
                Ok(format!(
                    "indexed 1 file ({n} chunks); run with a directory for incremental sync"
                ))
            } else {
                anyhow::bail!("path does not exist: {}", path.display());
            }
        })
        .await;

        match outcome {
            Ok(Ok(summary)) => Ok(tool_text(&format!("Index complete: {summary}"))),
            Ok(Err(e)) => Err(internal(format!("indexing failed: {e:#}"))),
            Err(e) => Err(internal(format!("index task panicked: {e}"))),
        }
    }

    #[tool(description = "Show knowledge base stats: documents, chunks, size, file types.")]
    async fn rag_status(&self) -> Result<CallToolResult, McpError> {
        match self.engine.store.stats() {
            Ok(stats) => {
                let mut pairs: Vec<_> = stats.by_extension.iter().collect();
                pairs.sort_by_key(|(_, c)| std::cmp::Reverse(**c));
                let exts = pairs
                    .iter()
                    .map(|(e, c)| format!(".{e}: {c}"))
                    .collect::<Vec<_>>()
                    .join(", ");
                Ok(tool_text(&format!(
                    "documents: {}\nchunks: {}\nsource bytes: {}\ntypes: {}\ndata dir: {}",
                    stats.documents,
                    stats.chunks,
                    stats.total_source_bytes,
                    if exts.is_empty() {
                        "(empty)".into()
                    } else {
                        exts
                    },
                    self.engine.data_dir.display(),
                )))
            }
            Err(e) => Err(internal(format!("status failed: {e:#}"))),
        }
    }

    #[tool(description = "Delete ALL indexed documents and embeddings from the knowledge base.")]
    async fn rag_clear(&self) -> Result<CallToolResult, McpError> {
        let outcome = tokio::task::spawn_blocking({
            let store = self.engine.store.clone();
            let bm25 = self.engine.bm25.clone();
            move || -> Result<()> {
                store.clear()?;
                bm25.rebuild_from(&store)?;
                Ok(())
            }
        })
        .await;

        match outcome {
            Ok(Ok(())) => Ok(tool_text("Knowledge base cleared.")),
            Ok(Err(e)) => Err(internal(format!("clear failed: {e:#}"))),
            Err(e) => Err(internal(format!("clear task panicked: {e}"))),
        }
    }
}

// ---------- helpers ----------

fn tool_text(s: &str) -> CallToolResult {
    CallToolResult::success(vec![ContentBlock::text(s.to_string())])
}

fn internal(msg: String) -> McpError {
    McpError::internal_error(msg, None)
}

fn expand_home(p: &str) -> String {
    if let Some(rest) = p.strip_prefix("~/") {
        if let Ok(home) = std::env::var("HOME") {
            return format!("{home}/{rest}");
        }
    }
    p.to_string()
}

#[tool_handler]
impl ServerHandler for PocketRag {
    fn get_info(&self) -> ServerInfo {
        ServerInfo::new(ServerCapabilities::builder().enable_tools().build())
            .with_server_info(
                Implementation::new("pocketrag", env!("CARGO_PKG_VERSION"))
                    .with_title("PocketRAG — single-binary local RAG"),
            )
            .with_instructions(
                "Local-first RAG over your own files. Workflow: call rag_index with a \
                 directory once (incremental afterwards), then rag_search questions against \
                 it. Hybrid dense (MiniLM, embedded in-binary) + BM25 retrieval runs fully \
                 offline; nothing leaves your machine."
                    .to_string(),
            )
    }
}
