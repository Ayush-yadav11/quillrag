//! quillrag library — exposes the engine internals so the CLI, the MCP
//! server, and external benches/tests all share one implementation.
//!
//! The engine (store, embedder, indexer, search) has no async or CLI
//! dependencies. The MCP server is behind the `mcp` feature; depend on this
//! crate with `default-features = false` to embed just the engine.

pub mod assets;
pub mod chunker;
pub mod embedder;
pub mod extract;
pub mod indexer;
pub mod search;
#[cfg(feature = "mcp")]
pub mod server;
pub mod store;

pub use chunker::chunk_text;
pub use embedder::Embedder;
#[cfg(feature = "pdf")]
pub use extract::PdfExtractor;
pub use extract::{Extractor, NoText, Section};
#[cfg(feature = "ocr")]
pub use extract::{ImageExtractor, Ocr};
pub use indexer::{IndexHooks, IndexOptions, IndexProgress, IndexReport};
pub use search::{hybrid_search, hybrid_search_with, rrf_fuse, SearchOptions, TantivyIndex};
pub use store::{ChunkKey, ChunkRow, Hit, Store, StoreStats};
