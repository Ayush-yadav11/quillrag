//! quillrag library — exposes the engine internals so the CLI, the MCP
//! server, and external benches/tests all share one implementation.

pub mod assets;
pub mod chunker;
pub mod embedder;
pub mod indexer;
pub mod search;
pub mod server;
pub mod store;

pub use chunker::chunk_text;
pub use embedder::Embedder;
pub use search::{hybrid_search, rrf_fuse, TantivyIndex};
pub use store::{ChunkRow, ChunkKey, Hit, Store, StoreStats};
