//! quillrag — a single-binary local RAG MCP server.
//!
//! Subcommands:
//!   serve    Run as an MCP stdio server (default for editor/agent config)
//!   index    Incrementally index a directory or file, then exit
//!   search   One-shot hybrid search from the command line
//!   status   Print knowledge base stats
//!   clear    Wipe the knowledge base
//!   bench    Measure per-phase latency (embed/dense/bm25/fuse) for N queries

// Engine lives in the lib crate (shared with tests/benches); re-export at
// the binary root so existing `store::`, `search::` etc. paths still work.
use quillrag::{embedder, indexer, search, server, store};

use anyhow::{Context, Result};
use clap::{Parser, Subcommand};
use rmcp::ServiceExt;
use std::io::Write;
use std::path::PathBuf;

fn default_data_dir() -> PathBuf {
    if let Ok(d) = std::env::var("QUILLRAG_DATA") {
        return PathBuf::from(d);
    }
    let base = std::env::var("XDG_DATA_HOME")
        .ok()
        .map(PathBuf::from)
        .or_else(|| {
            std::env::var("HOME")
                .ok()
                .map(|h| PathBuf::from(h).join(".local/share"))
        })
        .or_else(|| std::env::var("APPDATA").ok().map(PathBuf::from))
        .unwrap_or_else(|| PathBuf::from("."));
    base.join("quillrag")
}

#[derive(Parser)]
#[command(
    name = "quillrag",
    version,
    about = "Single-binary local RAG MCP server — embeddings compiled in, zero runtime downloads.",
    long_about = None
)]
struct Cli {
    /// Data directory for the index + materialized model cache.
    #[arg(long, global = true, env = "QUILLRAG_DATA", default_value_os_t = default_data_dir())]
    data_dir: PathBuf,

    #[command(subcommand)]
    cmd: Cmd,
}

#[derive(Subcommand)]
enum Cmd {
    /// Run as an MCP server over stdio (what you point your editor at).
    Serve,
    /// Incrementally index a directory or file.
    Index {
        /// Directory or file to index.
        path: PathBuf,
        /// Extra extensions to include beyond defaults.
        #[arg(short = 'e', long, value_delimiter = ',')]
        extensions: Vec<String>,
        /// Re-chunk and re-embed every known document from scratch.
        #[arg(long)]
        force: bool,
    },
    /// Search the index once and print results.
    Search {
        query: String,
        #[arg(short = 'k', long, default_value_t = 5)]
        top_k: usize,
    },
    /// Show index statistics.
    Status,
    /// Delete everything in the knowledge base.
    Clear,
    /// Measure per-phase latency for N queries (spike-only).
    Bench {
        /// Number of queries to run (use the sample-docs corpus).
        #[arg(short = 'n', long, default_value_t = 10)]
        iterations: usize,
    },
}

fn setup_tracing() {
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env().unwrap_or_else(|_| {
                // Our logs at info, third-party crate noise (tantivy segment
                // churn) suppressed unless explicitly requested.
                tracing_subscriber::EnvFilter::new("quillrag=info,warn")
            }),
        )
        .with_writer(std::io::stderr)
        .with_ansi(false)
        .init();
}

/// Shared engine handles for CLI subcommands.
struct EngineBundle {
    store: store::Store,
    bm25: search::TantivyIndex,
    embedder: embedder::Embedder,
}

/// Load order note: Store/TantivyIndex are instant; the embedder pays the
/// one-time model cost. Kept separate so future subcommands can skip it.
fn open_engine(data_dir: &std::path::Path) -> Result<EngineBundle> {
    Ok(EngineBundle {
        store: store::Store::open(data_dir)?,
        bm25: search::TantivyIndex::open(data_dir)?,
        embedder: embedder::Embedder::load(&data_dir.join("model"))?,
    })
}

fn print_hits(hits: &[store::Hit]) {
    if hits.is_empty() {
        println!("(no results)");
        return;
    }
    for (i, hit) in hits.iter().enumerate() {
        println!(
            "[{}] {}#chunk{} (score {:.2})",
            i + 1,
            hit.path,
            hit.chunk_index,
            hit.score
        );
        for line in hit.text.lines().take(6) {
            println!("    {line}");
        }
        println!();
    }
}

#[tokio::main]
async fn main() -> Result<()> {
    let cli = Cli::parse();

    match cli.cmd {
        // MCP mode: logs MUST go to stderr — stdout is the protocol channel.
        Cmd::Serve => {
            setup_tracing();
            let t0 = std::time::Instant::now();
            let svc = server::QuillRag::new(cli.data_dir.clone())
                .context("initializing quillrag engine")?;
            tracing::info!(
                "quillrag {} ready in {:?}",
                env!("CARGO_PKG_VERSION"),
                t0.elapsed()
            );

            let running = svc.serve(rmcp::transport::stdio()).await.inspect_err(|e| {
                tracing::error!("serving error: {e:?}");
            })?;
            running.waiting().await?;
            Ok(())
        }
        Cmd::Index {
            path,
            extensions,
            force,
        } => {
            setup_tracing();
            let mut eng = open_engine(&cli.data_dir)?;
            if force {
                eng.store.clear()?;
            }
            let report = if path.is_dir() {
                indexer::index_directory(
                    &path,
                    &extensions,
                    &eng.store,
                    &eng.bm25,
                    &mut eng.embedder,
                )?
            } else if path.is_file() {
                let n = indexer::index_one(&path, &eng.store, &eng.bm25, &mut eng.embedder)?;
                indexer::IndexReport {
                    indexed: vec![format!("{} ({n} chunks)", path.display())],
                    skipped_unchanged: 0,
                    removed_missing: 0,
                    failed: vec![],
                }
            } else {
                anyhow::bail!("path does not exist: {}", path.display());
            };
            println!("{}", report.summary());
            Ok(())
        }
        Cmd::Search { query, top_k } => {
            setup_tracing();
            let mut eng = open_engine(&cli.data_dir)?;
            let hits =
                search::hybrid_search(&query, top_k, &eng.store, &eng.bm25, &mut eng.embedder)?;
            print_hits(&hits);
            Ok(())
        }
        Cmd::Status => {
            let s = store::Store::open(&cli.data_dir)?.stats()?;
            println!(
                "documents: {}\nchunks:     {}\nsource bytes: {}\ndata dir: {}",
                s.documents,
                s.chunks,
                s.total_source_bytes,
                cli.data_dir.display()
            );
            Ok(())
        }
        Cmd::Clear => {
            setup_tracing();
            let eng = open_engine(&cli.data_dir)?;
            eng.store.clear()?;
            eng.bm25.rebuild_from(&eng.store)?;
            println!("cleared.");
            Ok(())
        }
        Cmd::Bench { iterations } => {
            setup_tracing();
            let mut eng = open_engine(&cli.data_dir)?;
            println!(
                "chunks: {} | iterations: {}",
                eng.store.stats()?.chunks,
                iterations
            );
            let queries = [
                "how does embedding",
                "bm25 search algorithm",
                "chunking strategy overlap",
                "vector cosine similarity",
                "mcp server protocol",
                "rag pipeline components",
                "miniLM model architecture",
                "redb storage internals",
                "tantivy index performance",
                "rust binary distribution",
                "hybrid retrieval fusion",
                "reciprocal rank fusion",
                "tokenizers crate behavior",
                "cargo build optimization",
                "clippy linter usage in rust",
            ];
            let mut embed_ms = Vec::new();
            let mut dense_ms = Vec::new();
            let mut bm25_ms = Vec::new();
            let mut fuse_ms = Vec::new();
            let mut total_ms = Vec::new();

            for (i, q) in queries.iter().cycle().take(iterations).enumerate() {
                let t0 = std::time::Instant::now();
                let qvec = eng.embedder.embed_one(q).unwrap();
                let t_embed = t0.elapsed();

                let t1 = std::time::Instant::now();
                let dense_keys = eng.store.dense_scan(&qvec).unwrap();
                let t_dense = t1.elapsed();

                let t2 = std::time::Instant::now();
                let sparse = eng.bm25.search_bm25(q, 25).unwrap();
                let t_bm25 = t2.elapsed();

                let t3 = std::time::Instant::now();
                let mut dense: Vec<(String, usize)> = Vec::new();
                for (key, _) in dense_keys.into_iter().take(25) {
                    if let Some(row) = eng.store.get_chunk_row(key).unwrap() {
                        dense.push((row.path, row.ordinal));
                    }
                }
                let fused = search::rrf_fuse(dense, sparse, 5);
                let t_fuse = t3.elapsed();

                let total = t_embed + t_dense + t_bm25 + t_fuse;
                if i > 0 {
                    embed_ms.push(t_embed.as_secs_f64() * 1000.0);
                    dense_ms.push(t_dense.as_secs_f64() * 1000.0);
                    bm25_ms.push(t_bm25.as_secs_f64() * 1000.0);
                    fuse_ms.push(t_fuse.as_secs_f64() * 1000.0);
                    total_ms.push(total.as_secs_f64() * 1000.0);
                }
            }

            if total_ms.is_empty() {
                println!("need at least 2 iterations for timing — rerun with -n 2");
                return Ok(());
            }

            let p = |v: &Vec<f64>| {
                let n = v.len() as f64;
                let s: f64 = v.iter().sum();
                let mean = s / n;
                let mut sorted = v.clone();
                sorted.sort_by(|a, b| a.total_cmp(b));
                let p50 = sorted[(n * 0.5) as usize];
                let p95 = sorted[(n * 0.95) as usize];
                (mean, p50, p95)
            };

            let (e_m, e_p50, e_p95) = p(&embed_ms);
            let (d_m, d_p50, d_p95) = p(&dense_ms);
            let (b_m, b_p50, b_p95) = p(&bm25_ms);
            let (f_m, f_p50, f_p95) = p(&fuse_ms);
            let (t_m, t_p50, t_p95) = p(&total_ms);

            println!();
            println!("phase    | count | mean  | p50   | p95");
            println!("---------|-------|-------|-------|------");
            println!(
                "embed    | {:<5} | {:>5.1} | {:>5.1} | {:>5.1} ms",
                iterations, e_m, e_p50, e_p95
            );
            println!(
                "dense    | {:<5} | {:>5.1} | {:>5.1} | {:>5.1} ms",
                iterations, d_m, d_p50, d_p95
            );
            println!(
                "bm25     | {:<5} | {:>5.1} | {:>5.1} | {:>5.1} ms",
                iterations, b_m, b_p50, b_p95
            );
            println!(
                "fuse     | {:<5} | {:>5.1} | {:>5.1} | {:>5.1} ms",
                iterations, f_m, f_p50, f_p95
            );
            println!("---------|-------|-------|-------|------");
            println!(
                "total    | {:<5} | {:>5.1} | {:>5.1} | {:>5.1} ms",
                iterations, t_m, t_p50, t_p95
            );
            writeln!(std::io::stderr(), "bench done @ {}", t_m).ok();
            Ok(())
        }
    }
}
