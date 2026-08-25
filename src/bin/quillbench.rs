//! quillbench — synthetic scale / latency probe for quillrag.
//!
//! Builds N synthetic chunks through the REAL index path (chunk -> embed ->
//! store -> BM25 rebuild) and then times steady-state hybrid_search across
//! corpus sizes. This measures exactly what a production query pays:
//! redb dense scan + tantivy BM25 + RRF fusion.
//!
//! Run:
//!   cargo build --release
//!   ./target/release/quillbench --sizes 1000,5000,10000,25000,50000,100000

use std::path::PathBuf;
use std::time::Instant;

use clap::{Parser, ValueEnum};
use quillrag::{chunk_text, hybrid_search, Embedder, Store, TantivyIndex};
use std::hash::{Hash, Hasher};

#[derive(Parser)]
#[command(name = "quillbench", about = "synthetic scale/latency probe for quillrag")]
struct Cli {
    /// Output data dir (fresh subdir per size).
    #[arg(long, default_value = "/tmp/quillbench")]
    data_dir: PathBuf,
    /// Chunk counts to probe, comma-separated.
    #[arg(long, default_value = "1000,5000,10000,25000,50000,100000")]
    sizes: String,
    /// Bytes per synthetic chunk body.
    #[arg(long, default_value_t = 600)]
    chunk_bytes: usize,
    /// Queries per size (steady-state, after warmup).
    #[arg(long, default_value_t = 20)]
    queries: usize,
    /// How to distribute chunks across documents.
    #[arg(long, value_enum, default_value_t = Mode::Single)]
    mode: Mode,
}

#[derive(Copy, Clone, PartialEq, Eq, ValueEnum)]
enum Mode {
    /// All chunks in one document (worst case for BM25 field routing).
    Single,
    /// One chunk per document (typical "many small files" corpus).
    Many,
}

fn main() -> anyhow::Result<()> {
    let cli = Cli::parse();
    let sizes: Vec<usize> = cli
        .sizes
        .split(',')
        .map(|s| s.trim().parse().expect("numeric size"))
        .collect();

    println!("size,mode,index_ms,query_ms_p50,query_ms_max,vectors_mb");

    for &n in &sizes {
        let dir = cli.data_dir.join(format!("n{n}"));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir)?;

        let store = Store::open(&dir)?;
        let bm25 = TantivyIndex::open(&dir)?;
        let mut emb = Embedder::load(&dir)?;

        // Build n chunks. Hash/mtime/size are dummy but fed through the real
        // upsert path so we exercise the same code the indexer uses.
        let t0 = Instant::now();
        match cli.mode {
            Mode::Single => {
                let mut chunks = Vec::with_capacity(n);
                for i in 0..n {
                    chunks.push(synthetic_chunk(i, cli.chunk_bytes));
                }
                let h = hash_u64(&chunks);
                store.upsert_document("bigdoc.txt", h, 0, 0, &chunks, &mut emb)?;
                bm25.rebuild_from(&store)?;
            }
            Mode::Many => {
                for i in 0..n {
                    let chunk = synthetic_chunk(i, cli.chunk_bytes);
                    let h = hash_u64(&chunk);
                    store.upsert_document(&format!("doc{i}.txt"), h, 0, 0, &[chunk], &mut emb)?;
                }
                bm25.rebuild_from(&store)?;
            }
        }
        let index_ms = t0.elapsed().as_millis();

        // Warmup (pays one-time model load inside hybrid_search's embedder).
        let _ = hybrid_search("warmup subject token retrieval benchmark", 5, &store, &bm25, &mut emb)?;

        let mut lat = Vec::with_capacity(cli.queries);
        let mut max = 0u128;
        for q in 0..cli.queries {
            let query = format!("subject token zz{q:08x} retrieval benchmark");
            let t0 = Instant::now();
            let _ = hybrid_search(&query, 5, &store, &bm25, &mut emb)?;
            let d = t0.elapsed().as_millis();
            lat.push(d);
            max = max.max(d);
        }
        lat.sort_unstable();
        let p50 = lat[lat.len() / 2];
        let mb = (n * 384 * 4) as f64 / 1e6;
        let mode = match cli.mode {
            Mode::Single => "single",
            Mode::Many => "many",
        };
        println!("{n},{mode},{index_ms},{p50},{max},{mb:.1}");
    }
    Ok(())
}

/// Deterministic synthetic chunk: unique token per index (so dense + BM25 can
/// both find it) plus filler to reach chunk_bytes.
fn synthetic_chunk(i: usize, bytes: usize) -> String {
    let base = format!(
        "subject token zz{i:08x} the quick brown fox jumps over the lazy dog \
         embedding retrieval benchmark synthetic corpus item number {i} filler"
    );
    if base.len() >= bytes {
        base
    } else {
        format!("{base} {}", "x".repeat(bytes - base.len()))
    }
}

fn hash_u64<T: Hash>(v: &T) -> u64 {
    let mut h = std::collections::hash_map::DefaultHasher::new();
    v.hash(&mut h);
    h.finish()
}
