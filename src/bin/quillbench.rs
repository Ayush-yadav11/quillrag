//! quillbench — synthetic scale/latency probe for quillrag.
//!
//! Builds N chunks through the REAL index path (chunk → embed → store) and
//! times steady-state hybrid_search across corpus sizes. Measures exactly
//! what production queries pay: redb dense scan + tantivy BM25 + RRF fusion.
//!
//! Usage:
//!   cargo build --release
//!   ./target/release/quillbench --sizes 1000,10000,100000

use std::time::Instant;

use clap::{Parser, ValueEnum};
use quillrag::{hybrid_search, Embedder, Store, TantivyIndex};
use std::hash::{Hash, Hasher};

#[derive(Parser)]
#[command(
    name = "quillbench",
    about = "synthetic scale/latency probe for quillrag"
)]
struct Cli {
    /// Base data dir; one fresh subdir per size.
    #[arg(long, default_value = "/tmp/quillbench")]
    data_dir: String,
    /// Chunk counts to probe, comma-separated.
    #[arg(long, default_value = "1000,5000,10000,25000,50000")]
    sizes: String,
    /// Queries per size (steady-state, after warmup).
    #[arg(long, default_value_t = 15)]
    queries: usize,
}

#[derive(Copy, Clone, PartialEq, Eq, ValueEnum)]
enum Mode {
    /// All chunks in one document.
    Single,
    /// One chunk per document.
    Many,
}

fn main() -> anyhow::Result<()> {
    let cli = Cli::parse();
    let sizes: Vec<usize> = cli
        .sizes
        .split(',')
        .map(|s| s.trim().parse().expect("numeric size"))
        .collect();

    println!("chunks,index_s,query_p50_ms,query_max_ms,vectors_mb");

    for &n in &sizes {
        let dir = format!("{}/n{}", cli.data_dir, n);
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir)?;

        let store = Store::open(std::path::Path::new(&dir))?;
        let bm25 = TantivyIndex::open(std::path::Path::new(&dir))?;
        let mut emb = Embedder::load(std::path::Path::new(&dir))?;

        // Index n synthetic chunks through the real upsert path. Each chunk
        // carries a unique token so both dense and BM25 can find it.
        let t0 = Instant::now();
        // One doc per chunk through the real upsert path (embed + store +
        // atomic commit per doc), then one BM25 rebuild like the indexer does.
        for i in 0..n {
            let text = synthetic_chunk(i);
            let id = format!("doc{i}.txt");
            store.upsert_document(&id, hash_u64(&text), 0, 0, &[text], &mut emb)?;
        }
        bm25.rebuild_from(&store)?;
        let index_s = t0.elapsed().as_secs_f64();

        // Warmup pays the one-time model load inside hybrid_search.
        let _ = hybrid_search(
            "warmup subject token retrieval benchmark",
            5,
            &store,
            &bm25,
            &mut emb,
        )?;

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
        println!("{n},{index_s:.2},{p50},{max},{mb:.1}");
    }
    Ok(())
}

fn synthetic_chunk(i: usize) -> String {
    format!(
        "subject token zz{i:08x} the quick brown fox jumps over the lazy dog \
         embedding retrieval benchmark synthetic corpus item number {i} filler \
         text padding to reach realistic chunk size for tokenizer exercise now"
    )
}

fn hash_u64<T: Hash>(v: &T) -> u64 {
    let mut h = std::collections::hash_map::DefaultHasher::new();
    v.hash(&mut h);
    h.finish()
}
