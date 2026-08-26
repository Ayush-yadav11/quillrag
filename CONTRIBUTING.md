# Contributing to quillrag

Thanks for your interest! quillrag is deliberately small — a focused engine,
not a framework — so contributions are best when they're surgical. This guide
gets you from clone to merged PR.

## Development setup

```sh
git clone https://github.com/Ayush-yadav11/quillrag.git
cd pocketrag                 # note: folder keeps the historical name; crate is quillrag
cargo build --release        # debug works too, but see the caveat below
cargo test --release
```

> **Always test with `--release`.** Debug-mode candle inference is orders of
> magnitude slower — the e2e suite will appear to hang.

Model weights (~90 MB safetensors) are committed under `assets/` as regular
files, so a fresh clone just works. First build takes a few minutes (the
release profile runs LTO).

## Project layout

| Path | What lives there |
|---|---|
| `src/assets.rs` | model bytes compiled in via `include_bytes!`, cache materialization |
| `src/embedder.rs` | candle BERT forward pass, masked mean-pool + L2 norm |
| `src/store.rs` | redb persistence: chunks, f32 vectors, doc metadata; `dense_scan` |
| `src/search.rs` | tantivy BM25 sidecar, RRF fusion, `hybrid_search` |
| `src/indexer.rs` | incremental walk, FNV-hash skip/prune |
| `src/chunker.rs` | paragraph-first splitting |
| `src/server.rs` | rmcp MCP layer: `rag_index/search/status/clear` |
| `src/lib.rs` | module exports shared by the CLI, tests, and benches |
| `src/bin/quillbench.rs` | synthetic latency/scale probe |

## Ground rules

- **stdio is protocol territory.** Logs go to stderr only (`tracing`). Never
  `println!` in library code — stdout carries JSON-RPC when serving.
- **Errors**: `anyhow::Result` with `.context()` at boundaries. MCP handlers
  translate to `ErrorData` in `server.rs`, nowhere else.
- **No `unsafe`.** Candle/redb/tantivy already encapsulate theirs.
- **Tests**: unit tests live next to the code; end-to-end coverage belongs in
  `tests/e2e.rs` (real binary, real stdio). If you change retrieval behavior,
  add an e2e assertion that would have caught the bug.
- **Formatting**: `cargo fmt` before committing; `cargo clippy` clean-ish (no
  new warnings).
- **Benchmarks**: if you touch the hot path (`dense_scan`, fusion, embedder),
  run `quillbench` before/after and paste both tables into the PR.

## Pull requests

1. Branch from `master`, keep the PR focused on one thing.
2. Update the README (and the Changelog section) if behavior, flags, or
   documented numbers change.
3. Describe *why*, not just *what* — the diff shows what.
4. CI builds release artifacts on version tags; PRs are validated by the test
   suite locally. (A PR-triggered test workflow would itself be a welcome
   contribution — see below.)

## Good first issues

These are scoped, self-contained, and reviewed promptly:

- **Multi-threaded dense scan** — parallelize `Store::dense_scan` (#2)
- **Scalar quantization (SQ8) for stored vectors** — 4× RAM cut (#3)
- **Reranker hook** — pluggable post-fusion rescoring stage (#4)

Bigger arcs live in [#1 — ANN index](https://github.com/Ayush-yadav11/quillrag/issues/1).

## Questions

Open a discussion or comment on the relevant issue with your plan before large
refactors — especially anything touching the storage format (schema migrations
are the one place where being clever hurts).
