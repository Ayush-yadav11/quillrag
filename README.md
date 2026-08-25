# quillrag

**A single-binary local RAG MCP server in Rust.** Starts in milliseconds. Ships
as one file for macOS / Linux / Windows. No Node, no Python, no model-download
dance at first query — the MiniLM embedding model (~90 MB) and its tokenizer
are compiled into the binary.

```
$ ./quillrag serve --data-dir ~/.local/share/quillrag
2026-08-25 INFO quillrag 0.1.0 ready in 41ms      <- handshake-ready before the model loads
```

## Why it's fast

| Stage | Cost |
|---|---|
| Binary start + MCP initialize | **~20 ms** (measured: store open + tool registration only) |
| First `rag_search` / `rag_index` call | +~300 ms one-time (mmap safetensors, build BERT graph) |
| Subsequent searches | **~25 ms** per query (2-core CPU, small corpus) |
| Re-indexing unchanged corpus | near-zero (FNV content hash skip) |

The embedding model is *lazy*: the MCP handshake and `rag_status` never touch
it, so editors see an instant server.

## Install

```sh
# from a release (recommended): grab the archive for your OS, done.
cargo install --path .          # or build from source
```

Cross-compile targets used by CI: `x86_64-unknown-linux-gnu`,
`aarch64-apple-darwin`, `x86_64-pc-windows-msvc`.

## Wire it into your editor

Claude Desktop / Cursor / any MCP client:

```json
{
  "mcpServers": {
    "quillrag": {
      "command": "/usr/local/bin/quillrag",
      "args": ["serve"],
      "env": { "QUILLRAG_DATA": "~/.local/share/quillrag" }
    }
  }
}
```

Or just run `./quillrag serve` and point any stdio client at it.

## Tools

| Tool | What it does |
|---|---|
| `rag_index`  | Incrementally index a directory/file. Skips unchanged files, prunes deleted ones, re-embeds only diffs. |
| `rag_search` | Hybrid retrieval: dense MiniLM cosine + BM25 keyword, fused with Reciprocal Rank Fusion. Returns ranked chunks with source paths. |
| `rag_status` | Document/chunk counts, bytes indexed, file-type breakdown. |
| `rag_clear`  | Wipe everything. |

CLI equivalents (same engine):

```sh
quillrag index ~/notes              # incremental walk
quillrag search "auth flow" -k 5    # one-shot search
quillrag status                     # stats
quillrag clear                      # wipe
```

## Design

- **Embeddings**: [candle](https://github.com/huggingface/candle) (pure Rust)
  running `sentence-transformers/all-MiniLM-L6-v2` — masked mean pooling +
  L2 norm, numerically matching sentence-transformers on CPU. Weights are
  `include_bytes!`-ed into the binary and mmap'd from a materialized cache on
  first load.
- **Storage**: single [redb](https://github.com/cberner/redb) file — chunk text,
  raw f32 vectors, document metadata. Atomic commits; crash-safe.
- **Keywords**: [tantivy](https://github.com/quickwit-inc/tantivy) BM25 sidecar
  index rebuilt per indexing pass (cheap at pocket scale).
- **Fusion**: Reciprocal Rank Fusion (`Σ 1/(60+rank)`) — no score-scale tuning,
  robust to heterogeneous rankings.
- **Chunking**: paragraph-first with 1000-char cap and 120-char overlap;
  oversized paragraphs hard-split at sentence boundaries.

### File types indexed by default
`md markdown txt rst json yaml yml toml csv tsv html htm xml log rs py js jsx ts
tsx go c h cpp hpp java rb sh bash zsh sql proto graphql dockerfile makefile ini
cfg conf env` — extend with `-e ext1,ext2` / `"extensions": [...]`.

Ignored dirs: `.git node_modules target dist build .venv venv __pycache__ …`

## Privacy & footprint

Everything runs locally: embeddings, storage, search. Nothing leaves the
machine — there is no network code path at all after installation.

Binary ≈ 105 MB (the model lives inside). RAM ≈ 120 MB resident while idle,
spiking to ~250 MB during batch embedding.

## Development

```sh
cargo test                    # unit + end-to-end (spawns real stdio servers)
cargo run -- serve            # dev server
RUST_LOG=debug cargo run ...  # verbose logs (stderr only)
```

License: MIT
