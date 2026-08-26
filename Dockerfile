# quillrag — single-binary local RAG MCP server.
#
# Multi-stage build: the release binary is compiled with the model assets
# baked in (assets/ are include_bytes!'d at build time), so the runtime
# stage needs nothing but the binary itself.
#
# Both stages pin Debian bookworm so the glibc versions match (a trixie
# builder produces a binary requiring GLIBC_2.39 that bookworm lacks).

FROM rust:1-slim-bookworm AS builder
WORKDIR /build

# esaxx-rs (via tokenizers) compiles C++; slim images lack g++.
RUN apt-get update && apt-get install -y --no-install-recommends g++ && rm -rf /var/lib/apt/lists/*

# Deps first for layer caching: manifest + lockfile, then sources.
COPY Cargo.toml Cargo.lock ./
RUN mkdir src && echo "fn main() {}" > src/main.rs && cargo build --release && rm -rf src

COPY . .
RUN touch src/main.rs && cargo build --release --bin quillrag

FROM debian:bookworm-slim
LABEL org.opencontainers.image.title="quillrag"
LABEL org.opencontainers.image.description="Single-binary local RAG MCP server in Rust. MiniLM compiled in, hybrid dense + BM25 search, zero runtime downloads."
LABEL org.opencontainers.image.source="https://github.com/Ayush-yadav11/quillrag"
LABEL org.opencontainers.image.licenses="MIT"

COPY --from=builder /build/target/release/quillrag /usr/local/bin/quillrag

# Index data lives here; mount a volume to persist it across container runs.
ENV QUILLRAG_DATA=/data
VOLUME /data

# MCP stdio transport: one JSON-RPC message per line on stdin/stdout.
ENTRYPOINT ["/usr/local/bin/quillrag", "serve"]
