//! End-to-end tests: exercise the real binary over real stdio, plus a full
//! index->search pipeline through the actual MiniLM weights.
//!
//! Run with `cargo test --release` — debug-mode candle inference is orders
//! of magnitude slower.

use serde_json::Value;
use std::io::{BufRead, BufReader, Write};
use std::process::{Child, Command, Stdio};
use std::sync::mpsc::{channel, Receiver};
use std::time::Duration;

struct TestServer {
    child: Child,
    stdin: std::process::ChildStdin,
    rx: Receiver<String>,
}

impl TestServer {
    fn send_raw(&mut self, msg: &Value) {
        writeln!(self.stdin, "{msg}").expect("write rpc");
        self.stdin.flush().expect("flush");
    }

    /// Send a request and wait for one response line with a hard timeout.
    fn rpc(&mut self, msg: Value) -> Value {
        self.send_raw(&msg);
        let line = self
            .rx
            .recv_timeout(Duration::from_secs(90))
            .expect("timed out waiting for MCP response");
        serde_json::from_str(&line).expect("valid JSON-RPC line")
    }
}

impl Drop for TestServer {
    fn drop(&mut self) {
        let _ = self.child.kill();
        let _ = self.child.wait();
    }
}

/// Spawn the pocketrag binary; a background thread pumps stdout lines into a
/// channel so a hung server can never hang the test forever.
fn spawn_server(args: &[&str]) -> TestServer {
    let mut child = Command::new(env!("CARGO_BIN_EXE_pocketrag"))
        .args(args)
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::null())
        .spawn()
        .expect("spawn pocketrag");
    let stdin = child.stdin.take().expect("stdin");
    let stdout = child.stdout.take().expect("stdout");

    let (tx, rx) = channel();
    std::thread::spawn(move || {
        let reader = BufReader::new(stdout);
        for line in reader.lines() {
            match line {
                Ok(l) => {
                    if tx.send(l).is_err() {
                        break;
                    }
                }
                Err(_) => break,
            }
        }
    });

    TestServer { child, stdin, rx }
}

#[test]
fn mcp_stdio_handshake_and_tools() {
    let data_dir = tempfile_dir("mcp-handshake");
    let mut srv = spawn_server(&["serve", "--data-dir", data_dir.to_str().unwrap()]);

    // 1. initialize
    let init = srv.rpc(serde_json::json!({
        "jsonrpc": "2.0",
        "id": 1,
        "method": "initialize",
        "params": {
            "protocolVersion": "2024-11-05",
            "capabilities": {},
            "clientInfo": { "name": "pocketrag-test", "version": "0.0.1" }
        }
    }));
    assert_eq!(init["id"], 1);
    assert_eq!(init["result"]["serverInfo"]["name"], "pocketrag");

    // 2. initialized notification (no response expected)
    srv.send_raw(&serde_json::json!({
        "jsonrpc": "2.0",
        "method": "notifications/initialized"
    }));

    // 3. tools/list — all four registered
    let tools = srv.rpc(serde_json::json!({
        "jsonrpc": "2.0", "id": 2, "method": "tools/list"
    }));
    let names: Vec<String> = tools["result"]["tools"]
        .as_array()
        .expect("tools array")
        .iter()
        .map(|t| t["name"].as_str().expect("tool name").to_string())
        .collect();
    for expected in ["rag_search", "rag_index", "rag_status", "rag_clear"] {
        assert!(
            names.iter().any(|n| n == expected),
            "missing tool {expected} in {names:?}"
        );
    }

    // 4. tools/call rag_status — must respond before any model loading
    let status = srv.rpc(serde_json::json!({
        "jsonrpc": "2.0",
        "id": 3,
        "method": "tools/call",
        "params": { "name": "rag_status", "arguments": {} }
    }));
    assert!(status["result"]["content"][0]["text"].is_string());

    cleanup(&data_dir);
}

#[test]
fn full_pipeline_index_then_search() {
    let data_dir = tempfile_dir("pipeline");
    let docs_dir = tempfile_dir("corpus");
    std::fs::write(
        docs_dir.join("rust_errors.md"),
        "# Common Rust Errors\n\nThe error E0382 means use of moved value. \
         Ownership was transferred to another binding, so the original \
         cannot be used again.\n\nBorrow checker violations happen when a \
         reference outlives its owner.\n",
    )
    .unwrap();
    std::fs::write(
        docs_dir.join("deploy.txt"),
        "Deployment runbook: always bump the version field before tagging. \
         The staging cluster accepts builds signed with the team key.\n",
    )
    .unwrap();

    // index subcommand
    let out = run(&[
        "index",
        docs_dir.to_str().unwrap(),
        "--data-dir",
        data_dir.to_str().unwrap(),
    ]);
    assert!(out.status, "index failed: {}", out.stderr);
    assert!(
        out.stdout.contains("indexed 2"),
        "unexpected summary: {}",
        out.stdout
    );

    // search subcommand — semantic query with no keyword overlap
    let out = run(&[
        "search",
        "why does the compiler say my variable was moved",
        "--top-k",
        "2",
        "--data-dir",
        data_dir.to_str().unwrap(),
    ]);
    assert!(out.status, "search failed: {}", out.stderr);
    assert!(
        out.stdout.contains("rust_errors.md"),
        "semantic search missed the ownership doc:\n{}",
        out.stdout
    );

    // incremental re-index: unchanged -> skipped
    let out = run(&[
        "index",
        docs_dir.to_str().unwrap(),
        "--data-dir",
        data_dir.to_str().unwrap(),
    ]);
    assert!(
        out.stdout.contains("unchanged 2"),
        "incremental skip failed: {}",
        out.stdout
    );

    // delete a file -> pruned on next pass
    std::fs::remove_file(docs_dir.join("deploy.txt")).unwrap();
    let out = run(&[
        "index",
        docs_dir.to_str().unwrap(),
        "--data-dir",
        data_dir.to_str().unwrap(),
    ]);
    assert!(
        out.stdout.contains("pruned 1"),
        "prune failed: {}",
        out.stdout
    );

    cleanup(&docs_dir);
    cleanup(&data_dir);
}

// ---------- helpers ----------

struct Out {
    status: bool,
    stdout: String,
    stderr: String,
}

fn run(args: &[&str]) -> Out {
    let out = Command::new(env!("CARGO_BIN_EXE_pocketrag"))
        .args(args)
        .output()
        .expect("run subcommand");
    Out {
        status: out.status.success(),
        stdout: String::from_utf8_lossy(&out.stdout).into_owned(),
        stderr: String::from_utf8_lossy(&out.stderr).into_owned(),
    }
}

fn tempfile_dir(label: &str) -> std::path::PathBuf {
    let dir = std::env::temp_dir().join(format!(
        "pocketrag-test-{label}-{}-{}",
        std::process::id(),
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .subsec_nanos()
    ));
    std::fs::create_dir_all(&dir).unwrap();
    dir
}

fn cleanup(dir: &std::path::Path) {
    let _ = std::fs::remove_dir_all(dir);
}
