//! Library-level indexing with extractor and progress hooks — the path a
//! desktop app takes when it embeds the engine with `default-features = false`.

use anyhow::Result;
use quillrag::indexer::index_directory_with;
use quillrag::{
    Embedder, Extractor, IndexHooks, IndexOptions, IndexProgress, Section, Store, TantivyIndex,
};
use std::path::Path;
use std::sync::Mutex;

fn temp(label: &str) -> std::path::PathBuf {
    let p = std::env::temp_dir().join(format!("qr-hooks-{label}-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&p);
    std::fs::create_dir_all(&p).unwrap();
    p
}

/// Pretends `.fake` files are binary and decodes them by reversing the bytes.
struct Reverse;
impl Extractor for Reverse {
    fn extensions(&self) -> &[&str] {
        &["fake"]
    }
    fn extract(&self, path: &Path) -> Result<Vec<Section>> {
        let text: String = std::fs::read_to_string(path)?.chars().rev().collect();
        Ok(vec![Section::text(text)])
    }
}

#[test]
fn extractor_files_are_discovered_extracted_and_searchable() {
    let root = temp("root");
    let docs = root.join("docs");
    let data = root.join("data");
    std::fs::create_dir_all(&docs).unwrap();
    std::fs::write(
        docs.join("plain.md"),
        "Plain markdown about sourdough bread.",
    )
    .unwrap();
    let hidden: String = "Quarterly zyxglorp invoice from the accountant."
        .chars()
        .rev()
        .collect();
    std::fs::write(docs.join("scan.fake"), hidden).unwrap();

    let store = Store::open(&data).unwrap();
    let bm25 = TantivyIndex::open(&data).unwrap();
    let mut embedder = Embedder::load(&data.join("model")).unwrap();

    let events = Mutex::new(Vec::new());
    let on_progress = |e: IndexProgress| events.lock().unwrap().push(e);
    let reverse = Reverse;
    let extractors: [&dyn Extractor; 1] = [&reverse];
    let hooks = IndexHooks {
        extractors: &extractors,
        progress: Some(&on_progress),
    };

    let report = index_directory_with(
        &docs,
        &[],
        &store,
        &bm25,
        &mut embedder,
        IndexOptions::default(),
        &hooks,
    )
    .unwrap();
    assert_eq!(report.indexed.len(), 2, "{:?}", report.failed);

    // The extractor's output (not the raw bytes) is what got indexed.
    let hits = bm25.search_bm25("zyxglorp", 3).unwrap();
    assert_eq!(hits.len(), 1);
    assert!(hits[0].0.ends_with("scan.fake"));

    let events = events.into_inner().unwrap();
    assert_eq!(
        events.first(),
        Some(&IndexProgress::Discovered { files: 2 })
    );
    assert!(events.contains(&IndexProgress::Scanned { done: 2, total: 2 }));
    assert!(events
        .iter()
        .any(|e| matches!(e, IndexProgress::Embedding { done, total } if done == total)));
    assert_eq!(events.last(), Some(&IndexProgress::Finished));

    // Without the hook, `.fake` is not a known extension and is left alone.
    let plain = index_directory_with(
        &docs,
        &[],
        &store,
        &bm25,
        &mut embedder,
        IndexOptions {
            no_prune: true,
            force: true,
            ..Default::default()
        },
        &IndexHooks::default(),
    )
    .unwrap();
    assert_eq!(plain.indexed.len(), 1);

    drop((store, bm25, embedder));
    std::fs::remove_dir_all(root).unwrap();
}
