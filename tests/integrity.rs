use quillrag::{Store, TantivyIndex};
use std::collections::HashMap;

fn temp(label: &str) -> std::path::PathBuf {
    let p = std::env::temp_dir().join(format!("qr-integrity-{label}-{}", std::process::id()));
    std::fs::create_dir_all(&p).unwrap();
    p
}

#[test]
fn batch_keeps_text_vectors_and_paths_aligned() {
    let dir = temp("alignment");
    let store = Store::open(&dir).unwrap();
    let mut docs = HashMap::new();
    for i in 0..4 {
        docs.insert(format!("doc{i}.md"), (0, 0, 0, vec![]));
    }
    let order: Vec<_> = docs.keys().cloned().collect();
    let chunks: Vec<_> = (0..4).map(|i| format!("marker{i}")).collect();
    let vectors: Vec<_> = (0..4)
        .map(|i| {
            let mut v = vec![0.0; 4];
            v[i] = 1.0;
            v
        })
        .collect();
    for (i, path) in order.iter().enumerate() {
        docs.get_mut(path).unwrap().3 = vec![(3 - i) as u64];
    }
    store.upsert_batch(&docs, &chunks, &vectors).unwrap();
    for (i, path) in order.iter().enumerate() {
        let key = 3 - i;
        let row = store.get_chunk_by_ordinal(path, 0).unwrap().unwrap();
        assert_eq!(row.2, chunks[key], "wrong content for {path}");
        let ranks = store.dense_scan(&vectors[key]).unwrap();
        assert_eq!(ranks[0].0, key as u64, "wrong vector for {path}");
    }
    drop(store);
    std::fs::remove_dir_all(dir).unwrap();
}

#[test]
fn bm25_returns_hydratable_paths_after_reopen() {
    let dir = temp("bm25");
    let store = Store::open(&dir).unwrap();
    let mut docs = HashMap::new();
    docs.insert("/notes/attention.md".to_string(), (0, 0, 0, vec![0]));
    store
        .upsert_batch(&docs, &["attention transformer".into()], &[vec![1.0; 384]])
        .unwrap();
    {
        let bm25 = TantivyIndex::open(&dir).unwrap();
        bm25.rebuild_from(&store).unwrap();
    }
    let bm25 = TantivyIndex::open(&dir).unwrap();
    let hits = bm25.search_bm25("attention", 1).unwrap();
    assert_eq!(hits, vec![("/notes/attention.md".into(), 0)]);
    drop(bm25);
    drop(store);
    std::fs::remove_dir_all(dir).unwrap();
}
