use quillrag::{indexer::discover_files, Store};
use std::{collections::HashMap, path::PathBuf, process::Command};

struct Fixture(PathBuf);
impl Fixture {
    fn new(label: &str) -> Self {
        let p = std::env::temp_dir().join(format!("qr-safety-{label}-{}", std::process::id()));
        std::fs::create_dir_all(p.join("docs")).unwrap();
        Self(p)
    }
    fn seed(&self, path: &str) {
        let store = Store::open(&self.0.join("data")).unwrap();
        let mut docs = HashMap::new();
        docs.insert(
            path.to_string(),
            (0, 0, 0, vec![store.next_chunk_key().unwrap()]),
        );
        store
            .upsert_batch(&docs, &["retained source".into()], &[vec![1.0; 384]])
            .unwrap();
    }
    fn count(&self) -> u64 {
        Store::open(&self.0.join("data"))
            .unwrap()
            .stats()
            .unwrap()
            .documents
    }
    fn index(&self, path: &std::path::Path, flags: &[&str]) -> std::process::Output {
        Command::new(env!("CARGO_BIN_EXE_quillrag"))
            .arg("index")
            .arg(path)
            .arg("--data-dir")
            .arg(self.0.join("data"))
            .args(flags)
            .output()
            .unwrap()
    }
}
impl Drop for Fixture {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.0);
    }
}

#[test]
fn no_prune_preserves_missing_document() {
    let f = Fixture::new("no-prune");
    f.seed(f.0.join("docs/missing.md").to_str().unwrap());
    let out = f.index(&f.0.join("docs"), &["--no-prune"]);
    assert!(
        out.status.success(),
        "{}",
        String::from_utf8_lossy(&out.stderr)
    );
    assert_eq!(f.count(), 1);
}

#[test]
fn force_invalid_path_preserves_store() {
    let f = Fixture::new("force");
    f.seed("/another-root/keep.md");
    assert!(!f.index(&f.0.join("absent"), &["--force"]).status.success());
    assert_eq!(f.count(), 1);
}

#[test]
fn empty_successful_walk_prunes_only_this_root() {
    let f = Fixture::new("empty");
    f.seed(f.0.join("docs/deleted.md").to_str().unwrap());
    f.seed(f.0.join("docs-other/keep.md").to_str().unwrap());
    let out = f.index(&f.0.join("docs"), &[]);
    assert!(
        out.status.success(),
        "{}",
        String::from_utf8_lossy(&out.stderr)
    );
    assert_eq!(f.count(), 1);
}

#[cfg(unix)]
#[test]
fn incomplete_walk_fails_without_pruning() {
    let f = Fixture::new("broken");
    f.seed(f.0.join("docs/missing.md").to_str().unwrap());
    std::os::unix::fs::symlink(f.0.join("absent"), f.0.join("docs/broken")).unwrap();
    assert!(discover_files(&f.0.join("docs"), &[]).is_err());
    assert!(!f.index(&f.0.join("docs"), &["--force"]).status.success());
    assert_eq!(f.count(), 1);
}

#[cfg(unix)]
#[test]
fn symlink_cycle_fails_without_pruning() {
    let f = Fixture::new("cycle");
    f.seed(f.0.join("docs/missing.md").to_str().unwrap());
    std::os::unix::fs::symlink(f.0.join("docs"), f.0.join("docs/loop")).unwrap();
    assert!(!f.index(&f.0.join("docs"), &[]).status.success());
    assert_eq!(f.count(), 1);
}

#[test]
fn excluded_existing_files_are_not_deleted() {
    let f = Fixture::new("excluded");
    let path = f.0.join("docs/existing.custom");
    std::fs::write(&path, "keep this previously indexed source").unwrap();
    f.seed(path.to_str().unwrap());
    assert!(f.index(&f.0.join("docs"), &[]).status.success());
    assert_eq!(f.count(), 1);
}

#[test]
fn explicitly_selected_hidden_root_is_walked() {
    let f = Fixture::new("hidden");
    let root = f.0.join(".notes");
    std::fs::create_dir(&root).unwrap();
    std::fs::write(root.join("note.md"), "text").unwrap();
    assert_eq!(discover_files(&root, &[]).unwrap().len(), 1);
}
