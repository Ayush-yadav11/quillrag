//! Pluggable text extraction.
//!
//! The indexer reads plain-text formats itself. Embedders that want to index
//! binary formats (PDFs, images via OCR, office docs) implement [`Extractor`]
//! and pass it through [`crate::indexer::IndexHooks`]; the indexer then
//! discovers files with those extensions and routes them to the extractor
//! instead of the UTF-8 reader.

use anyhow::{Context, Result};
use std::path::Path;

#[cfg(feature = "ocr")]
mod ocr;
#[cfg(feature = "pdf")]
mod pdf;
#[cfg(feature = "ocr")]
pub use ocr::{ImageExtractor, Ocr};
#[cfg(feature = "pdf")]
pub use pdf::PdfExtractor;

/// Files above this size are skipped during discovery unless their
/// extractor raises the limit.
pub const DEFAULT_MAX_FILE_BYTES: u64 = 8 * 1024 * 1024;

/// A run of extracted text, optionally tied to a page of the source.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Section {
    pub text: String,
    /// 1-based page number, for paged formats like PDF.
    pub page: Option<u32>,
}

impl Section {
    /// Unpaged text.
    pub fn text(text: impl Into<String>) -> Self {
        Self {
            text: text.into(),
            page: None,
        }
    }
}

/// Error for a file that was read fine but holds no indexable text (a photo
/// without words, a blank scan). The indexer remembers such files as empty
/// documents so unchanged ones aren't re-extracted on every pass; see
/// [`crate::IndexOptions::retry_empty`].
#[derive(Debug, Clone, Copy)]
pub struct NoText(pub &'static str);

impl std::fmt::Display for NoText {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.0)
    }
}

impl std::error::Error for NoText {}

/// Turns a file into indexable text.
pub trait Extractor: Send + Sync {
    /// Lowercase extensions (no leading dot) this extractor handles.
    fn extensions(&self) -> &[&str];

    /// Files larger than this are skipped during discovery.
    fn max_file_bytes(&self) -> u64 {
        DEFAULT_MAX_FILE_BYTES
    }

    /// Extract the file's text, e.g. one section per PDF page. Chunks never
    /// span sections, so each search hit maps to one page. Return
    /// [`NoText`] (or no non-blank sections) for files without text.
    fn extract(&self, path: &Path) -> Result<Vec<Section>>;
}

/// Pick the extractor registered for `ext`, if any. Later entries win so a
/// caller can override an earlier registration.
pub fn find<'a>(extractors: &[&'a dyn Extractor], ext: &str) -> Option<&'a dyn Extractor> {
    extractors
        .iter()
        .rev()
        .find(|e| e.extensions().iter().any(|x| x.eq_ignore_ascii_case(ext)))
        .copied()
}

/// Read a file as text: UTF-8 (BOM stripped), falling back to lossy decoding.
pub fn read_text(path: &Path) -> Result<String> {
    let bytes = std::fs::read(path).with_context(|| format!("reading {}", path.display()))?;
    if bytes.starts_with(b"\xef\xbb\xbf") {
        // Strip BOM and try UTF-8.
        let sans_bom = &bytes[3..];
        return String::from_utf8(sans_bom.to_vec())
            .with_context(|| format!("decoding {} as utf-8", path.display()));
    }
    match String::from_utf8(bytes) {
        Ok(s) => Ok(s),
        // Fall back to lossy: replace invalid UTF-8 with U+FFFD.
        Err(e) => Ok(String::from_utf8_lossy(e.as_bytes()).into_owned()),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    struct Fake(&'static [&'static str], &'static str);
    impl Extractor for Fake {
        fn extensions(&self) -> &[&str] {
            self.0
        }
        fn extract(&self, _: &Path) -> Result<Vec<Section>> {
            Ok(vec![Section::text(self.1)])
        }
    }

    #[test]
    fn find_is_case_insensitive_and_last_wins() {
        let a = Fake(&["pdf"], "a");
        let b = Fake(&["PDF", "png"], "b");
        let list: Vec<&dyn Extractor> = vec![&a, &b];
        let got = find(&list, "pdf").unwrap();
        assert_eq!(got.extract(Path::new("x")).unwrap()[0].text, "b");
        assert!(find(&list, "docx").is_none());
    }
}
