//! PDF text extraction (feature `pdf`): one [`Section`] per page, read from
//! the PDF's text layer. With the `ocr` feature and [`PdfExtractor::with_ocr`],
//! pages without a text layer (scans) are rendered and OCR'd instead.

use super::{Extractor, NoText, Section};
use anyhow::{bail, Context, Result};
use pdf_extract::{Document, PlainTextOutput};
use std::panic::{catch_unwind, AssertUnwindSafe};
use std::path::Path;

#[cfg(feature = "ocr")]
use super::ocr::{has_text, Ocr};
#[cfg(feature = "ocr")]
use std::sync::Arc;

/// Extracts PDF text page by page with `pdf-extract` (pure Rust, offline).
#[derive(Clone)]
pub struct PdfExtractor {
    /// PDFs above this size are skipped.
    pub max_file_bytes: u64,
    /// At most this many pages of one PDF are OCR'd (~2-4 s each on CPU),
    /// so a 500-page scanned book can't stall a whole index pass.
    #[cfg(feature = "ocr")]
    pub max_ocr_pages: usize,
    #[cfg(feature = "ocr")]
    ocr: Option<Arc<Ocr>>,
}

impl Default for PdfExtractor {
    fn default() -> Self {
        Self {
            max_file_bytes: 128 * 1024 * 1024,
            #[cfg(feature = "ocr")]
            max_ocr_pages: 100,
            #[cfg(feature = "ocr")]
            ocr: None,
        }
    }
}

#[cfg(feature = "ocr")]
impl PdfExtractor {
    /// OCR pages that have no text layer.
    pub fn with_ocr(mut self, ocr: Arc<Ocr>) -> Self {
        self.ocr = Some(ocr);
        self
    }
}

impl Extractor for PdfExtractor {
    fn extensions(&self) -> &[&str] {
        &["pdf"]
    }

    fn max_file_bytes(&self) -> u64 {
        self.max_file_bytes
    }

    fn extract(&self, path: &Path) -> Result<Vec<Section>> {
        let bytes = std::fs::read(path).with_context(|| format!("reading {}", path.display()))?;
        // pdf-extract panics on some malformed fonts and streams. Contain it
        // to this file (requires `panic = "unwind"` in the final binary).
        match catch_unwind(AssertUnwindSafe(|| self.extract_pages(bytes))) {
            Ok(result) => result,
            Err(_) => bail!("PDF parser failed on this file"),
        }
    }
}

impl PdfExtractor {
    fn extract_pages(&self, bytes: Vec<u8>) -> Result<Vec<Section>> {
        let mut doc = Document::load_mem(&bytes).context("not a readable PDF")?;
        if doc.is_encrypted() && doc.decrypt("").is_err() {
            bail!("password-protected PDF");
        }

        let pages: Vec<u32> = doc.get_pages().into_keys().collect();
        let mut sections = Vec::new();
        // Pages with no extractable text: scans, or text-layer failures.
        let mut textless = Vec::new();
        let mut unreadable = 0;
        for &page in &pages {
            let mut raw = String::new();
            let result = {
                let mut out = PlainTextOutput::new(&mut raw);
                pdf_extract::output_doc_page(&doc, &mut out, page)
            };
            if let Err(e) = result {
                // One broken page shouldn't cost the rest of the document.
                unreadable += 1;
                tracing::debug!(page, "unreadable PDF text layer: {e:?}");
                textless.push(page);
                continue;
            }
            let text = clean(&raw);
            if text.is_empty() {
                textless.push(page);
            } else {
                sections.push(Section {
                    text,
                    page: Some(page),
                });
            }
        }
        drop(doc);

        #[cfg(feature = "ocr")]
        if let Some(ocr) = &self.ocr {
            if !textless.is_empty() {
                sections.extend(self.ocr_pages(ocr, bytes, &textless)?);
                sections.sort_by_key(|s| s.page);
            }
        }

        if sections.is_empty() {
            if !pages.is_empty() && unreadable == pages.len() {
                bail!("could not read any page");
            }
            #[cfg(feature = "ocr")]
            if self.ocr.is_some() {
                return Err(NoText("no text found (blank or unreadable scan)").into());
            }
            return Err(NoText("no text layer (scanned PDF?)").into());
        }
        Ok(sections)
    }

    /// Render `pages` (1-based) with hayro and OCR them.
    #[cfg(feature = "ocr")]
    fn ocr_pages(&self, ocr: &Ocr, bytes: Vec<u8>, pages: &[u32]) -> Result<Vec<Section>> {
        use hayro::hayro_interpret::InterpreterSettings;
        use hayro::hayro_syntax::Pdf;
        use hayro::vello_cpu::color::palette::css::WHITE;
        use hayro::{render, RenderCache, RenderSettings};

        // ~200 DPI: small print stays legible without huge bitmaps.
        const SCALE: f32 = 200.0 / 72.0;
        const MAX_SIDE: f32 = 3000.0;

        if pages.len() > self.max_ocr_pages {
            tracing::warn!(
                pages = pages.len(),
                limit = self.max_ocr_pages,
                "scanned PDF has more pages than the OCR limit; indexing the first ones"
            );
        }
        let pdf = Pdf::new(bytes).map_err(|e| anyhow::anyhow!("rendering PDF: {e:?}"))?;
        let cache = RenderCache::new();
        let settings = InterpreterSettings::default();
        let mut out = Vec::new();
        for &number in pages.iter().take(self.max_ocr_pages) {
            let Some(page) = pdf.pages().get(number as usize - 1) else {
                continue;
            };
            let (w, h) = page.render_dimensions();
            let scale = SCALE.min(MAX_SIDE / w.max(h).max(1.0));
            let pixmap = render(
                page,
                &cache,
                &settings,
                &RenderSettings {
                    x_scale: scale,
                    y_scale: scale,
                    bg_color: WHITE,
                    ..Default::default()
                },
            );
            let (pw, ph) = (pixmap.width() as u32, pixmap.height() as u32);
            if pw == 0 || ph == 0 {
                continue;
            }
            let text = ocr.read_pixels(pixmap.data_as_u8_slice(), pw, ph)?;
            if has_text(&text) {
                out.push(Section {
                    text,
                    page: Some(number),
                });
            }
        }
        Ok(out)
    }
}

/// Undo common text-layer artifacts: ligature glyphs, words hyphenated
/// across line breaks, and runs of spaces from glyph positioning. Blank lines
/// survive as paragraph breaks for the chunker.
fn clean(raw: &str) -> String {
    let text = raw
        .replace('\u{fb00}', "ff")
        .replace('\u{fb01}', "fi")
        .replace('\u{fb02}', "fl")
        .replace('\u{fb03}', "ffi")
        .replace('\u{fb04}', "ffl");

    let mut out = String::with_capacity(text.len());
    let mut paragraph_break = false;
    for line in text.lines() {
        let line = line.split_whitespace().collect::<Vec<_>>().join(" ");
        if line.is_empty() {
            paragraph_break = !out.is_empty();
            continue;
        }
        if paragraph_break {
            out.push_str("\n\n");
            paragraph_break = false;
        } else if out.ends_with('-') && line.starts_with(char::is_lowercase) {
            out.pop(); // "migra-" + "tion" -> "migration"
        } else if !out.is_empty() {
            out.push('\n');
        }
        out.push_str(&line);
    }
    out
}

#[cfg(test)]
mod tests {
    use super::clean;

    #[test]
    fn clean_joins_hyphenation_and_keeps_paragraphs() {
        let raw = "  The data-\nbase   migra-\ntion plan\n\n\n\u{fb01}nal  step\nNext-\nYear";
        // Lowercase continuations are joined; "Next-" + "Year" is not.
        assert_eq!(
            clean(raw),
            "The database migration plan\n\nfinal step\nNext-\nYear"
        );
    }

    #[test]
    fn clean_of_whitespace_is_empty() {
        assert_eq!(clean(" \n \n\t"), "");
    }
}
