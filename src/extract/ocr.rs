//! OCR (feature `ocr`): text from screenshots, photos and scanned PDF pages
//! via the ocrs engine, with its detection and recognition models compiled
//! into the binary. The recognition model reads English / basic Latin text.
//!
//! Models: <https://huggingface.co/robertknight/ocrs>, by Robert Knight,
//! licensed CC BY-SA 4.0 (trained on HierText and synthetic data).

use super::{Extractor, NoText, Section};
use anyhow::{Context, Result};
use image::imageops::FilterType;
use image::DynamicImage;
use ocrs::{ImageSource, OcrEngine, OcrEngineParams};
use rten::Model;
use std::path::Path;
use std::sync::Arc;

static DETECTION_MODEL: &[u8] = include_bytes!("../../assets/ocr/text-detection.rten");
static RECOGNITION_MODEL: &[u8] = include_bytes!("../../assets/ocr/text-recognition.rten");

/// Longest image side fed to OCR. Detection time grows with pixel count,
/// and text in a 4K screenshot is still legible at this size.
pub const MAX_OCR_SIDE: u32 = 2560;

/// A loaded OCR engine. Loading parses ~12 MB of model data, so build one
/// and share it (`Arc<Ocr>`) between extractors.
pub struct Ocr {
    engine: OcrEngine,
}

impl Ocr {
    /// Load the models compiled into this binary.
    pub fn bundled() -> Result<Self> {
        let detection_model =
            Model::load_static_slice(DETECTION_MODEL).context("loading OCR detection model")?;
        let recognition_model =
            Model::load_static_slice(RECOGNITION_MODEL).context("loading OCR recognition model")?;
        let engine = OcrEngine::new(OcrEngineParams {
            detection_model: Some(detection_model),
            recognition_model: Some(recognition_model),
            ..Default::default()
        })?;
        Ok(Self { engine })
    }

    /// Recognize text in tightly packed RGB or RGBA pixels.
    pub fn read_pixels(&self, pixels: &[u8], width: u32, height: u32) -> Result<String> {
        let source = ImageSource::from_bytes(pixels, (width, height))?;
        let input = self.engine.prepare_input(source)?;
        Ok(tidy(&self.engine.get_text(&input)?))
    }

    /// Recognize text in an image, downscaling very large ones first.
    pub fn read_image(&self, image: DynamicImage) -> Result<String> {
        let image = if image.width().max(image.height()) > MAX_OCR_SIDE {
            image.resize(MAX_OCR_SIDE, MAX_OCR_SIDE, FilterType::Triangle)
        } else {
            image
        };
        let rgb = image.into_rgb8();
        self.read_pixels(rgb.as_raw(), rgb.width(), rgb.height())
    }
}

/// Whether OCR output is worth indexing. Photos without text often produce a
/// few stray characters from textures and edges.
pub(crate) fn has_text(text: &str) -> bool {
    text.split_whitespace()
        .filter(|w| w.chars().filter(|c| c.is_alphanumeric()).count() >= 2)
        .count()
        >= 2
}

/// One line per recognized text line, whitespace collapsed, blanks dropped.
fn tidy(raw: &str) -> String {
    raw.lines()
        .map(|l| l.split_whitespace().collect::<Vec<_>>().join(" "))
        .filter(|l| !l.is_empty())
        .collect::<Vec<_>>()
        .join("\n")
}

/// OCRs screenshots and photos.
pub struct ImageExtractor {
    ocr: Arc<Ocr>,
}

impl ImageExtractor {
    pub fn new(ocr: Arc<Ocr>) -> Self {
        Self { ocr }
    }
}

impl Extractor for ImageExtractor {
    fn extensions(&self) -> &[&str] {
        &["png", "jpg", "jpeg", "webp", "bmp", "gif"]
    }

    fn max_file_bytes(&self) -> u64 {
        40 * 1024 * 1024
    }

    fn extract(&self, path: &Path) -> Result<Vec<Section>> {
        let image = image::open(path).context("not a readable image")?;
        let text = self.ocr.read_image(image)?;
        if !has_text(&text) {
            return Err(NoText("no text found in image").into());
        }
        Ok(vec![Section::text(text)])
    }
}

#[cfg(test)]
mod tests {
    use super::{has_text, tidy};

    #[test]
    fn tidy_collapses_and_drops_blank_lines() {
        assert_eq!(
            tidy("  Invoice   4821 \n\n  due  soon\n"),
            "Invoice 4821\ndue soon"
        );
    }

    #[test]
    fn stray_marks_are_not_text() {
        assert!(!has_text(""));
        assert!(!has_text("| . ~ i"));
        assert!(!has_text("Hello"));
        assert!(has_text("Invoice 4821"));
    }
}
