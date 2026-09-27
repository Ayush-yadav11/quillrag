//! OCR end to end with the bundled models: text rendered to an image (a
//! stand-in screenshot), a scanned PDF (image only, no text layer), and
//! text-less files that must not be re-OCR'd on every pass.

use lopdf::content::{Content, Operation};
use lopdf::{dictionary, Dictionary, Document, Object, Stream};
use quillrag::indexer::index_directory_with;
use quillrag::{
    hybrid_search, Embedder, Extractor, ImageExtractor, IndexHooks, IndexOptions, NoText, Ocr,
    PdfExtractor, Store, TantivyIndex,
};
use std::sync::Arc;

const LINES: &[&str] = &[
    "Electricity bill September",
    "Account number 58213",
    "Amount due 2450 rupees",
];

fn temp(label: &str) -> std::path::PathBuf {
    let p = std::env::temp_dir().join(format!("qr-ocr-{label}-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&p);
    std::fs::create_dir_all(&p).unwrap();
    p
}

/// A one-page Letter-size (612x792 pt) PDF. `fill` adds the objects the
/// page needs (fonts, images) and returns its content and resources.
fn one_page_pdf(fill: impl FnOnce(&mut Document) -> (Vec<Operation>, Dictionary)) -> Vec<u8> {
    let mut doc = Document::with_version("1.5");
    let pages_id = doc.new_object_id();
    let (operations, resources) = fill(&mut doc);
    let resources_id = doc.add_object(resources);
    let content = Content { operations }.encode().unwrap();
    let content_id = doc.add_object(Stream::new(dictionary! {}, content));
    let page_id = doc.add_object(dictionary! {
        "Type" => "Page", "Parent" => pages_id, "Contents" => content_id,
    });
    doc.objects.insert(
        pages_id,
        Object::Dictionary(dictionary! {
            "Type" => "Pages", "Kids" => vec![page_id.into()], "Count" => 1,
            "Resources" => resources_id,
            "MediaBox" => vec![0.into(), 0.into(), 612.into(), 792.into()],
        }),
    );
    let catalog_id = doc.add_object(dictionary! { "Type" => "Catalog", "Pages" => pages_id });
    doc.trailer.set("Root", catalog_id);
    let mut out = Vec::new();
    doc.save_to(&mut out).unwrap();
    out
}

/// `LINES` as a text PDF, rasterized at 144 DPI: a stand-in screenshot.
fn text_image() -> image::RgbImage {
    let pdf = one_page_pdf(|doc| {
        let font = doc.add_object(dictionary! {
            "Type" => "Font", "Subtype" => "Type1", "BaseFont" => "Helvetica",
        });
        let mut ops = Vec::new();
        for (i, line) in LINES.iter().enumerate() {
            ops.push(Operation::new("BT", vec![]));
            ops.push(Operation::new("Tf", vec!["F1".into(), 28.into()]));
            ops.push(Operation::new(
                "Td",
                vec![60.into(), (700 - 50 * i as i64).into()],
            ));
            ops.push(Operation::new("Tj", vec![Object::string_literal(*line)]));
            ops.push(Operation::new("ET", vec![]));
        }
        (ops, dictionary! { "Font" => dictionary! { "F1" => font } })
    });

    let pdf = hayro::hayro_syntax::Pdf::new(pdf).unwrap();
    let pixmap = hayro::render(
        &pdf.pages()[0],
        &hayro::RenderCache::new(),
        &hayro::hayro_interpret::InterpreterSettings::default(),
        &hayro::RenderSettings {
            x_scale: 2.0,
            y_scale: 2.0,
            bg_color: hayro::vello_cpu::color::palette::css::WHITE,
            ..Default::default()
        },
    );
    let (w, h) = (pixmap.width() as u32, pixmap.height() as u32);
    let rgba = image::RgbaImage::from_raw(w, h, pixmap.data_as_u8_slice().to_vec()).unwrap();
    image::DynamicImage::ImageRgba8(rgba).to_rgb8()
}

/// A PDF whose only page is `img` as a picture: no text layer, like a scan.
fn scanned_pdf(img: &image::RgbImage) -> Vec<u8> {
    one_page_pdf(|doc| {
        let image = doc.add_object(Stream::new(
            dictionary! {
                "Type" => "XObject", "Subtype" => "Image",
                "Width" => img.width() as i64, "Height" => img.height() as i64,
                "ColorSpace" => "DeviceRGB", "BitsPerComponent" => 8,
            },
            img.as_raw().clone(),
        ));
        let ops = vec![
            Operation::new("q", vec![]),
            Operation::new(
                "cm",
                vec![
                    612.into(),
                    0.into(),
                    0.into(),
                    792.into(),
                    0.into(),
                    0.into(),
                ],
            ),
            Operation::new("Do", vec!["Im1".into()]),
            Operation::new("Q", vec![]),
        ];
        (
            ops,
            dictionary! { "XObject" => dictionary! { "Im1" => image } },
        )
    })
}

#[test]
fn reads_text_from_an_image() {
    let dir = temp("image");
    let png = dir.join("bill.png");
    text_image().save(&png).unwrap();

    let ocr = Arc::new(Ocr::bundled().unwrap());
    let sections = ImageExtractor::new(ocr).extract(&png).unwrap();
    let text = sections[0].text.to_lowercase();
    assert!(text.contains("electricity"), "{text}");
    assert!(text.contains("58213"), "{text}");
    std::fs::remove_dir_all(dir).unwrap();
}

#[test]
fn scanned_pdf_pages_are_ocrd_only_when_enabled() {
    let dir = temp("scan");
    let pdf = dir.join("scan.pdf");
    std::fs::write(&pdf, scanned_pdf(&text_image())).unwrap();

    let err = PdfExtractor::default().extract(&pdf).unwrap_err();
    assert!(err.is::<NoText>(), "{err:#}");

    let ocr = Arc::new(Ocr::bundled().unwrap());
    let sections = PdfExtractor::default().with_ocr(ocr).extract(&pdf).unwrap();
    assert_eq!(sections.len(), 1);
    assert_eq!(sections[0].page, Some(1));
    let text = sections[0].text.to_lowercase();
    assert!(text.contains("amount due"), "{text}");
    std::fs::remove_dir_all(dir).unwrap();
}

#[test]
fn textless_files_are_remembered_not_reocrd() {
    let root = temp("index");
    let docs = root.join("docs");
    let data = root.join("data");
    std::fs::create_dir_all(&docs).unwrap();
    let img = text_image();
    img.save(docs.join("bill.png")).unwrap();
    std::fs::write(docs.join("scan.pdf"), scanned_pdf(&img)).unwrap();
    image::RgbImage::from_pixel(400, 300, image::Rgb([250, 250, 250]))
        .save(docs.join("blank.png"))
        .unwrap();
    std::fs::write(
        docs.join("todo.md"),
        "Renew the car insurance before March.",
    )
    .unwrap();

    let store = Store::open(&data).unwrap();
    let bm25 = TantivyIndex::open(&data).unwrap();
    let mut embedder = Embedder::load(&data.join("model")).unwrap();
    let ocr = Arc::new(Ocr::bundled().unwrap());
    let images = ImageExtractor::new(ocr.clone());
    let pdf = PdfExtractor::default().with_ocr(ocr);
    let extractors: [&dyn Extractor; 2] = [&images, &pdf];
    let hooks = IndexHooks {
        extractors: &extractors,
        progress: None,
    };
    let mut pass = |options| {
        index_directory_with(&docs, &[], &store, &bm25, &mut embedder, options, &hooks).unwrap()
    };

    let first = pass(IndexOptions::default());
    assert_eq!(first.indexed.len(), 3, "{:?}", first.failed);
    assert_eq!(first.failed.len(), 1, "{:?}", first.failed);
    assert!(first.failed[0].contains("blank.png"), "{:?}", first.failed);
    assert!(
        first.failed[0].contains("no text found"),
        "{:?}",
        first.failed
    );

    // The blank image is remembered: nothing is extracted again.
    let second = pass(IndexOptions::default());
    assert_eq!(second.skipped_unchanged, 4);
    assert!(second.indexed.is_empty() && second.failed.is_empty());

    // retry_empty re-examines just the text-less file.
    let retry = pass(IndexOptions {
        retry_empty: true,
        ..Default::default()
    });
    assert_eq!(retry.skipped_unchanged, 3);
    assert_eq!(retry.failed.len(), 1);

    let hits = hybrid_search("electricity bill amount", 5, &store, &bm25, &mut embedder).unwrap();
    let scan = hits
        .iter()
        .find(|h| h.path.ends_with("scan.pdf"))
        .expect("scanned PDF is searchable");
    assert_eq!(scan.page, Some(1));
    assert!(
        hits.iter().any(|h| h.path.ends_with("bill.png")),
        "{hits:?}"
    );
    assert!(!hits.iter().any(|h| h.path.ends_with("blank.png")));

    drop((store, bm25, embedder));
    std::fs::remove_dir_all(root).unwrap();
}
