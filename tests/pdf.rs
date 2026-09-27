//! PDF extraction end to end: real PDFs built with lopdf, indexed through
//! the extractor hook, searched with page numbers.

use lopdf::content::{Content, Operation};
use lopdf::{dictionary, Document, Object, Stream};
use quillrag::indexer::index_directory_with;
use quillrag::{
    hybrid_search, Embedder, Extractor, IndexHooks, IndexOptions, PdfExtractor, Store, TantivyIndex,
};
use std::path::Path;

fn temp(label: &str) -> std::path::PathBuf {
    let p = std::env::temp_dir().join(format!("qr-pdf-{label}-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&p);
    std::fs::create_dir_all(&p).unwrap();
    p
}

/// Write a PDF with one page per entry; each page holds the given lines.
/// A page with no lines has no text layer, like a scanned image.
fn write_pdf(path: &Path, pages: &[&[&str]]) {
    let mut doc = Document::with_version("1.5");
    let pages_id = doc.new_object_id();
    let font_id = doc.add_object(dictionary! {
        "Type" => "Font",
        "Subtype" => "Type1",
        "BaseFont" => "Helvetica",
    });
    let resources_id = doc.add_object(dictionary! {
        "Font" => dictionary! { "F1" => font_id },
    });
    let mut kids = Vec::new();
    for lines in pages {
        let mut ops = Vec::new();
        for (i, line) in lines.iter().enumerate() {
            ops.push(Operation::new("BT", vec![]));
            ops.push(Operation::new("Tf", vec!["F1".into(), 12.into()]));
            ops.push(Operation::new(
                "Td",
                vec![72.into(), (760 - 18 * i as i64).into()],
            ));
            ops.push(Operation::new("Tj", vec![Object::string_literal(*line)]));
            ops.push(Operation::new("ET", vec![]));
        }
        let content = Content { operations: ops };
        let content_id = doc.add_object(Stream::new(dictionary! {}, content.encode().unwrap()));
        let page_id = doc.add_object(dictionary! {
            "Type" => "Page",
            "Parent" => pages_id,
            "Contents" => content_id,
        });
        kids.push(page_id.into());
    }
    let count = kids.len() as i64;
    doc.objects.insert(
        pages_id,
        Object::Dictionary(dictionary! {
            "Type" => "Pages",
            "Kids" => kids,
            "Count" => count,
            "Resources" => resources_id,
            "MediaBox" => vec![0.into(), 0.into(), 612.into(), 792.into()],
        }),
    );
    let catalog_id = doc.add_object(dictionary! {
        "Type" => "Catalog",
        "Pages" => pages_id,
    });
    doc.trailer.set("Root", catalog_id);
    doc.save(path).unwrap();
}

#[test]
fn extracts_one_section_per_page() {
    let dir = temp("sections");
    let pdf = dir.join("report.pdf");
    write_pdf(
        &pdf,
        &[
            &["Quarterly report", "Revenue grew in every region."],
            &[],
            &["Appendix: the zyxglorp invoice is overdue."],
        ],
    );
    let sections = PdfExtractor::default().extract(&pdf).unwrap();
    let pages: Vec<_> = sections.iter().map(|s| s.page).collect();
    // The blank page 2 yields no section; numbering keeps the real pages.
    assert_eq!(pages, vec![Some(1), Some(3)]);
    assert!(
        sections[0].text.contains("Revenue grew"),
        "{:?}",
        sections[0]
    );
    assert!(sections[1].text.contains("zyxglorp"), "{:?}", sections[1]);
    std::fs::remove_dir_all(dir).unwrap();
}

#[test]
fn scanned_and_corrupt_pdfs_fail_cleanly() {
    let dir = temp("bad");
    let scanned = dir.join("scan.pdf");
    write_pdf(&scanned, &[&[], &[]]);
    let err = PdfExtractor::default().extract(&scanned).unwrap_err();
    assert!(format!("{err:#}").contains("no text layer"), "{err:#}");

    let corrupt = dir.join("corrupt.pdf");
    std::fs::write(&corrupt, b"%PDF-1.5\nthis is not really a pdf").unwrap();
    let err = PdfExtractor::default().extract(&corrupt).unwrap_err();
    assert!(format!("{err:#}").contains("not a readable PDF"), "{err:#}");
    std::fs::remove_dir_all(dir).unwrap();
}

#[test]
fn indexed_pdf_hits_carry_page_numbers() {
    let root = temp("index");
    let docs = root.join("docs");
    let data = root.join("data");
    std::fs::create_dir_all(&docs).unwrap();
    write_pdf(
        &docs.join("handbook.pdf"),
        &[
            &["Employee handbook", "Welcome to the company."],
            &[
                "Leave policy",
                "Everyone gets twenty days of paid vacation per year.",
            ],
            &["Expenses", "Submit travel receipts within thirty days."],
        ],
    );
    write_pdf(&docs.join("scan.pdf"), &[&[]]);
    std::fs::write(docs.join("notes.md"), "Plain notes about sourdough bread.").unwrap();

    let store = Store::open(&data).unwrap();
    let bm25 = TantivyIndex::open(&data).unwrap();
    let mut embedder = Embedder::load(&data.join("model")).unwrap();
    let pdf = PdfExtractor::default();
    let extractors: [&dyn Extractor; 1] = [&pdf];
    let hooks = IndexHooks {
        extractors: &extractors,
        progress: None,
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
    assert_eq!(report.failed.len(), 1);
    assert!(report.failed[0].contains("scan.pdf"), "{:?}", report.failed);

    let hits = hybrid_search(
        "how much holiday time do I get",
        3,
        &store,
        &bm25,
        &mut embedder,
    )
    .unwrap();
    let top = &hits[0];
    assert!(top.path.ends_with("handbook.pdf"), "{hits:?}");
    assert_eq!(top.page, Some(2), "{hits:?}");

    // Unpaged formats have no page.
    let hits = hybrid_search("sourdough", 1, &store, &bm25, &mut embedder).unwrap();
    assert!(hits[0].path.ends_with("notes.md"));
    assert_eq!(hits[0].page, None);

    // A second pass re-extracts nothing: the unchanged PDFs are skipped, and
    // the text-less scan was remembered rather than reported again.
    let again = index_directory_with(
        &docs,
        &[],
        &store,
        &bm25,
        &mut embedder,
        IndexOptions::default(),
        &hooks,
    )
    .unwrap();
    assert_eq!(again.indexed.len(), 0);
    assert_eq!(again.skipped_unchanged, 3);
    assert!(again.failed.is_empty(), "{:?}", again.failed);

    drop((store, bm25, embedder));
    std::fs::remove_dir_all(root).unwrap();
}
