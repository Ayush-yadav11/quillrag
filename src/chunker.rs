//! Chunking: paragraph-first splitting with hard cap at ~1000 chars and
//! ~120-char overlap between consecutive chunks.

pub const TARGET_CHUNK_CHARS: usize = 1000;
pub const OVERLAP_CHARS: usize = 120;

/// Split text into overlapping chunks. Prefers paragraph boundaries, falls
/// back to sentence-ish boundaries (". "), then raw character windows.
pub fn chunk_text(text: &str) -> Vec<String> {
    let mut chunks = Vec::new();
    let mut current = String::new();

    for para in split_paragraphs(text) {
        if para.chars().count() > TARGET_CHUNK_CHARS {
            // Flush what we have, then hard-split the oversized paragraph.
            if !current.trim().is_empty() {
                chunks.push(current.trim().to_string());
                current.clear();
            }
            chunks.extend(hard_split(&para));
            continue;
        }
        if current.chars().count() + para.chars().count() + 1 > TARGET_CHUNK_CHARS
            && !current.is_empty()
        {
            chunks.push(current.trim().to_string());
            current.clear();
            // Carry overlap from the previous chunk's tail for continuity.
            let tail = tail_chars(chunks.last().unwrap(), OVERLAP_CHARS);
            if !tail.is_empty() {
                current.push_str(&tail);
                current.push(' ');
            }
        }
        current.push_str(para);
        current.push('\n');
    }
    if !current.trim().is_empty() {
        chunks.push(current.trim().to_string());
    }
    chunks.retain(|c| !c.trim().is_empty() && c.chars().count() >= 8);
    chunks
}

fn split_paragraphs(text: &str) -> Vec<&str> {
    text.split("\n\n")
        .flat_map(|p| p.split("\r\n\r\n"))
        .map(str::trim)
        .filter(|p| !p.is_empty())
        .collect()
}

fn tail_chars(s: &str, n: usize) -> String {
    let chars: Vec<char> = s.chars().collect();
    if chars.len() <= n {
        return s.to_string();
    }
    // Snap to a word boundary so we don't start mid-word.
    let start = chars.len() - n;
    let slice: String = chars[start..].iter().collect();
    match slice.find(char::is_whitespace) {
        Some(i) => slice[i..].trim_start().to_string(),
        None => slice,
    }
}

fn hard_split(s: &str) -> Vec<String> {
    let mut out = Vec::new();
    let mut start = 0;
    let chars: Vec<char> = s.chars().collect();
    while start < chars.len() {
        let end = (start + TARGET_CHUNK_CHARS).min(chars.len());
        let mut cut = end;
        if end < chars.len() {
            // Prefer breaking at the last ". " before the hard limit.
            let window: String = chars[start..end].iter().collect();
            if let Some(pos) = window.rfind(". ") {
                cut = start + pos + 1; // keep the period
            }
        }
        out.push(
            chars[start..cut]
                .iter()
                .collect::<String>()
                .trim()
                .to_string(),
        );
        start = cut;
    }
    out.into_iter().filter(|c| !c.is_empty()).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn splits_long_text() {
        let para = "word ".repeat(300); // 1500 chars
        let text = format!("{para}\n\nsecond paragraph with distinct content here");
        let chunks = chunk_text(&text);
        assert!(chunks.len() >= 2);
        assert!(chunks
            .iter()
            .all(|c| c.chars().count() <= TARGET_CHUNK_CHARS + 50));
    }

    #[test]
    fn short_text_single_chunk() {
        let chunks = chunk_text("hello world this is a short document");
        assert_eq!(chunks.len(), 1);
        assert_eq!(chunks[0], "hello world this is a short document");
    }

    #[test]
    fn empty_and_tiny_input() {
        assert!(chunk_text("").is_empty());
        assert!(chunk_text("ab").is_empty()); // below min length
    }
}
