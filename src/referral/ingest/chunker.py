"""Split documents into paragraph chunks that keep page and character offsets for citations."""
import re

from referral.schemas import Chunk, Document

MAX_CHARS = 1200


def chunk_document(doc: Document, max_chars: int = MAX_CHARS, sentence_cuts: bool = False) -> list[Chunk]:
    chunks: list[Chunk] = []
    for page_no, text in enumerate(doc.pages, start=1):
        for start, end in _paragraph_spans(text, max_chars, sentence_cuts):
            body = text[start:end].strip()
            if not body:
                continue
            chunks.append(
                Chunk(
                    # case_id prefix keeps ids unique when chunks from many cases share one index (E2)
                    chunk_id=f"{doc.case_id}/{doc.doc_id}:p{page_no}:{start}",
                    case_id=doc.case_id,
                    doc_id=doc.doc_id,
                    doc_type=doc.doc_type,
                    doc_date=doc.doc_date,
                    page=page_no,
                    start=start,
                    end=end,
                    text=body,
                )
            )
    return chunks


def _paragraph_spans(text: str, max_chars: int, sentence_cuts: bool = False) -> list[tuple[int, int]]:
    """Paragraph boundaries (blank lines); long paragraphs are cut at max_chars.

    Fixed cuts split words and sentences: at 600 and 300 characters only 84.0% and 78.8% of dev gold passages
    sat inside one chunk, and a 1,203-character paragraph left the 3-character chunk "f 1" (dev-021). With
    sentence_cuts, a long paragraph is cut at a section header, sentence end, line break or space in the second
    half of the window (see _cut_point), and only then at max_chars.
    """
    spans, cursor = [], 0
    for match in re.finditer(r"\n\s*\n", text + "\n\n"):
        para_end = match.start()
        while para_end - cursor > max_chars:
            cut = _cut_point(text, cursor, max_chars, para_end) if sentence_cuts else cursor + max_chars
            spans.append((cursor, cut))
            cursor = cut
        if para_end > cursor:
            spans.append((cursor, para_end))
        cursor = match.end()
    return spans


# A line that opens a section: "EXAM:", "Cardiac exam:" on its own line, or an all-caps title.
_SECTION_START = re.compile(r"\n(?=[A-Z][A-Za-z /&()-]{1,40}:[ \t]*\n|[A-Z][A-Z /&()-]{3,}[ \t]*\n)")
_SENTENCE_END = re.compile(r"[.!?](?=\s)")


def _cut_point(text: str, start: int, max_chars: int, para_end: int) -> int:
    """Where to end a chunk that starts at `start`: the best boundary in the second half of the window.

    Preference: before a section header (keeps "EXAM:" with its lines), then a sentence end, then a line break
    (OCR-style wraps put line breaks inside sentences: "HDL: 44\nmg/dL"), then a space, then a hard cut. The
    window stops max_chars // 8 before the paragraph's end, so a paragraph just over the limit splits into two
    real chunks instead of leaving a remnant like "f 1".
    """
    half = start + max_chars // 2
    end = min(start + max_chars, para_end - max_chars // 8)
    window = text[half:end]
    for pattern in (_SECTION_START, _SENTENCE_END):
        if ends := list(pattern.finditer(window)):
            return half + ends[-1].end()
    for sep in ("\n", " "):
        if (i := window.rfind(sep)) >= 0:
            return half + i + 1
    return end


def chunk_case(docs: list[Document], max_chars: int = MAX_CHARS, sentence_cuts: bool = False) -> list[Chunk]:
    return [c for d in docs for c in chunk_document(d, max_chars, sentence_cuts)]
