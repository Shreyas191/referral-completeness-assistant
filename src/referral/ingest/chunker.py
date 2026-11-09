"""Split documents into paragraph chunks that keep page and character offsets for citations."""
import re

from referral.schemas import Chunk, Document

MAX_CHARS = 1200


def chunk_document(doc: Document, max_chars: int = MAX_CHARS) -> list[Chunk]:
    chunks: list[Chunk] = []
    for page_no, text in enumerate(doc.pages, start=1):
        for start, end in _paragraph_spans(text, max_chars):
            body = text[start:end].strip()
            if not body:
                continue
            chunks.append(
                Chunk(
                    chunk_id=f"{doc.doc_id}:p{page_no}:{start}",
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


def _paragraph_spans(text: str, max_chars: int) -> list[tuple[int, int]]:
    """Paragraph boundaries (blank lines); long paragraphs are cut at max_chars."""
    spans, cursor = [], 0
    for match in re.finditer(r"\n\s*\n", text + "\n\n"):
        para_end = match.start()
        while para_end - cursor > max_chars:
            spans.append((cursor, cursor + max_chars))
            cursor += max_chars
        if para_end > cursor:
            spans.append((cursor, para_end))
        cursor = match.end()
    return spans


def chunk_case(docs: list[Document]) -> list[Chunk]:
    return [c for d in docs for c in chunk_document(d)]
