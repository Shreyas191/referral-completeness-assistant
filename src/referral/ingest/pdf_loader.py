"""Turn a case folder of PDFs into Document objects."""
import re
from datetime import date, datetime
from pathlib import Path

import pdfplumber

from referral.schemas import Document

DOC_TYPE_HINTS = {
    "referral": "referral_letter",
    "progress": "progress_note",
    "lab": "lab_report",
    "imaging": "imaging_report",
    "echo": "imaging_report",
    "medication": "medication_list",
}

DATE_PATTERNS = [
    (r"\b(\d{1,2})/(\d{1,2})/(\d{4})\b", "%m/%d/%Y"),
    (r"\b(\d{4})-(\d{2})-(\d{2})\b", "%Y-%m-%d"),
]


def guess_doc_type(filename: str) -> str:
    name = filename.lower()
    for hint, doc_type in DOC_TYPE_HINTS.items():
        if hint in name:
            return doc_type
    return "unknown"


def guess_doc_date(text: str) -> date | None:
    """First date found in the document header; None if nothing readable."""
    head = text[:1500]
    for pattern, fmt in DATE_PATTERNS:
        match = re.search(pattern, head)
        if match:
            try:
                return datetime.strptime(match.group(0), fmt).date()
            except ValueError:
                continue
    return None


def load_pdf(path: Path, case_id: str) -> Document:
    with pdfplumber.open(path) as pdf:
        pages = [page.extract_text() or "" for page in pdf.pages]
    return Document(
        doc_id=path.stem,
        case_id=case_id,
        doc_type=guess_doc_type(path.name),
        doc_date=guess_doc_date(pages[0] if pages else ""),
        pages=pages,
    )


def load_text(path: Path, case_id: str) -> Document:
    """Plain-text document; pages separated by form feeds (\\f). Handy for hand-made cases and tests."""
    pages = path.read_text(encoding="utf-8").split("\f")
    return Document(
        doc_id=path.stem,
        case_id=case_id,
        doc_type=guess_doc_type(path.name),
        doc_date=guess_doc_date(pages[0] if pages else ""),
        pages=pages,
    )


def load_case(case_dir: Path, reverse: bool = False) -> list[Document]:
    """Load every .pdf and .txt document in a case folder, sorted by file name (reverse=True: E4 position
    experiment, same documents in the opposite order)."""
    case_id = case_dir.name
    docs = []
    for path in sorted(case_dir.iterdir(), reverse=reverse):
        if path.suffix.lower() == ".pdf":
            docs.append(load_pdf(path, case_id))
        elif path.suffix.lower() == ".txt":
            docs.append(load_text(path, case_id))
    return docs
