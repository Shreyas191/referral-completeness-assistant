"""Value-term expansion: bring the second document of a possible contradiction in front of the verifier.

Observed failure (dev, every run): a contradiction is missed when only one of the two documents is retrieved.
  dev-009 bmp   the letter's "Most recent potassium is 5.9 mmol/L" scores 0 in BM25: "potassium" occurs in half
                of the case's 4 chunks, so its IDF is log(1) = 0, and zero-score chunks are dropped.
  dev-012 meds  the note's "Continue tiotropium 36 mcg" shares no word with the item or its synonyms; drug
                names are patient-specific and cannot be listed in the checklist.

Change: take the terms that carry a value in the top-ranked chunk ("Potassium: 4.2", "Tiotropium 18 mcg") and
add up to `max_extra` chunks from other documents of the same case that state a value for any of those terms.
Matching is on words, not BM25 scores, so it does not depend on IDF. Switch: PipelineConfig.value_expansion.
"""
import re

from referral.schemas import Chunk

# Words that sit next to numbers in headers and narrative but are not measured things.
STOP = {
    "date", "dob", "mrn", "page", "age", "phone", "fax", "ext", "suite", "room", "time", "year", "years", "day",
    "days", "week", "weeks", "month", "months", "since", "about", "for", "and", "the", "with", "was", "were",
    "has", "had", "than", "from", "last", "most", "recent", "dated", "total", "grade", "lead", "visit",
    "collected", "reported", "approximately", "over", "under", "per", "every", "each", "into", "patient", "name",
}
_TERM = re.compile(r"\b([A-Za-z]{3,})\s*:?\s*\d")


def value_terms(text: str) -> set[str]:
    """Lower-cased words directly followed by a number: analytes, drugs, measures."""
    return {t.lower() for t in _TERM.findall(text)} - STOP


def value_mentions(text: str, terms: set[str]) -> set[str]:
    """Terms that `text` gives a value for (the term, then a number within the same clause)."""
    low = text.lower()
    return {t for t in terms if re.search(rf"\b{re.escape(t)}\b[^.\n]{{0,25}}?\d", low)}


def value_expansion(shown: list[Chunk], case_chunks: list[Chunk], max_extra: int = 2) -> list[Chunk]:
    """Chunks from other documents that state a value for a term the top chunk gives a value for.

    Ranked by how many such terms they mention; ties keep packet order. Never returns a shown chunk or a chunk
    from the top chunk's own document, and only ever searches `case_chunks` (one case).
    """
    if not shown:
        return []
    top = shown[0]
    terms = value_terms(top.text)
    if not terms:
        return []
    seen = {c.chunk_id for c in shown}
    scored = [(len(value_mentions(c.text, terms)), n, c) for n, c in enumerate(case_chunks)
              if c.chunk_id not in seen and c.doc_id != top.doc_id]
    scored = [s for s in scored if s[0] > 0]
    scored.sort(key=lambda s: (-s[0], s[1]))
    return [c for _, _, c in scored[:max_extra]]
