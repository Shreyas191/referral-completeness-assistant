"""Deterministic checks that turn a verifier proposal into the final status.

Order of decisions (answers the Present / Stale / Needs review question):
  1. Verify each citation: the quote must actually appear in the cited chunk.
  2. No verified citation left for a Present/Contradictory claim -> NEEDS_REVIEW.
  3. Verified evidence but its date is missing or unreadable, and the item has a window -> NEEDS_REVIEW.
  4. Verified, dated, and the newest evidence is older than the window -> STALE.
  5. Otherwise -> PRESENT.
Stale is certain about the age; Needs review means the evidence itself could not be verified.
"""
import re
from datetime import date

from rapidfuzz import fuzz

from referral.config import settings
from referral.pipeline.verifier import VerifierOutput
from referral.schemas import ChecklistItem, Chunk, Citation, ItemResult, Status


def quote_score(quote: str, chunk: Chunk, length_check: bool = True) -> float:
    """How well the quote matches somewhere in the chunk (0-100, lower-cased, whitespace-normalized).

    rapidfuzz partial_ratio aligns the shorter string inside the longer one, so a quote longer than its chunk
    scores 100 whenever it contains the chunk, whatever else it says: in dev-021 the chunk "f 1" (a page-break
    remnant) gives 100 to "Echo shows LVEF of 15% with severe MR.". With length_check, a quote longer than the
    chunk is compared with the whole chunk (fuzz.ratio), so any text the chunk does not contain lowers the score.
    """
    q, text = " ".join(quote.lower().split()), " ".join(chunk.text.lower().split())
    if not q:
        return 0.0
    if length_check and len(q) > len(text):
        return fuzz.ratio(q, text)
    return fuzz.partial_ratio(q, text)


_PIECE_BREAK = re.compile(r"\.{3,}|\u2026|\[\s*\.\.\.\s*\]|\n")  # ellipses and line breaks
MIN_PIECE_CHARS = 8  # shorter pieces ("and", "...") are skipped; they carry no evidence


def quote_supported(quote: str, chunk: Chunk, threshold: int = settings.quote_match_threshold,
                    piecewise: bool = False, length_check: bool = True) -> bool:
    """The quote appears in the chunk. With piecewise, a quote also passes if every line or "..." piece does.

    Observed failure (rq-dev, elide-dev): the model joins non-adjacent lines of the cited chunk into one quote,
    with or without an ellipsis: "Date: 11/03/2026 ... Cardiac exam: ...", or "CURRENT MEDICATION LIST\n
    Date: 10/05/2026\nLisinopril 10 mg ..." skipping the name, DOB and MRN lines. Each line is verbatim, but the
    joined quote scores 82-89.5 against 90, and a correct Present goes to Needs review. Each piece must still
    match on its own, so nothing invented passes; a quote with no piece of MIN_PIECE_CHARS or more fails.
    """
    if quote_score(quote, chunk, length_check) >= threshold:
        return True
    if not piecewise:
        return False
    pieces = [p.strip() for p in _PIECE_BREAK.split(quote) if len(p.strip()) >= MIN_PIECE_CHARS]
    return len(pieces) > 1 and all(quote_score(p, chunk, length_check) >= threshold for p in pieces)


def decide(
    item: ChecklistItem,
    proposal: VerifierOutput,
    chunks_by_id: dict[str, Chunk],
    reference_date: date,
    piecewise_quotes: bool = False,
    quote_length_check: bool = True,
) -> ItemResult:
    flags: list[str] = []
    proposed = Status(proposal.status)

    if proposed in {Status.MISSING, Status.NOT_APPLICABLE}:
        return ItemResult(item_id=item.item_id, status=proposed, rationale=proposal.rationale)

    citations: list[Citation] = []
    for ev in proposal.evidence:
        chunk = chunks_by_id.get(ev.chunk_id)
        if chunk and quote_supported(ev.quote, chunk, piecewise=piecewise_quotes,
                                          length_check=quote_length_check):
            citations.append(
                Citation(chunk_id=chunk.chunk_id, doc_id=chunk.doc_id, page=chunk.page,
                         quote=ev.quote, doc_date=chunk.doc_date)
            )
        else:
            flags.append(f"citation_failed:{ev.chunk_id}")

    def result(status: Status) -> ItemResult:
        return ItemResult(item_id=item.item_id, status=status, citations=citations,
                          rationale=proposal.rationale, flags=flags)

    if proposed == Status.CONTRADICTORY:
        # A contradiction needs at least two verified sources from different documents.
        return result(Status.CONTRADICTORY if len({c.doc_id for c in citations}) >= 2 else Status.NEEDS_REVIEW)

    if not citations:
        return result(Status.NEEDS_REVIEW)

    if item.recency_days is None:
        return result(Status.PRESENT)

    dated = [c.doc_date for c in citations if c.doc_date]
    if not dated:
        flags.append("date_unreadable")
        return result(Status.NEEDS_REVIEW)

    age_days = (reference_date - max(dated)).days
    if age_days > item.recency_days:
        flags.append(f"age_days:{age_days}")
        return result(Status.STALE)
    return result(Status.PRESENT)
