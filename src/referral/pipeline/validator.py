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


def quote_score(quote: str, chunk: Chunk) -> float:
    """How well the quote matches somewhere in the chunk (rapidfuzz partial_ratio, 0-100, whitespace-normalized)."""
    if not quote.strip():
        return 0.0
    return fuzz.partial_ratio(" ".join(quote.lower().split()), " ".join(chunk.text.lower().split()))


_ELLIPSIS = re.compile(r"\.{3,}|\u2026|\[\s*\.\.\.\s*\]")
MIN_PIECE_CHARS = 8  # shorter pieces ("and", "...") are skipped; they carry no evidence


def quote_supported(quote: str, chunk: Chunk, threshold: int = settings.quote_match_threshold,
                    allow_elision: bool = False) -> bool:
    """The quote appears in the chunk. With allow_elision, a quote with "..." passes if every piece does.

    Observed failure (rq-dev): the model joins two lines of a note, e.g. "Date: 11/03/2026 ... Cardiac exam:
    regular rate and rhythm, ...". Each piece is verbatim, but the joined quote scores 84 and 87.9 against 90,
    and a correct Present goes to Needs review. Each piece must still match on its own, so nothing invented
    passes; a quote with no piece of MIN_PIECE_CHARS or more fails.
    """
    if quote_score(quote, chunk) >= threshold:
        return True
    if not allow_elision or not _ELLIPSIS.search(quote):
        return False
    pieces = [p.strip() for p in _ELLIPSIS.split(quote) if len(p.strip()) >= MIN_PIECE_CHARS]
    return bool(pieces) and all(quote_score(p, chunk) >= threshold for p in pieces)


def decide(
    item: ChecklistItem,
    proposal: VerifierOutput,
    chunks_by_id: dict[str, Chunk],
    reference_date: date,
    allow_elision: bool = False,
) -> ItemResult:
    flags: list[str] = []
    proposed = Status(proposal.status)

    if proposed in {Status.MISSING, Status.NOT_APPLICABLE}:
        return ItemResult(item_id=item.item_id, status=proposed, rationale=proposal.rationale)

    citations: list[Citation] = []
    for ev in proposal.evidence:
        chunk = chunks_by_id.get(ev.chunk_id)
        if chunk and quote_supported(ev.quote, chunk, allow_elision=allow_elision):
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
