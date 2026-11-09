"""Deterministic checks that turn a verifier proposal into the final status.

Order of decisions (answers the Present / Stale / Needs review question):
  1. Verify each citation: the quote must actually appear in the cited chunk.
  2. No verified citation left for a Present/Contradictory claim -> NEEDS_REVIEW.
  3. Verified evidence but its date is missing or unreadable, and the item has a window -> NEEDS_REVIEW.
  4. Verified, dated, and the newest evidence is older than the window -> STALE.
  5. Otherwise -> PRESENT.
Stale is certain about the age; Needs review means the evidence itself could not be verified.
"""
from datetime import date

from rapidfuzz import fuzz

from referral.config import settings
from referral.pipeline.verifier import VerifierOutput
from referral.schemas import ChecklistItem, Chunk, Citation, ItemResult, Status


def quote_supported(quote: str, chunk: Chunk, threshold: int = settings.quote_match_threshold) -> bool:
    if not quote.strip():
        return False
    return fuzz.partial_ratio(" ".join(quote.lower().split()), " ".join(chunk.text.lower().split())) >= threshold


def decide(
    item: ChecklistItem,
    proposal: VerifierOutput,
    chunks_by_id: dict[str, Chunk],
    reference_date: date,
) -> ItemResult:
    flags: list[str] = []
    proposed = Status(proposal.status)

    if proposed in {Status.MISSING, Status.NOT_APPLICABLE}:
        return ItemResult(item_id=item.item_id, status=proposed, rationale=proposal.rationale)

    citations: list[Citation] = []
    for ev in proposal.evidence:
        chunk = chunks_by_id.get(ev.chunk_id)
        if chunk and quote_supported(ev.quote, chunk):
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
