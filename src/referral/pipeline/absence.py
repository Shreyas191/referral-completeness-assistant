"""Absence check: a Missing label is a claim that nothing in the packet supports the item.

A retrieval miss looks exactly like a missing item, so before accepting Missing we search
again with a much wider net (larger k plus a plain synonym scan over every chunk) and
re-verify. If the second look finds evidence, the Missing label is overturned and flagged.
"""
import re

from referral.schemas import ChecklistItem, Chunk


def synonym_scan(item: ChecklistItem, chunks: list[Chunk]) -> list[Chunk]:
    terms = [t.lower() for t in [item.description, *item.synonyms] if t.strip()]
    patterns = [re.compile(rf"\b{re.escape(t)}\b") for t in terms]
    return [c for c in chunks if any(p.search(c.text.lower()) for p in patterns)]


def candidates_for_recheck(
    item: ChecklistItem,
    first_pass: list[Chunk],
    wide_hits: list[Chunk],
    all_chunks: list[Chunk],
) -> list[Chunk]:
    """Chunks worth showing the verifier again: anything new from the wide search or the scan."""
    seen = {c.chunk_id for c in first_pass}
    out: list[Chunk] = []
    for chunk in [*wide_hits, *synonym_scan(item, all_chunks)]:
        if chunk.chunk_id not in seen:
            seen.add(chunk.chunk_id)
            out.append(chunk)
    return out
