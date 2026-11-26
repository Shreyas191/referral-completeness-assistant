"""Absence check: a Missing label is a claim that nothing in the packet supports the item.

A retrieval miss looks exactly like a missing item, so Missing is only accepted after two more looks:
  1. Wide re-check (per item): a larger-k search plus a synonym scan over every chunk, re-verified.
  2. Long-context final read (per case, #4): every item still Missing is checked in one call that
     reads the whole packet. Long packets are split into windows; any window finding evidence wins.
Each stage that overturns Missing leaves a flag, so its effect can be measured in the ablation.
"""
import re

from pydantic import BaseModel

from referral.llm import LLM
from referral.pipeline.verifier import SYSTEM, VerifierOutput, format_passages
from referral.retrieval.expansion import STOP
from referral.retrieval.hybrid import BM25Retriever
from referral.schemas import ChecklistItem, Chunk

# ~75k tokens of packet text per call; longer packets are read in several windows.
WINDOW_CHARS = 300_000


def _item_patterns(item: ChecklistItem) -> list[re.Pattern]:
    terms = [t.lower() for t in [item.description, *item.synonyms] if t.strip()]
    return [re.compile(rf"\b{re.escape(t)}\b") for t in terms]


def synonym_scan(item: ChecklistItem, chunks: list[Chunk]) -> list[Chunk]:
    patterns = _item_patterns(item)
    return [c for c in chunks if any(p.search(c.text.lower()) for p in patterns)]


def names_item(item: ChecklistItem, proposal: VerifierOutput) -> bool:
    """Does any quoted evidence name the item, by its description or a synonym?

    Used to hold the long-context read to item-specific evidence (PipelineConfig.strict_long_context).
    Observed failure (dev): its only Missing -> Present overturns quoted text that does not name the item,
    e.g. "Patient is on warfarin." for the medication list and "Please accept her for care at your cardiology
    clinic." for the reason for referral. Condition calls (not_applicable) are not affected.
    """
    patterns = _item_patterns(item)
    return any(p.search(e.quote.lower()) for e in proposal.evidence for p in patterns)


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


CONDITION_TOP_K = 3
_FILLER = STOP | {"is", "on", "a", "an", "of", "to", "in", "or", "not", "taking", "takes", "currently", "uses"}


def condition_terms(item: ChecklistItem) -> set[str]:
    """Content words of the item's condition: "patient is on warfarin" -> {"warfarin"}."""
    return {w for w in re.findall(r"[a-z0-9]+", (item.condition or "").lower()) if w not in _FILLER and len(w) > 2}


def condition_candidates(item: ChecklistItem, first_pass: list[Chunk], all_chunks: list[Chunk],
                         k: int = CONDITION_TOP_K) -> list[Chunk]:
    """Unseen chunks that mention the item's condition, best BM25 match first.

    Observed failure (sent600-dev, dev-024 inr): retrieval searches for the item ("INR result"), not its
    condition, so neither chunk saying "warfarin" was shown, and the verifier called INR Not applicable. The
    absence check only re-examines Missing, so nothing caught it. A chunk qualifies only if it contains a
    content word of the condition, so a case with no mention costs no extra call.
    """
    terms = condition_terms(item)
    if not terms:
        return []
    seen = {c.chunk_id for c in first_pass}
    pattern = re.compile(r"\b(" + "|".join(map(re.escape, sorted(terms))) + r")\b")
    pool = [c for c in all_chunks if c.chunk_id not in seen and pattern.search(c.text.lower())]
    return BM25Retriever(pool).search(" ".join(sorted(terms)), k) if len(pool) > 2 else pool[:k]


class LongContextOutput(BaseModel):
    items: list[VerifierOutput]


def windows(chunks: list[Chunk], max_chars: int = WINDOW_CHARS) -> list[list[Chunk]]:
    """Consecutive groups of chunks, each under max_chars of text (a single huge chunk gets its own)."""
    groups: list[list[Chunk]] = [[]]
    size = 0
    for chunk in chunks:
        if groups[-1] and size + len(chunk.text) > max_chars:
            groups.append([])
            size = 0
        groups[-1].append(chunk)
        size += len(chunk.text)
    return [g for g in groups if g]


def final_read(llm: LLM, items: list[ChecklistItem], chunks: list[Chunk]) -> dict[str, VerifierOutput]:
    """Re-check items still marked Missing against the whole packet. Returns item_id -> proposal.

    An item stays Missing only if no window finds it. The first window that proposes anything other
    than missing wins, so the validator still has to verify its quote afterwards.
    """
    if not items or not chunks:
        return {}
    listed = "\n".join(
        f"- {i.item_id}: {i.description}"
        + (f" (also: {', '.join(i.synonyms)})" if i.synonyms else "")
        + (f" (required only if {i.condition})" if i.condition else "")
        + (f" Definition: {i.definition}" if i.definition else "")
        for i in items
    )
    best: dict[str, VerifierOutput] = {}
    for window in windows(chunks):
        user = (
            "A first search did not find these checklist items. Read every passage carefully, "
            "including abbreviations, tables and passages late in long notes, and decide again.\n\n"
            f"Items:\n{listed}\n\nPassages:\n{format_passages(window)}\n\n"
            "Return one entry per item, using the item ids above."
        )
        out = llm.structured(SYSTEM, user, LongContextOutput)
        for proposal in out.items:
            if proposal.item_id not in {i.item_id for i in items}:
                continue  # ignore ids the model invented
            current = best.get(proposal.item_id)
            if current is None or (current.status == "missing" and proposal.status != "missing"):
                best[proposal.item_id] = proposal
    return best
