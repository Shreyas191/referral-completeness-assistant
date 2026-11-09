"""Cross-document conflict checker for values the verifier may miss when judging one item.

Starts with medication doses and laterality; extend with dates and diagnoses as failures show up.
"""
import re
from collections import defaultdict

from referral.schemas import Chunk

DOSE_RE = re.compile(r"\b([a-z][a-z\-]{3,})\s+(\d+(?:\.\d+)?)\s*(mg|mcg|g|units)\b", re.I)
SIDE_RE = re.compile(r"\b(left|right|bilateral)\b", re.I)


def dose_conflicts(chunks: list[Chunk]) -> dict[str, set[tuple[str, str]]]:
    """drug -> {(dose, doc_id)} for drugs whose dose differs between documents."""
    seen: dict[str, set[tuple[str, str]]] = defaultdict(set)
    for chunk in chunks:
        for drug, amount, unit in DOSE_RE.findall(chunk.text):
            seen[drug.lower()].add((f"{float(amount):g} {unit.lower()}", chunk.doc_id))
    return {
        drug: entries
        for drug, entries in seen.items()
        if len({dose for dose, _ in entries}) > 1 and len({doc for _, doc in entries}) > 1
    }


def laterality_conflict(chunks: list[Chunk]) -> bool:
    """True if documents disagree on side (left vs right) in the cited evidence for one item."""
    sides_by_doc = defaultdict(set)
    for chunk in chunks:
        for side in SIDE_RE.findall(chunk.text):
            sides_by_doc[chunk.doc_id].add(side.lower())
    sides = {s for doc_sides in sides_by_doc.values() for s in doc_sides} - {"bilateral"}
    return len(sides) > 1 and len(sides_by_doc) > 1
