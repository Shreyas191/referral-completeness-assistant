"""LLM item verifier: reads retrieved chunks and proposes a status with quoted evidence.

The LLM only proposes present / missing / contradictory / not_applicable. Stale and
Needs review are decided afterwards in code (validator.py), so those rules stay deterministic.
"""
from typing import Literal

from pydantic import BaseModel

from referral.llm import LLM
from referral.schemas import ChecklistItem, Chunk

SYSTEM = """You check whether a specialist referral packet documents a checklist item.
Rules:
- Use only the passages given. Never use outside knowledge about the patient.
- A negated mention ("no prior colonoscopy") is not evidence the item exists.
- An order that was not resulted ("echo scheduled") does not satisfy the item.
- Quote evidence verbatim, copied exactly from the passage, and give its chunk_id.
- If two passages give conflicting values (dose, side, date, diagnosis), answer contradictory and cite both.
- If the item is conditional and the condition does not hold, answer not_applicable."""


class Evidence(BaseModel):
    chunk_id: str
    quote: str


class VerifierOutput(BaseModel):
    item_id: str
    status: Literal["present", "missing", "contradictory", "not_applicable"]
    evidence: list[Evidence] = []
    rationale: str


def format_passages(chunks: list[Chunk]) -> str:
    return "\n\n".join(
        f"[chunk_id={c.chunk_id} | doc={c.doc_id} ({c.doc_type}) | page={c.page} | date={c.doc_date}]\n{c.text}"
        for c in chunks
    )


def verify_item(llm: LLM, item: ChecklistItem, chunks: list[Chunk]) -> VerifierOutput:
    condition = f"\nCondition: required only if {item.condition}." if item.condition else ""
    user = (
        f"Checklist item {item.item_id}: {item.description}{condition}\n"
        f"Also known as: {', '.join(item.synonyms) or 'n/a'}\n\n"
        f"Passages:\n{format_passages(chunks) or '(none retrieved)'}"
    )
    return llm.structured(SYSTEM, user, VerifierOutput)
