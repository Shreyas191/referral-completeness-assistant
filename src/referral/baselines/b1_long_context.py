"""B1, primary baseline: the whole packet and checklist in one LLM call, same output schema."""
from typing import Literal

from pydantic import BaseModel

from referral.ingest.chunker import chunk_case
from referral.llm import LLM
from referral.pipeline.verifier import SYSTEM, Evidence, format_passages
from referral.schemas import CaseResult, Checklist, Citation, Document, ItemResult, Status


class B1Item(BaseModel):
    item_id: str
    status: Literal["present", "missing", "contradictory", "stale", "needs_review", "not_applicable"]
    evidence: list[Evidence] = []
    rationale: str


class B1Output(BaseModel):
    referral_reason: str
    items: list[B1Item]


def run_b1(docs: list[Document], checklist: Checklist, reference_date, llm: LLM) -> CaseResult:
    chunks = chunk_case(docs)
    by_id = {c.chunk_id: c for c in chunks}
    items_text = "\n".join(
        f"- {i.item_id}: {i.description}"
        + (f" (only if {i.condition})" if i.condition else "")
        + (f" (stale if older than {i.recency_days} days)" if i.recency_days else "")
        for i in checklist.items
    )
    user = (
        f"Referral date: {reference_date}\nChecklist:\n{items_text}\n\n"
        f"Packet:\n{format_passages(chunks)}\n\nGive a status for every checklist item."
    )
    out = llm.structured(SYSTEM, user, B1Output)
    results = [
        ItemResult(
            item_id=it.item_id,
            status=Status(it.status),
            rationale=it.rationale,
            citations=[
                Citation(chunk_id=e.chunk_id, doc_id=by_id[e.chunk_id].doc_id, page=by_id[e.chunk_id].page,
                         quote=e.quote, doc_date=by_id[e.chunk_id].doc_date)
                for e in it.evidence if e.chunk_id in by_id
            ],
        )
        for it in out.items
    ]
    return CaseResult(case_id=docs[0].case_id, checklist_id=checklist.checklist_id,
                      referral_reason=out.referral_reason, items=results)
