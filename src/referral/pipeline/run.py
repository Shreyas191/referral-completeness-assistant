"""End-to-end pipeline. Each optional stage can be switched off for ablations (E2, E5)."""
from dataclasses import dataclass
from datetime import date

from referral.config import settings
from referral.ingest.chunker import chunk_case
from referral.ingest.identity import quarantine
from referral.llm import LLM
from referral.pipeline.absence import candidates_for_recheck
from referral.pipeline.conflicts import dose_conflicts, laterality_conflict
from referral.pipeline.validator import decide
from referral.pipeline.verifier import VerifierOutput, verify_item
from referral.retrieval.hybrid import CaseRetriever, DenseRetriever
from referral.schemas import CaseResult, Checklist, Document, FollowUp, ItemResult, Status


@dataclass
class PipelineConfig:
    retrieval_mode: str = "hybrid"   # bm25 | dense | hybrid   (E3)
    scope: str = "case"              # case | global           (E2)
    top_k: int = settings.top_k
    identity_check: bool = True
    absence_check: bool = True
    validator: bool = True           # (E5)
    conflict_checker: bool = True    # (E5)


def run_case(
    docs: list[Document],
    checklist: Checklist,
    reference_date: date,
    llm: LLM,
    cfg: PipelineConfig = PipelineConfig(),
    dense: DenseRetriever | None = None,
) -> CaseResult:
    case_id = docs[0].case_id
    quarantined: list[str] = []
    if cfg.identity_check:
        docs, quarantined = quarantine(docs)

    chunks = chunk_case(docs)
    chunks_by_id = {c.chunk_id: c for c in chunks}
    if dense:
        dense.add(chunks)
    retriever = CaseRetriever(chunks, dense=dense, mode=cfg.retrieval_mode, scope=cfg.scope)

    results: list[ItemResult] = []
    for item in checklist.items:
        hits = retriever.search(item, cfg.top_k)
        proposal = verify_item(llm, item, hits)
        flags: list[str] = []

        if proposal.status == "missing" and cfg.absence_check:
            extra = candidates_for_recheck(item, hits, retriever.search(item, settings.absence_top_k), chunks)
            if extra:
                second = verify_item(llm, item, hits + extra)
                if second.status != "missing":
                    proposal, flags = second, ["absence_check_overturned"]

        if cfg.validator:
            result = decide(item, proposal, chunks_by_id, reference_date)
        else:
            result = _unvalidated(proposal)
        result.flags.extend(flags)

        if cfg.conflict_checker and result.status == Status.PRESENT:
            cited = [chunks_by_id[c.chunk_id] for c in result.citations]
            if dose_conflicts(cited) or laterality_conflict(cited):
                result.status = Status.CONTRADICTORY
                result.flags.append("conflict_checker")
        results.append(result)

    return CaseResult(
        case_id=case_id,
        checklist_id=checklist.checklist_id,
        items=results,
        quarantined_docs=quarantined,
        followups=draft_followups(checklist, results),
    )


def _unvalidated(proposal: VerifierOutput) -> ItemResult:
    return ItemResult(item_id=proposal.item_id, status=Status(proposal.status), rationale=proposal.rationale)


def draft_followups(checklist: Checklist, results: list[ItemResult]) -> list[FollowUp]:
    """Template requests for staff to edit and send. The system never sends anything itself."""
    items = {i.item_id: i for i in checklist.items}
    out = []
    for r in results:
        desc = items[r.item_id].description
        if r.status == Status.MISSING:
            out.append(FollowUp(item_id=r.item_id, request=f"Please send: {desc}."))
        elif r.status == Status.STALE:
            out.append(FollowUp(item_id=r.item_id, request=f"Please send a more recent {desc}."))
        elif r.status == Status.CONTRADICTORY:
            docs = ", ".join(sorted({c.doc_id for c in r.citations}))
            out.append(FollowUp(item_id=r.item_id, request=f"Please clarify {desc}; records disagree ({docs})."))
    return out
