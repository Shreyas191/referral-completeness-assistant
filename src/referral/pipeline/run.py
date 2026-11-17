"""End-to-end pipeline. Each optional stage can be switched off for ablations (E2, E5, absence)."""
from dataclasses import asdict, dataclass
from datetime import date

from referral.config import settings
from referral.ingest.chunker import chunk_case
from referral.ingest.identity import quarantine
from referral.llm import LLM
from referral.pipeline.absence import candidates_for_recheck, final_read
from referral.pipeline.conflicts import dose_conflicts, laterality_conflict
from referral.pipeline.trace import TRACE_DEPTH, CaseTrace, ItemTrace
from referral.pipeline.validator import decide
from referral.pipeline.verifier import VerifierOutput, verify_item
from referral.retrieval.hybrid import CaseRetriever, DenseRetriever
from referral.schemas import CaseResult, Checklist, Chunk, Document, FollowUp, ItemResult, Status


@dataclass
class PipelineConfig:
    retrieval_mode: str = "bm25"     # bm25 | dense | hybrid   (E3)
    scope: str = "case"              # case | global           (E2)
    top_k: int = settings.top_k
    identity_check: bool = True
    absence_check: bool = True       # wide per-item re-check before accepting Missing
    absence_long_context: bool = True  # whole-packet final read for items still Missing (#4)
    validator: bool = True           # (E5)
    conflict_checker: bool = True    # (E5)


def run_case(
    docs: list[Document],
    checklist: Checklist,
    reference_date: date,
    llm: LLM,
    cfg: PipelineConfig | None = None,
    dense: DenseRetriever | None = None,
    global_chunks: list[Chunk] | None = None,
    trace: CaseTrace | None = None,
) -> CaseResult:
    """Check one case. Pass `trace` to record retrieval and per-stage decisions for trajectory metrics."""
    cfg = cfg or PipelineConfig()
    case_id = docs[0].case_id
    trace = trace if trace is not None else CaseTrace(case_id=case_id)
    trace.case_id, trace.config = case_id, asdict(cfg)

    quarantined: list[str] = []
    if cfg.identity_check:
        docs, quarantined = quarantine(docs)
    trace.quarantined_docs = quarantined

    chunks = chunk_case(docs)
    # In the global-scope ablation, citations to other cases' chunks must still validate, so they can be counted.
    pool = global_chunks if cfg.scope == "global" and global_chunks else chunks
    chunks_by_id = {c.chunk_id: c for c in [*chunks, *pool]}
    if dense:
        dense.add(chunks)
    retriever = CaseRetriever(chunks, dense=dense, mode=cfg.retrieval_mode, scope=cfg.scope,
                              global_chunks=global_chunks)

    proposals: dict[str, VerifierOutput] = {}
    flags: dict[str, list[str]] = {i.item_id: [] for i in checklist.items}
    for item in checklist.items:
        ranked = retriever.search(item, max(cfg.top_k, TRACE_DEPTH))
        shown = ranked[: cfg.top_k]
        proposal = verify_item(llm, item, shown)
        t = ItemTrace(item_id=item.item_id, ranked=[c.chunk_id for c in ranked],
                      shown=[c.chunk_id for c in shown], first_status=proposal.status)

        if proposal.status == "missing" and cfg.absence_check:
            wide = retriever.search(item, settings.absence_top_k)
            extra = candidates_for_recheck(item, shown, wide, chunks)
            t.absence_extra = [c.chunk_id for c in extra]
            if extra:
                second = verify_item(llm, item, shown + extra)
                t.absence_status = second.status
                if second.status != "missing":
                    proposal = second
                    flags[item.item_id].append("absence_check_overturned")
        proposals[item.item_id] = proposal
        trace.items.append(t)

    if cfg.absence_check and cfg.absence_long_context:
        still_missing = [i for i in checklist.items if proposals[i.item_id].status == "missing"]
        for item_id, proposal in final_read(llm, still_missing, chunks).items():
            trace.item(item_id).long_context_status = proposal.status
            if proposal.status != "missing":
                proposals[item_id] = proposal
                flags[item_id].append("long_context_overturned")

    results: list[ItemResult] = []
    for item in checklist.items:
        proposal = proposals[item.item_id]
        result = decide(item, proposal, chunks_by_id, reference_date) if cfg.validator else _unvalidated(proposal)
        result.flags.extend(flags[item.item_id])

        if cfg.conflict_checker and result.status == Status.PRESENT:
            cited = [chunks_by_id[c.chunk_id] for c in result.citations if c.chunk_id in chunks_by_id]
            if dose_conflicts(cited) or laterality_conflict(cited):
                result.status = Status.CONTRADICTORY
                result.flags.append("conflict_checker")
        trace.item(item.item_id).final_status = result.status.value
        results.append(result)

    return CaseResult(
        case_id=case_id,
        checklist_id=checklist.checklist_id,
        items=results,
        quarantined_docs=quarantined,
        followups=draft_followups(checklist, results),
    )


def _unvalidated(proposal: VerifierOutput) -> ItemResult:
    """E5 ablation: take the LLM's word, keeping its quotes as citations without checking them."""
    from referral.schemas import Citation

    return ItemResult(
        item_id=proposal.item_id,
        status=Status(proposal.status),
        rationale=proposal.rationale,
        citations=[Citation(chunk_id=e.chunk_id, doc_id=e.chunk_id.split("/")[-1].split(":")[0], page=0,
                            quote=e.quote) for e in proposal.evidence],
    )


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
