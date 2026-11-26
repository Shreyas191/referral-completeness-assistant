"""End-to-end pipeline. Each optional stage can be switched off for ablations (E2, E5, absence)."""
from dataclasses import asdict, dataclass
from datetime import date

from referral.config import settings
from referral.ingest.chunker import MAX_CHARS, chunk_case
from referral.ingest.identity import quarantine
from referral.llm import LLM
from referral.pipeline.absence import candidates_for_recheck, final_read, names_item
from referral.pipeline.conflicts import dose_conflicts, laterality_conflict
from referral.pipeline.trace import TRACE_DEPTH, CaseTrace, ItemTrace
from referral.pipeline.validator import decide, quote_score, quote_supported
from referral.pipeline.verifier import VerifierOutput, verify_item
from referral.retrieval.expansion import value_expansion
from referral.retrieval.hybrid import CaseRetriever, DenseRetriever
from referral.schemas import CaseResult, Checklist, ChecklistItem, Chunk, Document, FollowUp, ItemResult, Status


@dataclass
class PipelineConfig:
    retrieval_mode: str = "bm25"     # bm25 | dense | hybrid   (E3)
    scope: str = "case"              # case | global           (E2)
    top_k: int = settings.top_k
    chunk_size: int = MAX_CHARS        # E3 chunk size, in characters
    sentence_cuts: bool = False      # cut long paragraphs at section / sentence boundaries, not mid-word
    identity_check: bool = True
    absence_check: bool = True       # wide per-item re-check before accepting Missing
    absence_long_context: bool = True  # whole-packet final read for items still Missing (#4)
    validator: bool = True           # (E5)
    conflict_checker: bool = True    # (E5)
    value_expansion: bool = True     # add other documents' values for the top chunk's terms (vexp-dev runs)
    strict_long_context: bool = True   # final read may overturn Missing only with a quote naming the item
    piecewise_quotes: bool = True    # validator accepts quotes whose lines / "..." pieces each match (pw-dev runs)
    quote_length_check: bool = True  # a quote longer than its chunk must match the whole chunk, not contain it


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

    chunks = chunk_case(docs, cfg.chunk_size, cfg.sentence_cuts)
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
        expansion = value_expansion(shown, chunks) if cfg.value_expansion else []
        first = shown + expansion
        proposal = verify_item(llm, item, first)
        t = ItemTrace(item_id=item.item_id, ranked=[c.chunk_id for c in ranked],
                      shown=[c.chunk_id for c in shown], expansion=[c.chunk_id for c in expansion],
                      first_status=proposal.status)

        if proposal.status == "missing" and cfg.absence_check:
            wide = retriever.search(item, settings.absence_top_k)
            extra = candidates_for_recheck(item, first, wide, chunks)
            t.absence_extra = [c.chunk_id for c in extra]
            if extra:
                second = verify_item(llm, item, first + extra)
                t.absence_status = second.status
                if second.status != "missing":
                    proposal = second
                    flags[item.item_id].append("absence_check_overturned")
        proposals[item.item_id] = proposal
        trace.items.append(t)

    if cfg.absence_check and cfg.absence_long_context:
        still_missing = [i for i in checklist.items if proposals[i.item_id].status == "missing"]
        by_id = {i.item_id: i for i in still_missing}
        for item_id, proposal in final_read(llm, still_missing, chunks).items():
            trace.item(item_id).long_context_status = proposal.status
            if proposal.status == "missing":
                continue
            if (cfg.strict_long_context and proposal.status in {"present", "contradictory"}
                    and not names_item(by_id[item_id], proposal)):
                flags[item_id].append("long_context_rejected")  # stays Missing; staff see the flag
                continue
            proposals[item_id] = proposal
            flags[item_id].append("long_context_overturned")

    results: list[ItemResult] = []
    for item in checklist.items:
        proposal = proposals[item.item_id]
        result = (decide(item, proposal, chunks_by_id, reference_date, piecewise_quotes=cfg.piecewise_quotes,
                         quote_length_check=cfg.quote_length_check)
                  if cfg.validator
                  else _unvalidated(item, proposal, chunks_by_id))
        result.flags.extend(flags[item.item_id])
        if cfg.validator and proposal.status in {"present", "contradictory"}:
            # Only these proposals have their quotes checked; Missing and Not applicable keep no citations.
            trace.item(item.item_id).rejected_quotes = [
                {"chunk_id": e.chunk_id, "quote": e.quote,
                 "score": round(quote_score(e.quote, chunks_by_id[e.chunk_id], cfg.quote_length_check), 1)
                 if e.chunk_id in chunks_by_id else None}
                for e in proposal.evidence
                if e.chunk_id not in chunks_by_id
                or not quote_supported(e.quote, chunks_by_id[e.chunk_id], piecewise=cfg.piecewise_quotes,
                                       length_check=cfg.quote_length_check)]

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


def _unvalidated(item: ChecklistItem, proposal: VerifierOutput, chunks_by_id: dict[str, Chunk]) -> ItemResult:
    """E5 ablation: take the LLM's status and quotes without checking them.

    The item id comes from the checklist, not the model, which sometimes echoes the description instead.
    Citation metadata (document, page, date) is read from the cited chunk when it exists, so citation metrics
    stay comparable with validated runs; the quote itself is still not checked.
    """
    from referral.schemas import Citation

    def citation(chunk_id: str, quote: str) -> Citation:
        chunk = chunks_by_id.get(chunk_id)
        if chunk:
            return Citation(chunk_id=chunk_id, doc_id=chunk.doc_id, page=chunk.page, quote=quote,
                            doc_date=chunk.doc_date)
        return Citation(chunk_id=chunk_id, doc_id=chunk_id.split("/")[-1].split(":")[0], page=0, quote=quote)

    return ItemResult(
        item_id=item.item_id,
        status=Status(proposal.status),
        rationale=proposal.rationale,
        citations=[citation(e.chunk_id, e.quote) for e in proposal.evidence],
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
