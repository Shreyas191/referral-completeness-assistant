"""Error attribution (#14): label every wrong item with the stage that caused it, from the saved traces.

    python -m referral.eval.errors runs/pipe-dev        # writes runs/pipe-dev/errors.json and errors.md

The harness also runs this after every pipeline run. B0 and B1 have no retrieval traces, so they are skipped.

Causes, checked in this order, so each error is charged to the earliest stage that broke:
  quarantined       the identity check removed the document holding the gold evidence
  retrieval_miss    the gold evidence never reached the model (not in the first-pass top k, not added by
                    the absence re-check, and the long-context read did not run). For a Contradictory item,
                    evidence from fewer than two of its documents reached the model.
  validator_reject  the model's proposal was turned into Needs review by the validator (a quote it could not
                    find in the cited chunk, or an unreadable date)
  recency_window    the item was found, but Present vs Stale came out wrong. That call is made in code from the
                    cited document's date, so this points at dating (e.g. a recent note quoting an old result)
  conflict_checker  the rule-based conflict checker turned a Present into Contradictory
  no_gold_evidence  gold says the item is in the packet but lists no evidence text, so retrieval can't be judged
                    (a generator gap to fix, not a system error)
  reasoning_error   the evidence was in front of the model and it still chose the wrong status
"""
import argparse
import json
from collections import Counter
from pathlib import Path

from referral.eval.gold import GoldCase, GoldEvidence, GoldItem, load_gold
from referral.eval.metrics import PRESENT_LIKE, _evidence_hit
from referral.eval.report import render_report
from referral.ingest.chunker import chunk_case
from referral.ingest.pdf_loader import load_case
from referral.pipeline.trace import CaseTrace, ItemTrace
from referral.schemas import CaseResult, Chunk, ItemResult, Status

CAUSES = ("quarantined", "retrieval_miss", "validator_reject", "recency_window", "conflict_checker",
          "no_gold_evidence", "reasoning_error")
EXAMPLES_PER_CAUSE = 5


def _hits(evidence: list[GoldEvidence], chunks: list[Chunk]) -> set[str]:
    """doc_ids of gold evidence found in these chunks."""
    return {ge.doc_id for ge in evidence for ch in chunks if ch.doc_id == ge.doc_id and _evidence_hit(ch.text, ge)}


def _deciding_stage(t: ItemTrace, flags: list[str]) -> str:
    """Which model call produced the proposal the validator received."""
    if "long_context_overturned" in flags or t.long_context_status is not None:
        return "long_context"
    if "absence_check_overturned" in flags or t.absence_status is not None:
        return "absence_recheck"
    return "first_pass"


def attribute_item(gold: GoldCase, gi: GoldItem, pred: ItemResult | None, t: ItemTrace | None, trace: CaseTrace,
                   chunks_by_id: dict[str, Chunk]) -> dict:
    """Cause and supporting detail for one wrong item."""
    gold_status, evidence = gi.status, gi.evidence
    flags = pred.flags if pred else []
    pred_status = pred.status if pred else Status.NEEDS_REVIEW
    record = {"case_id": gold.case_id, "item_id": gi.item_id, "gold": gold_status.value,
              "pred": pred_status.value, "details": []}
    details = record["details"]

    if t is None:
        record["cause"] = "reasoning_error"
        details.append("no_trace_for_item")
        return record

    stage = _deciding_stage(t, flags)
    record["stage"] = stage
    # Everything the model read before its final proposal: first pass, absence extras, or the whole case.
    viewed_ids = [*t.shown, *t.absence_extra]
    if t.long_context_status is not None:
        viewed_ids += [cid for cid, ch in chunks_by_id.items()
                       if ch.case_id == gold.case_id and ch.doc_id not in trace.quarantined_docs]
    viewed = [chunks_by_id[c] for c in dict.fromkeys(viewed_ids) if c in chunks_by_id]
    ranked = [chunks_by_id[c] for c in t.ranked if c in chunks_by_id]

    evidence_docs = {ge.doc_id for ge in evidence}
    seen_docs = _hits(evidence, viewed)
    needed = min(2, len(evidence_docs)) if gold_status == Status.CONTRADICTORY else 1
    first_rank = next((rank for rank, ch in enumerate(ranked, start=1) if _hits(evidence, [ch])), None)
    record.update(gold_evidence=[ge.text for ge in evidence], first_gold_rank=first_rank,
                  shown=t.shown, absence_extra=t.absence_extra,
                  citations=[{"doc_id": c.doc_id, "quote": c.quote} for c in (pred.citations if pred else [])],
                  rationale=pred.rationale if pred else "", flags=flags)
    if pred and any(c.doc_id in gold.wrong_patient_docs for c in pred.citations):
        details.append("cites_wrong_patient_doc")

    if evidence and evidence_docs <= set(trace.quarantined_docs):
        record["cause"] = "quarantined"
        details.append("evidence_docs_quarantined:" + ",".join(sorted(evidence_docs)))
    elif evidence and len(seen_docs) < needed:
        record["cause"] = "retrieval_miss"
        if seen_docs:
            details.append(f"saw_{len(seen_docs)}_of_{len(evidence_docs)}_evidence_docs")
        elif first_rank is not None:
            details.append(f"ranked_{first_rank}_below_k_{len(t.shown)}")
        else:
            details.append(f"not_in_top_{len(t.ranked)}")
    elif pred_status == Status.NEEDS_REVIEW and any(
            f.startswith("citation_failed") or f == "date_unreadable" for f in flags):
        record["cause"] = "validator_reject"
        details += sorted({f.split(":")[0] for f in flags if f.startswith("citation_failed")} |
                          {f for f in flags if f == "date_unreadable"})
    elif {gold_status, pred_status} == {Status.PRESENT, Status.STALE}:
        record["cause"] = "recency_window"
        dates = sorted({str(c.doc_date) for c in pred.citations if c.doc_date}) if pred else []
        details += [f"cited_doc_dates:{','.join(dates) or 'none'}", *(f for f in flags if f.startswith("age_days"))]
    elif "conflict_checker" in flags and pred_status == Status.CONTRADICTORY:
        record["cause"] = "conflict_checker"
    elif not evidence and gold_status in PRESENT_LIKE:
        record["cause"] = "no_gold_evidence"
    else:
        record["cause"] = "reasoning_error"
        details.append(f"{gold_status.value}->{pred_status.value}")
    return record


def attribute_run(golds: list[GoldCase], preds: list[CaseResult], traces: dict[str, CaseTrace],
                  chunks_by_id: dict[str, Chunk]) -> list[dict]:
    """One record per wrong item in cases that have a trace."""
    by_case = {p.case_id: p for p in preds}
    records = []
    for g in golds:
        trace = traces.get(g.case_id)
        if trace is None:
            continue
        pred_items = {r.item_id: r for r in by_case[g.case_id].items} if g.case_id in by_case else {}
        trace_items = {t.item_id: t for t in trace.items}
        for gi in g.items:
            pi = pred_items.get(gi.item_id)
            if (pi.status if pi else Status.NEEDS_REVIEW) == gi.status:
                continue
            records.append(attribute_item(g, gi, pi, trace_items.get(gi.item_id), trace, chunks_by_id))
    return records


def cause_counts(records: list[dict]) -> dict[str, int]:
    counts = Counter(r["cause"] for r in records)
    return {cause: counts[cause] for cause in CAUSES}


def _short(text: str, limit: int = 160) -> str:
    text = " ".join(text.split())
    return text if len(text) <= limit else text[: limit - 3] + "..."


def render_errors(records: list[dict], total_items: int, run_name: str = "") -> str:
    counts = cause_counts(records)
    lines = [f"# Error attribution{f' for `{run_name}`' if run_name else ''}", "",
             f"{len(records)} wrong items out of {total_items}. Each is charged to the earliest stage that broke "
             "(see referral/eval/errors.py for the rules).", "",
             "| Cause | Items | Share of errors |", "| --- | --- | --- |"]
    for cause, n in counts.items():
        lines.append(f"| {cause} | {n} | {n / len(records):.0%} |" if records else f"| {cause} | 0 | - |")

    stage_counts = Counter((r["cause"], r.get("stage", "-")) for r in records)
    if stage_counts:
        lines += ["", "## By deciding stage", "",
                  "Which model call made the final proposal: first pass, absence re-check, or long-context read.",
                  "", "| Cause | Stage | Items |", "| --- | --- | --- |"]
        lines += [f"| {c} | {s} | {n} |" for (c, s), n in sorted(stage_counts.items())]

    flips = Counter((r["cause"], f"{r['gold']} -> {r['pred']}") for r in records)
    if flips:
        lines += ["", "## Gold -> predicted, by cause", "", "| Cause | Gold -> pred | Items |", "| --- | --- | --- |"]
        lines += [f"| {c} | {f} | {n} |" for (c, f), n in sorted(flips.items(), key=lambda kv: (kv[0][0], -kv[1]))]

    for cause in CAUSES:
        examples = [r for r in records if r["cause"] == cause][:EXAMPLES_PER_CAUSE]
        if not examples:
            continue
        lines += ["", f"## {cause}: examples ({len(examples)} of {counts[cause]})", ""]
        for r in examples:
            lines.append(f"- **{r['case_id']} / {r['item_id']}**: gold {r['gold']}, predicted {r['pred']}"
                         + (f"; stage {r['stage']}" if r.get("stage") else "")
                         + (f"; {', '.join(r['details'])}" if r["details"] else ""))
            if r.get("gold_evidence"):
                lines.append(f"  - Gold evidence: \"{_short(r['gold_evidence'][0])}\"")
            for c in r.get("citations", [])[:2]:
                lines.append(f"  - Cited ({c['doc_id']}): \"{_short(c['quote'])}\"")
            if r.get("rationale"):
                lines.append(f"  - Model rationale: {_short(r['rationale'])}")
    return "\n".join(lines) + "\n"


def write_errors(out_dir: Path, golds: list[GoldCase], preds: list[CaseResult], traces: dict[str, CaseTrace],
                 chunks_by_id: dict[str, Chunk]) -> dict[str, int]:
    """Write errors.json and errors.md into the run folder; return counts per cause."""
    records = attribute_run(golds, preds, traces, chunks_by_id)
    total = sum(len(g.items) for g in golds if g.case_id in traces)
    (out_dir / "errors.json").write_text(json.dumps(records, indent=2))
    (out_dir / "errors.md").write_text(render_errors(records, total, out_dir.name), encoding="utf-8")
    return cause_counts(records)


def from_run_dir(run_dir: Path, cases_dir: Path | None = None) -> dict[str, int]:
    """Rebuild the report for a finished run from its predictions, traces and the case folders."""
    metrics = json.loads((run_dir / "metrics.json").read_text())
    cases_dir = cases_dir or Path(metrics.get("cases_dir", "data/cases"))
    traces = {p.name.removesuffix(".trace.json"): CaseTrace.model_validate_json(p.read_text())
              for p in run_dir.glob("*.trace.json")}
    if not traces:
        raise SystemExit(f"{run_dir} has no traces; error attribution needs a pipeline run.")
    case_ids = metrics.get("case_ids") or sorted(traces)
    golds = [load_gold(cases_dir / cid) for cid in case_ids]
    preds = [CaseResult.model_validate_json((run_dir / f"{cid}.json").read_text()) for cid in case_ids
             if (run_dir / f"{cid}.json").exists()]
    # Chunk every case in the run: under global scope (E2) a trace can point at another case's chunks.
    chunks = [c for cid in case_ids for c in chunk_case(load_case(cases_dir / cid))]
    counts = write_errors(run_dir, golds, preds, traces, {c.chunk_id: c for c in chunks})
    metrics["error_attribution"] = counts
    (run_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))
    (run_dir / "report.md").write_text(render_report(metrics), encoding="utf-8")
    return counts


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("run_dir", type=Path)
    ap.add_argument("--cases-dir", type=Path, default=None)
    args = ap.parse_args()
    counts = from_run_dir(args.run_dir, args.cases_dir)
    print(json.dumps(counts, indent=2))
    print(f"Wrote {args.run_dir / 'errors.md'} and errors.json")


if __name__ == "__main__":
    main()
