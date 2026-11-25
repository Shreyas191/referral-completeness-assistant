"""Run a system over a split and write predictions, traces and metrics to runs/<name>/.

    python -m referral.eval.harness --system b1 --split dev --name b1-dev
    python -m referral.eval.harness --system pipeline --split dev --name pipe-dev
    python -m referral.eval.harness --system pipeline --split dev --name e2-global --scope global
    python -m referral.eval.harness --system pipeline --split dev --name abl-no-absence --no-absence-check

Never pass --split test before the freeze.
"""
import argparse
import json
import time
from pathlib import Path

from referral.baselines.b0_keywords import run_b0
from referral.baselines.b1_long_context import run_b1
from referral.config import settings
from referral.eval.errors import write_errors
from referral.eval.gold import GoldCase, load_gold
from referral.eval.metrics import bootstrap_cis, retrieval_recall, slice_table, summarize
from referral.eval.report import cost_usd, render_report, run_info
from referral.ingest.chunker import chunk_case
from referral.ingest.pdf_loader import load_case
from referral.llm import LLM
from referral.pipeline.run import PipelineConfig, run_case
from referral.pipeline.trace import CaseTrace
from referral.schemas import Checklist

CASES = Path("data/cases")
CHECKLISTS = Path("data/checklists")


def load_checklist(checklist_id: str, checklists_dir: Path = CHECKLISTS) -> Checklist:
    return Checklist.model_validate_json((checklists_dir / f"{checklist_id}.json").read_text())


def case_dirs(cases_dir: Path, split: str) -> list[tuple[Path, GoldCase]]:
    out = []
    for case_dir in sorted(p for p in cases_dir.iterdir() if p.is_dir() and (p / "gold.json").exists()):
        gold = load_gold(case_dir)
        if gold.split == split:
            out.append((case_dir, gold))
    return out


def run_split(
    system: str,
    split: str,
    out_dir: Path,
    cfg: PipelineConfig | None = None,
    llm: LLM | None = None,
    cases_dir: Path = CASES,
    checklists_dir: Path = CHECKLISTS,
) -> dict:
    """Run one system over every case in a split. Returns the metrics dict (also written to disk)."""
    cfg = cfg or PipelineConfig()
    out_dir.mkdir(parents=True, exist_ok=True)
    cases = case_dirs(cases_dir, split)
    docs_by_case = {gold.case_id: load_case(case_dir) for case_dir, gold in cases}

    # Every chunk in the split, for recall@k lookups and for the global-scope ablation (E2).
    all_chunks = [c for docs in docs_by_case.values() for c in chunk_case(docs)]
    chunks_by_id = {c.chunk_id: c for c in all_chunks}
    global_chunks = all_chunks if system == "pipeline" and cfg.scope == "global" else None

    dense = None
    if system == "pipeline" and cfg.retrieval_mode != "bm25":
        from referral.retrieval.hybrid import DenseRetriever
        dense = DenseRetriever(str(out_dir / "chroma"))
        if global_chunks:
            dense.add(global_chunks)

    golds, preds, traces, latencies = [], [], {}, []
    for _, gold in cases:
        docs = docs_by_case[gold.case_id]
        checklist = load_checklist(gold.checklist_id, checklists_dir)
        start = time.time()
        if system == "b0":
            pred = run_b0(docs, checklist)
        elif system == "b1":
            pred = run_b1(docs, checklist, gold.reference_date, llm)
        else:
            trace = CaseTrace(case_id=gold.case_id)
            pred = run_case(docs, checklist, gold.reference_date, llm, cfg, dense=dense,
                            global_chunks=global_chunks, trace=trace)
            traces[gold.case_id] = trace
            (out_dir / f"{gold.case_id}.trace.json").write_text(trace.model_dump_json(indent=2))
        latencies.append(time.time() - start)
        golds.append(gold)
        preds.append(pred)
        (out_dir / f"{gold.case_id}.json").write_text(pred.model_dump_json(indent=2))

    metrics: dict = {"run": out_dir.name, "system": system, "split": split, "cases": len(golds),
                     "model": getattr(llm, "name", type(llm).__name__) if llm is not None else None,
                     **run_info(), "cases_dir": str(cases_dir), "case_ids": [g.case_id for g in golds]}
    if system == "pipeline":
        metrics["config"] = traces[golds[0].case_id].config if golds else {}
    metrics.update(summarize(golds, preds))
    metrics["ci95"] = bootstrap_cis(golds, preds)
    if traces:
        metrics.update(retrieval_recall(golds, traces, chunks_by_id))
    metrics["mean_latency_s"] = sum(latencies) / len(latencies) if latencies else 0.0
    metrics["max_latency_s"] = max(latencies, default=0.0)
    if llm is not None:
        cost = cost_usd(llm.input_tokens, llm.output_tokens)
        metrics.update(llm_calls=llm.calls, input_tokens=llm.input_tokens, output_tokens=llm.output_tokens,
                       price_per_mtok={"input": settings.price_input_per_mtok,
                                       "output": settings.price_output_per_mtok},
                       cost_usd=cost, cost_per_case_usd=cost / len(golds) if cost is not None and golds else None)
    else:
        metrics.update(cost_usd=0.0, cost_per_case_usd=0.0)
    metrics["slices"] = slice_table(golds, preds)
    if traces:
        metrics["error_attribution"] = write_errors(out_dir, golds, preds, traces, chunks_by_id)
    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))
    (out_dir / "report.md").write_text(render_report(metrics), encoding="utf-8")
    return metrics


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--system", choices=["b0", "b1", "pipeline"], required=True)
    ap.add_argument("--split", default="dev")
    ap.add_argument("--name", required=True)
    ap.add_argument("--cases-dir", type=Path, default=CASES)
    ap.add_argument("--retrieval", default="bm25", choices=["bm25", "dense", "hybrid"])
    ap.add_argument("--scope", default="case", choices=["case", "global"], help="E2 ablation")
    ap.add_argument("--top-k", type=int, default=PipelineConfig.top_k)
    ap.add_argument("--no-identity-check", action="store_true")
    ap.add_argument("--no-absence-check", action="store_true", help="drop both absence stages")
    ap.add_argument("--no-long-context", action="store_true", help="keep the wide re-check, drop the final read")
    ap.add_argument("--no-validator", action="store_true", help="E5 ablation")
    ap.add_argument("--no-conflict-checker", action="store_true", help="E5 ablation")
    ap.add_argument("--no-value-expansion", action="store_true",
                    help="drop value-term expansion (contradiction recall)")
    args = ap.parse_args()

    if args.split == "test":
        print("Warning: running on the held-out test split. This should happen once, at the freeze.")

    cfg = PipelineConfig(
        retrieval_mode=args.retrieval, scope=args.scope, top_k=args.top_k,
        identity_check=not args.no_identity_check,
        absence_check=not args.no_absence_check,
        absence_long_context=not (args.no_absence_check or args.no_long_context),
        validator=not args.no_validator, conflict_checker=not args.no_conflict_checker,
        value_expansion=not args.no_value_expansion,
    )
    llm = LLM() if args.system != "b0" else None
    metrics = run_split(args.system, args.split, Path("runs") / args.name, cfg, llm, cases_dir=args.cases_dir)
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
