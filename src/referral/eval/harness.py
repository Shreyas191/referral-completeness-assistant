"""Run a system over a split and write predictions + metrics to runs/<name>/.

    python -m referral.eval.harness --system b1 --split dev --name b1-dev
"""
import argparse
import json
import time
from pathlib import Path

from referral.baselines.b0_keywords import run_b0
from referral.baselines.b1_long_context import run_b1
from referral.eval.gold import load_gold
from referral.eval.metrics import summarize
from referral.ingest.pdf_loader import load_case
from referral.llm import LLM
from referral.pipeline.run import PipelineConfig, run_case
from referral.schemas import Checklist

CASES = Path("data/cases")
CHECKLISTS = Path("data/checklists")


def load_checklist(checklist_id: str) -> Checklist:
    return Checklist.model_validate_json((CHECKLISTS / f"{checklist_id}.json").read_text())


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--system", choices=["b0", "b1", "pipeline"], required=True)
    ap.add_argument("--split", default="dev")
    ap.add_argument("--name", required=True)
    ap.add_argument("--no-absence-check", action="store_true")
    ap.add_argument("--no-identity-check", action="store_true")
    ap.add_argument("--retrieval", default="bm25", choices=["bm25", "dense", "hybrid"])
    args = ap.parse_args()

    out_dir = Path("runs") / args.name
    out_dir.mkdir(parents=True, exist_ok=True)
    llm = LLM() if args.system != "b0" else None
    cfg = PipelineConfig(retrieval_mode=args.retrieval, absence_check=not args.no_absence_check,
                         identity_check=not args.no_identity_check)

    golds, preds, latencies = [], [], []
    for case_dir in sorted(p for p in CASES.iterdir() if p.is_dir()):
        gold = load_gold(case_dir)
        if gold.split != args.split:
            continue
        docs = load_case(case_dir)
        checklist = load_checklist(gold.checklist_id)
        start = time.time()
        if args.system == "b0":
            pred = run_b0(docs, checklist)
        elif args.system == "b1":
            pred = run_b1(docs, checklist, gold.reference_date, llm)
        else:
            dense = None
            if args.retrieval != "bm25":
                from referral.retrieval.hybrid import DenseRetriever
                dense = DenseRetriever(str(out_dir / "chroma"))
            pred = run_case(docs, checklist, gold.reference_date, llm, cfg, dense=dense)
        latencies.append(time.time() - start)
        golds.append(gold)
        preds.append(pred)
        (out_dir / f"{gold.case_id}.json").write_text(pred.model_dump_json(indent=2))

    metrics = summarize(golds, preds)
    metrics["cases"] = len(golds)
    metrics["mean_latency_s"] = sum(latencies) / len(latencies) if latencies else 0.0
    if llm:
        metrics.update(llm_calls=llm.calls, input_tokens=llm.input_tokens, output_tokens=llm.output_tokens)
    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
