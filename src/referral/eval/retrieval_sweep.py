"""E3 retrieval-only sweep: recall@k for BM25, dense and hybrid at several chunk sizes. No LLM calls, no cost.

Three recall measures per item with gold evidence:
  any        at least one gold evidence passage is in the top k (the harness's recall@k)
  all        every gold evidence passage is in the top k
  multi-doc  `all`, restricted to items whose evidence spans two or more documents; a contradiction can only be
             seen when both sides are retrieved
plus `findable`: the share of gold passages that sit inside a single chunk at all. Below 1.000 the chunker cut
gold sentences apart, so lower recall at that size is partly the cut, not the ranking.

    python -m referral.eval.retrieval_sweep --split dev --chunk-sizes 1200 600 300
Dense and hybrid need chromadb (its default embedding model downloads on first use).
"""
import argparse
import json
import tempfile
from pathlib import Path

from referral.eval.gold import load_gold
from referral.eval.harness import CASES, CHECKLISTS, load_checklist
from referral.eval.metrics import _evidence_hit
from referral.ingest.chunker import MAX_CHARS, chunk_document
from referral.ingest.pdf_loader import load_case
from referral.retrieval.hybrid import CaseRetriever

KS = (1, 3, 5, 10)
MODES = ("bm25", "dense", "hybrid")


def sweep(case_dirs: list[Path], checklist_name: str, chunk_size: int, modes=MODES, dense=None,
          sentence_cuts: bool = False) -> dict:
    items = {i.item_id: i for i in load_checklist(checklist_name, CHECKLISTS).items}
    out = {m: {"items": 0, "multi_doc_items": 0, **{f"{r}@{k}": 0 for r in ("any", "all", "multi") for k in KS}}
           for m in modes}
    passages = findable = n_chunks = 0
    for d in case_dirs:
        gold = load_gold(d)
        chunks = [c for doc in load_case(d) for c in chunk_document(doc, chunk_size, sentence_cuts)]
        n_chunks += len(chunks)
        if dense is not None:
            dense.add(chunks)
        for gi in gold.items:
            for e in gi.evidence:
                passages += 1
                findable += any(c.doc_id == e.doc_id and _evidence_hit(c.text, e) for c in chunks)
        for mode in modes:
            retriever = CaseRetriever(chunks, dense=dense, mode=mode)
            r = out[mode]
            for gi in gold.items:
                if not gi.evidence:
                    continue
                ranked = retriever.search(items[gi.item_id], max(KS))
                multi = len({e.doc_id for e in gi.evidence}) > 1
                r["items"] += 1
                r["multi_doc_items"] += multi
                for k in KS:
                    hits = [any(c.doc_id == e.doc_id and _evidence_hit(c.text, e) for c in ranked[:k])
                            for e in gi.evidence]
                    r[f"any@{k}"] += any(hits)
                    r[f"all@{k}"] += all(hits)
                    r[f"multi@{k}"] += multi and all(hits)
    for r in out.values():
        for k in KS:
            r[f"any@{k}"] = round(r[f"any@{k}"] / r["items"], 3) if r["items"] else 0.0
            r[f"all@{k}"] = round(r[f"all@{k}"] / r["items"], 3) if r["items"] else 0.0
            r[f"multi@{k}"] = round(r[f"multi@{k}"] / r["multi_doc_items"], 3) if r["multi_doc_items"] else 0.0
    return {"chunk_size": chunk_size, "sentence_cuts": sentence_cuts, "cases": len(case_dirs), "chunks": n_chunks,
            "findable": round(findable / passages, 3) if passages else 0.0, "modes": out}


def render(results: list[dict]) -> str:
    lines = ["| Chunk size | Cuts | Chunks | Findable | Mode | any@3 | any@5 | all@3 | all@5 | all@10 | multi-doc@5 "
             "| multi-doc@10 |", "| --- " * 12 + "|"]
    for res in results:
        for mode, r in res["modes"].items():
            cuts = "sentence" if res.get("sentence_cuts") else "fixed"
            lines.append(f"| {res['chunk_size']} | {cuts} | {res['chunks']} | {res['findable']:.3f} | {mode} | "
                         f"{r['any@3']:.3f} | {r['any@5']:.3f} | {r['all@3']:.3f} | {r['all@5']:.3f} | "
                         f"{r['all@10']:.3f} | {r['multi@5']:.3f} (n={r['multi_doc_items']}) | {r['multi@10']:.3f} |")
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--split", default="dev", choices=["dev", "adversarial"])
    ap.add_argument("--checklist", default="cardiology-v1")
    ap.add_argument("--cases-dir", type=Path, default=CASES)
    ap.add_argument("--chunk-sizes", type=int, nargs="+", default=[MAX_CHARS])
    ap.add_argument("--sentence-cuts", action="store_true", help="section / sentence boundary cuts")
    ap.add_argument("--modes", nargs="+", default=list(MODES), choices=MODES)
    ap.add_argument("--out", type=Path, help="write the results as JSON here")
    args = ap.parse_args()

    case_dirs = sorted(d for d in args.cases_dir.iterdir()
                       if (d / "gold.json").exists() and load_gold(d).split == args.split)
    results = []
    for size in args.chunk_sizes:
        dense = None
        if set(args.modes) - {"bm25"}:
            from referral.retrieval.hybrid import DenseRetriever

            dense = DenseRetriever(tempfile.mkdtemp(), collection=f"sweep{size}{'s' if args.sentence_cuts else 'f'}")
        results.append(sweep(case_dirs, args.checklist, size, args.modes, dense, args.sentence_cuts))
    print(render(results))
    if args.out:
        args.out.write_text(json.dumps(results, indent=1))


if __name__ == "__main__":
    main()
