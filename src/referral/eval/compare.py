"""Compare two run folders side by side, with paired McNemar tests on the items both runs scored.

    python -m referral.eval.compare runs/b1-dev runs/pipe-dev
    python -m referral.eval.compare runs/pipe-dev runs/pipe-dev-k10 --out runs/k10-vs-k5.md

Metrics and CIs come from each run's metrics.json. The McNemar tests reload predictions and gold labels,
so they need the case folders the runs were made from (recorded in metrics.json, or --cases-dir).
"""
import argparse
import json
from math import comb
from pathlib import Path

from referral.eval.gold import GoldCase, load_gold
from referral.eval.report import HEADLINE, fmt
from referral.schemas import GAP_STATUSES, CaseResult, Status

SIDE_METRICS = (*HEADLINE, "recall@3", "recall@5", "recall@10", "cost_per_case_usd", "mean_latency_s",
                "max_latency_s", "llm_calls", "input_tokens", "output_tokens")


def load_run(run_dir: Path) -> tuple[dict, dict[str, CaseResult]]:
    metrics = json.loads((run_dir / "metrics.json").read_text())
    preds = {}
    for path in run_dir.glob("*.json"):
        if path.name != "metrics.json" and not path.name.endswith(".trace.json") and path.name != "errors.json":
            pred = CaseResult.model_validate_json(path.read_text())
            preds[pred.case_id] = pred
    return metrics, preds


def mcnemar_exact(a_only: int, b_only: int) -> float:
    """Two-sided exact McNemar p-value from the discordant counts (binomial test with p = 0.5)."""
    n = a_only + b_only
    if n == 0:
        return 1.0
    tail = sum(comb(n, i) for i in range(min(a_only, b_only) + 1)) / 2 ** n
    return min(1.0, 2 * tail)


def paired_tests(golds: list[GoldCase], preds_a: dict[str, CaseResult], preds_b: dict[str, CaseResult]) -> dict:
    """Item-level paired outcomes. 'status' = exact status right; 'gap' = gap / no-gap call right."""
    counts = {test: {"both": 0, "a_only": 0, "b_only": 0, "neither": 0} for test in ("status", "gap")}
    for g in golds:
        items_a = {r.item_id: r.status for r in preds_a[g.case_id].items}
        items_b = {r.item_id: r.status for r in preds_b[g.case_id].items}
        for gi in g.items:
            sa, sb = items_a.get(gi.item_id, Status.NEEDS_REVIEW), items_b.get(gi.item_id, Status.NEEDS_REVIEW)
            gold_gap = gi.status in GAP_STATUSES
            for test, right_a, right_b in (
                ("status", sa == gi.status, sb == gi.status),
                ("gap", (sa in GAP_STATUSES) == gold_gap, (sb in GAP_STATUSES) == gold_gap),
            ):
                key = "both" if right_a and right_b else "a_only" if right_a else "b_only" if right_b else "neither"
                counts[test][key] += 1
    for c in counts.values():
        c["p_value"] = mcnemar_exact(c["a_only"], c["b_only"])
    return counts


def _cell(m: dict, name: str) -> str:
    if name not in m:
        return "–"
    ci = (m.get("ci95") or {}).get(name)
    return f"{fmt(m[name])} [{fmt(ci[0])}, {fmt(ci[1])}]" if ci else fmt(m[name])


def _delta(a: dict, b: dict, name: str) -> str:
    va, vb = a.get(name), b.get(name)
    if not isinstance(va, (int, float)) or not isinstance(vb, (int, float)):
        return ""
    return f"{vb - va:+.3f}" if isinstance(va, float) or isinstance(vb, float) else f"{vb - va:+d}"


def compare(run_a: Path, run_b: Path, cases_dir: Path | None = None) -> str:
    ma, preds_a = load_run(run_a)
    mb, preds_b = load_run(run_b)
    a, b = run_a.name, run_b.name
    lines = [f"# `{a}` (A) vs `{b}` (B)", "", "| | A | B |", "| --- | --- | --- |"]
    for key in ("system", "split", "cases", "model", "git_commit", "git_dirty", "created_utc"):
        va, vb = ma.get(key), mb.get(key)
        if key == "git_commit":
            va, vb = va and va[:12], vb and vb[:12]
        lines.append(f"| {key} | {fmt(va)} | {fmt(vb)} |")
    cfg_a, cfg_b = ma.get("config") or {}, mb.get("config") or {}
    for key in sorted(set(cfg_a) | set(cfg_b)):
        if cfg_a.get(key) != cfg_b.get(key):
            lines.append(f"| config.{key} | {fmt(cfg_a.get(key))} | {fmt(cfg_b.get(key))} |")

    lines += ["", "## Metrics", "", "Brackets: 95% bootstrap CI over cases. Diff = B - A.", "",
              "| Metric | A | B | Diff |", "| --- | --- | --- | --- |"]
    for name in SIDE_METRICS:
        if name in ma or name in mb:
            lines.append(f"| {name} | {_cell(ma, name)} | {_cell(mb, name)} | {_delta(ma, mb, name)} |")

    slices_a = {r["slice"]: r for r in ma.get("slices", [])}
    slices_b = {r["slice"]: r for r in mb.get("slices", [])}
    if slices_a or slices_b:
        lines += ["", "## Slices", "", "| Slice | Cases | Status acc A | Status acc B | Gap R A | Gap R B |",
                  "| --- | --- | --- | --- | --- | --- |"]
        for name in dict.fromkeys([*slices_a, *slices_b]):
            ra, rb = slices_a.get(name, {}), slices_b.get(name, {})
            lines.append(f"| {name} | {ra.get('cases', rb.get('cases', 0))} | {fmt(ra.get('status_accuracy'))} | "
                         f"{fmt(rb.get('status_accuracy'))} | {fmt(ra.get('gap_recall'))} | "
                         f"{fmt(rb.get('gap_recall'))} |")

    attr_a, attr_b = ma.get("error_attribution"), mb.get("error_attribution")
    if attr_a or attr_b:
        lines += ["", "## Error attribution", "", "| Cause | A | B |", "| --- | --- | --- |"]
        for cause in dict.fromkeys([*(attr_a or {}), *(attr_b or {})]):
            lines.append(f"| {cause} | {fmt((attr_a or {}).get(cause))} | {fmt((attr_b or {}).get(cause))} |")

    lines += ["", "## Paired tests (McNemar, exact)", ""]
    cases_dir = cases_dir or Path(ma.get("cases_dir", "data/cases"))
    shared = sorted(set(preds_a) & set(preds_b))
    golds = [load_gold(cases_dir / cid) for cid in shared if (cases_dir / cid / "gold.json").exists()]
    if len(shared) < max(len(preds_a), len(preds_b)):
        lines.append(f"Runs share {len(shared)} of {len(preds_a)} / {len(preds_b)} cases; tests use the shared ones.")
    if not golds:
        lines.append(f"No gold labels found under {cases_dir}; pass --cases-dir.")
    else:
        lines += ["Items right in one run but not the other. A small p-value means the difference is unlikely "
                  "to be chance.", "", "| Test | Both right | Only A | Only B | Neither | p |",
                  "| --- | --- | --- | --- | --- | --- |"]
        for test, c in paired_tests(golds, preds_a, preds_b).items():
            lines.append(f"| {test} correct | {c['both']} | {c['a_only']} | {c['b_only']} | {c['neither']} | "
                         f"{c['p_value']:.3g} |")
    return "\n".join(lines) + "\n"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("run_a", type=Path)
    ap.add_argument("run_b", type=Path)
    ap.add_argument("--cases-dir", type=Path, default=None)
    ap.add_argument("--out", type=Path, default=None, help="also write the comparison to this file")
    args = ap.parse_args()
    text = compare(args.run_a, args.run_b, args.cases_dir)
    if args.out:
        args.out.write_text(text, encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
