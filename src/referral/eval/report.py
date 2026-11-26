"""Run provenance, cost, and the human-readable report.md the harness writes into every run folder."""
import subprocess
from datetime import datetime, timezone
from pathlib import Path

from referral.config import settings

REPO = Path(__file__).resolve().parents[3]

# Targets from the project plan: (metric, ">=" or "<=", value).
TARGETS = {
    "gap_recall": (">=", 0.90),
    "gap_precision": (">=", 0.80),
    "status_macro_f1": (">=", 0.80),
    "status_macro_f1_gold": (">=", 0.80),
    "citation_precision": (">=", 0.95),
    "cost_per_case_usd": ("<=", 0.10),
    "mean_latency_s": ("<=", 60.0),
}
HEADLINE = ("gap_recall", "gap_precision", "gap_f1", "status_macro_f1", "status_macro_f1_gold", "status_accuracy",
            "citation_precision",
            "citation_doc_precision", "citation_page_precision",
            "false_missing_rate", "false_contradictory_wrong_patient", "foreign_citation_rate")


def run_info() -> dict:
    """Git commit and dirty flag (so a number can be traced to code), plus a UTC timestamp."""
    def git(*args: str) -> str | None:
        try:
            out = subprocess.run(["git", *args], cwd=REPO, capture_output=True, text=True, timeout=10, check=True)
            return out.stdout.strip()
        except (OSError, subprocess.SubprocessError):
            return None

    status = git("status", "--porcelain")
    return {
        "git_commit": git("rev-parse", "HEAD"),
        "git_dirty": bool(status) if status is not None else None,
        "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }


def cost_usd(input_tokens: int, output_tokens: int, price_in: float | None = settings.price_input_per_mtok,
             price_out: float | None = settings.price_output_per_mtok) -> float | None:
    """Token cost in USD, or None when the prices for the model are not configured."""
    if price_in is None or price_out is None:
        return None
    return (input_tokens * price_in + output_tokens * price_out) / 1_000_000


def fmt(value, digits: int = 3) -> str:
    if value is None:
        return "-"
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value)


def _target(name: str, value) -> str:
    if name not in TARGETS:
        return ""
    op, goal = TARGETS[name]
    if value is None:
        return f"{op} {goal}"
    ok = value >= goal if op == ">=" else value <= goal
    return f"{op} {goal} {'(met)' if ok else '(MISSED)'}"


def run_log_row(m: dict) -> str:
    """A row for docs/run-log.md; Who, System / change and Takeaway are left for a person to fill in."""
    commit = (m.get("git_commit") or "")[:7] + ("*" if m.get("git_dirty") else "")
    cost = m.get("cost_per_case_usd")
    cells = [
        (m.get("created_utc") or "")[:10], m.get("run", ""), "", m.get("system", ""),
        f"{m.get('split', '')} ({m.get('cases', 0)})",
        fmt(m.get("gap_recall"), 2), fmt(m.get("gap_precision"), 2), fmt(m.get("status_macro_f1"), 2),
        fmt(m.get("false_missing_rate"), 2), fmt(m.get("citation_precision"), 2),
        f"${cost:.3f}" if cost is not None else "-", f"{m.get('mean_latency_s', 0.0):.1f} s", commit, "",
    ]
    return "| " + " | ".join(cells) + " |"


def render_report(m: dict) -> str:
    ci = m.get("ci95") or {}
    commit = m.get("git_commit") or "unknown"
    lines = [
        f"# Run `{m.get('run', '')}`",
        "",
        f"- System: **{m.get('system')}** on split **{m.get('split')}** ({m.get('cases')} cases)",
        f"- Model: {m.get('model') or 'none'}",
        f"- Commit: `{commit[:12]}`{' (uncommitted changes)' if m.get('git_dirty') else ''}",
        f"- Created: {m.get('created_utc', '')}",
    ]
    if m.get("config"):
        lines.append("- Config: " + ", ".join(f"{k}={v}" for k, v in m["config"].items()))
    lines += ["", "## Headline metrics", "", "95% CIs: bootstrap over cases (2000 resamples).", "",
              "| Metric | Value | 95% CI | Target |", "| --- | --- | --- | --- |"]
    for name in HEADLINE:
        if name in m:
            lo_hi = ci.get(name)
            interval = f"[{fmt(lo_hi[0])}, {fmt(lo_hi[1])}]" if lo_hi else ""
            lines.append(f"| {name} | {fmt(m[name])} | {interval} | {_target(name, m[name])} |")

    lines += ["", "## Cost and latency", "", "| Metric | Value | Target |", "| --- | --- | --- |"]
    for name in ("llm_calls", "input_tokens", "output_tokens", "cost_usd", "cost_per_case_usd",
                 "mean_latency_s", "max_latency_s"):
        if name in m:
            lines.append(f"| {name} | {fmt(m[name], 4)} | {_target(name, m[name])} |")
    if m.get("system") != "b0" and m.get("cost_usd") is None:
        lines += ["", "Cost not reported: set LLM_PRICE_INPUT_PER_MTOK and LLM_PRICE_OUTPUT_PER_MTOK in .env."]

    recall = [k for k in m if k.startswith("recall@")]
    if recall:
        lines += ["", "## Retrieval (first pass)", "", "| " + " | ".join(recall) + " |",
                  "| " + " | ".join("---" for _ in recall) + " |",
                  "| " + " | ".join(fmt(m[k]) for k in recall) + " |"]

    if m.get("slices"):
        lines += ["", "## Slices", "",
                  "Case-level tags from gold.json; every item in a tagged case counts toward its slice.", "",
                  "| Slice | Cases | Items | Status acc | Gap R | Gap P | False-Missing | Cite P |",
                  "| --- | --- | --- | --- | --- | --- | --- | --- |"]
        for row in m["slices"]:
            lines.append(
                f"| {row['slice']} | {row['cases']} | {row['items']} | {fmt(row['status_accuracy'])} | "
                f"{fmt(row['gap_recall'])} | {fmt(row['gap_precision'])} | {fmt(row['false_missing_rate'])} | "
                f"{fmt(row['citation_precision'])} |"
            )

    if "error_attribution" in m:
        attr = m["error_attribution"]
        wrong = sum(attr.values())
        lines += ["", "## Error attribution", ""]
        if wrong:
            lines += [f"{wrong} wrong items; examples in errors.md.", "",
                      "| Cause | Items | Share |", "| --- | --- | --- |"]
            lines += [f"| {cause} | {n} | {n / wrong:.0%} |" for cause, n in attr.items() if n]
        else:
            lines.append("No wrong items.")

    lines += ["", "## Run-log row", "", "Paste into docs/run-log.md and fill in Who, System / change and Takeaway:",
              "", "```", run_log_row(m), "```", ""]
    return "\n".join(lines)
