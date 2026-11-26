"""E4: accuracy by where the evidence sits in the packet, and by packet length. No LLM calls.

For every gold item with evidence, the evidence's packet page is its page counted across the documents in the
order the run read them (file names, reversed for --doc-order reverse runs), and its depth is that page over the
packet's page count. Items without evidence (Missing, Not applicable) have no position and are left out of the
position table; they still count in the packet-length table.

    python -m referral.eval.position runs/b1-hard runs/pipe-hard --cases-dir data/hard_cases
"""
import argparse
from pathlib import Path

from referral.eval.compare import load_run
from referral.eval.gold import GoldCase, load_gold
from referral.ingest.pdf_loader import load_case
from referral.schemas import Status

PAGE_BINS = [(1, 5), (6, 15), (16, 30), (31, 1000)]
DEPTH_BINS = [(0.0, 0.25), (0.25, 0.5), (0.5, 0.75), (0.75, 1.01)]
LENGTH_BINS = [(1, 20), (21, 40), (41, 1000)]


def evidence_positions(case_dir: Path, gold: GoldCase,
                       reverse: bool = False) -> tuple[int, dict[str, tuple[int, float]]]:
    """(packet pages, {item_id: (deepest evidence page, depth)}). Deepest, because an item is only fully seen
    once its last piece of evidence is read; for a contradiction that is the second document."""
    offset, starts = 0, {}
    for doc in load_case(case_dir, reverse=reverse):
        starts[doc.doc_id] = offset
        offset += len(doc.pages)
    out = {}
    for gi in gold.items:
        pages = [starts[e.doc_id] + e.page for e in gi.evidence if e.doc_id in starts]
        if pages:
            out[gi.item_id] = (max(pages), max(pages) / offset)
    return offset, out


def _bin(value, bins):
    return next((lo, hi) for lo, hi in bins if lo <= value <= hi or (isinstance(lo, float) and lo <= value < hi))


def table(run_dirs: list[Path], cases_dir: Path) -> str:
    loaded = {d.name: load_run(d) for d in run_dirs}
    runs = {name: preds for name, (_, preds) in loaded.items()}
    reverse = {name: metrics.get("doc_order") == "reverse" for name, (metrics, _) in loaded.items()}
    case_ids = sorted(set.intersection(*(set(p) for p in runs.values())))
    rows: dict[tuple, dict[str, list[int]]] = {}

    def count(key, run, ok):
        cell = rows.setdefault(key, {}).setdefault(run, [0, 0])
        cell[0] += ok
        cell[1] += 1

    for cid in case_ids:
        gold = load_gold(cases_dir / cid)
        for run, preds in runs.items():
            n_pages, pos = evidence_positions(cases_dir / cid, gold, reverse[run])
            items = {r.item_id: r.status for r in preds[cid].items}
            for gi in gold.items:
                ok = items.get(gi.item_id, Status.NEEDS_REVIEW) == gi.status
                count(("length", _bin(n_pages, LENGTH_BINS)), run, ok)
                if gi.item_id in pos:
                    page, depth = pos[gi.item_id]
                    count(("page", _bin(page, PAGE_BINS)), run, ok)
                    count(("depth", _bin(depth, DEPTH_BINS)), run, ok)
    names = list(runs)
    out = [f"{len(case_ids)} cases in common. Each cell: items correct / items (accuracy).", ""]
    titles = {"page": "Evidence packet page (deepest piece)", "depth": "Evidence depth (page / packet pages)",
              "length": "Packet length in pages (all items)"}
    for kind, bins in (("page", PAGE_BINS), ("depth", DEPTH_BINS), ("length", LENGTH_BINS)):
        out += [f"### {titles[kind]}", "", "| Bin | " + " | ".join(names) + " |", "| --- " * (len(names) + 1) + "|"]
        for b in bins:
            if (kind, b) not in rows:
                continue
            label = f"{b[0]:.2f}-{min(b[1], 1.0):.2f}" if kind == "depth" else (
                f"{b[0]}+" if b[1] >= 1000 else f"{b[0]}-{b[1]}")
            cells = []
            for name in names:
                ok, n = rows[(kind, b)].get(name, [0, 0])
                cells.append(f"{ok}/{n} ({ok / n:.3f})" if n else "-")
            out.append(f"| {label} | " + " | ".join(cells) + " |")
        out.append("")
    return "\n".join(out)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("runs", type=Path, nargs="+")
    ap.add_argument("--cases-dir", type=Path, default=Path("data/cases"))
    args = ap.parse_args()
    print(table(args.runs, args.cases_dir))


if __name__ == "__main__":
    main()
