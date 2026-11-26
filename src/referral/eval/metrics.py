"""Outcome and trajectory metrics from the Milestone 2 plan."""
import random
from collections import Counter

from rapidfuzz import fuzz

from referral.eval.gold import GoldCase, GoldEvidence
from referral.schemas import GAP_STATUSES, CaseResult, Chunk, Status


def pairs(golds: list[GoldCase], preds: list[CaseResult]):
    """Aligned (gold item, predicted item, case) triples. A missing prediction counts as NEEDS_REVIEW."""
    by_case = {p.case_id: p for p in preds}
    for g in golds:
        pred_items = {r.item_id: r for r in by_case[g.case_id].items} if g.case_id in by_case else {}
        for gi in g.items:
            yield gi, pred_items.get(gi.item_id), g


def gap_prf(golds: list[GoldCase], preds: list[CaseResult]) -> dict[str, float]:
    """Positive class = Missing or Contradictory (what staff must chase)."""
    tp = fp = fn = 0
    for gi, pi, _ in pairs(golds, preds):
        gold_pos = gi.status in GAP_STATUSES
        pred_pos = pi is not None and pi.status in GAP_STATUSES
        tp += gold_pos and pred_pos
        fp += pred_pos and not gold_pos
        fn += gold_pos and not pred_pos
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {"gap_precision": precision, "gap_recall": recall, "gap_f1": f1}


def status_macro_f1(golds: list[GoldCase], preds: list[CaseResult]) -> float:
    tp, fp, fn = Counter(), Counter(), Counter()
    labels = set()
    for gi, pi, _ in pairs(golds, preds):
        pred = pi.status if pi else Status.NEEDS_REVIEW
        labels |= {gi.status, pred}
        if pred == gi.status:
            tp[pred] += 1
        else:
            fp[pred] += 1
            fn[gi.status] += 1
    f1s = []
    for lab in labels:
        p = tp[lab] / (tp[lab] + fp[lab]) if tp[lab] + fp[lab] else 0.0
        r = tp[lab] / (tp[lab] + fn[lab]) if tp[lab] + fn[lab] else 0.0
        f1s.append(2 * p * r / (p + r) if p + r else 0.0)
    return sum(f1s) / len(f1s) if f1s else 0.0


def status_accuracy(golds: list[GoldCase], preds: list[CaseResult]) -> float:
    total = right = 0
    for gi, pi, _ in pairs(golds, preds):
        total += 1
        right += (pi.status if pi else Status.NEEDS_REVIEW) == gi.status
    return right / total if total else 0.0


def false_missing_rate(golds: list[GoldCase], preds: list[CaseResult]) -> float:
    """Share of items that are really present (in some form) but predicted Missing."""
    present_like = {Status.PRESENT, Status.STALE, Status.CONTRADICTORY}
    total = wrong = 0
    for gi, pi, _ in pairs(golds, preds):
        if gi.status in present_like:
            total += 1
            wrong += pi is not None and pi.status == Status.MISSING
    return wrong / total if total else 0.0


def false_contradictory_rate(golds: list[GoldCase], preds: list[CaseResult], slice_name: str) -> float:
    """E.g. slice_name='wrong_patient': how often a merged foreign document creates a fake conflict."""
    total = wrong = 0
    for gi, pi, g in pairs(golds, preds):
        if slice_name in g.slices and gi.status != Status.CONTRADICTORY:
            total += 1
            wrong += pi is not None and pi.status == Status.CONTRADICTORY
    return wrong / total if total else 0.0


def _evidence_hit(text: str, gold: GoldEvidence, threshold: int = 85) -> bool:
    return fuzz.partial_ratio(gold.text.lower(), text.lower()) >= threshold


def citation_precision(golds: list[GoldCase], preds: list[CaseResult]) -> float:
    """Cited quotes that match a planted gold evidence sentence for that item."""
    total = good = 0
    for gi, pi, _ in pairs(golds, preds):
        if not pi:
            continue
        for c in pi.citations:
            total += 1
            good += any(c.doc_id == ge.doc_id and _evidence_hit(c.quote, ge) for ge in gi.evidence)
    return good / total if total else 0.0


def citation_levels(golds: list[GoldCase], preds: list[CaseResult]) -> dict[str, float]:
    """How close each citation lands to the gold evidence, at three levels of strictness.

    document  the cited document holds gold evidence for that item
    page      the cited document and page hold gold evidence for that item
    passage   the quote itself matches a gold evidence passage (this is `citation_precision`)
    All three share one denominator, every citation made, so the gaps between them show where citations
    go wrong: right document but wrong page, or right page but a header or date line instead of the value.
    """
    by_case = {p.case_id: p for p in preds}
    m = metrics_from_counts(_total([case_counts(g, by_case.get(g.case_id)) for g in golds]))
    return {"citation_doc_precision": m["citation_doc_precision"],
            "citation_page_precision": m["citation_page_precision"]}


def recall_at_k(gold_evidence: list[GoldEvidence], retrieved: list[Chunk]) -> bool:
    """Trajectory: did retrieval surface at least one gold evidence passage?"""
    return any(ch.doc_id == ge.doc_id and _evidence_hit(ch.text, ge) for ge in gold_evidence for ch in retrieved)


def retrieval_recall(golds: list[GoldCase], traces: dict, chunks_by_id: dict[str, Chunk],
                     ks: tuple[int, ...] = (3, 5, 10)) -> dict[str, float]:
    """Trajectory: share of items with gold evidence whose evidence is in the first-pass top k.

    traces maps case_id -> CaseTrace; chunks_by_id must cover every chunk id the traces mention.
    """
    hits = {k: 0 for k in ks}
    total = 0
    for g in golds:
        trace = traces.get(g.case_id)
        if trace is None:
            continue
        by_item = {t.item_id: t for t in trace.items}
        for gi in g.items:
            if not gi.evidence or gi.item_id not in by_item:
                continue
            total += 1
            ranked = [chunks_by_id[c] for c in by_item[gi.item_id].ranked if c in chunks_by_id]
            for k in ks:
                hits[k] += recall_at_k(gi.evidence, ranked[:k])
    return {f"recall@{k}": hits[k] / total if total else 0.0 for k in ks}


def foreign_citation_rate(preds: list[CaseResult]) -> float:
    """E2: share of citations that point at another case's chunk (chunk ids are '<case_id>/...')."""
    total = foreign = 0
    for p in preds:
        for item in p.items:
            for c in item.citations:
                total += 1
                foreign += not c.chunk_id.startswith(f"{p.case_id}/")
    return foreign / total if total else 0.0


def overturn_counts(preds: list[CaseResult]) -> dict[str, int]:
    """How often each absence stage rescued an item from a wrong Missing."""
    flags = Counter(f for p in preds for item in p.items for f in item.flags)
    return {name: flags[name] for name in ("absence_check_overturned", "long_context_overturned",
                                           "long_context_rejected", "conflict_checker")}


def bootstrap_ci(per_case_values: list[float], n: int = 2000, seed: int = 0) -> tuple[float, float]:
    rng = random.Random(seed)
    means = sorted(
        sum(rng.choice(per_case_values) for _ in per_case_values) / len(per_case_values) for _ in range(n)
    )
    return means[int(0.025 * n)], means[int(0.975 * n)]


# --- Case-level counts: the same metrics as above, built from per-case tallies that can be summed. ---
# Bootstrap CIs and slice tables resample or filter cases and re-aggregate these tallies, so the
# fuzzy citation matching runs once per case instead of once per resample.

PRESENT_LIKE = {Status.PRESENT, Status.STALE, Status.CONTRADICTORY}
CI_METRICS = ("gap_recall", "gap_precision", "gap_f1", "status_macro_f1", "status_accuracy",
              "citation_precision", "citation_doc_precision", "citation_page_precision", "false_missing_rate")
# Slices from the dataset plan; always listed so tables line up across runs. Others found in gold are appended.
REPORT_SLICES = ("buried", "contradiction", "stale", "wrong_patient", "abbreviation")


def case_counts(gold: GoldCase, pred: CaseResult | None) -> Counter:
    """Tallies for one case. A missing prediction counts as NEEDS_REVIEW, as in `pairs`."""
    c: Counter = Counter()
    pred_items = {r.item_id: r for r in pred.items} if pred else {}
    for gi in gold.items:
        pi = pred_items.get(gi.item_id)
        ps = pi.status if pi else Status.NEEDS_REVIEW
        c["items"] += 1
        c["correct"] += ps == gi.status
        gold_pos, pred_pos = gi.status in GAP_STATUSES, ps in GAP_STATUSES
        c["gap_tp"] += gold_pos and pred_pos
        c["gap_fp"] += pred_pos and not gold_pos
        c["gap_fn"] += gold_pos and not pred_pos
        c[("label", gi.status)] += 1
        c[("label", ps)] += 1
        if ps == gi.status:
            c[("tp", ps)] += 1
        else:
            c[("fp", ps)] += 1
            c[("fn", gi.status)] += 1
        if gi.status in PRESENT_LIKE:
            c["fm_total"] += 1
            c["fm_wrong"] += ps == Status.MISSING
        if "wrong_patient" in gold.slices and gi.status != Status.CONTRADICTORY:
            c["fc_total"] += 1
            c["fc_wrong"] += ps == Status.CONTRADICTORY
        for cit in pi.citations if pi else []:
            c["cite_total"] += 1
            c["cite_good"] += any(cit.doc_id == ge.doc_id and _evidence_hit(cit.quote, ge) for ge in gi.evidence)
            # doc_ids repeat across cases ("lab_report"), so another patient's document never counts (E2).
            own = cit.chunk_id.startswith(f"{gold.case_id}/")
            c["cite_doc"] += own and any(cit.doc_id == ge.doc_id for ge in gi.evidence)
            c["cite_page"] += own and any(cit.doc_id == ge.doc_id and cit.page == ge.page for ge in gi.evidence)
    if pred:
        for item in pred.items:
            for cit in item.citations:
                c["foreign_total"] += 1
                c["foreign"] += not cit.chunk_id.startswith(f"{pred.case_id}/")
    return c


def metrics_from_counts(c: Counter, undefined: float | None = 0.0) -> dict[str, float | None]:
    """Metrics from summed tallies. `undefined` is returned for a rate whose denominator is zero."""
    def ratio(a: str, b: str) -> float | None:
        return c[a] / c[b] if c[b] else undefined

    tp, fp, fn = c["gap_tp"], c["gap_fp"], c["gap_fn"]
    precision = tp / (tp + fp) if tp + fp else undefined
    recall = tp / (tp + fn) if tp + fn else undefined
    p, r = precision or 0.0, recall or 0.0
    if p + r:
        gap_f1 = 2 * p * r / (p + r)
    else:
        gap_f1 = undefined if precision is None and recall is None else 0.0
    f1s = []
    for lab in {k[1] for k in c if isinstance(k, tuple) and k[0] == "label" and c[k]}:
        lp = c[("tp", lab)] / (c[("tp", lab)] + c[("fp", lab)]) if c[("tp", lab)] + c[("fp", lab)] else 0.0
        lr = c[("tp", lab)] / (c[("tp", lab)] + c[("fn", lab)]) if c[("tp", lab)] + c[("fn", lab)] else 0.0
        f1s.append(2 * lp * lr / (lp + lr) if lp + lr else 0.0)
    return {
        "gap_precision": precision,
        "gap_recall": recall,
        "gap_f1": gap_f1,
        "status_macro_f1": sum(f1s) / len(f1s) if f1s else undefined,
        "status_accuracy": ratio("correct", "items"),
        "citation_precision": ratio("cite_good", "cite_total"),
        "citation_doc_precision": ratio("cite_doc", "cite_total"),
        "citation_page_precision": ratio("cite_page", "cite_total"),
        "false_missing_rate": ratio("fm_wrong", "fm_total"),
        "false_contradictory_wrong_patient": ratio("fc_wrong", "fc_total"),
        "foreign_citation_rate": ratio("foreign", "foreign_total"),
    }


def _total(counts: list[Counter]) -> Counter:
    total: Counter = Counter()
    for c in counts:
        total.update(c)
    return total


def bootstrap_cis(golds: list[GoldCase], preds: list[CaseResult], metrics: tuple[str, ...] = CI_METRICS,
                  n: int = 2000, seed: int = 0) -> dict[str, list[float] | None]:
    """95% percentile bootstrap CIs, resampling cases (not items, which are correlated within a case).

    Resamples where a metric is undefined (e.g. no gold gaps drawn) are skipped for that metric.
    """
    by_case = {p.case_id: p for p in preds}
    per_case = [case_counts(g, by_case.get(g.case_id)) for g in golds]
    if not per_case:
        return {m: None for m in metrics}
    rng = random.Random(seed)
    draws: dict[str, list[float]] = {m: [] for m in metrics}
    for _ in range(n):
        sample = metrics_from_counts(_total([rng.choice(per_case) for _ in per_case]), undefined=None)
        for m in metrics:
            if sample[m] is not None:
                draws[m].append(sample[m])
    out: dict[str, list[float] | None] = {}
    for m, vals in draws.items():
        vals.sort()
        out[m] = [vals[int(0.025 * len(vals))], vals[min(int(0.975 * len(vals)), len(vals) - 1)]] if vals else None
    return out


def slice_table(golds: list[GoldCase], preds: list[CaseResult]) -> list[dict]:
    """One row per slice (cases tagged with it in gold.json), plus 'all'. Undefined rates are None."""
    by_case = {p.case_id: p for p in preds}
    per_case = {g.case_id: case_counts(g, by_case.get(g.case_id)) for g in golds}
    extra = sorted({s for g in golds for s in g.slices} - set(REPORT_SLICES))
    rows = []
    for name in ("all", *REPORT_SLICES, *extra):
        members = [g for g in golds if name == "all" or name in g.slices]
        total = _total([per_case[g.case_id] for g in members])
        m = metrics_from_counts(total, undefined=None)
        rows.append({"slice": name, "cases": len(members), "items": total["items"],
                     **{k: m[k] for k in ("status_accuracy", "gap_recall", "gap_precision",
                                          "false_missing_rate", "citation_precision")}})
    return rows


def summarize(golds: list[GoldCase], preds: list[CaseResult]) -> dict[str, float]:
    return {
        **gap_prf(golds, preds),
        "status_macro_f1": status_macro_f1(golds, preds),
        "status_accuracy": status_accuracy(golds, preds),
        "citation_precision": citation_precision(golds, preds),
        **citation_levels(golds, preds),
        "false_missing_rate": false_missing_rate(golds, preds),
        "false_contradictory_wrong_patient": false_contradictory_rate(golds, preds, "wrong_patient"),
        "foreign_citation_rate": foreign_citation_rate(preds),
        **overturn_counts(preds),
    }
