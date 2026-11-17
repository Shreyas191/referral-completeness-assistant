"""Outcome and trajectory metrics from the Milestone 2 plan."""
import random
from collections import Counter

from rapidfuzz import fuzz

from referral.eval.gold import GoldCase, GoldEvidence
from referral.schemas import GAP_STATUSES, CaseResult, Chunk, Citation, Status


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


def recall_at_k(gold_evidence: list[GoldEvidence], retrieved: list[Chunk]) -> bool:
    """Trajectory: did retrieval surface at least one gold evidence passage?"""
    return any(ch.doc_id == ge.doc_id and _evidence_hit(ch.text, ge) for ge in gold_evidence for ch in retrieved)


def attribute_error(gold_evidence: list[GoldEvidence], retrieved: list[Chunk], citations: list[Citation],
                    flags: list[str]) -> str:
    """Label one wrong item: retrieval_miss | validator_reject | reasoning_error."""
    if gold_evidence and not recall_at_k(gold_evidence, retrieved):
        return "retrieval_miss"
    if any(f.startswith("citation_failed") for f in flags):
        return "validator_reject"
    return "reasoning_error"


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
    return {name: flags[name] for name in ("absence_check_overturned", "long_context_overturned", "conflict_checker")}


def bootstrap_ci(per_case_values: list[float], n: int = 2000, seed: int = 0) -> tuple[float, float]:
    rng = random.Random(seed)
    means = sorted(
        sum(rng.choice(per_case_values) for _ in per_case_values) / len(per_case_values) for _ in range(n)
    )
    return means[int(0.025 * n)], means[int(0.975 * n)]


def summarize(golds: list[GoldCase], preds: list[CaseResult]) -> dict[str, float]:
    return {
        **gap_prf(golds, preds),
        "status_macro_f1": status_macro_f1(golds, preds),
        "citation_precision": citation_precision(golds, preds),
        "false_missing_rate": false_missing_rate(golds, preds),
        "false_contradictory_wrong_patient": false_contradictory_rate(golds, preds, "wrong_patient"),
        "foreign_citation_rate": foreign_citation_rate(preds),
        **overturn_counts(preds),
    }
