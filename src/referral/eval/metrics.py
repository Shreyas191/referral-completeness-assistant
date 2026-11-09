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
    }
