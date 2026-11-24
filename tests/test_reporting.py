"""Run reporting (#13) and error attribution (#14), on hand-built gold, predictions and traces."""
from datetime import date

import pytest
from conftest import make_chunk

from referral.eval.compare import mcnemar_exact
from referral.eval.errors import attribute_item
from referral.eval.gold import GoldCase, GoldEvidence, GoldItem
from referral.eval.metrics import (
    REPORT_SLICES,
    _total,
    bootstrap_cis,
    case_counts,
    metrics_from_counts,
    slice_table,
    summarize,
)
from referral.eval.report import cost_usd, render_report, run_info, run_log_row
from referral.pipeline.trace import CaseTrace, ItemTrace
from referral.schemas import CaseResult, Citation, ItemResult, Status

S = Status


def gold(case_id, statuses, slices=(), evidence=None):
    items = [GoldItem(item_id=f"i{n}", status=s, evidence=(evidence or {}).get(f"i{n}", []))
             for n, s in enumerate(statuses)]
    return GoldCase(case_id=case_id, checklist_id="cardiology-v1", reference_date=date(2026, 11, 1), split="dev",
                    slices=list(slices), items=items)


def pred(case_id, statuses, citations=None):
    return CaseResult(case_id=case_id, checklist_id="cardiology-v1", items=[
        ItemResult(item_id=f"i{n}", status=s, citations=(citations or {}).get(f"i{n}", []))
        for n, s in enumerate(statuses)])


EV = GoldEvidence(doc_id="labs", page=1, text="creatinine 1.1 mg/dL")
GOOD = Citation(chunk_id="c0/labs:p1:0", doc_id="labs", page=1, quote="creatinine 1.1 mg/dL")
FOREIGN = Citation(chunk_id="c9/labs:p1:0", doc_id="labs", page=1, quote="LDL 98")

GOLDS = [
    gold("c0", [S.PRESENT, S.MISSING, S.STALE], ["buried"], {"i0": [EV]}),
    gold("c1", [S.CONTRADICTORY, S.PRESENT, S.NOT_APPLICABLE], ["contradiction", "wrong_patient"]),
    gold("c2", [S.PRESENT, S.PRESENT, S.MISSING], ["stale", "rare_slice"]),
    gold("c3", [S.MISSING, S.PRESENT, S.PRESENT]),
]
PREDS = [
    pred("c0", [S.PRESENT, S.MISSING, S.PRESENT], {"i0": [GOOD, FOREIGN]}),
    pred("c1", [S.CONTRADICTORY, S.CONTRADICTORY, S.NOT_APPLICABLE]),
    pred("c2", [S.MISSING, S.PRESENT, S.PRESENT]),
    # c3 has no prediction at all: every item counts as Needs review
]


def test_count_based_metrics_match_summarize():
    from_counts = metrics_from_counts(_total([case_counts(g, p) for g, p in
                                              zip(GOLDS, [*PREDS, None])]))
    direct = summarize(GOLDS, PREDS)
    for key, value in from_counts.items():
        assert value == pytest.approx(direct[key]), key


def test_bootstrap_cis_bracket_the_point_estimate_and_are_reproducible():
    ci = bootstrap_cis(GOLDS, PREDS, n=500)
    point = summarize(GOLDS, PREDS)
    for name, (lo, hi) in ci.items():
        assert lo <= point[name] + 1e-9 and point[name] - 1e-9 <= hi, name
    assert ci == bootstrap_cis(GOLDS, PREDS, n=500)


def test_bootstrap_skips_resamples_where_a_metric_is_undefined():
    golds = [gold("a", [S.PRESENT]), gold("b", [S.PRESENT])]  # no gold gaps anywhere
    ci = bootstrap_cis(golds, [pred("a", [S.PRESENT]), pred("b", [S.PRESENT])], n=200)
    assert ci["gap_recall"] is None and ci["status_accuracy"] == [1.0, 1.0]


def test_slice_table_lists_planned_slices_even_when_empty():
    rows = {r["slice"]: r for r in slice_table(GOLDS, PREDS)}
    assert list(rows)[: 1 + len(REPORT_SLICES)] == ["all", *REPORT_SLICES]
    assert rows["all"]["cases"] == 4 and rows["all"]["items"] == 12
    assert rows["abbreviation"]["cases"] == 0 and rows["abbreviation"]["status_accuracy"] is None
    assert rows["contradiction"]["gap_recall"] == 1.0 and rows["contradiction"]["gap_precision"] == 0.5
    assert rows["rare_slice"]["cases"] == 1  # slices outside the plan are still reported


def test_mcnemar_exact():
    assert mcnemar_exact(0, 0) == 1.0
    assert mcnemar_exact(0, 6) == pytest.approx(2 / 64)
    assert mcnemar_exact(2, 7) == mcnemar_exact(7, 2)


def test_cost_needs_both_prices():
    assert cost_usd(1_000_000, 100_000, None, 5.0) is None
    assert cost_usd(1_000_000, 100_000, 1.0, 5.0) == pytest.approx(1.5)


def test_run_info_records_the_commit():
    info = run_info()
    assert len(info["git_commit"]) == 40 and isinstance(info["git_dirty"], bool)


def test_report_and_run_log_row_render_without_optional_fields():
    m = {"run": "x", "system": "b0", "split": "dev", "cases": 2, "gap_recall": 0.5, "mean_latency_s": 0.1,
         "cost_per_case_usd": 0.0, "error_attribution": {"retrieval_miss": 0}}
    assert "No wrong items." in render_report(m)
    assert run_log_row(m).count("|") == 15


# --- error attribution on single items ---

CHUNKS = {
    "c0/labs:p1:0": make_chunk("BMP: creatinine 1.1 mg/dL", doc_id="labs", case_id="c0"),
    "c0/note:p1:0": make_chunk("Patient discussed diet.", doc_id="note", case_id="c0"),
}
for cid, ch in CHUNKS.items():
    ch.chunk_id = cid


def attribute(gold_status, pred_status, shown, ranked=None, flags=(), citations=(), quarantined=(), **trace_kw):
    g = gold("c0", [gold_status], evidence={"i0": [EV]} if gold_status != S.MISSING else {})
    p = ItemResult(item_id="i0", status=pred_status, flags=list(flags), citations=list(citations))
    t = ItemTrace(item_id="i0", ranked=ranked or shown, shown=shown, **trace_kw)
    trace = CaseTrace(case_id="c0", quarantined_docs=list(quarantined), items=[t])
    return attribute_item(g, g.items[0], p, t, trace, CHUNKS)


def test_evidence_never_shown_is_a_retrieval_miss():
    r = attribute(S.PRESENT, S.MISSING, shown=["c0/note:p1:0"], ranked=["c0/note:p1:0", "c0/labs:p1:0"])
    assert r["cause"] == "retrieval_miss" and r["details"] == ["ranked_2_below_k_1"]


def test_long_context_read_counts_as_seeing_the_whole_case():
    r = attribute(S.PRESENT, S.MISSING, shown=["c0/note:p1:0"], long_context_status="missing")
    assert r["cause"] == "reasoning_error" and r["stage"] == "long_context"


def test_quarantined_evidence_doc():
    r = attribute(S.PRESENT, S.MISSING, shown=["c0/note:p1:0"], quarantined=["labs"])
    assert r["cause"] == "quarantined"


def test_failed_citation_is_a_validator_reject():
    r = attribute(S.PRESENT, S.NEEDS_REVIEW, shown=["c0/labs:p1:0"], flags=["citation_failed:c0/labs:p1:0"])
    assert r["cause"] == "validator_reject" and r["details"] == ["citation_failed"]


def test_present_vs_stale_is_a_recency_window_error():
    r = attribute(S.STALE, S.PRESENT, shown=["c0/labs:p1:0"], citations=[GOOD])
    assert r["cause"] == "recency_window"


def test_rule_based_conflict_is_charged_to_the_conflict_checker():
    r = attribute(S.PRESENT, S.CONTRADICTORY, shown=["c0/labs:p1:0"], flags=["conflict_checker"])
    assert r["cause"] == "conflict_checker"


def test_hallucinated_present_is_a_reasoning_error():
    r = attribute(S.MISSING, S.PRESENT, shown=["c0/note:p1:0"])
    assert r["cause"] == "reasoning_error" and r["details"] == ["missing->present"]


def test_citation_levels_separate_document_page_and_passage():
    # Gold evidence: the creatinine line on page 1 of "labs".
    header = Citation(chunk_id="c0/labs:p1:0", doc_id="labs", page=1, quote="BASIC METABOLIC PANEL")
    other_page = Citation(chunk_id="c0/labs:p2:0", doc_id="labs", page=2, quote="Date: 10/02/2026")
    other_doc = Citation(chunk_id="c0/note:p1:0", doc_id="note", page=1, quote="Labs reviewed.")
    g = [gold("c0", [S.PRESENT], evidence={"i0": [EV]})]
    p = [pred("c0", [S.PRESENT], {"i0": [GOOD, header, other_page, other_doc]})]
    m = summarize(g, p)
    assert m["citation_doc_precision"] == 0.75   # all but the other document
    assert m["citation_page_precision"] == 0.5   # the passage and the header share the gold page
    assert m["citation_precision"] == 0.25       # only the passage itself
    assert set(bootstrap_cis(g, p)) >= {"citation_doc_precision", "citation_page_precision"}


def test_citation_on_item_without_gold_evidence_fails_every_level():
    m = summarize([gold("c0", [S.MISSING])], [pred("c0", [S.PRESENT], {"i0": [GOOD]})])
    assert m["citation_doc_precision"] == m["citation_page_precision"] == m["citation_precision"] == 0.0


def test_another_patients_document_is_not_the_right_document():
    # Same doc_id and page as the gold evidence, but the chunk belongs to case c9.
    m = summarize([gold("c0", [S.PRESENT], evidence={"i0": [EV]})], [pred("c0", [S.PRESENT], {"i0": [FOREIGN]})])
    assert m["citation_doc_precision"] == m["citation_page_precision"] == 0.0
