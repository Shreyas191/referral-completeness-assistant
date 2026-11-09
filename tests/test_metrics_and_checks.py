import json
from datetime import date
from pathlib import Path

from conftest import make_chunk

from referral.eval.gold import GoldCase, GoldItem
from referral.eval.metrics import false_missing_rate, gap_prf, status_macro_f1
from referral.pipeline.absence import candidates_for_recheck
from referral.pipeline.conflicts import dose_conflicts
from referral.schemas import CaseResult, Checklist, ItemResult, Status

ROOT = Path(__file__).resolve().parents[1]


def gold(*statuses):
    return GoldCase(case_id="c1", checklist_id="x", reference_date=date(2026, 11, 1), split="dev",
                    items=[GoldItem(item_id=f"i{n}", status=s) for n, s in enumerate(statuses)])


def pred(*statuses):
    return CaseResult(case_id="c1", checklist_id="x",
                      items=[ItemResult(item_id=f"i{n}", status=s) for n, s in enumerate(statuses)])


def test_gap_metrics_and_false_missing():
    g = [gold(Status.MISSING, Status.PRESENT, Status.CONTRADICTORY, Status.PRESENT)]
    p = [pred(Status.MISSING, Status.MISSING, Status.PRESENT, Status.PRESENT)]
    m = gap_prf(g, p)
    assert m["gap_precision"] == 0.5 and m["gap_recall"] == 0.5
    assert false_missing_rate(g, p) == 1 / 3
    perfect = [pred(Status.MISSING, Status.PRESENT, Status.CONTRADICTORY, Status.PRESENT)]
    assert status_macro_f1(g, perfect) == 1.0


def test_absence_check_finds_abbreviation_missed_by_retrieval(bmp_item):
    hidden = make_chunk("Labs 10/1: BMP within normal limits", doc_id="labs", page=37)
    other = make_chunk("Patient reports fatigue.", doc_id="note")
    extra = candidates_for_recheck(bmp_item, first_pass=[other], wide_hits=[], all_chunks=[other, hidden])
    assert extra == [hidden]


def test_dose_conflict_across_documents():
    a = make_chunk("Metformin 500 mg twice daily", doc_id="letter")
    b = make_chunk("metformin 1000 mg BID", doc_id="meds")
    assert "metformin" in dose_conflicts([a, b])
    assert dose_conflicts([a, make_chunk("Metformin 500 mg", doc_id="meds")]) == {}


def test_example_checklist_is_valid():
    data = json.loads((ROOT / "data/checklists/cardiology-v1.json").read_text())
    assert Checklist.model_validate(data).items
