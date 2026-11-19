"""Quota selection of defect plans (issue #10): small splits must still match the dataset mix."""
import json
from pathlib import Path

import pytest

from datagen.defect_plan import MIX, eligible, plan_features, quota_targets, select_plans

RAW = json.loads((Path(__file__).parents[1] / "data/checklists/cardiology-v1.json").read_text())
ITEMS = {i["item_id"]: i for i in RAW["items"]}


@pytest.mark.parametrize("n", [40, 110])
def test_selection_hits_every_target(n):
    plans, targets, achieved = select_plans(RAW, n, seed=10, prefix="cv-dev")
    assert achieved == targets == quota_targets(n)
    assert targets["wrong_patient"] >= 2  # the slice that random draws most often miss
    counted = {k: sum(k in plan_features(p) for p in plans) for k in MIX}
    assert counted == achieved
    assert [p.case_id for p in plans] == [f"cv-dev-{k:03d}" for k in range(n)]
    for p in plans:  # still valid plans
        assert all(eligible(i.defect, ITEMS[i.item_id]) for i in p.items if i.defect)


def test_selection_is_deterministic_and_seed_dependent():
    a, _, _ = select_plans(RAW, 40, seed=10, prefix="x")
    b, _, _ = select_plans(RAW, 40, seed=10, prefix="x")
    c, _, _ = select_plans(RAW, 40, seed=20, prefix="x")
    assert a == b and a != c
