"""Step 2 of the generator: decide which defects to plant in a case.

    python datagen/defect_plan.py --checklist cardiology-v1 --n 5

Each defect only goes to items where it can exist (see `eligible`): Stale needs a recency window,
Contradict needs a value two documents can disagree on, and so on. Each plan also decides whether every
conditional item's condition holds (e.g. "patient is on warfarin"). When it does not, the item is
not_applicable and gets no defect.
"""
import argparse
import json
import random
import re
from pathlib import Path

from pydantic import BaseModel

DEFECTS = ["omit", "contradict", "stale", "bury", "abbreviate"]

# Target share of cases containing each property (Milestone 2 dataset mix).
MIX = {"complete": 0.30, "omit": 0.40, "abbreviate": 0.25, "contradict": 0.20,
       "bury": 0.20, "stale": 0.15, "wrong_patient": 0.05}

CONDITION_RATE = 0.30  # share of cases in which a conditional item's condition holds

ABBREVIATION_RE = re.compile(r"^[A-Z][A-Z0-9/\-]{1,5}$")  # e.g. ECG, TTE, LDL, PT/INR


class ItemPlan(BaseModel):
    item_id: str
    defect: str | None = None  # None = present and clean


class DefectPlan(BaseModel):
    case_id: str
    checklist_id: str
    items: list[ItemPlan]
    wrong_patient_doc: bool = False
    condition_holds: dict[str, bool] = {}  # conditional item_id -> does its condition hold for this patient


def eligible(defect: str, item: dict) -> bool:
    """Can this defect be planted in this checklist item?"""
    if defect == "omit":
        return True
    if defect == "stale":
        return item.get("recency_days") is not None
    if defect == "contradict":  # needs a value (dose, lab result, measurement) that two documents can disagree on
        return item["evidence_type"] in {"lab", "imaging", "medication_list"}
    if defect == "bury":  # a result or list that can sit deep inside a long narrative document
        return item["evidence_type"] in {"lab", "test", "imaging", "medication_list"}
    if defect == "abbreviate":  # needs an abbreviation to write the evidence in
        return any(ABBREVIATION_RE.match(s) for s in item.get("synonyms", []))
    raise ValueError(f"unknown defect {defect!r}")


def make_plan(case_id: str, checklist: dict, rng: random.Random) -> DefectPlan:
    items = {i["item_id"]: i for i in checklist["items"]}
    holds = {i: rng.random() < CONDITION_RATE for i, item in items.items() if item.get("condition")}
    applicable = [i for i in items if holds.get(i, True)]
    plans = {i: ItemPlan(item_id=i) for i in items}
    if rng.random() >= MIX["complete"]:
        if rng.random() < MIX["omit"] / (1 - MIX["complete"]):
            for i in rng.sample(applicable, k=min(len(applicable), rng.randint(1, 3))):
                plans[i].defect = "omit"
        for defect in ["contradict", "stale", "bury", "abbreviate"]:
            free = [i for i in applicable if plans[i].defect is None and eligible(defect, items[i])]
            if free and rng.random() < MIX[defect] / (1 - MIX["complete"]):
                plans[rng.choice(free)].defect = defect
    return DefectPlan(case_id=case_id, checklist_id=checklist["checklist_id"], items=list(plans.values()),
                      wrong_patient_doc=rng.random() < MIX["wrong_patient"], condition_holds=holds)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--checklist", required=True)
    ap.add_argument("--n", type=int, default=5)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    checklist = json.loads(Path(f"data/checklists/{args.checklist}.json").read_text())
    rng = random.Random(args.seed)
    for n in range(args.n):
        print(make_plan(f"{args.checklist}-{n:03d}", checklist, rng).model_dump_json())
