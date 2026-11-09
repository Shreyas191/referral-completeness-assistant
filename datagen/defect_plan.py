"""Step 2 of the generator: decide which defects to plant in a case.

    python datagen/defect_plan.py --checklist cardiology-v1 --n 5
"""
import argparse
import json
import random
from pathlib import Path

from pydantic import BaseModel

DEFECTS = ["omit", "contradict", "stale", "bury", "abbreviate"]

# Target share of cases containing each property (Milestone 2 dataset mix).
MIX = {"complete": 0.30, "omit": 0.40, "abbreviate": 0.25, "contradict": 0.20,
       "bury": 0.20, "stale": 0.15, "wrong_patient": 0.05}


class ItemPlan(BaseModel):
    item_id: str
    defect: str | None = None  # None = present and clean


class DefectPlan(BaseModel):
    case_id: str
    checklist_id: str
    items: list[ItemPlan]
    wrong_patient_doc: bool = False


def make_plan(case_id: str, checklist: dict, rng: random.Random) -> DefectPlan:
    item_ids = [i["item_id"] for i in checklist["items"]]
    plans = {i: ItemPlan(item_id=i) for i in item_ids}
    if rng.random() >= MIX["complete"]:
        if rng.random() < MIX["omit"] / (1 - MIX["complete"]):
            for i in rng.sample(item_ids, k=rng.randint(1, 3)):
                plans[i].defect = "omit"
        for defect in ["contradict", "stale", "bury", "abbreviate"]:
            free = [i for i in item_ids if plans[i].defect is None]
            if free and rng.random() < MIX[defect] / (1 - MIX["complete"]):
                plans[rng.choice(free)].defect = defect
    return DefectPlan(case_id=case_id, checklist_id=checklist["checklist_id"], items=list(plans.values()),
                      wrong_patient_doc=rng.random() < MIX["wrong_patient"])


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
