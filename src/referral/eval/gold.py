"""Gold labels written by the case generator (labels by construction, then audited)."""
from datetime import date
from pathlib import Path

from pydantic import BaseModel

from referral.schemas import Status


class GoldEvidence(BaseModel):
    doc_id: str
    page: int
    text: str  # the planted sentence; used to score citations and retrieval recall@k


class GoldItem(BaseModel):
    item_id: str
    status: Status
    evidence: list[GoldEvidence] = []


class GoldCase(BaseModel):
    case_id: str
    checklist_id: str
    reference_date: date
    split: str  # dev | test | adversarial
    slices: list[str] = []  # e.g. buried, contradiction, stale, wrong_patient, abbreviation
    wrong_patient_docs: list[str] = []
    items: list[GoldItem]


def load_gold(case_dir: Path) -> GoldCase:
    return GoldCase.model_validate_json((case_dir / "gold.json").read_text())
