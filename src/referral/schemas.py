"""Shared data models. Every component reads and writes these."""
from datetime import date
from enum import Enum

from pydantic import BaseModel, Field


class Status(str, Enum):
    PRESENT = "present"
    MISSING = "missing"
    CONTRADICTORY = "contradictory"
    STALE = "stale"                # evidence verified, date known and older than the window
    NEEDS_REVIEW = "needs_review"  # evidence could not be verified (bad citation, no/unreadable date)
    NOT_APPLICABLE = "not_applicable"  # conditional item whose condition is absent


GAP_STATUSES = {Status.MISSING, Status.CONTRADICTORY}


class ChecklistItem(BaseModel):
    item_id: str
    description: str
    evidence_type: str = Field(description="e.g. lab, imaging, note, medication_list")
    required: bool = True
    condition: str | None = Field(None, description="Only required when this holds, e.g. 'on warfarin'")
    recency_days: int | None = Field(None, description="Clinic-owned staleness window; None = no limit")
    synonyms: list[str] = []


class Checklist(BaseModel):
    checklist_id: str
    clinic: str
    specialty: str
    version: str
    items: list[ChecklistItem]


class Document(BaseModel):
    doc_id: str
    case_id: str
    doc_type: str = "unknown"
    doc_date: date | None = None
    pages: list[str]


class Chunk(BaseModel):
    chunk_id: str
    case_id: str
    doc_id: str
    doc_type: str
    doc_date: date | None
    page: int
    start: int
    end: int
    text: str


class Citation(BaseModel):
    chunk_id: str
    doc_id: str
    page: int
    quote: str
    doc_date: date | None = None


class ItemResult(BaseModel):
    item_id: str
    status: Status
    citations: list[Citation] = []
    rationale: str = ""
    flags: list[str] = []  # e.g. "citation_failed", "absence_check_overturned", "date_unreadable"


class FollowUp(BaseModel):
    item_id: str
    request: str


class CaseResult(BaseModel):
    case_id: str
    checklist_id: str
    referral_reason: str = ""
    items: list[ItemResult]
    quarantined_docs: list[str] = []  # wrong-patient or unidentifiable documents
    followups: list[FollowUp] = []
    summary: str = ""

    @property
    def complete(self) -> bool:
        return all(r.status in {Status.PRESENT, Status.NOT_APPLICABLE} for r in self.items)
