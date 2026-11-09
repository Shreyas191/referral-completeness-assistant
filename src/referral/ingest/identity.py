"""Identity check: keep other patients' documents out of the evidence pool.

Each document's name / DOB / MRN is compared with the referral letter. Mismatched or
unidentifiable documents are quarantined so they cannot produce false Contradictory labels.
"""
import re

from pydantic import BaseModel

from referral.schemas import Document


class PatientId(BaseModel):
    name: str | None = None
    dob: str | None = None
    mrn: str | None = None

    def known_fields(self) -> int:
        return sum(v is not None for v in (self.name, self.dob, self.mrn))


NAME_RE = re.compile(r"(?:Patient(?: Name)?|Name)\s*:\s*([A-Z][A-Za-z'\-]+(?:,? [A-Z][A-Za-z'\-]+)+)")
DOB_RE = re.compile(r"(?:DOB|Date of Birth)\s*:\s*(\d{1,2}/\d{1,2}/\d{4}|\d{4}-\d{2}-\d{2})")
MRN_RE = re.compile(r"MRN\s*[:#]?\s*([A-Z0-9\-]{4,})")


def extract_id(text: str) -> PatientId:
    def first(rx: re.Pattern) -> str | None:
        m = rx.search(text)
        return m.group(1).strip() if m else None

    name = first(NAME_RE)
    return PatientId(name=_normalize_name(name) if name else None, dob=first(DOB_RE), mrn=first(MRN_RE))


def _normalize_name(name: str) -> str:
    if "," in name:  # "Doe, Jane" -> "jane doe"
        last, first = [p.strip() for p in name.split(",", 1)]
        name = f"{first} {last}"
    return " ".join(name.lower().split())


def matches(reference: PatientId, candidate: PatientId) -> bool | None:
    """True = same patient, False = conflict on any shared field, None = nothing to compare."""
    shared = [
        (getattr(reference, f), getattr(candidate, f))
        for f in ("mrn", "dob", "name")
        if getattr(reference, f) and getattr(candidate, f)
    ]
    if not shared:
        return None
    return all(a == b for a, b in shared)


def quarantine(docs: list[Document]) -> tuple[list[Document], list[str]]:
    """Return (kept documents, quarantined doc_ids). The referral letter defines the patient."""
    reference_doc = next((d for d in docs if d.doc_type == "referral_letter"), docs[0] if docs else None)
    if reference_doc is None:
        return [], []
    reference = extract_id("\n".join(reference_doc.pages))
    kept, flagged = [], []
    for doc in docs:
        verdict = matches(reference, extract_id("\n".join(doc.pages)))
        if doc is reference_doc or verdict is True:
            kept.append(doc)
        else:  # mismatch or unidentifiable -> staff decide
            flagged.append(doc.doc_id)
    return kept, flagged
