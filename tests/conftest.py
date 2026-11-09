from datetime import date

import pytest

from referral.schemas import ChecklistItem, Chunk


def make_chunk(text, doc_id="labs", page=1, doc_date=date(2026, 10, 1), case_id="c1"):
    return Chunk(chunk_id=f"{doc_id}:p{page}:0", case_id=case_id, doc_id=doc_id, doc_type="lab_report",
                 doc_date=doc_date, page=page, start=0, end=len(text), text=text)


@pytest.fixture
def bmp_item():
    return ChecklistItem(item_id="bmp", description="basic metabolic panel", evidence_type="lab",
                         recency_days=90, synonyms=["BMP", "creatinine"])
