from datetime import date

from conftest import make_chunk

from referral.pipeline.validator import decide
from referral.pipeline.verifier import Evidence, VerifierOutput
from referral.schemas import Status

REF = date(2026, 11, 1)


def proposal(status, *evidence):
    return VerifierOutput(item_id="bmp", status=status, rationale="r",
                          evidence=[Evidence(chunk_id=c, quote=q) for c, q in evidence])


def test_present_when_quote_verified_and_recent(bmp_item):
    chunk = make_chunk("BMP: creatinine 1.1 mg/dL, potassium 4.2")
    result = decide(bmp_item, proposal("present", (chunk.chunk_id, "creatinine 1.1 mg/dL")),
                    {chunk.chunk_id: chunk}, REF)
    assert result.status == Status.PRESENT


def test_stale_when_verified_but_old(bmp_item):
    chunk = make_chunk("BMP: creatinine 1.1 mg/dL", doc_date=date(2026, 1, 5))
    result = decide(bmp_item, proposal("present", (chunk.chunk_id, "creatinine 1.1")), {chunk.chunk_id: chunk}, REF)
    assert result.status == Status.STALE


def test_needs_review_when_quote_invented(bmp_item):
    chunk = make_chunk("Patient denies chest pain.")
    result = decide(bmp_item, proposal("present", (chunk.chunk_id, "creatinine 1.1 mg/dL")),
                    {chunk.chunk_id: chunk}, REF)
    assert result.status == Status.NEEDS_REVIEW
    assert any(f.startswith("citation_failed") for f in result.flags)


def test_needs_review_when_date_unreadable(bmp_item):
    chunk = make_chunk("BMP: creatinine 1.1 mg/dL", doc_date=None)
    result = decide(bmp_item, proposal("present", (chunk.chunk_id, "creatinine 1.1")), {chunk.chunk_id: chunk}, REF)
    assert result.status == Status.NEEDS_REVIEW
    assert "date_unreadable" in result.flags


def test_contradiction_needs_two_documents(bmp_item):
    a = make_chunk("creatinine 1.1 mg/dL", doc_id="labs")
    b = make_chunk("creatinine 2.4 mg/dL", doc_id="letter")
    both = decide(bmp_item, proposal("contradictory", (a.chunk_id, "creatinine 1.1"), (b.chunk_id, "creatinine 2.4")),
                  {a.chunk_id: a, b.chunk_id: b}, REF)
    one = decide(bmp_item, proposal("contradictory", (a.chunk_id, "creatinine 1.1")), {a.chunk_id: a}, REF)
    assert both.status == Status.CONTRADICTORY
    assert one.status == Status.NEEDS_REVIEW


def test_missing_passes_through(bmp_item):
    assert decide(bmp_item, proposal("missing"), {}, REF).status == Status.MISSING
