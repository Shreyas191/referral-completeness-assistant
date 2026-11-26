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


def test_names_item_matches_description_or_synonym_in_quotes(bmp_item):
    from referral.pipeline.absence import names_item
    from referral.pipeline.verifier import Evidence, VerifierOutput

    def prop(quote):
        return VerifierOutput(item_id="bmp", status="present", rationale="",
                              evidence=[Evidence(chunk_id="labs:p1:0", quote=quote)])

    assert names_item(bmp_item, prop("Creatinine 1.1 mg/dL"))
    assert names_item(bmp_item, prop("Basic metabolic panel resulted."))
    assert not names_item(bmp_item, prop("Labs were reviewed with the patient."))
    assert not names_item(bmp_item, VerifierOutput(item_id="bmp", status="present", rationale=""))


NOTE = make_chunk("PROGRESS NOTE\nDate: 11/03/2026\nPatient Name: Jane Doe\nSUBJECTIVE:\nFatigue for months.\n"
                  "EXAM:\nCardiac exam: regular rate and rhythm, normal S1 and S2, no murmurs.",
                  doc_id="progress_note", doc_date=date(2026, 11, 3))
ELIDED = "Date: 11/03/2026\n...\nCardiac exam: regular rate and rhythm, normal S1 and S2, no murmurs."


def test_joined_quote_rejected_by_default_and_accepted_piecewise(bmp_item):
    # rq-dev: the model joined two lines of a note with "...", and a correct Present went to Needs review.
    p = proposal("present", (NOTE.chunk_id, ELIDED))
    assert decide(bmp_item, p, {NOTE.chunk_id: NOTE}, REF).status == Status.NEEDS_REVIEW
    result = decide(bmp_item, p, {NOTE.chunk_id: NOTE}, REF, piecewise_quotes=True)
    assert result.status == Status.PRESENT and result.citations[0].quote == ELIDED


def test_piecewise_still_needs_every_piece_in_the_chunk(bmp_item):
    from referral.pipeline.validator import quote_supported

    invented = "Date: 11/03/2026 ... Cardiac exam: irregularly irregular rhythm with a loud diastolic rumble."
    assert not quote_supported(invented, NOTE, piecewise=True)
    assert not quote_supported("... and ...", NOTE, piecewise=True)        # no piece long enough
    assert quote_supported("Date: 11/03/2026 … Fatigue for months.", NOTE, piecewise=True)  # unicode ellipsis
    # elide-dev: non-adjacent lines joined by line breaks, with no ellipsis
    joined = "PROGRESS NOTE\nFatigue for months.\nCardiac exam: regular rate"
    assert not quote_supported(joined, NOTE)
    assert quote_supported(joined, NOTE, piecewise=True)
    # one long invented line still fails, even if another line is real
    assert not quote_supported("PROGRESS NOTE\nEjection fraction 25% on today's echo.", NOTE, piecewise=True)


def test_quote_longer_than_its_chunk_must_match_the_whole_chunk(bmp_item):
    # dev-021: a page-break remnant "f 1" is a chunk of its own; partial_ratio gave any quote containing it 100.
    from referral.pipeline.validator import quote_score, quote_supported

    remnant = make_chunk("f 1")
    padded = "Echo shows LVEF of 15% with severe MR."
    assert quote_score(padded, remnant, length_check=False) == 100       # the old hole
    assert not quote_supported(padded, remnant)
    assert quote_supported(padded, remnant, length_check=False)
    p = proposal("present", (remnant.chunk_id, padded))
    assert decide(bmp_item, p, {remnant.chunk_id: remnant}, REF).status == Status.NEEDS_REVIEW
    assert decide(bmp_item, p, {remnant.chunk_id: remnant}, REF, quote_length_check=False).status == Status.PRESENT


def test_length_check_keeps_whole_chunk_quotes_and_ocr_noise(bmp_item):
    from referral.pipeline.validator import quote_supported

    short = make_chunk("Allergies: penicillin (rash).")
    assert quote_supported("Allergies: penicillin (rash).", short)      # the whole chunk, exactly
    assert quote_supported("Allergies:  Penicillin (rash). ", short)    # case and whitespace differ
    assert quote_supported("Allergies: penicilin (rash).", short)       # one OCR-style slip, still shorter
    assert quote_supported("Allergies: penicillin (rash)..", short)     # one extra character
    assert not quote_supported("Allergies: penicillin (rash). No known drug interactions.", short)
    # piecewise pieces are held to the same rule
    assert not quote_supported("Fatigue for months.\nAllergies: penicillin (rash) and sulfa (hives).", short,
                               piecewise=True)
