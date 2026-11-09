from referral.ingest.chunker import chunk_document
from referral.ingest.identity import quarantine
from referral.schemas import Document

HEADER = "Patient Name: Jane Doe\nDOB: 03/14/1961\nMRN: AB12345\n\n"


def doc(doc_id, text, doc_type="unknown"):
    return Document(doc_id=doc_id, case_id="c1", doc_type=doc_type, pages=[text])


def test_chunks_keep_page_and_offsets():
    d = Document(doc_id="note", case_id="c1", pages=["First paragraph.\n\nSecond paragraph.", "Page two."])
    chunks = chunk_document(d)
    assert [c.page for c in chunks] == [1, 1, 2]
    assert d.pages[0][chunks[1].start:chunks[1].end].strip() == "Second paragraph."


def test_wrong_patient_document_is_quarantined():
    docs = [
        doc("letter", HEADER + "Referred for chest pain.", "referral_letter"),
        doc("labs", HEADER + "Creatinine 1.1"),
        doc("other", "Patient Name: John Smith\nDOB: 01/01/1950\nMRN: ZZ99999\n\nCreatinine 3.0"),
        doc("cover", "Fax cover sheet, 4 pages"),
    ]
    kept, flagged = quarantine(docs)
    assert [d.doc_id for d in kept] == ["letter", "labs"]
    assert flagged == ["other", "cover"]
