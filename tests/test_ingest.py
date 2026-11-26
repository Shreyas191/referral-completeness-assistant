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


NOTE = ("SUBJECTIVE:\nFatigue for months. Denies chest pain at rest or with exertion over the past year.\n"
        "Fasting lipid panel this admission: total cholesterol 225 mg/dL, LDL 131 mg/dL, HDL 44\nmg/dL.\n"
        "EXAM:\nGeneral: well appearing, no distress.\nNeck: no jugular venous distension.\n"
        "Cardiac exam: regular rate and rhythm, no murmurs.")


def test_fixed_cuts_split_sections_and_sentences():
    # The old behaviour, kept as the default: cuts land every max_chars, wherever that is.
    chunks = chunk_document(doc("note", NOTE), max_chars=120)
    assert all(len(c.text) <= 120 for c in chunks)
    assert not any("EXAM:\nGeneral: well appearing" in c.text and "Cardiac exam" in c.text for c in chunks)


def test_sentence_cuts_keep_a_section_together_and_never_cut_mid_word():
    chunks = chunk_document(doc("note", NOTE), max_chars=200, sentence_cuts=True)
    assert all(len(c.text) <= 200 for c in chunks)
    assert any(c.text.startswith("EXAM:") and "Cardiac exam" in c.text for c in chunks)   # cut before the header
    assert any(c.text.endswith("HDL 44\nmg/dL.") for c in chunks)                     # sentence end, not the wrap
    d = doc("note", NOTE)
    for c in chunks:                                                                     # offsets still exact
        assert d.pages[0][c.start:c.end].strip() == c.text


def test_sentence_cuts_fall_back_to_a_space_then_a_hard_cut():
    words = " ".join(["word"] * 60)                 # no line breaks, no sentence ends
    assert all(not c.text.endswith("wor") for c in chunk_document(doc("n", words), 50, sentence_cuts=True))
    blob = "x" * 130                                # no boundary at all: hard cuts
    assert [len(c.text) for c in chunk_document(doc("n", blob), 50, sentence_cuts=True)] == [50, 50, 30]


def test_sentence_cuts_leave_no_tiny_remnant():
    # dev-021: a paragraph 3 characters over max_chars left the chunk "f 1".
    para = "Assessment: stable. " * 60 + "f 1"            # 1,203 characters
    assert [len(c.text) for c in chunk_document(doc("n", para), 1200)][-1] == 3
    sizes = [len(c.text) for c in chunk_document(doc("n", para), 1200, sentence_cuts=True)]
    assert min(sizes) >= 1200 // 8 and max(sizes) <= 1200
