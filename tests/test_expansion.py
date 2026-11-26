"""Value-term expansion: the second document of a contradiction reaches the verifier (dev-009, dev-012)."""
from conftest import make_chunk

from referral.retrieval.expansion import value_expansion, value_mentions, value_terms

LAB = make_chunk("Date: 09/11/2026\nMRN: 20236\nBASIC METABOLIC PANEL\nPotassium: 4.2 mmol/L\nSodium: 136 mmol/L",
                 doc_id="lab_report")
LETTER = make_chunk("Reason for referral: palpitations.\nMost recent potassium is 5.9 mmol/L.",
                    doc_id="referral_letter")
NOTE = make_chunk("Seen 09/01/2026. Diet discussed for 30 minutes.", doc_id="progress_note")
LAB_P2 = make_chunk("Potassium: 4.0 mmol/L (repeat)", doc_id="lab_report", page=2)


def test_value_terms_skip_header_words():
    assert value_terms(LAB.text) == {"potassium", "sodium"}


def test_value_mentions_need_a_number_after_the_term():
    assert value_mentions("Most recent potassium is 5.9 mmol/L.", {"potassium"}) == {"potassium"}
    assert value_mentions("Potassium supplements were stopped.", {"potassium"}) == set()


def test_adds_another_documents_value_for_the_same_term():
    assert value_expansion([LAB], [LAB, LETTER, NOTE, LAB_P2]) == [LETTER]


def test_drug_names_from_the_med_list_find_the_note():
    meds = make_chunk("Lisinopril 10 mg once daily\nTiotropium 18 mcg once daily", doc_id="medication_list")
    note = make_chunk("Continue tiotropium 36 mcg once daily.", doc_id="progress_note")
    assert value_expansion([meds], [meds, note, NOTE]) == [note]


def test_never_repeats_shown_chunks_and_respects_the_cap():
    others = [make_chunk(f"Potassium: 4.{n} mmol/L", doc_id=f"note_{n}") for n in range(4)]
    assert value_expansion([LAB, others[0]], [LAB, *others], max_extra=2) == others[1:3]


def test_nothing_to_expand():
    assert value_expansion([], [LAB]) == []
    assert value_expansion([NOTE], [NOTE, LAB]) == []
