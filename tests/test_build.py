"""Generator steps 4-6 (noise, PDFs, gold.json, leak check), offline: cases are rendered with the scripted LLM
from test_render, built into real PDFs, read back with the pipeline's loader and checked like any case."""
import random
import re

import pytest
from test_render import CHECKLIST, ScriptedLLM, plan_with

from datagen.build import add_noise, build_case, layout, locate
from datagen.check_cases import check_case
from datagen.render import BURY_PAGE, RenderedCase, render_case
from referral.eval.gold import load_gold
from referral.ingest.pdf_loader import load_case

HEADER = "PROGRESS NOTE\nDate: 10/05/2026\nPatient Name: Avery Lane\nDOB: 03/31/1951\nMRN: MRN-60855"
EVIDENCE = "Cardiac exam: regular rate and rhythm, normal S1 and S2, grade 2/6 systolic murmur, no radiation."
PAGE = (f"{HEADER}\n\nShe describes worsening breathlessness climbing stairs over several months.\n\n"
        f"{EVIDENCE}\n\nContinue metoprolol tartrate 25 mg twice daily and recheck potassium 4.2 next month.")


@pytest.mark.parametrize("seed", range(20))
def test_noise_changes_letters_only_and_tracks_evidence(seed):
    start = PAGE.index(EVIDENCE)
    noisy, [(s, e)] = add_noise(PAGE, [(start, start + len(EVIDENCE))], rate=1.0, rng=random.Random(seed))
    assert noisy != PAGE
    assert noisy.startswith(HEADER + "\n\n")                       # header never touched
    assert re.findall(r"\d+(?:\.\d+)?", noisy) == re.findall(r"\d+(?:\.\d+)?", PAGE)  # values intact
    assert noisy.count("\n") == PAGE.count("\n")                    # line structure intact
    assert noisy.split("\n\n")[2] == noisy[s:e]                     # span still covers the evidence paragraph
    for short in ["S1", "S2", "mg", "2/6"]:                         # short tokens never corrupted
        assert short in noisy


def test_zero_noise_is_identity():
    span = (PAGE.index(EVIDENCE), PAGE.index(EVIDENCE) + len(EVIDENCE))
    assert add_noise(PAGE, [span], rate=0.0, rng=random.Random(0)) == (PAGE, [span])


def test_layout_keeps_paragraphs_whole_and_each_rendered_page_separate():
    long_para = "\n".join(f"line {n}" for n in range(30))
    pdf = layout([long_para + "\n\n" + long_para, "short page"])
    assert len(pdf) == 3 and pdf[2][0] == "short page"
    assert all(line.startswith("line") or line == "" for line in pdf[0])


def test_locate_ignores_wrapping():
    assert locate("a b  c", ["x\na\nb c y"]) == (1, "a\nb c")
    assert locate("missing", ["nothing here"]) is None


BUILD_PLANS = [plan_with("b-complete", holds=False), plan_with("b-bury-ecg", {"ecg": "bury"}),
               plan_with("b-wrong-stale", {"bmp": "stale"}, wrong=True),
               plan_with("b-contradict-abbrev", {"meds": "contradict", "echo": "abbreviate"}),
               plan_with("b-omit-reason", {"reason": "omit", "progress_note": "stale", "lipids": "contradict"})]


@pytest.mark.parametrize("plan", BUILD_PLANS, ids=lambda p: p.case_id)
def test_built_case_passes_every_check(plan, tmp_path):
    case = render_case(plan, CHECKLIST, ScriptedLLM(), seed=3)
    case_dir = build_case(case, tmp_path, "dev", noise=0.08, seed=3)
    assert sorted(p.name for p in case_dir.glob("*.pdf")) == sorted(f"{d.doc_id}.pdf" for d in case.docs)
    assert check_case(case_dir) == []
    gold = load_gold(case_dir)
    assert gold.split == "dev" and {i.item_id: i.status for i in gold.items} == case.expected
    defects = {p.defect for p in plan.items if p.defect}
    assert ("buried" in gold.slices) == ("bury" in defects)
    assert ("wrong_patient" in gold.slices) == plan.wrong_patient_doc
    if "bury" in defects:
        (ev,) = [e for i in gold.items for e in i.evidence if i.item_id == "ecg"]
        assert ev.doc_id == "discharge_summary" and ev.page >= BURY_PAGE


def test_rebuild_is_identical_and_replaces_old_files(tmp_path):
    case = render_case(plan_with("again"), CHECKLIST, ScriptedLLM())
    first = build_case(case, tmp_path, "dev")
    (first / "stray.txt").write_text("old text output")
    snapshot = {p.name: p.read_bytes() for p in first.glob("*.pdf")} | {"gold.json": (first / "gold.json").read_bytes()}
    second = build_case(case, tmp_path, "dev")
    assert not (second / "stray.txt").exists()
    assert {p.name: p.read_bytes() for p in second.glob("*.pdf")} | {"gold.json": (second / "gold.json").read_bytes()} \
        == snapshot


def test_leak_check_flags_an_omitted_item_mentioned_anyway(tmp_path):
    case = render_case(plan_with("leaky", {"ecg": "omit"}), CHECKLIST, ScriptedLLM())
    letter = next(d for d in case.docs if d.doc_id == "referral_letter")
    letter.pages[0] += "\nA recent EKG was unremarkable.\n"
    problems = check_case(build_case(case, tmp_path, "dev", noise=0.0))
    assert any(p.startswith("leak: ecg is missing") and "referral_letter" in p for p in problems)


def test_generated_case_cannot_overwrite_hand_cases(tmp_path):
    case = render_case(plan_with("hand-009"), CHECKLIST, ScriptedLLM())
    with pytest.raises(ValueError):
        build_case(case, tmp_path, "dev")


def test_pdf_reads_back_through_the_pipeline_loader(tmp_path):
    case = render_case(plan_with("loader"), CHECKLIST, ScriptedLLM())
    docs = {d.doc_id: d for d in load_case(build_case(case, tmp_path, "dev", noise=0.0))}
    letter = docs["referral_letter"]
    assert letter.doc_type == "referral_letter" and letter.doc_date == case.reference_date
    assert "Confidential" in letter.pages[0] and "Page 1 of 1" in letter.pages[0]


def test_gold_page_follows_the_pdf_when_a_long_note_overflows(tmp_path):
    case = render_case(plan_with("overflow"), CHECKLIST, ScriptedLLM())
    note = next(d for d in case.docs if d.doc_type == "progress_note" and "EXAM:" in d.pages[0])
    head, rest = note.pages[0].split("\n\n", 1)
    note.pages[0] = head + "\n\n" + "\n\n".join(f"Interval history item {n}." for n in range(40)) + "\n\n" + rest
    case_dir = build_case(case, tmp_path, "dev", noise=0.0)
    assert check_case(case_dir) == []
    (exam,) = [e for i in load_gold(case_dir).items if i.item_id == "progress_note" for e in i.evidence
               if e.doc_id == note.doc_id]
    assert exam.page == 2  # one rendered page became two PDF pages


def test_case_folder_carries_its_rendering(tmp_path):
    case = render_case(plan_with("selfcontained"), CHECKLIST, ScriptedLLM())
    case_dir = build_case(case, tmp_path, "dev")
    assert RenderedCase.model_validate_json((case_dir / "rendered.json").read_text()) == case
