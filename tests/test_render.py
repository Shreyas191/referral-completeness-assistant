"""Generator steps 2-3 (defect plan, rendering), offline with a scripted LLM.

Every rendered case is checked against the rules the pipeline applies, so its labels are right by construction:
evidence verbatim on the cited page, recency windows, no leaks of omitted or stale items, identity quarantine,
and dose conflicts only where a contradiction was planted.
"""
import json
import random
import re
from pathlib import Path

import pytest

from datagen.defect_plan import DEFECTS, DefectPlan, ItemPlan, eligible, make_plan
from datagen.render import (
    BURY_PAGE,
    LONG_PAGES,
    MAX_DOCS,
    NO_REASON_RULE,
    PRESENTATIONS,
    LeakError,
    LetterProse,
    LongNoteProse,
    Medication,
    NoteProse,
    ProfileDraft,
    RenderedCase,
    Section,
    condition_keyword,
    find_terms,
    item_terms,
    render_case,
    write_case,
)
from referral.ingest.chunker import chunk_case
from referral.ingest.identity import quarantine
from referral.ingest.pdf_loader import guess_doc_date, load_case
from referral.pipeline.conflicts import dose_conflicts
from referral.schemas import Checklist, Document, Status

RAW = json.loads((Path(__file__).parents[1] / "data/checklists/cardiology-v1.json").read_text())
CHECKLIST = Checklist.model_validate(RAW)
ITEMS = {i.item_id: i for i in CHECKLIST.items}


class ScriptedLLM:
    """Answers every generator prompt with fixed text; `leaks` answers start with a forbidden term.
    With warfarin=True the profile lists warfarin whatever the prompt asked, like a model ignoring the rule."""

    def __init__(self, leaks: int = 0, warfarin: bool = False):
        self.leaks, self.warfarin = leaks, warfarin
        self.profile_prompts: list[str] = []
        self.prose_prompts: list[str] = []
        self.calls = self.input_tokens = self.output_tokens = 0

    def structured(self, system, user, schema):
        self.calls += 1
        self.input_tokens += len(user) // 4
        self.output_tokens += 100
        leak = ""
        if self.leaks:
            self.leaks -= 1
            leak = " A recent EKG was normal."
        if schema is ProfileDraft:
            self.profile_prompts.append(user)
            return ProfileDraft(
                first_name="Avery", last_name="Lane", conditions=["hypertension", "type 2 diabetes"],
                medications=[Medication(name="metoprolol tartrate", dose=25, unit="mg", frequency="twice daily"),
                             Medication(name="lisinopril", dose=10, unit="mg", frequency="once daily")]
                + ([Medication(name="warfarin", dose=4, unit="mg", frequency="once daily")] if self.warfarin else []),
                allergies="no known drug allergies", referral_reason="Evaluation of exertional chest tightness.",
                history="Six weeks of chest tightness on stairs." + leak, referring_physician="Dr. Sam Reed",
                practice="Elm Street Clinic")
        if schema is not ProfileDraft:
            self.prose_prompts.append(user)
        if schema is LetterProse:
            return LetterProse(paragraphs=["Symptoms began six weeks ago." + leak, "She remains active."])
        if schema is NoteProse:
            return NoteProse(subjective="Tired on exertion." + leak,
                             assessment_plan=["Hypertension, continue regimen.", "Refer to specialist."])
        if schema is LongNoteProse:
            return LongNoteProse(sections=[Section(title=f"Part {n}", text="Uneventful day." + leak)
                                           for n in range(LONG_PAGES)])
        raise NotImplementedError(schema)


def plan_with(name: str, defects: dict | None = None, holds: bool = True, wrong: bool = False) -> DefectPlan:
    defects = defects or {}
    return DefectPlan(case_id=name, checklist_id=RAW["checklist_id"], wrong_patient_doc=wrong,
                      items=[ItemPlan(item_id=i, defect=defects.get(i)) for i in ITEMS], condition_holds={"inr": holds})


# Every eligible (item, defect) pair on its own, plus edge combinations and a batch of random plans.
PLANS = [plan_with("complete-no-warfarin", holds=False), plan_with("complete-warfarin"),
         plan_with("wrong-patient", wrong=True),
         plan_with("note-omitted-abbrev-in-letter", {"progress_note": "omit", "ecg": "abbreviate"}),
         plan_with("note-stale-contradiction-in-letter", {"progress_note": "stale", "meds": "contradict"}),
         plan_with("bury-inr-short-window", {"inr": "bury", "bmp": "abbreviate"})]
PLANS += [plan_with(f"{item}-{defect}", {item: defect}) for item in ITEMS for defect in DEFECTS
          if eligible(defect, RAW["items"][list(ITEMS).index(item)])]
_rng = random.Random(7)
PLANS += [make_plan(f"random-{n:02d}", RAW, _rng) for n in range(30)]


def rendered(plan: DefectPlan) -> RenderedCase:
    return render_case(plan, CHECKLIST, ScriptedLLM(), seed=1)


def case_docs(case: RenderedCase):
    return [d for d in case.docs if d.patient == "case"]


def hits(docs, terms) -> set[str]:
    return {d.doc_id for d in docs if find_terms(d.text(), terms)}


def test_plans_only_plant_possible_defects():
    rng, seen = random.Random(0), set()
    for n in range(500):
        plan = make_plan(f"p{n}", RAW, rng)
        assert set(plan.condition_holds) == {i for i, it in ITEMS.items() if it.condition}
        for p in plan.items:
            if p.defect:
                seen.add(p.defect)
                assert eligible(p.defect, RAW["items"][list(ITEMS).index(p.item_id)]), (p.item_id, p.defect)
                assert plan.condition_holds.get(p.item_id, True), f"defect on not-applicable {p.item_id}"
    assert seen == set(DEFECTS)
    assert not eligible("stale", {"item_id": "meds", "evidence_type": "medication_list"})
    assert not eligible("contradict", {"item_id": "reason", "evidence_type": "note"})


@pytest.mark.parametrize("plan", PLANS, ids=lambda p: p.case_id)
def test_rendered_case_is_labelled_by_construction(plan):
    case = rendered(plan)
    docs = {d.doc_id: d for d in case.docs}
    own = case_docs(case)
    assert set(case.expected) == set(ITEMS)
    assert 2 <= len(case.docs) <= MAX_DOCS

    for d in case.docs:  # the loader must read the date we meant
        assert guess_doc_date(d.pages[0]) == d.doc_date, d.doc_id

    for e in case.evidence:
        assert e.text in docs[e.doc_id].pages[e.page - 1]
        assert docs[e.doc_id].patient == "case"

    defect = {p.item_id: p.defect for p in plan.items}
    for item_id, item in ITEMS.items():
        status, ev = case.expected[item_id], [e for e in case.evidence if e.item_id == item_id]
        applicable = plan.condition_holds.get(item_id, True)
        assert status == (Status.NOT_APPLICABLE if not applicable else
                          {"omit": Status.MISSING, "stale": Status.STALE,
                           "contradict": Status.CONTRADICTORY}.get(defect[item_id], Status.PRESENT))
        terms = item_terms(item)
        if status in (Status.MISSING, Status.NOT_APPLICABLE):
            assert not ev and not hits(own, terms), f"{item_id} leaks into {hits(own, terms)}"
        else:
            assert ev
        if status == Status.CONTRADICTORY:
            assert len({e.doc_id for e in ev}) >= 2
        if status in (Status.PRESENT, Status.STALE) and item.recency_days:
            age = (case.reference_date - max(docs[e.doc_id].doc_date for e in ev)).days
            assert (age > item.recency_days) == (status == Status.STALE), f"{item_id}: {age}d"
        if status == Status.STALE:  # no newer document may mention it
            assert hits(own, terms) <= {e.doc_id for e in ev}
        if defect[item_id] == "bury":
            assert [(e.doc_id, e.page) for e in ev] == [("discharge_summary", BURY_PAGE)]
            assert len(docs["discharge_summary"].pages) == LONG_PAGES
        if defect[item_id] == "abbreviate":
            assert all(e.doc_id.startswith(("progress_note", "referral_letter")) for e in ev)

    for item_id, holds in plan.condition_holds.items():
        key = condition_keyword(ITEMS[item_id])
        assert bool(hits(case.docs, [key])) == holds

    as_docs = [Document(doc_id=d.doc_id, case_id=case.case_id, doc_type=d.doc_type, doc_date=d.doc_date,
                        pages=d.pages) for d in case.docs]
    kept, flagged = quarantine(as_docs)
    assert sorted(flagged) == sorted(case.wrong_patient_docs)
    assert bool(dose_conflicts(chunk_case(kept))) == (defect["meds"] == "contradict")


def test_written_case_loads_like_any_case(tmp_path):
    case = rendered(plan_with("roundtrip", {"ecg": "bury"}, wrong=True))
    loaded = {d.doc_id: d for d in load_case(write_case(case, tmp_path))}
    assert {d.doc_id: d.pages for d in case.docs} == {k: d.pages for k, d in loaded.items()}
    assert RenderedCase.model_validate_json((tmp_path / "roundtrip" / "rendered.json").read_text()) == case


def test_condition_is_enforced_in_code_not_trusted_to_the_llm():
    off = render_case(plan_with("no-warfarin", holds=False), CHECKLIST, ScriptedLLM(warfarin=True))
    assert not hits(off.docs, ["warfarin"]) and off.expected["inr"] == Status.NOT_APPLICABLE
    on = render_case(plan_with("warfarin"), CHECKLIST, ScriptedLLM(warfarin=True))
    assert [m.dose for m in on.patient.medications if m.name == "warfarin"] == [5]  # one entry, set by code


def test_wrong_patient_document_differs_on_every_identifier():
    case = rendered(plan_with("wrong", wrong=True))
    (foreign,) = [d for d in case.docs if d.patient == "other"]
    p, text = case.patient, foreign.text()
    assert f"{p.first_name} {p.last_name}" not in text and p.mrn not in text and p.dob.strftime("%m/%d/%Y") not in text
    mrn = text.split("MRN: ")[1].split()[0]
    assert sorted(mrn) == sorted(p.mrn), "MRN should be the case MRN with two digits swapped"


def test_code_varies_the_patient_the_llm_is_asked_for():
    llm = ScriptedLLM()
    for n in range(20):
        render_case(plan_with(f"v{n}"), CHECKLIST, llm)
    problems = {re.search(r"Presenting problem: (.*?);", u)[1] for u in llm.profile_prompts}
    sexes = {re.search(r"-year-old (\w+)\.", u)[1] for u in llm.profile_prompts}
    assert len(problems) >= 6 and sexes == {"female", "male"}
    for problem in PRESENTATIONS["cardiology"]:  # a presenting problem must not name a checklist item
        assert not find_terms(problem, [t for i in ITEMS.values() for t in item_terms(i)]), problem


def test_missing_reason_is_never_shown_to_the_prose_model():
    llm = ScriptedLLM()
    case = render_case(plan_with("no-reason", {"reason": "omit"}), CHECKLIST, llm)
    assert llm.prose_prompts and case.expected["reason"] == Status.MISSING
    for prompt in llm.prose_prompts:
        assert case.patient.referral_reason not in prompt and case.patient.history not in prompt
        assert NO_REASON_RULE in prompt


def test_crowded_packet_drops_older_notes_to_stay_within_eight_documents():
    for seed in range(10):  # older-note count is random; the cap must hold for every draw
        case = render_case(plan_with("crowded", {"bmp": "stale", "ecg": "bury"}, wrong=True), CHECKLIST,
                           ScriptedLLM(), seed=seed)
        assert len(case.docs) == MAX_DOCS
        assert sum(d.doc_type == "progress_note" for d in case.docs) == 1


def test_leak_guard_retries_then_gives_up():
    clean, leaky = ScriptedLLM(), ScriptedLLM(leaks=1)
    render_case(plan_with("retry"), CHECKLIST, clean)
    case = render_case(plan_with("retry"), CHECKLIST, leaky)
    assert leaky.calls == clean.calls + 1 and case.llm_calls == leaky.calls
    assert not hits(case.docs, ["EKG"])
    with pytest.raises(LeakError):
        render_case(plan_with("hopeless"), CHECKLIST, ScriptedLLM(leaks=100))


def test_find_terms_matches_whole_words_only():
    assert find_terms("Recent EKG, normal.", ["EKG"]) == {"EKG"}
    assert find_terms("She echoed the plan.", ["echo"]) == set()
    assert find_terms("PT/INR 2.1", ["PT/INR", "prothrombin time"]) == {"PT/INR"}
    assert find_terms("bilateral 12-lead tracing", ["12-lead"]) == {"12-lead"}
