"""Step 3 of the generator: turn a DefectPlan into the documents of one referral packet.

    python -m datagen.render --checklist cardiology-v1 --n 5 --seed 0

Code decides every fact a checklist item depends on: values, dates, which document and page holds it,
and the exact evidence text, with the plan's defects applied. The LLM writes only the patient profile and
the narrative prose around those facts (letter body, note sections, a long discharge summary), and must
avoid every checklist term, so an omitted or stale item cannot leak back in through the prose. Labels are
therefore known by construction: RenderedCase.expected and RenderedCase.evidence.

Defects:
  omit        no evidence anywhere
  stale       the evidence sits in its own document dated past the item's recency window
  contradict  the note (or letter) states a value that conflicts with the item's own document
  bury        the evidence is on page BURY_PAGE of a LONG_PAGES-page discharge summary
  abbreviate  the evidence is a clinical-shorthand line in the note (or letter), with no formal report
A conditional item whose condition does not hold is not_applicable, and the condition's key word
(e.g. "warfarin") appears nowhere in the packet.

Output per case in --out (default data/rendered/): <case_id>/rendered.json (input to step 4: noise, PDF
and gold.json) and one <doc_id>.txt per document, pages split by form feeds, for spot checks.
"""
import argparse
import json
import random
import re
from dataclasses import dataclass
from datetime import date, timedelta
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field

from datagen.defect_plan import DefectPlan, make_plan
from referral.schemas import Checklist, ChecklistItem, Status

LONG_PAGES = 12  # bury: length of the discharge summary...
BURY_PAGE = 11   # ...and the page that holds the evidence ("buried past page 10")
REF_START = date(2026, 9, 1)  # referral (reference) dates fall in the 90 days after this
LEAK_ATTEMPTS = 3
MAX_DOCS = 8  # packets are 2-8 documents (CLAUDE.md); older progress notes are dropped first
NO_REASON_RULE = ("Do not say why the patient is being referred, what should be evaluated, or mention any planned "
                  "procedure or the current complaint.")

SYSTEM = ("You write synthetic clinical documents for testing software. Every patient is fictional. "
          "Write in the plain, terse style of transcribed US clinic notes (as in MTSamples). Return only JSON.")


# ---------------------------------------------------------------- LLM-written parts

NAME = r"^[A-Z][A-Za-z'\-]+$"


class Medication(BaseModel):
    name: str = Field(pattern=r"^[A-Za-z][A-Za-z \-]+$", description="generic name, e.g. metoprolol tartrate")
    dose: float = Field(gt=0)
    unit: Literal["mg", "mcg", "units"]
    frequency: str = Field(description="e.g. once daily, twice daily, at bedtime")


class ProfileDraft(BaseModel):
    first_name: str = Field(pattern=NAME)
    last_name: str = Field(pattern=NAME)
    conditions: list[str] = Field(min_length=2, max_length=6)
    medications: list[Medication] = Field(min_length=2, max_length=8)
    allergies: str
    referral_reason: str = Field(description="one sentence: the clinical problem the specialist should evaluate")
    history: str = Field(description="2-4 sentences of history of present illness, no test results")
    referring_physician: str = Field(description='e.g. "Dr. Jane Smith"')
    practice: str


class Patient(ProfileDraft):
    sex: str
    age: int
    dob: date
    mrn: str


class LetterProse(BaseModel):
    paragraphs: list[str] = Field(min_length=2, max_length=4)


class NoteProse(BaseModel):
    subjective: str
    assessment_plan: list[str] = Field(min_length=2, max_length=5, description="one problem and its plan each")


class Section(BaseModel):
    title: str
    text: str


class LongNoteProse(BaseModel):
    sections: list[Section] = Field(min_length=LONG_PAGES, max_length=LONG_PAGES)


# ---------------------------------------------------------------- output

class RenderedDoc(BaseModel):
    doc_id: str
    doc_type: str
    doc_date: date
    pages: list[str]
    patient: Literal["case", "other"] = "case"  # "other" = wrong-patient document

    def text(self) -> str:
        return "\f".join(self.pages)


class PlantedEvidence(BaseModel):
    item_id: str
    doc_id: str
    page: int  # 1-based, like Chunk.page
    text: str  # exactly as it appears on the page


class RenderedCase(BaseModel):
    case_id: str
    checklist_id: str
    reference_date: date
    plan: DefectPlan
    patient: Patient
    docs: list[RenderedDoc]
    evidence: list[PlantedEvidence]
    expected: dict[str, Status]
    wrong_patient_docs: list[str] = []
    llm_calls: int = 0
    input_tokens: int = 0
    output_tokens: int = 0


class LeakError(RuntimeError):
    """The LLM kept using a forbidden term, so the case cannot be labelled by construction."""


# ---------------------------------------------------------------- per-item facts (cardiology-v1)

@dataclass(frozen=True)
class ItemSpec:
    home: str                      # where the clean evidence lives: letter | note | lab | test | imaging | meds
    extra_terms: tuple[str, ...] = ()  # words the prose must avoid beyond the checklist synonyms


SPECS = {
    "reason": ItemSpec("letter"),
    "progress_note": ItemSpec("note"),
    "ecg": ItemSpec("test", ("electrocardiogram", "QTc")),
    "echo": ItemSpec("imaging", ("echocardiogram", "ejection fraction", "LVEF")),
    "lipids": ItemSpec("lab", ("lipid panel", "lipids")),
    "bmp": ItemSpec("lab", ("basic metabolic panel", "sodium", "kidney function")),
    "meds": ItemSpec("meds"),
    "inr": ItemSpec("lab", ("coagulation",)),
}


# Presenting problems by specialty, drawn per case for variety. None may name a checklist item.
PRESENTATIONS = {
    "cardiology": [
        "chest pain or pressure on exertion", "an episode of fainting", "a newly heard heart murmur",
        "leg swelling and breathlessness suggesting heart failure", "palpitations",
        "blood pressure that stays high despite treatment", "risk assessment before planned non-cardiac surgery",
        "an irregular pulse found at a routine visit", "breathlessness on exertion",
        "fatigue with a strong family history of early heart disease",
    ],
}


def _flag(value: float, low: float | None, high: float | None) -> str:
    if low is not None and value < low:
        return " (L)"
    if high is not None and value > high:
        return " (H)"
    return ""


def _ecg(rng: random.Random) -> dict[str, str]:
    rhythm, impression, short = rng.choice([
        ("Normal sinus rhythm", "Normal sinus rhythm with no acute ST-T wave changes", "NSR, no acute ST-T changes"),
        ("Sinus bradycardia", "Sinus bradycardia, otherwise normal tracing", "SB, otherwise nl"),
        ("Sinus rhythm with occasional premature atrial complexes",
         "Sinus rhythm with occasional premature atrial complexes and no acute changes", "SR w/ occ PACs"),
        ("Normal sinus rhythm", "Normal sinus rhythm with voltage criteria for left ventricular hypertrophy",
         "NSR, LVH by voltage"),
    ])
    hr = rng.randint(50, 58) if "bradycardia" in rhythm else rng.randint(62, 92)
    pr, qrs, qtc = rng.randint(140, 200), rng.randint(80, 110), rng.randint(400, 460)
    return {
        "block": f"Findings: {rhythm} at {hr} bpm. Normal axis. PR interval {pr} ms, QRS {qrs} ms, QTc {qtc} ms.\n"
                 f"Impression: {impression}.",
        "abbrev": f"EKG: {short}, rate {hr}, PR {pr}, QRS {qrs}.",
        "bury": f"12-lead ECG on admission: {rhythm.lower()} at {hr} bpm, PR interval {pr} ms, QRS {qrs} ms, "
                f"QTc {qtc} ms. {impression}.",
    }


def _echo(rng: random.Random) -> dict[str, str]:
    ef = rng.choice([30, 35, 40, 45, 50, 55, 60, 60, 65])
    grade, short = (("Normal", "nl") if ef >= 55 else ("Mildly reduced", "mildly reduced") if ef >= 45
                    else ("Moderately reduced", "mod reduced") if ef >= 35
                    else ("Severely reduced", "severely reduced"))
    extra = rng.choice(["Trace mitral regurgitation", "Mild mitral regurgitation", "Grade 1 diastolic dysfunction",
                        "Mild aortic valve sclerosis without stenosis", "Mild left atrial enlargement"])
    alt = ef - 25 if ef >= 55 else min(ef + 25, 65)
    return {
        "block": f"Findings: Normal left ventricular size. Left ventricular ejection fraction estimated at {ef}%. "
                 f"{extra}. No pericardial effusion.\nImpression: {grade} left ventricular systolic function.",
        "abbrev": f"TTE: LVEF {ef}%, {short} LV fxn, no sig valve dz.",
        "alt": f"Most recent echo showed an ejection fraction of {alt}%.",
        "bury": f"Transthoracic echo this admission: LVEF {ef}%. {extra}. "
                f"{grade} left ventricular systolic function.",
    }


def _lipids(rng: random.Random) -> dict[str, str]:
    ldl, hdl, tg = rng.randint(60, 190), rng.randint(32, 75), rng.randint(70, 260)
    tc = ldl + hdl + tg // 5
    alt = ldl + 60 if ldl < 130 else ldl - 60
    return {
        "block": f"LIPID PANEL\nTotal Cholesterol: {tc} mg/dL{_flag(tc, None, 199)}\n"
                 f"LDL Cholesterol: {ldl} mg/dL{_flag(ldl, None, 129)}\n"
                 f"HDL Cholesterol: {hdl} mg/dL{_flag(hdl, 40, None)}\n"
                 f"Triglycerides: {tg} mg/dL{_flag(tg, None, 149)}",
        "abbrev": f"Lipids: TC {tc}, LDL {ldl}, HDL {hdl}, TG {tg}.",
        "alt": f"Most recent LDL is {alt} mg/dL.",
        "bury": f"Fasting lipid panel this admission: total cholesterol: {tc} mg/dL, LDL: {ldl} mg/dL, "
                f"HDL: {hdl} mg/dL, triglycerides: {tg} mg/dL.",
    }


def _bmp(rng: random.Random) -> dict[str, str]:
    na, cl, co2 = rng.randint(134, 144), rng.randint(98, 107), rng.randint(21, 29)
    k, cr = round(rng.uniform(3.5, 5.2), 1), round(rng.uniform(0.6, 1.6), 1)
    bun, glu, ca = rng.randint(8, 30), rng.randint(80, 180), round(rng.uniform(8.6, 10.2), 1)
    alt = 5.9 if k < 4.8 else 3.2
    return {
        "block": f"BASIC METABOLIC PANEL\nSodium: {na} mmol/L{_flag(na, 135, 145)}\n"
                 f"Potassium: {k} mmol/L{_flag(k, 3.5, 5.0)}\nChloride: {cl} mmol/L\nCO2: {co2} mmol/L\n"
                 f"BUN: {bun} mg/dL{_flag(bun, None, 20)}\nCreatinine: {cr} mg/dL{_flag(cr, None, 1.2)}\n"
                 f"Glucose: {glu} mg/dL{_flag(glu, None, 99)}\nCalcium: {ca} mg/dL",
        "abbrev": f"BMP: Na {na}, K {k}, Cl {cl}, CO2 {co2}, BUN {bun}, Cr {cr}, Glu {glu}.",
        "alt": f"Most recent potassium is {alt} mmol/L.",
        "bury": f"Basic metabolic panel on discharge: sodium: {na} mmol/L, potassium: {k} mmol/L, "
                f"creatinine: {cr} mg/dL, BUN: {bun} mg/dL, glucose: {glu} mg/dL.",
    }


def _inr(rng: random.Random) -> dict[str, str]:
    inr = round(rng.uniform(1.8, 3.4), 1)
    pt = round(inr * 11.5 + rng.uniform(-1, 1), 1)
    return {
        "block": f"COAGULATION\nProthrombin Time: {pt} sec\nINR: {inr}",
        "abbrev": f"PT/INR: {pt}/{inr}.",
        "alt": f"Most recent INR is {1.3 if inr > 2 else 3.8}.",
        "bury": f"Coagulation on discharge: INR {inr}.",
    }


def med_line(m: Medication) -> str:
    route = "subcutaneously" if m.unit == "units" else "by mouth"
    return f"{m.name[0].upper()}{m.name[1:]} {m.dose:g} {m.unit} {route} {m.frequency}"


def _meds(rng: random.Random, patient: Patient) -> dict[str, str]:
    block = "\n".join(med_line(m) for m in patient.medications)
    texts = {"block": block, "bury": f"Discharge medications:\n{block}"}
    # The conflict checker keys doses on the word before the number, so pick a med whose last word is a real word.
    candidates = [m for m in patient.medications if len(m.name.split()[-1]) >= 4]
    if candidates:
        m = rng.choice(candidates)
        alt = m.dose * 2 if m.dose < 100 else m.dose / 2
        texts["alt_note"] = f"Continue {m.name} {alt:g} {m.unit} {m.frequency}."
        texts["alt_letter"] = f"Takes {m.name} {alt:g} {m.unit} {m.frequency}."
    return texts


FACTS = {"ecg": _ecg, "echo": _echo, "lipids": _lipids, "bmp": _bmp, "inr": _inr}


def _exam(rng: random.Random) -> str:
    return "\n".join([
        "EXAM:",
        f"General: {rng.choice(['well appearing, no distress', 'alert, no acute distress', 'comfortable at rest'])}.",
        "Neck: " + rng.choice(["no jugular venous distension", "no carotid bruits",
                               "supple, no jugular venous distension"]) + ".",
        "Cardiac exam: regular rate and rhythm, normal S1 and S2, "
        + rng.choice(["no murmurs, rubs, or gallops", "soft systolic murmur at the sternal border, no gallop",
                      "no murmurs or gallops", "grade 2/6 systolic ejection murmur, no radiation"]) + ".",
        f"Lungs: {rng.choice(['clear to auscultation bilaterally', 'clear bilaterally', 'faint bibasilar crackles'])}.",
        f"Extremities: {rng.choice(['no peripheral edema', 'trace ankle edema', '1+ pitting edema of both ankles'])}.",
    ])


def _vitals(rng: random.Random) -> str:
    return (f"BP {rng.randint(112, 158)}/{rng.randint(66, 94)}, HR {rng.randint(58, 92)}, RR {rng.randint(12, 18)}, "
            f"SpO2 {rng.randint(94, 99)}% on room air, weight {rng.randint(120, 240)} lb.")


# ---------------------------------------------------------------- helpers

def fmt(d: date) -> str:
    return d.strftime("%m/%d/%Y")


def item_terms(item: ChecklistItem) -> list[str]:
    return [item.description, *item.synonyms, *SPECS[item.item_id].extra_terms]


def condition_keyword(item: ChecklistItem) -> str:
    return item.condition.split()[-1].lower()  # "patient is on warfarin" -> "warfarin"


def find_terms(text: str, terms: list[str]) -> set[str]:
    """Terms that occur in text as whole words, ignoring case."""
    return {t for t in terms if t and re.search(rf"(?<![\w/-]){re.escape(t)}(?![\w/-])", text, re.I)}


def _strings(value, skip: frozenset = frozenset()) -> list[str]:
    if isinstance(value, str):
        return [value]
    if isinstance(value, dict):
        return [s for k, v in value.items() if k not in skip for s in _strings(v, skip)]
    if isinstance(value, list):
        return [s for v in value for s in _strings(v, skip)]
    return []


def ask(llm, user: str, schema: type[BaseModel], forbidden: list[str], skip: frozenset = frozenset()):
    """llm.structured with a leak guard: retry with feedback while any forbidden term appears."""
    prompt = user
    for _ in range(LEAK_ATTEMPTS):
        out = llm.structured(SYSTEM, prompt, schema)
        leaked = find_terms("\n".join(_strings(out.model_dump(), skip)), forbidden)
        if not leaked:
            return out
        prompt = f"{user}\n\nYour last answer used forbidden words: {', '.join(sorted(leaked))}. Rewrite without them."
    raise LeakError(f"{schema.__name__}: still used {sorted(leaked)} after {LEAK_ATTEMPTS} attempts")


def header(title: str, when: date, p: Patient, *extra: str) -> str:
    return "\n".join([title, f"Date: {fmt(when)}", f"Patient Name: {p.first_name} {p.last_name}",
                      f"DOB: {fmt(p.dob)}", f"MRN: {p.mrn}", *extra])


def running_header(p: Patient, title: str, page: int, total: int) -> str:
    return f"{p.first_name} {p.last_name} | {p.mrn} | {title} | Page {page} of {total}"


def _rules(forbidden: list[str], extra: str = "") -> str:
    return ("Rules:\n"
            f"- Never use these words or phrases, in any capitalisation: {', '.join(sorted(set(forbidden)))}.\n"
            "- Do not name any medication, dose, lab value, test or imaging result, or calendar date; "
            "those are added separately.\n"
            "- No physical examination or vital signs. Plain sentences, no markdown or headings in the fields.\n"
            + (f"- {extra}\n" if extra else ""))


# ---------------------------------------------------------------- the renderer

class _Builder:
    def __init__(self, plan: DefectPlan, checklist: Checklist, llm, rng: random.Random):
        self.plan, self.checklist, self.llm, self.rng = plan, checklist, llm, rng
        self.items = {i.item_id: i for i in checklist.items}
        missing_specs = set(self.items) - set(SPECS)
        if missing_specs:
            raise NotImplementedError(f"no fact templates for items {sorted(missing_specs)}; add them to SPECS")
        self.defect = {p.item_id: p.defect for p in plan.items}
        self.applicable = {i for i in self.items if plan.condition_holds.get(i, True)}
        self.ref = REF_START + timedelta(days=rng.randint(0, 89))
        self.docs: list[RenderedDoc] = []
        self.evidence: list[PlantedEvidence] = []
        # Prose must avoid every checklist term and every condition key word, whatever the plan says.
        self.forbidden = [t for i in self.items.values() for t in item_terms(i)]
        self.forbidden += [condition_keyword(i) for i in self.items.values() if i.condition]

    def d(self, item_id: str) -> str | None:
        return self.defect[item_id] if item_id in self.applicable else "n/a"

    def fresh(self, window: int | None, cap: int = 120) -> date:
        hi = min(cap, int(window * 0.7)) if window else cap
        return self.ref - timedelta(days=self.rng.randint(2, max(2, hi)))

    def old(self, window: int) -> date:
        return self.ref - timedelta(days=window + self.rng.randint(15, 150))

    def window(self, item_id: str) -> int | None:
        return self.items[item_id].recency_days

    def add(self, doc: RenderedDoc, planted: list[tuple[str, str]] = ()) -> None:
        """Add a document and record where each (item_id, text) landed."""
        self.docs.append(doc)
        for item_id, text in planted:
            page = next((n for n, pg in enumerate(doc.pages, 1) if text in pg), None)
            if page is None:
                raise AssertionError(f"{item_id}: planted text not found in {doc.doc_id}")
            self.evidence.append(PlantedEvidence(item_id=item_id, doc_id=doc.doc_id, page=page, text=text))

    # -- profile

    def make_patient(self) -> Patient:
        rng = self.rng
        sex, age = rng.choice(["female", "male"]), rng.randint(42, 88)
        dob = self.ref - timedelta(days=age * 365 + rng.randint(30, 330))
        letters = "ABCDEFGHJKLMNPRSTW"
        initials = [rng.choice(letters) for _ in range(5)]
        # Left alone, the model writes the same patient every time (spot check 1: 5/5 palpitations, 4/5 AF,
        # three "Dr. Helen ..."), so code picks the presenting problem and the initials of every name.
        problem = rng.choice(PRESENTATIONS.get(self.checklist.specialty, ["a problem typical for this clinic"]))
        conds = []
        for item_id, holds in self.plan.condition_holds.items():
            cond = self.items[item_id].condition
            conds.append(f'- The statement "{cond}" must be {"TRUE" if holds else "FALSE"} for this patient.'
                         + (" Include it among the medications." if holds else ""))
        user = (f"Create a fictional patient being referred to a {self.checklist.specialty} clinic.\n"
                f"Fixed facts: {age}-year-old {sex}. First name starts with {initials[0]}, last name starts "
                f"with {initials[1]}. Presenting problem: {problem}; referral_reason and history must be about it. "
                f"Referring physician's first name starts with {initials[2]} and last name with {initials[3]}; "
                f"practice name starts with {initials[4]}.\n" + "\n".join(conds) + "\n"
                "Fields: conditions = 2-6 chronic diagnoses; medications = 2-8 current medications that fit them, "
                "generic name, numeric dose, unit (mg, mcg or units) and frequency; referral_reason = one sentence "
                "starting 'Evaluation of'; history = 2-4 sentences, no test results.\n"
                f"Never use these words in any field except medications: {', '.join(sorted(set(self.forbidden)))}.")
        draft = ask(self.llm, user, ProfileDraft, self.forbidden, skip=frozenset({"medications"}))
        meds = list(draft.medications)
        for item_id, holds in self.plan.condition_holds.items():  # enforce the condition in code
            key = condition_keyword(self.items[item_id])
            meds = [m for m in meds if key not in m.name.lower()]
            if holds:
                meds.append(Medication(name=key, dose=5, unit="mg", frequency="once daily"))
        mrn = f"MRN-{rng.randint(10000, 99999)}"
        return Patient(**draft.model_dump(exclude={"medications"}), medications=meds, sex=sex, age=age,
                       dob=dob, mrn=mrn)

    def patient_summary(self, p: Patient, with_reason: bool) -> str:
        """Patient context for prose prompts. Without the reason, the prompt gets only the chronic conditions: the
        history is about the presenting problem, and spot check 2 showed the model reconstructing the reason from it
        ("scheduled for elective knee replacement")."""
        text = (f"Patient: {p.first_name} {p.last_name}, {p.age}-year-old {p.sex}. "
                f"Conditions: {', '.join(p.conditions)}.\n")
        if with_reason:
            text += f"Background: {p.history}\nBeing referred to {self.checklist.specialty} for: {p.referral_reason}\n"
        return text

    # -- main

    def build(self) -> RenderedCase:
        rng = self.rng
        p = self.make_patient()
        # From here on prose must not name the patient's medications either (doses live in code-made text only).
        self.forbidden += [w for m in p.medications for w in m.name.split() if len(w) >= 4]
        facts = {i: FACTS[i](rng) for i in FACTS if i in self.applicable}
        meds = _meds(rng, p) if "meds" in self.applicable else {}
        if self.d("meds") == "contradict" and "alt_note" not in meds:
            raise ValueError("meds contradiction needs a medication whose name ends in a word of 4+ letters")

        reason_known = self.d("reason") not in ("omit",)
        note_ok = self.d("progress_note") is None  # a current progress note exists to carry inserted lines
        target = "note" if note_ok else "letter"

        # Lines inserted into the narrative document: shorthand evidence, contradicting values, conditions.
        inserts: list[tuple[str, str]] = []
        for item_id in self.items:
            if self.d(item_id) == "abbreviate":
                inserts.append((item_id, facts[item_id]["abbrev"]))
            elif self.d(item_id) == "contradict":
                alt = meds[f"alt_{target}"] if item_id == "meds" else facts[item_id]["alt"]
                inserts.append((item_id, alt))
        condition_lines = [f"{c[0].upper()}{c[1:]}." for i, c in
                           ((i, self.items[i].condition) for i, holds in self.plan.condition_holds.items() if holds)]

        # Narrative note date must keep every present item it carries inside that item's window.
        windows = [self.window("progress_note")] + [self.window(i) for i, _ in inserts
                                                    if self.d(i) == "abbreviate" and self.window(i)]
        windows = [w for w in windows if w]
        if self.d("progress_note") == "stale":
            note_date = self.old(self.window("progress_note"))
        else:
            note_date = self.fresh(min(windows) if windows else None, cap=45)

        others = self._count_other_docs(facts)
        notes_wanted = self.d("progress_note") in (None, "stale")
        if others + notes_wanted > MAX_DOCS:
            raise ValueError(f"packet would have {others + notes_wanted} documents (max {MAX_DOCS})")

        in_letter = target == "letter"
        self._letter(p, reason_known, inserts if in_letter else [], condition_lines if in_letter else [])
        if self.d("progress_note") in (None, "stale"):
            self._notes(p, note_date, reason_known, inserts if target == "note" else [],
                        condition_lines if target == "note" else [], max_older=MAX_DOCS - others - 1)
        if self.d("meds") in (None, "contradict"):
            when = note_date if note_ok else self.ref - timedelta(days=rng.randint(0, 14))
            self.add(RenderedDoc(doc_id="medication_list", doc_type="medication_list", doc_date=when,
                                 pages=[header("CURRENT MEDICATION LIST", when, p) + "\n\n" + meds["block"]
                                        + f"\n\nAllergies: {p.allergies}\n"]),
                     [("meds", meds["block"])])
        self._reports(p, facts)
        self._labs(p, facts)
        buried = [i for i in self.items if self.d(i) == "bury"]
        if buried:
            item_id = buried[0]
            text = meds["bury"] if item_id == "meds" else facts[item_id]["bury"]
            self._discharge_summary(p, item_id, text, reason_known)

        expected = {}
        for item_id in self.items:
            d = self.d(item_id)
            expected[item_id] = {"n/a": Status.NOT_APPLICABLE, "omit": Status.MISSING, "stale": Status.STALE,
                                 "contradict": Status.CONTRADICTORY}.get(d, Status.PRESENT)
        return RenderedCase(
            case_id=self.plan.case_id, checklist_id=self.plan.checklist_id, reference_date=self.ref,
            plan=self.plan, patient=p, docs=self.docs, evidence=self.evidence, expected=expected,
            wrong_patient_docs=[doc.doc_id for doc in self.docs if doc.patient == "other"])

    # -- documents

    def _letter(self, p: Patient, reason_known: bool, inserts, condition_lines) -> None:
        extra = "Do not include a greeting or sign-off." + ("" if reason_known else " " + NO_REASON_RULE)
        prose = ask(self.llm, f"Write the body of a referral letter from {p.referring_physician} ({p.practice}) to a "
                              f"{self.checklist.specialty} clinic: 2-3 short paragraphs.\n"
                              f"{self.patient_summary(p, reason_known)}" + _rules(self.forbidden, extra),
                    LetterProse, self.forbidden)
        reason_line = f"Reason for referral: {p.referral_reason}"
        parts = [header("REFERRAL LETTER", self.ref, p), "",
                 f"To: {self.checklist.specialty.title()} Clinic Intake",
                 f"From: {p.referring_physician}, {p.practice}", "", "Dear Colleague,", ""]
        if reason_known:
            parts += [reason_line, ""]
        for para in prose.paragraphs:
            parts += [para, ""]
        for _, text in inserts:
            parts += [text, ""]
        for line in condition_lines:
            parts += [line, ""]
        parts += ["Thank you for seeing this patient.", "", p.referring_physician, p.practice]
        planted = ([("reason", reason_line)] if reason_known else []) + list(inserts)
        self.add(RenderedDoc(doc_id="referral_letter", doc_type="referral_letter", doc_date=self.ref,
                             pages=["\n".join(parts) + "\n"]), planted)

    def _note_prose(self, p: Patient, reason_known: bool, older: bool) -> NoteProse:
        kind = "an earlier routine follow-up visit" if older else "the visit at which the referral was decided"
        extra = "" if reason_known and not older else NO_REASON_RULE
        return ask(self.llm, f"Write the narrative parts of a clinic progress note for {kind}: a subjective paragraph "
                             "and 2-5 assessment-and-plan entries, one problem each, without numbering.\n"
                             f"{self.patient_summary(p, reason_known and not older)}" + _rules(self.forbidden, extra),
                   NoteProse, self.forbidden)

    def _note(self, p: Patient, when: date, prose: NoteProse, doc_id: str, inserts, condition_lines) -> None:
        exam = _exam(self.rng)
        parts = [header("PROGRESS NOTE", when, p, f"Provider: {p.referring_physician}"), "",
                 "SUBJECTIVE:", prose.subjective, "", "VITALS:", _vitals(self.rng), "", exam, ""]
        if inserts or condition_lines:
            parts += ["DATA REVIEWED:", ""]
            for _, text in inserts:
                parts += [text, ""]
            for line in condition_lines:
                parts += [line, ""]
        parts += ["ASSESSMENT AND PLAN:"] + [f"{n}. {entry}" for n, entry in enumerate(prose.assessment_plan, 1)]
        self.add(RenderedDoc(doc_id=doc_id, doc_type="progress_note", doc_date=when, pages=["\n".join(parts) + "\n"]),
                 [("progress_note", exam), *inserts])

    def _count_other_docs(self, facts: dict) -> int:
        """Documents besides the progress notes: letter, med list, reports, lab reports, discharge summary."""
        lab_items = [i for i in self.items if SPECS[i].home == "lab" and i in facts]
        labs = (any(self.d(i) in (None, "contradict") for i in lab_items)
                + sum(self.d(i) == "stale" for i in lab_items) + self.plan.wrong_patient_doc)
        reports = sum(i in facts and self.d(i) in (None, "stale", "contradict") for i in ("ecg", "echo"))
        meds = "meds" in self.applicable and self.d("meds") in (None, "contradict")
        bury = any(self.d(i) == "bury" for i in self.items)
        return 1 + meds + reports + labs + bury

    def _notes(self, p: Patient, note_date: date, reason_known: bool, inserts, condition_lines,
               max_older: int = 2) -> None:
        n_older = min(self.rng.randint(0, 2), max_older)  # 1-3 progress notes in total, within MAX_DOCS
        dates = [note_date]
        for _ in range(n_older):
            dates.append(dates[-1] - timedelta(days=self.rng.randint(60, 240)))
        ids = ["progress_note"] if n_older == 0 else [f"progress_note_{k}" for k in range(1, n_older + 2)]
        # Oldest note gets the lowest number, so the ids don't reveal which note is current.
        for k, when in enumerate(sorted(dates)):
            current = when == note_date
            prose = self._note_prose(p, reason_known, older=not current)
            self._note(p, when, prose, ids[k], inserts if current else [], condition_lines if current else [])

    def _reports(self, p: Patient, facts: dict) -> None:
        for item_id, doc_id, title, window in [("ecg", "ecg_report", "12-LEAD ECG REPORT", 180),
                                               ("echo", "imaging_report", "TRANSTHORACIC ECHOCARDIOGRAM REPORT", 365)]:
            if item_id not in facts or self.d(item_id) not in (None, "stale", "contradict"):
                continue
            window = self.window(item_id) or window
            when = self.old(window) if self.d(item_id) == "stale" else self.fresh(window)
            block = facts[item_id]["block"]
            page = header(title, when, p, f"Ordering provider: {p.referring_physician}") + "\n\n" + block + "\n"
            self.add(RenderedDoc(doc_id=doc_id, doc_type="imaging_report" if item_id == "echo" else "unknown",
                                 doc_date=when, pages=[page]), [(item_id, block)])

    def _lab_page(self, p: Patient, when: date, blocks: list[str], provider: str) -> str:
        top = header("LABORATORY REPORT", when, p, f"Ordering provider: {provider}")
        return top + "\n\n" + "\n\n".join(blocks) + "\n"

    def _labs(self, p: Patient, facts: dict) -> None:
        lab_items = [i for i in self.items if SPECS[i].home == "lab" and i in facts]
        current = [i for i in lab_items if self.d(i) in (None, "contradict")]
        stale = [i for i in lab_items if self.d(i) == "stale"]
        planned: list[tuple[date, list[str], str]] = []  # (date, item_ids, patient)
        if current:
            windows = [self.window(i) for i in current if self.window(i)]
            planned.append((self.fresh(min(windows) if windows else None, cap=25), current, "case"))
        for i in stale:
            planned.append((self.old(self.window(i)), [i], "case"))
        if self.plan.wrong_patient_doc:
            planned.append((self.ref - timedelta(days=self.rng.randint(2, 20)), ["lipids", "bmp"], "other"))
        self.rng.shuffle(planned)  # numbering must not reveal which report is stale or foreign
        ids = ["lab_report"] if len(planned) == 1 else [f"lab_report_{k}" for k in range(1, len(planned) + 1)]
        for doc_id, (when, item_ids, who) in zip(ids, planned):
            if who == "other":
                other = self.other_patient(p)
                blocks = [_lipids(self.rng)["block"], _bmp(self.rng)["block"]]
                self.add(RenderedDoc(doc_id=doc_id, doc_type="lab_report", doc_date=when, patient="other",
                                     pages=[self._lab_page(other, when, blocks, other.referring_physician)]))
                continue
            blocks = [facts[i]["block"] for i in item_ids]
            self.add(RenderedDoc(doc_id=doc_id, doc_type="lab_report", doc_date=when,
                                 pages=[self._lab_page(p, when, blocks, p.referring_physician)]),
                     [(i, facts[i]["block"]) for i in item_ids])

    def other_patient(self, p: Patient) -> Patient:
        """A near-identical identity, as in a misfiled report: same surname, other first name, DOB, and an MRN
        with two adjacent digits swapped."""
        rng = self.rng
        first = rng.choice([n for n in ["Ellen", "Evan", "Grace", "Glen", "Rose", "Ross", "Nina", "Neil"]
                            if n != p.first_name])
        digits = list(p.mrn[4:])
        k = next((j for j in range(len(digits) - 1) if digits[j] != digits[j + 1]), None)
        if k is None:
            digits[0] = str((int(digits[0]) % 9) + 1)
        else:
            digits[k], digits[k + 1] = digits[k + 1], digits[k]
        return p.model_copy(update={"first_name": first, "mrn": "MRN-" + "".join(digits),
                                    "dob": p.dob - timedelta(days=rng.randint(400, 1500)),
                                    "referring_physician": "Dr. " + rng.choice(["Angela Ruiz", "Tom Becker"])})

    def _discharge_summary(self, p: Patient, item_id: str, text: str, reason_known: bool) -> None:
        window = self.window(item_id)
        when = self.ref - timedelta(days=self.rng.randint(5, max(5, min(45, int(window * 0.7)) if window else 45)))
        admission = self.rng.choice(["community-acquired pneumonia", "cellulitis of the lower leg",
                                     "elective total knee replacement", "acute kidney stone with obstruction",
                                     "diverticulitis managed without surgery"])
        extra = "Do not describe a physical examination. Do not mention the referral."
        prose = ask(self.llm, f"Write a hospital discharge summary for an admission for {admission}, as exactly "
                              f"{LONG_PAGES} sections (e.g. admission diagnosis, history, hospital course by day, "
                              "consults, procedures, condition at discharge, instructions, follow-up). Each section "
                              "120-200 words.\n" + self.patient_summary(p, False) + _rules(self.forbidden, extra),
                    LongNoteProse, self.forbidden)
        pages = []
        for n, section in enumerate(prose.sections, 1):
            top = header("DISCHARGE SUMMARY", when, p, f"Attending: {p.referring_physician}") if n == 1 \
                else running_header(p, "Discharge summary", n, LONG_PAGES)
            body = f"{section.title.upper()}:\n{section.text}"
            if n == BURY_PAGE:
                paragraphs = section.text.split("\n\n")
                mid = max(1, len(paragraphs) // 2)
                body = f"{section.title.upper()}:\n" + "\n\n".join(paragraphs[:mid] + [text] + paragraphs[mid:])
            pages.append(f"{top}\n\n{body}\n")
        self.add(RenderedDoc(doc_id="discharge_summary", doc_type="unknown", doc_date=when, pages=pages),
                 [(item_id, text)])


def render_case(plan: DefectPlan, checklist: Checklist, llm, seed: int | str = 0) -> RenderedCase:
    """Render one case. Deterministic given the seed and the LLM's answers."""
    before = (llm.calls, llm.input_tokens, llm.output_tokens)
    case = _Builder(plan, checklist, llm, random.Random(f"{seed}:{plan.case_id}")).build()
    case.llm_calls = llm.calls - before[0]
    case.input_tokens = llm.input_tokens - before[1]
    case.output_tokens = llm.output_tokens - before[2]
    return case


def write_case(case: RenderedCase, out_dir: Path) -> Path:
    case_dir = out_dir / case.case_id
    case_dir.mkdir(parents=True, exist_ok=True)
    (case_dir / "rendered.json").write_text(case.model_dump_json(indent=2))
    for doc in case.docs:
        (case_dir / f"{doc.doc_id}.txt").write_text(doc.text(), encoding="utf-8")
    return case_dir


def main() -> None:
    from referral.eval.report import cost_usd
    from referral.llm import LLM

    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--checklist", required=True)
    ap.add_argument("--n", type=int, default=5)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", type=Path, default=Path("data/rendered"))
    args = ap.parse_args()

    raw = json.loads(Path(f"data/checklists/{args.checklist}.json").read_text())
    checklist = Checklist.model_validate(raw)
    rng, llm = random.Random(args.seed), LLM()
    failed = 0
    for n in range(args.n):
        plan = make_plan(f"{args.checklist}-{args.seed}-{n:03d}", raw, rng)
        defects = {p.item_id: p.defect for p in plan.items if p.defect}
        try:
            case = render_case(plan, checklist, llm, seed=args.seed)
        except (LeakError, ValueError) as err:
            failed += 1
            print(f"FAIL {plan.case_id}: {err}")
            continue
        write_case(case, args.out)
        pages = sum(len(d.pages) for d in case.docs)
        print(f"ok   {case.case_id}: {defects or 'complete'}"
              f"{' +wrong_patient' if case.wrong_patient_docs else ''} | {len(case.docs)} docs, {pages} pages | "
              f"{case.llm_calls} calls, {case.input_tokens} in / {case.output_tokens} out tokens")
    cost = cost_usd(llm.input_tokens, llm.output_tokens)
    print(f"\n{args.n - failed}/{args.n} rendered with {llm.name}: {llm.calls} calls, {llm.input_tokens} input / "
          f"{llm.output_tokens} output tokens" + (f", ${cost:.4f}" if cost is not None else
                                                  " (set LLM_PRICE_*_PER_MTOK in .env for cost)"))


if __name__ == "__main__":
    main()
