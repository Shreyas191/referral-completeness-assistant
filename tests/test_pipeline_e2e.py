"""End-to-end runs of the pipeline and harness on one text-file case, with a fake LLM (no API key)."""
import json
import re
from datetime import date
from pathlib import Path

import pytest
from fakes import PASSAGE_RE, FakeLLM

from referral.eval.harness import load_checklist, run_split
from referral.ingest.pdf_loader import load_case
from referral.pipeline.absence import LongContextOutput, windows
from referral.pipeline.run import PipelineConfig, run_case
from referral.pipeline.trace import CaseTrace
from referral.pipeline.verifier import Evidence, VerifierOutput
from referral.schemas import Status

ROOT = Path(__file__).resolve().parents[1]
CHECKLISTS = ROOT / "data/checklists"
HEADER = "Patient Name: Jane Doe\nDOB: 03/14/1961\nMRN: AB12345"
FILLER = "\n\n".join(f"Visit narrative paragraph {n}: patient discussed diet, sleep and exercise." for n in range(12))

DOCS = {
    "referral_letter": f"Date: 11/01/2026\n{HEADER}\n\nReason for referral: exertional chest pain.\n\n"
                       "Current medications: Metformin 1000 mg twice daily.",
    "labs": f"Date: 10/20/2026\n{HEADER}\n\nBMP: creatinine 1.1 mg/dL, potassium 4.2 mmol/L.",
    "old_labs": f"Date: 01/05/2025\n{HEADER}\n\nLipid panel: LDL 150 mg/dL, HDL 41 mg/dL.",
    "med_list": f"Date: 10/25/2026\n{HEADER}\n\nCurrent medications: Metformin 500 mg twice daily.",
    # Echo evidence sits on page 2, after a long note, and only as the abbreviation TTE.
    "progress_note": f"Date: 10/28/2026\n{HEADER}\n\nProgress note. Cardiac exam: regular rate, no murmur.\n\n"
                     f"{FILLER}\f{FILLER}\n\nTTE 10/15: EF 55%, normal wall motion.",
    "other_patient_lab": "Date: 10/22/2026\nPatient Name: John Smith\nDOB: 01/01/1950\nMRN: ZZ99999\n\n"
                         "BMP: creatinine 3.0 mg/dL.",
    # Ordered-not-resulted distractor: outranks the real TTE result for the echo query.
    "imaging_order": f"Date: 10/02/2026\n{HEADER}\n\nImaging order: transthoracic study report requested; "
                     "report to follow.",
}
EXPECTED = {
    "reason": Status.PRESENT,
    "ecg": Status.MISSING,              # never mentioned
    "echo": Status.PRESENT,             # buried, abbreviation only
    "lipids": Status.STALE,             # 22 months old, window 365 days
    "bmp": Status.PRESENT,              # the other patient's creatinine 3.0 must not interfere
    "meds": Status.CONTRADICTORY,       # metformin 1000 mg vs 500 mg
    "progress_note": Status.PRESENT,
    "inr": Status.NOT_APPLICABLE,       # only required on warfarin
}


@pytest.fixture
def case_dir(tmp_path):
    d = tmp_path / "cases" / "hand-000"
    d.mkdir(parents=True)
    for name, text in DOCS.items():
        (d / f"{name}.txt").write_text(text)
    gold = {
        "case_id": "hand-000", "checklist_id": "cardiology-v1", "reference_date": "2026-11-01", "split": "dev",
        "slices": ["buried", "abbreviation", "contradiction", "stale", "wrong_patient"],
        "wrong_patient_docs": ["other_patient_lab"],
        "items": [{"item_id": k, "status": v.value} for k, v in EXPECTED.items()],
    }
    gold["items"][2]["evidence"] = [{"doc_id": "progress_note", "page": 2, "text": "TTE 10/15: EF 55%"}]
    (d / "gold.json").write_text(json.dumps(gold))
    return d


def run(case_dir, llm, **cfg):
    trace = CaseTrace(case_id="")
    result = run_case(load_case(case_dir), load_checklist("cardiology-v1", CHECKLISTS), date(2026, 11, 1),
                      llm, PipelineConfig(**cfg), trace=trace)
    return {r.item_id: r for r in result.items}, result, trace


def test_full_pipeline_gets_every_item_right(case_dir):
    items, result, trace = run(case_dir, FakeLLM())
    assert {k: v.status for k, v in items.items()} == EXPECTED
    assert result.quarantined_docs == ["other_patient_lab"]
    assert {f.item_id for f in result.followups} == {"ecg", "lipids", "meds"}
    assert trace.item("echo").final_status == "present" and trace.item("echo").ranked


class EchoesDescription(FakeLLM):
    """A model that returns the item's description as its item_id, as gpt-6-luna did for `reason`."""

    def structured(self, system, user, schema):
        out = super().structured(system, user, schema)
        if isinstance(out, VerifierOutput):
            out.item_id = re.search(r"Checklist item \S+: (.*)", user)[1].split("\n")[0]
        return out


def test_ablation_without_validator_keeps_checklist_item_ids_and_chunk_pages(case_dir):
    # Regression: with the validator off, a model-written item_id reached draft_followups and raised KeyError.
    items, result, _ = run(case_dir, EchoesDescription(), validator=False)
    assert set(items) == set(EXPECTED)
    assert {f.item_id for f in result.followups} <= set(EXPECTED)
    echo = items["echo"]
    assert echo.citations and all(c.page > 0 and c.doc_date for c in echo.citations)


def test_value_expansion_is_traced_and_changes_no_status_here(case_dir):
    items, _, trace = run(case_dir, FakeLLM(), value_expansion=True)
    assert {k: v.status for k, v in items.items()} == EXPECTED
    assert all(not set(t.expansion) & set(t.shown) for t in trace.items)
    assert all("other_patient_lab" not in c for t in trace.items for c in t.expansion)  # quarantined stays out


def test_wide_recheck_recovers_a_retrieval_miss(case_dir):
    # With k=1 the verifier only sees the imaging order, so the real TTE result is a retrieval miss.
    items, _, trace = run(case_dir, FakeLLM(), top_k=1)
    echo = trace.item("echo")
    assert echo.first_status == "missing" and echo.absence_status == "present"
    assert items["echo"].status == Status.PRESENT
    assert "absence_check_overturned" in items["echo"].flags


def test_reasoning_miss_on_shown_evidence_is_left_to_the_final_read(case_dir):
    # Evidence was retrieved but misjudged: the wide re-check adds nothing new, the final read catches it.
    items, _, trace = run(case_dir, FakeLLM(miss_first={"echo"}))
    assert trace.item("echo").absence_status is None
    assert items["echo"].flags == ["long_context_overturned"]


def test_long_context_read_overturns_when_per_item_looks_fail(case_dir):
    llm = FakeLLM(miss_per_item={"echo"})
    items, _, trace = run(case_dir, llm)
    assert items["echo"].status == Status.PRESENT
    assert "long_context_overturned" in items["echo"].flags
    assert llm.long_context_calls == 1  # one call per case, covering every still-missing item
    assert trace.item("echo").long_context_status == "present"


class FinalReadOverreaches(FakeLLM):
    """The whole-packet read calls the never-mentioned ECG Present from a quote that does not name it,
    as gpt-6-luna did with "Patient is on warfarin." for the medication list (dev-010)."""

    def structured(self, system, user, schema):
        out = super().structured(system, user, schema)
        if isinstance(out, LongContextOutput):
            letter = next(m for m in PASSAGE_RE.finditer(user) if "chest pain" in m["text"])
            overreach = VerifierOutput(item_id="ecg", status="present", rationale="fake: overreach",
                                       evidence=[Evidence(chunk_id=letter["id"], quote=letter["text"].strip())])
            out.items = [overreach if o.item_id == "ecg" else o for o in out.items]
        return out


def test_strict_long_context_rejects_a_present_without_an_item_term(case_dir):
    loose, _, _ = run(case_dir, FinalReadOverreaches(), strict_long_context=False)
    assert loose["ecg"].status == Status.PRESENT
    items, _, trace = run(case_dir, FinalReadOverreaches(), strict_long_context=True)
    assert items["ecg"].status == Status.MISSING and items["ecg"].flags == ["long_context_rejected"]
    assert trace.item("ecg").long_context_status == "present"  # the read's proposal is still traced
    assert items["inr"].status == Status.NOT_APPLICABLE  # condition calls are untouched


def test_strict_long_context_keeps_a_rescue_that_names_the_item(case_dir):
    # Echo evidence is the abbreviation "TTE", a listed synonym, so the final read's rescue stands.
    items, _, _ = run(case_dir, FakeLLM(miss_per_item={"echo"}), strict_long_context=True)
    assert items["echo"].status == Status.PRESENT and "long_context_overturned" in items["echo"].flags


def test_ablation_without_long_context_leaves_false_missing(case_dir):
    items, _, _ = run(case_dir, FakeLLM(miss_per_item={"echo"}), absence_long_context=False)
    assert items["echo"].status == Status.MISSING


def test_ablation_without_identity_check_creates_false_contradiction(case_dir):
    items, _, _ = run(case_dir, FakeLLM(), identity_check=False)
    # This patient's creatinine 1.1 vs the merged-in patient's 3.0: the error the professor asked about.
    assert items["bmp"].status == Status.CONTRADICTORY


def test_harness_writes_traces_and_recall(case_dir, tmp_path):
    metrics = run_split("pipeline", "dev", tmp_path / "run", PipelineConfig(), FakeLLM(),
                        cases_dir=case_dir.parent, checklists_dir=CHECKLISTS)
    assert metrics["cases"] == 1 and metrics["gap_recall"] == 1.0 and metrics["gap_precision"] == 1.0
    assert metrics["recall@10"] == 1.0
    assert metrics["foreign_citation_rate"] == 0.0
    assert (tmp_path / "run" / "hand-000.trace.json").exists()


def test_harness_writes_report_with_cis_slices_and_provenance(case_dir, tmp_path):
    metrics = run_split("pipeline", "dev", tmp_path / "run", PipelineConfig(), FakeLLM(),
                        cases_dir=case_dir.parent, checklists_dir=CHECKLISTS)
    saved = json.loads((tmp_path / "run" / "metrics.json").read_text())
    assert saved["git_commit"] and saved["case_ids"] == ["hand-000"] and saved["run"] == "run"
    assert saved["ci95"]["gap_recall"] == [1.0, 1.0]
    assert {r["slice"]: r["cases"] for r in saved["slices"]}["wrong_patient"] == 1
    assert metrics["error_attribution"] == dict.fromkeys(metrics["error_attribution"], 0)
    report = (tmp_path / "run" / "report.md").read_text(encoding="utf-8")
    assert "## Slices" in report and "## Run-log row" in report


def test_error_attribution_and_compare_on_a_retrieval_miss(case_dir, tmp_path):
    from referral.eval.compare import compare
    from referral.eval.errors import from_run_dir

    kwargs = dict(cases_dir=case_dir.parent, checklists_dir=CHECKLISTS)
    run_split("pipeline", "dev", tmp_path / "full", PipelineConfig(), FakeLLM(), **kwargs)
    # k=1 with no absence check: the TTE result is never shown, so echo is a false Missing from retrieval.
    weak = run_split("pipeline", "dev", tmp_path / "k1", PipelineConfig(top_k=1, absence_check=False),
                     FakeLLM(), **kwargs)
    assert weak["error_attribution"]["retrieval_miss"] >= 1
    errors = json.loads((tmp_path / "k1" / "errors.json").read_text())
    echo = next(r for r in errors if r["item_id"] == "echo")
    assert echo["cause"] == "retrieval_miss" and echo["gold"] == "present" and echo["pred"] == "missing"
    assert "retrieval_miss: examples" in (tmp_path / "k1" / "errors.md").read_text(encoding="utf-8")

    assert from_run_dir(tmp_path / "k1") == weak["error_attribution"]  # rebuilt from the saved run alone

    text = compare(tmp_path / "full", tmp_path / "k1")
    assert "config.top_k | 5 | 1" in text and "| status correct |" in text


def test_e2_global_scope_cites_other_patients(case_dir, tmp_path):
    # A second patient in the same split, with their own BMP and lipid values.
    other = case_dir.parent / "hand-001"
    other.mkdir()
    hdr = "Patient Name: Raj Patel\nDOB: 07/02/1958\nMRN: CD67890"
    (other / "referral_letter.txt").write_text(f"Date: 11/03/2026\n{hdr}\n\nReason for referral: palpitations.")
    (other / "labs.txt").write_text(f"Date: 10/30/2026\n{hdr}\n\nBMP: creatinine 0.9 mg/dL.\n\nLipid panel: LDL 98.")
    gold = json.loads((case_dir / "gold.json").read_text())
    gold.update(case_id="hand-001", reference_date="2026-11-03", slices=[], wrong_patient_docs=[])
    (other / "gold.json").write_text(json.dumps(gold))

    kwargs = dict(cases_dir=case_dir.parent, checklists_dir=CHECKLISTS)
    scoped = run_split("pipeline", "dev", tmp_path / "case", PipelineConfig(), FakeLLM(), **kwargs)
    pooled = run_split("pipeline", "dev", tmp_path / "global", PipelineConfig(scope="global"), FakeLLM(), **kwargs)
    assert scoped["foreign_citation_rate"] == 0.0
    assert pooled["foreign_citation_rate"] > 0.0


def test_windows_split_long_packets():
    from conftest import make_chunk
    chunks = [make_chunk("x" * 100, doc_id=f"d{n}") for n in range(5)]
    assert [len(w) for w in windows(chunks, max_chars=250)] == [2, 2, 1]
