# CLAUDE.md

Context for coding agents working in this repo.

## Project

**Specialist Referral Completeness Assistant**: NYU Generative AI Engineering final project (Fall 2026).
Team: Shreyas Kaldate (pipeline), Neel Gajiwala (data), Adhyayan Verma (evaluation).

Given a referral packet (2–8 PDFs for one patient: referral letter, progress notes, labs, imaging, medication
list) and a specialty clinic's checklist, the system gives every checklist item one status with cited
evidence, and drafts follow-up requests for gaps. Intake staff review everything. **The system never
contacts anyone on its own.**

| Status | Rule |
| --- | --- |
| `present` | Verified quote supports the item, within its recency window |
| `missing` | Nothing in the packet supports it, after the absence check |
| `contradictory` | Verified quotes from at least two different documents conflict |
| `stale` | Verified evidence, date known, newest evidence older than the item's `recency_days` |
| `needs_review` | Evidence could not be verified: citation failed, or date missing/unreadable |
| `not_applicable` | Conditional item whose condition does not hold |

Out of scope: triage or urgency decisions, prior authorization, handwritten notes, EHR integration,
real patient data.

## How this project is graded (shapes every decision)

The course rewards rigorous evaluation and justified decisions, not feature count:
- Start from the simplest credible baseline (B1) and add a component only when evaluation shows the
  failure it fixes. Every component must be switchable off for ablations.
- Evaluation is part of the system: outcome metrics, trajectory metrics (retrieval, error attribution),
  edge-case slices, cost and latency.
- Failure analysis and insight matter more than a polished demo.

When proposing changes, say which observed failure or experiment they serve.

## Deadlines

- Milestone 3 (Nov 19, 2026): results, failures and iteration, 2–3 pages
- Final submission (Dec 3, 2026, 11:59 PM ET): PDF report + code + 8–10 min demo

## Architecture

```
PDFs → ingest + chunk → identity check → case-scoped retrieval (BM25 / dense / hybrid)
     → LLM item verifier → absence check (before Missing) → citation + recency validator
     → conflict checker → follow-up drafts → review UI
```

Key design decisions and why (several answer professor feedback, so keep them):
- **Retrieval is filtered to one `case_id`.** Another patient's record must never be cited. A global index
  exists only as ablation E2 (`scope="global"`).
- **Identity check** (`ingest/identity.py`): documents whose name/DOB/MRN don't match the referral letter,
  or have no identifiers, are quarantined and flagged, so a wrongly merged document can't create false
  Contradictory labels.
- **Absence check** (`pipeline/absence.py`): a retrieval miss looks like a missing item, so Missing is only
  accepted after a wider search plus a synonym scan over every chunk, then re-verification.
- **The LLM only proposes** present / missing / contradictory / not_applicable. Stale and Needs review are
  decided in code (`pipeline/validator.py`) in a fixed order: verify citations → check date → check window.
- **Recency windows live in the checklist JSON** (`recency_days` per item), owned by the receiving clinic.
  Age is measured from the referral date.

## Layout

```
src/referral/
  schemas.py            Pydantic models shared by everything; change with care
  config.py             settings from .env
  llm.py                Anthropic wrapper: structured JSON output, one repair retry, token counting
  ingest/               pdf_loader.py, chunker.py (page + char offsets), identity.py
  retrieval/hybrid.py   BM25, ChromaDB dense, reciprocal rank fusion, CaseRetriever
  pipeline/             verifier.py (LLM), validator.py, absence.py, conflicts.py, run.py (PipelineConfig)
  baselines/            b0_keywords.py (lower bound), b1_long_context.py (primary baseline)
  eval/                 gold.py (label format), metrics.py, harness.py (CLI)
  api.py                FastAPI backend for the review UI
datagen/                synthetic case generator (only defect_plan.py exists so far)
data/checklists/        checklist JSON (cardiology-v1.json is the only one so far)
data/cases/<case_id>/   PDFs + gold.json; not committed
runs/<name>/            per-run predictions + metrics.json; not committed
tests/                  unit tests; must pass without an API key
```

## Commands

```bash
pip install -e ".[dev]"
cp .env.example .env        # ANTHROPIC_API_KEY, LLM_MODEL
pytest -q
ruff check src tests datagen
python -m referral.eval.harness --system b0 --split dev --name b0-dev
python -m referral.eval.harness --system b1 --split dev --name b1-dev
python -m referral.eval.harness --system pipeline --split dev --name pipe-dev --retrieval bm25
uvicorn referral.api:app --reload
```

## Dataset

- 4 specialty checklists planned (cardiology, gastroenterology, endocrinology, neurology), adapted from
  public referral guidelines (UC Davis Health, Partnership HealthPlan of California, Health Plan of Nevada).
- 150 synthetic cases: patient seed (Synthea, or LLM profiles if Synthea is too slow) → defect plan
  (omit, contradict, stale, bury, abbreviate, wrong-patient doc) → LLM-rendered documents in MTSamples
  note style → OCR-style noise → PDF. Labels are known by construction and written to `gold.json`.
- Target mix: 30% complete, 40% with 1–3 missing items, 25% abbreviation-only evidence, 20% contradictions,
  20% buried past page 10, 15% stale, 5% wrong-patient document. Packets 3–60 pages.
- Splits: 40 dev, 110 held-out test, 15 hand-written adversarial.

## Metrics and targets

- Gap recall ≥ 0.90 and precision ≥ 0.80 (positive = missing or contradictory)
- Status macro-F1 ≥ 0.80; citation precision ≥ 0.95
- Cost ≤ $0.10 and latency ≤ 60 s per case
- Also reported: false-Missing rate (by cause), false-Contradictory rate on the wrong-patient slice,
  retrieval recall@k (k = 3, 5, 10), error attribution (retrieval miss / validator reject / reasoning error)
- Bootstrap 95% CIs over cases; paired McNemar tests between systems

Planned experiments: E1 B0 vs B1 vs pipeline; E2 case vs global retrieval; E3 BM25 vs dense vs hybrid,
chunk size, k; E4 evidence position and packet length; E5 with and without validator and conflict checker;
plus the absence-check ablation.

## Current status (Nov 8, 2026)

Written and unit-tested: schemas, chunking, identity check, validator rules, absence-check candidate
search, dose/laterality conflict checks, core metrics, defect planner.

Written but never run end to end (expect bugs): LLM wrapper, verifier, B1, `pipeline/run.py`, harness,
dense retrieval, API.

Not written yet, roughly in priority order:
1. Data generator after the defect plan: render documents, add noise, write PDFs and `gold.json`, leak check
2. Harness saving retrieved chunks per item, so recall@k and error attribution can be computed
3. Absence check final step: long-context read of the whole packet before accepting Missing
4. Bootstrap CIs in harness output; per-run report
5. Other three checklists; LLM judge for citations; referral reason and handoff summary
6. React review UI (last)

## Rules

- **Never use real patient data.** All cases are synthetic.
- **Never run on the test split while developing.** Tune on dev only; test is for the final frozen run.
- Keep every pipeline stage switchable through `PipelineConfig` so ablations stay possible.
- Tests must run offline without an API key; mock or avoid LLM calls in tests.
- Keep `schemas.py` and the `gold.json` format stable; other teammates' code depends on them.
- Run `pytest` and `ruff` before committing.
- **Do not add Claude or any AI as a co-author in commits**, and no AI attribution lines in commit
  messages or PR descriptions.
