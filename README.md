# Specialist Referral Completeness Assistant

NYU Generative AI Engineering final project (Fall 2026) · Shreyas Kaldate, Neel Gajiwala, Adhyayan Verma

Given a referral packet (2–8 PDFs for one patient) and a specialty clinic's checklist, the system labels
every checklist item with cited evidence and drafts follow-up requests for gaps. Staff review everything;
the system never contacts anyone.

| Status | Meaning |
| --- | --- |
| `present` | Verified quote supports the item, within its recency window |
| `missing` | Nothing in the packet supports it, confirmed by the absence check |
| `contradictory` | Two documents give conflicting values |
| `stale` | Verified evidence, date known and older than the item's window |
| `needs_review` | Evidence could not be verified (citation failed, date missing or unreadable) |
| `not_applicable` | Conditional item whose condition does not hold |

## Pipeline

```
PDFs → ingest + chunk → identity check → case-scoped retrieval (BM25 / dense / hybrid)
     → LLM item verifier → absence check (before Missing) → citation + recency validator
     → conflict checker → follow-up drafts → review UI
```

Every optional stage can be switched off in `PipelineConfig` for ablations.

## Layout

```
src/referral/
  schemas.py            shared Pydantic models (statuses, checklist, chunks, results)
  ingest/               PDF loading, chunking with offsets, wrong-patient identity check
  retrieval/hybrid.py   BM25 + ChromaDB, reciprocal rank fusion, case_id filter
  pipeline/             verifier (LLM), validator (code), absence check, conflict checker, run.py
  baselines/            B0 keyword rules, B1 long-context single call
  eval/                 gold format, metrics, harness CLI
  api.py                FastAPI backend for the review UI
datagen/                synthetic case generator (Synthea → defect plan → LLM render → noise)
data/checklists/        clinic checklists as JSON (recency windows live here)
tests/                  unit tests, no API key needed
```

## Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
cp .env.example .env   # add ANTHROPIC_API_KEY
pytest
```

## Running

```bash
python -m referral.eval.harness --system b0 --split dev --name b0-dev
python -m referral.eval.harness --system b1 --split dev --name b1-dev
python -m referral.eval.harness --system pipeline --split dev --name pipe-dev --retrieval hybrid
uvicorn referral.api:app --reload
```

Cases live in `data/cases/<case_id>/` as PDFs plus `gold.json` (not committed).

## Ownership

- **Shreyas:** pipeline — ingestion, retrieval, verifier, validator, absence and conflict checks
- **Neel:** data — checklists, Synthea seeding, document generator, noise, leak check
- **Adhyayan:** evaluation — harness, baselines, metrics, LLM-judge validation, review UI
