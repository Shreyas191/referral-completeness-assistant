# Synthetic case generator

Builds referral packets whose gold labels are known by construction.

1. **Patient seed:** run Synthea and export FHIR JSON into `data/synthea/`.
2. **Defect plan:** `defect_plan.py` picks a specialty, a checklist, and which items to omit, contradict,
   make stale, bury past page 10, abbreviate, or add a wrong-patient document for.
3. **Render:** an LLM writes each document from the seed and plan in MTSamples note style.
4. **Noise:** OCR-style character errors, headers and footers; render to PDF.
5. **Gold:** write `gold.json` (see `src/referral/eval/gold.py`) next to the PDFs in `data/cases/<case_id>/`.
6. **Leak check:** flag cases where an omitted item's synonyms appear in the rendered text anyway.

Owner: Neel.
