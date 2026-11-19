# Synthetic case generator

Builds referral packets whose gold labels are known by construction.

One command does everything, reusing any case already rendered (so a rebuild costs no API calls):

```bash
python -m datagen.build --checklist cardiology-v1 --n 5 --seed 0 --split dev   # --noise 0 for clean text
```

The splits (issue #10) use `--quota`, which picks plans so the split matches the dataset mix exactly:

```bash
python -m datagen.build --checklist cardiology-v1 --n 40  --seed 10 --split dev  --quota            # committed
python -m datagen.build --checklist cardiology-v1 --n 110 --seed 20 --split test --quota --out data/test_cases
```

Dev cases (`data/cases/cardiology-v1-dev-*`) are committed. Test cases go to `data/test_cases/`, which is
gitignored and must not be run before the freeze; the final run passes `--cases-dir data/test_cases`. Each
case folder carries its `rendered.json`, so PDFs and gold can be rebuilt without the API.

1. **Patient profile:** written by the LLM (`render.py`). Code fixes age, sex, DOB, MRN, the presenting
   problem and name initials, so cases differ and identities are exact. (Synthea was dropped for time.)
2. **Defect plan:** `defect_plan.py` picks which items to omit, contradict, make stale, bury past page 10,
   or abbreviate, and whether to add a wrong-patient document. Each defect only goes to items where it can
   exist, and each plan decides whether conditional items apply (e.g. on warfarin).
3. **Render:** `render.py` (`python -m datagen.render`). Code writes every fact the labels depend on (values,
   dates, which document and page); the LLM writes only the narrative prose around them and must avoid
   every checklist term. Output: `data/rendered/<case_id>/rendered.json` plus `.txt` documents.
4. **Noise and PDF:** `build.py` adds OCR-style letter noise (never digits, never the header block), lays
   documents out as PDFs with a practice header and page footer (no dates), and reads them back with the
   pipeline's loader.
5. **Gold:** `gold.json` (see `src/referral/eval/gold.py`) is written from what the loader reads, with each
   evidence `text` being the whole supporting block (e.g. the full BMP panel), so any correct citation
   inside it scores as correct.
6. **Checks and leak check:** `check_cases.py` checks verbatim evidence, recency against the label, identity
   quarantine and dose conflicts, and flags any missing, not-applicable or stale item mentioned elsewhere.

Owner: Neel.
