# Milestone 3: Results, Failures and Iteration

**Specialist Referral Completeness Assistant.** Shreyas Kaldate, Neel Gajiwala, Adhyayan Verma.
Updated Nov 25, 2026 (the Nov 19 version is commit `43f6c54`).

Numbers come from saved dev runs (`openai:gpt-6-luna`); older runs were rescored for newer metrics from their
saved predictions. Full list: [run-log.md](run-log.md).

## 1. System implemented so far

Ingest and chunk → identity check → BM25 retrieval filtered to one case → **value-term expansion** →
per-item LLM verifier → absence check (wide re-check, then a whole-packet read held to **item-specific
evidence**) → citation and recency validator → conflict checker → follow-up drafts. Every stage has a
`PipelineConfig` switch. Evaluation adds bootstrap CIs, McNemar tests, slices, error attribution, citation
precision at three levels (document, page, passage) and macro-F1 over the statuses that occur in gold.
Not built: three more checklists, dense and hybrid retrieval, the review UI.

**Data.** 40 synthetic dev cases, cardiology only: 320 items (243 present, 33 missing, 30 not applicable,
8 contradictory, 6 stale), packets of 4–19 pages. The test split has not been touched.

## 2. Baseline results

| Run (dev, n = 40) | Gap recall | Gap prec. | Macro-F1 (gold classes) | Accuracy | Cite: doc | Cite: passage | $/case | s/case |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| B0 keywords | 0.780 [0.652, 0.886] | 0.492 | 0.323 [0.294, 0.343] | 0.850 | 0.992 | 0.000 | 0 | 0.0 |
| B1 whole packet, run 1 | 0.902 [0.793, 0.979] | 1.000 | 0.936 [0.856, 0.981] | 0.959 | 0.895 | 0.444 | 0.0010 | 10.4 |
| B1 whole packet, run 2 | 0.902 [0.800, 0.977] | 1.000 | 0.951 [0.888, 0.986] | 0.963 | 0.893 | 0.500 | 0.0010 | 9.1 |
| Pipeline as of Nov 19 | 0.854 [0.750, 0.947] | 1.000 | 0.924 [0.823, 0.978] | 0.978 | 0.987 | 0.635 | 0.0016 | 18.3 |
| **Current pipeline, run 1** | 0.976 [0.921, 1.000] | 1.000 | 0.978 [0.926, 1.000] | 0.991 | 0.997 | 0.642 | 0.0017 | 15.5 |
| **Current pipeline, run 2** | 0.927 [0.838, 1.000] | 1.000 | 0.967 [0.906, 1.000] | 0.988 | 0.993 | 0.653 | 0.0017 | 16.0 |
| Nov 19 pipeline, no absence check | 0.902 [0.816, 0.977] | 0.597 | 0.752 [0.672, 0.812] | 0.909 | 0.993 | 0.631 | 0.0012 | 14.8 |
| Nov 19 pipeline, no long-context read | 0.927 [0.852, 1.000] | 0.594 | 0.754 [0.682, 0.807] | 0.903 | 0.993 | 0.617 | 0.0013 | 14.3 |
| Nov 19 pipeline, no validator | 0.927 [0.852, 1.000] | 1.000 | 0.765 [0.704, 0.798] | 0.972 | 0.894 | 0.581 | 0.0016 | 15.8 |
| Nov 19 pipeline, global scope (E2) | 0.000 [0.000, 0.000] | 0.000 | 0.255 [0.209, 0.304] | 0.753 | 0.029 | 0.159 | 0.0022 | 24.4 |

Table 1 covers all runs. Brackets are 95% bootstrap CIs over cases. Gap = Missing or Contradictory.

B1 (whole packet in one call) is a strong baseline: 12–13 of 320 items wrong, no false gaps, far above B0
(McNemar p ≤ 3.62e-07 for both runs). Its two runs agree closely.

## 3. Current results

The current pipeline gets 3 and 4 items wrong in its two runs, with no false gaps. **It now beats B1 on
status correctness in all four pairings of their runs (McNemar p = 0.013 to 0.039)**; on Nov 19 it did not
(p = 0.18). On gap detection the difference is not significant (p = 0.25 and 1). It costs $0.0017 per case
against B1's $0.0010 and takes 15–16 s against 9–10 s.

## 4. Experiments and comparisons completed

E1 (B0, B1, pipeline; also on the 5 hand-made cases), E2 (global scope), E5 (validator off, conflict checker
off), both absence-check ablations, identity check off (2 cases), and two targeted fixes, each run twice.
B1 and the pipeline were repeated to size run-to-run noise. Not run yet: E3 and E4, which need longer packets.

## 5. What worked

- **Value-term expansion.** The two contradictions missed in all three earlier pipeline runs were caught in
  every run with it; all 44 items it added chunks to ended correct.
- **Case scoping.** Global scope sends 96.9% of citations to other patients and gap recall to 0.000.
- **The absence check**, for conditional items: 26–28 INR items per run corrected to Not applicable.
- **The validator.** Without it, 0 of 6 stale items are found and right-document citations fall to 0.894.
- **Identity check.** All 3 wrong-patient documents quarantined; none wrongly in the other 38 cases.

## 6. What failed

- **Never fired:** the wide re-check (retrieval does not miss on 4–19 page packets; recall@3 is 1.000) and
  the conflict checker (it only re-checks items already labelled Present).
- **Untestable live:** the stricter long-context read; the read did not overreach in either run.
- **Single runs:** repeating the Nov 19 pipeline moved gap recall from 0.854 to 0.902 with no code change.
- **The dev set is too easy** to show where retrieval helps over long context.

## 7. Failure analysis and likely causes

| Category | Current pipeline (2 runs) | B1 (2 runs) | Example |
| --- | --- | --- | --- |
| Present or Stale sent to Needs review after a failed citation | 3 | 0 | `dev-000` progress note (gold Stale); rationale names the right note dated 05/03/2026 |
| Contradiction missed | 2 | 5 | `dev-023` bmp: "Most recent potassium is 3.2 mmol/L" vs "Potassium: 5.1 mmol/L (H)", both shown |
| Missing called Present | 2 | 2 | `dev-038` reason, every run: "Atrial fibrillation remains under ongoing management. Referral placed." |
| Item dropped from output | 0 | 9 | `dev-011` reason: B1 fills the top-level reason field and omits the item |
| Model chose Needs review itself | 0 | 9 | `dev-025` inr: "does not establish whether the patient is taking warfarin" |

1. **Most of the pipeline's advantage is reliability, not reasoning.** Asking per item means no item can be
   dropped and the model never answers Needs review itself; B1 lost 18 items that way over its two runs.
2. **Failed citations are now the largest error type.** The rejected quote is not saved, so the cause cannot
   be confirmed from the runs; the long header-and-exam chunks and OCR noise ("SUBJEECTIVE") are suspects.
3. **The other errors are the model:** contradictions missed with both documents shown, and `dev-038`
   taking "Referral placed." as a reason for referral (a first-pass error no fix touches).

**Citation quality.** 99.3–99.7% of citations hit the right document and page (B1: 89.3–89.5%); the passage
score is lower mostly because quotes are header or date lines ("BASIC METABOLIC PANEL"), not the value.

## 8. Changes attempted and their effects

| Change | Measured effect |
| --- | --- |
| Case scoping (M2 feedback) | Foreign citations 0.000 vs 0.969; gap recall 0.854 vs 0.000. Kept. |
| Absence check (M2 feedback) | Gap precision 1.000 vs 0.597 without it, from INR condition calls, not retrieval misses. Kept. |
| Identity check (M2 feedback) | Off (2 cases): one Missing item called Present from another patient's lab. Kept. |
| Stale vs Needs review rule, recency windows (M2 feedback) | Stale items correct: 5 and 6 of 6 now, 0 of 6 without the validator. Per-clinic windows untested. |
| Value-term expansion | Contradictions caught 6, 8, 8, 6 of 8 in four runs, vs 5, 6, 6 without. Kept. |
| Stricter long-context read | Never fired live; replayed on six earlier runs it rejects all 4 wrong overturns and nothing else. Kept. |
| Macro-F1 over gold classes | Two runs with equal accuracy (0.991): 0.967 and 0.827 on the old metric, 0.967 and 0.993 on the new. |

## 9. Remaining gaps relative to success criteria

| Target | Current pipeline (run 1 / run 2) | Status |
| --- | --- | --- |
| Gap recall ≥ 0.90 | 0.976 / 0.927 | Met in both runs; run 2's CI goes down to 0.838 |
| Gap precision ≥ 0.80 | 1.000 / 1.000 | Met |
| Macro-F1 ≥ 0.80 (gold classes) | 0.978 / 0.967 | Met |
| Citation precision ≥ 0.95 | passage 0.642 / 0.653; document 0.997 / 0.993 | Met at document and page level, missed at passage level |
| Cost ≤ $0.10, latency ≤ 60 s per case | $0.0017; 15.5 / 16.0 s mean (21.0 s max) | Met |

**Limitations.** 40 synthetic cases, one checklist, one model, two runs per configuration, 8 contradictions.
Both fixes were designed on the dev cases they are measured on; the test split is the real check.

## 10. Prioritized plan, Nov 25 – Dec 3

1. **Harder dev cases (Neel, Nov 25–27):** packets toward 60 pages, more contradictions and stale items, so
   E3 and E4 can run and the whole-packet read is tested on real retrieval misses.
2. **Save rejected quotes** in the trace, then fix the failed-citation cause if it is mechanical.
3. **E3 and E4** on the harder cases (Nov 27–28).
4. **Freeze Nov 29:** B0, B1 and the pipeline on the 110 test cases, twice each. Report and demo Nov 30 – Dec 3.
