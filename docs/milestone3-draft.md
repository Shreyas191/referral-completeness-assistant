# Milestone 3: Results, Failures and Iteration (draft)

**Specialist Referral Completeness Assistant.** Shreyas Kaldate, Neel Gajiwala, Adhyayan Verma. Nov 19, 2026.

Headline numbers come from `runs/*/metrics.json` and `referral.eval.compare`; item counts in the failure
analysis were counted from the saved predictions and traces of the same runs. Commit `fa8fb45`, model
`openai:gpt-6-luna`, one run per configuration.

## 1. System implemented so far

The full path runs end to end on generated PDFs: ingest and chunk (page and character offsets) → identity
check → BM25 retrieval filtered to one case → per-item LLM verifier → absence check (wide re-check, then a
whole-packet read) before Missing is accepted → citation and recency validator → rule-based conflict
checker → follow-up drafts. Every stage has a `PipelineConfig` switch. The evaluation harness writes
per-item traces, bootstrap CIs over cases, slice tables, paired McNemar tests and error attribution.
The generator now produces noisy PDFs with gold labels known by construction.

Not built yet: the other three checklists, dense and hybrid retrieval runs (E3), the LLM citation judge,
and the review UI.

**Data used here.** 40 synthetic dev cases, cardiology checklist only (8 items, 320 item decisions): 243
present, 33 missing, 30 not applicable, 8 contradictory, 6 stale. Packets are 4–8 documents and 4–19 pages
(median 7). Plus 5 hand-made cases (40 items). The test split was not touched.

## 2. Results

| Run (dev, n = 40 unless noted) | Gap recall | Gap precision | Macro-F1 | Citation precision | $/case | s/case |
| --- | --- | --- | --- | --- | --- | --- |
| B0 keywords | 0.780 [0.652, 0.886] | 0.492 [0.346, 0.603] | 0.323 [0.294, 0.343] | 0.000 | 0.0000 | 0.0 |
| B1 whole packet, one call | 0.902 [0.793, 0.979] | 1.000 [1.000, 1.000] | 0.780 [0.713, 0.818] | 0.444 [0.412, 0.477] | 0.0010 | 10.4 |
| **Pipeline** (BM25, case scope, k = 5) | 0.854 [0.750, 0.947] | 1.000 [1.000, 1.000] | 0.770 [0.689, 0.978] | 0.635 [0.605, 0.663] | 0.0016 | 18.3 |
| Pipeline, no absence check | 0.902 [0.816, 0.977] | 0.597 [0.467, 0.694] | 0.752 [0.672, 0.812] | 0.631 [0.599, 0.661] | 0.0012 | 14.8 |
| Pipeline, no long-context read | 0.927 [0.852, 1.000] | 0.594 [0.458, 0.693] | 0.754 [0.682, 0.807] | 0.617 [0.586, 0.646] | 0.0013 | 14.3 |
| Pipeline, global scope (E2) | 0.000 [0.000, 0.000] | 0.000 (no CI) | 0.212 [0.175, 0.278] | 0.159 [0.133, 0.186] | 0.0022 | 24.4 |
| B1, hand-made (n = 5) | 1.000 | 1.000 | 1.000 | 0.565 [0.507, 0.686] | 0.0011 | 11.1 |
| Pipeline, hand-made (n = 5) | 0.500 [0.000, 1.000] | 1.000 | 0.797 [0.657, 1.000] | 0.500 [0.444, 0.619] | 0.0015 | 18.2 |

Brackets are 95% bootstrap CIs over cases. Gap = Missing or Contradictory. False-Missing rate is 0.000 in
every model run and 0.012 for B0.

**Baseline.** B1 is already strong: status accuracy 0.959 (13 of 320 items wrong) and no false gaps. It
beats B0 clearly (status McNemar p = 3.62e-07).

**Current system vs baseline.** The pipeline gets more items right (status accuracy 0.978 vs 0.959; 10
items only the pipeline gets right, 4 only B1) but the difference is not significant (p = 0.18), and on gap
detection it is not better (gap recall 0.854 vs 0.902; p = 0.625). It costs 1.6× as much and takes 7.9 s
longer per case. **On this dev set we have not shown that the pipeline beats B1.**

## 3. Experiments and what they showed

**E2, case vs global retrieval: case scoping is required.** With one shared index, 96.9% of citations
point at another patient's record (0.000 with case scope). Every missing item is "found" in someone else's
packet, so gap recall falls from 0.854 to 0.000 and status accuracy from 0.978 to 0.753 (p = 4.24e-22).

**Absence check: it helps, but not for the reason we built it.** Turning it off drops gap precision from
1.000 to 0.597 and status accuracy from 0.978 to 0.909 (p = 5.95e-05). The false-Missing rate on present
items is 0.000 with and without it, so it fixed no retrieval-caused false Missing; there were none to fix.
What it fixes is the conditional INR item: the per-item verifier sees only INR-related chunks, cannot tell
whether the patient is on warfarin, and proposes Missing. The whole-packet read turned 26 such items into
Not applicable. The two stages contribute very differently:

- *Wide re-check (k = 20 plus synonym scan):* 0 overturns in every run. It had new chunks to show for only
  6 of 58 first-pass Missing items, because BM25 returns only chunks that share a query term and a median
  packet has 7 chunks.
- *Long-context read:* 28 overturns in `pipe-dev`. 26 were correct (Missing → Not applicable) and 2 were
  wrong (truly Missing → Present). Those 2 errors account for most of the gap-recall difference between
  the pipeline and the no-long-context run (0.854 vs 0.927).

**Identity check.** Both wrong-patient documents in dev, and the one in the hand-made set, were
quarantined, with no document quarantined wrongly in the other 38 dev cases. An extra two-case run without
the check (`abl-no-identity-wp2`) had one error: a Missing lipid panel called Present from the other
patient's lab report. No run produced a false Contradictory on these cases, so the specific risk the check
was designed for was not observed. B1, which has no identity check, noticed the name mismatch on its own in
that case. This is two cases; it is an anecdote, not a result.

**Stale vs Needs review rule and recency windows.** With the validator deciding Stale in code from the
cited document's date and the checklist's `recency_days`, 5 of 6 stale items were labelled correctly and
the error attribution shows 0 recency-window errors. The sixth went to Needs review because the model's
quote failed the citation match. B1 labelled all 6 correctly by itself. Windows are per clinic by design,
but with one checklist that is untested.

## 4. Failure analysis

Pipeline: 7 wrong items. Error attribution: 4 reasoning errors, 2 retrieval misses, 1 validator reject.

| Category | Pipeline | B1 | Example |
| --- | --- | --- | --- |
| Contradiction missed (→ Present) | 3 | 3 | `dev-009` bmp: referral letter says "Most recent potassium is 5.9 mmol/L", lab report says "Potassium: 4.2 mmol/L". Only the lab report was retrieved; predicted Present. |
| Missing called Present | 3 | 1 | `dev-030` reason: long-context read cited "Please accept her for care at your cardiology clinic." `dev-010` meds: cited "Patient is on warfarin." as a medication list. |
| Stale → Needs review | 1 | 0 | `dev-000` progress_note: flag `citation_failed`; rationale shows the model had the right note ("dated 05/03/2026"). |
| Item dropped from output | 0 | 5 | `dev-011` reason: B1 filled the top-level `referral_reason` field and returned no `reason` item. |
| Model chose Needs review itself | 0 | 4 | `dev-025` inr (gold Not applicable): "does not establish whether the patient is taking warfarin". `dev-005` echo: found "TTE: LVEF 45%…" but asked for a standalone report. |

Likely causes:

1. **Contradictions are the main shared weakness** (5 of 8 found by each system). The conflicting statement
   uses different words from the item ("potassium" vs "basic metabolic panel"), so BM25 returns one
   document and the verifier has nothing to compare (`dev-009`, `dev-012`, and `hand-003` on the hand-made
   set). In `dev-023` both documents were shown ("3.2 mmol/L" vs "5.1 mmol/L") and the model still said
   Present. The rule-based conflict checker fired 0 times. Recall@k reads 1.000 because it counts one gold
   passage as a hit, so it hides this.
2. **The long-context read over-accepts weak evidence** for `reason` and `meds`, turning true gaps into
   Present. This is the "known behavior" in reverse: a second look that can only add evidence.
3. **Citation precision (0.635) is mostly a measurement problem.** Of 111 pipeline citations that did not
   match gold evidence, 105 were short header or date lines from the right document ("BASIC METABOLIC
   PANEL", "Date: 09/12/2026"); 1 cited a different document. The metric needs gold evidence that includes
   header and date lines, or the planned LLM judge, before we can say whether 0.95 is met.
4. **Macro-F1 is pulled down by a class with no gold examples.** Needs review has 0 gold items, so one
   predicted Needs review scores that class 0.0 and it is averaged in. Per-class F1 for the pipeline is
   present 0.988, not applicable 1.000, missing 0.952, stale 0.909, contradictory 0.769.

**Are the cases too easy?** Largely, yes. B1 is at 0.959 accuracy, first-pass retrieval recall@3 is 1.000,
and "buried" evidence sits on page 11 of packets of at most 19 pages. The dataset cannot yet show where
retrieval helps over long context, which is the central question of the project.

## 5. Gaps against targets (pipeline, dev)

| Target | Result | Status |
| --- | --- | --- |
| Gap recall ≥ 0.90 | 0.854 [0.750, 0.947] | Missed; CI includes the target |
| Gap precision ≥ 0.80 | 1.000 | Met |
| Macro-F1 ≥ 0.80 | 0.770 [0.689, 0.978] | Missed; CI includes the target |
| Citation precision ≥ 0.95 | 0.635 [0.605, 0.663] | Missed as measured; metric under review |
| Cost ≤ $0.10 per case | $0.0016 | Met |
| Latency ≤ 60 s per case | 18.3 s mean, 23.7 s max | Met |

## 6. Limitations

40 synthetic cases from one generator, one checklist, one model, one unseeded run per configuration.
There are only 8 contradictory and 6 stale items and 2 wrong-patient cases, so slice results are a handful
of items each, and repeated pipeline configurations differed by 1–2 items on the contradiction cases. CIs
are wide: the gap-recall interval spans about 0.20. Dense and hybrid retrieval (E3), packet length and
evidence position (E4), and the validator and conflict-checker ablations (E5) were not run. Nothing here
was measured on real referrals.

## 7. Plan, Nov 19 – Dec 3

1. **Fix contradiction handling (highest impact, Nov 19–22).** Retrieve by the item's component terms
   (e.g. potassium, creatinine for the BMP; drug names for the medication list) so both documents reach the
   verifier, and ask it to compare values across documents. Measure on the 8 dev contradictions and the
   contradiction slice; keep only if it helps.
2. **Make the long-context read stricter and narrower (Nov 19–22).** Use it to resolve conditions
   (Not applicable) and require item-specific evidence before it overturns a Missing. Target: keep the 26
   INR corrections, lose the 2 false Presents.
3. **Evaluation fixes, same window:** header and date lines in gold evidence or the LLM citation judge;
   report macro-F1 over classes with gold support alongside the current number; seed or repeat runs.
4. **Harder data (Neel):** longer packets toward the planned 60 pages, more contradictions and stale items,
   more wrong-patient cases, so E3 and E4 can say something.
5. **Freeze Nov 23–28:** one run of B0, B1 and the pipeline on the 110 test cases. Report Nov 28 – Dec 3.
