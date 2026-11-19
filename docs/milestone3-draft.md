# Milestone 3: Results, Failures and Iteration (draft)

**Specialist Referral Completeness Assistant.** Shreyas Kaldate, Neel Gajiwala, Adhyayan Verma. Nov 19, 2026.

Metrics come from `runs/*/metrics.json` and `referral.eval.compare`; item counts in the failure analysis
were counted from the same runs' predictions and traces. Commit `fa8fb45`, model `openai:gpt-6-luna`, one
run per configuration. Full run list: [run-log.md](run-log.md).

## 1. System so far

The pipeline runs end to end on generated PDFs: ingest and chunk → identity check → BM25 retrieval filtered
to one case → per-item LLM verifier → absence check (wide re-check, then a whole-packet read) → citation
and recency validator → conflict checker → follow-up drafts. Every stage has a `PipelineConfig` switch.
The harness writes traces, bootstrap CIs, slices, McNemar tests and error attribution. Not built yet: three
more checklists, dense and hybrid retrieval, the LLM citation judge, the review UI.

**Data.** 40 synthetic dev cases, cardiology checklist only: 320 item decisions (243 present, 33 missing,
30 not applicable, 8 contradictory, 6 stale). Packets are 4–19 pages (median 7). Plus 5 hand-made cases.
The test split was not touched.

## 2. Results

| Run (dev, n = 40) | Gap recall | Gap precision | Macro-F1 | Citation precision | $/case | s/case |
| --- | --- | --- | --- | --- | --- | --- |
| B0 keywords | 0.780 [0.652, 0.886] | 0.492 [0.346, 0.603] | 0.323 [0.294, 0.343] | 0.000 | 0.0000 | 0.0 |
| B1 whole packet, one call | 0.902 [0.793, 0.979] | 1.000 [1.000, 1.000] | 0.780 [0.713, 0.818] | 0.444 [0.412, 0.477] | 0.0010 | 10.4 |
| **Pipeline** (BM25, case scope, k = 5) | 0.854 [0.750, 0.947] | 1.000 [1.000, 1.000] | 0.770 [0.689, 0.978] | 0.635 [0.605, 0.663] | 0.0016 | 18.3 |
| Pipeline, no absence check | 0.902 [0.816, 0.977] | 0.597 [0.467, 0.694] | 0.752 [0.672, 0.812] | 0.631 [0.599, 0.661] | 0.0012 | 14.8 |
| Pipeline, no long-context read | 0.927 [0.852, 1.000] | 0.594 [0.458, 0.693] | 0.754 [0.682, 0.807] | 0.617 [0.586, 0.646] | 0.0013 | 14.3 |
| Pipeline, global scope (E2) | 0.000 [0.000, 0.000] | 0.000 (no CI) | 0.212 [0.175, 0.278] | 0.159 [0.133, 0.186] | 0.0022 | 24.4 |

Brackets are 95% bootstrap CIs over cases. Gap = Missing or Contradictory. False-Missing rate is 0.000 in
every model run. On the 5 hand-made cases B1 got 40 of 40 items right and the pipeline 39 of 40.

**Baseline vs current system.** B1 is already strong: status accuracy 0.959 (13 of 320 wrong), no false
gaps, and far above B0 (McNemar p = 3.62e-07). The pipeline gets more items right (0.978; 7 wrong), but the
difference is not significant (p = 0.18), its gap recall is lower (p = 0.625), and it costs 1.6× as much.
**We have not shown that the pipeline beats B1 on this dev set.**

## 3. Experiments and changes since Milestone 2

**Case-scoped retrieval (E2): required.** With one shared index, 96.9% of citations point at another
patient's record. Every missing item is "found" in someone else's packet, so gap recall falls from 0.854
to 0.000 and status accuracy from 0.978 to 0.753 (p = 4.24e-22).

**Absence check: it helps, but not for the reason we built it.** Without it, gap precision drops from
1.000 to 0.597 and status accuracy from 0.978 to 0.909 (p = 5.95e-05). It fixed no false Missing, because
there were none: the rate is 0.000 with and without it. What it fixes is the conditional INR item. The
per-item verifier sees only INR chunks, cannot tell whether the patient is on warfarin, and proposes
Missing. The two stages differ sharply:

- *Wide re-check:* 0 overturns in every run. It had new chunks to show for only 6 of 58 first-pass Missing
  items, because a median packet has 7 chunks.
- *Long-context read:* 28 overturns. 26 were correct (Missing → Not applicable); 2 turned truly Missing
  items into Present, which is most of the gap-recall difference from the no-long-context run.

**Identity check.** All three wrong-patient documents (2 dev, 1 hand-made) were quarantined, and nothing
was quarantined wrongly in the other 38 dev cases. A two-case run without the check had one error: a
Missing lipid panel called Present from the other patient's lab report. No false Contradictory appeared in
any run, so the risk the check was designed for was not observed. Two cases is an anecdote.

**Stale vs Needs review, and recency windows.** With Stale decided in code from the cited document's date
and the checklist's `recency_days`, 5 of 6 stale items were correct and error attribution shows 0
recency-window errors. The sixth went to Needs review after a citation failed to match. Per-clinic windows
are untested with one checklist.

## 4. Failure analysis

Error attribution for the pipeline's 7 wrong items: 4 reasoning errors, 2 retrieval misses, 1 validator
reject.

| Category | Pipeline | B1 | Example |
| --- | --- | --- | --- |
| Contradiction missed (→ Present) | 3 | 3 | `dev-009` bmp: letter says "Most recent potassium is 5.9 mmol/L", lab report says "Potassium: 4.2 mmol/L". Only the lab report was retrieved. |
| Missing called Present | 3 | 1 | `dev-030` reason: long-context read cited "Please accept her for care at your cardiology clinic." |
| Stale → Needs review | 1 | 0 | `dev-000` progress_note: flag `citation_failed`, though the rationale names the right note. |
| Item dropped from output | 0 | 5 | `dev-011` reason: B1 filled the top-level reason field and returned no `reason` item. |
| Model chose Needs review itself | 0 | 4 | `dev-005` echo: found "TTE: LVEF 45%…" but asked for a standalone report. |

Likely causes:

1. **Contradictions are the shared weakness** (each system found 5 of 8). The conflicting statement uses
   different words from the item ("potassium" vs "basic metabolic panel"), so BM25 returns one document.
   In `dev-023` both were shown ("3.2 mmol/L" vs "5.1 mmol/L") and the model still said Present. The
   conflict checker fired 0 times. Recall@k reads 1.000 because one gold passage counts as a hit.
2. **The long-context read over-accepts weak evidence** for `reason` and `meds`.
3. **Citation precision is mostly a measurement problem.** Of 111 pipeline citations that did not match
   gold evidence, 105 were header or date lines from the right document ("BASIC METABOLIC PANEL").
4. **Macro-F1 averages in a class with no gold examples.** One predicted Needs review scores that class
   0.0. Per-class F1: present 0.988, not applicable 1.000, missing 0.952, stale 0.909, contradictory 0.769.

**The cases are too easy.** B1 is at 0.959, recall@3 is 1.000, and "buried" evidence sits on page 11 of at
most 19. The data cannot yet show where retrieval helps over long context.

## 5. Gaps against targets (pipeline, dev)

| Target | Result | Status |
| --- | --- | --- |
| Gap recall ≥ 0.90 | 0.854 [0.750, 0.947] | Missed; CI includes the target |
| Gap precision ≥ 0.80 | 1.000 | Met |
| Macro-F1 ≥ 0.80 | 0.770 [0.689, 0.978] | Missed; CI includes the target |
| Citation precision ≥ 0.95 | 0.635 [0.605, 0.663] | Missed as measured; metric under review |
| Cost ≤ $0.10, latency ≤ 60 s per case | $0.0016, 18.3 s mean (23.7 s max) | Met |

**Limitations.** 40 synthetic cases, one checklist, one model, one unseeded run per configuration; repeat
configurations differed by 1–2 items. Only 8 contradictory items, 6 stale items and 2 wrong-patient cases,
and the gap-recall CI spans about 0.20. E3, E4 and E5 were not run. Nothing was measured on real referrals.

## 6. Plan, Nov 19 – Dec 3

1. **Contradictions (Nov 19–22).** Retrieve by the item's component terms (potassium, drug names) so both
   documents reach the verifier, and have it compare values. Keep only if the 8 dev contradictions improve.
2. **Stricter long-context read (Nov 19–22).** Use it to resolve conditions; require item-specific evidence
   to overturn a Missing. Goal: keep the 26 INR corrections, lose the 2 false Presents.
3. **Alongside:** fix the citation metric (gold header and date lines, or the LLM judge), report macro-F1
   over classes with gold support, repeat runs, and generate longer, harder packets for E3 and E4.
4. **Freeze Nov 23–28:** one run of B0, B1 and the pipeline on the 110 test cases. Report Nov 28 – Dec 3.
