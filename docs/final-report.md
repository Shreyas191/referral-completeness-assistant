# Specialist Referral Completeness Assistant: Final Report

**Shreyas Kaldate, Neel Gajiwala, Adhyayan Verma.** NYU Generative AI Engineering, Fall 2026. Draft, Nov 26.

Every number below comes from a saved run (`runs/<name>/metrics.json`) or from `referral.eval.compare` on two
runs, and the run log ([run-log.md](run-log.md)) records each one with its commit. Unless noted, the model is
`openai:gpt-6-luna`. Brackets are 95% bootstrap CIs over cases; p-values are exact paired McNemar tests on items.

## 1. Problem

A specialty clinic receives a referral packet (2–8 PDFs for one patient: referral letter, progress notes, labs,
imaging, medication list) and must check it against its own intake checklist before booking. Intake staff do this by
hand, and incomplete packets cause delays and repeat requests. Our system gives every checklist item one status, with
cited evidence, and drafts follow-up requests for the gaps. Staff review everything; the system never contacts anyone.

| Status | Rule |
| -------------- | ---------------------------------------------------------------------- |
| Present | A verified quote supports the item, within its recency window |
| Missing | Nothing in the packet supports it, after the absence check |
| Contradictory | Verified quotes from at least two documents conflict |
| Stale | Verified evidence with a known date, older than the item's window |
| Needs review | Evidence could not be verified: the citation failed, or the date is unreadable |
| Not applicable | A conditional item whose condition does not hold |

The cost of an error is not symmetric. A false Missing sends staff chasing a document they already have; a false
Present lets an incomplete packet through. Gaps (Missing or Contradictory) are the positive class.

## 2. System

```
PDFs -> chunk -> identity check -> BM25 retrieval (one case only) -> value-term expansion -> LLM verifier (per item)
     -> absence check (wide re-check, condition re-check, whole-packet read) -> citation + recency validator
     -> conflict checker -> follow-up drafts
```

The LLM only proposes Present, Missing, Contradictory or Not applicable. Stale and Needs review are decided in code,
in a fixed order: verify each quote against its chunk, then the date, then the item's window, which the clinic sets in
its checklist JSON. Every stage has a `PipelineConfig` switch, so every ablation below is one flag.

Each component is there because a run showed the failure it fixes:

- **Case-scoped retrieval.** Evidence is searched only within the case. A global index (experiment E2) cites another
  patient 96.9% of the time.
- **Identity check.** Documents whose name, date of birth or MRN do not match the letter are quarantined, so a
  misfiled report cannot create a false Contradictory.
- **Absence check.** A retrieval miss looks exactly like a missing item, so Missing is accepted only after a wider
  search, a synonym scan and a whole-packet read. The read may overturn Missing only with a quote that names the item.
- **Value-term expansion.** When the top chunk states a value ("Potassium: 4.2"), chunks from other documents that
  state a value for the same term are shown too, so both sides of a contradiction reach the verifier.
- **Condition re-check.** A Not applicable call is re-verified with any unseen chunk naming the item's condition
  ("warfarin"), since retrieval searches for the item, not its condition.
- **Validator.** Quotes are fuzzy-matched to the cited chunk. Quotes that join non-adjacent lines are checked line by
  line, and a quote longer than its chunk must match the whole chunk, not merely contain it.
- **Checklist definitions.** The clinic can say what counts as evidence; for the medication list, a single drug
  mentioned in passing is not a list but can contradict one. The definition is shown to the baseline too.

**Baselines.** B0 matches keywords. B1 gives the whole packet and checklist to the same model in one call and asks
for a status for every item, with quotes; it decides all six statuses itself, and its citations are recorded but not
checked. B1 is the primary baseline.

## 3. Data

All cases are synthetic; no real patient data was used. A generator (Neel) writes each packet: code fixes every fact a
label depends on (values, dates, which document and page), and the LLM writes only the surrounding prose, which may
not contain any checklist term. So labels are known by construction. A defect plan omits, contradicts, ages, buries
or abbreviates items, or adds a wrong-patient document. The PDFs carry OCR-style letter noise, and a checker verifies
every case (evidence verbatim, recency, identity, and leaks of absent items).

| Set | Cases | Items | Pages | Use |
| ---------------------- | ----- | ----- | ------ | -------------------------------------------------- |
| Dev | 40 | 320 | 4–19 | Development and ablations |
| Hard (`--profile hard`) | 18 | 144 | 43–46 | E3 and E4: two 20-page hospital documents per packet, evidence on random pages |
| Hand-written | 5 | 40 | short | Adversarial checks |
| Test (held out) | 110 | 880 | short, as dev | Generated Nov 26, run once at the freeze |

The test split matches the target mix exactly: 33 complete cases, 44 with missing items, 28 abbreviation-only, 22
contradictions, 22 buried, 16 stale and 6 wrong-patient. It was not opened or run before the freeze.

## 4. Evaluation

- **Outcome:** gap recall and precision, status macro-F1, accuracy, false-Missing rate.
- **Macro-F1 over the statuses gold uses.** The plain version averages in an F1 of 0 for any status gold never uses
  (Needs review), so two dev runs with equal accuracy (0.991) scored 0.967 and 0.827. We report the gold-class version.
- **Citations at three levels:** right document, right page, and right passage. A passage counts only if the quote
  matches the planted evidence.
- **Trajectory:** retrieval recall@k, and error attribution from per-item traces: retrieval miss, validator reject,
  recency window, or reasoning error, with the stage that made the final call.
- **Slices, cost, latency,** and an offline retrieval sweep (`retrieval_sweep`) and position table (`position`)
  that need no LLM calls.
- **Repeats.** The model is unseeded: repeating the Nov 19 pipeline moved dev gap recall from 0.854 to 0.902 with no
  code change. So main configurations were run twice and compared item by item.

## 5. Development: what changed and why

The Milestone 3 pipeline (Nov 19) did not beat B1: dev accuracy 0.978 vs 0.959 and 0.963, McNemar p = 0.18. Each
change below targets an observed error and was measured on dev; it was kept if it helped, or, for the quote length
check, if it closed a hole without changing any result.

| Change | Failure it targets | Evidence (dev unless noted) | Kept |
| ---------------- | ---------------------------------- | ---------------------------------- | -------- |
| Value-term expansion | Contradictions missed because BM25 scored the second document 0 (in a 4-chunk case a term in half the chunks has IDF 0) or it shared no word with the item | Contradictions caught 6, 8, 8, 6 of 8 in four runs vs 5, 6, 6 without | Yes |
| Strict long-context read | The whole-packet read overturning Missing with a generic quote | Replayed on six earlier runs it rejects all 4 wrong overturns and nothing else; fired once live (`dev-030`) and was right | Yes |
| Save rejected quotes, then check joined quotes piece by piece | Present items sent to Needs review, the largest error type: every rejected quote joined two lines of the right chunk and scored 82.1–89.5 against 90 | Citation-caused Needs review 0 and 0 vs 2, 1, 2; wrong items 1 and 1 vs 3, 4, 4 | Yes |
| Quote length check | Found by reading the code: a quote longer than its chunk scored 100 if it contained it (an invented echo finding passed against the 3-character chunk "f 1") | Replay of 5,372 saved citations in 18 runs: none newly rejected, so no saved result changes | Yes |
| Condition re-check | INR called Not applicable when no "warfarin" chunk was retrieved (`dev-024`, 600-character chunks) | Fires on exactly that item in 8 saved runs; replayed 3 times, correct 3 of 3 | Yes |
| Medication-list definition | "Patient is on warfarin." taken as a medication list (4 runs, dev and hard) | Both targeted items fixed; no item worse in four runs (p = 0.5–1) | Yes |
| Re-check Stale when a newer note was shown | `dev-018`: the model cited the older of two notes | Measured offline: would re-decide 52–82 correct Stale items to fix 1 error seen in 1 of 16 runs | No |
| Dense / hybrid retrieval (E3) | — | See Section 6 | No (BM25 stays) |
| Section/sentence-boundary chunking | Fixed cuts split 16% / 21% of gold passages at 600 / 300 characters | Makes E3 chunk size fair; no pipeline gain at 1200 (2 wrong vs 1) | Optional, off |

After these changes the pipeline gets 1 of 320 dev items wrong (`dev-038`, where "Referral placed." is taken as a
reason for referral) and beats B1 on dev in every pairing: 9–12 items vs 0, p = 0.00049 to 0.0039.

## 6. Experiments

**E1: baselines vs pipeline (dev).**

| Run | Gap recall | Gap prec. | Macro-F1 | Accuracy | Cite: doc | $/case | s/case |
| ---------------- | ------------------- | ------ | ------------------- | ------- | ------ | ------ | ------ |
| B0 | 0.780 | 0.492 | 0.323 | 0.850 | 0.992 | 0 | 0.0 |
| B1 (final checklist) | 0.951 [0.870, 1.000] | 1.000 | 0.979 [0.938, 0.997] | 0.969 | 0.896 | 0.0010 | 9.0 |
| Pipeline, Nov 19 | 0.854 | 1.000 | 0.924 | 0.978 | 0.987 | 0.0016 | 18.3 |
| Pipeline, final | 0.976 [0.921, 1.000] | 1.000 | 0.997 [0.988, 1.000] | 0.997 | 0.997 | 0.0017 | 16.2 |

**E2: case vs global retrieval.** A global index across all 40 cases sends 96.9% of citations to other patients, and
gap recall falls to 0.000. Case scoping is not optional.

**E5 and absence ablations (Nov 19 pipeline, dev).**

| Removed | Effect |
| ---------------------- | ---------------------------------------------------------------------- |
| Absence check | Gap precision 1.000 to 0.597, accuracy 0.978 to 0.909: INR called Missing instead of Not applicable |
| Long-context read only | Gap precision 0.594; the wide re-check never fired (retrieval does not miss on 4–19 page packets) |
| Validator | 0 of 6 stale items found; right-document citations 0.987 to 0.894 |
| Conflict checker | It never fired in any run; the ablation run differed on 7 items (p = 0.375), within run-to-run noise |
| Identity check (2 wrong-patient cases) | One Missing item called Present from another patient's lab |

**E3: retrieval.** On dev, dense and hybrid retrieval did not help: wrong items BM25 1 and 1, hybrid 2 and 1, dense 3
and 3 (p = 0.5–1), and both cost about 30% more because they always fill all five slots. Retrieval alone tells the
same story: all gold evidence in the top five for 0.992 (BM25), 0.946 (dense) and 0.992 (hybrid) of items. Hybrid
finds both sides of multi-document evidence more often (0.966 vs 0.931). With boundary-aware cuts, 600-character
chunks retrieve almost as well as 1200 (BM25 0.977) and 300 is clearly worse (0.689); before the chunker fix, small
chunks looked worse only because gold sentences were cut in half. On the 46-page hard set, hybrid matched BM25 item by
item at 20% more cost.

**E4: evidence position and packet length (hard set).** On 43–46 page packets the pipeline gets 2 of 144 items wrong
and B1 11 (p = 0.0039). Loading files by name puts the two long documents first, so 106 of 112 evidence items sit in
the last quarter. Reversing the document order (same content) moves 102 of them to the first quarter. B1 then drops
from 11 to 6 wrong (6 items fixed, 1 broken, p = 0.125); the pipeline does not change (2 to 3, p = 1). With evidence
at the front, the pipeline's lead shrinks to 5 items vs 2 (p = 0.453). The direction matches a position effect on the
long-context baseline, but with 18 cases it is not significant.

## 7. Final results (frozen test split)

The system was frozen at commit `824f8f4` on Nov 26 and run once on the 110 held-out cases: B0 once (it is
deterministic), B1 and the pipeline twice each.

| Run | Gap recall | Gap prec. | Macro-F1 | Accuracy | Cite: doc / passage | $/case | Mean / max s |
| -------------- | ------------------- | ------ | ------------------- | ------- | ----------- | ------ | --------- |
| B0 | 0.786 [0.708, 0.861] | 0.533 | 0.332 | 0.867 | 0.990 / 0.001 | 0 | 0.0 |
| B1, run 1 | 0.920 [0.862, 0.968] | 1.000 | 0.975 [0.952, 0.992] | 0.973 | 0.905 / 0.520 | 0.0010 | 9.1 / 13.9 |
| B1, run 2 | 0.938 [0.890, 0.977] | 0.991 | 0.974 [0.954, 0.989] | 0.960 | 0.898 / 0.516 | 0.0010 | 9.0 / 14.8 |
| **Pipeline, run 1** | **0.955 [0.914, 0.990]** | 1.000 | **0.978 [0.952, 0.994]** | **0.994** | **0.995 / 0.704** | 0.0016 | 15.4 / 19.6 |
| **Pipeline, run 2** | **0.946 [0.900, 0.983]** | 1.000 | **0.972 [0.946, 0.992]** | **0.993** | **0.995 / 0.702** | 0.0016 | 14.7 / 22.4 |

- **The pipeline is more often right than B1:** in all four pairings, 22–33 items are right only in the pipeline and
  3–4 only in B1 (p = 2.27e-07 to 0.000534). On gaps alone the difference is not significant (6–8 vs 3–4, p = 0.39 to
  0.75): B1 is a strong baseline when it answers.
- **It is more consistent:** its two runs differ on 3 items (p = 1); B1's differ on 39 (p = 0.108).
- **Slices (pipeline, run 1 / run 2):** buried evidence 1.000 / 0.989 accuracy, stale 1.000 / 0.992, wrong-patient
  1.000 / 1.000 (no false Contradictory from the 6 misfiled documents), abbreviation 0.991 / 0.991, contradiction
  0.972 / 0.966. False-Missing rate 0.000.

| Target | Pipeline (run 1 / run 2) | Met |
| ---------------------- | ---------------------------------------- | -------------------- |
| Gap recall ≥ 0.90 | 0.955 / 0.946 (CI lower bound 0.914 / 0.900) | Yes |
| Gap precision ≥ 0.80 | 1.000 / 1.000 | Yes |
| Status macro-F1 ≥ 0.80 | 0.978 / 0.972 | Yes |
| Citation precision ≥ 0.95 | document and page 0.995; passage 0.70 | At document and page level only |
| Cost ≤ $0.10, latency ≤ 60 s per case | $0.0016; 22.4 s max | Yes |

## 8. Failure analysis

**The pipeline's test errors (5 and 6 items)** are all of two kinds.

1. **Missed contradictions (4 and 5).** Three are retrieval misses in both runs: the second document was never shown.
   Value-term expansion should have added it, and the traces show why it did not:
   - in `test-026` and `test-093` OCR noise spells the drug differently on the two sides ("metfrmin" and
     "metformin", "mtformin" and "metformin"), so the exact word match fails;
   - in `test-046` the lab report gives cholesterol values and the letter says "LDL is 78": different terms.

   The same mechanism appears on the hand-written cases. In `hand-003` the top-ranked chunk is "Enclosed with this
   referral: ... current medication list", which names the item but states no value, so expansion has nothing to
   start from. The current pipeline misses that contradiction, and B1 gets all five hand-written cases right. The
   remaining misses had both sides in front of the model, which still answered Present.
2. **`test-021` reason** (both runs): a Missing reason for referral called Present from a generic sentence, the same
   first-pass error as `dev-038`.

**B1's test errors (24 and 35)** are mostly items it left out of its answer: 18 and 26. One call that must return
eight structured items is less reliable than eight calls that return one each. All of B1's errors on items with
evidence sit in the last quarter of the packet. On the test set that is also where the referral letter sorts, so
there position and the dropped reason item cannot be separated; the reversed-order experiment (E4) is the cleaner
evidence.

**Passage-level citations (0.70)** miss the target, but not because the model cites the wrong place. Of the 239
quotes per run that miss the planted passage, 232 and 231 are the document's title or date line
("TRANSTHORACIC ECHOCARDIOGRAM REPORT", "Date: 11/15/2026") instead of the value line, inside the right document and
page (0.995). The model uses the title to show what the document is, which a reviewer can still follow, but it is not
the evidence.

## 9. Limitations

- **Synthetic data, one checklist, one model.** The generator forbids checklist terms in filler prose, so long packets
  add length but no near-miss text; on the hard set BM25 still finds 0.991 of evidence. Real packets would contain
  ordered-not-resulted tests, other patients' values and copied-forward notes.
- **The test split comes from the same generator as dev,** so the test results measure generalisation to new
  patients and defect combinations, not to new document styles.
- **The hard set and E4 have 18 cases and one run per condition,** and every fix since Nov 19 was designed on dev.
  The frozen test run is the only unbiased estimate.
- **Unseeded model:** single runs vary by several items (B1 test runs differ on 39 items), which is why every claim
  above rests on paired tests across repeated runs.

## 10. Cost

Model calls across all 45 saved runs cost $2.64, including $0.59 for the frozen test run. Generating the hard and test
sets cost $0.30 more. Per case, the pipeline costs $0.0016 and B1 $0.0010, both well under the $0.10 target.

## 11. What we would do next

1. **Fuzzy, multi-chunk value expansion:** match terms within one edit and start from every shown chunk, not just the
   top one. The traces show this would have reached all three test retrieval misses and `hand-003`; it needs a dev
   measurement before it can be claimed.
2. **Ask the verifier to quote the value line, not the header**, for passage-level citation precision.
3. **Harder distractors** in the generator (ordered-not-resulted tests, values for other patients in prose), and the
   three remaining checklists.
4. **The review UI** for intake staff, with every status, its citations and the follow-up drafts.

*Team contributions: to be completed by the team.*
