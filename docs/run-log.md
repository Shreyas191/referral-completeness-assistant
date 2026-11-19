# Run log

One row per run worth comparing. Run folders (`runs/<name>/`) are not committed, so this table is the
shared record. Dev split only until the freeze.

**Gap R / P** = recall and precision on Missing + Contradictory. **FM** = false-Missing rate.
**Cite P** = citation precision. Cost and latency are per case. Each run's `report.md` ends with this row
pre-filled; `python -m referral.eval.compare runs/A runs/B` gives the side-by-side with CIs and McNemar p.

| Date | Run name | Who | System / change | Split (n) | Gap R | Gap P | Macro-F1 | FM | Cite P | Cost | Latency | Commit | Takeaway |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 2026-11-19 | b0-dev | Shreyas | B0 keyword lower bound | dev (40) | 0.78 | 0.49 | 0.32 | 0.01 | 0.00 | $0.000 | 0.0 s | fa8fb45 | Lower bound. Keywords over-flag gaps (precision 0.49) and cite nothing. |
| 2026-11-19 | b1-dev | Shreyas | B1 whole packet in one call (gpt-6-luna) | dev (40) | 0.90 | 1.00 | 0.78 | 0.00 | 0.44 | $0.001 | 10.4 s | fa8fb45 | Strong baseline: 13 of 320 items wrong. 5 are a dropped `reason` item, 4 are Needs review it chose itself, 3 are missed contradictions. Packets (4–19 pages) fit easily in context. |
| 2026-11-19 | pipe-dev | Shreyas | Full pipeline, BM25, case scope, k=5 | dev (40) | 0.85 | 1.00 | 0.77 | 0.00 | 0.63 | $0.002 | 18.3 s | fa8fb45 | 7 of 320 wrong, but not better than B1 on gaps (McNemar p = 0.18 status, 0.625 gap). 3 missed contradictions, 3 Missing called Present (2 introduced by the long-context read), 1 Stale sent to Needs review by a failed citation. recall@3 = 1.00. |
| 2026-11-19 | e2-global | Shreyas | E2: `--scope global` | dev (40) | 0.00 | 0.00 | 0.21 | 0.00 | 0.16 | $0.002 | 24.4 s | fa8fb45 | 96.9% of citations point at another patient's record; every gap is "found" elsewhere, so gap recall is 0. Case scoping is required. |
| 2026-11-19 | abl-no-absence | Shreyas | `--no-absence-check` (both absence stages off) | dev (40) | 0.90 | 0.60 | 0.75 | 0.00 | 0.63 | $0.001 | 14.8 s | fa8fb45 | Gap precision falls to 0.60: 25 not-applicable INR items stay Missing. False-Missing on present items is 0 with or without the check. |
| 2026-11-19 | abl-no-long-context | Shreyas | `--no-long-context` (wide re-check kept, final read off) | dev (40) | 0.93 | 0.59 | 0.75 | 0.00 | 0.62 | $0.001 | 14.3 s | fa8fb45 | Same as no-absence within noise, so the wide re-check alone changes nothing (0 overturns; it found extra chunks for only 6 items). All of the gain comes from the final read. |
| 2026-11-19 | b0-hand5 | Shreyas | B0 on Neel's hand-made cases | adversarial (5) | 0.50 | 0.17 | 0.25 | 0.00 | 0.50 | $0.000 | 0.0 s | fa8fb45 | Lower bound on the hand-made cases: 33 of 40 items right, gap precision 0.17. Run after the docs were edited, so its `metrics.json` marks the commit dirty; no code differed. |
| 2026-11-19 | b1-hand5 | Shreyas | B1 on Neel's hand-made cases | adversarial (5) | 1.00 | 1.00 | 1.00 | 0.00 | 0.56 | $0.001 | 11.1 s | fa8fb45 | All 40 items right. Only 2 gold gaps in these 5 cases, so this is a smoke test, not a result. |
| 2026-11-19 | pipe-hand5 | Shreyas | Full pipeline on Neel's hand-made cases | adversarial (5) | 0.50 | 1.00 | 0.80 | 0.00 | 0.50 | $0.002 | 18.2 s | fa8fb45 | 1 of 40 wrong: hand-003 meds contradiction called Present because only the medication list was retrieved. Same failure as on dev. |
| 2026-11-19 | abl-no-identity-wp2 | Shreyas | `--no-identity-check`, only the 2 wrong-patient dev cases (dev-014, dev-028) | dev (2) | 0.67 | 1.00 | 0.92 | 0.00 | 0.65 | $0.001 | 16.4 s | fa8fb45 | Extra run, not in the plan. Without the check, dev-014 lipids (gold Missing) is called Present from the other patient's lab report. No false Contradictory either way. n = 2. |

## Hypotheses tried

Record every change, including ones that did not help.

| Date | Observed failure | Hypothesis | Change | Result | Kept? |
| --- | --- | --- | --- | --- | --- |
| 2026-11-19 | A retrieval miss can look like a missing item (design concern from Milestone 2; not yet observed) | A wider search plus a whole-packet read before accepting Missing removes false Missing | Absence check on (`pipe-dev`) vs off (`abl-no-absence`) | False-Missing rate 0.000 in both. Gap precision 1.000 vs 0.597, gap recall 0.854 vs 0.902, status accuracy 0.978 vs 0.909 (McNemar p = 5.95e-05). The gain is 26 not-applicable INR items that the per-item verifier calls Missing; the check also turned 2 truly Missing items into Present. | Yes, but for a different reason than planned |
| 2026-11-19 | Same | The wide re-check (k = 20 + synonym scan) is enough without the whole-packet read | `abl-no-long-context` vs `pipe-dev` | Wide re-check overturned 0 items in every run; it had new chunks to show for only 6 of 58 first-pass Missing items. Gap precision 0.594 without the final read. | Final read kept; wide re-check unproven on these short packets |
| 2026-11-19 | Design concern: a shared index could cite another patient | Filtering retrieval to one `case_id` is necessary, not just tidy | `--scope global` (`e2-global`) vs `pipe-dev` | Foreign-citation rate 0.969 vs 0.000; gap recall 0.000 vs 0.854; status accuracy 0.753 vs 0.978 (McNemar p = 4.24e-22); recall@5 0.588 vs 1.000. | Case scope kept |
| 2026-11-19 | Design concern: a wrongly merged document could create false labels | Quarantining documents whose identifiers do not match prevents them | `--no-identity-check` on the 2 wrong-patient dev cases | Check on: both foreign documents quarantined, 16 of 16 items right. Check off: 15 of 16, one Missing item called Present from the foreign lab report. No false Contradictory in either run. | Kept; evidence is 2 cases |

Notes for the rows above: dates are UTC, as written by the harness. All model runs use `openai:gpt-6-luna`
at $0.10 / $0.50 per million input / output tokens. Total spend for the ten runs was $0.31. The runs were
made from PR #25's branch (commit `fa8fb45`) before it merged. Each configuration was run once; the model
is not seeded, and repeated pipeline configurations differed by 1–2 items on the contradiction cases.
