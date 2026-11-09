# How we work

## Order of work

```
#1 API key ───────────┐
#7 5 hand-made cases ─┼─> #2 pipeline run + #12 B0/B1 run ─> numbers we trust
                      │
#8 render → #9 noise/PDF/gold → #10 40 dev cases ─> #15 E1 on dev ─> #17 failure analysis ─> #18 Milestone 3
                                                   └> #5 E3, #6 E2 + absence ablation ──┘
#3 save traces → #14 error attribution ─────────────────────────────────┘
#13 harness CIs + report ────────────────────────────────────────────────┘
```

Everything starts from the 5 hand-made cases (#7) and the API key (#1). While the generator is being
built (#8–10), the pipeline and evaluation gaps (#3, #4, #13) are developed against those 5 cases.

| Dates | Phase | Output |
| --- | --- | --- |
| Nov 9–11 | First slice | All three systems run on 5 cases |
| Nov 11–15 | Scale | 40 dev cases, E1, retrieval traces |
| Nov 15–19 | Diagnose | Failure taxonomy, E2/E3, Milestone 3 (due Nov 19) |
| Nov 19–23 | Improve | 1–2 targeted fixes chosen from the failure analysis |
| Nov 23–28 | Freeze | One final run on the 110 test cases |
| Nov 28–Dec 3 | Report | Tables, failure examples, demo video, report (due Dec 3) |

## Experiment loop

For every change after the first dev run:

1. Run the current system on **dev** and keep the run folder.
2. Read the wrong items and name the failure, e.g. "BM25 misses abbreviations like TTE".
3. Write the hypothesis and make **one** change, behind a `PipelineConfig` flag.
4. Re-run on dev and compare with the previous run, including confidence intervals.
5. Keep the change only if it helps. Log the result in [`docs/run-log.md`](docs/run-log.md) either way.

**Never run on the test split before the freeze.** It gets one run, Nov 23–28.

## Branches and pull requests

- One branch per issue: `<name>/<issue>-<short-topic>`, e.g. `neel/8-render-documents`.
- Open a PR with `Closes #N`, fill in the template, and get one teammate's review before merging.
- CI (`pytest`, `ruff`) must pass. Run both locally first.
- Keep PRs small; split large work into several.
- Do not add AI co-author or attribution lines to commits or PRs.

## Shared contracts

`src/referral/schemas.py` and the `gold.json` format (`src/referral/eval/gold.py`) connect the pipeline,
the data and the evaluation. Tell the other two before changing either, and update the tests in the same PR.

## Run log

Every run worth comparing gets one row in [`docs/run-log.md`](docs/run-log.md): what changed, the key
metrics, and the takeaway. This log becomes most of the Milestone 3 and final reports, so write the
takeaway while it is fresh.

## Sync

15 minutes every two or three days: what is done, what is blocked, latest numbers.
