# Run log

One row per run worth comparing. Run folders (`runs/<name>/`) are not committed, so this table is the
shared record. Dev split only until the freeze.

**Gap R / P** = recall and precision on Missing + Contradictory. **FM** = false-Missing rate.
**Cite P** = citation precision. Cost and latency are per case. Each run's `report.md` ends with this row
pre-filled; `python -m referral.eval.compare runs/A runs/B` gives the side-by-side with CIs and McNemar p.

| Date | Run name | Who | System / change | Split (n) | Gap R | Gap P | Macro-F1 | FM | Cite P | Cost | Latency | Commit | Takeaway |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| | | | | | | | | | | | | | |

## Hypotheses tried

Record every change, including ones that did not help.

| Date | Observed failure | Hypothesis | Change | Result | Kept? |
| --- | --- | --- | --- | --- | --- |
| | | | | | |
