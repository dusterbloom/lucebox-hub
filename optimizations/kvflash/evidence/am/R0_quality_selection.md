# R0-QuALITY selection block (articles 0-19, 357 questions, target 0.10): PARTIAL

Pre-registered cells S1-S6 in `PREREG.md` Amendment 1 (b966c2a9). Written at 20:35 CEST with 2 of 4 invocations finished.
Raw: `/qwen3-4b/sel_*_t0.1/` on volume `kvflash-am-results`. `--compute-stats 0`, `best` config, on-policy variants, H100.

| Cell | Query source | Keys | Budget | Accuracy (n=357) | 95% CI (article bootstrap) | GPU-s/article | Status |
|---|---|---|---|---|---|---|---|
| original (articles 0-19, from the vs-random run) | - | - | - | 72.3% (258/357) | - | 0 | done |
| S2 | repeat | HighestAttnKeys | uniform | 62.2% | [57.1, 67.0] | 26.7 | done |
| S3 | context-prefill | HighestAttnKeys | per-head | 70.3% | [65.2, 75.1] | 13.3 | done |
| S1 | repeat | HighestAttnKeys | per-head | - | - | - | running (invocation `sel_repeat_perhead_t0.1`) |
| S4 | repeat | OMP-fast | per-head | - | - | - | running (same invocation) |
| S5 | ss-plus-repeat | HighestAttnKeys | per-head | - | - | - | running (`sel_ss_perhead_t0.1`) |
| S6 | ss-plus-repeat | OMP-fast | per-head | - | - | - | running (same invocation) |

What the two finished cells say (selection split, exploratory until the held-out is run): per-head budgets plus context-prefill queries are 8 points above uniform plus repeat queries at the same target (70.3% vs 62.2%) and 2 points below the full cache (72.3%). This matches the paper's direction (per-head budgets are what bring HighestAttnKeys close to the original cache) but the two cells differ in both budget and query source, so the 8 points cannot be attributed to either factor alone; S1 (repeat, per-head) separates them.

## Cost gate
The invocations `sel_repeat_perhead_t0.1` and `sel_ss_perhead_t0.1` have run 4.3 h each (OMP-fast with per-head budgets is slower than the pilot suggested, and self-study on top). Cost to date for this launch: AIME Table 9 $56.4 + pilot $28.6 + vs-random $33.7 + finished selection cells $5.9 = $124.6, plus ~$34 accrued by the two running invocations = ~$159 > the $150 cap. Per the coordinator's instruction, the held-out stage (articles 20-49: winner, its self-study variant, S6) is **not launched**, and the winner is not declared until S1 and S4 finish. The two running apps cannot be stopped by me (`modal app stop` was denied).
