# R0-QuALITY selection block (articles 0-19, 357 questions, target 0.10)

Cells S1-S6 pre-registered in `PREREG.md` Amendment 1 (b966c2a9). Raw: `/qwen3-4b/sel_*_t0.1/` on volume `kvflash-am-results`.
`--compute-stats 0`, `best` config, on-policy variants, H100. CI = article-cluster bootstrap. Paired CI vs original on the same questions.

| Cell | Query source | Keys | Budget | Accuracy (n=357) | 95% CI | vs original (paired) | GPU-s/article (compaction + query gen) |
|---|---|---|---|---|---|---|---|
| original | - | - | - | 72.3% (258) | [66.7, 77.6] | - | 0 |
| S1 | repeat | HighestAttnKeys | per-head | 72.3% (258) | [68.2, 76.1] | 0.00 [-2.29, +2.28] | 29.2 |
| S2 | repeat | HighestAttnKeys | uniform | 62.2% | [57.1, 67.0] | (not paired here) | 26.7 |
| S3 | context-prefill | HighestAttnKeys | per-head | 70.3% | [65.2, 75.1] | (not paired here) | 13.3 |
| S4 | repeat | OMP-fast | per-head | 72.5% (259) | [67.5, 77.1] | +0.28 [-3.66, +4.14] | 713.9 |
| S5 | ss-plus-repeat | HighestAttnKeys | per-head | pending | | | |
| S6 | ss-plus-repeat | OMP-fast | per-head | pending | | | |

## What S1 separates
S1 vs S2 differ only in budget (both repeat queries, HighestAttnKeys): per-head budgets add +10.1 points (72.3% vs 62.2%). S1 vs S3 differ only in query source (both per-head): repeat-prefill is +2.0 points over context-prefill (72.3% vs 70.3%), inside the noise at n=357. So the 8-point jump seen between S2 and S3 in the partial table was the per-head budget, not the query source.

## Winner by the PREREG rule
Online cells = S1-S4 (query source context-prefill or repeat). Highest accuracy: S4 259/357, S1 258/357: a tie within 1 question, which the rule sends to the lower compaction cost: **S1 (HighestAttnKeys, repeat queries, per-head budgets, on-policy; 29 s per article vs 714 s for S4)**. S1 matches the full cache on the selection split.

## Cost
Selection invocations (S1+S4 19,579 s; S2, S3 and S5/S6 see manifests): see `R0_summary.md` for the final total.
