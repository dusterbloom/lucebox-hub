# R0 first pass: Attention Matching on Qwen3-4B (2026-10-13)

Exploratory, not pre-registered. Raw outputs and per-run `manifest.json`:
Modal volume `kvflash-am-results`, `/qwen3-4b/<run>/`.

Setup: github.com/adamzweiger/compaction @ 2020f6d3, Qwen/Qwen3-4B @ 1cfa9a72,
torch 2.8.0, transformers 4.57.1, vllm 0.11.0, H100 80GB. Scripts:
`modal/am_repro.py`, `modal/aggregate.py`. Cost ~10.8 H100-h (~$43).

## AIME 2025, 30 problems, seed 1 only (95% CI about ±18 points)

Repo script settings: max-seq-len 2048, target 0.2, AM-HighestAttnKeys-basic,
repeat queries. Matched baseline = similar total decoded length.

| Run | Correct | Avg decoded tokens | Matched baseline |
|---|---|---|---|
| baseline 2048 | 3/30 | 1884 | |
| baseline 4096 | 8/30 | 3932 | |
| baseline 8192 | 10/30 | 7325 | |
| baseline 16384 | 15/30 | 12045 | |
| 1 compaction | 6/30 | 3498 | 8/30 (4096) |
| 3 compactions | 7/30 | 6535 | 10/30 (8192) |
| 8 compactions | 13/30 | 12930 | 15/30 (16384) |

Paper (Table 9, AM-HighestAttnKeys-fast, 50% per compaction, mean of 4 runs):
baselines 1.25 / 7.75 / 13; compaction 8 at 4096, 13 at 8192. Different
method config, so only the trend compares. Not reproduced, not contradicted.

## QuALITY, 10 articles, 181 questions

Query config ss-plus-repeat, method highest_attn_keys_rms_nnls2_-3_3_lsq_on-policy.

| Target | Budgets | Full context | AM | random keys + fit | truncate + fit |
|---|---|---|---|---|---|
| 0.05 | uniform | 134 | 110 | 108 | 93 |
| 0.05 | per-head | 134 | 129 | | |
| 0.10 | uniform | 133 | 119 | 111 | 85 |
| 0.10 | per-head | 136 | 130 | | |

Full context varies 133–136 across runs: nondeterminism. Per-head budgets
close most of the gap. AM beats truncation clearly; against random keys with
the same fitted β and values the margin (1–4 points) is within noise.

## Not done yet (needed before "reproduced")

- AIME seeds 2–4 and the paper's own config (fast, 50% per compaction).
- Full method matrix (OMP, OMP-fast, context-prefill, self-study) and R0s.
- QuALITY numbers read from the paper's Figure 4 for comparison.
