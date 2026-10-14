# R1b/R1c: Qwen3.8-27B, four configurations and head budgets

Pre-registration: `PREREG.md` sections "Rung R1b/R1c" and "R1b Amendment 1" (commits f8645db9, 88dc84ec). Code: `modal/am_qwen35/{r1b.py, r1b4.py, r1b_core.py, r1b_stats.py, r1b_report.py, r1b_merge.py}`, patches 0001-0003 on `adamzweiger/compaction` @ 2020f6d3, overlay `models/qwen3_5/`; CPU tests `tests/test_{adapter,onpolicy,influence}.py` (13 pass, tiny hybrid with the real tokenizer). Per-question rows: `R1b/27b/*.jsonl` and `R1b/4b/*.jsonl` (one file per cell, target and split). Budget file, curves, validation: `R1b/r1c/`. Tables below are produced by `python3 modal/am_qwen35/r1b_report.py R1b/27b R1b/4b`.

## Protocol as run

27B: Qwen3.8-27B BF16, transformers 5.19.0, torch 2.8.0, sdpa, non-thinking, greedy, at most 256 new tokens, adapter exact without compaction (R1a). 4B reference: Qwen3-4B on the repo's own unpatched modeling, transformers 4.57.1, same driver, prompts, splits, query configs, key selection and decoding (Amendment 1 item 1). Selection split = loader indices 0-19 (357 questions, 20 articles); held-out = 20-49 (537 questions, 30 articles). Targets 0.10 and 0.05 of the article tokens. Per-head budgets: 27B `R1b/r1c/optimized_agnostic.json`, 4B the repo's shipped `Qwen3-4B/optimized_agnostic.json`; both with `max_ratio_per_head` 0.75. Self-study generation for C uses the HF shim (Amendment 1 item 3) in both models, seed 1. Paired cluster bootstrap over articles, 10,000 resamples, seed 0; McNemar exact as a secondary.

Cell definitions, targets and the pass rule were not changed after the first result was seen. Every priority-list item ran; nothing was cut. The R0 selection winner on the 4B (S1 = HA + repeat + per-head + on-policy) is configuration O1 here, so there is no gap against R0.

## Verdict

**R1b is NOT PASSED.** The registered primary (selection, 0.10, 20 articles / 357 questions) gives FAIL for every configuration including the ceiling C (Delta = 27B degradation minus matched-protocol 4B degradation, pass if the upper 95% bound <= +3.0, fail if the lower bound > 0 and the point estimate > +3.0):

| cell | D 27B | D 4B | Delta [95% CI], points | verdict |
|---|---|---|---|---|
| C ceiling | 2.8 | -0.6 | +3.4 [+0.8, +6.0] | FAIL |
| O2 | 3.1 | -1.7 | +4.8 [+1.9, +7.9] | FAIL |
| F floor | 3.1 | -1.1 | +4.2 [+0.3, +8.1] | FAIL |
| O1 | 7.6 | -0.6 | +8.1 [+3.4, +12.8] | FAIL |

Overall rule (C passes and one of O1/O2 passes) is not met. What the numbers say: at 0.10 the 27B loses about 3 points with C, F or O2 (its own baseline is 89.6%) and 7.6 with O1; the matched 4B (original 67.2% in this protocol) shows no measurable loss at the same ratio (every 4B CI contains 0). Against that yardstick the 27B hybrid compacts worse than the 4B by 3.4 to 8 points. The 4B result is itself a floor effect of this protocol (67% accuracy in non-thinking greedy leaves little to lose), so the FAIL is a statement relative to the registered yardstick, not that the 27B compaction is broken: it keeps 83-87% of 89.6% with a 10x smaller cache in the full-attention layers.

## R1(b), 27B, selection split, original = 89.6% (n = 357 q, 20 articles)

D = accuracy(original) - accuracy(cell), points [95% CI]. Time columns are per article on one H100/H200; the second is the part of the first spent on query generation (for C the HF self-study shim).

Target 0.10:

| cell | acc % | D [95% CI] | McNemar only-orig / only-X, p | compaction s | query gen s |
|---|---|---|---|---|---|
| C ceiling (ss-plus-repeat, OMP-fast, per-head) | 86.8 | 2.8 [0.3, 5.2] | 14 / 4, 0.031 | 345 | 225 |
| F floor (context-prefill, HA, per-head) | 86.6 | 3.1 [0.6, 5.7] | 16 / 5, 0.027 | 7 | 3 |
| O2 (repeat, OMP-fast, per-head) | 86.6 | 3.1 [1.4, 5.0] | 13 / 2, 0.007 | 122 | 11 |
| O1 (repeat, HA, per-head) | 82.1 | 7.6 [3.4, 11.4] | 35 / 8, <0.001 | 18 | 13 |
| U-AM (repeat, HA, uniform budget) | 79.8 | 9.8 [6.3, 13.4] | 44 / 9, <0.001 | 16 | 11 |
| random keys (uniform) | 82.4 | 7.3 [4.9, 9.8] | 33 / 7, <0.001 | 16 | 11 |
| truncate (uniform) | 60.5 | 29.1 [22.7, 35.7] | 111 / 7, <0.001 | 16 | 11 |

Target 0.05:

| cell | acc % | D [95% CI] | McNemar only-orig / only-X, p | compaction s | query gen s |
|---|---|---|---|---|---|
| C | 84.6 | 5.0 [2.5, 7.5] | 21 / 3, <0.001 | 306 | 271 |
| F | 84.0 | 5.6 [2.5, 8.5] | 26 / 6, 0.001 | 6 | 4 |
| O2 | 84.9 | 4.8 [2.0, 7.4] | 23 / 6, 0.002 | 44 | 11 |
| O1 | 79.6 | 10.1 [6.1, 14.3] | 46 / 10, <0.001 | 16 | 13 |
| U-AM | 75.9 | 13.7 [9.3, 18.1] | 59 / 10, <0.001 | 15 | 11 |
| random keys | 75.1 | 14.6 [10.5, 18.7] | 59 / 7, <0.001 | 14 | 11 |
| truncate | 54.6 | 35.0 [29.5, 40.5] | 131 / 6, <0.001 | 14 | 11 |

Online vs ceiling, selection (online minus C, points; R0 rule: lower bound >= -3 -> within 3 points):

| cell | 0.10 | verdict | 0.05 | verdict |
|---|---|---|---|---|
| O2 | -0.3 [-2.6, +1.9] | within 3 points | +0.3 [-1.9, +2.5] | within 3 points |
| F | -0.3 [-2.3, +1.7] | within 3 points | -0.6 [-3.7, +2.4] | inconclusive |
| O1 | -4.8 [-7.7, -1.4] | inconclusive | -5.0 [-8.9, -1.4] | inconclusive |

U-AM vs controls (AM minus control; R0 rule):

| control | 0.10 | verdict | 0.05 | verdict |
|---|---|---|---|---|
| truncate | +19.3 [+14.0, +24.7] | AM better | +21.3 [+16.1, +26.5] | AM better |
| random keys, same fit | -2.5 [-5.4, +0.3] | inconclusive | +0.8 [-3.8, +5.3] | inconclusive |

Per-head budgets vs uniform for the same method (O1 minus U-AM, points): 0.10 +2.2 [-0.8, +5.3]; 0.05 +3.6 [+0.3, +6.8]. On the 4B, R0 measured +10 points for per-head budgets over uniform; on the 27B they are worth 2 to 4 points here.

## R1(b), 27B, held-out split (20-49; n = 537 q, 30 articles), original = 85.7%

| cell | target | acc % | D [95% CI] | online minus C | verdict |
|---|---|---|---|---|---|
| C | 0.10 | 83.6 | 2.0 [0.4, 3.6] | | |
| O2 | 0.10 | 83.4 | 2.2 [-0.2, 4.8] | -0.2 [-1.9, +1.3] | within 3 points |
| O1 | 0.10 | 79.1 | 6.5 [3.8, 9.4] | -4.5 [-7.0, -2.0] | inconclusive |
| C | 0.05 | 80.8 | 4.8 [2.8, 7.1] | | |
| O2 | 0.05 | 79.1 | 6.5 [3.9, 9.2] | -1.7 [-3.9, +0.6] | inconclusive |
| O1 | 0.05 | 76.0 | 9.7 [6.1, 13.5] | -4.8 [-7.6, -2.1] | inconclusive |

Held-out agrees with selection: O2 is within 3 points of the ceiling at 0.10 on both splits and O1 is 4.5-5 points below it on every split and target.

## Matched-protocol 4B (Qwen3-4B, same driver), selection split, original = 67.2%

| cell | acc % 0.10 | D 0.10 | acc % 0.05 | D 0.05 |
|---|---|---|---|---|
| C | 67.8 | -0.6 [-2.6, 1.4] | 66.1 | 1.1 [-1.7, 3.8] |
| F | 68.3 | -1.1 [-5.1, 2.8] | 69.2 | -2.0 [-5.4, 1.4] |
| O1 | 67.8 | -0.6 [-4.0, 2.5] | 63.3 | 3.9 [-1.2, 8.8] |
| O2 | 68.9 | -1.7 [-4.8, 1.1] | 67.2 | 0.0 [-4.0, 3.9] |

Compaction time per article on the 4B: O1 29 s, F 13 s, O2 717 s, C 1016 s (target 0.10).

Secondary, Delta at 0.05 (descriptive, same rule): C +3.9 [+0.3, +7.8] FAIL; O2 +4.8 [+0.8, +8.9] FAIL; F +7.6 [+2.7, +12.6] FAIL; O1 +6.2 [-0.6, +13.2] inconclusive.

## R1(c): head-sensitivity curves and the budget file

- Fit articles loader indices 90-94 (5 articles, 64 heads = 16 full-attention layers x 4 KV heads, 21 evaluation points 0 to 0.5, method `highest_attn_keys_rms_nnls2_-3_3_lsq`, query config `repeat`, baseline uniform at 0.05; reference answers greedy non-thinking, at most 128 tokens, Amendment 1 item 2). About 40-58 min per article; the projected block cost ($19) was below the $45 gate, so no reduction was made.
- Ratio-agnostic solve for 0.05 and 0.10: 271 iterations; the file is spiky (min 0.0006, median 0.0156, max 0.0866 at L43H0; sums to 1.0). The curve deltas are of order 0.01 log-perplexity, so the solver's own objective improvement is largely noise.
- Registered check (articles 95-99, not in the fit; mean log-perplexity of reference answers after compaction at 0.10, per-head file vs uniform, same method and queries): per-head lower on 5 of 5 articles, mean difference -0.094 (0.14 vs 0.20, 0.77 vs 0.80, 0.79 vs 0.90, 0.40 vs 0.66, 0.42 vs 0.44). Expectation (per-head <= uniform) met, on 5 articles.

## Cost ledger (Modal workspace davide-83641, estimated at H200 $4.54/h and H100 $3.95/h; cap $270)

| item | est. $ |
|---|---|
| R1(c) curves, articles 90-94 (5 jobs, 14,970 s; reconstructed from job logs at the H200 rate, upper bound, because concurrent writes lost their cost rows) | 18.9 |
| R1(c) validation | 0.4 |
| 27B timing cell | 0.7 |
| 27B selection 0.10 (8 cells, 3 jobs) | 13.3 |
| 27B held-out 0.10 (4 cells, 3 jobs) | 18.5 |
| 27B selection 0.05 (8 cells, 3 jobs) | 12.7 |
| 27B held-out 0.05 (4 cells, 3 jobs) | 16.0 |
| 4B timing (articles 0-1) | 2.4 |
| 4B selection 0.10 (5 cells, 3 jobs) | 41.4 |
| 4B selection 0.05 (5 cells, 3 jobs) | 14.6 |
| total | about 139 |

Under the registered $270 cap with $131 unspent. The R1a block ($15 cap) and R0 ($150 cap) are accounted separately.

## Not run / limits

- 4B held-out cells (the pass rule is selection-only; not needed) and 4B controls were not run.
- n is 20 articles / 357 questions for the primary; paired SE of Delta is 2.5-3 points, as registered. The primary FAIL is outside noise for all four cells (lower bounds +0.3 to +3.4), but the margin for F (+0.3) and C (+0.8) is thin.
- One run per cell, greedy decoding; only self-study sampling has randomness (seed 1). Runs landed on a mix of H100 and H200; decode is greedy in bf16, so cross-GPU numerics may flip a few borderline answers.
- Query generation timing for C is dominated by the HF `generate` shim (225-292 s per article), not by a tuned inference server; a production number would be lower.

## Amendment 2 result: cell F on the 27B held-out split (post hoc, exploratory)

Added after the R1b results were read (PREREG R1b Amendment 2). F = context-prefill queries, HighestAttnKeys, per-head budgets (`r1c/optimized_agnostic.json`, max ratio per head 0.75), held-out loader indices 20-49, n = 537 q, 30 articles. Rows: `R1b/27b/F__t0.1__held.jsonl`, `F__t0.05__held.jsonl`.

GPU parity: F ran on the GPU type of the `orig` cell of the same article range. Target 0.10: articles 20-29 H200, 30-39 H100 (pinned `H100!`), 40-49 H200. Target 0.05: all H200 (as the existing 0.05 held-out cells). A first launch used the wrong `--start` (it is relative to the split start) and produced only articles 40-49 plus two empty jobs; the missing ranges were then run with corrected arguments, no article was run twice. Cost about $1.8 (cost records $1.75 plus $0.08 for the two empty jobs), cap $15.

Correct answers out of 537:

| cell | 0.10 | 0.05 |
|---|---|---|
| orig | 460 (85.7%) | 460 (85.7%) |
| C | 449 (83.6%) | 434 (80.8%) |
| O2 | 448 (83.4%) | 425 (79.1%) |
| F | 439 (81.8%) | 431 (80.3%) |
| O1 | 425 (79.1%) | 408 (76.0%) |

Degradation D = orig - F (points, paired, cluster bootstrap 10,000, seed 0): 0.10: 3.9 [2.0, 5.9], McNemar 32/11, p = 0.002; 0.05: 5.4 [3.1, 7.9], McNemar 46/17, p < 0.001.

Paired differences, F minus cell (points, 95% CI; "within 3 points" = lower bound >= -3, unchanged R1b rule):

| comparison | 0.10 | verdict | 0.05 | verdict |
|---|---|---|---|---|
| F - C | -1.9 [-3.6, -0.2] | inconclusive | -0.6 [-3.0, +1.7] | within 3 points |
| F - O2 | -1.7 [-3.8, +0.6] | inconclusive | +1.1 [-1.1, +3.6] | within 3 points |
| F - O1 | +2.6 [-0.2, +5.5] | within 3 points | +4.3 [+0.6, +8.3] | within 3 points |

Reading: at 0.05 F is statistically indistinguishable from C and O2 and better than O1 (lower bound +0.6). At 0.10 F sits between O2/C and O1: F - O1 is positive with a CI touching zero, and F - C and F - O2 are not resolved by this n. Compaction time per article: F 6 s (0.10) and 4 s (0.05) against 346 s and 327 s for C. F is not shown to match the ceiling at 0.10. Exploratory; no earlier verdict changes.

## Amendment 3 result: F-UNI, F with a uniform budget (post hoc, exploratory)

Added after the R1b results were read (PREREG R1b Amendment 3). F-UNI = context-prefill queries, HighestAttnKeys, the same uniform budget for every head (no `optimized_agnostic.json`). Selection split (loader indices 0-19, 357 q) and held-out split (20-49, 537 q), targets 0.10 and 0.05. Rows: `R1b/27b/FUNI__t{0.1,0.05}__{sel,held}.jsonl`; driver cell `FUNI` in `r1b_core.py`, launch `r1b_funi_launch.sh`, tables `r1b_funi_report.py`. Cluster bootstrap 10,000 resamples, seed 0.

GPU parity: each range ran on the GPU type of its `orig` cell. Selection 0.10: H100 (articles 0-19); selection 0.05: H200; held-out 0.10: 20-29 H200, 30-39 H100 (pinned `H100!`), 40-49 H200; held-out 0.05: H200. `--start` was relative to the split start; each article ran once. Cost $2.50 (cost records: 0.36 + 0.48 + 0.39 + 0.27 + 0.28 + 0.72), cap $5.

Correct answers (the F, orig and U-AM columns reuse the existing rows; U-AM exists on the selection split only):

| split | target | n q | orig | F | FUNI | UAM |
|---|---|---|---|---|---|---|
| sel | 0.1 | 357 | 320 (89.6%) | 309 (86.6%) | 306 (85.7%) | 285 (79.8%) |
| sel | 0.05 | 357 | 320 (89.6%) | 300 (84.0%) | 280 (78.4%) | 271 (75.9%) |
| held | 0.1 | 537 | 460 (85.7%) | 439 (81.8%) | 435 (81.0%) | - |
| held | 0.05 | 537 | 460 (85.7%) | 431 (80.3%) | 417 (77.7%) | - |

| split | target | comparison | n q | diff points [95% CI] | McNemar only-A/only-B, p | verdict |
|---|---|---|---|---|---|---|
| sel | 0.1 | D = orig - F-UNI | 357 | 3.9 [1.2, 6.5] | 20/6, 0.009 | |
| sel | 0.1 | F - F-UNI | 357 | +0.8 [-1.1, +3.0] | 12/9, 0.664 | uniform within 3 points of F |
| sel | 0.1 | F-UNI - U-AM | 357 | +5.9 [+3.2, +8.5] | 31/10, 0.001 | AM better |
| sel | 0.05 | D = orig - F-UNI | 357 | 11.2 [7.8, 14.8] | 46/6, 0.000 | |
| sel | 0.05 | F - F-UNI | 357 | +5.6 [+1.7, +9.8] | 29/9, 0.002 | inconclusive |
| sel | 0.05 | F-UNI - U-AM | 357 | +2.5 [-1.1, +5.9] | 26/17, 0.222 | inconclusive |
| held | 0.1 | D = orig - F-UNI | 537 | 4.7 [2.6, 6.8] | 39/14, 0.001 | |
| held | 0.1 | F - F-UNI | 537 | +0.7 [-0.6, +2.0] | 15/11, 0.557 | uniform within 3 points of F |
| held | 0.05 | D = orig - F-UNI | 537 | 8.0 [5.5, 10.6] | 60/17, 0.000 | |
| held | 0.05 | F - F-UNI | 537 | +2.6 [+0.2, +5.1] | 22/8, 0.016 | inconclusive |

Verdict column for F - F-UNI uses the R1b rule in the direction of the decision: "uniform within 3 points of F" when the upper 95% bound of F - F-UNI is <= +3 (equivalently the lower bound of F-UNI - F >= -3). Other verdicts: inconclusive when the CI is wider than the margin; for F-UNI - U-AM the R0 rule (lower bound > 0 is "AM better").

Reading: at 0.10 per-head budgets add +0.8 [-1.1, +3.0] on selection and +0.7 [-0.6, +2.0] on held-out, both inside 3 points, so uniform budgets cost nothing resolvable at the 0.10 target and the engine can use them (decision rule of Amendment 3 item 5 met). At 0.05 per-head budgets add +5.6 [+1.7, +9.8] (selection) and +2.6 [+0.2, +5.1] (held-out), significant and not bounded within 3 points. Under uniform budgets, context-prefill queries beat the repeat-prompt queries at 0.10 (F-UNI - U-AM +5.9 [+3.2, +8.5]); at 0.05 the difference is +2.5 [-1.1, +5.9], inconclusive. Exploratory; no earlier verdict changes.
