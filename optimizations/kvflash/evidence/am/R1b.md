# R1b/R1c: Qwen3.8-27B, four configurations and head budgets (running log)

Pre-registration: `PREREG.md` sections "Rung R1b/R1c" and "R1b Amendment 1" (commits f8645db9, 88dc84ec). Code: `modal/am_qwen35/{r1b.py, r1b4.py, r1b_core.py, r1b_stats.py, r1b_report.py, r1b_merge.py}`, patches 0001-0003 on `adamzweiger/compaction` @ 2020f6d3, overlay `models/qwen3_5/`. Per-question rows: `R1b/27b/*.jsonl`, `R1b/4b/*.jsonl` (one file per cell, target and split; columns `article_id, qi, gold, parsed, correct, finite, answer, compaction_s, query_gen_s, ...`). Budget file and curves: `R1b/r1c/`.

## Protocol as run

27B: Qwen3.8-27B BF16, transformers 5.19.0, torch 2.8.0, sdpa, non-thinking, greedy, at most 256 new tokens, adapter exact without compaction (R1a). 4B reference: Qwen3-4B on the repo's own modeling, transformers 4.57.1, same driver and prompts (Amendment 1 item 1). Selection split = loader indices 0-19 (357 questions, 20 articles); target 0.10 of the article tokens. Per-head budgets: 27B `R1b/r1c/optimized_agnostic.json` (R1c below), 4B the repo's shipped `Qwen3-4B/optimized_agnostic.json`; both with `max_ratio_per_head` 0.75. Self-study generation for C uses the HF shim (Amendment 1 item 3) in both models. Paired cluster bootstrap over articles, 10,000 resamples, seed 0.

Nothing in the cell definitions, targets or pass rule changed after the first result was seen.

## R1(c): head-sensitivity curves and the budget file

- Fit articles loader indices 90-94 (5 articles, 18-19 reference answers each; 64 heads = 16 full-attention layers x 4 KV heads; 21 evaluation points 0 to 0.5; method `highest_attn_keys_rms_nnls2_-3_3_lsq`, query config `repeat`, baseline uniform at 0.05; reference answers greedy non-thinking, at most 128 tokens, Amendment 1 item 2).
- Cost: about 50-58 min of H200 per article (14,777 s of evaluation for the five), projected from article 90 as about $4.3 per article, so the full block (about $19) was below the $45 gate and was run with 5 fit articles and 21 points as registered.
- Solver: ratio-agnostic, target ratios 0.05 and 0.10, step 0.001; 271 iterations. The file is spiky: min 0.0006, median 0.0156 (uniform is 0.0156), max 0.0866 (L43H0), sums to 1.0. The solver's own objective (on the interpolated aggregated curves) improves from -0.57 to -1.05 at 0.10; the curves have deltas of order 0.01 log-perplexity, so most of that objective is within noise. Treat the file as a noisy allocation, not a validated one; the registered paired check (item 8) is pending.

## R1(b), selection split, target 0.10, 27B (n = 357 questions, 20 articles)

Own baseline (`orig` through the adapter): 89.6%.

| cell | acc % | degradation D vs own original, points [95% CI] | McNemar only-orig / only-X, p | compaction s / article | query generation s / article |
|---|---|---|---|---|---|
| C ceiling (ss-plus-repeat, OMP-fast, per-head) | 86.8 | 2.8 [0.3, 5.2] | 14 / 4, 0.031 | 345 | 225 |
| F floor (context-prefill, HA, per-head) | 86.6 | 3.1 [0.6, 5.7] | 16 / 5, 0.027 | 7 | 3 |
| O2 (repeat, OMP-fast, per-head) | 86.6 | 3.1 [1.4, 5.0] | 13 / 2, 0.007 | 122 | 11 |
| O1 (repeat, HA, per-head) | 82.1 | 7.6 [3.4, 11.4] | 35 / 8, <0.001 | 18 | 13 |
| U-AM (repeat, HA, uniform budget) | 79.8 | 9.8 [6.3, 13.4] | 44 / 9, <0.001 | 16 | 11 |
| random keys (uniform) | 82.4 | 7.3 [4.9, 9.8] | 33 / 7, <0.001 | 16 | 11 |
| truncate (uniform) | 60.5 | 29.1 [22.7, 35.7] | 111 / 7, <0.001 | 16 | 11 |

Timings are per article on one H100/H200 (mixed), compaction time includes query generation (the second column is the part of it spent generating queries; for C that is the HF self-study shim).

Controls (AM minus control, points; R0 rule): U-AM vs truncate +19.3 [+14.0, +24.7] -> AM better. U-AM vs random keys -2.5 [-5.4, +0.3] -> inconclusive (HA with uniform budgets is not better than random keys with the same fit on this model and split).

Online vs ceiling (online minus C): O2 -0.3 [-2.6, +1.9] -> within 3 points; F -0.3 [-2.3, +1.7] -> within 3 points; O1 -4.8 [-7.7, -1.4] -> inconclusive (not within 3, not clearly worse).

## R1(b) primary verdict: Delta against the matched-protocol 4B (selection, 0.10)

4B (matched protocol: same driver, non-thinking greedy 256): original 67.2%; C 67.8, F 68.3, O1 67.8, O2 68.9 (n = 357, 20 articles each). At target 0.10 the 4B does not degrade under this protocol (D = -0.6 to -1.7 points, every CI contains 0), so Delta(X) = D_27B(X) - D_4B(X) is essentially the 27B's own degradation plus a small amount.

| cell | D 27B | D 4B | Delta [95% CI], points | verdict (pass: upper <= +3.0; fail: lower > 0 and point > +3.0) |
|---|---|---|---|---|
| C | 2.8 | -0.6 | +3.4 [+0.8, +6.0] | FAIL |
| O2 | 3.1 | -1.7 | +4.8 [+1.9, +7.9] | FAIL |
| F | 3.1 | -1.1 | +4.2 [+0.3, +8.1] | FAIL |
| O1 | 7.6 | -0.6 | +8.1 [+3.4, +12.8] | FAIL |

By the registered rule, C fails, so **R1b overall is NOT PASSED** (it needs C to pass and one of O1/O2). The four verdicts are all FAIL (lower bound > 0 and point > +3.0), not inconclusive. Reading: the 27B loses 3 points at best (C, F, O2 cluster at 2.8-3.1, with overlapping CIs) and 7.6 for O1; the Qwen3-4B at 67% original accuracy shows no measurable loss at the same ratio in the same protocol, so the 27B hybrid is worse than the 4B by the registered yardstick. The R0 4B cells (thinking-mode, sampled, 2048 tokens, vLLM `original`; secondary reference, protocol-different) are in `R0_quality_selection.md`.

Observation, not a registered result: C, F and O2 are indistinguishable from each other (within 0.3 points); the extra queries of the ceiling (self-study, 225 s/article) buy nothing over `context-prefill` (3 s/article) with the same keys-and-per-head setup on this split, and OMP-fast on `repeat` matches both.

## Cost ledger (Modal, workspace davide-83641; running total, estimated at H200 $4.54/h or H100 $3.95/h)

Per-job cost records are in `modal/_dl/cost27` (scratch; the shared `cost_log.jsonl` lost entries to concurrent writes from the first parallel batch, so those rows are reconstructed from job wall times printed in the logs and counted at the H200 rate, an upper bound).

| item | wall | est. $ |
|---|---|---|
| R1(c) curves, articles 90-94 | 14,970 s (5 jobs) | 18.9 |
| 27B timing cell (article 0, 5 cells) | 625 s | 0.7 |
| 27B selection 0.10, 8 cells, 3 jobs | 12,094 s | 13.3 |
| 4B timing (articles 0-1) | about 2,100 s | 2.4 |
| 4B selection 0.10, 5 cells, 3 jobs | 36,019 s | 41.4 |

Subtotal about $77 of $270 at this commit. Remaining registered items are in progress (see below); this file is updated as blocks complete.

## Not yet run (at this commit)

Held-out (20-49) 0.10 for orig/O1/O2/C (running); selection 0.05 and controls; held-out 0.05; R1(c) validation on articles 95-99; 4B at 0.05.
