# R0-QuALITY held-out block (target 0.10, Qwen3-4B, H100)

Selection winner S1 chosen on articles 0-19 (`R0_quality_selection.md`); held-out per `PREREG.md` Amendment 2 (commit 39acd6b8, written before any held-out result was read; the held-out jobs were submitted a few minutes before that commit, results were read after it).
Raw: `/qwen3-4b/heldout_win_a{20,30,40}_10/` and `/qwen3-4b/heldout_ceil_a20_10/` on volume `kvflash-am-results`. CIs: article-cluster bootstrap, 10,000 resamples.

| Arm | Articles | n questions | Accuracy | 95% CI | GPU-s/article (compaction + query gen) |
|---|---|---|---|---|---|
| original (full cache) | 20-49 | 537 | 72.4% | [67.8, 77.0] | 0 |
| S1 online: HighestAttnKeys, repeat queries, per-head, on-policy | 20-49 | 537 | 70.8% | [66.1, 75.5] | 29.1 |
| S6 ceiling: OMP-fast, ss-plus-repeat, per-head, on-policy | 20-29 | 177 | 76.3% | [69.2, 82.8] | 831 |

Paired:
- S1 - original (n=537, 30 articles): -1.68 points, 95% CI [-4.49, +1.12], discordant 65 (28/37), McNemar p=0.32. S1 stays within ~3 points of the full cache on held-out; the CI cannot exclude a 4-point loss.
- **Primary comparison, S1 (best online) - S6 (ceiling), n=177 (10 articles): -5.08 points, 95% CI [-7.95, -2.27]**, discordant 19 (S6 right only 14, S1 right only 5), McNemar p=0.064.
- S6 - original (n=177): +1.69 points [-1.18, +5.00].

## Verdicts per PREREG
- Primary (online within 3 points of the ceiling, lower bound >= -3): **not met**. The lower bound is -7.95; the CI excludes zero in favour of the ceiling. The selection split agrees in direction and size (S6 - S1 = +3.08 [+0.27, +6.11], n=357, exploratory because S1 was selected there). Caveats: the held-out n is 177 questions, so the point estimate is imprecise (the CI width is ~5.7 points); the ceiling is OMP-fast, not plain OMP (Amendment 1/2); the two arms differ in key selection and in query source, so the gap is not attributed to either one. Cost ratio: the ceiling's compaction costs 29x more (831 vs 29 s per article), before the vLLM self-study generation it requires.
- Stale queries (R0s): not run on held-out (cost cut, Amendment 2). On the selection split S5 - S1 = 0.00 [-2.06, +2.23], n=357: no measurable loss from fitting with prefill-time queries only; that result is on the same articles used for selection and has a smaller n than pre-registered, so treat it as supporting, not confirming.
- Ran with held-out n below the pre-registered 95 articles for all arms (30 and 10 articles); minimum detectable differences are ~6 points (S1 vs original) and ~10 points (primary), so only large effects are resolved. The primary result happened to be large enough to resolve.

## Paper Figure 4 (Qwen3-4B QuALITY, read from the plot, approximate)
Paper, compacted size 0.10 (50 articles): original ~0.71; AM-HighestAttnKeys ~0.71; AM-OMP not plotted at 0.10 (at 0.05: ~0.71); KVzip ~0.645; H2O+ ~0.605; no-context ~0.37.

| Ours at target 0.10 | Accuracy | Where |
|---|---|---|
| original | 72.3% (art 0-19), 72.4% (20-49), 76.5% (all 115) | selection, held-out, vs-random run |
| AM HighestAttnKeys, repeat, per-head (S1) | 72.3% (0-19), 70.8% (20-49) | matches the paper's ~0.71 within noise |
| AM OMP-fast + self-study + per-head (S6) | 75.4% (0-19), 76.3% (20-29) | paper's AM-OMP ~0.71 at 0.05; our value is above the paper's plot, n small |
| AM HighestAttnKeys, repeat, uniform (S2; 115 articles) | 62.2% (0-19); 65.0% (115 articles) | below the paper: the paper uses per-head budgets |
| random keys, same fit (115 articles) | 59.6% | `R0_quality_vsrandom.md` |

Our absolute accuracies are within ~2 points of the paper's plot for the per-head configurations on 20-30 articles (our full-cache accuracy varies 72-76% by article subset; the paper's is ~71% on its 50 articles). The paper reports no number for per-head vs uniform on this model at this size in text, so the 10-point per-head benefit is our measurement (S1 vs S2, same 357 questions, difference of accuracies).
