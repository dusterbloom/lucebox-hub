# R0-QuALITY: AM-HighestAttnKeys vs random keys with the same fit (all 115 articles)

Pre-registered in `PREREG.md` (6f49fa0f) and Amendment 1 (b966c2a9: target 0.10 only, truncate not run).
Code sha 2020f6d37940fe013e7e7f1ebf409e5d107c5670, Qwen3-4B, H100, `--compute-stats 0`, algorithm config `key-selection`,
query config `repeat`, uniform budget, target 0.10, 115 articles / 2086 questions (5 article-range shards, 23 articles each).
Raw: volume `kvflash-am-results` `/qwen3-4b/vsrandom_t0.1_repeat_a{0,23,46,69,92}_23/`.

| Method | n questions | Accuracy | 95% CI (article-cluster bootstrap) | Compaction + query-gen s / article |
|---|---|---|---|---|
| original (full cache) | 2086 | 76.5% | [74.0, 78.9] | 0 |
| AM `highest_attn_keys_rms_nnls2_-3_3_lsq_on-policy` | 2086 | 65.0% | [62.3, 67.7] | 24.5 |
| random subset keys + same NNLS beta / lsq value fit (`random_subset_keys_nnls2_-3_3_lsq_on-policy`) | 2086 | 59.6% | [56.6, 62.7] | 24.1 |

Paired differences (same questions, 10,000 article-cluster bootstrap resamples):
- AM - random keys: **+5.37 points, 95% CI [+3.26, +7.60]**; discordant 460 (AM right only 286, random right only 174), exact McNemar p < 0.001.
- original - AM: +11.46 points, 95% CI [+9.50, +13.46].

## Verdict per PREREG
H1 (AM beats random keys by >= 3 points): the CI lower bound is above 0 (AM better) and also above 3 (+3.26): supported. The first-pass margin (1-4 points at n=181) was consistent with this but not resolvable; at n=2086 it is resolved at ~5 points.
So selecting keys by attention matters, not only the beta/value fit applied to any subset. Both methods cost the same to fit (~24 s per article, H100).

## Caveats
- Uniform budget, repeat-prefill queries, no self-study: this is the weakest AM variant. The 11.5-point gap to the full cache is expected here; the paper's ~0.70 vs 0.71 (Fig. 4, Qwen3-4B, 5% and 10%) uses per-head budgets and self-study queries (see the selection table for those cells).
- Random keys use the repo's seed handling (not varied by us): one draw per article.
- Target 0.05 and truncate were not run (cost gate, Amendment 1).

Cost: 8.5 H100-h (summed shard wall time), ~$34 (projection was $25: model/vLLM load per shard and generation from the compacted cache dominate).

Reproduce: `modal run --detach optimizations/kvflash/modal/am_repro.py::stage_vs_random`; analysis `python3 -I optimizations/kvflash/modal/analyze_quality.py <downloaded /qwen3-4b dir> 'vsrandom_*' highest_attn`.
