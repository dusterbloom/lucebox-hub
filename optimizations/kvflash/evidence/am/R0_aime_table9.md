# R0-AIME: paper Table 9 (online compaction), Qwen3-4B, 4 seeds

Pre-registered in `PREREG.md` (commit 6f49fa0f). Code sha 2020f6d37940fe013e7e7f1ebf409e5d107c5670, unmodified. H100 80GB.
Config: `AM-HighestAttnKeys-basic` (= paper's AM-HighestAttnKeys-fast), query `repeat`, algorithm config `default`,
`--max-seq-len 2048 --target-size 0.5`, `--max-compactions {2,6}`, seeds 1-4, 30 AIME 2025 problems (opencompass/AIME2025, I + II).
Baselines: `--mode baseline` at 2048/4096/8192, seeds 1-4 (seed 1 reused from the first pass: same command).
Raw JSONs + manifests: volume `kvflash-am-results`, `/qwen3-4b/t9_compact_2048_t0.5_c{2,6}_s{1..4}_p{0..4}/` and `/qwen3-4b/baseline_{2048,4096,8192}_s{1..4}_p*/` (5 shards of 6 problems per run).

| Config (eff. length) | Per-seed correct /30 | Mean /30 | 95% CI (t, df=3) | Paper Table 9 | Paper in CI |
|---|---|---|---|---|---|
| baseline 2048 | 3, 1, 3, 2 | 2.25 | [0.73, 3.77] | 1.25 | yes |
| baseline 4096 | 8, 8, 8, 9 | 8.25 | [7.45, 9.05] | 7.75 | yes |
| baseline 8192 | 10, 12, 14, 12 | 12.00 | [9.40, 14.60] | 13 | yes |
| compaction c<=2 (eff ~4096) | 8, 7, 7, 8 | 7.50 | [6.58, 8.42] | 8 | yes |
| compaction c<=6 (eff ~8192) | 12, 13, 12, 11 | 12.00 | [10.70, 13.30] | 13 | yes |

Paired (compaction - matching baseline), per-problem scores averaged over the 4 seeds, bootstrap over problems (10,000):
- c<=2 vs baseline 4096: -0.75 problems, 95% [-1.50, +0.00].
- c<=6 vs baseline 8192: 0.00 problems, 95% [-1.50, +1.50].

Mean generated tokens: c<=2 ~3.9k, c<=6 ~7.1k (matches the baselines' effective lengths). Compaction cost per problem: ~22 s (c<=2), ~80 s (c<=6) of fit time on H100, on top of ~86 s / ~240 s generation (HF eager, batch 1).

## Verdict per PREREG
1. Paper's Table 9 values lie inside our 95% CI for all 5 configurations: reproduced.
2. Non-inferiority (lower paired bound >= -3 problems): passes at both lengths (-1.50 and -1.50).
So the claim "reasoning survives repeated mid-trajectory compaction at the same effective length" holds in our run: up to 6 compactions at physical cache 2048 cost no measurable accuracy (0 +- 1.5 problems of 30).

Caveats: 30 problems x 4 seeds; the CI with df=3 is wide (the paper's 4-seed means are in it, but it cannot exclude differences of ~1-2 problems). Our baseline 2048 is above the paper's (2.25 vs 1.25), inside noise. Baseline 2048 and 4096 outputs are nearly seed-independent (forced answer at the cap), so seed variance there is mostly the final-answer sampling.

Exploratory first pass (target 0.2 per compaction, repo script `qwen-reasoning.sh`, seed 1; not pre-registered): c=1/3/8 gave 6/7/13 of 30 vs baselines at 4096/8192/16384 of 8/10/15 (`R0_first_pass.md`).

## Cost
This block: 85 shards, 14.3 H100-h, ~$56 at the Modal H100 list rate (manifests: `est_cost_usd`). Over my $50 projection by $6 (shard model-load overhead).

Reproduce: `modal run --detach optimizations/kvflash/modal/am_repro.py::aime_table9`; analysis: `python3 -I optimizations/kvflash/modal/analyze_t9.py <downloaded /qwen3-4b dir>`.
