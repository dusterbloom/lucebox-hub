# Pre-registration: Attention Matching reproduction (R0, R0s)

Written before any run of this second launch. Follows `../../PLAN.md` fidelity rules 1-8.
Anything changed after seeing data is appended under "Deviations" with date and reason.
The first pass (`R0_first_pass.md`, 1 seed, 10 articles) is exploratory and is not used for any decision below
except to know which numbers are already seen (selection articles 0-9 overlap it).

## Fixed provenance

- Code: https://github.com/adamzweiger/compaction at sha 2020f6d37940fe013e7e7f1ebf409e5d107c5670, unmodified. Configs used as shipped; no new config files, no algorithm edits.
- Image: python 3.12, torch 2.8.0, transformers 4.57.1, vllm 0.11.0, accelerate 1.12.0, datasets 4.4.1. Model Qwen/Qwen3-4B rev 1cfa9a7208912126459214e8b04321603b3df60c. GPU H100 80GB.
- Driver: `optimizations/kvflash/modal/am_repro.py`. Results on Modal volume `kvflash-am-results` under `/qwen3-4b/<run>/` with `manifest.json` per run.
- Budget cap for this launch: $150 at the Modal H100 rate (~$3.95/h, 38 H100-h). The first pass overspent its cap; this launch is cost-gated (see "Cost gate").

## QuALITY splits (QuALITY.v1.0.1 dev, repo-bundled; 115 unique articles, 2086 questions, loader order)

- Selection split: loader indices 0-19 (`--start-article 0 --n-articles 20`), 357 questions. Article ids:
  52845 30029 62139 63523 63401 62476 63041 30035 61285 62261 62314 61430 52855 62085 62498 61119 63616 61467 60412 63855
- Held-out split: loader indices 20-114 (`--start-article 20 --n-articles 95`), 1729 questions. Article ids:
  63633 63473 61434 63150 63645 61146 63936 20007 20008 20006 62382 63862 40965 32665 55815 63812 63392 63130 63916 63833 20002 51650 51483 51461 50818 51687 51027 51267 51351 51605 49897 51126 51350 50948 51320 51395 51436 51362 51150 51274 51413 20041 20048 20020 20051 20056 20044 20031 20029 20027 20055 51295 20064 20068 51688 20075 20073 51256 50826 20077 22073 22102 22218 22346 22462 22524 22579 22590 22867 22875 22876 22958 22966 22967 23104 23160 23563 24290 31736 59368 59679 31282 60897 31599 60745 99905 99922 99916 99915 99919 99930 99927 99929 99911 99914
- Methods are chosen on the selection split only. Held-out is run once per chosen configuration.
- Timing-only pilot (a few articles from the selection split, throwaway) is allowed to size the matrix; it informs cost, not method choice. It is declared in Amendment 1.

## Statistics (all rungs)

- Unit: question (QuALITY), problem x seed (AIME). Paired comparisons share the same questions/problems.
- QuALITY CI: paired cluster bootstrap over articles, 10,000 resamples, percentile 95%. Report n questions, n articles, mean, CI. McNemar exact as a secondary check.
- AIME CI: per-seed accuracy over 30 problems; mean over 4 seeds with t-interval (df=3) and, as a second view, bootstrap over problems of the seed-averaged per-problem score.
- "Reproduced" = the paper's number lies inside our 95% CI. Where the paper gives only a plot, its value is read from the figure by eye and marked approximate; no pass/fail is claimed from it.

## Rung R0-AIME: online compaction, paper's own config (Table 9, App. F.3)

- Hypothesis: compacting mid-trajectory preserves AIME accuracy at the same effective length (paper: 8 and 13 of 30 vs baselines 7.75 and 13).
- Config: `AM-HighestAttnKeys-basic` (= the paper's AM-HighestAttnKeys-fast: no on-policy, uniform), query config `repeat`, algorithm config `default`, `--max-seq-len 2048 --target-size 0.5`, `--max-compactions 2` (effective ~4096) and `6` (effective ~8192), protected tokens 20 (repo default). Baselines: `--mode baseline` at max-seq-len 2048, 4096, 8192. Seeds 1, 2, 3, 4. Seed-1 baselines from the first pass are the same command and are reused.
- Metric: pass@1 over 30 problems per seed. n = 30 problems x 4 seeds.
- Pass: (1) paper's Table 9 values (baselines 1.25 / 7.75 / 13; compaction 8 / 13) lie in our 95% CI; (2) non-inferiority: paired (compaction - matching baseline), lower 95% bound >= -3 problems of 30 (-10 points) at each effective length. If (1) fails but (2) holds, report "claim reproduced, numbers differ" and keep both facts.
- Exploratory, not pre-registered pass criteria: the first-pass runs at target 0.2 (repo script `qwen-reasoning.sh`).

## Rung R0-QuALITY: method matrix on the selection split

- Hypothesis: the paper's ranking holds: self-study + OMP + on-policy + per-head budgets >= self-study + HighestAttnKeys > prefill-only queries; per-head budgets help; AM beats trivial compactors.
- Factors (all from `best.py`, unmodified): query source {context-prefill, repeat, ss-plus-repeat} x key selection {HighestAttnKeys `highest_attn_keys_rms_nnls2_-3_3_lsq`, OMP `omp_nnls0_-inf_7_drop-7_lsq`, OMP-fast `..._k4int2`} x on-policy {off, on (`_on-policy` suffix)} x budget {uniform, optimized_agnostic with max-ratio-per-head 0.75} x target {0.05, 0.10}. Algorithm config `best`. Full factorial = 72 cells. Controls in the same run: `original` (full cache) in every invocation; `random_subset_keys_nnls2_-3_3_lsq_on-policy` and `truncate_nnls2_-3_3_lsq_on-policy` (config `key-selection`) at each target and query source run.
- Metric: question accuracy; also log-perplexity vs the original cache and GPU-seconds per compaction (compaction time + query-generation time; rule 7). n = 357 questions per cell on the selection split (20 articles).
- Deterministic apart from sampling inside self-study generation; one run per cell. The 72-cell plan is cost-gated, see below; unrun cells are reported as not run.
- Pass (R0): the paper's QuALITY numbers (Fig. 4, Qwen3-4B, read approximately from the plot) are compared next to ours for the held-out confirmation of the chosen config; AM beats truncate and random-keys on the pooled comparison below.

## Primary comparison (one, pre-registered)

- Best online candidate vs ceiling, on the held-out split, pooled over targets 0.05 and 0.10, uniform-vs-per-head budget as chosen.
- Online candidate = a cell whose query source is context-prefill or repeat (queries exist the moment the segment has been prefilled; no extra generation). On-policy and key selection are free factors. Best online candidate = highest mean accuracy over the two targets on the selection split; ties within 1 question go to the lower GPU-seconds per compaction.
- Ceiling = ss-plus-repeat + OMP (`omp_nnls0_-inf_7_drop-7_lsq_on-policy`) + optimized per-head budgets.
- Hypothesis: best online candidate is within 3 accuracy points of the ceiling.
- Pass: lower 95% paired bound of (online - ceiling) >= -3.0 points, pooled over targets (n = 1729 questions x 2 targets, 95 articles). Per-target results reported too, not used for pass/fail. Report the cost ratio next to the accuracy.

## Finding to test properly: AM vs random keys with the same fit

- First-pass observation (seen, exploratory): AM-HighestAttnKeys beat random-subset keys by 1.1 and 4.4 points (n=181, 10 articles, ss-plus-repeat) - inside noise.
- Test: `highest_attn_keys_rms_nnls2_-3_3_lsq_on-policy` vs `random_subset_keys_nnls2_-3_3_lsq_on-policy` (config `key-selection`, both unmodified, both include the same NNLS beta and lsq value fit), query config `repeat`, uniform budget, targets 0.05 and 0.10, `truncate_nnls2_-3_3_lsq_on-policy` and `original` in the same run. Run on all 115 articles (2086 questions): the contenders are fixed in advance, so no selection leakage. Held-out-only (1729 q) is reported as a sensitivity check.
- Power: paired question-level test, assumed discordant fraction 0.2 and a design effect 1.4 from article clustering gives SE ~ 1.2 points at n=2086; a 3-point true difference is detected with ~80% power (minimum detectable ~3.3 points). The dataset has no more questions, so n=2086 is the maximum; with the 1729 held-out only the minimum detectable difference is ~3.6.
- Hypothesis H1: AM beats random keys by >= 3 points. Decision rule: CI lower bound > 0 -> AM better (report size); CI within +-3 -> equivalent at this resolution; otherwise inconclusive, and the report says so. Same for AM vs truncate.

## Rung R0s: stale queries

- Hypothesis: fitting with only the queries available when the fit starts (context-prefill / repeat-prefill) loses <= 3 points vs fitting with all queries (ss-plus-repeat), same key selection, same budget, same target.
- Test (a): from the selection matrix, paired cells that differ only in query source. Test (b): on held-out, the chosen online candidate's key selection/budget with `repeat` vs `ss-plus-repeat` (the ceiling's queries), both targets, 1729 q x 2.
- Pass: lower 95% paired bound of (stale - all) >= -3.0 points on (b) pooled over targets. Decides whether non-pausing compaction is safe. Note: this definition equates "available at fit start" with prefill-derived queries; queries from decoding after the segment is sealed are what ss-plus-repeat adds.

## Cost gate

- Pilot timing (selection articles 0-1) before the matrix, then Amendment 1 fixes which cells run and in what priority order. Cells run in that order and stop when the cumulative projection reaches the allocation.
- Allocation target: AIME ~$50; QuALITY matrix on selection ~$40; held-out confirmations (online candidate, ceiling, stale test) ~$30; AM-vs-random test ~$25; reserve ~$5. If the running projection exceeds $150, stop and report.

## Deviations

(none yet)
