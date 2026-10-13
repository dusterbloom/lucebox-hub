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

## Rung R1a: Qwen3.8-27B adapter is exact without compaction

Written 2026-10-13 before any run of the adapter on the 27B model. Only tokenizer-length counts of the candidate prompts were taken (CPU, no model weights) to fix the 8K+ prompts. Adapter code: `../../modal/am_qwen35/` (patches on top of `adamzweiger/compaction` @ 2020f6d3). Cost cap for the whole R1a task: $15.

- Hypothesis: with a `CompactedPrefixCache` that holds the uncompacted prefix (C1/C2 = the original K/V of the 16 full-attention layers, beta = 0, DeltaNet conv/recurrent state carried through as-is), the adapter reproduces HF transformers exactly: same greedy tokens, same first-step logits.
- Model: `Qwen/Qwen3.8-27B` BF16, `Qwen3_5ForConditionalGeneration`, transformers 5.19.0, torch 2.8.0, one H200 (H100 80GB if unavailable). The same loaded weights serve reference and adapter in one process (reference runs first; the adapter classes are swapped in afterwards).
- Prompts (all through the HF chat template, `add_generation_prompt=True, enable_thinking=False`, one user turn; article text = QuALITY.v1.0.1 dev article as shipped in the repo, `article.strip()`, id = QuALITY article_id):
  - P1 (short): "Explain in three paragraphs why the sky is blue and why sunsets are red."
  - P2 (code): "Write a Python function that merges overlapping intervals, with a docstring and three unit tests, then explain its time complexity."
  - P3 (2.3K tokens, measured 2281): article 20041, then "\n\nQuestion: Summarize the story in five sentences."
  - P4 (8.3K tokens, measured 8277): article 51350, then "\n\nQuestion: Summarize the story in five sentences."
  - P5 (16.6K tokens, measured 16581): article 51350, "\n\n", article 22876, then "\n\nQuestion: Summarize the two stories in five sentences."
  P4 and P5 are 8K+; every full-attention layer attends over thousands of cached tokens there.
- Split: the tokenised prompt is cut after `len(ids) - 32` tokens. The first part is the context (prefilled by unmodified HF into a `DynamicCache`), the last 32 tokens are the suffix (question and assistant header).
- Arms, all greedy (argmax), 256 new tokens, EOS not treated as a stop (the 256 tokens are compared one for one):
  - R_split (gating reference): unmodified HF; context prefill, then suffix, then decode, continuing the same `DynamicCache`.
  - R_whole (informational): unmodified HF, whole prompt in one prefill, then decode.
  - ADAPTER: context prefilled by unmodified HF, the `DynamicCache` converted to a `CompactedPrefixCache` (C1/C2 = the cached K/V, beta = 0, linear-attention state moved over unchanged), then suffix and decode through the adapter classes.
- Two attention backends, same arms in each:
  - `eager` (gating). With beta = 0 the adapter's mask differs from HF's only by `+ 0.0`, so the result must be bit-identical; this isolates the adapter logic (positions, mask, cache carry-over, gate, DeltaNet state) from kernel dispatch.
  - `sdpa` (production backend of the repo; reported, not gating). HF passes mask=None for single-token decode (SDPA GQA path), while the adapter always passes an explicit additive mask because beta lives there, so different kernels may be selected and differ by a bf16 ulp. Any token divergence under `sdpa` is reported with the top-1/top-2 logit margin at the divergence step; a margin below one bf16 ulp of the logit is classed as kernel noise, anything else as an adapter fault.
- Metrics: (1) greedy tokens ADAPTER vs R_split, count of prompts with all 256 tokens identical; (2) max over vocab of |logit_ADAPTER - logit_R_split| (BF16 logits cast to float32) at the first step, i.e. the logits at the last prompt position; (3) the same max over all 256 steps (informational); (4) R_whole vs R_split tokens identical count (informational: the noise floor of re-chunking the prefill, not an adapter quantity).
- Pass threshold (`eager`): tokens identical on 5/5 prompts and first-step max |logit diff| <= 1e-2 on every prompt. Why 1e-2: BF16 has 8 bits of mantissa, so one ulp of a logit of magnitude 2-4 is 0.0156 and of magnitude 16-32 is 0.125. A threshold of 1e-2 admits only bit-identical logits for every logit of magnitude >= 2 (and sub-ulp differences below that); it is that strict on purpose because the adapter has to be a no-op at beta = 0. A genuine fault (wrong RoPE offset, mask, dropped DeltaNet state, changed gate) moves logits by >= 0.1 and flips tokens within a few steps.
- Failure handling: a fail is reported as a fail with the numbers; the threshold is not relaxed afterwards. Fixes to the adapter are allowed (they are bugs); each re-run is logged and counted in the cost.

Smoke test (not pass/fail, same task): one compaction at target 0.1 of the article tokens on QuALITY dev article loader index 0 with `AM-HighestAttnKeys` (config as in the repo's `examples/qa_demo.py`: rms scores, nnls_iters 2, bounds 0.05/20, c2_method lsq, uniform budget because no head-budget file exists for this model), query config `repeat`, then the first 3 questions of the article answered from the compacted cache, greedy, max 256 new tokens, thinking off. Reported: answers, correctness against the gold label, any NaN or crash.
