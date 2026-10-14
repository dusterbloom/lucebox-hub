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

(see Amendment 1; no outcome data of the matrix had been seen when it was written. The pilot only gave timings and 33-question accuracies on 2 articles, not used for any choice.)

## Amendment 1 (written after the timing pilot and the AIME Table 9 block, before any matrix, held-out or vs-random run)

Measured (H100, selection articles 0-1, target 0.10, uniform budget, per article; generating ~17 answers is ~75 s per cell regardless of method):

| Key selection | compaction time | all-in per article-cell (prefill/repeat queries) |
|---|---|---|
| HighestAttnKeys | ~12 s | ~1.6 min (~$0.10) |
| OMP-fast (k4 int2) | ~250 s | ~5.6 min (~$0.37) |
| OMP | ~1450 s | ~25 min (~$1.65) |

Self-study queries add roughly 1.5-2.5 min per article-cell (first pass: ~2.5 min per article-cell all-in for HA). The full 72-cell matrix on 20 articles projects to ~$320, the held-out ceiling with OMP on 95 articles to ~$100+, the vs-random test on 115 articles at both targets to ~$50. Total > $450 against the $150 cap, so the cost gate trips. Spent before this amendment: first pass ~$43 (previous task), AIME Table 9 $56, pilot ~$25 (the self-study pilot arm was still finishing). Remaining for this launch: ~$65.

Reductions (all from cost only, decided before seeing any matrix outcome):
1. Target 0.10 only. Target 0.05 is not run.
2. Selection matrix (articles 0-19, 357 q): 6 cells instead of 72, all on-policy except where stated, all `best` config methods:
   S1 HA repeat per-head; S2 HA repeat uniform; S3 HA context-prefill per-head; S4 OMP-fast repeat per-head; S5 HA self-study (ss-plus-repeat) per-head; S6 OMP-fast self-study per-head.
   Not run: plain OMP in every cell; non-on-policy variants; OMP-fast context-prefill; uniform-budget variants other than S2; target 0.05.
3. Ceiling = S6 key selection (OMP-fast `omp_nnls0_-inf_7_drop-7_lsq_k4int2_on-policy`) + ss-plus-repeat + per-head budgets, replacing plain OMP in the definition above (plain OMP costs ~5x more). The ceiling is therefore possibly below the paper's ceiling.
4. Held-out for the online-vs-ceiling and stale-query comparisons: articles 20-49 (30 articles, ~540 q) instead of 20-114. Minimum detectable difference at that n is ~6 points, so a verdict is given only when the CI is inside the +-3 margin; otherwise "inconclusive at this n".
5. AM vs random keys stays at all 115 articles (2086 q) but target 0.10 only, repeat queries, uniform budget, `highest_attn_keys_rms_nnls2_-3_3_lsq_on-policy` vs `random_subset_keys_nnls2_-3_3_lsq_on-policy` plus `original`; `truncate` not run (first pass: 9-19 points worse at n=181). Minimum detectable ~3.3 points as pre-registered.
6. `--compute-stats 0` for all QuALITY runs (skips reconstruction statistics; accuracy and perplexity unaffected).
7. Selection rule unchanged: best online cell among S1-S4 by selection accuracy at 0.10; ties within 1 question go to lower compaction time.
8. The vs-random run is split into 5 article ranges (start-article/n-articles) and merged for analysis; same configs in every shard.
9. AIME Table 9 block cost $56 vs $50 projected; design unchanged.

## Amendment 2 (written after S1-S4 selection results, before any held-out result; the user raised the cap to $190)

- Winner by the Amendment-1 rule: S1 (HighestAttnKeys, repeat queries, per-head budget, on-policy), see `R0_quality_selection.md` (S1 258/357, S4 259/357, tie -> lower cost).
- Selection OMP-fast cells cost ~714 s per article (more than the pilot suggested), so the ceiling arm (S6) on 30 held-out articles would cost ~$30+ and the projection would pass $190. Held-out is therefore: winner S1 + original on articles 20-49 (30 articles, all 3 shards); ceiling S6 (OMP-fast + ss-plus-repeat + per-head) on articles 20-29 only (10 articles). The R0s stale arm on held-out is dropped (the instructed first cut); R0s is answered on the selection split by S5 (HA ss-plus-repeat, per-head) vs S1 (same, repeat queries), n=357.
- Consequence: the primary comparison (online S1 vs ceiling S6) runs on 10 held-out articles (~180 q); the minimum detectable difference is ~10 points, so the verdict is "within margin" only if the paired CI lies inside +-3, otherwise "inconclusive at this n". The selection-split S1 vs S6 comparison (n=357) is reported next to it as an exploratory (selection-contaminated) number.

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

### R1a deviations (2026-10-13, found at the first GPU run, before any result)
1. P1 has 29 tokens and P2 36, shorter than or about as long as the 32-token suffix, so the "cut at len - 32" rule leaves P1 without context. Rule applied: suffix = min(32, len // 2) (P1: 15 + 14, P2: 18 + 18; P3-P5 unchanged at 32). The first attempt crashed on the empty context and produced no result.
2. The smoke test uses `algorithm=global_highest_attention_keys` (the AM-HighestAttnKeys registry method), not the per-layer-head variant of `examples/qa_demo.py`, with `query_config=repeat`, `rms` scores, `nnls_iters=2`, `lsq` values.

Smoke test (not pass/fail, same task): one compaction at target 0.1 of the article tokens on QuALITY dev article loader index 0 with `AM-HighestAttnKeys` (config as in the repo's `examples/qa_demo.py`: rms scores, nnls_iters 2, bounds 0.05/20, c2_method lsq, uniform budget because no head-budget file exists for this model), query config `repeat`, then the first 3 questions of the article answered from the compacted cache, greedy, max 256 new tokens, thinking off. Reported: answers, correctness against the gold label, any NaN or crash.

## Rung R1b/R1c: Qwen3.8-27B, four configurations and head budgets

Written 2026-10-13 after R1a passed (commit a90ea3bf) and before any R1b or R1c run on the 27B. Only the one smoke compaction of R1a (3 questions, not used for any choice) has been seen. Budget for this block: $270 Modal, separate from the R1a $15 and the R0 $150. H200 at $4.54/h; a running cost total is kept in `R1b.md`; cells run in the priority order below and the block stops when the cumulative actual cost plus the projected cost of the next item exceeds $270.

### Fixed provenance

- Adapter: `modal/am_qwen35/` at a90ea3bf (patch on `adamzweiger/compaction` @ 2020f6d3 + overlay), extended only where a method needs hybrid handling (`per_layer_head*.py`, head-budget code). Every extension keeps R1a's exactness test passing (eager, 5 prompts) and adds a CPU unit test. Model `Qwen/Qwen3.8-27B` BF16, transformers 5.19.0, torch 2.8.0, sdpa attention for the QuALITY runs (R1a: eager and sdpa agree at beta = 0 up to kernel noise; sdpa is the production kernel), one H200 per job, up to 3 jobs in parallel.
- QuALITY: same splits as R0 (loader indices 0-19 selection, 357 q; 20-49 held-out as in Amendment 1, about 540 q). Prompts via the repo's `format_context` / `format_question` / `parse_model_choice`, non-thinking (`enable_thinking=False`), greedy, at most 256 new tokens. Query configs as shipped (`repeat`, `context-prefill`, `ss-plus-repeat`); self-study sampling seed 1.
- `original` = the same 27B uncompacted through the adapter (bit-identical to HF by R1a): the model's own baseline, computed once per article and paired with every cell. (The repo's own `original` method uses vLLM; the 27B has no vLLM here, and within-model pairing makes this immaterial.)
- Compaction targets: 0.10 and 0.05 of the article tokens (non-article tokens are kept, as in the repo).

### R1(c) first: head-sensitivity curves and the per-head budget file

- Procedure: the paper's `head_budget_optimization.run` (per-head influence curves of the log-perplexity of reference answers against the head's budget, then the solver), adapted to the 16 full-attention layers x 4 KV heads = 64 heads; DeltaNet layers have no budget. Method `highest_attn_keys_rms_nnls2_-3_3_lsq`, baseline schedule uniform, query config `repeat` (the paper uses `ss-plus-repeat`; `repeat` is used because self-study on the 27B per fit article multiplies the cost; the budget file is then used for every per-head cell), target ratio 0.05, `--max-ratio 0.5`, 21 evaluation points (paper: 51), solve ratios 0.05 and 0.10, fit articles = loader indices 90-94 (5 articles; none in 0-49).
- Deliverables on volume `kvflash-am-results`: `head_curves.json` and `Qwen3.8-27B/optimized_agnostic.json` (per-head budgets, same schema as `head_budget_optimization/head_budgets/Qwen3-4B/optimized_agnostic.json`). All per-head cells use it with `--max-ratio-per-head 0.75`.
- Check (reported, not a gate): on loader indices 95-99 (not in the fit), mean log-perplexity of reference answers after compaction at 0.10 with the per-head file vs uniform budgets, same method and queries, paired over articles. Expectation: per-head <= uniform.

### R1(b): configurations (`best` algorithm config; all `_on-policy`)

| id | role | query source | key selection | budget | R0 4B reference cell |
|---|---|---|---|---|---|
| F | floor | context-prefill | `highest_attn_keys_rms_nnls2_-3_3_lsq_on-policy` | per-head | S3 |
| O1 | online candidate 1 | repeat | same HA | per-head | S1 |
| O2 | online candidate 2 | repeat | `omp_nnls0_-inf_7_drop-7_lsq_k4int2_on-policy` (OMP-fast) | per-head | S4 |
| C | ceiling | ss-plus-repeat | OMP-fast | per-head | S6 |

Why these two online candidates, named before data: both use queries that exist the moment a segment has been prefilled (no extra generation, hence no pause, which is the property E needs). O1 is the cheapest per-head option. O2 pays about 20x more compaction time for OMP-fast keys on the same queries; it measures whether key selection matters on the hybrid at equal query information, and it shares the ceiling's key selection, so C - O2 isolates the query source. F differs from O1 only in the query source (the context as queries, nothing generated), so O1 - F isolates the query source for HA. The four cells coincide with R0 cells S3, S1, S4, S6 of Amendment 1, which is what gives a paired 4B reference at 0.10 on the same 357 questions. The ceiling is OMP-fast, not plain OMP, as in Amendment 1 item 3; plain OMP is not run.

Controls (27B, selection split, both targets, `repeat`, uniform budget, on-policy NNLS/lsq fit): `highest_attn_keys_rms_nnls2_-3_3_lsq_on-policy` ("U-AM"), `random_subset_keys_nnls2_-3_3_lsq_on-policy`, `truncate_nnls2_-3_3_lsq_on-policy`.

### R1(b): statistics and pass criteria

- Unit: question; paired cluster bootstrap over articles (10,000 resamples, 95% percentile); McNemar exact as a secondary check. n questions and n articles are reported with every number.
- Degradation of configuration X on model M: D_M(X) = acc_M(original) - acc_M(X), in points, per question.
- PASS per configuration (primary: selection split, target 0.10, n = 357 q, 20 articles): Delta(X) = D_27B(X) - D_4B(X), paired per question against the matching R0 4B cell in the table, same questions. X passes if the upper 95% bound of Delta(X) <= +3.0 points (the 27B degrades at most 3 points more than the 4B). X fails if the lower 95% bound of Delta(X) > 0 and the point estimate > +3.0. Otherwise the verdict is "inconclusive at this n" and is not reported as a pass. The paired SE of Delta at n = 357 is expected near 2.5-3 points, so a pass needs the 27B to degrade about as little as, or less than, the 4B; this is accepted.
- R1b overall PASS (the go/no-go for E): C passes and at least one of O1, O2 passes. F is reported and does not gate. Otherwise R1b is reported as not passed, with the numbers.
- Secondary, not gating: Delta at target 0.05 where a matched 4B cell exists (descriptive otherwise); the held-out (20-49) online-vs-ceiling gap for O1 and O2 vs C by the R0 rule (lower bound of online - ceiling >= -3 points -> within 3 points; a CI wider than the margin -> inconclusive at this n); the 27B degradation with its 95% CI; U-AM vs random and vs truncate (R0 rule: lower bound > 0 -> AM better; inside +-3 -> equivalent; else inconclusive); GPU-seconds per compaction (compaction + query generation) of every cell.
- If an R0 4B reference cell is missing when R1b is analysed, that 4B cell is run once with the unmodified `am_repro.py` and its cost is counted here.

### n, seeds, order, cost cutting

- n: 20 articles / 357 q (selection) and 30 articles / about 540 q (held-out) per cell. One run per cell, greedy decoding; the only randomness is self-study generation (seed 1). No cell is repeated.
- Priority order (cells launched in this order, after one cell has been timed):
  1. R1(c) budget file (5 fit articles; if the projection exceeds $45, drop to 3 fit articles and 11 evaluation points, stated in the report).
  2. Selection, 0.10: `original`, O1, F, O2 (articles 0-19).
  3. Selection, 0.10: C.
  4. Selection, 0.10 controls: U-AM, random, truncate.
  5. Held-out (20-49), 0.10: `original`, O1, O2, C.
  6. Selection, 0.05: `original`, O1, O2, C, F, then the controls at 0.05.
  7. Held-out, 0.05: O1, O2, C.
  8. R1(c) validation on articles 95-99.
  9. Missing 4B reference cells.
  When the projection of the items not yet run exceeds the remaining budget, items are removed from the bottom of this list (9 first), never reordered, and the report lists every cell not run. An item is run completely or not at all, except that it may be split by article range with an identical configuration.
- Timing: before step 2, one cell (O1 on 2 selection articles) is timed and the per-cell cost is projected from it for each configuration; projection and actual are both logged in `R1b.md`.

### R1b Amendment 1 (2026-10-13, written while building the driver, before any R1b or R1c run)

1. Protocol finding. Reading `evaluation/qa_evaluator.py` shows that the R0 QuALITY cells (4B) answer in thinking mode (`format_question` default `enable_thinking=True`), sampled (temperature/top-p/top-k from `get_generation_params`), `max_new_tokens=2048`, with `original` served by vLLM. The 27B protocol above is non-thinking, greedy, 256 tokens, HF for every arm. Delta against R0 4B cells would therefore compare degradations measured under different protocols. Decision: the primary 4B reference for the PASS rule becomes matched-protocol 4B cells, run through the same driver (`modal/am_qwen35/r1b_core.py`, same prompts, split, queries, key selection, budgets, non-thinking, greedy, 256 tokens; Qwen3-4B with the repo's own Qwen3 modeling in the transformers 4.57.1 image, shipped `Qwen3-4B/optimized_agnostic.json` with `--max-ratio-per-head 0.75`). The R0 4B cells (S1, S3, S4, S6) are reported next to it as a secondary, protocol-different reference. The matched 4B cells enter the priority list as item 3b, right after the ceiling C (cost expected below one third of the matching 27B cells because the 4B has no hybrid layers and 36x8 heads but fast kernels; measured and logged).
2. R1(c) reference answers. The paper computes influence curves from thinking-mode reference answers of up to 4096 tokens. At 27B each curve point costs one perplexity forward per question, so that is more than 50x the cost at the budget given. R1(c) uses reference answers from the uncompacted 27B, greedy, non-thinking, at most 128 tokens (the repo's question prompt asks to think very briefly and then give the letter), and the perplexity prompt is formatted non-thinking to match. Everything else in R1(c) is unchanged.
3. Self-study generation (config C only): the repo needs vLLM for the generation step of self-study. The 27B hybrid is not served by the vLLM pinned in the repo, so generation uses a small HF `generate` shim (`overlay/models/qwen3_5/hf_generate.py`) with the same sampling parameters as the repo (temperature 0.7, top-p 0.8, top-k 20, non-thinking; temperature 0.6, top-p 0.95, top-k 20 with thinking) and the same prompts; query extraction is unchanged. Sampling seed 1.
4. Per-head budgets of the 4B shipped file are keyed `L{layer}H{head}` over 36 layers x 8 KV heads; the 27B file is keyed over the full-model layer index (the 16 full-attention layers are 3, 7, ..., 63) x 4 KV heads. The repo's own loader maps missing keys to None, which the on-policy compactor treats as unallocated, but only for layers it compacts; DeltaNet layers are never compacted.

### R1b Amendment 2 (2026-10-14, written after the R1b results, including the held-out cells, were read; before any held-out run of F)

1. Status. Cell F was run on the selection split only. This cell is added AFTER the R1b results were read, so it is exploratory and post hoc. It gates nothing and does not change any verdict already reported.
2. Cell. F (context-prefill queries, `highest_attn_keys_rms_nnls2_-3_3_lsq_on-policy`, per-head budgets from `R1b/r1c/optimized_agnostic.json`, `--max-ratio-per-head 0.75`; configuration unchanged from the F row above), targets 0.10 and 0.05, held-out split (loader indices 20-49, 30 articles, 537 questions). No other cell is run or re-run.
3. Comparisons, all paired over the same 537 questions and 30 articles, with the same cluster bootstrap over articles (10,000 resamples, seed 0, 95% percentile) and McNemar exact as secondary: F - C, F - O2, F - O1, and F vs orig (degradation D = acc(orig) - acc(F)). Same table layout as the existing held-out section.
4. Rule, unchanged from R1b: an online cell is "within 3 points" of the reference when the lower 95% bound of (cell - reference) is >= -3 points; a CI wider than the margin is "inconclusive at this n". No new threshold.
5. GPU parity. Earlier held-out cells ran on a mix of H100 and H200 (from the per-job cost records on `kvflash-am-results`): at 0.10 the `orig` cells ran on H200 for articles 20-29, H100 for 30-39, H200 for 40-49; at 0.05 all on H200. Borderline greedy answers can flip across GPU types, so F is run on the same GPU type as the `orig` cell of the same article range and target (H100 pinned without upgrade to H200, `H100!`). This parity rule overrides the cheapest-GPU rule because the comparison is paired against existing rows. The GPU type per job is recorded in the cost record and in R1b.md.
6. Cost: projected from the F selection timings (11-17 s per article plus prefill, plus about 2 min model load per job); under $5 for four jobs, cap $15. Results go to new directories on `kvflash-am-results`; nothing is deleted.

### R1b Amendment 3 (2026-10-14, written after the R1b results and Amendment 2 results were read; before any run of F-UNI)

1. Status. Post hoc and exploratory. It gates nothing and does not change any verdict already reported.
2. Question. How much do per-head budgets add to F? F-UNI is F (context-prefill queries, `highest_attn_keys_rms_nnls2_-3_3_lsq_on-policy`) with a uniform budget, the same for every head, instead of `R1b/r1c/optimized_agnostic.json`. Everything else is identical to F.
3. Runs. F-UNI at targets 0.10 and 0.05 on the selection split (loader indices 0-19, 357 questions) and the held-out split (loader indices 20-49, 537 questions). No other cell is run or re-run. GPU parity as in Amendment 2 item 5: each article range runs on the GPU type of the `orig` cell of that range and target (`--start` is relative to the split start).
4. Comparison. F - F-UNI, paired over the same questions and articles, with the same cluster bootstrap over articles (10,000 resamples, seed 0, 95% percentile) and McNemar exact as secondary, same rule as Amendment 2 item 4 (within 3 points when the lower 95% bound >= -3). Also reported on the selection split: F-UNI - U-AM (query source effect under uniform budgets). Degradation vs orig is reported for F-UNI.
5. Decision use. If F - F-UNI is within 3 points on held-out at 0.10, the engine can use uniform budgets (one 640-token segment per 64-slot pool block, no per-head storage).
6. Cost: short jobs (F cells took about 4 min per 10 articles), projected under $3, cap $5. Results go to a new directory `r1b_funi` on `kvflash-am-results`; nothing is deleted.
