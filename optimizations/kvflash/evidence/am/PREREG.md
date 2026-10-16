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

## Rung R2: multi-round compaction of F on Qwen3.8-27B

Written 2026-10-14 after R1b Amendments 2 and 3 and before any R2 run on a GPU. What has been seen: the existing single-round F rows (selection 309/357 against orig 320/357, held-out 439/537 against 460/537), the repo source read for this design, 22 CPU unit tests passing on the tiny hybrid and one CPU budget computation (item 6). No R2 accuracy has been seen. F is itself post hoc (R1b Amendment 2), so R2 inherits that status: it gates nothing already reported.

### Question and what the repo's online loop does

Does an F-compacted answer stay accurate after several rounds of compaction? Two readings of "rounds", both measured: segment-wise compaction that never revisits old tokens (our engine's design), and recompaction of the whole cache every round (the repo's online loop).

The repo's online loop (the AIME Table 9 path, `--max-compactions`) re-compacts the whole cache each round, not only the new tokens:
- `evaluation/run_reasoning_evaluation.py:552` `indices_to_compact = range(0, compactable_len)` with `compactable_len = total_seq_len - protected_tokens` (:537-539; `total_seq_len` is the longest layer of the current compacted cache, :191-204, :538; `protected_tokens` defaults to 20, :367). The compacted prefix and everything generated since are one cache and are compacted together. Only the first round does an HF prefill (:508-520); later rounds reuse the compacted cache (:521-524). Target per round: `int(compactable_len * target_size)` (:545-546), so the cache is brought to a fixed fraction of its CURRENT length every round, and old tokens are compacted again every round.
- The previous beta is carried in the loop's own method. The loop's method is `AM-HighestAttnKeys-basic` (`evaluation/configs/algorithms/default.py:12`, `run_reasoning_evaluation.py:924`), which the registry builds as `PerLayerHeadCompaction` (no `on_policy` flag, `compaction/compaction_methods/registry.py:146,159-169`). That class accepts `(K, beta, V)` layers (`per_layer_head.py:138-143,192`), passes `attention_bias` into the algorithm (:459-501, :525) and keeps the old beta of kept tokens (:568-576).
- F is the on-policy variant (`..._on-policy` sets `on_policy: True`, registry :146,159), `PerLayerHeadOnPolicyCompaction`. It reads only K and V (`per_layer_head_on_policy.py:316-317`), calls the algorithm with `(K, V, queries, target)` only (:476-478) and sets beta of kept tokens to zero (:512-518). So on re-compaction F drops the previous beta. The loop cannot even call F verbatim: F's `compact_kv_cache` has no `full_query_extraction` argument (`per_layer_head_on_policy.py:70-85`; the loop passes it at `run_reasoning_evaluation.py:566`).
- A beta-carrying F-RE would need the loop's method on the hybrid. `per_layer_head.py` is not among the files patched by 0001-0003 (patch headers: `global_highest_attention_keys.py`, `per_layer_head_on_policy.py`, `query_generation/*`, `models/*`, `evaluation/utils.py`, `head_budget_optimization/influence.py`) and indexes queries by absolute layer (`per_layer_head.py:505` `queries[0, layer_idx, ...]`; the hybrid has 16 query rows, not 64). That needs code changes, so per the instruction "no patching" there is NO F-RE-beta arm. F-RE below is the loop's shape (whole cache, every round, fixed fraction of current length) executed with F's method, which drops beta. This is stated in every result table.

### Context and questions

Context = article A (loader index in the probe set) followed by k distractor QuALITY articles, k in {1, 2, 4, 8}. Segment 0 is exactly the R1b context (`format_context`: system turn plus user turn with A). Each distractor is one more user turn (`<|im_start|>user\n{article}<|im_end|>\n`, the part of `format_context` after the system turn). Distractors: loader indices 50-89 (outside selection 0-19, held-out 20-49 and R1c fit/val 90-99), 8 per probe article drawn by `random.Random("r2-0-{probe_index}").sample(range(50, 90), 8)` and nested (the k-distractor context uses the first k). The questions of A (all of them, same set as R1b) are asked after all k distractors, with the R1b prompt (`format_question`, non-thinking), greedy, at most 256 new tokens, same parser and scoring as R1b. Positions of the question tokens continue from the full logical length.

### Arms (all at 0.10; k = 0 reuses the existing rows and is not rerun)

- FULL: one-shot prefill of the whole context, no compaction, answered through the adapter with beta = 0 (the same path as the `orig` rows). k = 0 is the existing `orig` row.
- F-SEG: each article is compacted once when it arrives, to int(0.10 x its article tokens) slots, non-article tokens (turn tags) kept; the earlier compacted articles, with their betas, are the prefix; never recompacted. A's step is the R1b call (`Session.compact("F", ...)`), so A alone reproduces the k = 0 F rows exactly. Later segments are prefilled behind the compacted prefix (RoPE continues from the logical length), their article tokens compacted by the unchanged repo compactor in the F configuration, and the result is spliced behind the prefix.
- F-BLK (the engine's granularity): the token stream (all segments, tags included) is cut by position into 640-token blocks. A block is sealed when 640 tokens exist; it is compacted ONCE to 64 slots (0.10) with the earlier compacted blocks as prefix. All tokens of a block are compacted (the engine does not know article boundaries). The open tail (< 640 tokens) stays uncompacted at question time. Per-head budgets: the repo's own formula (`per_layer_head_on_policy.py:241-266`) with the same proportions file and cap 0.75, applied to a 64-slot block. This needs no new allocation logic, so no uniform-budget fallback is used.
- F-RE: step 0 is the R1b F call on A. For each new article: prefill it behind the current compacted cache, then recompact the WHOLE cache to int(0.10 x L) slots, where L is the longest layer (as the loop's `total_seq_len`). A is compacted k + 1 times. Deviations from the loop, forced by F with per-head budgets: (a) `protected_tokens = 0` (indices = all L slots) because per-head padding puts -inf padding, not the last tokens, in the final 20 slots; (b) layers are padded to a common length L with zero K/V and beta = -inf, because the compactor takes one index range for all layers while F's budgets make layer lengths differ (the repo already pads heads within a layer the same way). Queries: the repo's unmodified context-prefill over the whole uncompacted text so far (the loop's `full_query_extraction` behaviour), subsampled by the repo to its 100,000-per-KV-head cap. Previous beta dropped (see above). The cache size is a fixed point of s = 0.10 (s + n), about 0.11 of one article regardless of k: F-RE spends one article's budget on k + 1 articles, F-SEG and F-BLK spend 0.10 per article. Both slot counts (tensor and effective, non-padding) are reported.

Queries for F-SEG and F-BLK steps are the queries the segment's own prefill computes behind the compacted prefix (captured by forward pre-hooks on the 16 full-attention layers, post-norm, post-RoPE at true positions), handed to the unmodified repo compactor by replacing `ContextPrefillQueryGenerator._extract_query_vectors_from_prefill` at run time (`r2_core.fixed_queries`; no repo file is edited, the image is the R1b image). This is the engine's behaviour: queries are available from the segment's own prefill. It differs from F-RE, whose queries come from an uncompacted forward as the repo's loop does; this is part of the arm definitions.

Hybrid model: the Gated DeltaNet conv and recurrent state is never compacted. `CompactedPrefixCache` carries it through every segment (`models/cache.py`, `linear_layer_indices`), and each segment is prefilled from the state the previous one left. CPU unit tests on the tiny hybrid (`tests/test_multiseg.py`, written before any GPU run, 22 passed in total with the 13 existing): a 2-segment and a 3-segment UNCOMPACTED prefix built segment by segment through the adapter equals a one-shot forward (K and V of the full-attention layers to 1e-4, conv and recurrent state to 1e-5 / 1e-4, logits of a following question chunk to 1e-3, all fp32), `extend` does not mutate its prefix, the queries captured behind a prefix equal the one-shot queries (1e-4), and an F-SEG step with an identity compactor reproduces the uncompacted cache and hands the repo compactor exactly the one-shot queries of the segment.

### Probe set, metrics, statistics

- Probe set: loader indices 0-19 (357 questions), so k = 0 matches the existing rows. If cost forces it, the fixed subset is loader indices 0-9 (declared now, chosen without looking at any result); the subset is used for every arm and k, and said so in the result.
- Per arm and k: correct counts out of N; for FULL(k) too.
- Paired differences, per question: D_arm(k) = FULL(k) - arm(k) (excess degradation at the same context), and arm(k) - arm(0) for the compacting arms. FULL(k) - FULL(0) is reported as the effect of distractors alone. k = 0 rows come from earlier runs on a mix of H100 and H200 (R1b Amendment 2 item 5); R2 arms run on one GPU type, so a cross-run comparison to k = 0 carries a small greedy-flip noise that FULL(k) - orig(0) shows directly.
- Cluster bootstrap over articles, 10,000 resamples, seed 0, 95% percentile; McNemar exact as secondary (`r1b_stats.py`, unchanged).
- Pre-registered question 1 (primary): F-SEG stays within 3 points of single-round F at k = 8: the lower 95% bound of F-SEG(8) - F(0) >= -3 points. A CI wider than the margin is reported as inconclusive at this n. Same statement for F-SEG(8) - FULL(8) is reported next to it.
- Pre-registered question 2: does F-RE degrade monotonically with k? Defined on the excess degradation D_RE(k) = FULL(k) - F-RE(k) over k = 1, 2, 4, 8: monotone when D_RE(k) is non-decreasing across consecutive k in point estimate; the number of violations and the CI of D_RE(8) - D_RE(1) are reported. The same is reported for the raw accuracy F-RE(k) - F-RE(0).
- F-BLK is exploratory: F-BLK(k) - F-SEG(k) and the same within-3 rule against F(0), no gate.
### GPU and parity

Cheapest GPU that holds BF16 27B (about 54 GB), the context (about 60K tokens at k = 8, 16 full-attention layers x 4 KV heads x 256 x 2 x 2 bytes = 131 KB per token, 7.9 GB) and the F-RE full-text query tensor (about 12 GB before subsampling): A100-80GB first. H100 on Modal is also 80 GB, so an OOM on the A100 is not fixed by the H100; the next step is H200 (141 GB). All arms of one run use the same GPU type, recorded in each row (`gpu`) and in the cost record. The pilot's type is the type the full-run projection is quoted for.

### Pilot and cost gate

Pilot (not inference): probe articles with loader indices 0 and 1, k in {1, 8}, arms FULL, F-SEG, F-BLK, F-RE, all questions of those articles. It reports wall time per arm and k (prefill, compaction, answering), peak GPU memory, correctness so far (no conclusions at this n), and a cost projection for the full design on 20 and on 10 articles. Cap $10. The full run is NOT started by this block. Results go to `r2_pilot` on `kvflash-am-results`, rows in `evidence/am/R2_multiround_pilot/`; nothing is deleted.

### Design check before any GPU run (budget arithmetic, no accuracy)

Per-head slots for one 64-slot block over a 640-token block with the R1b budget file and cap 0.75 (CPU job `r2.py::budgets`): of the 64 compacted heads, 0 get 0 slots, 0 get 1 slot, 12 get 2 slots, the largest gets 354, mean 63.6 (truncation by `int()` drops 23 of 4096 slots, 0.6%). Over a 6626-token article at 662 slots: no head under 3 slots, largest 3670, mean 661.6 (drops 23 of 42368). Layer tensors are padded to their largest head with beta = -inf, so the tensor slot count per block exceeds the effective count; both are reported.

### R2 Amendment 1 (2026-10-14, written after pilot probe 0 and the first F-RE row of pilot probe 1 were read, before any other R2 run)

1. What was seen. Pilot probe 0 (14 questions, H200, k in {1, 8}): FULL 10/14 and 10/14; F-SEG 9/14 and 10/14; F-BLK 11/14 and 12/14; F-RE 7/14 at k = 1 and 0/14 at k = 8, where the F-RE answers are token gibberish with finite logits (every answer runs to the 256-token cap). Pilot probe 1 (19 questions): FULL 19/19 at k = 1 and k = 8, F-SEG 19/19 and 19/19, F-BLK 18/19 and 18/19, F-RE 3/19 at k = 1. n is far too small for any accuracy conclusion on FULL, F-SEG or F-BLK; the F-RE collapse is the one finding, and it is a failure of the cache, not a few lost answers. On an A100-80GB the F-RE arm ran out of memory (CUDA OOM in `torch.linalg.lstsq`, `compaction/algorithms/base.py:168`, 76 GB allocated) at some k <= 8; on the H200 its peak at k = 8 is 89 GB, F-SEG's 71.6 GB, FULL's 61.7 GB.
2. Candidate cause, to be tested and not asserted. After an F compaction, about 40% of the tensor slots are padding (probe 0, k = 1: tensor 1843 against 1108 effective for F-RE; F-SEG k = 8: 7699 against 4651), because the repo pads every head to its layer's largest head with zero K/V and beta = -inf (`per_layer_head_on_policy.py:486-497, 572-594`). F's compactor ignores the incoming beta (Rung R2, online-loop findings), so on a re-compaction those zero keys are live keys with logit 0 and value 0. The same pass drops the beta of the real compacted keys.
3. Added arm F-RE-UNI, a labelled diagnostic and not a gate: identical to F-RE (same step order, same query source, same `protected_tokens = 0`, same layer padding to a common length) but with the uniform-budget compactor of R1b Amendment 3 (cell FUNI: context-prefill queries, the F algorithm, one target for every head), so no head is padded and the only remaining difference to the repo's loop is the dropped beta. Its step 0 is the F-UNI call on A. If F-RE-UNI is healthy and F-RE is not, the F-RE collapse is the padding artifact of per-head budgets on a beta-dropping re-compaction; if both collapse, it is the dropped beta. Either reading is reported next to F-RE, and F-RE (per-head, as defined above) stays the arm of pre-registered question 2.
4. GPU. F-RE needs more than 80 GB (A100 OOM; 89 GB peak on H200 at k = 8 with a 46K-token context), so every R2 arm of the pilot and of the full run runs on one H200; A100 and H100 are excluded for R2. The H200 rate is $4.54/h (`r1b.py` `USD_PER_S`), the rate used for the projection. The k = 0 F and orig rows ran on the H100/H200 mix of R1b Amendment 2 item 5.
5. Distractor lengths: the 8 distractors of probe 0 and probe 1 average about 5K tokens, so k = 8 contexts are about 46K tokens, not 60K.
6. Cost: this amendment adds one more diagnostic arm to the pilot and to the full-run projection; the pilot cap of $10 is unchanged.

### R2 Amendment 2 (2026-10-14, written after the whole pilot table was read and before any full-run row; the user approved the full run without F-RE)

1. What was seen: the pilot table only (probes 0-1, 33 questions, H200), correct at k = 1 / k = 8: FULL 29 / 29, F-SEG 28 / 29, F-BLK 29 / 30, F-RE 10 / 0, F-RE-UNI 23 / 0. F-RE-UNI collapses like F-RE, so by Amendment 1 item 3 the collapse goes with the dropped beta on re-compaction, not with per-head padding. This is a pilot reading at n = 33, reported as such.
2. Arms of the full run: FULL, F-SEG, F-BLK. F-RE and F-RE-UNI are not run. Both produce gibberish at k = 8 on both pilot articles, the engine never re-compacts compacted tokens (M8 compacts each sealed block once), and F-RE is the most expensive arm (pilot k = 8: 166 s compaction and 251 s answering per article). Pre-registered question 2 is withdrawn; it is reported as "not run, the pilot shows collapse at k = 8" next to the pilot rows.
3. k in {1, 8}. k = 2 and 4 served only question 2. Question 1 needs k = 8; k = 1 is kept as the midpoint.
4. Probe set: loader indices 0-19 (357 questions), the pilot articles 0-1 included and re-run; pilot rows are not merged into the full run.
5. GPU: one H200 for the whole run (Amendment 1 item 4), one container, articles in order, output `r2_full` on `kvflash-am-results`, entry point `r2.py::pilot` with `--probes 0..19 --ks 1,8 --arms FULL,F-SEG,F-BLK --out r2_full`.
6. Cost: the pilot timings project about 200 s per article for the three arms at k in {1, 8}, about 70 minutes and $5.5 with the model load. Cap $8: the run is stopped if the running cost passes it, and the articles done so far are reported as a prefix of the probe set.
7. Analysis unchanged: question 1 (lower 95% bound of F-SEG(8) - F(0) >= -3 points) with F-SEG(8) - FULL(8) next to it, F-BLK exploratory, `r2_report.py --stats`. Rows go to `evidence/am/R2/`.

## R1(e) on lucebox6 iGPU: segment-local vs global F queries, uniform budgets

Written 2026-10-14 before any R1(e) run. What has been seen: the existing R1b rows (orig, F, F-UNI on Modal H100/H200), the R2 pilot, and the repo source (`r2_core.py` `gen_fblk`). No LOCAL-arm accuracy has been seen.

1. Question. M8 (segment-local compaction in the engine) can only ship if compacting each sealed block from its own prefill queries loses little against compacting the whole article from whole-article queries. Everything else is held equal: same algorithm (`highest_attn_keys_rms_nnls2_-3_3_lsq_on-policy`, cell FUNI), same uniform budget (every head the same), same total budget 0.10 of the article tokens.
2. Arms, all run on this hardware (lucebox6 integrated GPU, Radeon 8060S gfx1151, ROCm torch, HIP_VISIBLE_DEVICES=1, one session, one code copy):
   - ORIG: uncompacted context, R1b `original_triples` path.
   - F-UNI-GLOBAL: R1b cell FUNI at target 0.10 (whole-article context-prefill queries, uniform budget), as in R1b Amendment 3.
   - F-UNI-LOCAL: the formatted context cut into sealed 640-token blocks by stream position; each block compacted once to 64 slots per head (uniform, all 640 block tokens as indices) from its OWN prefill queries computed behind the already compacted prefix (the `gen_fblk` / `segment_step` path of `r2_core.py` with cell FUNI instead of F). The open tail (< 640 tokens) stays exact. DeltaNet state carried through as in R2.
3. Splits and order. Selection split first (loader indices 0-19, 357 questions), held-out second (indices 20-49, 537 questions). Greedy decoding, 256 new tokens, non-thinking, same question prompts as R1b. Model Qwen/Qwen3.8-27B bf16, transformers 5.19.0, SDPA attention, pinned code adamzweiger/compaction @ 2020f6d3 with patches 0001-0003 and the overlay. torch is a ROCm build (not the torch 2.8.0 CUDA build of Modal), so kernels differ.
4. Comparability. These rows are not comparable to the H100/H200 rows of R1b/R2 (different GPU, torch build, kernels). All three arms run on this hardware and only the within-session differences are interpreted. The ORIG row here is the reference for degradation; the Modal ORIG row is reported next to it as a sanity check of the platform, not as a gate.
5. Statistics. LOCAL - GLOBAL, paired over the same questions and articles, cluster bootstrap over articles (10,000 resamples, seed 0, 95% percentile), McNemar exact as secondary (`r1b_stats.py`). Also reported: each arm vs ORIG, n questions and n articles with every number, per-arm prefill/compaction wall time and peak memory.
6. Decision rule (held-out, primary). If the lower 95% bound of LOCAL - GLOBAL is above -3 points, M8 ships segment-local F. Otherwise M8 needs a global or query-reservoir design. The selection split is reported first for sanity and does not decide.
7. Cost: no Modal spend. Runs are resumable per article; a smoke article measures prefill/decode speed before the split is launched.

## Rung R2b: the shipping uniform-block design over 1 and 8 rounds, on the held-out articles

Written 2026-10-15 before any R2b row exists. What has been seen: the R2 rows and their report (selection split, 20 articles, per-head budgets), the R2 pilot, the R1(e) rows on both splits (uniform blocks at k = 0 only), and the engine plan "Revision 2". Nobody has run the shipping design (uniform 64-slot blocks) with distractor articles in front of the questioned article, on any article. R2's F-SEG(8) - F(0) = -0.6 [-3.3, +2.2] is the reason for this rung: its lower bound missed the -3 rule, so that question stays open. The reading of R2 (point estimate near zero) sets an expectation; it does not set the rule below.

### Question

Does the design the engine ships hold up over many rounds, on more articles? The design: every sealed 640-token block is compacted once, to 64 slots in every head and layer (uniform), from the queries of that block's own prefill, and is never compacted again. The test: put 1 or 8 other articles in front of the questioned article (so the cache holds blocks from 2 or 9 articles, about 12K or 50K tokens) and ask the questioned article's questions. Primary: how much accuracy does the shipping arm lose between k = 0 and k = 8, with a confidence interval tight enough to rule a 3-point loss in or out.

### Arms (all at ratio 0.10; k = number of distractor articles in front of the questioned article)

- FULL: no compaction, same context as the shipping arm. To keep memory bounded on this machine the context is fed in the same 640-token blocks (each block prefilled behind the uncompacted blocks before it), not in one pass. The CPU tests for R2 show that block-by-block prefill equals one-shot prefill (1e-4 on keys and values, 1e-3 on logits, fp32); on the GPU the two can still differ by rounding, so a few greedy answers may flip. The k = 0 reference is the existing R1(e) ORIG row, which was prefilled in one pass; FULL(k) - ORIG(0) therefore carries this small noise and is reported only as context.
- F-BLK-UNI (the shipping arm): the arm R2 calls F-BLK (`r2_core.gen_fblk`: the token stream of all articles, turn tags included, cut by position into 640-token blocks; each sealed block compacted once to 64 slots with the earlier compacted blocks as prefix; the open tail under 640 tokens stays exact; the Gated DeltaNet state carried through) but with the R1b cell FUNI as the compactor, which gives every head the same slot count. This is exactly the R1(e) F-UNI-LOCAL setup, run on a longer stream. At k = 0 it is R1(e)'s F-UNI-LOCAL, so k = 0 is not rerun.
- k in {1, 8}. k = 0 rows are reused from R1(e) (`evidence/am/R1e/{held,sel}/`): ORIG for FULL, F-UNI-LOCAL for the shipping arm. R1(e) ran on the same machine with the same code copy (`r2_core.py` on lucebox6 and in this repo have the same md5), so no cross-machine noise enters the primary contrast. A repeatability check re-runs the shipping arm at k = 0 on held-out articles 20-22 and reports how many answers differ from the R1(e) rows (expected 0; any difference is reported next to the primary result as the greedy-flip floor).
- Not run: F-RE and F-RE-UNI (collapse at k = 8, R2 Amendment 2), and the per-head-budget arms F-SEG and F-BLK (R2 covers them; the engine no longer uses them).

### Probe set: which articles, and why

- Stage 1, decides: the 30 held-out articles, loader indices 20-49, 537 questions (the R1(e) held-out set). Run in loader order. Distractors follow the R2 rule, `random.Random("r2-0-{probe_index}").sample(range(50, 90), 8)`, nested (the k = 1 context uses the first distractor of the k = 8 list). The pool 50-89 does not overlap 0-49.
- Stage 2, reported separately and pooled: the 20 selection articles, loader indices 0-19, 357 questions (k = 0 rows exist in R1(e) `sel/`), run only after stage 1 is complete and only if the time gate below allows it.
- Why both: the primary has to be decided on articles the design was not tuned on, so the held-out set decides. But the held-out set alone gives a lower bound that reaches -3 only if the point estimate is already close to zero (see "Planning arithmetic"). Adding the selection articles costs about 4 hours on a machine that costs nothing, and takes the interval from 30 to 50 articles. The selection articles are not clean for choosing a method, but this question does not choose a method: it compares one fixed design against itself at k = 8 and k = 0, so the selection history leaks very little into it. Stage 2 can change the verdict only in the one way written in the decision rule.
- Context length: about 50K tokens at k = 8 (R2 measured 50,024 on average), about 12K at k = 1. Questions, prompt, parser, scoring and greedy decoding (at most 256 new tokens) are the R1b ones, as in R2.

### Metrics and statistics

- Correct counts out of N per arm and k, with the number of questions and articles next to every figure.
- Paired differences per question, on the same questions: ship(8) - ship(0) (primary), ship(1) - ship(0), ship(8) - ship(1) (does the loss grow with rounds), FULL(8) - ship(8) (secondary), FULL(1) - ship(1), and FULL(1) - ORIG(0), FULL(8) - ORIG(0) (the effect of distractors alone, with the chunked-prefill noise noted above).
- Cluster bootstrap over articles, 10,000 resamples, seed 0, 95% percentile interval; McNemar exact as the secondary check. Same code as R2 and R1(e): `r1b_stats.py`, unchanged.
- Also reported per arm and k: context tokens, effective slots kept, prefill and compaction seconds, answer seconds, peak GPU memory.
- Planning arithmetic (not part of the rule). R2's interval at 20 articles was 5.5 points wide, a standard error of about 1.4 points; at 30 articles that is about 1.1, at 50 articles about 0.9. For the lower bound to reach -3 the point estimate must be about -0.8 or better at 30 articles and about -1.3 or better at 50. If the true loss is the -0.6 that R2 measured, the chance of passing is about 57% at 30 articles and about 79% at 50. So an inconclusive stage 1 is a likely outcome, and the rule below says what happens then.

### GPU, machine and parity

- Machine: lucebox6, integrated GPU (Radeon 8060S, gfx1151, device 1 via HIP_VISIBLE_DEVICES=1, shared RAM), the machine R1(e) used. Environment as in R1(e): `lucebox/r1e/env.sh`, everything under `$HOME/r1e` (`TMPDIR` is `$R1E/tmp`), nothing in /tmp. Model Qwen/Qwen3.8-27B bf16, transformers 5.19.0, ROCm torch 2.9.1, SDPA attention, pinned compaction repo @ 2020f6d3 with patches 0001-0003 and the overlay: the R1(e) install, unchanged.
- Parity contract: model file and revision, sampling (greedy), question prompts and the 256-token limit are the R1(e) ones. Both arms run in one session on one code copy. The k = 0 reference rows come from the same machine and code. Nothing from Modal (H100 or H200) is mixed into any contrast. Caching does not apply (no server, no prefix cache: every context is built from scratch per article and arm).
- Memory (read-only look at lucebox6 on 2026-10-15, nothing stopped): 125 GiB total, 28 GiB free, 59 GiB available (32 GiB of the total is page cache), no process above 70 MB resident, GPU 1 idle; the GPU can map up to 120 GiB of that RAM (kernel option `amdgpu.gttsize=122880`). R2's peak at k = 8 on the H200 was FULL 65.1 GB and F-SEG 80.2 GB; F-SEG is not run here. The shipping arm needs about the weights plus a small cache: in R1(e) its peak was 51.0 GB on every one of the 30 held-out articles, because blocks are prefilled 640 tokens at a time, and R2's F-BLK at k = 8 peaked at 54.7 GB. FULL at k = 8 needs the weights (about 51 GB) + the full key-value cache of 50K tokens (131 KB per token, 6.6 GB) + one block's activations, about 62 GB, below the 66.2 GB that R1(e) already reached on this machine with ORIG on an 8.3K-token article. So both arms fit within what R1(e) already used. The one risk is FULL, whose peak is only estimated; the smoke run below measures it before the run starts.
- Time (from R1(e) held-out rows on this machine). Shipping arm: 3.5 to 3.6 s per 1K tokens to build (24.3 s for a 7,016-token article), 17 to 20 s to answer an article's questions. A 50K-token stream builds in about 180 to 200 s at k = 8; k = 1 is a checkpoint of the same pass, so the arm costs about 200 s + 2 x 20 s = 240 s per article. FULL, block by block: no compaction step, so at most the same per-token cost, about 250 s for the 50K stream, plus about 30 s to answer at each of k = 1 and k = 8 (answering ORIG at 7K takes 23 to 26 s; the extra 6.6 GB of cache adds about 12% to the weight reads per token), about 310 s per article. Both arms: about 550 s per article, 700 s with 25% added for the occasional long answers R1(e) saw (56 to 253 s, in 9 of 90 article-arm cells, none in the shipping arm). Stage 1: 30 x 700 s = 21,000 s = 5.8 h. Stage 2: 20 x 700 s = 14,000 s = 3.9 h. With the smoke run and the repeatability check (about 0.3 h): about 10 h in total, inside the 24-hour allowance.
- Alternative, Modal H200 at $4.54 per hour: R2 took 2952 s for three arms at k in {1, 8} on 20 articles, about 148 s per article, of which F-SEG (not needed) was the most expensive. FULL plus the shipping arm at k in {0, 1, 8} on 50 articles, about 120 s per article, is about 6,000 s plus model loads, about 1.8 h, about $8. It would also need k = 0 rerun on the H200 so the primary contrast does not cross machines. It is faster, but it costs about $8, while the iGPU answers the same question for $0 within the time allowed and with the k = 0 rows already on the same machine. Decision: lucebox6 iGPU, cost $0.00, no Modal spend. Fallback if the iGPU becomes unavailable for a day: an amendment first, then the H200 with a hard cap of $12 (about 2.6 h of H200 time).

### Smoke, cost gate and stopping rules

- Smoke (throwaway, rows not kept, written under `$R1E/out/r2b_smoke`): held-out article 20, first 3 questions, k in {1, 8}, both arms, to measure build time, answer time and the peak memory of FULL at k = 8. The smoke accuracies are not used for any decision; article 20 is re-run in the full run, as R2 re-ran its pilot articles.
- Gate 1 (memory): if FULL at k = 8 runs out of memory or its peak is above 90 GiB, FULL is dropped at k = 8 (the secondary contrast is reported as "not run on this machine"); the shipping arm and the primary contrast are unaffected. FULL is not moved to Modal without an amendment, because that would cross machines.
- Gate 2 (time): if the smoke projects stage 1 above 14 hours, stage 2 is not run. If stage 1 itself passes 24 hours of wall, the run stops and the articles done, in loader order from 20, are reported as a prefix of the held-out set, with the prefix size stated in every number.
- Cost cap: $0 on Modal; no Modal job is started under this preregistration.
- Rows go to `evidence/am/R2b/held/` and `evidence/am/R2b/sel/`, the report to `evidence/am/R2b.md`. Runs are resumable per article.

### Decision rule (set now, before any row)

Primary contrast, stage 1 only: D = ship(8) - ship(0) on the 30 held-out articles, 537 questions, in points, with the 95% cluster-bootstrap interval [lo, hi].

1. PASS: lo >= -3. The shipping design is cleared to 8 rounds at 50K tokens within the 3-point margin. Engine plan: Revision 2 stays as written; the on-engine QuALITY gate (M7) also runs multi-article contexts, with the R2b rows as its reference; the claim "the loss does not compound over rounds" goes into the plan with this interval.
2. INCONCLUSIVE: lo < -3 and hi >= -3. Go to stage 2 and pool the 50 articles (894 questions; article is the cluster). The pooled interval [lo50, hi50] decides under the same margin: lo50 >= -3 counts as PASS, hi50 < -3 as FAIL, anything else stays INCONCLUSIVE. A pooled INCONCLUSIVE means the plan ships the design on the point estimate with the risk stated, and adds a multi-article arm to the on-engine gate (M7) so that the real engine measures what this rung could not.
3. FAIL: hi < -3 on the held-out articles alone, meaning every plausible value is a loss of more than 3 points. Engine plan: the never-re-compacted design is not the default for long contexts. The next rung compares a higher ratio (0.20) and keeping the most recent blocks exact; M7 waits for that result. Revision 2 gets a note that its R1(e) basis held for one article but not over rounds. M9 (pager and prefix-cache storage) does not depend on this result and keeps its place.

Secondary contrast, no gate on the primary: G = FULL(8) - ship(8), held-out. R1(e) measured the single-round gap on this machine at 4.28 points [2.21, 6.37] (ORIG - F-UNI-LOCAL). If G's point estimate is above 6 points (1.5 times the single-round gap), the plan says compaction costs more with a long context than the R1(e) number, and the agentic gate (item 8 of Revision 2) gets a "compaction off" arm next to the compaction arm. If G is 6 or below, the plan quotes about 4 to 5 points against full KV. Also reported, no rule: ship(1) - ship(0), ship(8) - ship(1), and the repeatability result.

What does not change the rule: the secondary contrast cannot rescue or fail the primary; stage 2 cannot turn a held-out PASS or FAIL into anything else; smoke and repeatability results are never used as accuracy evidence.

### Driver changes the run needs (described here, not made; to be made and unit-tested on the CPU before any GPU run)

1. `modal/am_qwen35/r2_core.py`: register two arms. `F-BLK-UNI` uses the existing `gen_fblk` with compactor cell `FUNI` (add it to `ARMS`, `ARM_GENS` and `ARM_CELL`; the runner is built with `budget = None`, as R1(e) does). `FULL-CHK` is a new generator: the same 640-token cut as `gen_fblk`, but each block goes through `extend` uncompacted, yielding at each requested k with the open tail added the same way. A CPU test on the tiny hybrid (next to `tests/test_multiseg.py`) must show that `FULL-CHK` equals `gen_full` on keys, values and following-question logits, and that `F-BLK-UNI` at k = 0 over one article equals R1(e)'s `local_triples`.
2. A new driver `lucebox/r2b/r2b_run.py`, modelled on `lucebox/r1e/r1e_run.py` (same model path, same ROCm environment, same resume behaviour), calling the existing `r2_core.run_r2` with the split label `held` or `sel` in the row file names, probes 20-49 then 0-19, ks 1 and 8, arms `FULL-CHK,F-BLK-UNI`, and an `--n_questions` option for the smoke. The Modal entry `r2.py::pilot` is not used.
3. A new report `lucebox/r2b/r2b_report.py`. `r2_report.py` reads only `__sel.jsonl` names and the R1b `orig`/`F` k = 0 rows, so it cannot be reused as is. The new one reads R2b rows for k = 1 and 8, takes k = 0 from the R1(e) rows (ORIG as FULL, F-UNI-LOCAL as the shipping arm) for the matching split, and computes every contrast above with `r1b_stats.py` for held-out, selection and pooled, plus the repeatability count.
4. Nothing else: no change to the compaction code, prompts, parser, scoring, budgets or patches.

### R2b Amendment 1 (2026-10-15, before any R2b row): run order and stage 2

Written after the drivers were committed (003bb8a2) and before the smoke or any row. Nothing in the arms, k values, articles, prompts, scoring or decision rule changes. Two things change.

1. **Run order.** The primary contrast uses only the shipping arm; the k = 0 reference rows already exist. So the shipping arm runs first, over all 30 held-out articles at k = 1 and 8. That takes about 30 x 240 s = 2.0 h, since R1(e) saw no long answers in this arm. Then the repeatability check runs, then FULL over the same 30 articles, which takes about 3.2 h. The primary verdict is computed as soon as the 30 shipping rows exist. FULL is not part of the primary, so this order cannot change the verdict. The secondary contrast G is reported once FULL completes.
2. **Stage 2 only on INCONCLUSIVE.** The decision rule uses the 20 selection articles only when stage 1 is INCONCLUSIVE ("stage 2 cannot turn a held-out PASS or FAIL into anything else"). So stage 2 runs only in that case, shipping arm first, then FULL. After a PASS or FAIL it is not run under this preregistration.

Why: the original order interleaved both arms and ran stage 2 regardless of the verdict. That put the primary verdict about 6 h and the full run about 10 h away, though only about 2 h of it decides anything. More hours do not tighten the interval. Its width is set by the number of articles (see "Planning arithmetic"), and that number is unchanged.

### R2b withdrawn (2026-10-15, before any R2b row)

R2b is withdrawn. No row was produced and no smoke was used.
- **The multi-round question is closed.** The project owner ruled on the R2 pilot: re-compaction damages accuracy (F-RE and F-RE-UNI collapse at k = 8), and the compact-once design holds (F-SEG(8) - F(0) = -0.6 points). R2b re-asked that question on more articles.
- **Accuracy is decided on agentic coding traces,** not on articles. The B0 harness is in `evidence/agentic/B0_3090/`: 60 OpenHands trajectories, recall probes and next tool call. The bar is to beat POOL16K's 75.6% retention.
- **The articles that remain have one job.** In M7 they check that the engine reproduces the Python reference rows that already exist. That is a parity check, not an accuracy question.
- **The drivers in 003bb8a2 stay.** They feed a token stream to the shipping arm (F-BLK-UNI) and to block-by-block full KV (FULL-CHK), which is the stream an agentic trace needs.

## Rung A1: the shipping design on the B0 agentic traces, offline

Written 2026-10-15 before any A1 row exists. What has been seen: all of B0 (`evidence/agentic/B0_3090/`, every table and the per-request rows), the R1(e) and R2 rows, the engine plan "Revision 2", and the harness and driver source. Nobody has run the shipping design on an agentic trace. Paths below are relative to `optimizations/kvflash/`; "key-value cache" is written KV. Section 1 records what was measured on the CPU while writing this text (token counts, prompt parity, cost arithmetic); those scripts read the harness data and write nothing but their own output under `/mnt/kimi-k3/build/a1_stats/`. Nothing was run on a GPU, nothing on Modal, and the only contact with lucebox6 was read-only commands (memory readings, file listings and checksums, the model config, a copy of `tokenizer.json` to the build host).

### Question

R2b is withdrawn (last section above) and accuracy is decided on agentic coding traces. A1 asks the one question the engine's final gate would otherwise answer too late and too expensively: if every sealed 640-token block is compacted once to 64 slots per head and layer (ratio 0.10, uniform, fitted from that block's own prefill queries), the Gated DeltaNet recurrent state carried through and the open tail (under 640 tokens) kept exact, does the model still recall what it recalled with full KV, and still choose the same next tool call? The design is `F-BLK-UNI` in `modal/am_qwen35/r2_core.py` (`gen_fblk`, lines 294-310, commit 003bb8a2); its uncompacted twin is `FULL-CHK` (`gen_fullchk`, lines 313-333), the same 640-token block-by-block prefill with no compaction (a CPU test shows it equals one-pass full KV). Both are fed the B0 requests, token for token, in the Python reference on Qwen3.8-27B bf16 (the R1(e) install on lucebox6).

### 1. The harness, what A1 reuses, and what it cannot

**How B0 builds a request** (`agentic_eval/run_arm.py`, `build_request`, lines 96-104). Messages are `traj["messages"][:cut_index]`; a probe appends one user message whose text is the question plus `ANSWER_SUFFIX` (`probes.py:20`, "Reply with the answer only: no explanation, no quotes, and do not call any tool."); a decision point appends nothing. Tools are `traj["tools"]` (five per trajectory). Temperature 0, seed 1, non-streaming, `chat_template_kwargs = {"enable_thinking": False, "preserve_thinking": True}` (`convert.py:21`; with these the rendered history does not change when a user turn is appended). The server renders with `--chat-template-file` = `qwen36_template.jinja` (`convert.TEMPLATE`, line 18; md5 a7f294a5f0be5f1903214304f259f87f), which is not the template shipped with the Qwen3.8 model directory (md5 519239a4...), so the Python side must use the harness template and not `apply_chat_template`. Answer limits are **64 new tokens for a probe and 256 for a decision** (`PROBE_MAX_TOKENS, DECISION_MAX_TOKENS`, line 30), not 256 for both. The engine ran with `--max-ctx 52224` (line 25). `run_arm.py` also sends a "primer" request (max_tokens 1, lines 159-167) so the server can snapshot the shared prefix; offline there is no server and no prefix cache, so the primer has no counterpart.

**Cut points and tasks** (`agentic_eval/tasks.py`, fixed seed 20261014, file `/mnt/kimi-k3/artifacts/agentic-eval/tasks_all60.jsonl`, md5 10d9da8d568463073ddb3c6234986ac8). For each trajectory and each P in {12288, 24576, 36864} (line 12), the cut is the last assistant message whose token boundary is at most P and that follows a tool message; it is skipped if that boundary is more than 4096 tokens under P (lines 25-31). Probe candidates are extracted from `messages[:cut]` by `probes.extract_candidates` (lines 62-149): a fact that occurs in exactly one tool output, at least 1500 tokens before the cut (`MIN_DIST_TOK`, line 19), up to 12 per (trajectory, P) round-robin over six types (`sample_probes`, lines 152-163). Decision points are assistant boundaries past 12288 tokens that follow a tool message, 24 per trajectory, one per equal-count bin (lines 38-50); the gold call is the call the OpenHands agent really made.

**B0 numbers, recomputed from the task and result files.**
- 60 trajectories, 3292 tasks: 1854 probes (P = 12K: 450 over 38 trajectories, 24K: 696 over 58, 36K: 708 over 59) and 1438 decision points. Probes by type: line 808, cmdline_text 457, cmdline_path 367, defname 181, traceback 23, failed_test 18. Probe distance from the cut (`distance_tok`): 0-4096: 558, 4096-8192: 252, 8192-16384: 601, over 16384: 443 (median 9491).
- Trajectories are truncated at the last assistant boundary at or under 48K tokens (`build_data.py:18-19`). Truncated length: min 32,904, median 47,802, mean 45,255, max 48,633 tokens; 2,715,297 in total. The stream that has to be built is cut at each trajectory's last task: 2,704,747 tokens in total (median 47,480 per trajectory). Request prompt lengths in the engine rows: probes median 24,454 (max 36,974), decisions median 29,488 (max 48,647). The system prompt plus tool schemas (the head) is 5,921 tokens in every trajectory, which is 9.25 blocks.
- 1527 distinct cut positions over the 60 trajectories (median 25 per trajectory, max 27): these are the checkpoints.
- Retained probes (engine full KV correct): 951 of 1854 over all 60 trajectories (7 to 27 per trajectory). Decision points where engine full KV produced a parsable call: 1204; the other 234 stopped at the 256-token limit. Engine completion lengths: probes mean 16.9 tokens (max 64), decisions mean 113.6 (1204 parsed: mean 85.9; total 163,313 over the 1438).
- POOL16K, the bar: 719 of 951 = 75.6% (r1b_stats gives [72.9, 78.2] with its 10,000-resample, seed-0 cluster bootstrap over trajectories, the same as B0's Wilson interval). By distance: 96%, 91%, 96%, 6%; by P: 95%, 94%, 45%; by type: defname 77%, line 77%, traceback 77%, failed_test 78%, cmdline_path 71%, cmdline_text 73%. Next-call agreement 76.2% (918 of 1204).

**Scoring** (`agentic_eval/score.py`). A probe is correct if `probes.score_exact(content, answer)` (lines 61-62; `normalize` and `score_exact`, `probes.py:28-45`). A decision agrees if the first parsed call has the same `(tool name, key)` as the reference's (`probes.call_key`, `parse_response_call`, lines 166-199; `score.call_of` lines 65-68); reference no-call points are excluded and counted, an arm parse failure counts as disagreement. `analyze` (lines 98-168) builds every table B0 reports, with the reference arm named `FULL`.

**Reused unchanged (harness code, not re-implementations):** the trajectories (`/mnt/kimi-k3/datasets/agentic-eval-prepared/trajectories.jsonl`, md5 4f2f96e2db420acb3b54e987c32ab6e4, 12.7 MB) and the task list; `run_arm.build_request` (messages, tools, limits, template keywords); `convert.render` (the template, run with `convert.TEMPLATE` pointed at the copied template file, an attribute assignment, not an edit); `probes.py` and `score.py`, and `report.py`, for correctness, parsing, key, normalization and every table. The reference's rows are written in the harness row schema under the file names `FULL.jsonl` (the Python FULL-CHK; the name is only what `score.py` requires), `SHIP.jsonl`, `FULLB.jsonl`.

**What cannot be reused, and its Python stand-in.**
1. *The server's tool-call parser.* The engine turns generated text into `tool_calls` with `server/src/server/tool_parser.cpp` (1870 lines, eleven syntaxes; `parse_tool_calls(text, tools)` in `tool_parser.h`). A Python port would be a re-implementation. Instead a small CPU-only command-line program is built from that file unchanged (g++ and nlohmann/json, both on lucebox6) that reads generated text and the tool list and prints `cleaned_text` and `tool_calls`; its output feeds `probes.parse_response_call` and, as `content`, `probes.score_exact`. It needs a round-trip test: for the 1438 gold calls, render the assistant turn with the template and parse it back; all names and arguments must match. If the program cannot be built standalone, A1 stops and is amended; a hand port is not substituted silently.
2. *The server's renderer and tokenizer.* Measured today on the CPU, with the Qwen3.8 tokenizer and `convert.render`, against `usage.prompt_tokens` of every FULL request in B0: the prefix property holds in all 3292 requests (tokens of the cut prefix are exactly the first tokens of the whole request), so one token stream with checkpoints is valid. But the Python prompt equals the server's token count in only 1893 of 3292 requests; in the other 1399 it is shorter by 1 to 44 tokens (median 6 of 25,000-48,000), growing with the number of earlier tool calls and zero at the head, so it is a formatting difference in how the server prints tool-call parameters. The cause is not identified here. It never touches the probe question or the facts, it is a few tokens in tens of thousands, and A1 does not try to imitate it. It is a known departure, reported in the text of the result next to the quantization difference.
3. *The tokenizer file.* The harness counts tokens with the Qwen3.6 NVFP4 repository's `tokenizer.json` (`convert.TOKENIZER`); the model's own file (sha256 0997f410...) has the same vocabulary but another pre-tokenizer, and tokenizes one of the 60 trajectories differently (Devanagari text, 48,208 against 48,147 tokens). The Python side tokenizes with the model's own tokenizer, and recomputes every cut position as the token length of the rendered cut prefix (checked equal to a prefix of the full token stream). The B0 labels (`distance_tok`, `P`) are kept as they are so that the tables line up.
4. *Different weights and cache type.* The engine ran Q4_K_M weights with a q4_0 cache; the reference runs bf16 weights and a bf16 cache. The reference's own full-KV yield (how many of the 1854 probes it answers correctly) will therefore not be 951, and its retained set is not B0's.

**Where the data lives and whether it can go to lucebox6.** On the build host: `/mnt/kimi-k3/datasets/agentic-eval-prepared/trajectories.jsonl` (12.7 MB), `/mnt/kimi-k3/artifacts/agentic-eval/tasks_all60.jsonl` (1.5 MB), the template, the B0 rows under `/mnt/kimi-k3/artifacts/agentic-eval/full/` and the committed copies in `evidence/agentic/B0_3090/`. Together about 16 MB; lucebox6's root disk has 45 GB free (98% used), so they fit. Copy to `~/r1e/data/a1/` and `~/r1e/code/agentic_eval/` and check the md5 sums above on arrival. Rows go to `~/r1e/out/a1/`, never `/tmp`.

### 2. Building the stream once per trajectory

Per trajectory and arm: render the longest cut, tokenize once, and run the arm over the stream with one checkpoint at each distinct cut position (about 25), answering every probe and decision of that cut from its checkpoint. No cut is rebuilt.

**Can the existing generators yield at arbitrary token positions?** Yes. `gen_fblk` and `gen_fullchk` use their `segs` argument only to form `stream = cat(seg.ids)` and the checkpoint ends `ends[k]` (lines 296-297 and 317-318). `articles` is not used, and `ks` is a list of segment indices. Their logic is positional: `sealed_blocks(consumed, ends[k])` seals whole 640-token blocks up to the checkpoint, and the open tail is added to a copy with `extend`, which does not mutate its prefix (`test_extend_does_not_mutate_its_prefix`, `tests/test_multiseg.py:117`), so earlier yields do not disturb later ones. A1 therefore builds `segs` as one `Seg` per interval between consecutive checkpoints and calls the generators with `ks = range(len(segs))`. `Seg.art` is not read by these two generators. Only `run_r2` (lines 362-399) is tied to QuALITY (`build_segments`, `Session.answer`), and A1 does not use it.

**Smallest driver changes (described, not made).**
1. `r2_core.py`: `gen_fullchk` gets an optional `block` argument (default `BLOCK`), needed for the block-512 control in section 4, and an optional flag (default off, so R1(e) and R2b behaviour does not change) that calls `torch.cuda.empty_cache()` after each block (section 3). No change to `gen_fblk`.
2. A generic answer helper next to `Session.answer` (`r1b_core.py:121-133` is QuALITY-specific): take the suffix token ids, clone the checkpoint with `Session._cache` (line 101), prefill the suffix, and decode with `Session._greedy` (line 110) for at most 64 (probe) or 256 (decision) tokens, stopping on `<|im_end|>` (248046) or 248044 as `Session.eos` does; return the decoded text.
3. A new driver `lucebox/a1/a1_run.py`, modelled on `lucebox/r2b/r2b_run.py`: same model path, same ROCm environment, resumable per trajectory and arm; order per trajectory is FULL-CHK (all probes and all decisions), then the shipping arm (the probes the reference got right and the decisions where the reference produced a parsable call), then, on the control trajectories, the block-512 control.
4. The tool-parser program of section 1, with its round-trip test.
5. Nothing else: no change to compaction code, patches, prompts, the harness, the scorer or `r1b_stats.py`.

**CPU test it needs, on the tiny random hybrid** (a new `tests/test_a1.py` next to `test_r2b.py`, same fixture and the small test block of 24 tokens). Checkpoint positions that are not multiples of the block size, including one inside the first block, one exactly on a block boundary, two inside the same block, and the stream end. Assertions: (a) FULL-CHK at each checkpoint equals a one-pass prefill of `stream[:cut]` on keys, values (1e-4) and the logits of a following suffix (1e-3, fp32); (b) F-BLK-UNI at each checkpoint equals the state from a fresh run with that single checkpoint (intermediate yields change nothing), and at one checkpoint it equals R1(e)'s `local_triples` path on the same ids; (c) the answer helper's tokens equal `model.generate` greedy on the one-pass cache of `stream[:cut] + suffix`, with the stop token and the `max_new` limit honoured; (d) `block=512` seals at 512. Also, on the CPU, on all 60 real trajectories: the cut-prefix token ids are a prefix of the whole-request ids, and a hash of the token ids of every request, computed here and again on lucebox6, is identical.

### 3. Size, time, machine, memory

**Tokens to build per arm:** 2,704,747 (1527 checkpoints, an average open tail of about 320 tokens added at each).

**Time, lucebox6 integrated GPU (iGPU: Radeon 8060S, bf16, `HIP_VISIBLE_DEVICES=1`).** Measured on this machine: build 3.5 s per 1K tokens for the shipping arm (R1(e) held-out rows: 24.3 s for 7,016 tokens, including compaction); FULL-CHK blocks 35.2 s for 9,089 tokens at short context (the aborted R2b smoke, 3.9 s per 1K); decode 2.28 tokens per second uncompacted and 2.88 compacted (`lucebox/r1e/r1e_bench.py` output on a 6.6K article), so about 0.43 s per extra token. Answer time from the R1(e) rows: 1.2 s per question for ORIG (21-26 s for 18-19 questions), made of a prefill of about 130 tokens (0.7 s at 220 tokens per second) and one decode step for the one-letter answer. A1 answers are much longer than those, so decode steps dominate: a probe costs 0.5 s (78-token suffix prefill, cache clone) plus about 15.9 steps; a decision costs 0.5 s plus its steps. Assumptions: 2.2 tokens per second for FULL-CHK at 25-48K context (the 3.1 GiB of KV adds about 6% of weight reads), 2.85 for the shipping arm; FULL-CHK build 5.8 s per 1K on average, because each 640-token block is attended against up to 48K keys (the smoke measured only the short end; the smoke below measures it at 48K).

| Piece | Basis | Seconds |
|---|---|---|
| FULL-CHK build + tails | 2.70M tokens x 5.8 s/1K, 1527 tails x 1.9 s | 18,600 |
| FULL-CHK probes | 1854 x (0.5 + 15.9 x 0.45) | 14,300 |
| FULL-CHK decisions | 1438 x 0.5 + 161.9K steps x 0.45 | 73,600 |
| Shipping build + tails | 2.70M x 3.5 s/1K, 1527 x 1.5 s | 11,800 |
| Shipping probes | about 950 retained x (0.5 + 13.3 x 0.35) | 5,000 |
| Shipping decisions | about 1204 parsed, 102.2K steps x 0.35 | 36,400 |
| Control (block 512, 8 trajectories) | 8/60 of the FULL-CHK total | 14,200 |
| Total | 48.3 h, plus smoke, model loads, determinism check about 1.5 h | about 50 h |

With 25% added for long answers and slower decode at 40K+ context, about 62 h; per trajectory both arms take about 44 minutes. The engine's decision completions are the model of cost; the Python bf16 model may answer longer or shorter. About 63% of the time is decode of decision-point answers.

**Modal H200 comparison ($4.54 per hour).** R2 rows on the H200: F-BLK (the same 640-token blocks with per-head budgets) at 46.5K tokens builds in about 25 s (9.5 s block prefill + 15.3 s compaction, 0.54 s per 1K), and a QuALITY question answers in 0.22 s (4.2 s for 19), about 5 times faster than the iGPU. Decode rate on the H200 is not measured in any committed row; assumed 25 tokens per second for the 27B in this Python loop. Decode steps in A1: about 332,000 (FULL-CHK 191,000, shipping 115,000, control 25,500) is 13,300 s; builds 1,460 + 810 + 110 s, tails 600 s, suffix prefills and clones 650 s, two model loads 180 s: about 17,100 s = 4.75 h = **$22**; $17 to $33 for a decode rate between 33 and 15 tokens per second. Cap $35.

**Machine decision, using the rules.** lucebox6 first, $0, about 50 h of wall (62 h with the allowance). Modal H200 only if lucebox6 cannot do the job; the fallback is stated now so no money is spent without a written amendment: both arms re-run from scratch on one H200, expected $22, hard cap $35, GPU and dollars restated in the amendment before the job starts. FULL-CHK, the shipping arm and the control always run on the same machine in the same session and code copy (parity contract below); nothing from Modal is mixed into a lucebox6 contrast, as in R1(e), because kernels and torch builds differ between the two.

**The memory warning, explained from the code.** In the aborted R2b smoke, FULL-CHK reached about 50K tokens with the machine at 122 of 125 GiB used. Per 640-token block, `extend` (`r2_core.py:141-171`) does this against a prefix of L keys, with L growing by 640 every block:
- `compacted_cache_from_triples(..., clone=True)` (`cache_convert.py:75-83`) clones the whole prefix KV. KV is 64 KiB per token (16 full-attention layers x 4 KV heads x 256 x K and V x 2 bytes; `num_key_value_heads` 4, `head_dim` 256 in the model config; the R2b text's 131 KB per token was twice too high), so the cache is 3.1 GiB at 50K and a clone, the old copy and the `cat` of the new block each exist briefly.
- The additive mask is built (1, 1, 640, L) and then `add_beta_to_mask` (`overlay/models/qwen3_5/masks.py:51-71`) expands it to all 24 query heads and clones it: 24 x 640 x 50,000 x 2 bytes = 1.4 GiB per layer per block, even for FULL-CHK where beta is all zero.
- Attention goes through PyTorch's scaled dot-product attention (SDPA) with that explicit mask. The smoke log says the flash and memory-efficient kernels are experimental on this GPU and need an environment switch that is not set, so the math path runs (inferred from those warnings and from the peak memory below, not observed directly): key and value repeated to 24 heads (0.6 GiB each in bf16), and a 24 x 640 x L score matrix, 1.4 GiB in bf16 or 2.9 GiB in fp32, with a second matrix of the same size for the softmax. The R1(e) ORIG run is consistent: 66.2 GB peak at 8.3K tokens against 51 GB for the block path, which is about four to five such score matrices.
So the peak allocated memory per block at 48K is about weights 51 GiB + 3 cache copies 9 GiB + 8 to 14 GiB of attention temporaries, 70 to 75 GiB: about what R1(e) already reached. The 122 GiB is probably not the live tensors but the allocator: the temporaries have a new, larger size on every one of the 78 blocks, so cached free blocks of the previous size cannot be reused, and PyTorch keeps them until an allocation fails; on this machine the GPU memory is ordinary RAM and an allocation fails only when RAM is gone. `gen_fullchk` never calls `empty_cache` between blocks (it only synchronizes). This is a hypothesis to be tested, not a finding. The shipping arm is not affected: its prefix is at most about 5K slots, so the same temporaries are about a tenth the size (R1(e) peak 51.0 GB on all 30 held-out articles).
Idle reading of lucebox6 on 2026-10-15, before anything starts: 125 GiB total, 65 GiB "used" with no process of ours and no GPU client, 61 GiB free, 60 GiB available; this baseline also existed before the aborted smoke (R2b text: 59 GiB available, 28 GiB free, 32 GiB page cache). So the 122 GiB figure included it. Its origin (kernel or driver memory) is not known and is not tested here, because testing it would mean running the GPU.

**Memory gate (set now).**
- Mitigation in the driver from the start, not as an afterthought: `torch.cuda.empty_cache()` after every block of FULL-CHK, and a per-process cap `torch.cuda.set_per_process_memory_fraction` for 100 GiB so the allocator frees its cache before RAM runs out. Both are numerically inert.
- Before launch: no GPU client; record `free -m`; start only if `used` is at most 70 GiB and available at least 55 GiB.
- During the run: a watchdog reads `free -m` every 10 s into `~/r1e/out/a1/mem.log` and stops the run (SIGTERM, resumable) when `used` reaches 120 GiB or swap use exceeds 1 GiB. The driver logs `max_memory_allocated` and `max_memory_reserved` after every checkpoint.
- Smoke gate: FULL-CHK over the 48,633-token trajectory must finish its last block and a probe answer with the machine's `used` never above 110 GiB and `max_memory_reserved` never more than 15 GiB above `max_memory_allocated` (no ballooning). If it fails, the ladder is one rung per amendment: expandable-segments allocator setting; then run FULL-CHK in 320-token blocks (a different reference numerics, which makes the block-512 control also the reference's sensitivity check); then the Modal H200 fallback above. The shipping arm and the primary contrast are not run on a different machine from FULL-CHK.

### 4. Parity contract, noise floor

Parity contract (the bench-parity axes, stated before the run): model Qwen/Qwen3.8-27B bf16, same files, for FULL-CHK, shipping arm and control; KV bf16 on all; sampling greedy (argmax) with the harness limits 64 and 256 and the same stop tokens; the same rendered ids per request on all arms (one tokenization per trajectory, hashed and checked equal between the build host and lucebox6); one session, one code copy (the pinned compaction checkout @ 2020f6d3 with patches 0001-0003, the overlay, `modal/am_qwen35`), SDPA attention as R1(e); no prefix cache anywhere, every stream built from scratch per trajectory and arm, so the caching axis is symmetric by construction. The ratio, the block size, the uniform 64 slots, the algorithm (`highest_attn_keys_rms_nnls2_-3_3_lsq_on-policy`, cell FUNI) are R1(e)'s.

**Noise floor.**
1. *Greedy determinism.* On the smoke trajectory, 3 probes and 2 decisions are answered twice from the same checkpoint on each arm and in two separate processes; token ids must be identical. If not, the difference is reported next to the result as an additional floor.
2. *Reference sensitivity (the analogue of B0's FULLB).* The FULL-CHK arm is re-run with 512-token blocks on the first 8 trajectories of the file (about 130 retained probes, 160 parsed decisions). A numerically equivalent full-KV run with different reduction order shows how many answers the reference itself flips; B0's engine repeat flipped none (330/330, 393/393); a bitwise repeat of the same Python run tests determinism (item 1) but says nothing about sensitivity to rounding, which is what the reference's noise floor has to be. Reported as agreement of block-512 with block-640 on the same probes and decisions. No gate; if more than 5 points of the retained probes flip, the verdict text says the reference is soft by that much. The one-pass full-KV run cannot be the control: its score matrix at 37K tokens would be 24 x 37K x 37K x 4 bytes = 131 GB.

### 5. Statistics

Same code as R2 and R1(e): `modal/am_qwen35/r1b_stats.py`, unchanged, `boot_mean` (lines 14-24: resamples whole clusters, 10,000 resamples, seed 0, 95% percentile) with the trajectory as the cluster, and `mcnemar_exact` (lines 27-35). `score.py` also runs unchanged for the B0-format tables, with its own bootstrap (seed 1); those tables are for reading, and every interval quoted for a decision comes from `r1b_stats`. n probes and n trajectories are stated with every number.

- Primary: retention R = P(shipping arm correct | reference FULL-CHK correct), the mean of the 0/1 vector over the retained probes, with `[lo, hi]`.
- Per-bucket retention as B0 does: by probe type, by P, by distance of the fact from the cut (0-4096, 4096-8192, 8192-16384, over 16384), for the shipping arm; B0's POOL16K values for the same buckets are quoted beside them and marked as another machine and precision.
- Secondary: next-call agreement of the shipping arm with the reference's first parsed call (full key and name only), over the decision points where the reference parsed, with `[lo, hi]`, overall and by decision position (cut 12-24K, 24-36K, 36-48K; decisions have no fact distance, so B0's distance split applies only to probes). Reference-no-call points are counted and excluded; the reference stopping at 256 tokens in more or fewer points than the engine is reported.
- Descriptive, same-probe comparison with the engine: on probes retained by both the engine's full KV (B0) and the reference, the per-probe difference (A1 shipping - B0 POOL16K) with `boot_mean` and `mcnemar_exact` over the two correct vectors. This removes the difference in which probes are retained; it does not remove the difference in machine or precision, and is labelled so.
- Control: agreement of block-512 with block-640 FULL-CHK; no interval beyond counts.

**Planning arithmetic (not part of the rule), from B0's real cluster sizes, 951 retained probes in 60 trajectories, simulated with `boot_mean`.** The B0 POOL16K retention interval is 75.6% [72.9, 78.2], half-width 2.7 points (the clusters add almost nothing: the design effect is about 1). For the shipping arm the half-width is about 2.5 points if the true retention is 80%, 2.2 at 85%, 1.9 at 90%, 1.4 at 95%, and 3.0 at 60%. Chance of each verdict of the rule below, by true retention (n = 951, 300 simulations each): 70%: FAIL 0.98; 75.6%: PASS 0.04, INCONCLUSIVE 0.93, FAIL 0.03; 80%: PASS 0.87, INCONCLUSIVE 0.13; 85% and above: PASS 1.00. The number of retained probes is the reference's own, not 951: with 700, widths grow by about 17%, with 1200 shrink by about 11%. Doubling to 120 trajectories (stage 2) gives a half-width of about 1.8 points at 80%.

### 6. Decision rule (set now, before any row)

The bar is B0's POOL16K retention, 75.6%, taken as a fixed number from the plan (it has its own interval, 72.9 to 78.2, and the rule does not widen for it). The comparison is each side's retention against its own full KV, measured on different weights, cache types and machines; it is not a comparison of accuracy between engines. The test needs at least 50 trajectories finished with at least 600 retained probes; otherwise the result is reported as a prefix with its size.

Primary: R and its 95% interval [lo, hi] (section 5).
1. **PASS:** lo > 75.6%. The shipping design recalls more of what full KV recalled than the pool without compaction did, with confidence. The plan proceeds in its order, M2 (key bias in the attention kernel), M3 (query capture), M4 + M5 (top-64 scoring, beta and value fit), M6 (compacted flag, beta upload and serde), M7 (QuALITY gate on the automatic path), M9 (compacted state through the pager blob), then the agentic gate. The final engine gate on the B0 harness stays as written: the engine's own retention must beat 75.6% while its speed stays close to the pool arms, which A1 cannot measure. A1's retention and per-probe rows become the offline expectation for that gate; an engine result more than 5 points under A1's lower bound is investigated as an engine or Q4 effect before it is judged.
2. **INCONCLUSIVE, point estimate at or above 75.6%** (interval contains 75.6): M2, M3 and M4 + M5 proceed, since any compaction variant needs them (about 13 engineering days). M6 does not start until stage 2 is in: 60 more trajectories, the next 60 kept by `build_data.py`'s fixed-seed walk (the first 60 come out identical; asserted), the same pipeline, about 50 h at $0, which finishes inside those 13 days. The pooled 120 trajectories decide under the same rule against the same 75.6% (the new trajectories have no engine rows; the bar is a number, and that cross-set use is stated). Pooled lo > 75.6% is PASS, hi < 75.6% is FAIL, otherwise the plan ships the design on the point estimate with the risk written down and the agentic gate on the engine decides.
3. **FAIL:** hi < 75.6%, or INCONCLUSIVE with the point estimate under 75.6%. M2 to M5 still proceed (they serve every ratio and variant), and M6 onward waits. The next rung, **A2**, is preregistered separately before it runs; its default is **ratio 0.20** (128 slots per 640-token block, uniform), shipping arm only, against the same A1 reference rows (no FULL-CHK re-run; about 15 h), same rule. One preregistered branch: if the shipping arm's retention under 8192 tokens of distance (the 0-4096 and 4096-8192 buckets pooled) is under 90% while POOL16K had 91 to 96% there, the loss is recency, and A2 first tests keeping the most recent blocks exact instead. Why 0.20 first: exact-recent blocks cannot help facts that are far back, which is where the pool arms lose most (6% beyond 16K), while a larger ratio helps every distance. If A2 also fails, the compaction track stops after M5 and the question of whether 2.0b ships without compaction goes to the project owner.
No ratio or window is tuned inside A1: no arm is run in A1 besides the reference, the shipping arm and the control.

What does not change the rule: the secondary agreement, the descriptive comparison and the control cannot rescue or fail the primary. A low next-call agreement with a PASS is reported as a finding for the agentic gate, not as a failure of A1.

### 7. Smoke run, gates and stopping rules

- **Before any GPU time (CPU, here and on lucebox6):** the CPU tests of section 2, the harness's own `test_agentic_eval.py`, the tool-parser program and its round-trip test, the identical request hashes on both machines. Gate: all pass.
- **Smoke** (throwaway, rows not kept, under `~/r1e/out/a1_smoke`): the trajectory with the longest stream, `chatcmpl-69d54ff3e1e5db963e2109a2e92865d1` (48,633 tokens), both arms, all its checkpoints, 3 probes (one per P) and its last 2 decisions answered. It measures: build seconds per 1K at 12K, 24K, 36K and 48K for FULL-CHK; decode tokens per second at those lengths; answer seconds; peak allocated and reserved memory and the watchdog log; and the determinism check. Smoke accuracies are not used for any decision; this trajectory is re-run in the full run.
- **Gate 1 (memory):** as in section 3. **Gate 2 (time):** if the smoke projects the whole run above 72 h, decisions are answered at every second decision point of each trajectory (positions 2, 4, 6, ... in cut order, fixed now), for both arms and the control; probes are never subsampled. **Gate 3 (pipeline sanity):** after the first 5 trajectories, the reference's own yield (probes correct / probes attempted) must lie between 35% and 70% (B0 engine: 51.3%); outside it the run stops and the render, the parser and the stop tokens are checked. This protects against a broken pipeline and is not an accuracy criterion. **Gate 4 (determinism):** as in section 4; an unexplained difference is reported, not hidden.
- **Stopping rules:** no stopping on results and no interim look at R. The run stops only for a gate, for the watchdog, or at 96 h of wall; if the 96 h is reached, the finished trajectories in file order are reported as a prefix, with the prefix size in every number. Rows are resumable per trajectory and arm.
- **Cost cap:** $0 on Modal; no Modal job is started under this preregistration. The fallback needs an amendment (section 3).
- **Rows and report:** `evidence/agentic/A1/{FULL,SHIP,FULLB}.jsonl` (FULL = the Python reference FULL-CHK), `score.json`, `retained.json`, report `evidence/agentic/A1.md` generated with the harness `report.py`, `r1b_stats` intervals added.

### The engine comparison, and what A1 cannot measure

**What the offline arm stands for.** In the engine a compacted block is one 64-slot pool chunk (Revision 2, item 3), and the pool counts slots. A 16,384-slot pool holds 256 compacted blocks, which is 163,840 tokens: the 10-times figure. B0's sessions end at 52,224 tokens at most (`--max-ctx`): that is 81 blocks, 5.2K slots plus an exact tail under 640 tokens, so nothing compacted would page out of a 16K pool, or even of an 8K pool (128 blocks = 82K tokens). The offline arm with every compacted block resident is therefore what the engine does at B0's lengths. This agrees with the capacity check in the plan, which is about something else: prefix snapshots in RAM carry a fixed part of about 350 MiB (recurrent state and buffers) plus 18 KiB per token, so a compacted session shrinks 4.4 times at 128K and 6 times at 262K, not 10 times; at 52K the snapshot goes from about 1.27 GiB to about 0.45 GiB, 2.8 times. That limits how many sessions fit in RAM; it does not change what stays in the pool. Two points where the engine plan keeps more exact than A1's arm, so A1 is the harsher test: section 2 of the plan says the pinned head (system and tools, 5,921 tokens here) and the current turn stay uncompacted, and Revision 2 does not repeal that; A1 compacts every sealed block including the head and keeps only the sub-640 tail. One point where the engine must match the reference: blocks are sealed on a positional grid over the session stream, independent of where requests begin and end.

**What A1 cannot measure.**
- *Speed.* The offline reference is not the engine; its decode and build times say nothing about the engine's prefill or decode, so "close to the pool arms' speed" is left to the engine gate.
- *Cross-machine numbers.* The engine ran Q4_K_M weights with a q4_0 cache; the reference runs bf16. The comparison with 75.6% is retention against each side's own full KV and is worded that way everywhere: "the shipping design keeps R of what bf16 full KV recalled; POOL16K kept 75.6% of what Q4_K_M full KV recalled."
- *Quantization effects of the compacted state in the engine* (compacted K and V stored in q4_0, beta in float): the engine gate sees them, A1 does not.
- *The engine's rendering* (section 1, item 2) to the token.

### A1 Amendment 1 (2026-10-15, before any A1 row): the probes decide first, with one early look

Written after efb31219 and before any smoke or row. Arms, the reference, the parity contract, the scoring, the 75.6% bar and the 95% rule for the final analysis do not change. Three things change.

1. **Stage 1 is the recall probes only.** The primary is decided on probes. Decisions are the secondary, yet they took about 110,000 of the roughly 172,000 s in section 3 (about 64%). In stage 1, each trajectory's stream is built only to its last probe cut, which is at most 36,974 tokens, against 48,647 for decisions. That is 155 probe checkpoints, not 1,527. The block-512 control runs on probes only, on the first 8 trajectories, after stage 1.
2. **One early look, at 20 trajectories.** After the first 20 trajectories in file order, retention is computed with `r1b_stats.boot_mean` (10,000 resamples, seed 0) at **99.9%**, using the 0.05th and 99.95th percentiles. This is the Haybittle-Peto rule: an early stop needs overwhelming evidence, so the final 95% analysis keeps its error rate. If the 99.9% lower bound is above 75.6%, the result is PASS and stage 1 stops. If the 99.9% upper bound is below 75.6%, the result is FAIL and stage 1 stops. Otherwise all 60 trajectories run and section 6 decides at 95%, unchanged. Gate 3 (the reference's yield after 5 trajectories) still applies. This replaces "no interim look at R" in section 7.
3. **Stage 2, the decision points,** runs after the stage-1 verdict, on the same machine and in the same session as stage 1. It uses all 1,438 decision points, with the full stream to 48K, and reports next-call agreement with no rule attached. The engine's agentic gate measures next-call agreement anyway. Gate 2's subsampling applies only to stage 2.

Time for stage 1, using section 3's rates: FULL-CHK build of about 2.2M tokens at about 5 s per 1K tokens (shorter contexts) is about 11,000 s; FULL-CHK probes 14,300 s; the shipping-arm build about 7,700 s; shipping-arm probes 5,000 s. All 60 trajectories take about 10.6 h on lucebox6, and the early look comes after about 3.5 h. On a Modal H200, with section 3's assumed decode of 25 tokens per second, all 60 trajectories take about 1.1 h, roughly $5. The machine is chosen in the next amendment.

### A1 Amendment 2 (2026-10-15, before any A1 row): lucebox6, and a 6-session pilot first

The project owner chose the machine and asked for a short pilot before the proper run.

- **Machine:** lucebox6 integrated GPU, $0. No Modal job is started.
- **Pilot:** the first 6 trajectories in file order, both arms, stage-1 probes only.
  - The pilot replaces the smoke. The memory gate and watchdog of section 3 apply, the determinism check of section 4 item 1 runs on its first trajectory, and Gate 3 (the reference's yield, 35% to 70%) is checked on its 6 trajectories.
  - It reports, with counts: the reference's correct probes, retention with its interval (6 clusters, so the interval is only indicative), retention by distance, build and answer seconds, and peak memory.
  - **The pilot decides nothing.** It only says whether the proper run (Amendment 1: early look at 20, then 60) goes ahead as written.
  - If nothing in the design changes after the pilot, its rows are the first 6 trajectories of the proper run. If anything changes (arms, ratio, render, limits), its rows are discarded and the proper run starts from trajectory 1.
- **Pilot time:** about 6/60 of stage 1, about 1.1 h plus the model load.

### R2b correction (2026-10-15): the smoke had written rows

An independent fact-check found that two R2b texts said "before any row" when that was not true.
- **The smoke started before Amendment 1.** The smoke run on lucebox6 had already started when Amendment 1 was committed (fe8a834c, 09:40:15). So "before the smoke" in that amendment is wrong.
- **The smoke wrote throwaway rows before the withdrawal.** It wrote `~/r1e/out/r2b_smoke/FULL-CHK__k1__t0.1__held.jsonl` at 09:40:49: 3 rows, held-out article 20, k = 1, full KV. The withdrawal came later (f8abef0d, 09:48:06), so "before any row" there is wrong. Both texts should say "before any kept row".
- **Nothing changes.** Smoke rows are throwaway under the R2b text. Nobody read their accuracy for any purpose; only build seconds, answer seconds and peak memory were reported. The file stays on lucebox6, unused.

### A1 Amendment 3 (2026-10-15, after the pilot started, before any pilot row was read): the server's tokenizer, and the swap rule

Written at about 12:05 CEST, after the pilot started on lucebox6 at 09:53:12 UTC. At that point `FULL.jsonl` held 12 rows, from trajectory 1's first probe cut. Nobody has read whether they are correct. Both changes below were already in the code the pilot runs. They are commits 7725ffc2 and 53cfda9a. The files on lucebox6 have the committed md5s, and each was last modified before the launch. So the pilot runs under this amendment, and Amendment 2's discard rule ("if anything changes ... its rows are discarded") is not triggered.

1. **A1 tokenizes with the server's own tokenizer.** This replaces the tokenizer choice in section 1 items 2 and 3.
   - **The cause of the 1-to-44-token shortfall is found.** The server's hand-written Qwen3.5 pre-tokenizer (`server/src/server/tokenizer.cpp`) splits a whitespace-only line such as "\n   \n" into "\n" and "   \n". Hugging Face `tokenizers` keeps it as one piece. The template engine is not the cause.
   - **The fix.** A1 now tokenizes with that `tokenizer.cpp`, compiled unchanged into `tokenize_cli` (`lucebox/a1/server_tok.py`). It runs over a tokenizer-only GGUF derived from the engine GGUF (`lucebox/a1/make_tok_gguf.py`).
   - **Checked against B0.** Against `usage.prompt_tokens` of every FULL request in B0, 3292 of 3292 counts are equal, on both machines. The known departure in item 2 no longer exists.
   - **Cut positions and the hash file.** Cut positions are recomputed with this tokenizer. The request-id hash file `evidence/agentic/A1/request_ids.sha256` (md5 b8da5217...) is generated under it and is identical on both machines. The prefix property holds on all 60 trajectories. The B0 labels are kept as before.
2. **The swap rule in section 3 counts growth.** "Swap use exceeds 1 GiB" becomes "swap use grows by more than 1 GiB over its level when the watchdog starts". The 120 GiB `used` limit is unchanged.
   - lucebox6 idles at about 459 MiB of swap.
   - The first launch (09:50:34 UTC) was stopped during the model load when swap reached 1027 MiB. It wrote no row. It is kept in `~/r1e/out/a1_pilot.try1_swapkill`.
3. **A record, not a change: the tool-parser round trip of section 1 item 1 passes.**
   - 1400 of 1400 gold calls match on names and argument dicts.
   - The other 38 of the 1438 decision points have no gold call, so there is nothing to round-trip. 37 are at the end of a trajectory and 1 has no call.

### A1 Amendment 4 (2026-10-15, after the pilot and D1, before the proper run): the proper run goes ahead as written

What has been seen:
- **The pilot** (`evidence/agentic/A1/pilot/`, 8f8f54f8). Retention is 10 of 118, 8.5% [4.4, 12.5], with only 6 clusters. B0 POOL16K gets 88 of 111 on the same trajectories.
- **The D1 identity diagnostic** (`evidence/agentic/A1/diag_identity/`, 7e531d7d). This is outside A1. The shipping-arm generator, with every block kept at 640 of 640 slots, answers 38 of 38 of the reference-correct probes on trajectories 1 and 2. The compaction path is therefore mechanically sound on agentic streams, and the pilot's 8.5% is the effect of ratio 0.10.

The proper run (Amendment 1: stage 1, probes, one early look at 20 trajectories at 99.9%) goes ahead as written, on the same machine, under the same rules.
- **The pilot rows stay as trajectories 1 to 6.** The only code change since the pilot is 7e531d7d. It adds a `ratio` option to `load_model` and `a1_run.py`, whose default (0.10) is `torch.equal` to the pilot's shipping arm, tensor for tensor (CPU test `test_d1_ratio_option_default_is_the_shipping_ratio_and_ratio_1_keeps_every_slot`). So nothing in the design changed, and Amendment 2's discard rule does not apply.
- **Expected outcome.** The early look at 20 is expected to stop stage 1 with FAIL. It runs anyway for two reasons: it is the preregistered verdict, which the pilot is not; and its FULL-CHK rows are the reference that A2 reuses (section 6, item 3).
- **What is run.** The block-512 control runs after stage 1 as in Amendment 1. Stage 2 (decisions) is not started by this amendment. Whether it runs, and on what, is set after the stage-1 verdict.
- **A2.** A2 is preregistered separately before any A2 row, as section 6 requires.

## A2: keep the most recent blocks exact (preregistered 2026-10-15, before any A2 row)

What has been seen:
- The A1 pilot rows: 6 trajectories, shipping-arm retention 10 of 118. By distance: 0-4096 2/41, 4096-8192 3/26, 8192-16384 3/33, over 16384 2/18. POOL16K on the same trajectories: 34/36, 23/25, 30/31, 1/19.
- D1: ratio 1.0 gives 38 of 38.
- Nothing from A1 trajectories 7 and up. The A1 proper run is in progress, and nobody has read its retention.
- Why A2 is next: section 6 item 3 makes A2 the next rung after a FAIL. Its preregistered branch applies. On the pilot, retention under 8192 tokens is 5 of 67 (7.5%), against POOL16K's 57 of 61 (93%). So the loss is recency, and A2 first tests keeping the most recent blocks exact.
- The project owner chose this A2: two arms, with the equal-memory arm as the primary.
- A2 starts after A1's stage-1 verdict, because it reuses A1's FULL-CHK rows. If A1's stage 1 does not stop with FAIL at the early look, A2 is not run, and this section is reconsidered first.

### Question

A compacted block in the pilot recalls about as many verbatim facts per kept row as an exact block does: 10% of the rows, about 10% of the facts. Can the same 16,384 rows as POOL16K, split into an exact recent window plus the compacted older blocks, recall more than POOL16K, which drops what falls outside its 16K?

### Arms

Both arms use the A1 shipping arm with one change.
- **What stays the same:** the 640-token blocks by stream position; each block compacted once (64 slots, ratio 0.10, the FUNI cell, a local fit on the block's own keys and values with that block's own prefill queries); the Gated DeltaNet state carried through; and the open tail exact.
- **The one change:** the last W sealed blocks are kept **exact**. A block's compacted form replaces its exact form only when the block leaves the window. The compacted form is computed once, at seal time, from that block's own prefill (the same local fit as A1). A block is never re-compacted.
- **What the block prefill attends to:** later blocks are prefilled behind what the cache holds at that moment, the compacted older blocks plus the exact window. An engine with query capture at prefill (M3) can do the same, by compacting at seal and keeping both forms of a block until it leaves the window, or by keeping its queries until then.

| Arm | Window W | Exact tokens | Rows at 52,224 tokens (81 blocks + 384 tail) | Role |
|---|---|---|---|---|
| `WIN18` | 18 blocks | 11,520 + tail | 18x640 + 63x64 + 384 = 15,936 (at most 16,384, like POOL16K) | **primary** |
| `WIN25` | 25 blocks | 16,000 + tail | 25x640 + 56x64 + 384 = 19,968 (more memory than POOL16K) | secondary |

W = 18 is the largest window whose rows at the engine's 52,224-token limit stay within POOL16K's 16,384 (W = 19 gives 16,512). The owner was shown "19 blocks, 12,160 tokens" for the primary. That miscounted the tail and is corrected here. W = 25 is the window closest to 16K exact.

### Reused unchanged

- Everything in A1 sections 1 to 4: the harness, render, server tokenizer, stream, checkpoints, limits, parser, parity contract, determinism check and machine (lucebox6 integrated GPU, $0).
- The reference is A1's FULL-CHK rows. A2 runs no reference for trajectories A1 already ran. If A2 continues past its early look and A1 stopped at 20, A2 runs FULL-CHK on trajectories 21 to 60 with the same code and the same flags.
- A2 asks only the probes the reference got right, as A1's shipping arm does.

### Code and CPU tests (before any A2 row)

`r2_core.py` gets a window parameter on the block generator (`gen_fblk(..., window=0)`). Window 0 is the A1 shipping arm. The driver gets arms `WIN18` and `WIN25`, with rows in `WIN18.jsonl` and `WIN25.jsonl`. Tests on the tiny hybrid (`tests/test_a1.py`, block 24):
- (a) **Window 0** is `torch.equal` to the current `gen_fblk` at every checkpoint.
- (b) **A window of at least the number of blocks** equals FULL-CHK at every checkpoint: keys and values to 1e-4, and the logits of a following suffix to 1e-3.
- (c) **Any other window:** at a checkpoint, each block older than the window holds exactly the triples the local fit produced at its seal time, each window block holds its exact keys and values, and the slot count per full-attention layer is 64 x (older blocks) + block x W + tail.
- (d) **The row arithmetic** in the table above, for 52,224 tokens.

### Statistics and decision rule

- **Statistics** are as A1 section 5 (`r1b_stats`, the trajectory as the cluster), for each arm. Retention by distance and by probe type are reported for both arms, next to POOL16K's.
- **Early look.** As A1 Amendment 1: one Haybittle-Peto look at 20 trajectories, at 99.9%, on the **primary** only.
  - If the 99.9% lower bound is above 75.6%, the result is PASS. If the 99.9% upper bound is below 75.6%, the result is FAIL. Either way, A2 stops.
  - Otherwise both arms run all 60 trajectories, and the primary is decided at 95% under A1 section 6's rule.
  - INCONCLUSIVE with the point at or above 75.6%: the design ships on the point estimate with the risk written down, and the engine's agentic gate decides. The 120-trajectory extension is not used in A2.
- **Primary `WIN18`:**
  - **PASS:** the 2.0b design becomes "exact recent 18 blocks + compacted older". The plan's order after M5 holds, with the window added to M6 and M8.
  - **FAIL:** under section 6 item 3, the compaction track stops after M5, and whether 2.0b ships without compaction goes to the project owner.
- **Secondary `WIN25`:**
  - It gets the same interval and is reported as not memory-matched. It cannot rescue or fail the primary.
  - If `WIN25` passes while `WIN18` does not, that is reported to the owner as a finding: compaction adds recall only with more memory than the pool.
- **Prediction, from the pilot distances only (not part of the rule).** Facts inside the window at about A1's exact rate, and facts outside it at the pilot's compacted rate of about 10%. `WIN18` is projected at about 71% against POOL16K's 79% on the pilot trajectories. `WIN25` is projected at about POOL16K plus 1 to 2 points.

### Time

- The window arms build like the shipping arm, about 140 s per trajectory. They answer behind about 12K to 16K exact rows, at about 10 s per probe for about 20 probes.
- That is about 340 s per arm per trajectory. Both arms on 20 trajectories take about 3.8 h. All 60 take about 11 h, plus FULL-CHK on trajectories 21 to 60 if A1 stopped at 20 (about 8 h).
- Machine: the lucebox6 integrated GPU, at $0.

## E1: router recall of the target-QK index at prefill end (diagnostic)

Preregistered 2026-10-15, before any E1 row. This is a diagnostic. It decides nothing and changes no default behaviour. It informs the design of a router that pages chunks back in by score.

### Question

At the end of prefill, before the first decode token, does luce_server's existing target-QK residency index (`LUCE_KVFLASH_POLICY=qk`, `server/src/common/kvflash_qk.h`) rank the chunk that holds the source of a recalled fact inside its top k? Today the reselect runs only during decode, every tau = max(64, history/45) generated tokens, so a probe answer of at most 64 tokens never triggers it. E1 measures what the index would pick if it ran once at prefill end.

### Binary and branch

- A new opt-in env var `KVFLASH_QK_DUMP=<dir>` on a branch `kvflash/2.0r-e1-router-recall`, created from `kvflash/2.0a-E-turn-checkpoints` (09bde623), the 2.0a stack that B0 ran on. With the variable unset, behaviour and graph are unchanged (unit-tested).
- The dump writes, at the end of every prefill and before the first decode token, one JSON line per request: a prompt hash, `n_chunks`, the score of every sealed chunk, and which chunks are resident.
- The B0 binary md5 was 13c65b5cdce446f00f348a4101432de9. E1 uses a rebuilt binary from the branch above. Its md5 and the ggml .so md5s are recorded in the evidence folder.

### Probe set

- All 951 retained B0 probes (`optimizations/kvflash/evidence/agentic/B0_3090/retained.json`, tasks `tasks_all60.jsonl` with `src_tok`, `cut_tok`, `distance_tok`, `answer`).
- Primary subset: the 207 probes whose fact is older than 16,384 tokens (`distance_tok` > 16384). POOL16K recalled 12 of these. The other 744 (707 recalled) are the secondary subset. The 951 together are reported as "all".

### Run

- Machine: the 3090, under `flock /mnt/kimi-k3/build/3090.lock`, one 27B server at a time, GPU confirmed idle first.
- B0's POOL16K server flags: `--kvflash 16384`, `KVFLASH_STATE_LADDER=1 KVFLASH_LIVE_RESTORE=1`, same GGUF, `--max-ctx 52224`, temperature 0, seed 1, same chat template. Added: `LUCE_KVFLASH_POLICY=qk` and `KVFLASH_QK_DUMP`.
- Replay: the B0 harness (`optimizations/kvflash/agentic_eval/`), probes only, same trajectories and same prompts.
- Disk KV cache: off (`--no-kv-cache` or a fresh per-run `--kv-cache-dir`), with `disk_hit=false` confirmed.

### Metric

- A fact is **hit at k** if any chunk (64 tokens) that covers the token range [src_tok, src_tok + gold length in tokens] is among the top k chunks by score. Gold length is the tokenized `answer`.
- **recall@k** for k in {8, 16, 32, 64, 128} chunks, over the primary subset, the secondary subset and all.
- The **rank distribution**: the best (smallest) rank of any covering chunk, as a histogram and as median and quartiles.
- The number of probes whose source chunk was already **resident** at prefill end (under LRU most of the 16K window is resident). Recall is also reported over the non-resident probes alone, since a resident hit costs nothing.
- Two query variants, if both are feasible:
  - (a) the query of the last prefill token;
  - (b) the max score over all of the question's prefill tokens.
  Variant (b) needs per-token prefill queries. If the engine captures only one query on the prefill graph, only (a) is run and (b) is reported as not feasible, with the reason.
- Chance level is reported next to each recall: k divided by n_chunks, averaged over probes.

### What this does not decide

No threshold is set and no pass or fail applies. The numbers go to the design of the router (does a prefill-end score find old facts, and at what k). They do not change a default, do not replace B0's POOL16K results and are not an accuracy result: E1 does not check that the model answers correctly once the chunk is paged in.

### Parity notes

- This is a within-engine measurement, not an engine comparison, so bench-engine-parity does not apply as a comparison. The workload is B0's own, with B0's flags.
- Different from B0: the residency policy (qk instead of lru) and the dump. Because the policy differs, the engine's recall of the probe answer is not compared with B0's and is not reported as such.
- Scores are read once per request at prefill end. Reselect at decode time is not exercised (answers are at most 64 tokens), as in B0.
- Median, mean, best and min are reported for rank. Decode and prefill speeds are not the subject; the run records the three speeds only as provenance.

## Diagnostics E2, E3, E5 before A2 (2026-10-15, before any row)

Context: the A1 pilot showed compacted blocks keep about 8.5% of verbatim facts (10 of 118). The design under test is one where the exact KV lives in host RAM, the compacted blocks act as summary and index, a router picks the blocks a question needs, their exact KV is paged in, and their compacted slots are masked. These three diagnostics ask whether that can work. They run on lucebox6 (`HIP_VISIBLE_DEVICES=1`, $0), **after A1 has exited and before A2**, on the 6 pilot trajectories, probes only, with the pilot's FULL-CHK rows (`~/r1e/out/a1_pilot/FULL.jsonl`, read only) as the reference and the same reference-correct probes the pilot asked (118). **They decide nothing**, change no rung or bar, and inform the retrieval design only. A2 runs after them, unchanged.

### E3: oracle exact source block

- **State.** The A1 shipping state, built by `gen_fblk_window` with window 0 (the same operations as the A1 shipping arm; checked below). Before each sealed 640-token block is compacted, its exact keys and values are kept (about 64 KiB per token).
- **Intervention.** For each probe, the blocks that hold the fact are swapped back to exact at question time: the block's compacted entry is **replaced** by its exact K/V (never both). Everything else is as in the shipping arm: other blocks compacted, the open tail exact, the DeltaNet state as carried. The probe is answered as in the pilot (`answer_ids`, 64 tokens, the harness scorer).
- **Which blocks (two variants, both reported).**
  - `E3`, located: the first occurrence of the gold string in the stream tokens from the probe's `src_tok` (the start of the source tool message) to the cut; the blocks overlapping that token span. If the gold string is not found in the span, the probe falls back to the interval below and is counted as not located.
  - `E3SPEC`, as first specified: every block overlapping [`src_tok`, `src_tok` + gold length in tokens]. `src_tok` is the start of the source message, so for a fact deep inside a long message this variant can swap the wrong block; it is reported for that reason, next to the located variant. When both variants swap the same blocks the answer is computed once.
- **Output.** Retention against the FULL-CHK reference on the retained probes, with the pilot report's tables: overall (bootstrap interval, trajectory as the cluster), by distance, by probe type; next to the pilot's shipping-arm numbers on the same probes. Number of blocks swapped and located share.

### E2: router recall

- **State and query.** The E3 state and the same question prefill. For each probe the question suffix tokens' queries are captured on the 16 full-attention layers (24 query heads, post-norm, post-RoPE, under the shipping cache).
- **Two routers**, each scoring every sealed block at the cut (the open tail is not a block):
  - (a) compacted router: the block's share of attention mass from its compacted slots, `sum over its slots of exp(q.k/sqrt(d) + beta)`, normalised over all blocks (max-subtracted for stability), mean over layers and query heads;
  - (b) the engine's mean-pooled-key cosine, computed on the **exact** keys of each 64-token chunk of the block exactly as `kvflash_qk_chunk_scores` (`server/src/common/kvflash_qk.h`): post-RoPE keys summed over the chunk and L2-normalised with the same `+1e-6`, the query normalised with the same `+1e-6`, cosine, max over the query heads of the KV group's heads (all 24 heads, each against its KV head), mean over layers; the block score is the max over its 10 chunks.
  - Both are aggregated over the question tokens two ways: max over tokens, and the last token only. Four score vectors per probe.
- **Metric.** The rank of the best-ranked source block (the located blocks of E3) among the sealed blocks; recall@k for k in {1, 2, 4, 8, 16}; the rank distribution (median, mean, quartiles). Reported for all retained probes and for those more than 16,384 tokens back. Probes whose source falls in the open tail, or that were not located, are counted and excluded from E2.
- **Reading.** The engine's own router (b) is the one the retrieval design would ship; (a) shows how much a router built from the compacted slots alone could do. E1 (the engine's index on the 3090) is a separate measurement.

### E5: shipping arm at ratio 0.25

- The A1 shipping arm with `--ratio 0.25` (64 slots become 160 per block; the option exists since D1), on the same 6 trajectories and probes, scored against the same reference rows. Output: retention with the same tables, next to the pilot's 8.5% at ratio 0.10 and D1's 38 of 38 at 1.0. It shows how retention moves with the slot budget.

### Reproducibility check

- In the E3/E2 pass the shipping arm is re-answered from the unswapped state and each answer's token ids must equal the pilot's `SHIP.jsonl` row for the same task. This is done for the first trajectory (about 950 s of answer time for all six; extended to more trajectories only if time allows). A mismatch is recorded in `ship_check.json` and reported, and E3/E2 numbers are then read with that mismatch stated. The same model, code copy, flags and `--no-kv-cache` equivalent (this driver has no disk cache) hold as in the pilot.

### Machine and rules

- lucebox6, `HIP_VISIBLE_DEVICES=1`, the A1 launch memory gate and watchdog (host used under 120 GiB, swap growth under 1 GiB). Outputs `~/r1e/out/e3e2/` and `~/r1e/out/e5/`; the pilot rows are copied, never edited. E3/E2 and E5 run one after the other, never together.
- No result of E1, E2, E3 or E5 changes A1's or A2's preregistered rule. A diagnostic that is read as a bar would need its own amendment first.

## Diagnostics E6, E7 (2026-10-15, before any row)

Context: Codex's read-only root-cause audit of the A1 pilot (`/mnt/kimi-k3/build/codex_a1_rootcause.md`) found no plumbing bug. The paper's only needle test (RULER, Appendix F.4: Qwen3-4B 90.2% at 12.5% cache against 93.9% full) uses repeat-prefill queries and keeps the system and template tokens exact. The shipping arm does neither: it fits every 640-token block on its own context-prefill queries, and it compacts every token, including the rendered system message (about 11.5K characters) and the tool schema (about 14.7K characters). E6 and E7 each remove one of those two differences. They decide nothing and change no A1 or A2 rule.

### Common rules

- Same 6 pilot trajectories and the same 118 reference-correct probes as the pilot, the pilot FULL-CHK rows as the reference (copied unedited into the output directory), the same tables as E2/E3/E5: retention overall (bootstrap interval, trajectory as the cluster), by distance, by probe type, next to the pilot's shipping arm (8.5%) on the same probes.
- The shipping arm otherwise unchanged: 640-token blocks, each compacted once to 64 slots (ratio 0.10) at seal time, DeltaNet state carried, open tail exact, answers by `answer_ids` (64 tokens, harness scoring). Both arms are options of `gen_fblk_window`; with the option off it is the shipping arm (checked by the code tests).
- lucebox6, `HIP_VISIBLE_DEVICES=1`, the A1 launch memory gate and watchdog. Outputs `~/r1e/out/e6/` and `~/r1e/out/e7/`. The runs go one after the other, after E3/E2 and E5 and before A2: E3/E2, E5, E6, E7, never together.

### E6: repeat-prefill queries at ratio 0.10

- **Change.** At each seal, take a clone of the cache the block was just prefilled into (earlier blocks compacted, the current block exact, the DeltaNet state after the block). On the clone, teacher-force the repeat instruction followed by the block's 640 tokens, at the logical positions that continue after the block. Capture the post-RoPE queries of every token of that pass on the 16 full-attention layers (24 query heads). Those queries replace the context-prefill queries in key selection, beta fit and value fit of that block. Then the clone is dropped: the generator's cache and DeltaNet state never see the repeat, so the next block is prefilled behind the same state as in the shipping arm (except that earlier blocks were fitted differently).
- **Instruction text, from the am-compaction source** (`compaction/query_generation/conversation_specs.py`, spec `repeat`: `conversation_starter="Repeat the previous context verbatim."`, `prefill_with_article=True`; and `self_study.py`, where the starter is rendered as one user turn through the chat template with `add_generation_prompt=True` and `enable_thinking=False`, followed by the prefilled article as the answer, and the queries of that whole suffix, prompt tokens and article tokens, are extracted behind the context cache). Read only; the source is not run, imported or copied. Rendered as the model's own tokenizer chat template gives it, expected `<|im_start|>user\nRepeat the previous context verbatim.<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n`; the run logs the exact string and token count and they are quoted in the report. Following the source, the captured queries are those of the instruction tokens plus the repeated block tokens (about 20 + 640 queries per block).
- **Differences from the paper stated.** The paper repeats the context to be compacted as a whole; here the repeat is per 640-token block behind the already compacted earlier blocks, because that is the shipping arm's unit. The repeated text is exactly the block's tokens. The fit's other inputs (block keys and values from the context prefill, slot count, algorithm and budgets) are unchanged.
- **Cost.** One extra forward of about 660 tokens per block.

### E7: system and tool prefix exact

- **Change.** Every sealed block that overlaps tokens [0, end of the rendered system message) stays exact, rounded up to block granularity; the end is one past the first `<|im_end|>` of the rendered stream (the template puts the tool schema inside the first system message). The kept blocks are a contiguous leading range, held exact with beta 0 and not fitted. All other sealed blocks are compacted at 0.10 exactly as in the shipping arm, each prefilled behind the cache as it stands (exact prefix plus compacted earlier blocks). A contiguous leading range does not trigger the noncontiguous partial-reconstruction bug Codex noted at `per_layer_head_on_policy.py:492` (uncompacted positions inside a noncontiguous compaction range disappear); no repo code path is used for the exact blocks.
- **Reported.** The exact-token count per trajectory (system end token, exact blocks, exact tokens), so the extra capacity is visible.
- **No matched-capacity control for now.** E7 holds more rows than the shipping arm (the exact prefix is not part of the 10% budget). A control with the same extra rows spent elsewhere (for example one more exact recent window) is skipped. It is added, with its own prereg section before any row, if E7 shows a gain.

### What these do not decide

No threshold applies and neither result changes the A1 or A2 rule. If E6 recovers retention, query mismatch is implicated and a repeat-prefill fit becomes a candidate for its own preregistered test; if E7 recovers it, the control-prompt tokens are implicated. Both are read next to E5 (capacity) and E3 (oracle exact source block). Decode and prefill speeds are recorded as provenance only.

### Code checks before any row (CPU, build host)

Stub tests (`lucebox/a1/test_a1_e67.py`): the clone does not mutate the generator's cache or state (the stub scribbles on its input in the repeat pass and the final state still equals the shipping arm's); the repeated tokens are teacher-forced from position (block end) onward; the fit receives the captured repeat queries and not the context ones; E7 keeps exactly the leading blocks exact and compacts the others; zero exact blocks equals the shipping state. Real-model tests (`modal/am_qwen35/tests/test_a1.py`, run on lucebox6 before the launch): E6 differs from the shipping arm in the fitted tensors; E7 with a zero-length prefix equals the shipping arm and a two-block prefix holds the FULL-CHK rows.

## Diagnostic E8: selection-only twin of the shipping arm (2026-10-15, before any row)

Context: arXiv 2608.00902 reports that selecting original keys and values matches or beats Attention Matching under imperfect query proxies. The shipping arm fits every block on its own context-prefill queries, which are an imperfect proxy (E6 asks about that from the query side). E8 asks whether the fit (the bias beta and the fitted values) earns anything over plain key selection on these agentic traces. It decides nothing and changes no A1, A2 or A3 rule or bar.

- **Arm.** The A1 shipping arm (F-BLK-UNI, 640-token sealed blocks, ratio 0.10, 64 slots per block and head, open tail exact, DeltaNet state carried) with one change: each block keeps the 64 original key and value rows the shipping fit selects, copied verbatim. The bias is 0 and no value is fitted. The selection is the shipping fit's own: the repo's HighestAttnKeys scoring (rms over the fit queries, `highest_attn_keys_rms_nnls2_-3_3_lsq_on-policy`, the class in `r1b_core.CELLS["FUNI"]`) ranks the keys and returns its indices `idx`; E8 wraps that class (`a1_core.select_only`, `lucebox/a1/a1_core.py`, swapped into `Runner.method.algorithm_class` by `a1_run.py --select-only`, the switch point `am_fixture_dump.py` already uses) and returns `K[idx]`, `V[idx]`, a zero bias of beta's shape and dtype, and the unchanged `idx`. The wrapped class still runs its fit internally; the result is discarded. The selection is therefore the one the shipping arm makes, by construction, not a reimplementation of it. `am_fixture_dump.py` (M4 fixtures) shows the selection equals `topk` of the key scores in rank order.
- **Everything else as E5.** The 6 pilot trajectories and the 118 reference-correct probes (md5 `93a29de59549954ebd9d46ffd9214047` for the reference); the pilot FULL-CHK rows are copied unedited into the output directory and `a1_e_report.py` asserts their md5 equals the pilot's; the same tables as E5: retention overall (bootstrap interval, trajectory as the cluster), by distance bin, by probe type. Reported next to the pilot's 8.5% at the same ratio, E5's 27.1% [19.4, 35.7] at ratio 0.25 and E3's 83.9% [77.5, 91.1]. Decode and prefill speeds are provenance only.
- **Reading.** Retention above 8.5% with non-overlapping intervals says the fit hurts on these traces; retention at or below it says the fit earns nothing measurable over selection (either way the engine's M5 fit is not removed or kept on this evidence alone). No threshold applies.
- **Order of runs on lucebox6** (`HIP_VISIBLE_DEVICES=1`, one at a time): E6, the A3 tests, E7, A3 stage 1, then E8 (`lucebox/a1/a1_launch_e8.sh`, output `~/r1e/out/e8/`, which refuses to start while any of e3e2, e5, e6, e7, a3sel or a3 is running).
- **Stub tests** (`lucebox/a1/test_a1_e8.py`): the selected indices equal the fit's own, the kept keys and values are byte-equal to the source rows at those indices, the bias is all zero with beta's shape and dtype, and the fitted tensors are discarded (list and tensor `idx`). Tiny-hybrid test (`modal/am_qwen35/tests/test_a1.py`, run on lucebox6 before the launch): the E8 state has the shipping arm's shapes, bias 0, and block 0's kept rows are rows of FULL-CHK's exact block 0.

## Diagnostic E9: the exact-mass router (2026-10-16, before any row)

Context: the AM fit sets beta so that a compacted block's attention mass matches the exact block's. E2's `a_last` therefore approximates the exact attention mass the question puts on each block, and the same mass computed on the exact keys is `a_last`'s ceiling. If exact mass ranks at least as well, routing needs no compacted index (no beta kernel, no value fit for routing; the exact pool plus host-side scoring of the old exact keys). E9 measures that. It decides nothing and changes no A1, A2 or A3 rule or bar.

- **Routers, all on identical state.** The 6 pilot trajectories, the 118 reference-correct probes, the same cuts, the shipping state (window 0) as in E2, no answer generated. Per probe one query capture of the question suffix (`r2_core.extend`, `want_queries`) feeds: `a` (E2: compacted slots with beta, `a1_e.router_mass`), `x` (new: the same `router_mass` on each block's **exact** 640 keys, the rows `on_block` keeps as in E3 and A3, with zero beta, so each block's share is the softmax mass of its exact keys among the listed blocks) and `b` (E2: kvflash_qk cosine). Variants `_last` (last question token) and `_max` (max over question tokens). Normalisation is over the listed blocks, as `a_last` does it.
- **Ranks.** (1) E2's: over every sealed block of the cut, the best rank of the located source blocks (E3's located span), routers `a_max a_last x_max x_last b_max b_last`: recall@1/2/4/8/16, median rank, the over-16K subset, and per probe the `x` rank against the `a_last` rank (counts better, equal, worse, and the list). (2) A3's, per W in {2, 6}: among the candidate blocks only (never the exact prefix P, never the window W; P from the rendered system message as in A3), normalised over the candidates as A3's router is, with the k that `a1_a3.choose_k` gives at the cut under the cut's reserve (longest question suffix plus 64 new tokens): the share of probes whose source block is a candidate that falls inside the fetched top k, for `a_last`, `x_last`, `x_max` and `b_last` (probes whose source block is not a candidate are counted separately and excluded from the denominator).
- **Reference and tables.** The pilot FULL-CHK rows are copied unedited (md5 checked by `a1_e_report.py`, which refuses a different file); the report is the additive `E9_exact_mass_router` section of `a1_e_report.py`, whose earlier arguments are unchanged. Next to E2's `a_last`/`b` on the same probes.
- **Reading.** If `x_last` recall matches or beats `a_last` (the intervals and the paired counts are descriptive, no threshold), routing does not need the compacted index. If it is clearly worse, beta carries ranking information the exact mass lacks at this granularity. No result changes a rule.
- **Cost.** One build pass per trajectory (E3/E2: 142 s each, about 15 minutes in all) plus per probe the compacted router (E2: about 1 s), the exact router over up to 57 blocks (ten times the keys, estimated 5 to 10 s) and the two candidate-only repeats (about 2 x 7 s): about 20 to 30 s per probe, 118 probes, so about 40 to 60 minutes of routing. **Expected about 1 hour, at most 1.5 hours**, plus the model load. Peak memory is E3/E2's (exact keys of every block, about 1.2 GiB more for the concatenated keys and fp32 logits of one layer). The runtime is an estimate; the first trajectory's `progress.jsonl` line gives the real figure.
- **Order and machine.** lucebox6, `HIP_VISIBLE_DEVICES=1`, one job at a time, after A3 stage 1 and before A3 stage 2 (`lucebox/a1/a1_launch_e9.sh`, output `~/r1e/out/e9/`, refuses to start while any of e3e2, e5, e6, e7, e8, e9, a3sel or a3 runs). Code: `lucebox/a1/a1_e9.py`; it reads `a1_a3.py`, `a1_e.py`, `a1_e67.py` and `a1_core.py` and changes none of them.
- **Stub tests** (`lucebox/a1/test_a1_e9.py`): exact-mass ranking with zero beta equals a brute-force softmax mass on random tensors (all blocks and a subset); a planted high-mass block ranks first; ranks map source ids onto the listed candidates and ignore sources outside them; the budget row takes k from `choose_k` and reports inside-k; the report table counts better, equal, worse and inside-k.

### E1 deviations (2026-10-15, written after the run; no PREREG-time rule is changed, only what the data forced)

- **Base commit.** The branch is based on 0697848b, not 09bde623: the B0 binary (md5 13c65b5c...) was built from this worktree's `server/` at 0697848b, and 09bde623 is a different lineage.
- **Hit rule.** `src_tok` in the tasks is the start of the assistant call that produced the tool message, not the fact. The primary hit uses the fact's own token span (the answer located in the tool message). The rule as written is reported as `lit`, and the whole source message as `msg`.
- **Variant (b).** Feasible only with a new capture (`q_cap_all`). It folds the per-token queries of every prefill batch ending within the last 512 prompt tokens, because the final batch of a probe holds only the tokens after the last assistant header.
- **Replay.** Probes only, so the request history differs from B0's; answers match B0's POOL16K on 783 of 951 probes (old facts 12 and 12 recalled, within 16K 688 and 707). Details in `evidence/agentic/E1_router_recall/README.md`.

## A4 R-LRU offline pilot (2026-10-16, before any row)

Written before any A4 row exists. Paths are relative to `optimizations/kvflash/`. "KV" is the key-value cache. "R-LRU" is the candidate engine design: the engine's least-recently-used (LRU) pool, meaning an exact, contiguous recent window, plus k whole 640-token blocks recalled exact from host memory and ranked by exact-key attention mass; there is no compaction and no GPU index. "B0" is the 3090 baseline run of the agentic harness and POOL16K its exact 16,384-row pool arm. "A3", "E2", "E3", "E7" and "E9" are the rungs and diagnostics above. "FULL-CHK" is full KV built block by block (the A1 reference). "DeltaNet" is the Gated DeltaNet recurrent layers of Qwen3.8, whose state is carried through, never compacted. "Sink" is the first rows of the cache that the engine never evicts.

What has been seen:
- **A3 stage 1** (`evidence/agentic/A3/sel/`): window 2 blocks, k filling the budget, router `a_last` on compacted blocks: 95 of 118 (80.5%). B0 POOL16K on the same trajectories: 88 of 111. A3 lost 13 probes to router misses at the longest cuts.
- **E3** (`evidence/agentic/E3E2/`): the oracle swap of the source block: 99 of 118.
- **E9** (`evidence/agentic/E9/`): on E2's shipping state, exact-key mass `x_last` ranks the source block better than `a_last` (recall@1 36.4% against 18.6%, median rank 2 against 4). That is not A3's state, and not an LRU state.
- **The engine's pure LRU** (`evidence/agentic/prefill_reselect/README.md`, OFF arm, 3090): 700 of 951 overall and 688 of 744 within 16K. Today's engine LRU does not keep the system and tool prefix. It protects only the first 64 tokens (one chunk) and evicts everything else least-recently-used; at a 36K cut its resident pool is chunk 0 plus the newest 255 chunks.
- **`rlru_est.py`** (`evidence/agentic/prefill_reselect/diag/`): an estimate, not a measurement, that a recent window plus 4 routed blocks reaches 95.4 of 105. It ranks over all blocks on E2's state.
- Nothing from any A4 configuration. Nobody has measured an exact-mass router on an LRU state, or an LRU state with a fetch, on the real model.

### Status: a pilot

A4 is a pilot. It decides nothing and changes no rule or bar: the A1 rule and the 75.6% bar, A3 stage 2, the A2 arms and E1 to E9 all stand. No number here is read as a result for any of them. Its two questions are descriptive.

1. **Do `x_last` ranks hold on an LRU state?** E9 ranked over E2's compacted state, where every block is present. In R-LRU the routing query is formed over a state in which the old blocks are absent. The question is whether the exact-mass rank of the source block among the evicted blocks is still as good.
2. **What does R-LRU retain at the fixed 16,384-row budget**, against an in-harness pure-LRU baseline (`A4_K0`)? That baseline removes the cross-engine comparison with B0.

### Arms (probes only, the 6 pilot trajectories, the 118 reference-correct probes)

| Arm | Exact prefix | Sink | Fetch k | Role |
|---|---|---|---|---|
| **A4_K4** | none | rows 0 to 63 | 4 | primary |
| **A4_K0** | none | rows 0 to 63 | 0 (pure LRU) | in-harness LRU baseline, the analogue of POOL16K |
| **A4_K4P** | P exact (A3 and E7's system and tool prefix, 10 blocks) | inside P | 4 | reported next to the others, never primary |

`A4_K4` and `A4_K0` share one build pass per trajectory. `A4_K4P` has its own build pass: its window is smaller by P, so its blocks are prefilled behind a different cache. P is the sealed blocks overlapping the tokens [0, end of the rendered system message), exactly as in A3 and E7 (`e67.system_end`, `e67.prefix_blocks`; 5,921 tokens, 10 blocks, 6,400 rows in every B0 trajectory).

### State at a cut (built over the trajectory's token stream by `a1_a4.gen_lru`)

- **Window.** A contiguous run of the newest whole 640-token sealed blocks, exact, plus the open tail (under 640 tokens), exact. Block granularity: the window edge differs from the engine's 64-token-chunk LRU by under 640 tokens.
- **Sink (A4_K4, A4_K0).** Rows 0 to 63 of block 0, exact and always resident, a row-level sink as in the engine. They are the first 64 rows of block 0 as prefilled, with nothing in front of them. While block 0 is a whole resident block these rows are part of it. After block 0 is evicted only these 64 rows stay. No row is ever counted twice: the state holds the whole blocks, the sink rows only when block 0 is not whole, and the tail (`a1_a4.state_rows`).
- **Evicted blocks.** Every sealed block that is not in P or the window. It is neither attended nor compacted. Its exact K/V, as computed while it was resident (behind the sink or P and the window of that time), is kept on host (CPU memory, `HostStore`, 64 KiB per token). A block that holds the sink rows is still a candidate for its other rows: block 0's candidate rows are rows 64 to 639 (576 rows), its keys and pooled chunks are scored without the sink rows, and fetching it brings back rows 64 to 639 behind the resident sink rows, giving one whole block 0. A fact located wholly inside rows 0 to 63 counts as resident, not routable.
- **DeltaNet state.** Carried through every token of the stream, evicted or not, as everywhere in the harness. Eviction does not touch it.
- **Positions.** Every token keeps its true position (rotary position embedding at the true index) after blocks are evicted, as in the compacted states.
- **Open tail.** Its K/V were computed behind the build window, before any cut-time eviction. This is what an engine does: tokens are computed while the older rows are still resident, and eviction follows when room is needed.

### Build-time eviction (block-granular LRU)

Let P be the number of resident pinned blocks (0 for the sink arms; `min(p, j)` for A4_K4P), W the number of resident unpinned whole blocks, and `B` = 640. Before block j is appended, evict the oldest unpinned whole blocks until

    BUILD:   B * (P + W + 1)  +  64 * [block 0 is not whole]  +  (B - 1)   <=   16,384

where the +1 is the block being appended, `[block 0 is not whole]` counts the sink rows when block 0 was evicted (the block being appended counts as whole when it is block 0), and `B - 1 = 639` reserves the largest open tail that can follow. It is `state_rows(resident - evicted + {j}, sink, tail = B - 1) <= cap` in `a1_a4.py`. If P pinned blocks plus the new block plus the tail do not fit, the run aborts. The question and the answer are not reserved at build time (they are not known); the cut inequality adds them.

Consequences, stated before any row: the build keeps a window of at most 24 whole blocks after an append (sink arms: 24 blocks, 15,360 rows, plus 64 sink rows once block 0 is gone, plus a tail of up to 639: at most 16,063 rows; A4_K4P: 10 pinned plus 14 window blocks). Nothing is evicted while the context is at most 24 blocks (15,360 tokens): up to there every cut holds the full-KV state, with no candidates. Evictions start at 25 sealed blocks.

### Cut-time window, candidates and k

The reserve is A3's: `reserve = q + max_new` (q of the cut's longest question suffix, `max_new` = 64), so that one k serves every probe of a cut. At the cut, evict further oldest window blocks (the build window can only shrink, never regrow) until the largest window `w` for which, with `k = min(k_req, candidates)`,

    CUT:   B * (P + w + k)  +  64 * [block 0 is not in P or the window]  +  tail  +  reserve   <=   16,384

i.e. `sink + window + tail + k * 640 + reserve <= 16,384` (`a1_a4.cut_plan`). Candidates are all sealed blocks outside P and the window. The bound charges every fetched block 640 rows and the sink rows whenever block 0 is not whole. If block 0 is fetched, its 64 sink rows were already resident, so the actual rows are 64 fewer than the bound (`a1_a4.final_rows`); the cap holds in every case. If even `w = 0` does not fit, the run aborts (`OverBudget`; as in A3, no exception path). `k_req` is 4 for A4_K4 and A4_K4P and 0 for A4_K0 (so A4_K0 keeps a larger window, up to the same cap). k is fixed, not budget-filling. When a cut has fewer than 4 candidates k is the number of candidates; with none, there is no routing pass and the state is the full-KV state.

The cap is asserted, not assumed: the routing pass (base rows + question) is checked against 16,384 before it runs; the answer state's rows equal the plan's (`final_rows`) and `rows + reserve <= 16,384`; and the answer runs under A3's `row_guard`.

Worked numbers (`a1_a4.py`; tail 0 with reserve 164, and tail 639 with reserve 364):

| Sealed blocks | Arm | Build window | Cut window | Candidates | k | Rows bound | Peak rows bound |
|---|---|---|---|---|---|---|---|
| 19 | K4, K0 | 19 | 19 | 0 | 0 | 12,160 to 12,799 | 12,324 to 13,163 |
| 24 | K4, K0 | 24 | 24 | 0 | 0 | 15,360 to 15,999 | 15,524 to 16,363 |
| 25 | K4 | 24 | 24 or 19 | 1 or 6 | 1 or 4 | 16,064 or 15,423 | 16,228 or 15,787 |
| 38 | K4 | 24 | 21 or 19 | 17 or 19 | 4 | 16,064 or 15,423 | 16,228 or 15,787 |
| 38 | K0 | 24 | 24 or 23 | 14 or 15 | 0 | 15,424 or 15,423 | 15,588 or 15,787 |
| 57 | K4 | 24 | 21 or 19 | 36 or 38 | 4 | 16,064 or 15,423 | 16,228 or 15,787 |
| 57 | K0 | 24 | 24 or 23 | 33 or 34 | 0 | 15,424 or 15,423 | 15,588 or 15,787 |
| 57 | K4P | 14 | 11 or 10 | 36 or 37 | 4 | 16,000 or 15,999 | 16,164 or 16,363 |

(K4P's build window counts unpinned blocks; its P is 10 more.) These are arithmetic facts about the design.

### Per probe

1. **Routing pass (A4_K4 and A4_K4P only; A4_K0 has k = 0).** The question suffix (the harness's question turn and generation prompt, the same token ids A1 appended) is prefilled under the cut's base state (sink or P, window, tail; before any fetch) and its queries are captured on the 16 full-attention layers (`r2_core.extend`, `want_queries`). It runs on the base state, not on the larger build state, because the build state plus the question can exceed the cap. The pass's KV is dropped at once.
2. **Router `x_last`.** For each candidate, the softmax mass of the question's last-token query on the candidate's exact keys (block 0: rows 64 to 639), zero beta, normalised over the candidates only, mean over layers and query heads: `a1_e9.exact_mass`, E9's function, reused. Ties go to the lower block index (`a1_a3.rank_candidates`). `x_last` alone decides the fetch.
3. **Fetch.** The top k candidates come back exact. The answer state is built by A3's insertion path: `a1_a3.a3_state`, i.e. `a1_e.oracle_triples` with every piece (sink rows, whole blocks in logical order) swapped in from the host copies, the DeltaNet entries and the open tail taken from the build state. The fetched blocks are the host copies, byte for byte.
4. **Answer.** The routing-pass KV is discarded; the question is prefilled again on the answer state and the probe is answered as in A1: `r2_core.answer_ids`, greedy, at most 64 new tokens, the harness scorer. When A4_K0's final state equals A4_K4's (no candidates at the cut), A4_K0 reuses A4_K4's answer, marked `answer_reused`; the inputs are identical, so greedy decoding gives the same ids.

### Logged for every routed probe, never part of a rule

- **Ranks of the source block among the candidates** (E3's located-span definition, as in A3; a rank is the best of the source blocks that are candidates): for `x_last`; `x_imend`, `x_ucontent`, `x_lit5`; `x_max`; and `b_last`.
  - `x_imend` (the "end - 5" router): the same mass at the query position of the `<|im_end|>` that closes the user message. The engine's prompt ends `...content<|im_end|>\n<|im_start|>assistant\n` with no empty think block, its end - 4 cut is just before `<|im_start|>`, and so its end - 5 query token is that `<|im_end|>`. It is defined from the template, not from an offset: the last `<|im_end|>` before the last `<|im_start|>` (the assistant header) of the question suffix (`a1_a4.imend_index`). A4's rendered suffix is `<|im_start|>user\n{question}<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n` (the B0 render, thinking off), so here the token is the ninth from the end, not the fifth. Each row logs its index, its distance from the end and the decoded token string.
  - `x_ucontent`: the mass at the last content token of the user message, the token just before that `<|im_end|>` (`a1_a4.ucontent_index`; one token earlier). Logged to separate the two.
  - `x_lit5`: the literal fifth token from the end of A4's suffix. With the think block it is the newline after `assistant`. Logged so a literal-offset reading can be matched too.
  - `x_max`: the max over the question tokens of each candidate's mass.
  - `b_last`: E2's kvflash_qk cosine (mean over layers of the max over query heads of the cosine between the last query and each 64-token chunk's pooled exact key, max over a block's chunks). It is available on this state because the pooled keys come from the exact blocks kept on host.
- Recall at 1, 2, 4, 8 and 16 (among probes whose source is a candidate), overall and for facts older than 16K; the share of those inside the cut's k.
- The numbers of probes whose source is resident (P, sink, window or open tail), a candidate, or not located.
- Per cut: window blocks at build time and at the cut, candidates, k, reserve, rows bound and peak rows bound (`A4_cuts_S.jsonl`, `A4_cuts_P.jsonl`); per probe: rows and peak rows actually held.

### Data and reference

The 6 pilot trajectories and the 118 reference-correct probes. The reference is the pilot FULL rows `~/r1e/out/a1_pilot/FULL.jsonl`, md5 `93a29de59549954ebd9d46ffd9214047`, copied unedited into the output directory and asserted by md5 both at the source and at the copy (`a1_a4.ensure_reference`; `a1_a4_report.py` asserts it again). The fixed universe is `a1_a3.PILOT_N` and `PILOT_PROBES` (6 and 118), checked by `a1_a3.check_universe`. A run with a different file stops.

### Report (`lucebox/a1/a1_a4_report.py`, additive; it changes no earlier report)

- Retention against the reference, per arm, with the bootstrap interval clustered by trajectory (`a1_core.boot_ci`, 10,000 resamples, seed 0, 95%): overall, by distance bin (0 to 4K, 4K to 8K, 8K to 16K, over 16K) and by probe type, each cell with its count and interval. Also split by whether the probe's cut evicted anything, with the sanity count that cuts without candidates reproduce the reference answers.
- The router tables of the two routed arms (ranks above).
- The paired `A4_K4` against `A4_K0` counts (both, only one, only the other, neither) with `r1b_stats.mcnemar_exact`; `A4_K4P` against `A4_K4` the same way.
- Next to: A3 stage 1 W = 2 (95 of 118), B0 POOL16K on the same trajectories (88 of 111) and the paired counts on the common probes, E3 (99 of 118), and the engine's pure LRU (700 of 951, 688 of 744 within 16K). The A3 and E3 numbers are recomputed from their row files when the report is given their directories; the others are quoted.
- **Reading.** Descriptive only; no threshold applies. Cuts of at most 24 blocks have no candidates, and every arm holds the full-KV state there (the sanity set, not evidence for R-LRU). The informative probes are those at cuts with candidates, so the report counts them separately. Intervals over 6 trajectories are wide.

### Differences from the engine and from B0 (stated, not tuned)

- The window edge is block-granular, under 640 tokens from the engine's 64-token-chunk LRU. The sink is 64 rows, as the engine's `sink_chunks = 1`.
- The build reserves a full tail (639 rows) for the open tail, and the cut charges 640 rows for every fetched block: up to about one block of rows can be left unused at a cut. The mean rows held are reported.
- A4 runs bf16 K/V and B0 runs the 4-bit q4_0 cache type; both count rows per full-attention layer (16,384), so the comparison is in rows, not bytes (A3's wording). The host archive stands in for B0's paged-out chunks.
- A4_K0 is the in-harness analogue of POOL16K, not the same machine: other precision, block granularity, and no prefix cache. The 3090 pure-LRU anchor (700 of 951, 688 of 744 within 16K) is quoted, never combined.
- Caching is symmetric across A4's arms and its reference (no prefix cache, no disk cache, everything built from scratch). It is not symmetric against B0.

### Parity contract

Model Qwen/Qwen3.8-27B bf16, the same files, for every arm and the reference; KV bf16; greedy with A1's limits (64 new tokens for a probe) and stop tokens; the rendered ids of each request are A1's; PyTorch scaled-dot-product attention (SDPA) as A1; one code copy whose md5s are recorded in `provenance.<id>.txt` and `manifest.json` (A4's files, which are `a1_a4.py`, `a1_a4_report.py` and `a1_launch_a4.sh`, plus `a1_a3.py`, `a1_e.py`, `a1_e9.py`, `a1_core.py`, `a1_e67.py`, `r2_core.py`, `r1b_core.py`); the same 118 retained probes in every arm; the cap 16,384 rows asserted at every cut.

### Cost, machine and order

- **Machine and order.** lucebox6 integrated GPU, `HIP_VISIBLE_DEVICES=1`, $0, no Modal job, one job at a time. **After A3 stage 2 ends.** `lucebox/a1/a1_launch_a4.sh` (output `~/r1e/out/a4/`) refuses to start while any of e3e2, e5, e6, e7, e8, e9, a3sel or a3 runs, or a1 python jobs are alive, runs under the A1 watchdog (`a1_watchdog.py`: stop, resumable, at 120 GiB used or swap growth over 1 GiB), the A1 launch gate and the 100 GiB per-process cap. Resume: rerun the launch; the manifest must match; a pass without its done marker is redone.
- **Runtime, derived from A3 stage 1 and E9** (estimates, not gates). A3 stage 1 took 88 minutes for 6 trajectories times 2 arms: 12 build passes of 181 to 215 s (with the compaction fit), 236 answers (1,285 s per 118 at a mean of 14.7K rows, 10.9 s each) and 236 routing passes (median 0.91 s). E9 took 19 minutes: builds of 132 to 148 s and all routing 205 s for 118 probes, 1.7 s per probe for three routers over all blocks plus two candidate repeats. A4 has 12 build passes with no fit and at most 24 blocks of attention: 120 to 200 s each, about 32 minutes. It answers up to 354 times (118 per arm, fewer where K0 reuses K4's answer), 11 s each: 54 to 65 minutes. It routes 236 times at about 3 s (pass 0.9 s plus exact mass over at most 48 candidates, E9's cost, plus the host-to-device copy of at most 1.2 GiB of keys per cut): about 12 minutes. **Expected about 1.7 hours, at most 2.5 hours**, plus the model load. The first trajectory's `progress.jsonl` line gives the real figure.
- **Peak memory.** A3 stage 1 allocated at most 59.4 GiB and the GPU reserved at most 88.0 GiB; E9 allocated 56.1 GiB. A4 adds at most about 6 GiB of KV-sized tensors at once (the build state, two base states, the answer state, the routing state, each at most about 1 GiB) and 1.2 GiB of candidate keys, so **expected allocated 58 to 63 GiB, reserved at most about 90 GiB** (A3's 88.0 plus allocator slack), under the 100 GiB cap. Host memory: the archive of one pass is 57 blocks times 42 MiB, about 2.4 GiB, freed per pass. The first trajectory's `mem.log` and `progress.jsonl` give the real figures.

### Code and CPU checks before any row

New files only. Nothing is modified in `a1_a3.py`, `a1_e.py`, `a1_core.py`, `a1_e67.py`, `r2_core.py`, `r1b_core.py` (the files A3 stage 2 checks by md5) or `a1_e9.py`.
- `lucebox/a1/a1_a4.py`: the generator that evicts instead of compacting (`gen_lru`, built on `r2_core.extend` and `sealed_blocks`), the pure rules (`state_rows`, `build_evictions`, `cut_plan`, `imend_index`, `ucontent_index`, `rank_row`, `source_status`), the host store and the driver. It reuses `a1_a3` (`a3_state`, `rank_candidates`, `row_guard`, the `PILOT_*` universe constants, the manifest, resume and reference helpers), `a1_e` (`oracle_triples` through `a3_state`, `locate_span`, `overlap_blocks`, `pooled_chunk_keys`, `qk_chunk_scores`, `best_rank`, `recall_table`), `a1_e9.exact_mass`, `a1_e67` and `a1_core`.
- `lucebox/a1/a1_a4_report.py`, `lucebox/a1/a1_launch_a4.sh`.
- `lucebox/a1/test_a1_a4.py` (stub tests, torch only, pure functions and a fake `extend`; 26 pass locally on the CPU): the state rows count the sink once; the build eviction keeps BUILD after every append, evicts oldest first, never evicts a pinned block, is minimal, and gives the 24-block windows above; it aborts when the pinned blocks do not fit; the cut plan reproduces the worked numbers, keeps CUT on a grid of 69 block counts, tails 0, 1 and 639, reserves 164 and 364 for all three arms, is maximal, never gives A4_K0 a smaller window than A4_K4, bounds the actual rows of any fetch and saves exactly the sink rows when block 0 is fetched, and aborts when nothing fits; the `x_imend` and `x_ucontent` finders on a synthetic template (including the engine's prompt without a think block, where the closer is the literal fifth token from the end, an earlier turn in the suffix, and a suffix without a header); ranks are restricted to the candidates and ignore resident sources; source status (sink, window, tail, not located); the sink-first piece list, the state with fetched blocks in logical order and byte-equal to the host copies with the tail and DeltaNet entry from the build state, block 0 fetched without double counting; the host store trims block 0's keys and pooled chunks; the generator against a fake `extend` (oldest evicted, sink kept, resident rows equal the host copies, evicted blocks keep the values computed while resident, DeltaNet state carried, a pinned prefix never evicted); a leak check that runs the generator over five segments, with and without a pinned prefix, and compares every `extend` call (for both full-attention layers, the input cache length, beta lengths and every row's token identity in K and V across all heads and channels; one DeltaNet accumulator; `orig_len` and the incoming tokens; tensor values are not compared exhaustively, and the empty-prefix call checks layer 0 only) against a plain reference eviction schedule written in the test, so a window one block late or early, or any evicted row, fails (checked by hand with two deliberate breaks of the eviction, both caught, then reverted); the answer path with a non-640 block size (the row assertion uses the given block size; the 640 default fails it); the report's counts, pairs and tables, including `inside_k_optimistic` and the real fetch coverage (none, partial, all).
- `modal/am_qwen35/tests/test_a1_a4_tiny.py` (tiny random hybrid, run on lucebox6 through `a1_pytest.sh` before the launch): shapes of the state with an evicted window; every resident row equals its host copy; fetched blocks' K/V byte-equal to the host copies in logical order; the rebuilt base state equals the generator's; with nothing evicted the state and the answer equal FULL-CHK's; the routing pass equals the brute-force exact mass over the candidates, leaves the base state untouched, and the answer runs under the row guard (which refuses a state over the cap).
Gate: all pass on lucebox6 before the launch. The tiny-hybrid test was not run locally (it needs the pinned compaction checkout and the model tokenizer).

### Details fixed here (readings of the design, chosen before any row)

1. The primary arms have no P: today's engine protects only a 64-token sink. A4_K4P (P exact) is reported next to them and is never the primary.
2. The sink is row-level (64 rows), not the whole of block 0; no 576-row deviation is needed.
3. Block 0 stays a candidate for rows 64 to 639; the cut charges it 640 rows (conservative by 64 rows when it is fetched).
4. The routing pass runs on the base state after cut-time eviction, because the build state plus the question can exceed the cap.
5. A fetched block's K/V are the host copies computed at seal time behind the build window of that time; they are not recomputed under the cut state. The open tail's K/V were computed before the cut-time eviction.
6. k is fixed (4), not budget-filling, and capped at the candidates. The reserve is A3's, taken per cut.
7. `x_imend` (the `<|im_end|>` closing the user message, the engine's end - 5 token), `x_ucontent` (the token before it) and `x_lit5` (the literal offset) are defined by the template or the offset as above; all are logged only, none decides a fetch.
8. Ranks use E2's `best_rank` (ties do not push the source down); the fetch uses A3's tie rule (lower block index). The share of probes whose rank is at most k (`inside_k_optimistic`) therefore counts ties in the source's favour and can exceed what the fetch delivers. The report adds the real outcome for `x_last`: the fact's candidate source blocks (`src_cand_blocks`) against the blocks actually fetched, counted as none, partial or all, as A3's source fetch coverage is.
9. A cut with no candidates is the full-KV state in every arm; the answers there are expected to equal the reference's, and the report counts how many do.
10. Block 0 may be evicted at build time, and the sink then stands in for it; the exact block 0 (all 640 rows, computed with nothing in front) stays in the host archive.
11. A fact wholly inside rows 0 to 63 counts as resident.

### What A4 does not change

The A1 rule and the 75.6% bar, A3 stage 2 and its selected window, the A2 arms, E1 to E9, the reference, scorer, limits and render. A4's numbers are not read as an A1, A2 or A3 result, and no A4 outcome changes a rule. A4 and E9's rank tables are different measurements (an LRU state with candidates only, against E2's all-compacted state) and are not combined.

### Outputs

Rows (harness row schema plus fields `attempt`, `build`, `k`, `k_plan`, `rows`, `reserve`, `peak_rows`, `n_cand`, `window_build`, `window_cut`, `fetch_bytes`, `src_blocks`, `src_status`, `answer_reused`, `fetched_blocks`): `~/r1e/out/a4/A4_K4.jsonl`, `A4_K0.jsonl`, `A4_K4P.jsonl`; router records `A4_K4_route.jsonl` and `A4_K4P_route.jsonl` (ranks, top 16, fetch plan, the `x_imend`, `x_ucontent` and `x_lit5` tokens); per-cut records `A4_cuts_S.jsonl` and `A4_cuts_P.jsonl`; `FULL.jsonl` (the copied reference), `done_S.jsonl`, `done_P.jsonl`, `manifest.json`, `progress.jsonl`, `summary.json`, per-attempt `provenance.<id>.txt`, `stdout.<id>.log`, `run.<id>.log`, `watchdog.<id>.out`, and `mem.log`. Report: `python3 a1_a4_report.py ~/r1e/out/a4 ~/r1e/out/a1_pilot ~/r1e/data/a1/B0_3090 ~/r1e/out/a3sel ~/r1e/out/e3e2` writes `a4_report.json`; evidence is copied to `evidence/agentic/A4/`.

### A4 Amendment 1 (2026-10-16, before any A4 row): compact-on-evict arms

No A4 row exists. The A4 section above is unchanged except that its code file `lucebox/a1/a1_a4.py` and report `a1_a4_report.py` gain the arms below (the A4 arms, their builds, rules and outputs are not altered; the original 26 A4 stub tests still pass). Nothing is modified in `a1_a3.py`, `a1_e.py`, `a1_core.py`, `a1_e67.py`, `a1_e9.py`, `r2_core.py` or `r1b_core.py`. "AM" is Attention Matching, the shipping compactor (cell FUNI); "selection only" is E8's rule (`a1_core.select_only`: the fit's own key ranking, the selected rows copied verbatim, beta 0).

**Why.**
- *The design rule (the user's).* The exact pool must work as it is, and compaction is active only when the pool is full. A cut that fits the 16,384-row budget must be the A4_K0 state exactly. Only when a block has to leave the window is it compacted, instead of dropped.
- *The paper* (arXiv 2608.00902, online KV compaction for agents; section 3.2, Tables 1 and 2) finds two things. Immediate fitting on a block's own prefill queries is the worst setting. Fitting with queries from later turns ("delayed") recovers almost all of the no-compaction score: Qwen3.5-27B scored 52.50 with no compaction, 51.00 with delayed AM and 52.00 with delayed selection-only (TE in its tables) at ratio 0.2 (Table 2; the ratio sweep says 0.1 is too aggressive and 0.2 improves every combination). Compact-on-evict gives that delay for free: a block is compacted when it leaves the window, with the queries that read it while it was resident. The paper's delay is one assistant turn; ours is every sealed block between the block and its eviction, a longer delay on a 640-token grid. That difference is stated, not tuned.

**State (the `gen_lru` variant, `ct=CtCompactor`).** Same 64-row sink, 640-token blocks, open tail and reserve as A4.
- When the window evicts block j it is compacted to ratio 0.2, `n // 5` slots per KV head (128 of 640), by the repo's compactor (`Runner._compact`, uniform slots, cell FUNI), and kept in a compacted region in logical order. Block 0 keeps its sink rows resident, so only its rows 64 to 639 are compacted (115 slots). Compacted blocks are frozen: never refit, never recompacted (a test follows the same tensors across cuts). DeltaNet state is carried as everywhere.
- *Delayed queries* of block j: the post-RoPE queries, on the 16 full-attention layers, of every token of the sealed blocks j+1 to (built - 1) that the build prefilled, where `built` is the number of blocks when j is evicted (the block being appended and the open tail, which is prefilled again as a block later, are not counted: each token once). They are captured by `r2_core.extend(..., want_queries=True)` under the arm's own cache and kept on host while the block is in the window. Subsampled uniformly (midpoints of 1,024 equal strides) to at most 1,024 tokens; the count available and used is recorded per compaction.
- *Budget.* `rows = sink + 640 * window_blocks + 128 * compacted_blocks + tail + reserve <= 16,384` (`ct_bound`; block 0 whole means no separate sink). Build time: `640*(W+1) + 64*[block 0 not whole] + 128*C + 639 <= 16,384` (`ct_build_step`), so a growing compacted region shrinks the window. Minimum window 2 blocks. If the compacted region plus that minimum does not fit, the oldest compacted block is dropped (a ceiling, marked `ponytail` in the code; its exact copy stays in the host archive, it is not routable; the drops are counted per build and per cut). If even that does not fit, the run aborts (`OverBudget`). The cap is asserted at the routing pass, at the answer state (`answer_ct`: actual rows equal the plan's, at most its bound, plus reserve at most the cap) and under A3's `row_guard`.
- *Cut.* `ct_cut_plan`: the largest window w, w at least 2, with the oldest window blocks beyond w compacted at the cut (on the queries available then, for this cut only; the build is untouched) and `k = min(k_req, compacted)`, such that `ct_bound + tail + reserve <= 16,384`; a swapped block is charged 640 instead of 128. At the window floor the oldest compacted blocks are dropped for this cut.
- *Invariant enforced on the data, not only in code.* At every cut where nothing is compacted or dropped, the cut record carries a sha256 digest of the state (K, beta and V of all 16 full-attention layers, the DeltaNet state; shapes and dtypes included), for A4_K0 (build S0) and for the CT, CA and CT0 builds; the generated ids of every probe answer are kept. A4_CT answers at those cuts are computed, never reused from A4_K0. The report exits non-zero naming the cut (or the probe) if any digest or id sequence differs from A4_K0's, or if a cut is evictionless in one and not the other (`check_no_eviction`). Every build and routing step also asserts equal row counts over all 16 full-attention layers and K, beta and V (`rows_all`, `build_guard`; A3's `_full_rows` read layer 0 only).
- *Invariant, asserted in code (`ct_cut_plan`) and tested.* When nothing is compacted or dropped at a cut, the plan's window and rows equal `cut_plan(..., k = 0)`, A4_K0's, and the generator is bit-equal to the original `gen_lru` (tests: all 24 grid block counts times tails 0, 1, 639 times reserves 164, 364; a generator comparison). The first eviction is at 25 sealed blocks, as in A4_K0; at 25 blocks the window is still 24 blocks and block 0 is the compacted one (64 + 15,360 + 128 + 639 = 16,191 rows).

**Arms** (probes only, the 6 pilot trajectories, the 118 reference-correct probes; every arm has its own state because later blocks are prefilled behind the compacted ones, so a compaction variant is a build):

| Arm | Build | How an evicted block is compacted | Fetch k | Role |
|---|---|---|---|---|
| **A4_CT** | CT | selection only (zero beta, original K/V rows), keys ranked by attention mass from the delayed queries, per head | 0 | primary |
| **A4_CA** | CA | the shipping fit (beta and values, FUNI) on the delayed queries | 0 | reported |
| **A4_CT0** | CT0 | selection only on the block's own seal-time prefill queries (640 tokens; the immediate control, isolates the delay) | 0 | control |
| **A4_CTK4** | CT (shared with A4_CT) | as A4_CT | 4 | reported |

A4_CTK4: routing pass on the cut's base state, `x_last` over the compacted slots (mass of the last question token on the compacted keys, normalised over the compacted blocks: `a1_e.router_mass`, beta 0), the top 4 are swapped for their exact host copies by A3's insertion path (E3's swap: the compacted entry is replaced, not duplicated), inside the budget (the window is smaller by the swap cost, minimum 2). Candidates are the compacted blocks present; dropped blocks are not routable. Row files `A4_CT.jsonl`, `A4_CTK4.jsonl` (+ `_route`), `A4_CA.jsonl`, `A4_CT0.jsonl`, cut records `A4_cuts_CT.jsonl`, `A4_cuts_CA.jsonl`, `A4_cuts_CT0.jsonl`, done markers `done_<build>.jsonl`; the same launch, manifest (now with the arms and the constants 128, 2, 1,024, the build order, and a `deps` digest of the compactor's resolved sources, the algorithm and query configs and the cache and query-capture overlay, so a change refuses to resume) and resume rules. Build and cut-time compaction logs are separate; each cut record carries the build's query stats (`q`) and that arm's cut-time stats (`q_cut`), the final cut included. The non-atomic `cp -n` that seeded the reference in `a1_launch_a4.sh` is removed; `ensure_reference` seeds it atomically. The report validates every done marker and cut record against the expected trajectories, probes and cuts and refuses an incomplete build. A4_K0 now has its own build pass (S0, the same state as S); the original A4 section's statement that K0 shares K4's build is superseded. The A4_CTK4 router reads only `x_last` over compacted slots (status "compacted").

**Primary rule.** Descriptive pilot as A4, with one stated rule so that the result cannot be argued afterwards. A4_CT is read as helping when, over the probes beyond the pool (distance 16,384 tokens or more, the report's last bin), the lower bound of the paired difference in retention A4_CT minus A4_K0 is above 0 (bootstrap, 95%, clustered by trajectory, 10,000 resamples, seed 0: `a1_a4_report.compact_on_evict`, `primary_rule`: POSITIVE or NOT_POSITIVE) **and** a within-pool guard holds: over ALL within-pool probes (distance under 16,384 tokens, cuts after evictions included), the probes A4_K0 answers right and A4_CT wrong, minus those A4_CT answers right and A4_K0 wrong, are at most 2 (`WITHIN_POOL_NET_LOSS`; the guard's counts are printed). The pilot has 118 probes: 18 beyond the pool and 100 within, and the beyond-pool probes sit in only 6 clusters (counts per trajectory 1, 7, 3, 3, 1, 3), so the lower bound rests on few clusters. POSITIVE is a preregistered descriptive screen whose only role is to decide whether a held-out run is worth doing; it is not confirmatory evidence. The 16,384 boundary is "at or above" everywhere (pool split, router filter). The A1 bar is unchanged: 75.6%, which is 719 of 951 probes. A4 runs the 118 reference-correct pilot probes, not B0's 951, so the bar is printed next to each arm's retention and its interval as a description (`descriptive_vs_bar`) and does not decide the rule. The same split (within the pool, under 16,384 tokens; beyond it) is reported per arm with its paired counts, McNemar exact and difference interval against A4_K0, plus the mean compacted blocks per cut and per probe, the drops (build, and per cut), the compactions done at the cut, and the delayed-query subsampling (compactions, those subsampled, mean tokens available and used). Within the pool the arms are expected to equal A4_K0 until the first eviction (the invariant), and the report counts how many of those answers equal the reference's (`no_candidate_identity`). A4_CA, A4_CT0 and A4_CTK4 are read against A4_CT descriptively (fit against selection, delayed against own queries, fetch against none); none of them is primary.

**Run order (arm-major, all 6 trajectories per build, resume and done markers per build):** A4_K0 (S0), then CT (A4_CT and A4_CTK4 answers), then CT0, then CA, then the original A4_K4 (S) and A4_K4P (P). The A4_CT-vs-A4_K0 answer lands about 1.5 hours after launch; `a1_a4_report.py --partial <a4_dir> <pilot_dir>` prints the primary rule (to `a4_partial.json`) as soon as A4_K0 and CT are complete, and marks other arms with missing probes "incomplete".

**Added time** (estimates from the A4 and A3 figures above, not gates). A4's builds take 120 to 200 s with no compaction. A3 stage 1's builds with the fit took 181 to 215 s against E9's 132 to 148 s without it, about 1 s per compacted block; with up to 1,024 delayed queries per fit instead of 640 and the host-to-device query gather, 1.5 to 2 s per compaction, about 33 compactions in the longest build, plus up to one cut-time compaction per cut.
- *A4_CT* (build CT, 6 trajectories): builds 200 to 300 s each, 20 to 30 minutes; 118 answers at about 11 s (A3 stage 1: 10.9 s at 14.7K rows), 22 minutes. About 45 to 50 minutes.
- *A4_CTK4* (shares the CT build): 118 answers, 22 minutes; 118 routing passes at about 1 to 3 s and cut-time compactions, 4 minutes. About 27 minutes.
- *A4_CA* (build CA): builds 220 to 320 s, 22 to 32 minutes; 22 minutes of answers. About 45 to 55 minutes.
- *A4_CT0* (build CT0): builds 190 to 280 s (own queries only, nothing gathered), 20 to 28 minutes; 22 minutes of answers. About 42 to 50 minutes.
- **Added: about 2.7 to 3.0 hours, at most 4 hours**, on top of A4's 1.7 hours (at most 2.5): about 4.6 hours, at most 6.5 hours, plus the model load, one job at a time on lucebox6, after A3 stage 2. Peak memory: the host holds the queries of at most 24 window blocks (of the order of 100 MB each over 16 layers, so about 3 GB) and the GPU adds the compacted entries (a few MiB per block) and a gathered query tensor (of the order of 200 MB), well inside A4's expected 58 to 63 GiB allocated and 100 GiB cap. The first trajectory's `progress.jsonl` line (it carries the build seconds and the `q` counts) gives the real figures. A4_CT answers at cuts without compaction could be reused from A4_K0 as A4 does; they are recomputed here (an upper bound on time).

**Checks before any row** (CPU, torch only): `lucebox/a1/test_a1_a4.py` now has 56 tests (the 26 above plus 30): arms and builds; the fits-equals-K0 invariant and its assertion; the 25-block first eviction; the budget cap at every build and cut for 129 block counts, tails 0, 1, 639 and reserves 164, 364, with any swap of k compacted blocks (block 0 included) inside the bound; the drop ceiling (oldest first, counted, an evicted block dropped at once is never compacted, aborts when the floor window alone does not fit, and the cut-time drop); cut-time evictions and the k = 4 swap paid by a smaller window; the delayed-query window boundaries (blocks j+1 to built-1, excluding j and the block being appended; the own-queries control); the 1,024-token subsample (evenly spaced from end to end, capped, gathered across blocks); the generator with a fake compactor (selection only, fit, own queries, the exact token ids of the queries passed, frozen across cuts, equal to the original generator while nothing is evicted, the drop ceiling); the piece order and the answer rows; the source status; the report's pool split, paired difference, rule and compaction statistics; the review fixes: the state digest over every tensor, the all-layer row guards, `check_no_eviction` refusals, the within-pool guard (net loss 2 positive, 3 not), the router filter with compacted sources and the 16,384 boundary, separate build and cut logs, done-marker and cut-record integrity, the arm-major build order, the partial report, and no `cp` in the launch script. `modal/am_qwen35/tests/test_a1_a4_tiny.py` gains test (f), a real CT-vs-K0 no-eviction comparison (state digest and generated ids, queries captured), and test (e) (the real compactor on the tiny hybrid: planned rows, zero beta, every kept row a row of the exact host copy, the swap byte-equal in logical order); it has not been run (it needs the pinned compaction checkout and the tokenizer) and is a gate for lucebox6 before the launch, like (a) to (d).

**What this amendment does not do.**
- It does not use turn-boundary segments: blocks stay on the fixed 640-token grid, so a block can start or end inside a turn and the delayed queries are those of whole blocks, not of the next assistant turn as in the paper.
- It does not run the next-tool-call task: the probes are A1's fact-recall questions, answered once at each cut.
- It does not test repeated tool calls: nothing is evicted and recalled twice in a row, no block is refit or recompacted, and the trajectories are the 6 pilot ones, so a result is neither a multi-step agent result nor a rule for A1, A2, A3 or the engine.
- It does not tune the ratio (0.2 only), the window floor (2), the subsample cap (1,024) or k (4), and it does not change any earlier rule or bar.

## A3 routed exact fetch (2026-10-15, before any row)

Written before any A3 row exists. Paths are relative to `optimizations/kvflash/`; "key-value cache" is written KV; "B0" is the 3090 baseline run of the agentic harness (`evidence/agentic/B0_3090/`) and POOL16K its exact 16,384-row pool arm, the bar's source; "A1" is the rung that ran the shipping design on the B0 probes (section "Rung A1" and its amendments above), "A2" the exact-recent-window rung, "E2", "E3" and "E7" the diagnostics above; "FULL-CHK" is full KV built block by block with no compaction (the A1 reference); "DeltaNet" is the Gated DeltaNet recurrent layers of Qwen3.8, whose state is carried through, never compacted; "SDPA" is PyTorch scaled-dot-product attention; "iGPU" is the integrated GPU of lucebox6.

What has been seen:
- **A1 proper** (`evidence/agentic/A1/proper/`): the shipping design (every sealed 640-token block compacted once to 64 slots) kept 38 of 337 reference-correct probes, 11.3%, over 20 trajectories; early look FAIL. FULL-CHK rows exist for trajectories 1 to 20 in file order (trajectories 1 to 6 are the pilot).
- **E3** (`evidence/agentic/E3E2/`, 6 pilot trajectories, 118 probes): swapping the one block that holds the fact back to exact K/V brings retention from 10 of 118 to 99 of 118 (83.9%, 95% interval [77.5, 91.1]), 1.04 blocks swapped per probe, the fact located in 118 of 118.
- **E2** (same run): ranking the compacted blocks by the attention mass their 64 slots receive from the question's last token puts the source block in the top 8 for 85.6% of probes and for 18 of 18 facts older than 16K (38.5 blocks per cut on average).
- Nothing from any A3 configuration. Nobody has run a router-driven fetch, and nobody has run any window-plus-prefix state with a fetch.

### Question

Can the design the 2.0b engine would ship, **an exact system/tool prefix and an exact recent window in front of compacted older blocks, whose exact K/V lives in host RAM and is fetched back by a router that reads only the compacted blocks**, recall more of what full KV recalled than POOL16K does, at the same number of rows? E3 says the right block is enough; E2 says the compacted blocks can find it. A3 is the first run where the router, not the located-block oracle, chooses the block.

### State at a cut (built over the trajectory's token stream by `r2_core.gen_fblk_window`, one build pass per trajectory and window)

- **P, the exact system/tool prefix.** Every sealed block that overlaps the tokens [0, end of the rendered system message) stays exact, as in E7 (`exact_blocks`, rounded up to whole blocks; 5,921 tokens in every B0 trajectory, 10 blocks, 6,400 rows).
- **W, the exact recent window**, stated in blocks throughout. The last W sealed blocks stay exact (`window` of `gen_fblk_window`, as A2). The grid is W = 2 blocks (1,280 tokens) and W = 6 blocks (3,840 tokens). A block is fitted once, at seal time, behind what the cache holds at that moment, and its compacted form replaces its exact form when it leaves the window; it is never re-fitted. P and W can overlap early in a trajectory and are then counted once.
- **The other sealed 640-token blocks**, the **candidates**, are compacted to 64 uniform slots (ratio 0.10, cell FUNI, the shipping fit). Their exact K/V, as computed at seal time, is kept (the `on_block` hook, as E3); it is what a fetch brings back. The engine keeps it in host RAM at 64 KiB per token (16 full-attention layers x 4 KV heads x 256 x K and V x 2 bytes); this offline run keeps it on the same device, which changes no number.
- **Carried state.** The DeltaNet state is carried through every block, and the open tail (under 640 tokens) stays exact, exactly as in the A1 shipping arm.

### Per probe

1. **Routing pass.** The question suffix (the harness's question turn and generation prompt, the same token ids A1 appended) is prefilled under the cut state and its queries are captured on the 16 full-attention layers (`r2_core.extend(..., want_queries=True)`, as E2). The pass's KV and DeltaNet state are dropped at once; nothing from it reaches the answer.
2. **Router `a_last`.** Each candidate block (never P, never W, and the open tail is not a block) is scored by the attention mass its compacted slots receive at the **last** question token: softmax of q.k/sqrt(d) + beta over the candidate blocks' slots only, summed per block, mean over layers and query heads (`a1_e.router_mass`, the E2 router (a) with the last-token aggregation). E2's normalisation ran over all blocks because all were compacted; here P and W are exact and are left out of the normalisation, so the router sees only what it ranks. Ties go to the lower block index. `a_last` is the only router A3 uses; the other three E2 scorers are not run.
3. **Fetch.** The top k candidates, k as the row budget fills (next section), are **swapped** to exact: the block's compacted entry is replaced by its exact K/V (`a1_e.oracle_triples`, E3's function), never kept next to it. The open tail and the DeltaNet state stay as the generator left them (the tail's KV was computed under the unswapped state, as in E3).
4. **Answer.** The routing-pass question KV is discarded, the question is prefilled again on the swapped state, and the probe is answered as in A1: `r2_core.answer_ids`, greedy, at most 64 new tokens, the harness scorer.

### Row budget (equal memory with B0 POOL16K)

Rows per full-attention layer at a cut with `nb` sealed blocks, an open tail of `tail` tokens, `E = |P union W|` exact blocks and `k` fetched blocks:

    rows = 640 * (E + k) + 64 * (nb - E - k) + tail      (= P + W + 64 * (compacted blocks not fetched) + 640 * k + tail)

B0's POOL16K holds the appended tokens inside its 16,384 rows (the pager allocates question and decode tokens within the fixed pool, `kvflash_pager.h:316`), so A3's budget covers them too (Amendment 1). The answer appends the question suffix (`q` tokens) and at most `max_new` generated tokens (64 for a probe; the last one is never fed back, so 63 rows, and the reserve below keeps the 64th as margin):

    reserve = q + max_new         (q of the longest question suffix at the cut, so that every probe of a cut has one k)
    peak rows = rows + reserve    must be at most 16,384

The cap is 16,384, POOL16K's pool. Each exact block in the window and each fetched block costs 576 more rows than a compacted one (640 - 64), so W and k trade one for one. **k fills the budget**: at every cut k is the largest number of router-ordered candidate blocks for which `rows + reserve` is at most 16,384, capped at the number of candidates. There is no requested k. At short contexts this fetches every candidate (the state is then all exact, as POOL16K is there); at long ones it fetches as many as fit. k is a function of the cut alone, so every probe of a cut uses the same k; the router only decides which blocks. The cut's k, rows, reserve, peak rows and rows at k = 0, the number of candidates, and whether k is below the number of candidates, are written into every row. The DeltaNet state is a fixed part not counted in the pool's rows and is the same in every arm.

The cap is **asserted, not assumed**: the routing pass (cut rows + question) is checked against 16,384 before it runs; the cut state is checked to have exactly the budgeted rows before the answer; and the answer runs under a guard on the model that adds every token fed to the cache (the question prefill, then each decode step) to the rows held and asserts the total is at most 16,384 at each call (`row_guard`). A violation aborts the run. Nothing is evicted and nothing is allowed to exceed the cap silently.

If `rows + reserve` at k = 0 is already above 16,384, the cut is **over budget** and the run **aborts** with an error (Amendment 2): there is no exception path, no unbounded cap and no flag. It does not happen on this grid at the probe cuts, which are at most 36,974 tokens (57 sealed blocks, tail 639): rows at k = 0 is at most 11,199 for W = 2 and 13,503 for W = 6, which leaves more than 2,600 rows for the question and the answer. Worked numbers (P = 10 blocks; `tail` 0 and 639; reserve between 164 and 364, i.e. a question of 100 to 300 tokens plus 64 new tokens, which give the same k; the real reserve is the cut's own and is logged):

| W (blocks) | cut tokens (sealed blocks) | candidates | k | rows at that k | peak rows |
|---|---|---|---|---|---|
| 2 | 12,288 (19) | 7 | 7 (all) | 12,160 to 12,799 | 12,324 to 13,163 |
| 2 | 24,576 (38) | 26 | 11 or 10 | 15,680 to 15,743 | 15,844 to 16,107 |
| 2 | 36,864 (57) | 45 | 9 or 8 | 15,744 to 15,807 | 15,908 to 16,171 |
| 6 | 12,288 (19) | 3 | 3 (all) | 12,160 to 12,799 | 12,324 to 13,163 |
| 6 | 24,576 (38) | 22 | 7 or 6 | 15,680 to 15,743 | 15,844 to 16,107 |
| 6 | 36,864 (57) | 41 | 5 or 4 | 15,744 to 15,807 | 15,908 to 16,171 |

Without the reserve k would be one higher at every long cut (12 or 11, 10 or 9; 8 or 7, 6 or 5), and a cut with 57 blocks, tail 64, P = 10, W = 6 would take k = 6 and exactly 16,384 rows before any question token; with any reserve it takes k = 5. These are arithmetic facts about the design, stated before any row. E2's recall at 8 (85.6%) is the measurement that says whether a k of 4 to 11 can be enough with 40 or more candidates.

### Stage 1: selection (6 pilot trajectories)

- **Data.** The 6 pilot trajectories, probes only, the 118 reference-correct probes of `evidence/agentic/A1/pilot/`, the pilot FULL-CHK rows (`~/r1e/out/a1_pilot/FULL.jsonl`, read only) as the reference. The probes are the ones E3 and E2 used.
- **Grid.** W in {2, 6} blocks (1,280 and 3,840 tokens), k filling the budget: two configurations, one state build per W and trajectory, one answer per probe. W = 8192 and a fixed k are not run.
- **Rule.** Retention of a configuration = correct probes over the 118 (A1's definition, trajectory as the cluster for the interval). The selected configuration is the one with the highest retention; ties go to the fewer mean rows (mean over its 118 probes of the cut's `rows`), then to the smaller W. There is no eligibility test: an over-budget cut aborts the run instead.
- **Completeness before any winner (Amendments 1 and 2), against a fixed universe.** Stage 1 expects exactly the 6 pilot trajectories and the 118 reference-correct probes, hard-coded in `a1_a3.py` (`PILOT_N`, `PILOT_PROBES`) next to the md5 of the pilot reference `evidence/agentic/A1/pilot/FULL.jsonl` (`93a29de59549954ebd9d46ffd9214047`). `a1_a3_report.py sel` names a selection only if the reference md5 and the 118 probes match, the arms found are exactly {A3_W2, A3_W6}, each arm has exactly one answer record and one router record for every one of the 118, no record is unexpected or duplicated, and each probe's answer and router records carry the same attempt id. Otherwise it stops with an error and writes no report. It is never complete against a set taken from whatever rows exist. A passing report carries `complete: true`, `reference_md5` and `expected_probes`.
- **Stage 2 universe.** Stage 2 expects A1 proper's held-out reference-correct probe set, scored the way A1 scored it, from A1 proper's reference `evidence/agentic/A1/proper/FULL.jsonl` (md5 `7ca44b131ecfb3b85f57169af8bc4501`, trajectories 1 to 20, 337 retained probes of which 118 are the pilot's): the 219 probes of trajectories 7 to 20 are asserted by count in the run and in the report. The reference for trajectories 21 to 60 is computed by this run's own FULL arm, so its probe set cannot be fixed in advance; for those the report checks only that every reference-correct probe of the finished trajectories has its records.
- **`load_selection` (Amendment 2).** Stage 2 takes W only from a report that passes all of: stage 1, `complete` true, arms exactly {A3_W2, A3_W6}, 6 trajectories, 118 probes per arm, the pilot reference md5, and a selected W equal to what the rule above gives when recomputed from the arm summaries (correct, n, mean rows). Empty or partial arm summaries are rejected. The manifest records the sha256 of the canonical JSON (sorted keys) of the report.
- **What stage 1 decides.** Only the window used in stage 2. It sets no bar and no verdict. Both configurations are reported with the retention table, mean rows, k and rows per cut, and the router recall below. Stage 2's trajectories (7 to 60) are disjoint from stage 1's (1 to 6), so the selection does not leak into stage 2.

### Stage 2: held-out proper run

- **Trajectories.** The 60 minus the 6 pilot ones: trajectories 7 to 60 in file order, 54 trajectories.
- **Configuration.** The one stage 1 selected, fixed before any stage-2 row, run as one arm (`A3_W<W>`, W in blocks). The run reads the stage-1 `a3_report.json` itself (`--sel-report`), refuses one that is not complete, not of exactly the registered arms, or that selected nothing, takes W from it (there is no W argument), and writes the report's sha256 into `manifest.json` before any row.
- **Resume (Amendment 1).** `manifest.json` (stage, arguments with `--ws` sorted, window, selection sha256 of the canonical JSON, the reference directory and the md5 of its FULL file, the md5 of the tasks, trajectories, template and tokenizer files, and of `a1_a3.py`, `a1_e.py`, `a1_core.py`, `a1_e67.py`, `r2_core.py` and `r1b_core.py`) is written at the first start of each stage; a rerun resumes only if it matches exactly, otherwise it refuses. A trajectory is complete only when its `done_A3_W<W>.jsonl` marker is written, after its last answer and router row; on restart a final line of an answer, router or done file that fails to parse (a torn tail) is dropped with a log line, a bad line anywhere else raises, and the rows of a trajectory without a marker (a watchdog stop, a crash) are removed and the trajectory is redone from its first cut. Every answer row and every router row carries the attempt id; every attempt writes its own `provenance.<id>.txt`, `stdout.<id>.log`, `run.<id>.log` and `watchdog.<id>.out`, and nothing earlier is overwritten. The launch scripts refuse only while the recorded pid is alive.
- **Reference.** FULL-CHK rows. For trajectories 7 to 20 they are copied unedited from the A1 proper run (`evidence/agentic/A1/proper/FULL.jsonl`, `~/r1e/out/a1/FULL.jsonl` on lucebox6), through a file in the output directory that is renamed into place, so the copy is whole or absent; those rows are never recomputed or replaced (the run asserts that no probe of a trajectory present in the A1 file is pending). For trajectories 21 to 60 they are computed with the same FULL arm code (`r2_core.gen_fullchk`, 640-token blocks, `empty_cache` after every block, bound into the generator itself with `functools.partial(gen_fullchk, empty_cache=True)` and also after every cut, same prompts, limits and stop tokens), in trajectory order, as each trajectory is reached. The functions `extend`, `gen_fullchk` and `answer_ids` of `r2_core.py` are unchanged between the A1 proper commit (9c0f1958) and A3: the later diff only adds code (checked with `git diff 9c0f1958 HEAD`); the md5 of the code in use is recorded in `provenance.txt`.
- **Rule.** The A1 proper rule and bar, unchanged. Quoted verbatim from this file.

  A1 Amendment 1, item 2: "**One early look, at 20 trajectories.** After the first 20 trajectories in file order, retention is computed with `r1b_stats.boot_mean` (10,000 resamples, seed 0) at **99.9%**, using the 0.05th and 99.95th percentiles. This is the Haybittle-Peto rule: an early stop needs overwhelming evidence, so the final 95% analysis keeps its error rate. If the 99.9% lower bound is above 75.6%, the result is PASS and stage 1 stops. If the 99.9% upper bound is below 75.6%, the result is FAIL and stage 1 stops. Otherwise all 60 trajectories run and section 6 decides at 95%, unchanged. Gate 3 (the reference's yield after 5 trajectories) still applies. This replaces "no interim look at R" in section 7."

  A1 section 6: "The bar is B0's POOL16K retention, 75.6%, taken as a fixed number from the plan (it has its own interval, 72.9 to 78.2, and the rule does not widen for it). The comparison is each side's retention against its own full KV, measured on different weights, cache types and machines; it is not a comparison of accuracy between engines. The test needs at least 50 trajectories finished with at least 600 retained probes; otherwise the result is reported as a prefix with its size." "Primary: R and its 95% interval [lo, hi] (section 5)." "1. **PASS:** lo > 75.6%." "2. **INCONCLUSIVE, point estimate at or above 75.6%** (interval contains 75.6)". "3. **FAIL:** hi < 75.6%, or INCONCLUSIVE with the point estimate under 75.6%." Interval method: `r1b_stats.boot_mean`, the trajectory as the cluster (`a1_core.boot_ci`, which equals it at 95%).

  How A3 applies it: the early look is taken after the first **20 held-out trajectories in file order** (trajectories 7 to 26), at 99.9%, on the A3 arm; PASS or FAIL ends stage 2. Otherwise all 54 run and the verdict is the 95% rule above (54 is at least 50; the retained-probe count is checked against 600 and the report states a prefix if either floor is missed). The 120-trajectory extension in A1 section 6, item 2, is not used. Gate 3 is not re-applied: it checks that the FULL arm's yield is plausible, A1 passed it (57.7%) on the same code, and trajectories 7 to 20 are bit-identical copies of its rows; the report prints the reference yield of the A3 trajectories anyway. What PASS, INCONCLUSIVE or FAIL mean for the engine plan is not set here; the numbers go to the project owner, as A2's FAIL branch does.
- **Reported next to the rule, never part of it.**
  - B0 POOL16K on the same probes: its retention over the same trajectories, and the per-probe comparison on the probes retained by both (counts and McNemar exact via `r1b_stats.mcnemar_exact`).
  - Retention **by distance bin** (0-4096, 4096-8192, 8192-16384, over 16384 tokens) and by probe type, next to POOL16K's bins, and **by k used** (probes grouped by the k of their cut).
  - Router recall of the source block, using E3's located-block definition (the blocks overlapping the first occurrence of the gold string, `a1_e.locate_span`): recall at k in {1, 2, 4, 8, 16} among the probes whose source block is a candidate, overall and for facts older than 16K; the numbers of probes whose source block is already exact (P or W), is in the open tail, or was not located; and the share of located probes whose source block is exact in the answer state.
  - **Source fetch coverage (Amendment 1)**, separate from the recall above and not replacing it: among the probes whose source block is a candidate, how many of the fact's candidate source blocks the fetch restored (none, partial, all), with the number of probes answered correctly in each class. It is descriptive. It does not say why a probe failed: with the source fully restored the other blocks, the history's K/V and the carried DeltaNet state are still affected by compaction, and an unfetched source does not prove the router caused the miss.
  - k and rows per cut: for each cut, the rows kept, the reserve, the peak rows, k used, the number of candidates, whether k is below the number of candidates; with the mean reserve, the mean and maximum peak rows, and the mean, the maximum and the histogram of k.
- **Coverage check before a verdict (Amendment 1).** `a1_a3_report.py proper` joins the answer and router records on (attempt, probe) and stops with an error unless every reference-correct probe of the finished trajectories has exactly one of each from one attempt; it never computes retention over a silent subset.
- **Provenance only, never a gate.** Routing-pass prefill seconds, fetch bytes (k x 640 tokens x 64 KiB), wall seconds per probe, answer decode tokens.

### Outputs

Rows (harness row schema plus A3 fields: `attempt`, `k`, `fetched_blocks`, `rows`, `reserve`, `peak_rows`, `rows_k0`, `n_cand`, `fetch_bytes`, `route_s`, `probe_wall_s`, `src_blocks`, `src_status`): `~/r1e/out/a3sel/A3_W<W>.jsonl` and `A3_W<W>_route.jsonl` (router ranks, fetch plan, `source_fetch_coverage`, `source_exact_at_answer`, `attempt`) for stage 1; `~/r1e/out/a3/` with the same names plus `FULL.jsonl` for stage 2; `done_A3_W<W>.jsonl` (trajectory markers), `manifest.json`. `progress.jsonl`, `checks.json` (early look), `summary.json`, per-attempt `provenance.<id>.txt`, `stdout.<id>.log`, `run.<id>.log`, `watchdog.<id>.out`, and `mem.log`. Report: `lucebox/a1/a1_a3_report.py` prints JSON and writes `a3_report.json`; evidence is copied to `evidence/agentic/A3/sel/` and `evidence/agentic/A3/proper/`. Runs resume per trajectory and window under the manifest rule above: rerun the same launch.

### Parity contract

Model Qwen/Qwen3.8-27B bf16, the same files, for every arm and the reference; KV bf16; greedy (argmax) with A1's limits (64 new tokens for a probe) and stop tokens; the rendered ids of each request are A1's (one tokenization per trajectory by the server's own tokenizer, hashes in `evidence/agentic/A1/request_ids.sha256`); SDPA attention as A1; one code copy, whose md5s are recorded in `provenance.txt` next to the A1 proper list (the FULL path of `r2_core.py` unchanged, above); no disk cache and no prefix cache anywhere (no server: every stream is built from scratch per trajectory and window, so the caching axis is symmetric by construction); ratio 0.10, 640-token blocks, 64 uniform slots, cell FUNI as in A1; the same retained probes (the reference's correct ones) in every arm. The B0 numbers come from another machine, other weights and another cache type and are worded as in A1: each side's retention against its own full KV.

### Machine, memory gates

lucebox6 iGPU, `HIP_VISIBLE_DEVICES=1`, $0, no Modal job. One job at a time: stage 1, then its report, then stage 2; never next to any other A1, A2, E or D job (the launch scripts refuse to start while one is running). The A1 launch gate (start only with host `used` at most 70 GiB and `available` at least 55 GiB), the per-process cap of 100 GiB, `empty_cache` after every cut and every FULL-CHK block, and the A1 watchdog (`a1_watchdog.py`: stop the run, resumable, when host `used` reaches 120 GiB or swap use grows by more than 1 GiB over its level when the watchdog starts) apply unchanged. A3's own peak is expected near E3/E2's 55 GiB allocated (weights plus the exact blocks kept aside, about 40 MiB per block, plus a state of at most 16,384 rows, about 1 GiB); stage 2's FULL-CHK builds reproduce A1's 65 GiB allocated and up to 100 GiB reserved. The two launch scripts are `lucebox/a1/a1_launch_a3sel.sh` and `lucebox/a1/a1_launch_a3.sh`; stage 2 reads the stage-1 selection from `a3_report.json` and cannot be given W; the selection's sha256 is in `manifest.json` and `provenance.<id>.txt`. A watchdog stop leaves a stale pid file: the next launch resumes (above).

### Time (estimates from E3/E2 and A1 proper, not gates)

- **Stage 1.** E3/E2 took 42 minutes for the 6 trajectories (build 142 s per trajectory, 15 minutes in total; about 240 answers at 6 to 7 s, inferred from the 1,456 s of answer time; 84 s of routing). A3 builds twice (one pass per W, same per-block cost, attention behind 12K to 17K exact rows adds under 10 s per trajectory: 2 x 6 x about 155 s, about 31 minutes), answers 2 x 118 = 236 times at 7 to 8 s (about 30 minutes) and routes 236 times at 1 to 2 s (about 6 minutes). Expected **about 1.1 hours, at most 1.75 hours**, plus the model load.
- **Stage 2.** Per held-out trajectory: build about 160 s, about 17 probes x about 10 s (routing plus answer), about 5.5 minutes. FULL-CHK for trajectories 21 to 60 takes about 750 s each (A1 proper). To the early look: 20 x 5.5 min + 6 x 12.5 min = about 3.1 hours. All 54 trajectories: about 5 hours of A3 plus about 8.3 hours of FULL-CHK, about 13 hours.

### Code and CPU checks before any row

`lucebox/a1/a1_a3.py` reuses `a1_e` (`router_mass`, `oracle_triples`, `locate_span`, `overlap_blocks`), `a1_e67` (`system_end`, `prefix_blocks`), `a1_core` (plan, rows, early look, interval) and `r2_core` (`gen_fblk_window`, `extend`, `answer_ids`, `gen_fullchk`); it adds only the budget arithmetic, the candidate set, the fetch plan and the driver. Tests, written first and seen failing before the code existed:
- `lucebox/a1/test_a1_a3.py` (stub model, runs anywhere with torch): k fills the budget (the most blocks that fit, every candidate at short contexts, capped at the candidates) and a cut over budget at k = 0 aborts; the grid is never over budget at the longest probe cut under the largest reserve; the row formula equals the rows of the real generator state; the router ranks only candidates even when the prefix and window slots would win; k = 0 is torch-equal to the prefix-plus-window state; fetching every candidate equals E3's swap of all blocks and is all exact; a fetched block replaces its compacted entry; the routing pass's KV is not what the answer sees and the cut state is untouched; the fetched blocks are the top k of the router order; no candidates means no routing pass; the selection and verdict helpers; the reference seeding (atomic: an interrupted copy leaves no destination); the report tables (cuts, retention by k used, source fetch coverage versus retention). Added with Amendment 1: k reserves the question and the decode tokens (nb = 57, tail = 64, P = 10, W = 6 gives k = 6 without a reserve and k = 5 with any); the routing pass is checked against the cap; the row guard raises at the question prefill and at the first decode step over the cap and is restored afterwards; the FULL fallback binds `empty_cache=True` into `gen_fullchk` through `run_unit` (the old call is shown to run no per-block cleanup); selection refuses `--cap` other than 16,384; stage 2 refuses to start without a complete stage-1 report and takes W from it; the manifest is written once and a changed run refuses to resume; a trajectory without a done marker is purged; the completeness checks refuse a missing, extra, duplicate or mixed-attempt record and a window set other than {W2, W6}; the source fetch coverage classes.
- `modal/am_qwen35/tests/test_a1.py`, the five A3 tests on the tiny random hybrid (run on lucebox6 through `a1_pytest.sh`, with the compaction checkout): row budget fills k and matches the real state's rows; the router never ranks P, W or the tail; k = 0 equals the prefix-plus-window state; fetching every candidate equals E3's swap and is all exact (explicit concatenation of the per-block exact K/V plus the tail); the routing pass's KV is discarded before the answer prefill, the answer's initial DeltaNet tensors equal the cut snapshot, changing a routing-pass tensor leaves every answer-state tensor unchanged (storage independence, in place of an object-id comparison), and the answer's tokens equal those of an answer that never routed.
Gate: all pass on lucebox6 before the stage-1 launch.

### What A3 does not change

The A1 rule and the 75.6% bar (above), the A1 reference, scorer, limits and render, and the A2 rule and its window arms (WIN18, WIN25), all stand. No ratio or window is tuned inside stage 2 (k is tuned nowhere: the budget sets it); stage 1's selection is the only choice, made on the 6 pilot trajectories. A3's results are not read as an A1 or A2 result, and neither changes A3's rule. Stage 1's router recall and E2's are different measurements (different state, normalisation over candidates only) and are not combined.

### Details fixed here (readings of the design, chosen before any row)

1. W is stated in blocks (2 and 6), as A2 did; the token figures (1,280 and 3,840) are 640 times that.
2. The router is normalised over the candidate blocks' slots only (E2: over all blocks, all compacted then; PREREG line 706).
3. There is no eligibility test in selection: a cut over budget at k = 0 aborts the run (Amendment 2), so every finished configuration kept the budget at every cut.
4. The exact K/V a fetch returns is the block's own prefill under the P-plus-window-plus-compacted state at seal time (what the engine would hold). E3's exact blocks were prefilled behind an all-compacted prefix, so A3's fetched K/V differ slightly from E3's.
5. Gate 3 is not re-applied (above).
6. The reserve is the cut's longest question suffix plus `max_new` (64), so the whole cut shares one k; probes with a shorter question simply leave a few rows free. The guard counts every fed token as one row, which is exact for the 16 full-attention layers.

### Amendment 1 (2026-10-15, before any A3 row): static review findings and the comparison wording

A static review (Codex, of the first two A3 commits, no code run) found: the budget left out the question and the answer's tokens; the FULL fallback never received its per-block cleanup flag; a partial run could name a stage-1 winner; stage 2 took W from an argument; "rerun the same launch" could not resume and mixed attempts; a copied reference could be silently replaced; "source fetched" meant any block; a tensor-id test; `--cap` ignored in selection. Fixes were written into the sections above (row budget with the reserve and the guards, the FULL generator binding, completeness before any winner, the selection report as the only source of W, the manifest and done markers, atomic reference seeding with a no-recompute assertion, attempt ids and the (attempt, probe) join, source fetch coverage none/partial/all, storage independence, `--cap`). No A3 row existed when it was written. The rule, the bar and the stage-2 trajectories are unchanged. A second review found the fixes incomplete (Amendment 2).

### Amendment 2 (2026-10-15, before any A3 row): abort, fixed universe, resume

A second static review (Codex, of 0c0079f9, no code run) still blocked stage 1. Changes: (1) a cut over budget at k = 0 aborts the run; the exception path, the over-cap and within-cap counts and the eligibility test are gone; (2) the reference universe is fixed in code: stage 1 is 6 trajectories and 118 probes under the pilot reference md5, stage 2 asserts the 219 held-out probes of trajectories 7 to 20 under A1 proper's reference md5, and trajectories 21 to 60 take their reference from this run's own FULL arm and so cannot be fixed in advance; (3) `load_selection` recomputes the rule and checks counts, `complete` and the md5; (4) the manifest adds the reference directory and md5s, the md5 of the data files and of `r1b_core.py`, a sorted `--ws` and a canonical-JSON hash of the selection report; (5) a torn final line of an answer, router or done file is dropped on resume, any other bad line raises; (6) attempt ids are unique (nanosecond time and pid); (7) the storage-independence test compares `untyped_storage().data_ptr()` across all tensor pairs; (8) the report keeps reserve and peak rows. The rule, the bar and the stage-2 trajectories are unchanged.

The comparison with B0 is worded as follows, and nothing else is claimed.
- **Rows on the GPU, not bytes.** A3 matches B0's POOL16K in rows per full-attention layer (16,384, question and answer included). A3 runs bf16 K/V and B0 runs q4_0, so the two arms do not use the same bytes; this is the framing A1 already has.
- **The exact archive.** A3's exact K/V of the compacted blocks stands in for B0's paged-out host chunks. KVFlash keeps every evicted exact chunk on host, so both arms hold the full exact history off the counted rows. What differs is how a block is chosen to come back: A3's router (the compacted-slot attention mass at the last question token) against the pager's own policy.
- **Caching.** The cache state is symmetric across A3's own arms and its reference (no prefix cache, no disk cache, everything built from scratch). It is not symmetric against B0, which uses a host prefix cache and primer requests. The bar is therefore each arm's retention against its own FULL reference, as in A1, and A3 is not an engine accuracy, speed or memory comparison.
