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
