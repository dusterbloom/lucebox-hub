# R1b check: effective cache size of per-head cells, and the 4B full-cache baseline gap

Read-only analysis. AM clone `/mnt/kimi-k3/repo/am-compaction @ 2020f6d` was read, never executed. Budget math reimplemented in
`modal/am_qwen35/checks/budget_math.py`; log pairing in `modal/am_qwen35/checks/baseline_diff.py` (both CPU, stdlib only). No Modal run.

## 1. Per-head cells: padded tensor is 17%, attended slots are 10.4%

How the repo stores non-uniform budgets (`compaction/compaction_methods/per_layer_head.py`):
- Per-head budget = `int(proportion * sub_target * total_heads)` (L313-316), after `apply_max_ratio_cap` (`base.py` L208-302, water-filling).
  `sub_target = int(n_art*target)`, `total_heads = global layers x KV heads`, proportions sum to 1.0, so the total over heads is conserved at `sub_target x total_heads`.
- Each head's C1/C2/beta is padded to the layer's longest head (L637-657): zero K/V, `beta = -inf`. Same for a head that returns fewer keys than asked (L535-543).
  The model adds beta to the additive attention mask (`overlay/models/qwen3_5/modeling_qwen3_5.py` L58-61), so padded slots get zero weight. `nan_layers` is empty in every row.
- `r1b_core.py` L155 `full_layer_len_mean` = `t[0].shape[2]` = the PADDED tensor length (max head per layer, plus the kept non-article tokens), not the attended count.

Recomputed from `R1b/r1c/optimized_agnostic.json` (27B: 16 full-attention layers x 4 KV heads = 64 heads, sum of proportions 1.0, max 0.0866, cap 0.75 not binding),
20 selection articles, mean 6560.9 article tokens, +25 kept prompt tokens:

| Quantity (27B, target 0.10) | Value |
|---|---|
| jsonl `full_layer_len_mean`, per-head cells | 1115.55 (17.0% of article) |
| recomputed padded length | 1115.55 (max abs diff 0.05 over 20 articles: reproduces the jsonl) |
| attended slots per head, mean (budget sum / heads + 25 kept) | 680.2 = 10.37% of article |
| jsonl, uniform cells (UAM/RND/TRN) | 687.0 for article 0, 680 mean = 10.37% |
| sum of head budgets / (sub_target x heads) | 0.9995 (int truncation only) |
| head budget min / max | 17 / 4124 |

Target 0.05: padded 569.7 (8.68%), attended 352.2 (5.37%); jsonl uniform 352.5. Same story.
Qwen3-4B (36 layers x 8 KV heads, shipped `Qwen3-4B/optimized_agnostic.json`, cap 0.75 binds): target 0.10 padded 2230.6 (34.7%) vs attended 666.1 (10.37%); target 0.05 padded 1147.6 (17.9%) vs 345.1 (5.37%). Recomputed padded matches the jsonl to 0.05.

Verdict: per-head and uniform cells are at the same effective compression (680.2 vs ~680 attended slots per head; the total budget is conserved). The 17% figure is padding, not extra retained context.
Per-head vs uniform, AM vs random/truncate, online vs ceiling, and the 27B-vs-4B degradation (Delta) comparisons are NOT confounded by cache size.
What the padded number does affect: any memory or decode-cost claim. The allocated per-head KV tensor is 1.64x (27B) and 3.35x (4B) the attended size at 0.10, so "10x smaller cache in the full-attention layers" (R1b.md L22) holds for attended slots and not for allocated memory (about 5.9x at 27B, 2.9x at 4B).
Caveat not recoverable from jsonl: O2/C (OMP) may return fewer keys than the budget (padding then masks more), so their attended size is at most 680.

## 2. 4B full-cache baseline: 72.3% (R0) vs 67.2% (R1b) is a protocol difference

Same split (articles 0-19, 357 questions), same model, same prompt text (`format_context`, `format_question` wording), same parser `parse_model_choice`, same torch 2.8.0 / transformers 4.57.1. Differences:

| Axis | R0 `original` | R1b `orig` |
|---|---|---|
| Thinking | ON: `format_question(..., enable_thinking=True)` default (`evaluation/utils.py` L424), called without the flag at `qa_evaluator.py` L927 | OFF: `answer(..., thinking=False)` (`r1b_core.py` L120), template adds empty think block |
| Decoding | sampled: model generation_config temperature/top_k/top_p via `get_generation_params`, `generate_with_vllm_batch` (`qa_evaluator.py` L934-940) | greedy argmax (`r1b_core.py` L112), no sampling |
| Max new tokens | 2048 (`run_qa_evaluation.py` L112-116) | 256 (`r1b_core.py` L17) |
| Engine | `original`: vLLM 0.11.0 batched (`am_repro.py` L22; log shows "Processed prompts ... toks/s"); compacted cells: HF batched `generate` | HF, one question at a time, no vLLM (`r1b4.py` L22 image has none) |
| Seed | none | n/a (greedy) |

Parsing is the same function; unparsed rows are 2 in each protocol (R0 thinking truncation did not matter much). The `PREREG.md` R1b Amendment 1 item 1 already records the thinking/sampling/2048/vLLM difference; this note adds the row-level count.

Per-question pairing: R0 per-question JSONs live only on Modal volume `kvflash-am-results` (`/qwen3-4b/sel_*_t0.1/`); not copied locally. Recoverable locally: the R0 stdout `modal/sel.log`, which has per-question choice, gold and mark for 14 of 20 original-method blocks (247 of 357 questions; the log is truncated for 6 articles: Lost in Translation, Captain Chaos, Coming of the Gods, The Star-Sent Knaves, Dangerous Quarry, Muck Man). Gold labels agree on all 247 (0 mismatches).

On the 247 paired questions: R0-original 181 correct (73.3%), R1b-4B-original 168 correct (68.0%).
- R0 right, R1b wrong: 22. R1b right, R0 wrong: 9. Both right 159, both wrong 57. Net +13 for R0 (5.3 points), matching the full-split gap of 5.1 points (72.3 vs 67.2); 31 discordant of 247.
- Examples R0 right / R1b wrong (R0 choice = gold; R1b answer text): quality_52845 q0 gold B, R1b "A. 7 years"; quality_63401 q6 gold A, R1b parsed B ("Because he was so exhausted ..."); quality_62476 q4 gold D, R1b "A".
- Examples R1b right / R0 wrong: quality_62476 q2 gold C, R0 chose A, R1b "C"; quality_62476 q15 gold B, R0 chose C, R1b "B. Duane would not have escaped"; quality_30035 q4 gold D, R0 chose C, R1b "D".

Reading: the baselines are two different measurements (thinking + sampling at 2048 tokens vs non-thinking greedy at 256), so R0 and R1b 4B absolute accuracies must not be compared; the paired deltas inside each protocol are fine. R0 numbers also include sampling noise (single sampled run, no seed), part of the 31 discordant pairs. To pair the remaining 110 questions, `modal volume get` of the R0 result JSONs (a file download, not a compute run) is needed.
