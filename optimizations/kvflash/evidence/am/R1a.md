# R1a: Qwen3.8-27B adapter is exact without compaction

Result: PASS (eager, gating). Pre-registration: `PREREG.md`, section "Rung R1a" and "R1a deviations".

Run 2026-10-13, one H200, torch 2.8.0+cu128, transformers 5.19.0, `Qwen/Qwen3.8-27B` BF16 loaded as `Qwen3_5ForCausalLM`
(text tower only; 16 full-attention + 48 Gated DeltaNet layers). Raw results on volume `kvflash-am-results`:
`qwen3.8-27b/r1a/r1a.json` (copy in `R1a/r1a.json`), `qwen3.8-27b/r1a/smoke.json` (copy in `R1a/smoke.json`).
Code: `../../modal/am_qwen35/` (`patches/0001-*.patch` on `adamzweiger/compaction` @ 2020f6d3, `overlay/models/qwen3_5/*`,
`r1a.py`, `r1a_core.py`, `tests/test_adapter.py`).

## Gating result, `eager`, ADAPTER vs R_split, 256 greedy tokens

| prompt | tokens (ctx + suffix) | tokens identical | max abs logit diff, first step | max over all 256 steps |
|---|---|---|---|---|
| P1 short | 29 (15 + 14) | yes | 0.0 | 0.0 |
| P2 code | 36 (18 + 18) | yes | 0.0 | 0.0 |
| P3 QuALITY 20041 | 2281 (2249 + 32) | yes | 0.0 | 0.0 |
| P4 QuALITY 51350 | 8277 (8245 + 32) | yes | 0.0 | 0.0 |
| P5 51350 + 22876 | 16581 (16549 + 32) | yes | 0.0 | 0.0 |

5/5 identical, every logit of every step bit-identical (threshold was 1e-2 on the first step). The adapter is a strict
no-op at beta = 0, including the DeltaNet state carry-over, the RoPE offset, the gate and the mask.

## Informational

- R_whole vs R_split (HF, one prefill vs chunked prefill + suffix), `eager`: tokens identical on P1 and P2 only; P3 diverges at
  step 74, P4 at 3, P5 at 32 (top1-top2 margins 0.125, 0.0, 0.0), first-step logit diff 0.25-0.77. This is the bf16 noise floor of
  re-chunking the prefill; it is not an adapter quantity and is the yardstick for the `sdpa` row.
- `sdpa` (not gating): first-step logit diff 0.0 on all five prompts; tokens identical on 1/5 (P2). Divergence steps: P1 135, P3 42,
  P4 3, P5 32, at top1-top2 margins of 0.0 / 0.125 / 0.125 / 0.0, i.e. at most one bf16 ulp (0.125 at |logit| 16-32). These sit inside the
  range of the R_whole noise above, so they are classed as kernel noise (HF decodes with mask=None and `enable_gqa`; the adapter passes
  an explicit additive mask because beta lives there). The sdpa decode is therefore not bit-equal to HF; bit-exactness is shown with `eager`.
- Unit tests (`modal run am_qwen35/r1a.py::unit`, CPU, 8 passed): mask helpers; a tiny random hybrid (4 layers) exact vs HF;
  duplicated keys with beta = -ln 2 leave logits unchanged (< 1e-4); a half-subsampled cache stays finite.
- `text_config.output_gate_type = "swish"` is ignored by transformers 5.19 (the gate is always sigmoid). Parity is unaffected
  (reference and adapter run the same code); it may cost answer quality. Not investigated.

## Adapter design (actual names)

- `models/qwen3_5/modeling_qwen3_5.py`: `Qwen3_5CompactableAttention(Qwen3_5Attention)` adds beta to the additive mask after the K/V
  cache update and before the kernel; the sigmoid gate is untouched. `Qwen3_5CompactableTextModel(Qwen3_5TextModel)` handles
  `CompactedPrefixCache`: RoPE positions continue from `original_seq_len`, the mask is built over the longest layer, DeltaNet
  layers run unchanged on the carried conv/recurrent state. `adapt_model(model)` / `restore_model(model)` swap classes in place (no new
  parameters, one set of weights for both arms).
- `masks.py`: `causal_additive_mask`, `add_beta_to_mask`, `repeat_kv_for_beta` (pure, unit tested). `query_capture.py`:
  `attention_layer_indices`, `project_queries` (queries over the 16 full-attention layers only, gate chunk removed).
  `cache_convert.py`: `dynamic_cache_to_triples`, `dynamic_cache_to_kv_tuples`, `inject_linear_state`,
  `compacted_cache_from_triples`, `greedy_decode`.
- Repo patch (6 files): `models/cache.py` (transformers 5 cache-layer API, `linear_layer_indices` carried unchanged),
  `compaction/compaction_methods/global_highest_attention_keys.py` (linear layers joined with the sliding set; query row map),
  `query_generation/{self_study,context_prefill,query_generator}.py` (queries only for attention layers).

## Smoke (not a rung)

One compaction on QuALITY `quality_52845` (6651 tokens, 6626 article tokens), target 0.1 (662 article tokens + 25 kept),
`AM-HighestAttnKeys` (`global_highest_attention_keys`, repeat queries): compaction 34 s, no NaN in any layer. Tokens per head average
the target; layers are padded to their per-layer maximum with beta = -inf (min 167, max 3958, mean 1559 slots per layer).
First 3 questions: uncompacted adapter 2/3 correct, compacted 1/3 (n = 3, no inference). Compacted answers D (gold B), C (gold B), C (gold C); all logits finite.

## Cost

Estimated from per-prompt walls plus load (H200 at $4.54/h): about $1.2 for the gating run (about 16 min), $0.14 for the smoke,
about $0.5 for four crashed attempts (empty-context P1, circular import, missing `requests`, log file inside the uploaded dir), about $0.2 CPU
(unit tests, weight prefetch). Total about $2 of the $15 cap. Modal billing was not read.
