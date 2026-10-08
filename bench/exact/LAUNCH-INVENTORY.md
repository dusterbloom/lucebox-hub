# Launch-count inventory and merge candidates — qwen4exp exact-stack, T=1 decode, gfx1151

Source data: `PROFILE-38.md` (wall/busy/idle decomposition), `rep1_mode1.json`
(machine-readable per-kernel table, 51 distinct kernel names, sums to exactly
**1871.0 launches/token**), `KNOCKOUT-HC.md` (measured ~2.4 µs/launch
critical-path cost + each kernel's own work, from the hc_* knockout arms),
`HC-AUDIT.md` (bandwidth-floor ranking of the hc_* family). Graph structure
from `server/src/qwen4exp/qwen4exp_graph.cpp` (comment at line 2: `embed ->
hc_init -> per-layer ([PLE], hc_mix(attn) -> linear|QSA -> hc_combine,
hc_mix(ffn) -> MoE -> hc_combine) -> hc_mix(output) -> lm_head`) and the
dispatch/fusion-matcher code in `server/deps/llama.cpp/ggml/src/ggml-cuda/ggml-cuda.cu`.

## 1. Per-layer-type kernel sequence and launch counts

48 decoder layers total: 36 linear-attention (GDN) layers + 12 full-attention
(QSA) layers, confirmed by the two layer-specific kernel counts in the trace
(`gated_delta_net_cuda_grouped_cols`=36/tok, `qsa_decode_wmma_partial`=12/tok).
Every "48/tok" row below recurs exactly once per decoder layer regardless of
type (shared HC-mix/combine machinery); every "36/tok" row is GDN-only; every
"12/tok" row is QSA-only.

### Per-GDN-layer sequence (×36, hc_mix -> linear GDN path -> hc_combine)

| order | kernel | launches/layer | µs/launch | small (<5µs)? | producer -> consumer |
|---|---|---|---|---|---|
| 1 | `hc_upmix_row8_exact` (HC mix-in, "xn" producer) | 1 | 20.74 | no | up-weight GEMV -> `mixed` |
| 2 | `hc_down_inject_mixed` | 1 | 21.01 | no | down-weight GEMV + inject -> `down_dst`,`inject_dst` |
| 3 | `quantize_hc_lo_q8_1` | ~2 (97/48≈2.0) | 1.71 | **yes** | silu(`mixed`) -> `lo_q8` for next-layer upmix |
| 4 | `hc_combine_norm_f32_b256` | ~2 (95/48≈2.0) | 10.03 | **yes** (latency-bound, see HC-AUDIT) | residual+block_out+gate -> `xn` |
| 5 | `gdn_ab_exact_f32` | 1 | 6.86 | borderline | A/B gate projections |
| 6 | `ssm_conv_step_f32` | 1 | 3.59 | **yes** | causal conv step |
| 7 | `gated_rms_norm2_kernel` | 1 | 2.28 | **yes** | GDN-tail rms_norm*gamma*sigmoid(z), **already PRODUCER_Q8-fused** (writes Q8_1 directly, `ggml_cuda_gdn_q8_match`, 36/tok confirmed sites) |
| 8 | `gated_delta_net_cuda_grouped_cols` | 1 | 34.84 | no | the GDN recurrence itself |
| 9 | `mul_mat_vec_q` (various types, q/k/v/out projections) | ~4.4 (210+48+5)/48≈5.3 shared across GDN+QSA | 25-120 | no | weight streaming |
| 10 | `quantize_q8_1` (generic activation producer) | ~5.3 (256/48≈5.33) | 2.09 | **yes** | **NOT yet producer-fused** — fuses rms_norm+gamma output into Q8_1 for the mul_mat_vec_q calls above |
| 11 | `rms_norm_f32` / `rms_norm_scale_f32` / `k_bin_bcast` / `cpy_scalar` etc. | several, each ≤2µs | 1-4 | **yes** | small glue ops around the above |

### Per-QSA-layer sequence (×12, hc_mix -> full attention -> hc_combine)

| order | kernel | launches/layer | µs/launch | small? |
|---|---|---|---|---|
| 1-4 | same hc_upmix/hc_down_inject/quantize_hc_lo_q8_1/hc_combine_norm as GDN | shared | shared | shared |
| 5 | `mul_mat_vec_f<bf16>` (Q/K/V/O projections, bf16 weights — no quantize step needed) | 6 | 14.96 | no |
| 6 | `rope_multi` | 4 | 3.15 | **yes** |
| 7 | `qsa_decode_wmma_partial` | 1 | 29.08 | no |
| 8 | `qsa_decode_ids` / `qsa_decode_merge` | 2 | ~10 each | borderline |
| 9 | `ds4_indexer_score_decode_wmma_kernel` | 1 | 10.84 | borderline |
| 10 | `k_topk_block_radix_f32_i32` / `k_argsort_f32_i32` | 2 | 6-9 | borderline |
| 11 | `soft_max_f32` | 1 | 1.90 | **yes** |
| 12 | `moe_fused_combine_shared_kernel_f32` | 1 | 4.55 | **yes** |

### Head (×1/token)

`argmax_f32` (129.69 µs, this is the GPU argmax over the full vocab logits —
large but called once/token, not a launch-count target), plus a handful of
1-launch glue kernels (`concat_dim0_dense_transpose_f32`, `reduce_rows_f32`,
`op_clamp_kernel`, `unary_op_kernel`×5) all <7 µs each, 1/tok.

### Full launches/token accounting (sums to 1871.0, confirmed against rep1_mode1.json)

| category | launches/tok |
|---|---|
| weight_stream (`mul_mat_vec_q`, `mul_mat_vec_f`) | 475.0 |
| hc_* family (`hc_down_inject_mixed`, `hc_upmix_row8_exact`, `hc_combine_norm_f32_b256`, `quantize_hc_lo_q8_1`) | 383.0 |
| generic `quantize_q8_1` | 256.0 |
| GDN-specific (`gated_delta_net_cuda_grouped_cols`, `gdn_ab_exact_f32`, `ssm_conv_step_f32`, `gated_rms_norm2_kernel`) | 144.0 |
| QSA-specific (`qsa_decode_*`, `ds4_indexer_*`, topk/argsort/softmax) | ~132.0 |
| norm/glue (`rms_norm_f32`, `rms_norm_scale_f32`, `k_bin_bcast`, `cpy_scalar`, `k_set_rows`, `k_get_rows_float`, `rope_multi`, `moe_fused_combine_*`, `unary_*`, `scale_f32`, `arange_f32`) | ~480.0 |
| head (`argmax_f32` + 1-launch glue) | ~1.0 |

## 2. Ranked merge candidates

| rank | kernel pair/chain | launches/tok removed | est. ms/tok saved | bit-identity risk | effort |
|---|---|---|---|---|---|
| 1 | **generic `quantize_q8_1` fused into its RMS_NORM\*gamma producer** (the plain pre-norm pattern, `ggml_cuda_op_rms_norm_fused`'s output currently re-read by a separate `quantize_row_q8_1_cuda` call before `mul_mat_vec_q`) | 256 | ~1.00 (0.40 ms own-work per KNOCKOUT-HC arm C, + 256×2.4µs≈0.61ms launch tax if the launch itself is eliminated, not just emptied) | **low** — same `q8_1_store_lane` formula (warp_reduce_max/sum over 32 lanes, d=amax/127, round, ds=(d,sum)) applied to the identical in-register `scale*x*gamma` value the unfused path recomputes from a stored-and-reread F32 buffer; no reassociation | medium — kernel itself is done (`launch-cut.cu`), wiring into `ggml-cuda.cu`'s dispatch loop + confirming which of the 256 sites aren't already PRODUCER_Q8-covered needs box access |
| 2 | `quantize_hc_lo_q8_1` folded into `hc_down_inject_mixed`'s epilogue | 97 | ~0.17-0.40 (own-work, HC-AUDIT §6 + KNOCKOUT-HC launch tax) | low (same silu+quantize formula) | **out of scope** — `hc_down_inject_mixed` internals owned by another agent, not touched |
| 3 | `hc_combine_norm_f32_b256` split or widened (HC-AUDIT §7 proposal) | 0 (same or +95 if split into 2 kernels) | ~0.2-0.88 | low (documented bit-identical reduction-tree split) | **out of scope** — hc_combine_norm internals owned by another agent |
| 4 | Collapse small QSA glue chain (`qsa_decode_ids` + `qsa_decode_merge` + `k_topk_block_radix_f32_i32` + `k_argsort_f32_i32`, 4 launches/QSA-layer = 48/tok total) into fewer launches | up to 36 (if 4->1 per layer, 12 layers) | ~0.3-0.5 (4×~8µs own-work + ~2.4µs×3 launch tax ×12 layers ≈ 0.44ms) | medium — multiple distinct reduction algorithms (top-k radix, argsort, merge) to fuse correctly, more surface area for a subtle ordering bug than a pure norm+quantize fusion | high — not attempted here, flagged for a future pass |
| 5 | Fold `rope_multi` (48/tok, 3.15µs mean) into the preceding Q/K-norm kernel for QSA layers | 48 | ~0.1-0.15 | medium (RoPE's position-dependent sin/cos table adds a second data dependency) | medium — not attempted here |

Candidate 1 is the only one that is simultaneously (a) bit-identical by
construction, (b) untouched by the other agent's hc_* ownership, and (c)
already measured as real recoverable time (KNOCKOUT-HC arm C: −0.40 ms/token
own-work; the launch-overhead half of its cost is the ~2.4 µs/launch ×256
figure from arm A−B of the same knockout). It is the one implemented below.

## 3. What was implemented

`server/deps/llama.cpp/ggml/src/ggml-cuda/launch-cut.cu` /
`launch-cut.cuh`: a new, self-contained kernel `rms_norm_mul_q8_1_f32<block_size>`
that fuses `rms_norm(x) * gamma` with Q8_1 quantization of that same
per-column value into one launch — bit-identical to today's two-kernel path
(`rms_norm_f32<block,do_multiply=true>` in `norm.cu` followed by
`quantize_q8_1` in `quantize.cu`, reading back the norm's F32 output).

- Norm half: same `block_reduce<SUM,block_size>` over sum-of-squares, same
  `rsqrtf(mean+eps)` scale, same per-column recompute `scale*x[col]*mul[col]`
  (no caching across the two passes, matching the unfused kernels' own
  behavior of recomputing `x[col]` fresh rather than reusing a register).
- Q8_1 half: same per-32-lane (one hardware warp) `warp_reduce_max<QK8_1>`/
  `warp_reduce_sum<QK8_1>`, `d = amax/127`, `q = round(v/d)`,
  `ds = make_half2(d, sum)` as `quantize_q8_1`'s `q8_1_store_lane`.
- Host launcher `ggml_cuda_rms_norm_mul_q8_1_cuda(...)` is **not yet wired**
  into `ggml-cuda.cu`'s dispatch loop — it is a tested, ready building block.
  Wiring requires box access to confirm the exact subset of the 256
  `quantize_q8_1` call sites this covers (PRODUCER_Q8 already claims 36 GDN +
  95 HC sites per the `GGML_ASSERT(producer_gdn_sites==36)` /
  `producer_hc_sites==95` invariants in `ggml-cuda.cu`; extending those exact
  sealed invariants blind, without the box, is the unsafe part — not the
  kernel itself).
- `bench/exact/test_launch_cut_bitexact.cpp`: standalone HIP unit test
  (no ggml graph), exercising the exported hook
  `ggml_cuda_test_launch_cut_rms_norm_mul_q8_1` — runs the fused kernel (A)
  and a from-scratch reproduction of the unfused baseline (B, duplicated in
  `launch-cut.cu` itself, zero dependency on `norm.cu`/`quantize.cu` internals)
  on identical random inputs and `memcmp`s both the F32 norm output and the
  raw Q8_1 bytes. 13 cases: production decode shape (2560 cols × 1 row,
  small+wide value ranges, ×5 reps each), multi-row (12, 48 rows), the
  ≥1024-column code path (10240 cols), small ncols (128, 32), and an
  all-zero row (exercises the `amax==0` branch, no div-by-zero).
- `bench/exact/build_launch_cut_test.sh`: build script for a **fresh**
  `~/qwen4exp-launch-cut/build1151` configure (never `cp -a`'d from another
  build dir), links only against `libggml-hip.so` + `libamdhip64.so` (no
  graph/driver dependencies needed for this test).

No env gate was added to `ggml-cuda.cu` yet since nothing is wired in; once
wired, the plan is `LUCE_QWEN_LAUNCH_CUT=1`, default off, old path untouched
— matching every other `LUCE_QWEN_*` gate in this tree.
