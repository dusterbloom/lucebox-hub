# qwen4exp fusion audit (in-situ, lucebox4 gfx1151, 2026-10-15)

Verification only: are the fusions claimed since 8714e452 actually engaged in the graph the
server runs (p4096 prefill, T=1 decode), and what copy / layout / conversion / quantize work
is left. No production code was changed.

## Provenance

- Branch `qwen4exp-exact-stack`, HEAD 60369ae2. Box tree `/home/duster/qwen4exp-exact-stack/src/server`
  checked with `rsync -rnc` against HEAD: identical except
  `deps/llama.cpp/ggml/src/ggml-cuda/mmvq.cu` and `test/unit/test_qwen4exp_decode_requant.cpp`,
  both md5-identical to their 37905d7e versions (the opt-in `LUCE_KQ_DENSE` variants, unset =
  generic kernel; production path unchanged). The MTP port agent was active on the box but had
  not synced newer files at trace time.
- Binary `build1151/luce_server` md5 `938bea37` (libggml-hip `2bc1443d`), built 14:04 after mmvq.cu 14:03.
- Model `Qwen3.8-Flash-Next-UD-Q4_K_XL` (blk.2 ffn_gate/up_exps are Q5_K, all other layers Q4_K;
  ffn_down_exps Q5_1 except blk.2/4/30/46/47 Q8_0). Flags `--max-ctx 20480 --prefix-cache-slots 0`,
  `HIP_VISIBLE_DEVICES=1`, lock `~/BOX_LOCK` held 14:20-14:31.
- Traces: `rocprofv3 --kernel-trace` (7.2.2) via `bench/exact/audit/trace.sh`, reduced with
  `bench/exact/audit/ktaudit.py`. Debug logs on: `LUCE_OP_TRACE=1 LUCE_QSA_GATE_DEBUG=1
  LUCE_NORM_ROPE_DEBUG=1 LUCE_PLE_DEBUG=1` (stderr only, kernels unchanged).
  - `pf`: warm-up `prompt_p4096_r1` (T=4131), measured `prompt_p4096_r0` (T=4136), isolated as the
    last trace segment with >1000 launches after a >1 s idle gap. 3158 launches, kernel-sum
    3146.2 ms, 1101.8 tok/s under the profiler (prefix_len=0 on both requests).
  - `dec`: HEAD defaults, 16-token warm-up then 2 x 128 tokens on `prompt_decode512_r0` (T=507 prompt);
    window = last 100 tokens before the last argmax. 1664 launches/token, wall 38.42 ms/token
    under the profiler, kernel-sum 39.07 ms (shared-expert stream overlaps, see below).
  - `req`: same with `LUCE_DECODE_REQ=q5_k LUCE_DECODE_REQ_ATTN=q6_k`. Wall 34.95 ms/token.
  - Trace dirs deleted after reduction (box disk ~13 GB free). Ledgers kept locally only.
- Profiler overhead: kernel-trace adds a few us per launch; the figures below are kernel durations
  (GPU) and ratios, not production wall times (HEAD no-profiler decode is ~28.6 tok/s = 35 ms).

## 1. Fusion status

Counts are per request (prefill) / per token (decode). "sites expected" comes from the graph:
12 attention layers, 36 GDN layers, 48 MoE layers (47 Q4_K + blk.2 Q5_K), 95 hc gate sites,
1 PLE layer.

| fusion | claimed by | kill switch / log | expected | seen | status | evidence |
|---|---|---|---|---|---|---|
| norm-rope (q/k RMS_NORM + ROPE -> `rope_multi_heads_norm`) | GUFO-PARITY prefill stack | `LUCE_NORM_ROPE_DEBUG` | 36 (3 per attn layer: q [256,24], k [256,2], qsa [128,4]) | 36 | ENGAGED | `[norm-rope] fused ne=[256,24,4136] cont=1` x12, `[256,2]` x12, `[128,4]` x12; `rope_multi_heads_norm<true,false,true>` 36 x 0.836 ms; unfused `rms_norm_f32<256,true>` and `rope_multi_heads` only 12 tiny launches (decode-prompt-shaped) |
| QSA3 gated attention (FLASH_ATTN_EXT + GATED_RMS_NORM_F16 -> `qsa3_attn_kernel<true>`) | QSA3 gate fusion | `LUCE_QSA_GATE_DEBUG`, `[qsa-gate] fused/unfused` | 12 | 12 | ENGAGED (p4096) | `[qsa-gate] fused n_q=4136` x12, 0 unfused; `qsa3_attn_kernel<true>` 12 x 11.40 ms; no `gated_rms_norm_f16_kernel`. Not built at T=507 (`[qsa-gate] unfused n_q=507 next=RESHAPE supported=0`, `qsa3_attn_kernel<false>`): the gated path requires `ggml_backend_cuda_mmb_f16_input_ok(L.wo, T)` which needs `T >= mmb_min_t() = 512` (mmb.cu:665/748). By design, not a bug |
| QSA union bitmap | QSA3 | - | 12 | 12 | ENGAGED | `qsa3_union_bitmap_kernel` 12 x 0.267 ms, `qsa_decode_ids` 12 |
| rope table (`rope_multi_table`) | prefill stack | - | 12 + 7 | 19 | ENGAGED | `rope_multi_table` 19 launches |
| GDN row-split kernel | prefill stack | - | 36 | 36 | ENGAGED | `gated_delta_net_rowsplit_cuda<2,4,8>` 36 x 3.06 ms |
| conv-folded q/k norm (`gdn_conv_direct_norm`) | prefill stack | - | 36 | 36 | ENGAGED | `gdn_conv_direct_norm_kernel<128,16>` 36 + `gdn_conv_direct_kernel<true,128>` 36 |
| wave gated norm (GDN output) | prefill stack | - | 36 | 36 | ENGAGED | `gated_rms_norm_wave_kernel<4>` 36 x 1.49 ms |
| hc norm RMS_NORM + MUL (`rms_norm_f32<1024,true>`) | hc_mix gamma-expand fix | - | 5 (2 hc_mix + 3 PLE grouped_norm) | 2 | PARTIAL 2/5 | `rms_norm_f32<1024,true,false>` 2 x 1.87 ms (hc_mix sites, gamma expanded at qwen4exp_graph.cpp:224-230); `rms_norm_f32<1024,false,false>` 3 x 1.84 ms + `k_bin_bcast op_mul` 3 x ~1.5 ms = the PLE `grouped_norm` key / query / conv sites. **Root cause**: `grouped_norm` (qwen4exp_graph.cpp:214-218, decode twin at 1391) does `ggml_mul(c, ggml_rms_norm(t), ggml_reshape_2d(c, nw, ...))` without expanding the gamma RESHAPE first, so the DFS emits RMS_NORM, RESHAPE, MUL and the adjacency check in `ggml_cuda_can_fuse` fails (same class as the two norm fusions that never matched before). **Fix (one line, both lambdas)**: `ggml_tensor * g = ggml_reshape_2d(c, nw, n_embd, hc); ggml_build_forward_expand(gf, g);` before `ggml_rms_norm`, exactly as `hc_mix` does. Saves ~5 ms/request (3 MUL launches + the norm's second pass; 0.16%) and 3 launches/token in decode |
| PLE conv (CONCAT + 4 taps + SILU -> `ple_conv_kernel`) | PLE conv fusion | `LUCE_PLE_DEBUG`, `[ple-conv]` | 1 | 1 | ENGAGED (tail CONT left) | `ple_conv_kernel<4,3,128,float>` 1 x 1.83 ms, `ple_concat_tail` 1. The `[ple-conv] no match at node N: ple-conv.cu:62/71` lines are the other CONCAT nodes (GDN conv state, line 62 = not a PLE concat, 71 = shape); the single PLE concat matches. The match ends at the SILU node (ple-conv.cu:124-128): the `ggml_cont(silu)` at qwen4exp_graph.cpp:1278 is outside it and executes as a 169 MB `__amd_rocclr_copyBuffer` (1.6-1.8 ms, the two biggest copyBuffer launches). The kernel already writes a contiguous [10240,T] f32 tensor, so the CONT is a pure memcpy |
| F16 routed GLU single launch (`mmb_routed_glu_f16_kernel`) | mmb GLU f16 (default `LUCE_MMB_GLU_F16`) | `LUCE_MMB_GLU_LEGACY=1` | 48 (47 Q4_K + 1 Q5_K) | 46 + 1 legacy | PARTIAL | `mmb_routed_glu_f16_kernel<128,44>` (44 = 32+Q4_K) 46 x 9.94 ms. blk.2 (Q5_K gate/up) takes mode 0: the legacy big+small pair `mmb_routed_glu_kernel<64,128,32,32,45>` 13.32 ms + `<64,64,16,32,45>` 28.56 ms (45 = 32+Q5_K, mmb.cu:137-145) = 41.9 ms vs ~10 ms for a Q4_K layer. Mode selection `mode = !(gw->type == GGML_TYPE_Q4_K && E <= 1024 && (M & 7) == 0) || mmb_glu_legacy() ? 0 : ...` (mmb.cu:993) only admits Q4_K. ~32 ms/request = 1.0%, the largest single item in this audit. 46 of 47 Q4_K layers seen; no other GLU kernel appears in the ledger for the 47th, so it is either a counting artefact of the segment boundary or a layer whose GLU output is consumed by the Q8_0 down path differently - not resolved by the trace |
| hc_scale_silu (SCALE + SILU on the [320,T] gate) | hc stack | `LUCE_QWEN_HC_SCALE_SILU` | 95 (T=1 only) | prefill 0 / decode n/a | ENGAGED by design | matcher rejects `ggml_nelements(mm) != 320` (ggml-cuda.cu:5206-5240), i.e. T=1 only. Prefill runs `scale_f32` + `unary_op_kernel<op_silu>` 95 x (0.013+0.017) = 2.9 ms (0.09%). In decode the kernel `hc_scale_silu_f32` does not appear in the ledger either: the gate SCALE+SILU is consumed by `hc_down_inject_mixed` / `hc_upmix_row8_exact` (the [320,1] gate path is absorbed by the decode-knob kernels, 0 `scale_f32`+`silu` pairs per token at the hc sites: decode has 3 `scale_f32`, 1 `silu` per token, all PLE/router) |
| hc_combine_norm (residual combine + norm) | hc stack | - | 95 + 48 | 95 + 46 + 3 | ENGAGED | prefill `hc_combine_norm_f32_b256` 49 + `hc_combine_norm_moe_f32_b256<true>` 46 + `hc_gate_mix_kernel` 95; decode `hc_combine_norm_f32_b256` 95/token |
| mmb_cvt_*_v8 (bf16 weight -> f16 convert at load) | mmb | - | load-time | not in ledgers | n/a | runs at model load, outside the measured segments; no per-request cost observed |
| q8f16 raster G=1 | mmb | - | - | - | not observable | raster choice is not visible in a kernel trace (same kernel name) |
| **decode knobs** (`qwen4exp_exact_stack_knobs()`, loader.cpp:1238-1250; master `LUCE_QWEN_EXACT_STACK=0`) | | | | | | all with `LUCE_DECODE_REQ` unset (HEAD defaults) and on the req arm |
| LUCE_QWEN_HC_DOWN_INJECT | decode stack | capture assert pairs=95 | 95/token | 95 | ENGAGED | `hc_down_inject_mixed` 9500/100 tokens, 21.3 us each |
| LUCE_QWEN_HC_UPMIX_ROW8 | decode stack | `[hc-upmix-row8-capture] sites=96` | 96 | 96 | ENGAGED | capture log on both arms, `hc_upmix_row8_exact<false>` 9600/100 |
| LUCE_QWEN_HC_LO_Q8 | decode stack | assert 97 | 97 | 97 | ENGAGED | `quantize_hc_lo_q8_1<true>` 9700/100 |
| LUCE_QWEN_GDN_AB_EXACT | decode stack | assert 36 | 36 | 36 | ENGAGED | `gdn_ab_exact_f32` 3600/100 |
| LUCE_QWEN_SHARED_EPILOGUE | decode stack | `[shared-epilogue-capture] uid=1..3 mode=1 sites=48` | 48 | 48 | ENGAGED | `moe_fused_combine_shared_kernel_f32<true>` 4800/100 |
| LUCE_QWEN_PRODUCER_Q8 | decode stack | assert gdn 36 / hc 95 | 36 + 95 | 36 + 95 | ENGAGED | ssm_out GEMV runs with no preceding `quantize_q8_1` in the token sequence; `quantize_q8_1` count 256/token matches the producer-q8 accounting (see section 2) |
| LUCE_GPU_ARGMAX | decode stack | - | 1 | 1 | ENGAGED | `argmax_f32<true>` 1/token (131 us) |
| LUCE_QWEN_EXPERT_ROW_WARPS=8 | decode stack | - | ffn_down GEMVs | 43 + 5 | ENGAGED | `mul_mat_vec_q<Q5_1,...,8>` 43/token, `<Q8_0,...,8>` 5/token (the 8-warp variants) |
| two-wave routed gate_up GEMV | decode stack | - | 48 | 48 | ENGAGED | `mul_mat_vec_q_routed_gate_up<Q4_K>` 47 + `<Q5_K>` 1 per token |
| bf16 router shadows | decode stack | - | 48 + 12 | 60 | ENGAGED | `mul_mat_vec_f<bf16,float,1,256>` 60/token |
| LUCE_QWEN_SHARED_OVERLAP | decode stack | - | 48 | - | ENGAGED (indirect) | shared-expert kernels on stream 1 show inflated durations (`quantize_q8_1` 40-52 us for the shared down input while overlapping the routed gate_up), kernel-sum 39.07 > wall 38.42 ms |
| LUCE_QWEN_GRAPH_SH / QSA_CONT_ELISION / EXACT_ROUTER_SUFFIX / HOST_PREFILL_GUARDS | decode stack | - | - | - | ENGAGED (structural) | graph-shape knobs, verified only by the launch count being exactly 1664/token and the greedy-64 byte-identity already recorded in decode-requant/RESULTS.md; no kernel-level signature |
| LUCE_DECODE_REQ=q5_k / _ATTN=q6_k (opt-in) | decode-requant 9281c696 | `decode copies (...)` log | 156 tensors | 156 | ENGAGED (req arm) | `decode copies (gdn q5_K, attn q6_K): 156 tensors, 1.79 GiB`; `mul_mat_vec_q<Q5_K>` 108 + `<Q6_K>` 48 per token replace 156 of the 210 Q8_0 GEMVs; wall 38.42 -> 34.95 ms/token under the profiler |

Totals: 27 fusions / fast paths checked. 23 ENGAGED, 2 PARTIAL (PLE grouped_norm 2/5, routed GLU
F16 46/47+blk.2), 0 NOT ENGAGED, 2 not observable in a kernel trace (cvt_v8 load-time, raster G).
The PLE conv fusion is engaged but leaves its trailing CONT as a 1.7 ms memcpy.

## 2. Redundancy ranking

Threshold flag: >= 0.3% of the request (9.4 ms) or of the token (115 us at 38.4 ms).

### Prefill (p4096, per request, kernel-sum 3146 ms; copy/cont class total 16.7 ms = 0.5%)

| rank | what | launches | ms/req | % | verdict |
|---|---|---|---|---|---|
| 1 | blk.2 Q5_K routed GLU on the legacy two-kernel path instead of `mmb_routed_glu_f16_kernel` | 2 | 41.9 (vs ~10) | 1.0 | REMOVABLE: extend the F16 GLU kernel's type admission to Q5_K (mmb.cu:993 + the Q5_K dequant tile) or requantize blk.2 gate/up to Q4_K at load. ~32 ms. FLAGGED |
| 2 | K/V cache round trip per attention layer: `cpy_scalar_contiguous<f32,f16>` (cast after rope) -> `cpy_1_scalar<half,half>` permuted CPY into the cache view (K and V, 24) -> `CONT` out of the cache view (24) -> pack CONTs (`cpy_scalar<half,half>` 12 + `cpy_scalar_transpose<half>` 12) | 60 + 12 + 24 | 14.1 + 1.1 + 0.8 = 16.0 | 0.5 | REMOVABLE in two steps: (a) `cpy_scalar` at [2117632x64] grid runs 1 element/thread at ~36 GB/s; a vectorised half copy alone is ~-11 ms. (b) `rope_multi_heads_norm` could write f16 directly into the strided cache view (it already has the output stride) and `qsa_pack_keys/values` could read the cache view instead of a CONT of it, removing the cast + CONT (24 + 24 launches, ~3 ms) on top. Graph: qwen4exp_graph.cpp:1023-1030 (cast + permute + cpy), 878-886 and 666-676 (CONT of permute before pack). FLAGGED |
| 3 | PLE grouped_norm unfused (3 x `rms_norm_f32<1024,false>` + 3 x `op_mul` [10240,T]) | 6 | ~10.0 | 0.3 | REMOVABLE: one-line gamma expand (section 1). ~5 ms. FLAGGED (borderline) |
| 4 | PLE `ggml_cont(silu(conv_out))` after the fused conv (`copyBuffer` 169 MB x1 + warm-up twin) | 1 | 1.7 | 0.05 | REMOVABLE: drop the `ggml_cont` at qwen4exp_graph.cpp:1278 (the kernel output is contiguous); or make the matcher absorb a following CONT of its SILU |
| 5 | shared-expert GLU unfused (`unary_gated_op_kernel<op_silu>` on [3072?,T]) | 47 | 5.4 | 0.17 | REMOVABLE only with a fused shared gate_up GEMM epilogue (mmb_q8f16 with GLU) - not a copy, listed for completeness |
| 6 | router: `k_argsort_f32_i32` 48 x 0.40 + `soft_max_f32` 48 x 0.23 + `k_topk_block_radix` 12 x 0.59 + `mm_ids_route_sort` 94 x 0.05 + `mm_ids_route_prefix` 94 | 296 | 19.4 + 11.0 + 7.1 + 4.6 + 1.6 = 43.7 | 1.4 | PARTLY REMOVABLE: argsort over 512 experts per token is a full sort for a top-k pick (radix top-k already exists for the 12 QSA sites, 0.59 ms vs argsort 0.40 + soft_max 0.23 per layer); `mm_ids_route_sort` runs twice per layer (gate_up and down each re-sort the same ids, 2.3 ms). ~15-20 ms if argsort -> topk and the route sort is shared. FLAGGED |
| 7 | compressed-K pooling chain per attention layer: CONT of the strided K view + 3 ADD + SCALE + CONT of an already-contiguous RESHAPE (`CONT (reshaped)(reshaped)` [128,1034] - no-op copy) + CPY f32->f16 | 7 x 12 | ~2.5 | 0.08 | the contiguous-RESHAPE CONT (12 launches, 0.3 ms) is a pure no-op: `ggml_cont` of a contiguous tensor; qwen4exp_graph.cpp:863-867. Fold the 3 ADD + SCALE + cast into one pooling kernel for the rest |
| 8 | PLE REPEAT x3 (`op_repeat` [10240,T] 1.08 ms each) + PLE big MUL/ADD | 3 + ~8 | 3.2 + ~6 | 0.3 | REMOVABLE: the three REPEATs broadcast per-hc weights that `k_bin_bcast` can broadcast implicitly (drop `ggml_repeat`, let MUL broadcast). FLAGGED (borderline) |
| 9 | GDN state copies: `CPY [128,128,48] f32` 36 (ssm state write-back) + `CONT [3,10240]` 36 + `CPY [3,10240]` 36 (conv state tail) | 108 | ~1.5 | 0.05 | NECESSARY (recurrent state must land in the state buffer); the conv tail CONT+CPY pair could be one strided CPY (36 launches) |
| 10 | QSA ids: `cpy_scalar_contiguous<float,int>` 24 (SCALE -> i32 index cast) | 24 | 0.03 | 0 | necessary, negligible |
| 11 | `__amd_rocclr_copyBuffer` 111 (2.46 ms; two ~1.7 ms = item 4, the rest 22 us each) + `fillBufferAligned` 73 | 184 | 3.4 | 0.1 | the small ones are memsets/graph-input copies, necessary |

Quantize in prefill: 0.01 ms (mmb takes f16 activations; the 5 `quantize_q8_1` are the 1-token tail). No
f32->f16->f32 round trip found: every `cpy_scalar_contiguous<float,half>` feeds an f16 consumer
(cache / pack / mmb), none is converted back. No bf16->f16 activation convert per request (the bf16
router weights are read by `mul_mat_vec_f<bf16>` directly).

### Decode (HEAD defaults, per token, kernel-sum 39.07 ms, wall 38.42 ms; copy/cont class 52 us = 0.1%)

| rank | what | launches/tok | us/tok | % | verdict |
|---|---|---|---|---|---|
| 1 | `quantize_q8_1` 256/token (grids 2560x256: 146, 1024x256: 96, 6144x256: 12, 10240x256: 2) | 256 | ledger 2437 (inflated); real ~256 x 1.6 = ~410 | ~1.0 | mostly NECESSARY (one Q8_1 activation per Q8_0 GEMV group). The ledger figure is inflated by the shared-expert down-input quantize (40-52 us per launch) sitting on stream 1 under the routed gate_up GEMV; in the token sequence the same grid off-stream is 1.5-2 us. The REMOVABLE part: the ffn_norm activation is quantized twice per MoE layer (once for the shared expert Q8_0 GEMV on stream 1, once for the routed Q4_K path) because the mmvq Q8 memo (mmvq.cu:3359-3400) is keyed on src0 type and only serves stream 0: 47 x ~1.6 us = ~75 us (0.2%). x_norm for attn_gate + qkv is memoised correctly (one quantize, two GEMVs) |
| 2 | router `k_argsort_f32_i32` 48 x 6.3 + `soft_max` 48 x 2.1 | 96 | 301 + 102 = 403 | 1.0 | PARTLY REMOVABLE: full 512-expert sort per layer for a top-k; a block-radix top-k (already in tree for QSA) is ~1 launch at ~2-3 us. ~150-200 us. FLAGGED |
| 3 | 1-row f32 GEMVs `mul_mat_vec_f<float,float,1,256>` 49/token (shared-expert logit / sigmoid gate + router shadows) | 49 | 609 | 1.6 | NECESSARY per op, but 12.4 us per launch for a [2560 x 1] dot product is launch-latency bound; fuse the shared-gate dot into the shared epilogue or the bf16 shadow GEMV (~47 launches, ~400 us). FLAGGED |
| 4 | `argmax_f32<true>` | 1 | 132 | 0.34 | NECESSARY (GPU argmax replaces the host round trip); 132 us for 151936 logits is a single-block reduction - a two-stage argmax would be ~20 us. FLAGGED (borderline) |
| 5 | PLE decode layer: 8 `cpy_scalar<f32,f32>` of the 4 weight-tap views (W[k] columns, [1,10240] strided -> contiguous, re-done every token on constant weights) + 3 unfused grouped_norm (3 `rms_norm_f32<1024,false>` + 3 `op_mul`) + 3 `op_repeat` + `concat_dim0_dense_transpose` | ~20 | ~100 | 0.26 | REMOVABLE: the tap-view copies are of constant weights (hoist to load time: 8 launches, ~18 us); grouped_norm gamma expand (section 1, 3 launches); REPEATs via implicit broadcast (3 launches). ~60 us |
| 6 | `cpy_scalar<f32,f32>` 21/token (30720-element GDN conv-state tails and the PLE taps above) | 21 | 46 | 0.12 | 13 necessary (state), 8 PLE (item 5) |
| 7 | `k_set_rows<float,int,half>` 24 + `<float,int,float>` 12 (K/V cache + ssm state writes) | 36 | 71 | 0.18 | NECESSARY (cache write with position index; the f32->f16 convert is folded into set_rows, no separate cast) |
| 8 | `__amd_rocclr_copyBuffer` 22 (1.7 us each) | 22 | 36 | 0.09 | graph-input / state copies, necessary |
| 9 | `concat_dim0_dense_transpose_f32` 1 (6 us) | 1 | 6 | 0.02 | PLE concat of the 3-token conv window, necessary unless the PLE decode conv is fused like the prefill one (the `ple_conv_kernel` matcher requires `T >= 256`, ple-conv.cu:73, so decode never fuses it) |

No quantize-dequantize round trip, no f32->f16->f32 chain and no bf16->f16 activation convert
exists per token: the only type conversions are inside `k_set_rows<float,int,half>` (cache write) and
the Q8_1 activation quantizes. The hc path quantizes once per site (`quantize_hc_lo_q8_1` 97 = hc_lo
sites) and `hc_down_inject_mixed` reads the f32 source directly.

### Requant arm (q5_k/q6_k) delta

Same 1664 launches/token, same copy / quantize counts (quantize_q8_1 256, hc_lo 97); only the
156 GEMVs change type. Nothing in the copy / conversion inventory is specific to that arm.

## 3. Summary

- Engaged as claimed: norm-rope 36/36, QSA3 gate 12/12 (p4096; T<512 by design unfused), union bitmap,
  rope table, GDN row-split, conv-folded norm, wave gated norm, PLE conv 1/1, hc_combine_norm, all 14
  decode knobs with their capture-asserted site counts (95/96/97/36/48), two-wave routed GEMV, bf16
  router shadows, 8-warp down GEMVs, requant copies 156/156.
- Broken / partial:
  1. PLE `grouped_norm` RMS_NORM+MUL never fuses (3 sites prefill, 3/token decode): gamma RESHAPE not
     expanded before `ggml_rms_norm`. Fix: `ggml_build_forward_expand(gf, gamma)` as in `hc_mix`
     (qwen4exp_graph.cpp:214-218 and 1391). ~5 ms/request.
  2. Routed GLU F16 kernel is Q4_K-only, so blk.2 (Q5_K gate/up in UD-Q4_K_XL) runs the legacy
     two-kernel path at 41.9 ms vs ~10 ms (1.0% of the request). Fix: admit Q5_K in mmb.cu:993 with a
     Q5_K tile loader, or requantize blk.2 gate/up to Q4_K in the loader.
  3. PLE conv fusion is matched but the trailing `ggml_cont(silu)` (graph.cpp:1278) is outside the
     match and costs a 169 MB memcpy (1.7 ms). Fix: delete the `ggml_cont`.
- Largest remaining copy/layout cost: the prefill K/V cache round trip (cast -> strided CPY -> CONT ->
  pack CONT), 16 ms = 0.5%/request, of which ~11 ms is the scalar 1-element/thread `cpy_scalar<half,half>`
  kernel at ~36 GB/s. Decode copies are 0.1%/token; the remaining decode non-GEMV cost is router
  sort+softmax (1.0%), the 1-row f32 GEMVs (1.6%) and the single-block argmax (0.34%).
