# Fusion candidates for qwen4exp T=1 decode graph-node reduction

Analysis only, no edits/build/timing. Source: `PROFILE-38.md`, `rep1_mode1.json` (1871.0
launches/token, exact), `KNOCKOUT-HC.md` (measured own-work + ~2.4 µs/launch critical-path tax),
`HC-AUDIT.md` (bandwidth-floor ranking), `HC-CN-FAST.md` (shipped latency cut, not a launch
reduction), `qwen4exp-launch-cut:bench/exact/LAUNCH-INVENTORY.md` (per-layer kernel sequence,
candidate-1 real-site falsification). Dispatch cost floor: ~1.8 µs/node (`launchcost/RESULTS.md`),
~2.4 µs/launch critical-path (knockout A−B). 48 decoder layers = 36 GDN + 12 QSA.

## 1. Ordered kernel sequence, one layer of each type

### GDN layer (×36)

| # | kernel | launches/layer | mean µs/launch | producer -> consumer |
|---|---|---|---|---|
| 1 | `hc_upmix_row8_exact` | 1 | 20.74 | up-weight GEMV -> `mixed` |
| 2 | `hc_down_inject_mixed` | 1 | 21.01 | down-weight GEMV + inject -> `down_dst`,`inject_dst` |
| 3 | `quantize_hc_lo_q8_1` | ~2.0 | 1.71 | silu(`mixed`) -> `lo_q8` (next layer's upmix input) |
| 4 | `hc_combine_norm_f32_b256` | ~2.0 | 10.03 | residual+block_out+gate -> `xn` |
| 5 | `gdn_ab_exact_f32` | 1 | 6.86 | A/B gate projections |
| 6 | `ssm_conv_step_f32` | 1 | 3.59 | causal conv step |
| 7 | `gated_rms_norm2_kernel` | 1 | 2.28 | GDN-tail norm, already PRODUCER_Q8-fused (writes Q8_1 directly) |
| 8 | `gated_delta_net_cuda_grouped_cols` | 1 | 34.84 | the GDN recurrence |
| 9 | `mul_mat_vec_q`/`mul_mat_vec_f` (q/k/v/out proj) | ~5.3 (shared GDN+QSA) | 25-120 | weight streaming |
| 10 | `quantize_q8_1` (generic) | ~5.3 | 2.09 | NOT producer-fused (see §3.1) |
| 11 | `rms_norm_f32`/`k_bin_bcast`/`cpy_scalar` glue | several, ≤2 µs each | 1-4 | small glue around the above |

### QSA layer (×12)

| # | kernel | launches/layer | mean µs/launch |
|---|---|---|---|
| 1-4 | hc_upmix/hc_down_inject/quantize_hc_lo_q8_1/hc_combine_norm (shared with GDN) | shared | shared |
| 5 | `mul_mat_vec_f<bf16>` (Q/K/V/O proj, no quantize needed) | 6 | 14.96 |
| 6 | `rope_multi` | 4 | 3.15 |
| 7 | `qsa_decode_wmma_partial` | 1 | 29.08 |
| 8 | `qsa_decode_ids`/`qsa_decode_merge` | 2 | ~10 each |
| 9 | `ds4_indexer_score_decode_wmma_kernel` | 1 | 10.84 |
| 10 | `k_topk_block_radix_f32_i32`/`k_argsort_f32_i32` | 2 | 6-9 |
| 11 | `soft_max_f32` | 1 | 1.90 |
| 12 | `moe_fused_combine_shared_kernel_f32` | 1 | 4.55 |

Head/tail (×1/token): `argmax_f32` (129.69 µs) + glue (`concat_dim0_dense_transpose_f32`,
`reduce_rows_f32`, `op_clamp_kernel`, `unary_op_kernel`×5), all <7 µs.

## 2. Fusion candidates

| # | kernels | launches saved/tok | est. ms/tok saved (1.8-2.4 µs/launch + own-work removed) | deps / fork-join crossing | existing coverage | feasibility |
|---|---|---|---|---|---|---|
| A | `hc_combine_norm_f32_b256` wide (256->1024 threads, same block count) | **0** (no launch change) | ~0.2-0.3 (latency cut only, from HC-AUDIT §7 alt. design; not a node removal) | none — single kernel, no stream crossing | HC-CN-FAST already shipped the smaller gamma-prefetch cut (−0.1 ms/tok, gated `LUCE_QWEN_HC_CN_FAST`); the 256->1024 widen is proposed, unimplemented, still bit-identical per HC-AUDIT inspection of `block_reduce` | low risk, not attempted — excluded from launch-count total below (0 nodes removed) |
| B | `quantize_hc_lo_q8_1` folded into `hc_down_inject_mixed` epilogue | 97 | 0.17-0.40 (own-work only, from quantize_hc_lo_q8_1's own-work share of KNOCKOUT-HC arm A-B; launch tax 97×2.4µs≈0.23ms adds to upper end) | in-stream, same producer->consumer edge, no SHARED_OVERLAP fork/join crossing (both run on the GDN/HC stream) | not covered by launch-cut's `rms_norm_mul_q8_1_f32`; silu+quantize formula differs from norm+quantize | medium — silu(mixed) -> lo_q8 write is a small, well-understood op; folding into down_inject's existing epilogue avoids a second kernel's launch, but down_inject's own internals are flagged out-of-scope by another agent in LAUNCH-INVENTORY |
| C | Generic `quantize_q8_1` (256/tok) fused into its producer | 0 real sites found (see below) | 0 (candidate falsified) | n/a | `rms_norm_mul_q8_1_f32` kernel exists (launch-cut branch) but has **zero applicable call sites** — on-box trace found 266 real producer sites, all `SCALE@2560`(101)/`GLU@640`(100)/`RESHAPE@2560`(50, not a real producer)/`MUL@6144,10240`(14, hc-family width, out of scope); none are `RMS_NORM` | **falsified** — kept here only to flag that the kernel is unused and the real producers are different ops |
| D | `GLU@640` (100 sites) producer -> `quantize_q8_1` consumer, fused analogous to launch-cut's norm+Q8 kernel | up to 100 | ~0.24-0.6 (100×2.4µs launch tax ≈0.24ms, plus own-work share of quantize_q8_1's measured 2.09µs×100≈0.21ms if the quantize work itself also collapses) | `g_producer_q8_handoff` is a **single-slot** claim struct, already held by GDN (36 sites) and HC (95 sites) producers; a third claimant needs a multi-slot/keyed redesign — touches shared infra adjacent to another agent's owned code | not covered; requires new multi-slot handoff mechanism, not a drop-in kernel reuse | high effort, best $/launch of the quantize_q8_1 family, flagged "needs scope sign-off" in LAUNCH-INVENTORY |
| E | QSA glue chain: `qsa_decode_ids`+`qsa_decode_merge`+`k_topk_block_radix_f32_i32`+`k_argsort_f32_i32` (4 launches/QSA-layer) | up to 36 (4->1 per layer × 12 layers, 48 total launches collapsing to 12) | ~0.3-0.5 (4×~8µs own-work + ~2.4µs×3 launch-tax savings ×12 layers ≈0.44ms) | all within the QSA per-layer sequence, no SHARED_OVERLAP crossing (single stream per layer) | not covered by any existing fused kernel | medium-high — top-k radix, argsort, and merge are 3 distinct reduction algorithms; correctness surface is larger than a pure norm+quantize fuse, not attempted |
| F | `rope_multi` (48/tok, 4/QSA-layer) folded into preceding Q/K-norm kernel | up to 36 (4->1 per QSA layer if fully collapsed; conservative: fold adjacent pairs only) | ~0.1-0.15 | RoPE needs a second data dependency (position-indexed sin/cos table) beyond the norm's own inputs — low-medium risk but untested | not covered | medium — not attempted, smallest candidate by ms/tok |
| G | hc_down_inject_mixed / hc_upmix_row8_exact fused into one combined down+up GEMV pass | 95-96 (one of the pair eliminated if a single launch computes both) | capped near 0 net — both kernels are legitimate weight streaming (3.32 MB Q8_0 each, ~1.4-1.5x off a 14 µs bandwidth floor per HC-AUDIT §5); fusing only relocates the bytes, does not remove them. Pure launch-tax saving only: ~95×2.4µs≈0.23ms | these two already sit back-to-back in the per-layer sequence (#1,#2 above) with `mixed` as the sole intermediate; no stream-fork crossing | not covered | low-medium — bit-identical in principle (two independent GEMVs into one launch, same math), but HC-AUDIT explicitly flags both kernels' internals as "owned by another agent," so marked out-of-scope for direct implementation here |

## 3. Notes on existing fused/elision machinery (do not re-fuse across these boundaries)

- `PRODUCER_Q8` / `GDN_AB_EXACT`: GDN-tail `gated_rms_norm2_kernel` already writes Q8_1 directly
  (36 sites sealed via `g_producer_q8_handoff`, asserted in `ggml-cuda.cu`). Any new producer-Q8
  fusion (candidate D) must not touch this slot.
- `HC_UPMIX_ROW8` / `SHARED_EPILOGUE`: hc_upmix/hc_down_inject already claim 95 of the handoff's
  slots. Candidate G's two kernels are the HC up/down GEMVs themselves — fusing them is a
  different axis (merging two launches into one), not a new Q8 handoff claim, but still inside
  code flagged as another agent's ownership.
- `SHARED_OVERLAP` stream fork/join: PROFILE-38.md notes busy time is de-overlapped across 4
  concurrent streams. None of candidates B/D/E/F/G cross a fork/join boundary — all sit within a
  single per-layer sequence on one stream, so none of this analysis's savings require touching
  SHARED_OVERLAP's scheduling.

## 4. Ranked by ms saved per unit implementation risk

| rank | candidate | ms/tok | risk | rationale |
|---|---|---|---|---|
| 1 | D — GLU@640 producer fusion | 0.24-0.6 | high (shared infra redesign) | largest real, un-falsified saving; blocked on multi-slot handoff, needs sign-off |
| 2 | B — quantize_hc_lo_q8_1 into hc_down_inject | 0.17-0.40 | medium (touches other agent's kernel) | second-largest, smaller blast radius than D |
| 3 | E — QSA glue chain (topk/argsort/merge) | 0.3-0.5 | medium-high (3 distinct algorithms) | comparable ms to B/D but more correctness surface |
| 4 | G — hc up/down single-launch merge | ~0.23 (launch-tax only; weight bytes not recoverable) | low-medium, but flagged out-of-scope | safest arithmetic, blocked by ownership not difficulty |
| 5 | F — rope_multi fold | 0.1-0.15 | medium | smallest payoff, second data dependency adds risk |
| — | A — hc_combine_norm widen | 0.2-0.3 | low | zero launches removed — latency cut, not a fusion candidate for this task's "nodes removed" framing |
| — | C — generic quantize_q8_1 x RMS_NORM | 0 | n/a | falsified on-box: zero RMS_NORM producer sites exist |

**Total removable launches (candidates with a real, un-falsified launch reduction: B+D+E+G)**:
97 + up to 100 + up to 36 + up to 95 (G's shared-launch merge counts one of the pair) =
**up to ~328 launches/token**, roughly 17.5% of the 1871/token total, estimated
**~0.94-1.73 ms/token** combined (B 0.17-0.40 + D 0.24-0.6 + E 0.3-0.5 + G ~0.23, not
strictly additive since launch-tax estimates share the same ~2.4 µs/launch assumption and GPU
streams overlap — treat as an optimistic upper bound, same caveat PROFILE-38.md's coordinator
note raises about the three buckets not being a strict critical-path budget).
