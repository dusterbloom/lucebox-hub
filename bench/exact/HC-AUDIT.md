# HC family audit — bandwidth floor vs measured, gfx1151 @ 242 GB/s

Model Qwen3.8-Flash-Next, 48 decoder layers, `n_embd = w.n_embd = 2560`, `hc = w.n_hc = 4`
streams, `hc_dim = n_embd*hc = 10240` (confirmed in `server/src/qwen4exp/qwen4exp_graph.cpp:1293`
and matches the `10240`/`320` literals baked into the kernels themselves). At T=1 decode
`n_tokens = 1`. Sources: `bench/exact/PROFILE-38.md` §3 (mean µs/launch, launches/tok),
`bench/exact/rep1_mode1.json` (same numbers, machine-readable), `bench/exact/KNOCKOUT-HC.md`
(own-work upper bound: hc family costs 5.36 ms/token of real work, not just launch overhead).

## 1–2. What each kernel computes, shapes, bytes, FLOPs

### `hc_down_inject_mixed` (mmvq.cu:341, launch mmvq.cu:3054, grid `<<<44, dim3(32,8,1)>>>`)
Two unrelated GEMVs fused into one launch, gated by `g_hc_down_inject`:
- Blocks 0–39 (320 output rows, 8 rows/block × 40 blocks): the HC **down** projection,
  `down[10240,320]` (Q8_0) · `xq[10240]` (Q8_1 activation) → `down_dst[320]`. This is a real
  weight matrix: 320 rows × 10240 cols, Q8_0 = 34 bytes / 32 elements ≈ 1.0625 B/elem →
  **3.32 MB** read, plus `xq` activation 320 × 36 B (q8_1 blocks) = 11.52 KB, write 320×4B=1.28 KB.
- Blocks 40–43 (4 blocks): the HC **inject** projection, `iw[10240,4]` (F32, 163.84 KB) ·
  `x[10240]` (F32, 40.96 KB, read once logically) → `inject_dst[4]` (16 B).
- FLOPs: down GEMV = 320×10240×2 ≈ 6.55 MFLOP; inject GEMV = 4×10240×2 ≈ 81.9 KFLOP. Negligible
  vs bandwidth floor either way (GEMV is always bandwidth-bound at this arithmetic intensity).
- Total bytes/launch ≈ 3.32 MB + 11.5 KB + 163.8 KB + 41.0 KB + 1.3 KB ≈ **3.54 MB**.
- Floor @242 GB/s = 3.54e6 / 242e9 = **14.6 µs**.

### `hc_upmix_row8_exact<false>` (mmvq.cu:307, launch mmvq.cu:3227, grid `<<<1280, dim3(32,8)>>>`)
The HC **up** projection: `weights[320,10240]` (Q8_0, same 320-col/10240-row shape, other
direction of the down matrix) · `lo_q8[320]` (Q8_1 packed activation, fixed 576 B tile per
mmb-w8a8 layout) → 10240 raw dot products, immediately gated (`sigmoid(sum)*xn[row]`) and
summed over the 4 hc streams (grid `e = 2*blockIdx.x + wave/4`, `c = wave%4`, 1280 blocks ×
8 waves = 10240 rows = 4 streams × 2560) into `mixed[2560]`.
- Weight bytes: 10240×320 elements × 1.0625 B ≈ **3.32 MB**.
- Activation: 576 B (lo_q8) + `xn` reread for the gate, 10240×4B = 40.96 KB.
- Write: `mixed[2560]`×4B = 10.24 KB.
- FLOPs: 10240×320×2 ≈ 6.55 MFLOP (GEMV, bandwidth-bound).
- Total bytes/launch ≈ 3.32 MB + 0.58 KB + 41.0 KB + 10.24 KB ≈ **3.37 MB**.
- Floor @242 GB/s = 3.37e6 / 242e9 = **13.9 µs**.

### `hc_combine_norm_f32_b256` (hc-cn.cu:81, launch hc-cn.cu:352, grid `<<<dim3(hc=4,n_tokens=1,1), 256>>>`)
Per hc-stream `c` (one block per stream, 4 blocks total): reads that stream's residual row
`residual[c,:,2560]`, the shared MoE/attn block output `block_out[:,2560]` (same 2560 floats,
re-read independently by all 4 blocks), the per-stream gamma `gamma[c,:,2560]`; computes
`a = residual + block_out * gate(inject[c])` (RMS-norm input), reduces `sum(a²)` across the
block (shared-mem tree, `block_reduce<SUM,256>`), then a second pass writes
`out_xn = rsqrt(mean+eps) * a * gamma` and optionally quantizes to q8/q8_1/bf16 copies.
- No weight matrix — every operand is an activation/parameter vector, not a model weight.
- Bytes per launch (4 blocks, each ~2560-float row): residual 10.24 KB + block_out 10.24 KB
  (re-read ×4 blocks = 40.96 KB total) + gamma 10.24 KB/block (41.0 KB total) + out_res write
  10.24 KB/block (41.0 KB) + out_xn write 10.24 KB/block (41.0 KB) + inject (16 B, negligible).
  Total ≈ 10.24+40.96+41.0+41.0+41.0 ≈ **174 KB** (using residual once, not re-read per block).
- FLOPs: ~2560×4×~6 ops ≈ 61 KFLOP — irrelevant, this is pure memory/latency bound.
- Floor @242 GB/s = 174e3 / 242e9 = **0.72 µs**.

### `quantize_hc_lo_q8_1<true>` (quantize.cu:56, launch quantize.cu:321/324, grid `<<<2, 256>>>`)
Applies `silu(scale*x+bias)` to the 320-element `mixed` vector in place, then quantizes it to
`block_q8_1` (QK8_1=32 → 10 blocks of 36 B, padded grid covers `MATRIX_ROW_PADDING=512` lanes,
only 320 active). This is the producer for `lo_q8` consumed by `hc_upmix_row8_exact`.
- Bytes: read 320×4B=1.28 KB, write (in-place) 320×4B=1.28 KB + q8_1 output 10×36B=360 B.
  Total ≈ **2.9 KB**.
- Floor @242 GB/s = 2.9e3 / 242e9 ≈ **0.012 µs** (sub-microsecond — this kernel is 100%
  launch-overhead-bound, there is nothing to stream).

## 3. Measured (PROFILE-38.md §3, rep1_mode1.json) vs floor

| kernel | mean µs/launch | floor µs | launches/tok | achieved GB/s | ratio meas/floor |
|---|---|---|---|---|---|
| `hc_down_inject_mixed` | 21.01 | 14.6 | 95.0 | ~168 GB/s | 1.44x |
| `hc_upmix_row8_exact` | 20.74 | 13.9 | 96.0 | ~162 GB/s | 1.49x |
| `hc_combine_norm_f32_b256` | 10.03 | 0.72 | 95.0 | ~17 GB/s | **13.9x** |
| `quantize_hc_lo_q8_1` | ~1.75* | 0.012 | 97.0 | ~1.7 GB/s | **~146x** (pure overhead, no real floor) |

\* `quantize_hc_lo_q8_1` isn't broken out in the PROFILE-38 top-15 table (too small); mean
derived from KNOCKOUT-HC.md's hc-family-own-work total (383 launches/tok, 5.36 ms/tok own work)
minus the other three kernels' measured µs/tok, divided by its 97 launches/tok. Treat as an
estimate, not a direct per-kernel trace read.

## 4. Launch geometry and the concrete reason for the ratio

- **`hc_down_inject_mixed`**: grid=44 blocks (40 for the down GEMV at 8 rows/block, 4 for
  inject), block=`dim3(32,8)`=256 threads. 44 blocks on a 40-CU GPU is close to 1 block/CU —
  reasonable occupancy for a GEMV. The 1.44x ratio is normal GEMV overhead (warp-level
  `vec_dot_q_mmvq` + `warp_reduce_sum`, one sync per block, no redundant weight re-reads).
  **Legitimate weight streaming** — 3.32 MB of real Q4/Q8_0-quantized down-weight is read every
  call; fusing this elsewhere would not remove that traffic, only relocate it.
- **`hc_upmix_row8_exact`**: grid=1280 blocks × `dim3(32,8)`=256 threads = well-shaped for 40 CUs
  (32 blocks/CU). 1.49x ratio is ordinary GEMV + warp-reduce + shared-mem cross-wave combine
  overhead. **Also legitimate weight streaming** — reads the full 3.32 MB up-weight matrix.
- **`hc_combine_norm_f32_b256`**: grid=**only 4 blocks** (`dim3(hc=4, n_tokens=1, 1)`) on a
  40-CU GPU — **36 of 40 CUs sit idle for the whole kernel**. `__launch_bounds__(256,4)` caps
  occupancy per CU at 4 blocks, which is irrelevant here since there are only 4 blocks *total*.
  Inside each block: a 6-iteration unrolled strided column loop (`KP=6`, 256 threads × 2
  elements/iter ≈ 3072 capacity for 2560 real columns, ~17% waste), one `block_reduce<SUM,256>`
  (shared-mem tree + `__syncthreads`), then a *second* unrolled loop re-reading `gamma` and
  writing `out_xn`/bf16/q8/q8_1 variants. The kernel is latency-bound, not bandwidth-bound:
  174 KB of real traffic takes 0.72 µs at floor bandwidth, but the measured 10.03 µs is
  dominated by kernel-launch latency + two serial `__syncthreads` round trips on a grid that
  can't hide any of it behind other active CUs. This is the textbook "tiny grid, serial
  reduction, GPU mostly idle" failure mode.
- **`quantize_hc_lo_q8_1`**: grid=2 blocks × 256 threads (`MATRIX_ROW_PADDING=512`/256), only
  320 of 512 lanes do real work (silu + 36-byte q8_1 write each). At ~2.9 KB of traffic this is
  pure dispatch/launch overhead — there is no bandwidth floor worth computing; its entire
  measured cost is the fixed ~2–2.5 µs HIP-graph-replay launch tax (matches KNOCKOUT-HC.md's
  measured ~2.4 µs/launch for the whole hc family, arm A−B).

## 5. Weight-matrix check (legitimacy of the time)

| kernel | reads a weight matrix? | which / size |
|---|---|---|
| `hc_down_inject_mixed` | **yes** | `down[10240,320]` Q8_0 (3.32 MB) + `iw[10240,4]` F32 (164 KB) |
| `hc_upmix_row8_exact` | **yes** | `up[320,10240]` Q8_0 (3.32 MB, same shape class as down, transposed role) |
| `hc_combine_norm_f32_b256` | **no** | only `gamma[2560]`/stream — a norm parameter, not a weight matrix; rest is activations |
| `quantize_hc_lo_q8_1` | **no** | pure activation quantization, no weight |

So ~2× 3.3 MB of the hc family's bytes are real weight streaming that would simply move to
whatever kernel absorbed the down/up GEMV if fused elsewhere — that portion of
`hc_down_inject_mixed`/`hc_upmix_row8_exact` time is not "recoverable" in any fusion, only in
better GEMV efficiency (already ~1.5x off floor, modest headroom). `hc_combine_norm_f32_b256`
and `quantize_hc_lo_q8_1` touch no weights at all — their time is 100% fusion/parallelism
overhead and is the only unambiguously recoverable bucket.

## 6. Ranking by recoverable ms/token = (measured − floor) × launches/token

| rank | kernel | measured µs | floor µs | launches/tok | recoverable ms/tok |
|---|---|---|---|---|---|
| 1 | `hc_combine_norm_f32_b256` | 10.03 | 0.72 | 95.0 | **0.882** |
| 2 | `hc_upmix_row8_exact` | 20.74 | 13.9 | 96.0 | 0.657 |
| 3 | `hc_down_inject_mixed` | 21.01 | 14.6 | 95.0 | 0.609 |
| 4 | `quantize_hc_lo_q8_1` | ~1.75 | ~0.01 | 97.0 | ~0.169 |

Sum ≈ 2.32 ms/tok. This is a *bandwidth-floor* lower bound on recoverability; it undercounts
against KNOCKOUT-HC.md's measured 5.36 ms/tok own-work total because down/upmix are also
latency/compute-bound beyond pure bytes (warp-reduce depth, 44/1280-block dispatch), not just
bandwidth. `hc_combine_norm_f32_b256` is unambiguously #1 both by this metric and by ratio
(13.9x vs 1.4–1.5x for the two GEMVs) — and it's the safest target because it touches **no**
weight matrix, so any speedup is pure win, not relocated weight-streaming time.

## 7. Proposed rewrite: `hc_combine_norm_f32_b256`

**Problem**: 4 blocks total on a 40-CU GPU; each block pays full kernel-launch + two
`__syncthreads` reduction-round-trip latency to process a trivial 174 KB of data. The fix must
not change the floating-point reduction order — `block_reduce<SUM,256>` does a shared-mem
binary tree over the 256 per-thread partial sums (shfl-reduce within each of 8 warps, then a
balanced tree combine of the 8 warp sums in increasing warp-id order). Splitting that 256-wide
tree at a power-of-two boundary and recombining the two halves with ordinary float addition
reproduces the identical bit pattern, because a balanced binary tree over 256 leaves computed
as (sum of leaves 0–127) + (sum of leaves 128–255) is exactly what the tree already computes at
its top level — no reassociation.

**Design** — split each of the 4 (stream) blocks into 2 half-blocks of 128 threads (8 blocks
total instead of 4; still small, but doubles parallelism and, more importantly, lets the
reduction kernel finish in roughly half the per-block latency since each half-block does 3
unrolled iterations instead of 6 and a 128-wide tree instead of 256-wide):

1. **Kernel A `hc_combine_f32_half`** — grid `<<<dim3(hc=4, 2, n_tokens), 128>>>` (y=0/1 selects
   `tid_global = threadIdx.x + 128*blockIdx.y`, otherwise identical column mapping
   `col = (tid_global + k*256)*2` and identical `hc_add_rn(r, hc_mul_rn(m, w))` formula — bit
   for bit the same per-element arithmetic as today, just computed by a different block).
   Writes `out_res`/`out_xn`'s unnormalized `a` values directly (already final — no change from
   today), and writes its half's partial `sum(a²)` to a tiny scratch buffer
   `partial[hc][2]` (32 bytes total).
2. **Kernel B `hc_combine_norm_finish`** — grid `<<<hc=4, 256>>>` (or even one block of `4*256`
   threads doing all 4 streams): reads `partial[c][0] + partial[c][1]` (that addition order
   matches the tree's top-level split exactly), computes `mean`/`scale` once, then re-reads the
   `a` values just written by kernel A from `out_res` (bit-identical, no recompute — a pure
   reload) together with `gamma`, and writes `out_xn`/bf16/q8/q8_1 exactly as today's second
   loop does.

**Net effect**: launch count for this op goes from 95/tok to 190/tok (+95 launches/tok,
≈+0.23 ms/tok dispatch tax at the measured ~2.4 µs/launch), but kernel A's per-launch latency
should drop roughly in proportion to the halved reduction depth and iteration count (10.03 µs →
an estimated ~6 µs based on the loop/reduction-depth ratio), and kernel B is tiny (256 threads,
no loop over `KP=6`, just a read-modify-write + one more small loop) and should run at
launch-floor (~2 µs). Projected: `95*(6+2) = 760 µs/tok` vs today's `95*10.03 = 953 µs/tok` —
a modest ~0.2 ms/tok net win after paying the extra-launch tax, confirming this single kernel's
headroom is real but small; the bigger win is eliminating the *second-pass column loop's*
redundant `gamma` global read by caching it in kernel A's registers across the two passes,
which kernel B cannot do once split — so this split-kernel design trades launch count for
reduction latency and is only worth it if kernel B's extra 95 launches/tok stay under the
~2.4 µs HIP-graph-replay floor measured in KNOCKOUT-HC.md (arm A−B / 383 launches). If not, the
higher-leverage alternative is to keep one kernel but enlarge `HC_CN_BLOCK2` from 256 to 1024
threads (same single block per stream, same tree-reduction algorithm family, no launch-count
change) so the column loop drops from `KP=6` to `KP=2` iterations and the warp-level reduction
tree depth grows by only one level — strictly less per-block latency with zero added launches,
at the cost of lower CU-count parallelism (still 4 blocks, now wider) which doesn't matter since
the GPU was never CU-bound here, only latency-bound.
