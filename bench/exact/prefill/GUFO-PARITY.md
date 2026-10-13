# Gufo vs lucebox prefill parity — lucebox4, Qwen3.8-Flash-Next UD-Q4_K_XL

Date: 2026-10-13. Box: lucebox4 (Strix Halo, two ROCm agents: `gfx1151` iGPU,
96 GiB apparent VRAM via `amdgpu.gttsize=98304`, and a newly-present discrete
`gfx1201`, 32 GiB — not present in this box's Sep-29 gufo-parity session, see
"Build notes" below). ROCm 7.2.2. Both engines pinned to the `gfx1151` device.

## Parity contract

- Same file: `/home/duster/lucebox-models/Qwen3.8-Flash-Next-UD-Q4_K_XL-00001-of-00004.gguf`
  (shards 2-4 alongside it), both engines.
- Same box, same GPU (`gfx1151`, forced via `ROCR_VISIBLE_DEVICES=1` for
  lucebox and `HIP_VISIBLE_DEVICES=1` for gufo — both hide the discrete
  `gfx1201` so neither engine can accidentally pick it).
- Same context budget: `--max-ctx 20480` (lucebox), gufo's `bench` tool sizes
  context from the requested prompt length automatically (no server KV
  budget to mismatch for a single pp-only call).
- Same KV type: both run their architecture's default K/V type (no
  `--cache-type-k/v` override on either side — neither CLI needed one; this
  is a single pp-timing call, not a generation run, so KV write cost is the
  only thing that matters and both default identically for this arch).
- Same sampling: greedy (temperature 0), `max_tokens`/`-n` = 1 (gufo) or
  `-n 0`/single-token-out (lucebox), cold server/process each session,
  **distinct random prompt per repetition** so neither engine can prefix-cache
  (lucebox: `restore=false slot=-1 prefix_len=0` verified in the server log
  for every rep; gufo's `bench` tool launches a fresh synthetic prompt per
  repetition by construction).
- 3 repetitions per point, median reported.
- p = 4096 and p = 16384.

## Parity numbers

| Engine | p | prefill tok/s (3 reps) | median |
| --- | ---: | --- | ---: |
| gufo (GPU-only, local fork rebuilt) | 4096  | 1741.68 ± 1.60 (tool-reported mean±stderr, effectively the median) | **1741.7** |
| gufo (GPU-only, local fork rebuilt) | 16384 | 1681.07 ± 3.83 | **1681.1** |
| lucebox (`build1151/luce_server`)   | 4096  | 1118.7, 1135.6, 1141.6 | **1135.6** |
| lucebox (`build1151/luce_server`)   | 16384 | 1193.3, 1247.0, 1272.2 | **1247.0** |

lucebox's p=16384 number (1247 tok/s) lands exactly inside the box's own
recorded 1,200–1,250 tok/s range in `server/docs/QWEN4EXP.md:82`, which
cross-validates the harness. gufo is **1.35x** faster at p=16384 (+34.9%)
and **1.53x** faster at p=4096 (+53.4%) — the gap is *larger* at shorter
prompts, consistent with lucebox carrying more fixed per-call overhead that
amortizes at longer p.

Flags used:
- gufo: `HIP_VISIBLE_DEVICES=1 ./build/release/gufo bench -m <shard1.gguf> -p 4096,16384 -n 0 -r 3 -v`
- lucebox: `ROCR_VISIBLE_DEVICES=1 ./build1151/luce_server <shard1.gguf> --port 19080 --max-ctx 20480`,
  then a small Python client posting distinct synthetic prompts with
  `stream:true`, `temperature:0`, `max_tokens:1`, reading `usage.prompt_tokens`
  and wall time (cross-checked against the server's own `prefill=Xs` log field
  — they agree to within noise).

IQ4_XS was not run: UD-Q4_K_XL is the one quant gufo's `bench` tool could load
cleanly on `gfx1151` once pointed at the right device (see below), and it is
exactly the quant lucebox's own recorded numbers use, so it is the stronger
parity point — no need for a second quant.

## Where the gap lives (by mechanism, not by profiling — see caveat)

`rocprofv3` kernel trace (step 5) was not run — given the mechanism is
already pinned down precisely by source + lucebox's own docs below, and time
budget, it was skipped as the lower-value remaining step. The claim below is
grounded in source comments and measured numbers the gufo authors already
put in their own code, not a fresh trace on this box.

**1. Dense projections: gufo adaptively dequants wide dense GEMMs to F16 and
runs WMMA instead of int8 dp4a; lucebox's Q4_K dense path does not have this
switch.**

`src/models/qwen38_flash_next/kernels/rocm/executor.cpp:126-134` (gufo, local
fork):
```
constexpr std::uint32_t kDenseF16MinRows = 2048;
constexpr std::uint32_t kDenseF16MaxCols = 2560;
```
with the comment directly above (lines 118-125) giving gufo's own
per-shape measurements: at 2048 tokens, `16384x2560` dense GEMM goes from
5.7ms (int8 W8A8) to 4.6ms (F16 WMMA after dequant) — a 19% win — and
similarly for `13312x2560` (4.3→3.9ms) and `10240x320`/`2560x2560`/`2560x640`.
Narrow-but-deep shapes (`2560x6144`, `640x2560`, `320x10240`) keep the int8
route because WMMA loses there. This is a **per-shape, empirically tuned
switch** (`executor.cpp:1742`: `w.rows >= kDenseF16MinRows && w.cols <= kDenseF16MaxCols`),
applied automatically at prefill (`prefill_phase` is true for the whole
chunk) regardless of the weight's stored quant type (Q4_K, Q5_K, Q6_K all
dequant-to-F16 then WMMA when the shape qualifies).

lucebox's own documentation (`server/docs/QWEN4EXP.md:63`) states its WMMA
F16 path is scoped to **`Q8_0` and `bf16`** tensors only: *"gfx1151 kernels:
MMB bf16 and Q8_0 -> F16 WMMA GEMMs"*. UD-Q4_K_XL's dense tensors are K-quants
(Q4_K/Q5_K/Q6_K), not Q8_0, so by lucebox's own stated scope those dense
projections stay on the int8 MMQ path
(`server/deps/llama.cpp/ggml/src/ggml-cuda/mmq.cu`,
`template-instances/mmq-instance-q4_k.cu`) for every prefill call, at every
row count — there is no equivalent "this shape is wide enough that dequant+
WMMA wins" escape hatch for K-quants. This is the single largest, most
concrete, most citable mechanism difference and the most likely primary
contributor to the ~35-53% gap, since the dense trunk (attention qkv, o-proj,
shared-expert gate/up/down, hyper-connection mixer) runs every token through
these exact wide/shallow shapes gufo's own comment calls out as WMMA wins.

**2. Routed MoE experts: gufo re-derives the GEMM tile width from the actual
per-expert token histogram on every forward call; lucebox's mmid path picks
one tile width from the aggregate column count.**

gufo (`executor.cpp:1944-1995`, `RoutedHints`): after the router produces
per-expert counts, it builds a tile map per (expert, 16-row-padded bucket),
then picks `kRoutedTileRowsNarrow=16` vs `kRoutedTileRowsWide=48`
(`executor.cpp:137-150`, `RoutedTileRows`) from the actual
`slots >= 16 * num_experts` fill ratio, and for prefill batches ≥1024 tokens
with the wide tile, **also** builds 64-token and (if it pays off, `tiles_128
* 4 <= routed_64_tiles_ * 3`) 128-token maps per layer
(`executor.cpp:3164-3179`) so the gate/up/down GEMMs can use whichever width
best amortizes weight dequantization for *that specific layer's actual
routing skew*, every call. Large buckets additionally fuse gate+up into one
paired kernel with SwiGLU applied without materializing the gate tensor
(`executor.cpp:3155-3176`, `RoutedGatedF16Gemm`).

lucebox's grouped-expert path (`mmid.cu:28-134`, the `ids_src1`/
`expert_bounds` compaction kernel) sorts tokens by expert the same way, and
`mmq-tile-selection.h`'s `ggml_cuda_mmq_select_x` does pick a tile width
adaptively — but from the batch's aggregate column count, not from a
per-layer, per-expert histogram re-evaluated every call, and there is no
analogue to gufo's fused gate+up SwiGLU-without-materializing-gate epilogue
or its 16/48/64/128 four-way per-call width ladder.

## Build notes (for anyone rebuilding gufo GPU-only here)

- `cmake --preset release` needs ICU/CURL/PNG/JPEG/libwebp/libsharpyuv-dev;
  only ICU/CURL/PNG/JPEG were already vendored under
  `~/gufo-parity/deps/usr` from an earlier session. `libwebp-dev` and
  `libsharpyuv-dev` were fetched with `apt-get download` (no sudo, no
  install) and `dpkg-deb -x`'d into that same deps tree; their `.pc` files'
  `prefix=/usr` had to be sed'd to the deps path for pkg-config to resolve
  headers.
- **The GPU-only (`GUFO_ENABLE_NPU=OFF`) build does not link.** Our local
  fork's `executor.cpp:1261` calls `NpuGemm::ABytes()/NewBuffer()/CBytes()`
  etc. unconditionally inside a runtime `if (npu != nullptr)` branch, but
  `src/models/qwen38_flash_next/CMakeLists.txt:65-76` only compiles
  `npu_gemm.cpp` (which needs XRT headers we don't have, per the task's hard
  rule) when `GUFO_ENABLE_NPU=ON`. README's "Without `GUFO_NPU_DIR` the build
  runs GPU-only" does not hold for this HEAD without a stub. Fixed by adding
  `kernels/rocm/npu_gemm_stub.cpp` (13 trivial dead-code bodies, since
  `model.npu()` is always null without NPU) and an `else()` branch in the
  CMakeLists to compile it when NPU is off. This is a build-only addition,
  not a functional change — worth upstreaming as a 1-file fix.
- **lucebox4 now has a second GPU** (`gfx1201`, 32 GiB VRAM) that was not
  present in the Sep 29 gufo-parity session on this box (that session's
  server log shows exactly one ROCm device). Both engines default to device
  ordinal 0, which is now the *discrete* `gfx1201` — gufo's `bench` tool
  prints `hipMalloc failed for blk.19...` (runs out of the 32 GiB dedicated
  VRAM aperture, confirmed via `hipMemGetInfo`/`hipExtMallocWithFlags` probes
  — neither can exceed it, it's a hard BIOS/kernel carve-out,
  `amdgpu.gttsize=98304` only governs the *other* device's GTT pool), and
  lucebox's `luce_server` prints a clean `cudaMalloc failed: out of memory`
  with a hint to pass `--target-device hip:1`. **Both engines must be pinned
  to the Strix iGPU explicitly** (`HIP_VISIBLE_DEVICES=1` for gufo,
  `ROCR_VISIBLE_DEVICES=1` for lucebox — `--target-device hip:1` and
  `--target-device auto` on lucebox's `build1151/luce_server` both load fine
  but then crash silently, no log line, on the first real request; hiding the
  second GPU from the process entirely via `ROCR_VISIBLE_DEVICES=1` so the
  server falls back to its old single-device `hip:0` path is what actually
  worked). Flag this to whoever added the discrete GPU to this box: the
  multi-GPU `--target-device hip:1` path in `build1151/luce_server` is
  broken for qwen4exp and needs its own bug report/test.

## Minimal-LOC fix for lucebox (ggml op level, benefits any model)

Two independent, additive changes to `ggml-cuda`, neither requiring new
tensor metadata:

1. **Dense GEMM: add a measured-shape escape to F16 WMMA for wide/shallow
   K-quant matrices at prefill**, mirroring gufo's exact heuristic. In
   `ggml-cuda/mmq.cu`'s dispatch (where it currently always routes
   `Q4_K`/`Q5_K`/`Q6_K` dense `mul_mat` through MMQ), add: if
   `src0->ne[1] (rows) >= 2048 && src0->ne[0] (cols... i.e. k) <= 2560` (the
   same two constants gufo measured) **and** the op is in the prefill batch
   (large `ne[1]` on src1), dequantize the weight tile to F16 once (reusing
   the existing Q8_0→F16 WMMA kernel lucebox already ships per
   `QWEN4EXP.md:63`, just widening its type gate to include `Q4_K`/`Q5_K`/
   `Q6_K`) and run the existing WMMA F16 GEMM instead of MMQ. This is a
   dispatch-condition change plus reusing an existing kernel — no new kernel
   code, bounded to a handful of lines in the op-selection function, and it
   is a ggml-level change so every K-quant model prefill benefits, not just
   qwen4exp.
2. **Routed experts: re-derive the MMQ tile width from the per-expert
   `expert_bounds` histogram already computed by `mmid.cu`'s sort**, instead
   of `ggml_cuda_mmq_select_x`'s aggregate-column estimate. `expert_bounds`
   is already on the device after the existing compaction kernel
   (`mmid.cu:28-134`); add a tiny host-side readback (already synchronous
   today around `ggml_cuda_op_mul_mat_id`) to pick `tile_x` per the same
   16/48 narrow/wide rule gufo uses (`slots >= 16*num_experts`), falling
   back to the current aggregate estimate when the readback would cost more
   than it saves (e.g. skip for small batches). This is the direct MoE
   analogue of fix 1: same mechanism (per-shape measured switch instead of
   one-size-fits-all), same bounded scope (the existing tile-selection
   function gains a second, more specific candidate).

Both are additive dispatch-condition changes on top of kernels lucebox
already has (WMMA F16 GEMM, adaptive tile selection) — not new kernels, not
an architecture change, and both apply to every ggml user of `mul_mat`/
`mul_mat_id` with K-quant weights, not just qwen4exp.

## Correction (2026-10-13, checked against the GGUF)

The "dense K-quant tensors stay on int8 MMQ" explanation above is **refuted**. Every dense tensor in UD-Q4_K_XL is Q8_0. That covers attn_*, ssm_out, the shexp triple and hc_*. The two exceptions are indexer.q_proj (BF16) and per_layer_token_embd (IQ4_NL). So the dense projections are already eligible for lucebox's Q8_0->F16 WMMA MMB path.

The only K-quant and Q5_1 tensors are the routed experts. The leading hypothesis is therefore the routed MoE. gufo's `RoutedF16GEMMKernel` decodes the weights inside the kernel into FP16 WMMA and picks the tile width from a histogram of tokens per expert. Lucebox instead falls back to integer MMQ for quantized 512-expert mul_mat_id. This is unproven until a rocprofv3 kernel trace shows where the gap is.

## Kernel trace (p=4096)

Box: lucebox4, Strix iGPU (`gfx1151`), discrete `gfx1201` hidden (same pinning as
the parity run). `rocprofv3 --kernel-trace --stats -f csv` from the extracted
`/home/duster/gufo-parity/rocprof-root/opt/rocm-7.2.2/bin/rocprofv3`. Same
binaries as the parity run (`/home/duster/gufo-parity/src/build/release/gufo`,
`/home/duster/qwen4exp-exact-stack/build1151/luce_server`), same GGUF
(`/home/duster/lucebox-models/Qwen3.8-Flash-Next-UD-Q4_K_XL-00001-of-00004.gguf`).
Scratch: `~/qwen4exp-f16/scratch/gufo/prof/`. GPU was idle (0-3% busy, no
resident server) before each run; no other processes were touched.

**gufo**: `gufo bench -m <gguf> -p 4096 -n 0 -r 1 -v` under rocprofv3
(`HIP_VISIBLE_DEVICES=1`). One repetition only, so the trace has exactly two
phases: model load (fillBufferAligned/copyBuffer, no GEMM kernels, ends at
t+20.2s) then the single pp4096 call (first `*GEMMKernel` launch to the last
kernel end). The prefill-only window used below is that second phase:
**5.574 s wall, 6926 kernel launches, 3.76% idle** (sum of kernel durations /
wall). Tool-reported throughput under instrumentation was 1532.5 tok/s (vs
1741.7 unprofiled — rocprofv3 serializes launches, so this run's wall time
is inflated and not comparable 1:1 to the parity numbers; only the
*composition* below is trustworthy).

**lucebox**: cold `luce_server` under rocprofv3 (`ROCR_VISIBLE_DEVICES=1`,
`--max-ctx 20480`), single request via a scratch client
(`single_pp_client.py`, distinct random-token prompt, `temperature:0`,
`max_tokens:1`, `stream:true` — not committed) calibrated to 587 words ->
4092 prompt tokens (gufo's tokenizer-exact `-p 4096` isn't available through
the HTTP API; 4092 vs 4096 is noise). The trace's first ~87s of wall time is
dominated by on-disk model streaming (103 GiB GGUF); a clean **5.17 s gap**
separates that from the single forward pass. The prefill-only window is
everything after that gap: **4.556 s wall, 7319 kernel launches, 16.78%
idle**. Server-reported `prefill_ms=4618.5` for 4092 tokens (886 tok/s under
instrumentation vs 1135.6 unprofiled median — same profiling-overhead
caveat as gufo).

### Bucket table (% of each engine's own prefill wall time)

| Class | gufo ms | gufo % | lucebox ms | lucebox % | Δ (luce − gufo, pp) |
| --- | ---: | ---: | ---: | ---: | ---: |
| Dense GEMMs | 2051.4 | 36.80 | 1177.9 | 25.85 | **−10.95** |
| Routed gate/up GEMM | 732.4 | 13.14 | — | — | — |
| Routed down GEMM | 1006.8 | 18.06 | — | — | — |
| Routed experts (all) | 1739.2 | 31.20 | 1032.6 | 22.66 | **−8.54** |
| Expert sort/scatter/gather | 23.7 | 0.42 | 359.3 | 7.89 | **+7.47** |
| GDN recurrence | 506.5 | 9.09 | 314.1 | 6.89 | −2.20 |
| Attention | 291.2 | 5.22 | 226.1 | 4.96 | −0.26 |
| hc_*/norms/elementwise | 715.1 | 12.83 | 669.4 | 14.69 | +1.86 |
| Other | 37.3 | 0.67 | 12.4 | 0.27 | −0.40 |
| **Idle (wall − Σkernels)** | 209.5 | 3.76 | 764.6 | 16.78 | **+13.02** |
| **Wall** | **5573.8** | 100 | **4556.4** | 100 | — |
| Launch count | 6926 | — | 7319 | — | — |

(gufo's routed gate/up vs down split comes from `RoutedF16GEMMKernel`'s
`WeightType` template param: 7/8 = gate/up tiles [48/64], 12/13 = down tiles
[64/128]; lucebox's routed-expert number above is not split gate/up vs down
— see kernel list below for why.)

### Top 15 kernels — gufo (post-load window)

| Total ms | Calls | Mean µs | Kernel |
| ---: | ---: | ---: | --- |
| 1739.18 | 384 | 4529.1 | `RoutedF16GEMMKernel` (WeightType 7/8/12/13, gate-up+down) |
| 1622.21 | 1256 | 1291.6 | `DenseF16GEMMKernel` |
| 369.73 | 144 | 2567.6 | `GdnRowSplitKernel` |
| 329.02 | 188 | 1750.1 | `HcCombineMoeF16Kernel` |
| 290.88 | 48 | 6060.1 | `WmmaCausalAttentionKernel` |
| 274.76 | 768 | 357.8 | `W8A8BlockedWmmaGEMMKernel` (dense, narrow-but-deep int8 route) |
| 247.03 | 196 | 1260.4 | `HcCombineVec4Kernel` |
| 95.67 | 48 | 1993.2 | `WKQuantA8BlockedWmmaGEMMKernel` |
| 92.05 | 376 | 244.8 | `HcMixEpilogueF16Kernel` |
| 87.38 | 144 | 606.8 | `GdnEpilogueKernel` |
| 46.23 | 72 | 642.0 | `Cijk_...` (Tensile/hipBLASLt dense GEMM) |
| 27.08 | 144 | 188.1 | `GdnPrepKqKernel` |
| 21.24 | 144 | 147.5 | `SsmConvBoundaryKernel` |
| 18.39 | 47 | 391.2 | `RmsNormKernel` |
| 16.34 | 55 | 297.1 | `QuantizeQ8TiledVec4Kernel` |

### Top 15 kernels — lucebox (post-gap window)

| Total ms | Calls | Mean µs | Kernel |
| ---: | ---: | ---: | --- |
| 814.93 | 301 | 2707.4 | `mmb_q8f16_kernel` (dense Q8_0 -> F16 WMMA, MMB) |
| 542.26 | 94 | 5768.8 | `mmb_routed_glu_kernel` (routed gate+up fused SwiGLU, F16 WMMA, MMB) |
| 325.01 | 235 | 1383.0 | `mm_ids_helper` (mul_mat_id expert sort/compaction, `mmid.cu`) |
| 282.00 | 94 | 3000.0 | `mmb_routed_kernel` (routed down-proj, F16 WMMA, MMB) |
| 251.28 | 72 | 3490.0 | `gated_delta_net_tiled_cuda` (GDN recurrence) |
| 198.68 | 630 | 315.4 | `mul_mat_q` (int8 MMQ, `mmq.cu` — see below) |
| 180.38 | 46 | 3921.4 | `hc_combine_norm_moe_f32_b256` |
| 136.04 | 95 | 1432.0 | `hc_gate_mix_kernel` |
| 132.47 | 144 | 919.9 | `hc_combine_norm_f32_b256` |
| 110.80 | 12 | 9233.2 | `qsa3_attn_kernel` |
| 97.56 | 142 | 687.0 | `mmb_small_m_kernel` |
| 85.84 | 93 | 923.0 | `mmb_w8a8_kernel` |
| 76.18 | 119 | 640.2 | `mmb_f32split_kernel` |
| 71.63 | 12 | 5968.9 | `qsa3_rows_kernel` |
| 68.14 | 36 | 1892.8 | `gated_rms_norm2_kernel` |

### Hypothesis verdict: refuted in its strong form, confirmed in a narrower form

lucebox's routed-expert path is **not** uniformly int8 MMQ. Most of the
routed-expert kernel time (824.3 ms of 1032.6 ms, 80%) already runs through
lucebox's own F16 WMMA path (`mmb_routed_kernel` / `mmb_routed_glu_kernel`,
the fused gate+up+SwiGLU analogue of gufo's `RoutedGatedF16Gemm`), dispatched
from `ggml-cuda.cu:3773` (`if (ggml_cuda_mmb_supported_mmid(...)) { ...
ggml_cuda_mul_mat_id_mmb(...); return; }`) inside `ggml_cuda_mul_mat_id()`
(`ggml-cuda.cu:3682`). `mmb_quant_type()` (`mmb-quant.cuh:220`) lists `Q4_K`
and `Q5_1` as MMB-eligible, so the K-quant/Q5_1 routed weights are not
blanket-excluded from WMMA the way the uncorrected hypothesis assumed.

The remaining 198.7 ms (19%) of routed-expert time **does** fall through to
classic int8 MMQ: when `ggml_cuda_mmb_supported_mmid()` returns false for a
given mul_mat_id call (shape/tile-count gate in `mmb.cu:898-911`, e.g.
`n_used > 64` or the tile-index-overflow check `(T*n_used)>>16 >= 1024`),
dispatch falls to `ggml_cuda_should_use_mmq(...)` at `ggml-cuda.cu:3780`,
which calls `ggml_cuda_mul_mat_q()` (`ggml-cuda.cu:3782`) ->
`ggml_cuda_mul_mat_q_switch_type()` (`mmq.cu:78`) -> `case GGML_TYPE_Q4_K:
mul_mat_q_case<GGML_TYPE_Q4_K>(...)` (`mmq.cu:146-147`) for the gate/up
weight at this batch size. So the MMQ fallback is real but partial, gated by
a tile/count condition rather than by quant type.

The two clearest trace-level contributors to the gap instead are:

1. **Dense GEMMs are a bigger relative share of gufo's wall time than
   lucebox's** (36.80% vs 25.85%, −10.95 pp) — consistent with gufo's
   adaptive F16-WMMA dequant switch applying to *more* dense shapes than
   lucebox's narrower Q8_0-only `mmb_q8f16_kernel`/`mmb_w8a8_kernel` gate
   (lucebox still has a non-trivial 274.76+85.84 ms narrow-but-deep int8
   dense fallback on the gufo side too — `W8A8BlockedWmmaGEMMKernel` — so
   this isn't a clean WMMA-vs-MMQ split either; it is a relative-weighting
   difference, not a missing capability).
2. **Expert sort/scatter and idle time are both far larger on lucebox.**
   `mm_ids_helper` (lucebox's host-driven `mmid.cu` compaction kernel) costs
   7.89% of wall time vs gufo's in-kernel `ExpertCountsKernel`-style
   bookkeeping at 0.42% (+7.47 pp), and lucebox's idle time (gaps between
   kernel launches — mostly the host-side readback/sync this compaction
   needs) is 16.78% of wall vs gufo's 3.76% (+13.02 pp). Combined,
   non-GEMM/non-recurrence overhead (sort/scatter + idle) is **24.67% of
   lucebox's prefill wall time vs 4.18% of gufo's** — the single largest
   bucket-level delta in this trace, and it matches the minimal-LOC fix #2
   proposed above (reuse the already-computed `expert_bounds` instead of a
   separate host round trip).

Caveat: rocprofv3 instrumentation inflates both engines' wall time
non-uniformly (gufo's unprofiled 1741.7 tok/s drops to 1532.5 profiled;
lucebox's unprofiled 1135.6 median drops to ~886 profiled), so the *absolute*
1.35-1.53x gap from the parity run should not be read directly off this
trace's wall times — only the per-class composition and idle/sort-scatter
shares, which are consistent with (and partly explain) the unprofiled gap
being larger at shorter p.

## Idle and sort root cause

Box: lucebox4, Strix iGPU pinned (`ROCR_VISIBLE_DEVICES=1`), GPU confirmed
idle (0-3%) before and after. Server: cold `build1151/luce_server` launched
with `LUCE_MMID_TELEMETRY=1`, same flags as the parity run
(`--port 19080 --max-ctx 20480`), one cold p≈4111 prefill via the existing
`lucebox_pp_bench.py` client (distinct random prompt, `max_tokens:1`). Scratch:
`~/qwen4exp-f16/scratch/gufo/idle/` on the box. Server was shut down after the
run; GPU returned to idle.

### 1. `mm_ids_helper` does NOT do a host round-trip — the claim is refuted

Grepped every `cudaStreamSynchronize` / `cudaMemcpy*ToHost` /
`ggml_backend_synchronize` reachable from a qwen4exp prefill forward:
`mmb.cu`, `mmid.cu`, `mmq.cu`, `mix-wmma-moe.cu`, `mmf.cu` have **zero** host
syncs or D2H copies. The only host round-trip in the whole `mul_mat_id`
dispatch tree is the generic `sync_fallback` path in
`ggml-cuda.cu:3793` (`cudaMemcpyAsync`+`cudaStreamSynchronize` at
`ggml-cuda.cu:3823-3824`, a second pair at `3862-3865`) — and the telemetry
census (`LUCE_MMID_TELEMETRY=1`, one cold p=4111 prefill) shows **zero**
`path=sync_fallback` dispatches. Every routed `mul_mat_id` call in this run
went through one of `mmb` (47 calls, the 4105-wide routed down-proj,
`ggml-cuda.cu:3773`), `mmq` (94 calls, all `ne11=1 width=6` — a narrow T=6
batch, not the main 4105-token GEMM), or `mmvq` (49 calls, also T=6). None of
these three paths synchronizes the host.

**Why the T=6 calls fail `ggml_cuda_mmb_supported_mmid`**: the gate at
`mmb.cu:910` is `if (T < mmb_min_t() || n_used > 64 || (T*n_used)>>16 >= 1024)
return false;` and `mmb_min_t()` returns **512** (`mmb.cu:790`). All 94 `mmq`
dispatches have `T=6` (`width=6` in the telemetry line), i.e. `6 < 512` —
that's the only failing term (`n_used=10 <= 64` and `(6*10)>>16=0 < 1024` both
pass). The q5_1/q8_0-typed T=6 calls instead land on `mmvq`
(`mmvq_max=8 >= 6`); the q4_K/q5_K-typed T=6 calls have a lower
`mmvq_mmid_max=4 < 6` so they fall through mmvq *and* mmb's 512-floor, landing
on `mmq` (`ggml-cuda.cu:3780-3784` → `mmq.cu:146-147`,
`mul_mat_q_case<GGML_TYPE_Q4_K>`). These T=6 calls are cheap (198.7 ms / 630
underlying kernel launches = 315 µs mean per the earlier trace) — they are
not the 19% "routed time" headline; that number is dominated by the
`mul_mat_q` launches' aggregate share of wall time, not by host stalls.

**Where the 764.6 ms / 16.78% idle actually comes from**: `ggml-cuda.cu`'s
CUDA-graph compatibility check (`ggml_cuda_graph_check_compability`,
`ggml-cuda.cu:4752-4762` `[TAG_MUL_MAT_ID_CUDA_GRAPHS]`) only tests
`mmid_mmvq_ok`, `mmid_mmq_ok`, `mmid_rocmfp3_ok` — it has **no `mmb_ok` or
`mix_wmma_ok` branch at all**, even though neither of those paths syncs the
host either (confirmed above). In this run's telemetry every
`event=graph node_eligible=1`, so this particular check is not disabling
graphs outright here — but warmup still never completes: a cold one-shot
p≈4111 benchmark forward's call is, by construction, the *first* call at that
exact graph shape (`ggml-cuda.cu:7776-7784`, "Warmup: need at least 2 calls
with no property change on the 2nd call"), and lucebox's chunk planner logs
show chunk row counts varying per request (`chunk-plan rows=2304/3328/.../
16640` in the server log) — a single pp-only bench exercises exactly one
shape once, so it always executes eagerly, never via `cudaGraphLaunch`
replay. Per-kernel host dispatch/bookkeeping overhead (pool allocs in
`ggml_cuda_pool_alloc` for `ids_src1`/`ids_dst`/`bounds`/`desc_big`/
`desc_small` on every `mmb`/`mmq` call, plus the full
`ggml_cuda_graph_update_required` property scan run every call since
`warmup_complete` never latches) sums to the 104 µs/launch average idle gap
(764.6 ms / 7319 launches) vs gufo's 30 µs/launch (209.5 ms / 6926) — gufo
does not appear to use HIP graphs either, so the ~3.5x gap is per-launch host
overhead, not graph-vs-no-graph. **Minimal-LOC fix**: none of the existing
fast paths need a sync, so the actual lever is reducing host-side
per-dispatch bookkeeping (skip the `ggml_cuda_graph_update_required` full
property re-scan once warmup has failed N times for a given one-shot shape,
and/or add `mmb_ok`/`mix_wmma_ok` to the graph-compatibility whitelist so a
*repeated* chunk shape, e.g. a real multi-turn server workload with
recurring prompt lengths, can actually warm up and replay) — ~5-15 lines,
estimated gain bounded by how much of the 764 ms is pool-alloc/property-scan
overhead vs irreducible per-kernel-launch latency; not fully quantified here
(would need per-call CPU-side timestamps, not just kernel timestamps, to
split the two).

### 2. Sort cost: lucebox sorts 2x per layer (glu-fused + down), not 3x

`mm_ids_helper` (`mmid.cu:33-134`) is called from **two independent sites per
layer** when the mmb path is taken: `ggml_cuda_mul_mat_id_mmb_glu`
(`mmb.cu:1112-1126`, the fused gate+up SwiGLU call — gate and up share **one**
sort here, not two) and `ggml_cuda_mul_mat_id_mmb` (`mmb.cu:1055-1069`, the
down-proj call). Both build their own independent `ids_src1`/`ids_dst`/
`bounds` from the *same* `ids` tensor (identical routing for gate, up, and
down within a layer — the router only runs once per layer) via two separate
calls to `ggml_cuda_launch_mm_ids_helper`/`ggml_cuda_launch_mm_ids_bounded`.
That is a real, avoidable 2x: the second call recomputes a sort whose input
and output are byte-identical to the first. 235 total `mm_ids_helper`/
`mm_ids_helper_fast` calls were observed across the run (2 sites x ~48
layers plus the `mmq`/`mix_wmma` fallback sites in `mmq.cu:415`,
`mix-wmma-moe.cu:436`, `mmf.cu:87` which also call it independently).

`mm_ids_helper`'s algorithm (`mmid.cu:33-130`) is a **serial-per-token, one
block per expert** compaction: 512 blocks (one per expert, `grid.x = E`),
each block loops `it` from 0 to `n_tokens` doing a per-warp scan + shared-mem
store + a second write-back loop — O(E x n_tokens) total work issued across
blocks even though each token only matches `top_k=10` of 512 experts, so
most of the 512 x 4105 iteration space is wasted "not my expert" checks.
Mean 1383 µs/call x 235 calls = 325 ms.

gufo's equivalent is `RoutedPadBoundsKernel` + `RoutedScatterKernel`
(`kernels.hip.cpp:5367-5420` and `mmq/mmid.hip.cpp`, local gufo clone at
`/home/peppi/.claude/jobs/559c1173/tmp/gufo`): `RoutedPadBoundsKernel` is a
single-block O(E) exclusive scan over per-expert counts (**no per-token
loop** — it consumes precomputed histogram counts), and `RoutedScatterKernel`
is a flat `O(slots)` grid-stride kernel doing one `atomicAdd` per
(token, slot) pair directly into the 16-padded bucket — no per-expert serial
loop, no double write-back pass. This is algorithmically cheaper (single
atomic scatter vs lucebox's warp-scan-then-rewrite) and is launched once,
consistent with gufo's 23.7 ms / 0.42% total.

**Minimal-LOC fix**: (a) cache `ids_src1`/`ids_dst`/`bounds` computed by the
glu-fused call and reuse them for the immediately-following down-proj call on
the same `ids` tensor within a layer — same mechanism (same `ids`, `E`, `T`,
`n_used`), a few lines threading a cache key/pointer through
`ggml_cuda_mul_mat_id_mmb`'s caller, saves ~half of 325 ms ≈ **162 ms**
directly. (b) Replatform `mm_ids_helper` on a histogram-then-atomic-scatter
scheme like gufo's instead of the per-expert serial token loop — bigger
change, out of scope for a "minimal-LOC" fix, but is the deeper reason
lucebox's sort is ~14x gufo's.

### 3. Prompt parity check: the two engines' synthetic prompts are not comparable

gufo's `bench -p` generates tokens via `MakeBenchmarkTokens`
(`src/cli/bench/bench.cpp:280-287`, local gufo clone):
```
tokens[i] = ((start_pos + i) % 1000) + 100
```
i.e. a deterministic **cyclic sequence of raw token IDs 100-1099**, repeating
every 1000 positions — not natural-language text, no tokenizer involved.
lucebox's `lucebox_pp_bench.py` instead builds a natural-language string from
a 40-word vocabulary (`WORDS = ("the quick brown fox jumps over lazy river
delta canyon ..." split())`) and runs it through the model's real 248k-vocab
BPE tokenizer.

These are **not** plausibly routing-equivalent. gufo's token-ID-cycling
scheme touches a narrow, low-ID band of the embedding table in a fixed
repeating pattern — hidden states (and therefore router logits) for
positions `i` and `i+1000` are identical inputs to the embedding lookup,
giving a highly periodic, likely much less diverse router activation
pattern than lucebox's natural-English-text embeddings, which draw from a
real word-frequency distribution and produce semantically varied hidden
states at every position. Since this is a **512-expert, top-10** MoE,
routing skew directly changes which experts get large vs. tiny token counts
each call — and that skew feeds straight into both engines' bucket-size-
dependent code paths (lucebox's 16/48-row-tile and mmb-vs-mmq-vs-sync_fallback
gates; gufo's 16/48/64/128 adaptive tile ladder). The dispatch-path split
measured here (mmb 47 / mmq 94 / mmvq 49 calls, almost all mmq/mmvq calls at
T=6) is itself a function of this specific prompt's routing distribution and
would very likely shift under gufo's cyclic-ID prompt or a different natural
prompt — the bucket-table percentages above should be read as this-prompt-
specific, not fundamental to either engine.

## Bounded sort gate experiment

One-line experiment: `ggml_cuda_launch_mm_ids_bounded`'s token-count gate in
`mmid.cu:374` rejected every prefill chunk (`tokens<=4096 || tokens>32768`),
forcing every 4096-row prefill chunk onto the slow one-warp-per-expert
`mm_ids_helper` (measured ~359 ms, 7.9% of wall at p=4092 in the prior kernel
trace). The bounded path's ordering claim — ascending `expert, then
token*used+slot` within each tile, replicated in increasing chunk order —
is identical to the helper's ascending-token-order compact layout whenever
`used == MM_IDS_ROUTE_EXPERTS` tile math (`MM_IDS_ROUTE_BITS=10` covers the
full 1024-route tile regardless of total token count `T`), so relaxing the
gate's lower bound should be a pure speed change, not a correctness change.

**Test first.** No existing test drove the bounded kernel (`rg
"mm_ids_bounded|mm_ids_route"` over `server/test` only found the classic-path
regression test, `test_ds4_mmid_regression.cpp`). Added
`server/test/unit/test_mmid_bounded_parity.cpp`: for experts=512, used=10
(the only shape the bounded path accepts), it runs `ggml_cuda_launch_mm_ids_helper`
and `ggml_cuda_launch_mm_ids_bounded` on the *same* randomized routing (with
masked/negative ids sprinkled in, same generator shape as the existing
regression test) and asserts `ids_src1`, `ids_dst`, and `expert_bounds` are
byte-identical, across `channels ∈ {1, 10}`, `inverse ∈ {false, true}`,
`tokens ∈ {512, 1000, 4096, 4097}`. Wired into `LUCE_TESTS` next to
`test_ds4_mmid_regression` in `server/CMakeLists.txt`.

**Gate change.** `mmid.cu:374`: `tokens<=4096` → `tokens<512` (cap at 32768
unchanged). `MM_IDS_ROUTE_BITS=10` encodes only the local index within one
1024-route tile, so it is independent of total `T` at every value in
[512, 4096] — confirmed by inspection, not just the test.

**Build provenance (lucebox4).** `/home/duster/qwen4exp-exact-stack/build1151`
was built from `/home/duster/qwen4exp-exact-stack/src/server`
(`mmid.cu` md5 `d32939834cb21a58e89eac9afb946865`, matches this worktree's
HEAD exactly — verified before touching anything). Copied that tree twice
into `~/qwen4exp-f16/scratch/sortgate/` (`src_a`, `src_b`), left `src_b` with
the gate change, reverted `src_a`'s `mmid.cu:374` back to `tokens<=4096`
(baseline). Configured and built both with `build1151`'s CMakeCache flags
(Release, HIP, `gfx1151`, `LUCE_TESTS=ON`, same `clang++`/`c++` compilers).

- A (gate unchanged, baseline): `luce_server` md5
  `41627a7b6da4ad24c38c0f52ee951c66`
- B (gate `tokens<512`): `luce_server` md5
  `5ac86fb6a22ff837bb5ddc76acd010d3`

**Test result on the iGPU** (`ROCR_VISIBLE_DEVICES=1`, binary B):
`test_mmid_bounded_parity` — **PASS**, all 16 cases (4 token counts × 2
channel counts × 2 inverse modes). On binary A (old gate, sanity check): the
12 cases with `tokens ∈ {512, 1000, 4096}` correctly report "bounded path
[not] accepted" (gate still closed) and the 4 cases at `tokens=4097`
(already past the old gate) pass — confirming the harness itself and that A
and B differ by exactly the one gate line.

**Timing A/B** (indicative: caches not dropped). GPU confirmed idle
(`rocm-smi` 0% on both agents, no other `luce_server`/`gufo` process) before
each load; one server loaded at a time, same flags as the gufo-parity run
above (`ROCR_VISIBLE_DEVICES=1 ./luce_server <shard1.gguf> --port <P>
--max-ctx 20480`), same `lucebox_pp_bench.py` client (distinct random prompt
per rep, `temperature=0`, cold each session), 3 reps per point:

| Binary | p | prefill tok/s (3 reps) | median |
| --- | ---: | --- | ---: |
| A (gate unchanged) | 4096  | 1008.6, 1102.3, 1126.6 | **1102.3** |
| B (gate `tokens<512`) | 4096  | 1068.1, 1137.5, 1193.9 | **1137.5** |
| A (gate unchanged) | 16384 | 1249.6, 1279.0, 1294.0 | **1279.0** |
| B (gate `tokens<512`) | 16384 | 1208.6, 1262.7, 1282.0 | **1262.7** |

B is +3.2% at p=4096 and −1.3% at p=16384 — both inside the ±4-8% rep-to-rep
spread seen on *each* binary alone, i.e. noise, not a measurable win. This is
expected: the prior kernel trace put the classic-helper sort at ~8% of wall
for a 4096-row chunk, and the bounded path only engages for that one 4096-row
chunk per prefill call (p=16384 still spends most of its chunks `>4096`,
already on the bounded path on both A and B) — the gate relaxation's
reachable savings are a fraction of that single chunk's sort time, within
this harness's noise floor. A finer, trace-level measurement (not a
wall-clock A/B) would be needed to isolate it; out of scope for this
one-line experiment.

**Output equivalence.** Fixed prompt ("Explain in one short paragraph why
the sky is blue."), `temperature=0`, `max_tokens=32`, non-streaming, on both
A (port 19080) and B (port 19081): generated text is **identical** —
`"\n\n\nThe sky appears blue due to a phenomenon called Rayleigh scattering.
As sunlight reaches Earth's atmosphere, it collides with gas molecules and
scatters in"` on both, same `content_tokens`/`total_tokens`. (This prompt is
only 64 tokens, below the gate on both binaries, so it exercises the shared
`mm_ids_helper` path rather than the changed one directly — the byte-level
ordering equivalence for the changed path is what the unit test above
proves; this is an end-to-end sanity check, not an isolated probe of the new
path.)

**Verdict:** correctness proven (test PASS, output identical); the gate
relaxation is safe to land, but on this harness it is not a measurable
prefill speedup — the 4096-row-chunk sort is too small a fraction of total
prefill wall time for a single-chunk win to clear the noise floor at p=4096
or p=16384.

## Text-prompt parity rerun (2026-10-13)

The parity numbers above are **void on prompt-content parity**: gufo's
`bench -p` tool feeds a deterministic cyclic token-ID sequence
(`((start_pos+i)%1000)+100`, `bench.cpp:280-287` — see "Prompt parity check"
above), while lucebox was fed tokenized natural-language text. For a
512-expert top-10 MoE, router logits (and therefore which per-shape/per-tile
code path each engine takes) depend on content, so the two engines were not
doing the same work. This section re-runs both engines through gufo's own
HTTP `serve` mode (OpenAI-compatible, not the `bench` CLI) so both consume
**byte-identical text** per rep.

### Parity contract (this rerun)

1. **Model**: same file, same shard path as the main parity run above
   (`/home/duster/lucebox-models/Qwen3.8-Flash-Next-UD-Q4_K_XL-00001-of-00004.gguf`).
   Binary md5s captured before touching anything:
   `luce_server` `7afa92f0da68d7b7a21f209a7c8e77b9`,
   `gufo` `c2a41962cc816a8eb6f1c827ed040898`,
   GGUF shard1 `1958631667f5320c7969d047b43b5829`.
2. **Device**: both pinned to the Strix iGPU (`gfx1151`) —
   `HIP_VISIBLE_DEVICES=1` for gufo, `ROCR_VISIBLE_DEVICES=1` for lucebox —
   same as the main run, hiding the discrete `gfx1201`.
3. **Context**: `--max-ctx 20480` (lucebox), `--context 20480` (gufo serve).
4. **Prompt content**: real text, not synthetic tokens. Source: concatenated
   CPython 3.12 stdlib `.py` files found on the box
   (`find /usr/lib/python3.12 -maxdepth 2 -iname '*.py' | sort | xargs cat`,
   9.77 MB corpus, `~/qwen4exp-f16/scratch/gufo/textparity/corpus.txt`).
   Six distinct, non-overlapping slices (3 reps × 2 lengths) were cut from
   this corpus at increasing offsets (so no rep repeats another rep's text)
   and calibrated against gufo's live tokenizer by proportional retry until
   `prompt_tokens` landed within 15 tokens of the 4096/16384 targets. The
   **same slice files** (byte-identical) were then posted to both engines —
   lucebox's own chat template adds a fixed ~40-43-token wrapper, so its
   `prompt_tokens` read ~40 tokens higher than gufo's for the same underlying
   text (4136 vs 4096, 16421 vs 16381, etc.) — this offset is the chat
   template, not a content mismatch, and is small relative to p.
5. **Sampling**: greedy (temperature 0), `max_tokens=1`, `stream=true`, same
   OpenAI-compatible client script against both engines
   (`text_bench_client.py`), reading `usage.prompt_tokens` and wall time as
   TTFT (first and only token).
6. **Caching symmetry — verified on both sides, not assumed:**
   - gufo: every request logged `cache=miss cached_tokens=0` in its verbose
     server log (`gufo_serve2.log`) — confirmed no prefix reuse across the 6
     distinct-content reps.
   - lucebox: **first attempt was NOT symmetric** — lucebox's prefix cache is
     ON by default (`prefix_cache = 32 slots`) and, even with 6 distinct
     prompt bodies, it cached the ~43-token chat-template preamble common to
     all of them (`restore=true slot=0 prefix_len=43` from rep 2 onward,
     `luce_serve.log`). That is a real, if small (43 of 4096-16430 tokens),
     asymmetry against gufo's zero-cache behavior, so that run was discarded.
     Server was restarted with `--prefix-cache-slots 0`; the rerun logged
     `restore=false slot=-1 prefix_len=0` on **every** request
     (`luce_serve_nocache.log`) — now symmetric with gufo's `cache=miss`.
     **All numbers below are from the `--prefix-cache-slots 0` run.**
7. One engine loaded at a time; GPU confirmed idle (0-3% busy on both
   `gfx1151`/`gfx1201` via `/sys/class/drm/card{1,2}/device/gpu_busy_percent`)
   before each load, no other process touched.

### Commands

- gufo server: `HIP_VISIBLE_DEVICES=1 LD_LIBRARY_PATH=/opt/rocm-7.2.2/lib
  /home/duster/gufo-parity/src/build/release/gufo serve --port 18090 llm
  -m <shard1.gguf> --context 20480 --served-model-name bench --think off
  --max-pending-per-client 8 -v`
- lucebox server: `ROCR_VISIBLE_DEVICES=1
  /home/duster/qwen4exp-exact-stack/build1151/luce_server <shard1.gguf>
  --port 19080 --max-ctx 20480 --prefix-cache-slots 0`
- slice generation: `python3 make_slices.py 18090 corpus.txt slices.json .`
  (binary-search-by-proportional-retry against gufo's live tokenizer)
- timed reps (same script, same files, against each engine's port):
  `python3 text_bench_client.py <port> prompt_p<4096|16384>_r<0|1|2>.txt`
- Scratch (box): `~/qwen4exp-f16/scratch/gufo/textparity/` — `corpus.txt`,
  `make_slices.py`, `text_bench_client.py`, `prompt_p{4096,16384}_r{0,1,2}.txt`,
  `slices.json`, `gufo_serve2.log`, `luce_serve_nocache.log`,
  `gufo_timed_results.jsonl`, `luce_timed_results_nocache.jsonl`.

### Raw per-rep arrays (prefill tok/s = prompt_tokens / client wall time)

| Engine | p target | rep0 | rep1 | rep2 | median | mean | best | min |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| gufo   | 4096  | 1252.9 | 1355.9 | 1333.4 | **1333.4** | 1314.1 | 1355.9 | 1252.9 |
| gufo   | 16384 | 1323.2 | 1323.6 | 1317.1 | **1323.2** | 1321.3 | 1323.6 | 1317.1 |
| lucebox| 4096  |  972.2 | 1151.6 | 1162.1 | **1151.6** | 1095.3 | 1162.1 |  972.2 |
| lucebox| 16384 | 1189.4 | 1216.5 | 1241.3 | **1216.5** | 1215.7 | 1241.3 | 1189.4 |

(prompt_tokens actually served, for the record: gufo 4096/4091/4095 and
16381/16379/16390; lucebox 4136/4131/4135 and 16421/16419/16430 — the
~40-token offset is lucebox's chat template, confirmed above.)

### New gap vs the old cyclic-token gap

| p | old gap (cyclic ids, gufo `bench` vs lucebox text) | new gap (real text, both engines) |
| ---: | ---: | ---: |
| 4096  | gufo 1741.7 vs lucebox 1135.6 → **1.53x (+53.4%)** | gufo 1333.4 vs lucebox 1151.6 → **1.16x (+15.8%)** |
| 16384 | gufo 1681.1 vs lucebox 1247.0 → **1.35x (+34.9%)** | gufo 1323.2 vs lucebox 1216.5 → **1.09x (+8.8%)** |

**Prompt content changed gufo's numbers, not lucebox's.** gufo's cyclic-token
`bench` numbers (1741.7 / 1681.1 tok/s) were **~24% and ~21% higher** than its
own real-text `serve` numbers (1333.4 / 1323.2 tok/s) on the same GGUF, same
device, same context — gufo's narrow, periodic, low-ID-band synthetic prompt
was measurably easier for its adaptive per-shape/per-expert-histogram tiling
than real natural-language-derived routing. lucebox's numbers, by contrast,
were already measured on real text in the original run and move only
slightly (1135.6→1151.6 at p=4096, 1247.0→1216.5 at p=16384 — both within
this harness's normal rep-to-rep noise band, ±5-8%, not a content effect).
So essentially all of the ~35-50% "gufo wins big" headline in the original
parity table was an artifact of gufo's own benchmark tool feeding itself an
easier synthetic workload — on matched real text the gap shrinks to
**+16% at p=4096 and +9% at p=16384**, still a real lucebox deficit but far
smaller than previously reported.

### Step 5 (lucebox fed gufo-style cyclic token IDs) — skipped

lucebox's HTTP API has no raw-token-ID prompt endpoint: `/v1/completions`
with an integer-array `prompt` returns `unknown endpoint` (only
`/v1/chat/completions` with string `content` is implemented), so there is no
way to feed lucebox the exact same cyclic-ID sequence gufo's `bench` tool
uses without going through its BPE tokenizer (which would re-encode the
integers as text, not reproduce raw IDs). Per the task's fallback
instruction, this half of the experiment was skipped rather than faked.

**Caveat:** indicative, not fully converged — 3 reps/point, caches not
dropped between reps (though confirmed non-reused per-request above), and
page/host memory state was whatever each engine's own cold-start left it in.
The qualitative result (content changes gufo's number by ~20%+, barely moves
lucebox's) is large relative to the ±5-8% rep-to-rep noise and is the
headline finding; the exact +16%/+9% gap figures should be read as
directionally solid but not noise-free to the last digit.

## Matched-text kernel trace (p=4096)

Box: lucebox4, Strix iGPU (`gfx1151`) pinned, discrete `gfx1201` hidden, GPU
confirmed idle (0-3%) before each load. Same binaries/GGUF md5s as the
text-parity rerun above. Scratch: `~/qwen4exp-f16/scratch/gufo/trace2/`
(`gufo_out_csv/`, `lucebox_out_csv/`, raw kernel-trace CSVs, not committed).

**Method.** Both engines launched in HTTP `serve` mode under
`rocprofv3 --kernel-trace --stats -f csv` (same tool as before). Each server
was warmed with one full text request (`prompt_p4096_r0.txt`, discarded),
then exactly one measured request on **distinct** text (`prompt_p4096_r1.txt`,
4091/4131 prompt tokens after chat-template offset) was sent through the same
`text_bench_client.py` used in the text-parity rerun. The server was then
SIGTERM'd to flush the rocprofv3 CSV before the next engine was started — one
engine at a time, nothing else touched. The measured request's kernel window
was isolated from the full process trace by finding the largest multi-second
gap between consecutive kernel end/start timestamps (clean separator between
warmup-request kernels and measured-request kernels, confirmed distinct from
the ~18-105s model-load gap at the front of each trace) and validated by
per-layer call counts: lucebox's measured window has exactly 36
`gated_delta_net_tiled_cuda` and 12 `qsa3_attn_kernel` calls — the expected
36-GDN/12-attention-layer count for one clean forward pass, no chunking
multiplier. gufo's measured window has 108 `GdnRowSplitKernel` / 36
`WmmaCausalAttentionKernel` calls — exactly 3x the 36/12 layer count, i.e.
gufo's serve path internally **chunks** a 4096-token prefill into 3 chunks
per request (confirmed identical 108/36 in both the warmup and measured
windows), not multiple duplicate forward passes of the kind that made the
earlier bench-mode trace unreliable (that one had a 4x multiplier from
bundled reps, not per-request chunking).

### Per-class kernel-duration sums (measured request only, ms)

| Class | gufo ms | lucebox ms | Δ (luce − gufo) |
| --- | ---: | ---: | ---: |
| Dense GEMM | 1024.21 | 1166.74 | **+142.53** |
| Routed gate/up | 431.47 | 609.18 | **+177.71** |
| Routed down | 590.23 | 340.41 | **−249.82** |
| Expert sort/scatter/gather | 8.54 | 31.65 | +23.11 |
| GDN | 253.16 | 393.33 | **+140.17** |
| Attention | 194.24 | 217.03 | +22.79 |
| hc_* | 338.92 | 456.44 | +117.52 |
| Norms/elementwise/quantize | 28.71 | 126.49 | +97.78 |
| Other | 17.41 | 1.83 | −15.58 |
| **Σ kernels** | **2886.88** | **3343.10** | **+456.22** |
| Idle (wall − Σkernels) | 108.32 (3.6%) | 309.35 (8.5%) | **+201.03** |
| **Wall (profiled, this request)** | **2995.21** | **3652.44** | **+657.23** |

### Kernel-vs-overhead verdict

The gap is **kernel time, not launch/idle overhead**. Σkernels alone account
for +456 ms of the +657 ms profiled-wall delta (69%); idle only adds +201 ms
(31%), and even lucebox's idle share (8.5%) is modest in absolute terms
compared to the kernel-time deltas. Against the unprofiled textparity wall
numbers (lucebox 1151.6 tok/s ≈ 3.56 s, gufo 1333.4 tok/s ≈ 3.07 s, gap
≈ 490 ms), the profiled delta (657 ms) is the same order of magnitude and
dominated by the same classes — rocprofv3 instrumentation inflates both
engines' wall time but does not manufacture the gap; it was already present
unprofiled, and the per-class composition here is the real explanation.

Top 3 delta classes (by |Δ|, excluding the small "other" residual):

1. **Routed down, −249.82 ms (lucebox wins this one).**
   - gufo: `RoutedF16GEMMKernel<WeightType=12, 128, 64, 2, true>` — 94 calls,
     5922.11 µs mean (556.68 ms), the dominant down-proj tile.
   - lucebox: `mmb_routed_kernel<128, 128, 32, 64, WT, false, true>` — 42
     calls, 3858.68 µs mean (162.06 ms) plus `mmb_routed_kernel<128, 64, 32,
     32, WT, false, true>` — 42 calls, 3322.90 µs mean (139.56 ms). Dispatch:
     `server/deps/llama.cpp/ggml/src/ggml-cuda/mmb.cu:1090-1094`
     (`mmb_routed_kernel<128, BN, 32, 64, WT, false, true>` /
     `mmb_routed_kernel<128, BN_SMALL, 32, 32, WT, false, true>`).

2. **Routed gate/up, +177.71 ms (gufo wins this one).**
   - gufo: `RoutedF16GEMMKernel<WeightType=7, 128, 48, 2, false>` — 86 calls,
     4325.78 µs mean (372.02 ms), the dominant gate/up tile.
   - lucebox: `mmb_routed_glu_kernel<64, 128, 32, 32, 44, false>` — 46 calls,
     6571.25 µs mean (302.28 ms) plus `mmb_routed_glu_kernel<64, 64, 16, 32,
     44, false>` — 46 calls, 5752.17 µs mean (264.60 ms). Dispatch:
     `server/deps/llama.cpp/ggml/src/ggml-cuda/mmb.cu:1146-1147`
     (`mmb_routed_glu_kernel<64, BN, 32, 32, WT>` /
     `mmb_routed_glu_kernel<64, BN_SMALL, 16, 32, WT>`). Per-call mean is
     ~1.4-1.7x gufo's despite fewer total calls — lucebox's fused glu kernel
     is doing meaningfully more work per launch than gufo's gate/up tile for
     this routing distribution.

3. **Dense GEMM, +142.53 ms (gufo wins this one; GDN is a near-tie at
   +140.17 ms, same order).**
   - gufo: `DenseF16GEMMKernel<256,128,2,8,1,1,false,true,false,false>` — 72
     calls, 4977.64 µs mean (358.39 ms), the largest single dense-GEMM
     instantiation.
   - lucebox: `mmb_q8f16_kernel<256, 128, 2, 8, 1, 1>` — 84 calls, 5920.25 µs
     mean (497.30 ms) plus `mmb_q8f16_kernel<256, 128, 2, 8, 1, 0>` — 97
     calls, 2713.59 µs mean (263.22 ms). Dispatch:
     `server/deps/llama.cpp/ggml/src/ggml-cuda/mmb-q8f16.cuh:254-256`
     (`mmb_q8f16_kernel<256, 128, 2, 8, 1, xt>` for `xt` in {0,1,2}).

### Top lever

**Routed gate/up is the single largest class where lucebox is slower in
absolute terms and the mechanism is visible in the trace**: lucebox's fused
`mmb_routed_glu_kernel` runs fewer total launches than gufo's gate/up tile
but at 1.4-1.7x the per-call mean latency for this routing distribution
(`mmb.cu:1146-1147`). Routed down is the mirror case where lucebox already
beats gufo (−250 ms) with the same kernel family (`mmb.cu:1090-1094`), so the
routed-expert story is not "lucebox's MMB path is slow" in general — it is
specifically the glu/gate-up tile shape/width choice that is the lever to
retune first, since the down-proj tile choice for the same kernel family is
already competitive or ahead. Dense GEMM and GDN deltas are smaller and
closely tied with each other; they are secondary to the gate/up tile.

## Prefill host-overhead breakdown

Box: lucebox4, Strix iGPU (`gfx1151`) pinned, discrete `gfx1201` hidden, GPU
confirmed idle (0-3%) before every load. **Measurement only** — no engine
source changes landed anywhere in this repo/worktree; all instrumentation
lives in a scratch copy and scratch build, per the task's hard rule.

**Scratch tree**: `~/qwen4exp-f16/scratch/idle2/` on lucebox4 —
`src_timers/` (a full copy of `~/qwen4exp-f16/scratch/sortgate/src_b`, i.e.
this worktree's HEAD-equivalent `server/` tree, `mmid.cu` md5
`61e9e8bd606bbd1ed9fe4e037dce1274`, matching `src_b`'s gate-relaxed `mmid.cu`
from the prior sortgate experiment — unrelated to this section's findings,
which are all in the chunk-graph/dispatch/sync path, not `mmid.cu`),
`build_timers/` (fresh `cmake -S src_timers/server -B build_timers
-DCMAKE_BUILD_TYPE=Release -DLUCE_GPU_BACKEND=hip
-DCMAKE_HIP_ARCHITECTURES=gfx1151 -DLUCE_TESTS=OFF`, built with
`cmake --build . --target luce_server -j32`), `patch_timers*.py` (the four
anchor-based source patches applied, kept for provenance), `server_timers2.log`
(raw timer output, not committed), reusing the existing
`~/qwen4exp-f16/scratch/gufo/textparity/` prompt slices and
`text_bench_client.py` for the driving requests (same distinct-text,
`--prefix-cache-slots 0`, `temperature=0`, `max_tokens=1` harness as the
text-parity rerun above).

**Timers added** (env-gated on `LUCE_PREFILL_TIMERS=1`, `std::chrono::steady_clock`,
printed once per prefill forward call, `n_tokens > 1` only):

1. `server/src/qwen4exp/qwen4exp_graph.cpp`, `forward_impl` — wall-clock split
   of: ctx-init → graph-built (ggml op construction), gallocr reserve+alloc,
   host-side input upload (positions/mask/embeddings), the
   `ggml_backend_graph_compute(backend, gf)` call itself, and the post-compute
   logits/argmax readback (`ggml_backend_tensor_get`).
2. `server/deps/llama.cpp/ggml/src/ggml-cuda/ggml-cuda.cu`,
   `ggml_cuda_graph_evaluate_and_capture` — wall-clock around the entire eager
   per-node dispatch loop (every `GGML_OP_*` dispatch call for every node in
   the graph, taken because `use_cuda_graph` is false for every one-shot
   prefill shape here, confirmed by the existing warmup-never-latches finding
   above).
3. `server/deps/llama.cpp/ggml/src/ggml-backend.cpp`, the generic
   `ggml_backend_graph_compute()` wrapper — split into
   `ggml_backend_graph_compute_async()` (issuing all the eager launches) vs.
   the subsequent `ggml_backend_synchronize()` call (blocking until the GPU
   drains the queue). This is the authoritative async-submit-vs-sync-wait
   split; the other two timers are diagnostic context around it.

### Results (unprofiled — no rocprofv3, so these are directly comparable to
the parity/text-parity wall-clock numbers above, unlike the earlier
rocprofv3 kernel-trace section)

One cold server load, one discarded warmup request (`prompt_p4096_r0.txt`),
then two more distinct-text requests at p=4096 (`r1`, `r2`) and, in a second
cold load, the same warmup/measured pattern at p=16384:

| p | rep | shape | n_nodes | async_submit_us | sync_wait_us | submit % of (submit+sync) | client prefill tok/s |
| ---: | --- | --- | ---: | ---: | ---: | ---: | ---: |
| 4096  | r0 (cold, first-ever call at this shape) | T=4136 | 5418 | 364,722 | 3,151,757 | 10.4% | 948.9 |
| 4096  | r1 (second distinct shape, same process) | T=4131 | 5502 |   9,158 | 3,232,083 |  0.3% | 1135.2 |
| 4096  | r2 | T=4135 | 5502 |   8,731 | 3,213,073 |  0.3% | 1135.9 |
| 16384 | r0 (cold, fresh server) | T=16384 | 5418 | 180,444 | 11,467,379 |  1.5% | 1211.8 |
| 16384 | r1 | T=16384 | 5418 |   9,802 | 11,676,890 |  0.08% | 1232.0 |

(n_nodes is the ggml graph's node count for the one prefill graph that covers
the whole request — confirmed single-shot, not chunked, consistent with the
matched-text kernel trace's "no chunking multiplier" finding above; the
chunk planner's configured chunk size for this `--max-ctx 20480` run is large
enough to cover both 4096 and 16384 in one graph.)

Host dispatch-loop cost per node (`luce-dispatch-timer`, the per-node eager
launch loop alone, a subset of `async_submit_us` above): **66.6 µs/node cold,
1.0-1.2 µs/node warm** at p=4096; **32.7 µs/node cold, 1.2-1.5 µs/node warm**
at p=16384 — a **~30-65x cold-vs-warm ratio**, but even the cold absolute
number (360.8 ms / 177.3 ms) is a small fraction of total wall time.

### Verdict: GPU-bound, not launch-bound

The CPU does **not** finish early and wait — `ggml_backend_synchronize()`
(the blocking call) is where **89.6-99.9%** of `ggml_backend_graph_compute`'s
wall time goes, at every p and every warm/cold state measured. The host-side
eager-dispatch loop — the mechanism the "7300 eager launches, ~42 µs/launch
of gap" framing in this task's brief was built on — accounts for at most
10.4% of wall time (the one-time cold-shape penalty) and **under 0.3-1.5%
once warm**, which is the steady-state a real multi-request server actually
runs in. This directly contradicts reading the earlier rocprofv3 kernel
trace's "764.6 ms / 16.78% idle, 104 µs/launch" figure as host-dispatch
overhead: rocprofv3 intercepts and serializes every kernel launch (already
flagged above as inflating both engines' wall time non-uniformly), and that
inflation is exactly where the earlier idle number came from — it is a
profiling artifact, not a real host stall in unprofiled operation. Unprofiled,
the host races ahead and submits the entire ~5400-5750-node graph in single-
digit milliseconds once warm; the remaining 3.2-3.3 s (p=4096) / 11.5-11.7 s
(p=16384) is the GPU actually executing the kernels Σkernels already measured
in the matched-text trace (lucebox 3343.10 ms at p=4096, same order as the
3.15-3.23 s `sync_wait_us` measured here).

### Top 3 host costs in µs/node

Since host dispatch is 1-2 µs/node warm, there is no large per-op host cost to
chase in steady state — the only measurable host cost worth naming is the
**one-time cold-shape tax** (66.6 µs/node at p=4096, 32.7 µs/node at
p=16384, vanishing to ~1.2 µs/node on the next distinct-shape call in the same
process). This is consistent with first-touch allocator-pool growth
(`ggml_cuda_pool_alloc` doing a real `hipMalloc` instead of a pool hit),
graph-key creation, and the one-time
`ggml_cuda_graph_check_compability`/property-scan cost that only runs fully
on a never-before-seen graph key — all amortized to near-zero by the second
call. No individual dispatch function (`ggml_cuda_mmb_supported_mmid`,
`ggml_cuda_pool_alloc`, the property-scan) was profiled in isolation because
the aggregate per-node host cost is already below the threshold where
splitting it further would change the GPU-bound verdict; `perf record` was
not run for the same reason (the chrono split already answers launch-bound
vs. GPU-bound unambiguously).

### A/B: forcing graph capture on the first call

No existing env/flag does this. Grepped every `getenv("LUCE_*")` and
`getenv("GGML_CUDA_*")` in the scratch tree (`ggml-cuda.cu` and every
`server/src/**/*.cpp`): `ggml_cuda_skip_props_check` (set via
`ScopedCudaGraphOverrides`'s `skip_property_check_`) only skips the
properties-changed check on a **subsequent** call to an **already-warm**
graph key — it does not shortcut the warmup state machine's hard "2 identical
calls" requirement at `ggml-cuda.cu:7776-7784`, and no `LUCE_*`/`GGML_CUDA_*`
variable bypasses that gate. **A/B skipped, with justification**: even in the
worst case this task could reach (the cold first-call penalty above),
forcing single-call capture would reclaim at most the `async_submit_us`
column — 364.7 ms of 3.52 s (10.4%) at p=4096, 180.4 ms of 11.65 s (1.5%) at
p=16384 — and only for literally the first request at a given exact chunk
shape per process lifetime; every subsequent request already pays ≤1.5 µs/node.
Given the verdict above (GPU-bound, Σkernels dominates), forcing graph
capture is not the lever regardless of whether a flag existed — it would not
touch the 89.6-99.9% of wall time spent waiting on the GPU.

### Smallest fix and expected gain

There is no cheap host-side fix with a meaningful payoff here, because the
host is not the bottleneck. The only host-side saving on the table is the
one-time cold-shape allocator/graph-key warmup tax (≤365 ms, once per
distinct chunk shape per process lifetime) — not worth engineering around for
a steady-state server. The actual lever remains what the matched-text kernel
trace above already identified: **kernel time**, specifically the routed
gate/up tile width/shape (`mmb.cu:1146-1147`, +177.71 ms vs. gufo at p=4096)
and the broader dense/GDN deltas — i.e., retuning GPU-side kernel tile
choices, not reducing host dispatch overhead, which this measurement shows is
already negligible.

## MMB 128-row tile threshold (Codex idea #1)

**Idea.** `mmb.cu`'s two routed MMB dispatchers (`ggml_cuda_mul_mat_id_mmb`,
`ggml_cuda_mul_mat_id_mmb_glu`) assign a single 128-row ("big") tile to
experts with `>= THRESH` routed rows, and one-or-more 64-row ("small") tiles
below that. `THRESH` was a duplicated `constexpr int ... = 128` in both
dispatchers. Codex's idea: lower it to 65, so a 65-127-row expert gets one
128-row tile instead of two 64-row tiles (each re-streaming the expert's
quantized weights). With T=4096 tokens, 10/512 experts used, the average is
~80 rows/expert — squarely in the affected band.

**Change.** Replaced the duplicated `constexpr THRESH = 128` in both
dispatchers with one named accessor, `mmb_tile128_min()` (`mmb.cu`, next to
`mmb_build_desc2`), defaulting to 65 and overridable via
`LUCE_MMB_TILE128_MIN` (read fresh per call, not cached, so a parity test or
operator can flip it without a restart). `BN_SMALL=64` stays a `constexpr`;
only the threshold moved.

**Correctness test.** New `server/test/unit/test_mmb_tile128_threshold.cpp`
(registered in `server/CMakeLists.txt` next to `test_mmb_cublas_diff`): 512
experts, 10 used, T=4096, routing built so **all 512 experts land in the
65-127 row band** (deterministic round-robin routing, exactly 80 rows/expert
— the worst case for exercising the change). Runs both routed dispatchers —
a plain Q5_1 "down" `MUL_MAT_ID`, and a fused Q4_K gate/up `SwiGLU` —
through the public `ggml_mul_mat_id`/`ggml_swiglu_split` graph API on the
real CUDA/HIP backend, once with `LUCE_MMB_TILE128_MIN=128` (old) and once
with `=65` (new), plus a CPU-backend reference for a loose sanity bound.
Result on lucebox4 (gfx1151, `ROCR_VISIBLE_DEVICES=1`):

```
[tile128] routing: 512/512 experts land in the 65-127 row band (min=80 max=80)
[tile128] down (Q5_1): bit-identical(128 vs 65)=YES maxdiff=0
[tile128] glu  (Q4_K): bit-identical(128 vs 65)=YES maxdiff=0
[tile128] PASS
```

Both dispatchers are **bit-identical** between THRESH=128 and THRESH=65, as
expected: the tile width changes how many tokens are processed per launch
per expert, not the K-accumulation order per output element.

**Prefill A/B.** lucebox4, gfx1151 iGPU, same `luce_server` binary (only
`LUCE_MMB_TILE128_MIN` differs), Qwen3.8-Flash-Next UD-Q4_K_XL,
`--max-ctx 20480 --prefix-cache-slots 0`, textparity harness
(`~/qwen4exp-f16/scratch/gufo/textparity/`), byte-identical CPython-text
slices, temp 0, 3 reps/arm. **The two arms could not run concurrently**: the
76 GB model does not fit twice in the iGPU's 98 GB UMA pool, so A and B are
two sequential server processes on the same port, not truly alternating
request-by-request. A third run of arm A (A2, bracketing B) was added to
estimate run-to-run noise for the same binary/threshold.

| length | A (128) raw tok/s | A2 (128, re-run) raw tok/s | B (65) raw tok/s | A median | B median | Δ% (A→B) |
|---|---|---|---|---|---|---|
| p≈4096  | 998.8, 1169.7, 1214.6 | 945.6, 1138.9, 1138.5 | 899.4, 1045.1, 1131.5 | 1169.7 | 1045.1 | **-10.7%** |
| p≈16384 | 1225.6, 1259.1, 1288.9 | — | 1176.0, 1230.8, 1253.1 | 1259.1 | 1230.8 | **-2.2%** |

A-vs-A2 run-to-run noise at p4096 was ~2.7% (median 1169.7 vs 1138.5); B is
8-11% below *both* A runs, i.e. outside that noise band. **Verdict: reject.**
This is not a wash, it is a small but real **regression** at p4096 and a
smaller one at p16384, the opposite of the idea's premise. A plausible
reason: with an exactly-80-rows-per-expert routing (and more generally with
real traffic, most experts well above 65), moving an expert that used to
split into 64+16 (one full 64-row tile, one 16-row tile) into a single
128-row tile means the 128-row GEMM now does ~1.6x the useful-row work for
the SAME wasted-row ratio as before per tile, but fewer, larger launches
reduce how well the GPU hides the per-launch fixed/setup cost against a
batch this small — i.e. the premise ("one bigger tile reads weights once
instead of twice") is true, but is outweighed here by larger, more serialized
per-tile work replacing more numerous, more parallelizable small tiles.
Per Codex's own rejection bar (<1% gain), this fails outright; it regresses.
**Determinism**: fixed-prompt, temp 0, max_tokens 256/32 generations were
byte-identical text between the 128 and 65 arms at every rep (see decode
table below) — the threshold change does not alter generated tokens, only
speed, and here it makes it slightly slower.

**Decode (same server runs, as requested for before/after reporting).**
512-token prompt, temp 0, max_tokens 256, decode tok/s = (completion_tokens-1)
/ (wall - time-to-first-token), 3 reps/arm:

| arm | raw decode tok/s | median | ms/token |
|---|---|---|---|
| A (128) | 23.79, 23.68, 23.90 | 23.79 | 42.04 |
| B (65)  | 23.75, 23.60, 23.83 | 23.75 | 42.11 |

Decode is flat (-0.17%, inside noise), as expected: decode batches are
T=1 per sequence, far below `mmb_min_t()=512`, so routed MMB (and this
threshold) never engages during decode. Generated text was byte-identical
between arms at every rep, confirming the change is prefill-path-only and
produces no token-level drift.

### Summary table (before=128, after=65)

| metric | before (128) | after (65) | Δ% |
|---|---|---|---|
| prefill p≈4096 (tok/s, median) | 1169.7 | 1045.1 | -10.7% |
| prefill p≈16384 (tok/s, median) | 1259.1 | 1230.8 | -2.2% |
| decode (tok/s, median) | 23.79 | 23.75 | -0.2% |

## Cold-shape tax on varied prompt lengths

Baseline arm (`LUCE_MMB_TILE128_MIN=128`), same server, 6 distinct-text
prompts at p≈{3000, 3500, 4000, 4500, 5200, 6100} tokens, sent once each
("pass 1"), then the identical 6 prompts sent again ("pass 2", same shapes,
now definitely warm):

| length | pass 1 tok/s | pass 2 tok/s | Δ% |
|---|---|---|---|
| 2997  | 1200.8 | 1178.3 | -1.9% |
| 3502  | 1243.2 | 1189.3 | -4.3% |
| 4011  | 1263.4 | 1270.9 | +0.6% |
| 4502  | 1277.7 | 1278.4 | +0.1% |
| 5195  | 1307.6 | 1301.3 | -0.5% |
| 6103  | 1328.3 | 1321.6 | -0.5% |

No cold-shape cliff is visible here (all deltas are within ~1-2 run-to-run
noise, except 3502 at -4.3% which is still far short of the ~66 µs/node /
~365 ms-at-p4096 cliff reported previously). **Caveat, and why this result is
inconclusive**: the prompt slices for pass 1 were generated by a calibration
script that queries the live server 1-6 times per target length while
narrowing in on the exact token count (see `make_varied_slices.py` attempts
in the transcript) — each calibration attempt is itself a prefill request at
a length very close to the final one. So by the time "pass 1" ran, the
server's graph-shape cache had very likely already seen each of these 6
shapes (or one within a few tokens of it) 1-5 times during calibration. This
measurement therefore can't distinguish a genuinely cold first-touch from an
already-warm shape; it is NOT the test the task asked for. A clean version
needs a freshly-launched server (zero prior requests) hitting each of the 6
lengths exactly once, in increasing or randomized order, with no calibration
traffic beforehand — not done here due to time/VRAM-reload budget (each
`luce_server` restart costs ~70-90s to reload the 76 GB model). Flagging this
as an open follow-up rather than reporting a false negative on the cold-tax
question.

## Fable #3: Routed GLU output-row tile 64 -> 128

Rejected. `ggml_cuda_mul_mat_id_mmb_glu`'s fused gate+up SwiGLU dispatch used a
64-row output tile (`mmb_routed_glu_kernel<64, ...>`) where the down-proj
dispatch (`ggml_cuda_mul_mat_id_mmb`) already used 128. Made the GLU tile
runtime-selectable (`mmb_glu_bm()`, env `LUCE_MMB_GLU_BM`, default 128,
override 64 for the A/B unit test only) and widened `WTM` proportionally
(32->64 for the big tile, 16->32 for the small tile) to keep
`WAVES_M * WAVES_N == MMB_NT/32 == 8`, the kernel's hard-coded occupancy
invariant.

**Risk found during review** (per the task's ask to check LDS/register budget
and why 64 was chosen): `mmb_tile_gemm_glu`'s Q4_K fast load path
(`Q4K_SPLIT`, stripe-packed register loads, two threads/row) is gated on
`4 * BM == MMB_NT` (256), which only holds at `BM=64`. At `BM=128` the kernel
silently falls back to the generic per-row dequant path for Q4_K -- same
arithmetic (confirmed bit-exact below) but a different, presumably slower,
load strategy for exactly the weight type (Q4_K gate/up) this change targets.
This is very likely why 64 was chosen originally and why the change
regresses below.

**Test.** `server/test/unit/test_mmb_glu_bm128.cpp` (new): E=512, n_used=10,
T=4096, Q4_K gate/up weights, routing skewed (8 distinct bases x 512 tokens
each) so 78 experts land >=128 rows (exercises the big-tile path) and 434
stay <128 rows (small-tile path). Runs the GLU GPU kernel with
`LUCE_MMB_GLU_BM=64` then `=128` in the same process and asserts
`memcmp`-exact output, plus a CPU-reference sanity bound (same caveat as
`test_mmb_cublas_diff`: CPU vec_dot quantizes activations, not an oracle).
**Result: PASS** -- `bit-identical(64 vs 128)=YES maxdiff=0`.

**Build (lucebox4).** `~/qwen4exp-f16/scratch/fable/3/{src,build}` (BM=128,
`git archive HEAD` + this change's diff) and `~/qwen4exp-f16/scratch/fable/3/
{base_src,base_build}` (unpatched HEAD, baseline A). Same cmake flags as
`~/qwen4exp-f16/scratch/tile128/build`. Binary md5s:
- A (baseline, HEAD): `luce_server` `5644a7469bb953ebee9e3bc19179fce6`
- B (BM=128 default): `luce_server` `aea3b3849f94bb8f12c1a9085fbeac82`

**A/B** (text-parity harness, `~/qwen4exp-f16/scratch/gufo/textparity/`,
`--prefix-cache-slots 0`, `--max-ctx 20480`, `ROCR_VISIBLE_DEVICES=1`, GPU
confirmed idle before each load, one server at a time, order A, B, A2):

| Point | A (median) | B (median) | A2 (median) | B vs A | B vs A2 | A/A2 noise |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| prefill p4096  | 1211.0 | 1143.2 | 1177.2 | -5.6% | -2.9% | 2.8% |
| prefill p16384 | 1230.9 | 1177.6 | 1239.2 | -4.3% | -5.0% | 0.7% |
| decode (512p/256t) | 23.93 | 23.74 | 23.84 | -0.8% | -0.4% | 0.4% |

Raw reps, tok/s: A p4096 [1163.5, 1218.6, 1211.0], B p4096 [1186.0, 1143.2,
1140.6], A2 p4096 [1182.3, 1125.5, 1177.2]; A p16384 [1215.2, 1230.9, 1245.7],
B p16384 [1148.1, 1198.4, 1177.6], A2 p16384 [1219.0, 1239.2, 1252.1]; A
decode [23.93, 23.58, 23.99], B decode [22.59, 23.74, 23.95], A2 decode
[23.84, 23.71, 23.96]. Generated text sampled byte-identical across A/B/A2 at
the one decode prompt checked.

**Verdict: REJECTED.** p16384's regression (-4.3%/-5.0%) is ~6-7x the
A/A2 noise floor at that length (0.7%); p4096's regression (-5.6%/-2.9%) is
also net negative against both baselines despite a noisier floor there
(2.8%). Decode is flat (within noise). Reverted locally
(`git checkout -- server/CMakeLists.txt server/deps/llama.cpp/ggml/src/
ggml-cuda/mmb.cu`, removed the new test file) -- HEAD is unchanged. Consistent
with the Q4K_SPLIT-loss hypothesis above: the fused-kernel fast path for Q4_K
only exists at BM=64, so doubling the output tile for this specific
weight type trades away more than the halved weight-restreaming saves.
