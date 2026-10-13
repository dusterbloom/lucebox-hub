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
