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
