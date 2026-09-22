# Plan: an efficient djev-spark equivalent for Strix Halo (gfx1151)

## Status (updated as work lands)

CPU-only progress so far — no GPU has been available in this environment,
so everything below is built and verified without one; nothing has run
against a real model yet. Branch `plan/djev-halo` (merges
`feat/diffusion-foundation`, the rebased DiffusionGemma backend, with
`experiment/openjev`, the causal `/v1/systemone` work — both were rebased
onto the same `upstream/main` tip independently, so the merge was clean).

- ✅ **Phase 2, core primitive — done, CPU-verified.** `run_diffusion_structured_read()`
  added to `diffusion_decoder.cpp`/`.h`: seeds `slot_count` noise positions
  after a causally-encoded prefix, runs `n_steps` bidirectional
  `forward_block` passes (refining toward the previous step's argmax
  between steps), returns the final step's raw per-slot logits — no
  commit/streaming/EOS handling, since a read wants a distribution, not
  tokens. 9 new unit tests against the module's existing
  `SyntheticModel`/spy-model pattern (single-slot, multi-slot, multi-step,
  Masked-vs-UniformState noise wiring, config guards). **57/57 tests pass**
  (48 pre-existing + 9 new), compiled and run with plain
  `g++ -std=c++17` — the module's own documented CPU-only path, no CUDA
  needed for this half.
- ✅ **Phase 2, wiring — done, syntax-verified only.** `DiffusionBackend::generate_impl`
  branches on `req.want_first_token_logits` before the normal
  `run_diffusion_generate()` call, routes to the structured-read primitive
  with `slot_count=1` and `n_steps=max(1, req.n_gen)` (`/v1/systemone`
  always sends `n_gen=1`, matching djev-spark's own one-step-suffices
  claim), and fills `GenerateResult::first_token_logits` — the same field
  the six causal backends already populate. **No `/v1/systemone` HTTP-layer
  changes were needed** beyond adding `"diffusion-gemma"` to its backend
  allowlist: the endpoint's rendering, tokenization, and
  restricted-softmax logic were already backend-agnostic. Verified with
  `g++ -fsyntax-only` against the real project headers (clean) on both
  `diffusion_backend.cpp` and `http_server.cpp` — **not** compiled as part
  of a full CUDA/HIP build, and never run.
- ❌ **Not built yet:** position-pinning for a *mixed* pinned+noise block
  (Phase 2's other bullet below) — deliberately out of scope for now since
  `/v1/systemone`'s one-slot-per-question protocol doesn't need it; still a
  documented follow-up if multi-position templates come up later.
- ⛔ **Blocked on hardware, not yet attempted:** Phase 0 (correctness
  validation against real weights), Phase 1 (running on real gfx1151),
  all of Phase 3 (every efficiency lever), Phase 4 (benchmarking). The
  structured-read primitive above has never executed against an actual
  DiffusionGemma forward pass — it's only been exercised against synthetic
  test models. Nothing in this status section changes that risk, flagged
  again below.

---

Target: reproduce [`mmastrac/djev-spark`](https://github.com/mmastrac/djev-spark)'s
capability — structured Jev-style decisions (noul/choice/score) served directly
off DiffusionGemma's native denoise-canvas mechanism — on AMD Strix Halo
(Ryzen AI Max+ 395, `gfx1151`, 128 GiB unified LPDDR5X-8000), using lucebox's
own native C++/HIP server instead of djev-spark's vLLM + FlashInfer + patched
structured-reads stack.

## Why this is a lucebox-native fit, not a from-scratch port

Two pieces already exist and are proven on exactly this hardware:

1. **Strix Halo is a first-class, benchmarked lucebox target already.**
   DeepSeek V4 Flash runs on it today (`server/README.md`): 37.0 tok/s DFlash
   decode on Qwen3.5-27B Q4_K_M, 3.08× decode / 2.24× prefill over llama.cpp
   HIP AR, with dedicated `gfx1151` tuning (`--ddtree-budget=22`,
   `-DLUCE_HIP_ARCHITECTURES=gfx1151`) and dedicated 4-bit MMQ kernels
   (`rocmfp4`/`rocmfpx`, `test_gfx1151_width4_dispatch.cpp`,
   `test_rocmfp4_hip_tail.cpp`) already tuned for this iGPU's bandwidth-bound
   profile.
2. **A DiffusionGemma backend already exists** (`server/src/diffusion/`,
   rebased onto current `upstream/main` on `feat/diffusion-foundation`), and
   it uses the *same* `ggml-cuda.h` / `ggml_backend_cuda_init` /
   `LUCE_BACKEND_CUDA` abstraction gemma4 already runs through on `gfx1151` —
   confirmed by direct grep, not assumed. It is **not yet GPU-validated at
   all** (its own README: "Phase 2 — implemented, GPU-build-gated, not yet
   run"), so this plan's Phase 0 is proving basic correctness before any
   Strix Halo-specific work, on whatever CUDA/HIP box is available first.

What's genuinely new work: the **structured-read algorithm itself** (canvas
seeding + position-pinning + single/few-step distribution read — djev-spark's
actual trick) doesn't exist in lucebox yet, and neither does **quantization,
batching, or step-count tuning for the diffusion path** on any hardware. Both
of those are the real content of this plan.

## djev-spark's numbers, for an honest target (not a promise to match)

DGX Spark (GB10/Blackwell, aarch64, CUDA 13, 121 GB unified, native NVFP4
tensor cores): single-read latency 98.4–101.7 ms at canvas width 32; 49.47
req/s at 32 concurrent clients (`MAX_SEQS=32`), p95 0.81s; DiffusionGemma
26B-A4B NVFP4 at 19 GB resident.

Strix Halo has no native FP4 tensor-core path (RDNA3.5, software 4-bit MMQ
only) and roughly half-to-a-third the memory bandwidth of GB10's HBM. **The
honest target is not matching DGX Spark's absolute throughput** — it's the
best achievable single-read latency and modest concurrent throughput on a
~$2-3k consumer box that owns its weights outright, benchmarked with the
*same methodology* (canvas width 32, p95 at N concurrent clients) so the
comparison is apples-to-apples rather than aspirational.

## Phases

### Phase 0 — Prove correctness (blocking, not Halo-specific)

The diffusion backend's own status notes flag two unvalidated things:
denoising contract (timestep/noise-level conditioning, schedule) is not
modeled — it currently just feeds uniform-state noise and reads logits — and
this "must be validated against published model behaviour." Do this first, on
any available CUDA or HIP box:

- Build `feat/diffusion-foundation`, load real DiffusionGemma weights, run
  `smoke_diffusion_gemma_forward` / `smoke_diffusion_gemma_sc_on` and compare
  outputs against a reference (e.g. the model's HF `transformers`
  implementation or vLLM's own DiffusionGemma path) on a handful of prompts.
- If the denoising contract is wrong (schedule/conditioning mismatch), fix it
  in `diffusion_decoder.cpp`/`diffusiongemma4_graph.cpp` before anything else
  — every later phase's numbers are meaningless against a model that isn't
  actually denoising correctly.
- Land model-card → `DiffusionConfig` plumbing through `BackendArgs` (the
  backend's own documented Phase 4 gap) so a `diffusiongemma-26b` model card
  entry can carry real decode defaults instead of the factory's hardcoded
  default config.

**Exit criteria:** DiffusionGemma forward pass output matches a trusted
reference within acceptable tolerance on a fixed prompt set.

### Phase 1 — Get it running on gfx1151, unquantized, no structured reads yet

- Build with `-DLUCE_HIP_ARCHITECTURES=gfx1151`, load DiffusionGemma
  26B-A4B in fp16/bf16 (won't fit comfortably alongside a large KV pool in
  128 GiB at fp16 for a 26B-A4B MoE — check real resident size; drop to Q8 if
  needed as an interim step before Phase 3's proper 4-bit work).
- Run the existing plain-generation path (`/v1/chat/completions` through the
  diffusion backend) end-to-end on real Strix Halo hardware. Establish a
  baseline: tokens/s, denoise-step latency, canvas-width scaling.
- Confirm no CUDA-only code path silently no-ops or misbehaves under HIP —
  the `#ifdef LUCE_BACKEND_CUDA` guards in `diffusion_gemma.cpp` need the same
  audit gemma4 already passed, not just a grep-level "it uses the same
  macro" assumption.

**Exit criteria:** DiffusionGemma denoises correctly on real gfx1151
hardware, with a measured (not estimated) baseline tok/s and per-step
latency.

### Phase 2 — Build the structured-read primitive (djev's actual algorithm)

**Status: core primitive + wiring done (see Status section above); the
real-model spot check below is not — still blocked on hardware.**

This was the piece lucebox didn't have yet, on any hardware. `/v1/systemone`
(the earlier openjev work) only ever supported causal backends — it forces a
single next-token position on an AR model and restricts vocab logits. That's
a *different, weaker* mechanism than djev's real trick, which needs genuine
bidirectional diffusion:

- ✅ **Canvas-seeding mode**, `run_diffusion_structured_read()` in
  `diffusion_decoder.cpp`/`.h`: given a rendered prompt/template ending
  exactly where the answer belongs (the same convention `/v1/systemone`
  already used for causal backends) and a slot count, seeds the canvas with
  noise at the answer slot(s) (reusing the decoder's existing
  masked/uniform-state noise machinery), runs N denoise steps (N=1 by
  default, matching djev-spark's own "one denoise step gives a distribution
  over each slot" claim), and reads the per-slot logit distribution
  directly — no full generation, no EOS handling.
- ❌ **Position-pinning for a *mixed* pinned+noise block** (fixed template
  tokens interleaved with free answer slots in one block, matching djev's
  `diffusion_pinned` behavior) — **not built**. Turned out unnecessary for
  the MVP: because `/v1/systemone` already renders prompts to end exactly at
  the answer position (no trailing fixed template after the noise slot),
  the causal prefix/bidirectional-block split handles pinning implicitly —
  everything before the block is prefix (fixed, causally encoded), the
  block itself is pure noise. A real mixed-block need (fixed tokens *inside*
  the bidirectional block) is still a documented follow-up if a future
  template shape requires it.
- ✅ **`/v1/systemone` routing** — turned out to need no new HTTP-layer
  branch at all: `DiffusionBackend::generate_impl` populates the exact same
  `GenerateResult::first_token_logits` field the causal backends already
  fill, so the endpoint's existing rendering/tokenization/restriction code
  (`systemone_softmax_restricted` et al.) works unmodified. Only change:
  `"diffusion-gemma"` added to the backend allowlist.

**Exit criteria (revised — split in two):**
- ✅ Correctness test suite on synthetic models — done, 57/57 (see Status).
- ⛔ Real-model spot check against djev-spark's own JevBench-style
  questions — **not done, blocked on hardware.** Nothing above has run a
  real DiffusionGemma forward pass. Do this as part of Phase 0/1 once a
  CUDA/HIP box is available, before trusting any answer this path produces.

### Phase 3 — Strix Halo-specific efficiency work

Strix Halo is **bandwidth-bound** (LPDDR5X-8000 unified, no dedicated VRAM;
the README says this explicitly for the existing DFlash/DDTree tuning). The
diffusion backend's current implementation is a **stateless full recompute
per denoise step** — its own README flags this as a known limitation. On a
bandwidth-bound iGPU, restreaming the full model's weights from unified
memory on every one of N denoise steps is the single biggest inefficiency to
attack, well before raw compute tuning matters:

1. **4-bit weights via `rocmfp4`/`rocmfpx`.** Reuse the existing HIP MMQ
   kernels and mixed-precision policy framework already built and tuned for
   `gfx1151` (the same ones DeepSeek4 uses there today) to quantize
   DiffusionGemma 26B-A4B. This is the direct Strix Halo analogue of
   djev-spark's NVFP4 — same motivation (cut memory traffic ~4×), different
   mechanism (no native FP4 tensor cores on RDNA3.5, so it's lucebox's
   existing software-packed MMQ path, not hardware FP4).
2. **Warm-prefix + block-incremental canvas updates**, replacing the
   stateless full-recompute-per-step path. Only the KV state for the prompt
   prefix and the (typically small) answer-slot region needs to be
   live-recomputed each denoise step; the rest of the canvas is unchanged
   between steps once padded/pinned. This is the single highest-leverage
   change for a bandwidth-bound chip, more valuable here than on DGX Spark's
   much higher-bandwidth HBM.
3. **Batch canvases across concurrent `/v1/systemone` requests** into one
   denoise-step GPU dispatch — the direct analogue of djev-spark's vLLM
   continuous batching (its main throughput lever, `MAX_SEQS=32`). Lucebox
   already has paged/multi-slot serving machinery for causal backends
   (`SeqEngine`, paged KV pools on qwen35/deepseek4); the diffusion path
   needs its own batched-canvas variant, since a denoise step is naturally
   a batched forward pass over positions and multiple concurrent canvases
   can share that batch dimension cheaply.
4. **Denoise-step count tuning for `gfx1151`'s bandwidth profile**
   (`eb_max_steps`/entropy-budget, already an existing knob), added to the
   per-arch tuning table `server/README.md` already keeps for DDTree budgets.
   Expect the optimal step count to be *lower* than a compute-bound GPU's
   default, since every extra step costs a full bandwidth-bound pass here.

**Exit criteria:** each of the four levers benchmarked independently
(before/after single-read latency and throughput), so the plan's final
numbers can be attributed rather than reported as one opaque total.

### Phase 4 — Benchmark and compare

Reproduce djev-spark's own benchmark shape exactly, so the result is
directly comparable rather than a different methodology dressed up as one:
single-read latency at canvas width 32, throughput and p95 latency at a swept
concurrency level (start at the same `MAX_SEQS=32` djev-spark used, scale
down if Strix Halo's memory/compute ceiling makes that unrealistic — report
the real ceiling rather than forcing the same number). Publish results
alongside djev-spark's numbers with the hardware disparity stated plainly
(GB10 Blackwell + native FP4 vs `gfx1151` + software 4-bit MMQ), not implied
away.

## Open risks to flag before committing engineering time

- **Phase 0 is a real gate, not a formality.** The diffusion backend has
  never run against real weights. If the denoising contract is wrong, every
  later phase is built on a broken foundation.
- **Memory headroom.** DiffusionGemma 26B-A4B at fp16 may not comfortably
  coexist with a meaningful KV/canvas budget in 128 GiB unified memory shared
  with the OS and other processes — verify real resident size on Phase 1
  before assuming Phase 3's 4-bit work is optional rather than required.
- **No native FP4 on RDNA3.5** means the memory-bandwidth win from
  quantization is real, but the *compute* speedup NVFP4 gets from Blackwell's
  dedicated tensor cores has no Strix Halo equivalent — don't extrapolate
  djev-spark's NVFP4 numbers onto ROCmFP4 without re-measuring.
- **Full-recompute-per-step is undocumented cost today.** Until Phase 3.2
  lands, every Phase 1/2 number is measuring the *unoptimized* diffusion
  path — treat early benchmarks as a floor, not a target.
