# Diffusion (dLLM) module

A general abstraction for serving **diffusion language models** (dLLMs) in
lucebox. Where the existing arches decode autoregressively (one token at a time,
left to right), diffusion models seed a block of positions with noise and refine
it over a few **denoising steps**, generating many tokens per forward. This
module lets such models plug into the existing `ModelBackend` factory exactly
like an AR arch.

Target families (both shipped mid-2026):

| Family | Backbone | Mode(s) | Noise | Maps onto |
|---|---|---|---|---|
| **DiffusionGemma** (Google) | Gemma 4 (MoE) | encoder (causal, writes KV) + decoder (bidirectional, denoise) | uniform-state | the existing `gemma4` graph + a bidirectional block mask |
| **Nemotron-Labs-Diffusion** (NVIDIA) | dense | AR / diffusion / self-speculation | masked | a dense backbone graph; self-spec → DFlash/DDTree (future) |

## Layers

```
DiffusionBackend  : ModelBackend      diffusion_backend.{h,cpp}
  └─ run_diffusion_generate(...)      diffusion_decoder.{h,cpp}   ← the abstraction
       └─ DiffusionModelGraph (seam)  diffusion_model.h
            ├─ DiffusionGemmaGraph    diffusion/diffusiongemma/   (implemented)
            └─ NemotronDiffusionGraph diffusion/nemotron/         (pending)
  DiffusionConfig / enums             diffusion_types.h
  family sub-factory                  diffusion_registry.{h,cpp}
```

- **`diffusion_decoder`** owns the algorithm and is **ggml-free** (depends only
  on `sampler.h` + the model seam) so it is unit-testable on CPU: semi-AR block
  decoding, per-step confidence, remasking policy (low-confidence / random /
  parallel-threshold), masked + uniform-state noise, block-granular streaming,
  EOS/length stop.
- **`DiffusionModelGraph`** is the per-family seam: `forward_block(canvas,
  block_begin, block_len, bidirectional) -> per-position logits`. Adding a dLLM
  family = implement this + register it in `diffusion_registry.cpp`.
- **`DiffusionBackend`** routes `ModelBackend::generate()` through the loop and
  stubs the AR-only surface (KV snapshots, pflash compress, DFlash). It is the
  only translation unit here that touches ggml (via `model_backend.h`).

## Integration points

- **Factory dispatch** — `common/backend_factory.cpp` routes
  `general.architecture ∈ {diffusiongemma, nemotron_diffusion}` to
  `create_diffusion_backend()`. (If a converted GGUF reuses a backbone arch like
  `gemma4`, prefer a `<arch>.diffusion.*` metadata flag read in
  `gguf_inspect.cpp` to disambiguate — a small follow-up.)
- **Model cards** — `share/model_cards/{diffusiongemma-26b,nemotron-diffusion-8b}.json`
  carry decode defaults under a `diffusion` block (schema in `_schema.json`).
  Wired end to end: `resolve_model_card` parses the block (string enums via the
  tested helpers in `diffusion_config.cpp`), the server copies it into
  `BackendArgs.diffusion` for diffusion arches, and the factory threads it into
  the `DiffusionBackend`.

## Status

- **Phase 1 (done, CPU-tested):** the abstraction, the decode loop, the backend
  + registry scaffolding, and `test/test_diffusion_decoder.cpp` (17 cases).
- **Phase 2 (DiffusionGemma — implemented, GPU-build-gated, not yet run):**
  `diffusiongemma/diffusion_gemma.{h,cpp}` wraps `Gemma4Weights` (reuses
  `load_gemma4_gguf` + cache) and runs `gemma4_denoise_batch` — a bidirectional
  variant of `gemma4_verify_batch` (every query attends to every key; per-block
  logit readout), reusing `build_gemma4_layer` for the exact gemma4 compute. The
  registry + backend_factory route `general.architecture == "diffusiongemma"` to
  it. **First-cut caveats to verify on GPU with real weights:** (a) stateless
  full-recompute per step, limited to `canvas <= SWA ring` (warm-prefix +
  block-incremental is the optimization follow-up); (b) the exact DiffusionGemma
  denoising contract (timestep/noise-level conditioning, schedule) is not modeled
  — the loop feeds uniform-state noise and reads logits, which must be validated
  against the published model behaviour.
- **Phase 3 (Nemotron — pending):** needs a dedicated dense backbone
  loader/graph (its arch differs from qwen3); best implemented against the real
  model config + a GPU build rather than fabricated blind. The registry branch is
  stubbed (`create_diffusion_model` returns nullptr with a diagnostic).
- **Phase 4 (partial):** model-card → `DiffusionConfig` plumbing **done**
  (`BackendArgs.diffusion` + server wiring + CPU-tested string↔enum helpers).
  Remaining: warm-prefix KV path (lifts the `canvas <= SWA ring` limit), a
  `smoke_diffusion_forward` harness, and `/v1/chat/completions` e2e.

> The ggml/CUDA code (Phase 2 + factory wiring) requires a GPU build (the CI
> runners or a local CUDA/ggml toolchain) to compile and test — it cannot be
> built in a CPU-only container. Phase 1's loop is verified independently via the
> CPU CTest, which is the gate that runs here.

## Build & test

```bash
# Full build (GPU toolchain + git submodules required):
cmake -S server -B server/build && cmake --build server/build -j
ctest --test-dir server/build -R diffusion_decoder

# Just the decode loop, CPU-only (no CUDA/ggml needed):
g++ -std=c++17 -O2 -Iserver/src -Iserver/src/diffusion -Iserver/src/common \
    server/test/test_diffusion_decoder.cpp \
    server/src/diffusion/diffusion_decoder.cpp \
    server/src/common/sampler.cpp -o /tmp/td && /tmp/td
```
