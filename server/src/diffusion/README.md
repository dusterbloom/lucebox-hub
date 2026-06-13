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
            ├─ DiffusionGemmaGraph    diffusion/diffusiongemma/   (phase 2)
            └─ NemotronDiffusionGraph diffusion/nemotron/         (phase 3)
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
  Plumbing card → `DiffusionConfig` through `BackendArgs` is a follow-up; the
  factory currently constructs a default `DiffusionConfig`.

## Status

- **Phase 1 (done, CPU-tested):** the abstraction, the decode loop, the backend
  + registry scaffolding, and `test/test_diffusion_decoder.cpp` (17 cases).
- **Phase 2 (DiffusionGemma):** implement `DiffusionGemmaGraph` wrapping
  `Gemma4Weights` (reuse `load_gemma4_gguf`, `gemma4_internal.h`). The forward
  needs a **bidirectional block mask + per-position logits**: the gemma4 graph
  already takes host-built masks as graph inputs and `gemma4_verify_batch` /
  `compute_gemma4_split_projection` already emit per-position logits/argmax — add
  a `gemma4_denoise_batch` that (a) attends the block to the prefix KV + within
  the block (non-causal) and (b) does **not** persist the noised block into the
  causal KV cache across steps.
- **Phase 3 (Nemotron):** dense backbone loader/graph (start from the qwen3
  dense loader/graph); causal mask = AR mode, bidirectional = diffusion mode.
- **Phase 4:** model-card → `DiffusionConfig` plumbing; server smoke
  (`smoke_diffusion_forward`) + `/v1/chat/completions` e2e.

> Phases 2–4 touch the ggml/CUDA graph and require a GPU build (the CI runners)
> to compile and test — they cannot be built in a CPU-only container. Phase 1's
> loop is verified independently via the CPU CTest.

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
