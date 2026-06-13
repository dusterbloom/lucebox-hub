// Shared value types for the abstracted diffusion (dLLM) module.
//
// lucebox serves autoregressive transformers today; diffusion language models
// (DiffusionGemma, Nemotron-Labs-Diffusion, …) instead generate a block of
// tokens in parallel and refine it over a few denoising steps. These types
// describe the noise scheme and remasking schedule that the model-agnostic
// decoder loop (diffusion_decoder.cpp) drives; the per-family forward seam
// lives in diffusion_model.h.
//
// Deliberately ggml-free so the loop is unit-testable on CPU without the GPU
// backend.

#pragma once

#include <cstdint>

namespace dflash::common {

// How corrupted/undetermined canvas positions are represented between steps.
enum class DiffusionNoise {
    Masked,        // absorbing-state: undetermined positions hold a [MASK] token id
    UniformState,  // uniform-state: undetermined positions hold a random vocab token
                   // (DiffusionGemma's scheme — no dedicated [MASK] token)
};

// Policy for choosing which positions to finalize (and which to keep
// noised/remasked) after each denoising step.
enum class DiffusionRemask {
    LowConfidence,     // finalize the highest-confidence positions on a linear
                       // schedule, keep the low-confidence ones noised (LLaDA-style)
    Random,            // finalize a random subset per step (ancestral sampling)
    ParallelThreshold, // finalize every position whose top probability exceeds
                       // `confidence_threshold` (Fast-dLLM-style parallel decode)
};

// Per-request decoding knobs. Populated from the model card defaults, then
// overridden by request fields / env where applicable.
struct DiffusionConfig {
    int             n_steps              = 0;   // denoising steps per block (<=0 => block_size)
    int             block_size           = 32;  // semi-AR block / canvas width
    DiffusionRemask remasking            = DiffusionRemask::LowConfidence;
    float           confidence_threshold = 0.9f; // used by ParallelThreshold
    DiffusionNoise  noise_scheme         = DiffusionNoise::Masked;
    int32_t         mask_token_id        = -1;  // required for the Masked scheme
                                                // (<0 => resolve from the model graph)
    bool            semi_ar              = true; // advance block-by-block (vs one canvas)
    uint64_t        seed                 = 0;    // Random remask / uniform-state noise
};

// Lightweight per-request counters surfaced to the server for /status + bench.
struct DiffusionStats {
    int forward_passes = 0;  // total model forwards (denoising steps across blocks)
    int blocks         = 0;  // committed semi-AR blocks
    int tokens         = 0;  // committed tokens
};

}  // namespace dflash::common
