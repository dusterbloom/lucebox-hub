// Model-agnostic diffusion decode loop — the core of the abstraction.
//
// Given any DiffusionModelGraph, a prompt and a DiffusionConfig, this runs
// semi-autoregressive masked-diffusion decoding: a fixed-width block is seeded
// with noise and refined over a few denoising steps, finalizing the
// highest-confidence positions on a schedule (or above a probability threshold)
// while keeping the rest noised. Each fully-denoised block is committed and
// streamed left-to-right, then the next block begins — giving unbounded length
// and ordinary token-streaming semantics on top of parallel generation.
//
// References: LLaDA / Dream low-confidence remasking; Fast-dLLM confidence-aware
// parallel decode; BD3-LM / LLaDA2 semi-autoregressive block diffusion.
//
// Deliberately ggml-free (depends only on the sampler + the model seam) so the
// loop is unit-testable on CPU. Streaming is delivered through a plain callback;
// DiffusionBackend adapts the server's DaemonIO to it.

#pragma once

#include <cstdint>
#include <functional>
#include <string>
#include <vector>

#include "diffusion_types.h"
#include "diffusion_model.h"
#include "sampler.h"

namespace dflash::common {

// Streaming sink. on_token is invoked once per committed token, in output
// order. Return false to abort generation (e.g. client disconnect); the loop
// then returns the tokens committed so far with ok=true.
struct DiffusionStream {
    std::function<bool(int32_t)> on_token;
};

struct DiffusionDecodeResult {
    bool                 ok = false;
    std::string          error;   // "" on success; e.g. "forward", "config"
    std::vector<int32_t> tokens;  // committed tokens (excludes the prompt)
    DiffusionStats       stats;
};

// Run a full diffusion generation. `model` must already be constructed; the
// loop calls model.prepare(prompt) then iterates blocks until `n_gen` tokens
// are committed, EOS is produced, the context limit is hit, or the stream is
// aborted. `sampler`/`do_sample` follow the same semantics as the AR backends
// (do_sample false => greedy/argmax). `stream.on_token` may be empty.
//
// When cfg.remasking == DiffusionRemask::EntropyBound the entropy-bound
// denoiser (DiffusionGemma-style) is used: single canvas of `n_gen` tokens,
// uniform-random init, linear temperature schedule, acceptance by sorted
// Shannon entropy within the MI bound, self-conditioning via model.set_sc().
DiffusionDecodeResult run_diffusion_generate(
    DiffusionModelGraph &        model,
    const std::vector<int32_t> & prompt,
    int                          n_gen,
    const DiffusionConfig &      cfg,
    const SamplerCfg &           sampler,
    bool                         do_sample,
    const DiffusionStream &      stream);

}  // namespace dflash::common
