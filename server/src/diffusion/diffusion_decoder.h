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

namespace luce::common {

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
    const DiffusionStream &      stream,
    int                          prepared_prefix_len = -1);

// ─── Structured read (djev-style canvas seeding) ───────────────────────────
//
// A "structured read" costs one (or a few) denoise steps, not a full
// generation: `prefix_tokens` is rendered so it ends exactly where the
// answer belongs (the same convention /v1/systemone already uses for
// causal backends — see http_server.cpp's systemone_question_suffix), and
// `slot_count` free positions right after it are seeded with noise and
// denoised in place. Unlike run_diffusion_generate, nothing is committed to
// a growing canvas and no EOS/length stopping applies — the caller wants
// the raw per-slot probability distribution, not committed tokens.
//
// `slot_count` > 1 supports a multi-token answer label read in one shot
// (all slot positions denoise jointly, bidirectionally, exactly like a
// generation block); it does NOT support mixing pinned/fixed tokens with
// noise within one block — every position in [prefix_len, prefix_len +
// slot_count) is noise. That's a deliberate MVP scope: it's enough for
// /v1/systemone's one-slot-per-question protocol today; a mixed pinned+
// noise block is a documented follow-up if multi-position templates are
// ever needed (see server/docs/djev-halo-plan.md, Phase 2).
struct DiffusionReadResult {
    bool                  ok = false;
    std::string           error;
    int                   slot_count = 0;
    int                   vocab      = 0;
    // Row-major: slot_count rows of `vocab` raw (pre-softmax) logits from the
    // FINAL denoise step. Caller restricts/softmaxes over candidate answer
    // tokens per slot, same as the causal /v1/systemone path already does.
    std::vector<float>    slot_logits;
    std::vector<int32_t>  slot_argmax;
    int                   forward_passes = 0;
};

// Runs `n_steps` (>=1) denoise steps over a `slot_count`-wide noise block
// seeded right after `prefix_tokens`, and returns the final step's per-slot
// logits. `n_steps<=0` defaults to 1, matching djev-spark's own claim that
// "one denoise step gives a distribution over each slot." `cfg.seed` (or an
// explicit `seed` override, taking precedence when nonzero) drives the
// initial noise draw for DiffusionNoise::UniformState; Masked noise instead
// seeds every slot with the model's mask token and ignores the seed.
DiffusionReadResult run_diffusion_structured_read(
    DiffusionModelGraph &        model,
    const std::vector<int32_t> & prefix_tokens,
    int                          slot_count,
    int                          n_steps,
    const DiffusionConfig &      cfg,
    uint64_t                     seed = 0,
    int                          prepared_prefix_len = -1);

}  // namespace luce::common
