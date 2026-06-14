// Per-family forward seam for diffusion (denoising) decoding.
//
// A DiffusionModelGraph wraps a transformer backbone (e.g. the gemma4 graph)
// and exposes a single batched forward returning logits for a contiguous block
// of canvas positions. The decoder loop (diffusion_decoder.cpp) owns the
// canvas, the noise/remask schedule, sampling and streaming; the graph owns
// weights, the compute backend, and attention-mask construction (causal over
// the prompt prefix, bidirectional within the block being denoised).
//
// This is the extension point: adding a new dLLM family = implement this
// interface and register it (diffusion_registry.cpp), exactly as adding a new
// autoregressive arch means implementing ModelBackend.
//
// Deliberately ggml-free so the decoder loop can be exercised against a
// synthetic implementation in unit tests.

#pragma once

#include <cstdint>
#include <vector>

namespace dflash::common {

struct DiffusionModelGraph {
    virtual ~DiffusionModelGraph() = default;

    // ── Static identity ──────────────────────────────────────────────
    virtual int     vocab() const = 0;       // logits row count
    virtual int32_t eos_token() const = 0;   // decoder stops a block early on this id
    virtual int     n_ctx_max() const = 0;   // max prompt + generated positions

    // Canonical [MASK] token id, or -1 when the family has none (uniform-state
    // noise). The decoder resolves DiffusionConfig::mask_token_id from this
    // when the request/card leaves it unset.
    virtual int32_t mask_token() const { return -1; }

    // ── Per-request lifecycle ────────────────────────────────────────
    // Prepare to denoise after `prompt`. Implementations may warm a KV cache
    // (encoder/causal pass) so later forward_block calls only recompute the
    // active block. `out_prefix_len` is the number of committed prompt
    // positions. Returns false on failure.
    virtual bool prepare(const std::vector<int32_t> & prompt,
                         int & out_prefix_len) = 0;

    // One denoising forward over canvas positions [block_begin, block_begin+block_len).
    // `canvas` holds the full sequence so far (prompt prefix + committed blocks +
    // the active block, the latter carrying mask/noise tokens). When
    // `bidirectional` is true the block positions attend to one another without a
    // causal mask (diffusion decode); when false the pass is causal (AR/encoder).
    // Fills `out_logits` row-major as block_len rows of `vocab()` floats.
    // Returns false on failure.
    virtual bool forward_block(const std::vector<int32_t> & canvas,
                               int block_begin, int block_len,
                               bool bidirectional,
                               std::vector<float> & out_logits) = 0;

    // Self-conditioning state injection (DiffusionGemma entropy-bound path).
    // Called by the EB decode loop once per step before forward_block.
    // sc_logits: [n_vocab * block_len] F32 raw canvas logits from the previous
    //            step. Borrowed — must remain valid until forward_block returns.
    //            Pass nullptr to disable SC (step 0).
    // sc_use:    0.0 on step 0 (zeroes SC signal), 1.0 thereafter.
    // sc_temp_inv: 1/t from the previous step's temperature.
    // Default is a no-op for non-DiffusionGemma families.
    virtual void set_sc(const float * /*sc_logits*/, float /*sc_use*/,
                        float /*sc_temp_inv*/) {}

    // Release per-request scratch (e.g. KV cache contents) before the next prompt.
    virtual void reset() {}
};

}  // namespace dflash::common
