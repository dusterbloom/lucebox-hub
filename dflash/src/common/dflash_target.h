// dflash_target.h — Interface that any target model must implement to support
// DFlash speculative decoding with the universal DFlash draft model.
//
// The DFlash draft model (z-lab/DFlashDraftModel) is a single generic Qwen3-style
// architecture that works with ANY target model. It cross-attends to intermediate
// features captured during the target's forward pass, and outputs hidden states
// in the target's representation space. The target's own lm_head then projects
// those hidden states to token IDs.
//
// A target backend implements this interface to opt into DFlash spec decode.

#pragma once

#include <cstdint>
#include <vector>

struct ggml_backend;
typedef struct ggml_backend * ggml_backend_t;

namespace dflash27b {

struct DFlashTarget {
    virtual ~DFlashTarget() = default;

    // Return the ggml backend used by this target's graph compute.  Default
    // returns nullptr; callers (e.g. Qwen3.6 MTP) that want to build CUDA
    // cgraphs against the same backend should check this and fall back if
    // it's null.
    virtual ggml_backend_t backend() const { return nullptr; }

    // ── Target forward ──────────────────────────────────────────────

    // Run a batch of tokens through the target model.  Returns the argmax
    // of the last token in `last_tok`.  If `all_argmax` is non-null, fills
    // it with argmax for every position (used during spec-decode verify).
    //
    // During forward, the target MUST capture intermediate activations at
    // the layers specified by capture_layer_ids() and store them in the
    // draft's feature ring (how this happens is implementation-defined).
    virtual bool verify_batch(const std::vector<int32_t> & tokens,
                              int base_pos,
                              int & last_tok,
                              std::vector<int32_t> * all_argmax = nullptr) = 0;

    // ── KV state management ─────────────────────────────────────────

    // Snapshot KV cache state before speculative verify, so it can be
    // rolled back if tokens are rejected.
    virtual bool snapshot_kv() = 0;

    // Restore KV cache to the last snapshot (undo speculative forward).
    virtual bool restore_kv() = 0;

    // ── Token utilities ─────────────────────────────────────────────

    // Check if a token is end-of-sequence for this model.
    virtual bool is_eos(int token) const = 0;

    // Embed token IDs using the target's embedding table.
    // Output: `out` must have space for `n * hidden_size()` floats.
    virtual bool embed_tokens(const int32_t * tokens, int n,
                              float * out) const = 0;

    // ── LM head projection ──────────────────────────────────────────

    // Project draft hidden states through the target's lm_head
    // (out_norm + output weight) to get token IDs via argmax.
    // `hidden` has shape [n_tokens * hidden_size()].
    virtual bool project_hidden_to_tokens(const float * hidden,
                                          int n_tokens,
                                          std::vector<int32_t> & tokens_out) = 0;

    // ── Configuration for draft model ───────────────────────────────

    // Target's hidden dimension (draft model must match).
    virtual int hidden_size() const = 0;

    // Mask token ID in the target's vocabulary (used for noise input).
    virtual int mask_token_id() const = 0;

    // Which target layers to capture intermediate activations from.
    // The draft model's fc layer expects exactly this many feature slices.
    virtual const std::vector<int> & capture_layer_ids() const = 0;

    // Return the backbone's final post-norm hidden state for the last committed
    // token (hidden_size() floats, F32).  Populated by verify_batch.
    // Returns nullptr if not yet available (e.g. before first verify_batch).
    // Default implementation returns nullptr; Qwen35DFlashTarget overrides it.
    virtual const float * last_hidden() const { return nullptr; }

    // Return the full post-norm hidden sequence from the MOST RECENT
    // verify_batch call: n_tokens * hidden_size() floats, F32, laid out as
    // [token_0_hidden, token_1_hidden, ..., token_{n_tokens-1}_hidden].
    // *out_n_tokens is set to the number of tokens captured (matches the
    // n_tokens passed to verify_batch).  Default returns nullptr.
    virtual const float * last_hidden_seq(int * out_n_tokens) const {
        if (out_n_tokens) *out_n_tokens = 0;
        return nullptr;
    }

    // Return the post-norm hidden at an ABSOLUTE sequence position, if that
    // position is covered by the most recent verify_batch's hidden capture.
    // The Qwen3.6 MTP head needs h_{base_pos-1} for its input pair at each
    // chain step, which equals last_hidden() only on the first chain step
    // (right after prefill); subsequent steps need a hidden from earlier in
    // the most recent verify_batch chunk.  Returns nullptr if out of range.
    virtual const float * hidden_at_pos(int abs_pos) const {
        (void)abs_pos;
        return nullptr;
    }
};

} // namespace dflash27b
