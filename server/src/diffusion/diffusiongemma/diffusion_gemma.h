// DiffusionGemmaGraph — DiffusionModelGraph over the Gemma 4 backbone.
//
// DiffusionGemma (Google) is a diffusion LLM built on a standard Gemma 4 MoE
// backbone with a bidirectional decoder mode (Uniform-State Diffusion). This
// adapter reuses lucebox's existing gemma4 loader / weights / KV cache and runs
// the bidirectional denoising forward (gemma4_denoise_batch); the model-agnostic
// loop in diffusion_decoder.cpp drives the noise/remask schedule and streaming.
//
// Uniform-state noise => no dedicated [MASK] token (mask_token() returns -1).

#pragma once

#include <cstdint>
#include <vector>

#include "diffusion_model.h"
#include "gemma4_internal.h"
#include "ggml-backend.h"
#include "ggml.h"

namespace dflash::common {

struct DiffusionGemmaConfig {
    const char * model_path = nullptr;
    int          gpu        = 0;
    int          max_ctx    = 4096;
};

class DiffusionGemmaGraph : public DiffusionModelGraph {
public:
    explicit DiffusionGemmaGraph(const DiffusionGemmaConfig & cfg);
    ~DiffusionGemmaGraph() override;

    DiffusionGemmaGraph(const DiffusionGemmaGraph &)             = delete;
    DiffusionGemmaGraph & operator=(const DiffusionGemmaGraph &) = delete;

    bool init();  // load weights + allocate KV cache; false on failure

    int     vocab() const override     { return w_.n_vocab; }
    int32_t eos_token() const override { return w_.eos_id; }
    int32_t mask_token() const override { return -1; }  // uniform-state noise
    int     n_ctx_max() const override { return cfg_.max_ctx; }

    bool prepare(const std::vector<int32_t> & prompt, int & out_prefix_len) override;

    // forward_block: canvas = full [prompt|canvas] token sequence; block_begin is
    // the first canvas position (= prompt length), block_len is C. SC state is
    // threaded in via set_sc() (called by the EB decode loop each step).
    bool forward_block(const std::vector<int32_t> & canvas, int block_begin,
                       int block_len, bool bidirectional,
                       std::vector<float> & out_logits) override;

    // Self-conditioning state setters (called by the EB decode loop each step).
    // sc_logits: [n_vocab * C] F32, prev-step raw canvas logits. Must remain valid
    //            until the next forward_block call completes.
    // sc_use:    0.0 on step 0 (zeroes SC signal), 1.0 thereafter.
    // sc_temp_inv: 1/temperature for SC softmax over vocab.
    void set_sc(const float * sc_logits, float sc_use, float sc_temp_inv);

    void reset() override;

private:
    // Lazily build sc_embT_: tok_embd transposed + dequantized to {n_vocab, n_embd}
    // F16 on the device. Called once on first SC-enabled forward.
    bool ensure_sc_embT();

    DiffusionGemmaConfig cfg_;
    ggml_backend_t       backend_ = nullptr;
    Gemma4Weights        w_;
    Gemma4Cache          cache_;
    std::vector<float>   embed_;   // scratch [n_embd * n_tokens]
    bool                 loaded_ = false;

    // Prompt length from last prepare() call — used as P in the unified forward.
    int prefix_len_ = 0;

    // Self-conditioning state (updated by set_sc each step)
    const float * sc_logits_ptr_ = nullptr;  // borrowed, valid for one forward
    float         sc_use_        = 0.0f;
    float         sc_temp_inv_   = 1.0f;

    // sc_embT: {n_vocab, n_embd} F16 device tensor (built lazily from tok_embd).
    // Persists across steps; only rebuilt if vocab/embd change (doesn't happen).
    ggml_context *        sc_embT_ctx_ = nullptr;
    ggml_backend_buffer_t sc_embT_buf_ = nullptr;
    ggml_tensor *         sc_embT_     = nullptr;
};

}  // namespace dflash::common
