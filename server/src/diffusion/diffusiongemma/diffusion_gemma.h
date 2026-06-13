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

#include <vector>

#include "diffusion_model.h"
#include "gemma4_internal.h"
#include "ggml-backend.h"

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
    bool forward_block(const std::vector<int32_t> & canvas, int block_begin,
                       int block_len, bool bidirectional,
                       std::vector<float> & out_logits) override;
    void reset() override;

private:
    DiffusionGemmaConfig cfg_;
    ggml_backend_t       backend_ = nullptr;
    Gemma4Weights        w_;
    Gemma4Cache          cache_;
    std::vector<float>   embed_;   // scratch [n_embd * canvas_len]
    bool                 loaded_ = false;
};

}  // namespace dflash::common
