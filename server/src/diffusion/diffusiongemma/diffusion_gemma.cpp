// DiffusionGemmaGraph implementation. See diffusion_gemma.h.

#include "diffusion_gemma.h"

#include "dflash27b.h"   // dflash27b_last_error
#include "ggml-cuda.h"   // ggml_backend_cuda_init

#include <cmath>
#include <cstdio>

namespace dflash::common {

DiffusionGemmaGraph::DiffusionGemmaGraph(const DiffusionGemmaConfig & cfg) : cfg_(cfg) {}

DiffusionGemmaGraph::~DiffusionGemmaGraph() {
    if (loaded_) {
        free_gemma4_cache(cache_);
        free_gemma4_weights(w_);
        loaded_ = false;
    }
    if (backend_) { ggml_backend_free(backend_); backend_ = nullptr; }
}

bool DiffusionGemmaGraph::init() {
    backend_ = ggml_backend_cuda_init(cfg_.gpu);
    if (!backend_) {
        std::fprintf(stderr, "[diffusiongemma] CUDA backend init failed (gpu=%d)\n", cfg_.gpu);
        return false;
    }
    if (!load_gemma4_gguf(cfg_.model_path, backend_, w_)) {
        std::fprintf(stderr, "[diffusiongemma] GGUF load failed: %s\n", dflash27b_last_error());
        return false;
    }
    if (!create_gemma4_cache(backend_, w_, cfg_.max_ctx, cache_)) {
        std::fprintf(stderr, "[diffusiongemma] cache alloc failed\n");
        free_gemma4_weights(w_);
        return false;
    }
    cache_.fa_window = 0;  // full attention — diffusion decode is global
    loaded_ = true;
    std::printf("[diffusiongemma] init ok: %d layers, embd=%d, vocab=%d, swa=%d, max_ctx=%d\n",
                w_.n_layer, w_.n_embd, w_.n_vocab, w_.sliding_window, cfg_.max_ctx);
    std::fflush(stdout);
    return true;
}

bool DiffusionGemmaGraph::prepare(const std::vector<int32_t> & prompt, int & out_prefix_len) {
    // Stateless full-recompute forward (see gemma4_denoise_batch): there is no
    // KV prefix to warm here — the whole canvas is re-embedded each step.
    out_prefix_len = (int)prompt.size();
    cache_.cur_pos = 0;
    return loaded_;
}

bool DiffusionGemmaGraph::forward_block(const std::vector<int32_t> & canvas,
                                        int block_begin, int block_len,
                                        bool /*bidirectional*/,
                                        std::vector<float> & out_logits) {
    if (!loaded_) return false;
    const int n = (int)canvas.size();
    if (block_begin < 0 || block_len <= 0 || block_begin + block_len > n) return false;

    const int hidden = w_.n_embd;
    embed_.resize((size_t)n * hidden);
    if (!w_.embedder.embed(canvas.data(), n, embed_.data())) {
        std::fprintf(stderr, "[diffusiongemma] embed failed (n=%d)\n", n);
        return false;
    }
    // Gemma scales input embeddings by sqrt(n_embd).
    const float scale = std::sqrt((float)hidden);
    for (size_t i = 0; i < embed_.size(); ++i) embed_[i] *= scale;

    return gemma4_denoise_batch(backend_, w_, cache_, embed_.data(),
                                canvas.data(), n, block_begin, block_len, out_logits);
}

void DiffusionGemmaGraph::reset() { cache_.cur_pos = 0; }

}  // namespace dflash::common
