// DiffusionGemmaGraph implementation. See diffusion_gemma.h.

#include "diffusion_gemma.h"

#include "dflash27b.h"   // dflash27b_last_error
#include "ggml-cuda.h"   // ggml_backend_cuda_init

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <thread>
#include <vector>

namespace dflash::common {

DiffusionGemmaGraph::DiffusionGemmaGraph(const DiffusionGemmaConfig & cfg) : cfg_(cfg) {}

DiffusionGemmaGraph::~DiffusionGemmaGraph() {
    if (sc_embT_buf_) { ggml_backend_buffer_free(sc_embT_buf_); sc_embT_buf_ = nullptr; }
    if (sc_embT_ctx_) { ggml_free(sc_embT_ctx_);                sc_embT_ctx_ = nullptr; }
    sc_embT_ = nullptr;
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
    // Phase-2: stateless full-recompute (no KV cache prefill). Record prompt length
    // so forward_block knows P (the prompt region size) for region-split logic.
    prefix_len_    = (int)prompt.size();
    out_prefix_len = prefix_len_;
    cache_.cur_pos = 0;
    return loaded_;
}

void DiffusionGemmaGraph::set_sc(const float * sc_logits, float sc_use, float sc_temp_inv) {
    sc_logits_ptr_ = sc_logits;
    sc_use_        = sc_use;
    sc_temp_inv_   = sc_temp_inv;
}

// Build sc_embT once: dequantize tok_embd {n_embd, n_vocab} and transpose to
// {n_vocab, n_embd} F16 in a device buffer. Mirrors dg_ensure_sc_embT in the
// reference (diffusion-gemma.cpp:591-651).
bool DiffusionGemmaGraph::ensure_sc_embT() {
    if (sc_embT_ != nullptr) return true;
    if (!w_.tok_embd) return false;

    ggml_tensor * src = w_.tok_embd;
    const int64_t n_embd  = src->ne[0];
    const int64_t n_vocab = src->ne[1];

    ggml_init_params ip{ ggml_tensor_overhead() * 2, nullptr, /*no_alloc=*/true };
    sc_embT_ctx_ = ggml_init(ip);
    if (!sc_embT_ctx_) return false;

    // sc_embT layout: {n_vocab, n_embd} F16 — A in ggml_mul_mat(sc_embT, probs)
    // computes sc_embT^T @ probs = {n_embd,n_vocab}^T @ {n_vocab,C} = {n_embd,C}.
    sc_embT_ = ggml_new_tensor_2d(sc_embT_ctx_, GGML_TYPE_F16, n_vocab, n_embd);
    ggml_set_name(sc_embT_, "sc_embT");

    ggml_backend_buffer_type_t buft = ggml_backend_get_default_buffer_type(backend_);
    sc_embT_buf_ = ggml_backend_alloc_ctx_tensors_from_buft(sc_embT_ctx_, buft);
    if (!sc_embT_buf_) {
        ggml_free(sc_embT_ctx_); sc_embT_ctx_ = nullptr; sc_embT_ = nullptr;
        return false;
    }
    ggml_backend_buffer_set_usage(sc_embT_buf_, GGML_BACKEND_BUFFER_USAGE_WEIGHTS);

    // Dequantize each vocab row on the host; scatter the transpose into F16.
    const ggml_type st       = src->type;
    const size_t    row_size = ggml_row_size(st, n_embd);
    std::vector<char> host_src((size_t)row_size * n_vocab);
    ggml_backend_tensor_get(src, host_src.data(), 0, host_src.size());

    std::vector<ggml_fp16_t> dstT((size_t)n_vocab * n_embd);
    const ggml_type_traits * tr = ggml_get_type_traits(st);

    const unsigned hw  = std::thread::hardware_concurrency();
    const unsigned nth = std::max(1u, std::min(hw ? hw : 1u, 32u));
    auto worker = [&](int64_t v0, int64_t v1) {
        std::vector<float> tmp(n_embd);
        for (int64_t v = v0; v < v1; ++v) {
            const char * row = host_src.data() + (size_t)v * row_size;
            if (st == GGML_TYPE_F32) {
                std::memcpy(tmp.data(), row, (size_t)n_embd * sizeof(float));
            } else {
                tr->to_float(row, tmp.data(), (int64_t)n_embd);
            }
            // Transpose: dstT[e * n_vocab + v] = tmp[e]  (row-major {n_vocab,n_embd})
            for (int64_t e = 0; e < n_embd; ++e) {
                dstT[(size_t)e * n_vocab + v] = ggml_fp32_to_fp16(tmp[e]);
            }
        }
    };
    std::vector<std::thread> pool;
    const int64_t chunk = (n_vocab + (int64_t)nth - 1) / (int64_t)nth;
    for (unsigned t = 0; t < nth; ++t) {
        const int64_t v0 = (int64_t)t * chunk;
        const int64_t v1 = std::min(v0 + chunk, n_vocab);
        if (v0 < v1) pool.emplace_back(worker, v0, v1);
    }
    for (auto & th : pool) th.join();

    ggml_backend_tensor_set(sc_embT_, dstT.data(), 0, dstT.size() * sizeof(ggml_fp16_t));
    return true;
}

bool DiffusionGemmaGraph::forward_block(const std::vector<int32_t> & canvas,
                                        int block_begin, int block_len,
                                        bool /*bidirectional*/,
                                        std::vector<float> & out_logits) {
    if (!loaded_) return false;
    const int n = (int)canvas.size();
    // block_begin is the first canvas position = number of prompt tokens (P).
    // block_len is C (canvas length for this block).
    if (block_begin < 0 || block_len <= 0 || block_begin + block_len > n) return false;

    const int P = block_begin;   // prompt token count (canvas = n_tokens - P)
    // The full sequence is canvas[0..P-1] (prompt) + canvas[P..P+block_len-1] (canvas).
    // n == P + block_len here; if canvas grew beyond one block (future) use n.

    const int hidden = w_.n_embd;
    embed_.resize((size_t)n * hidden);
    if (!w_.embedder.embed(canvas.data(), n, embed_.data())) {
        std::fprintf(stderr, "[diffusiongemma] embed failed (n=%d)\n", n);
        return false;
    }
    // Scale ALL input embeddings by sqrt(n_embd). Canvas rows will be re-normed
    // inside gemma4_denoise_batch (bare rms_norm replaces the scale for canvas).
    const float scale = std::sqrt((float)hidden);
    for (size_t i = 0; i < embed_.size(); ++i) embed_[i] *= scale;

    // SC state: build embT lazily, pass nullptr if SC is disabled this step.
    const float * sc_logits = sc_logits_ptr_;
    ggml_tensor * sc_embT   = nullptr;
    if (sc_logits && sc_use_ != 0.0f && w_.sc_pre_norm) {
        if (!ensure_sc_embT()) {
            std::fprintf(stderr, "[diffusiongemma] sc_embT build failed\n");
            // Non-fatal: fall through with SC disabled
        } else {
            sc_embT = sc_embT_;
        }
    }

    // Reset sc_logits_ptr_ after reading (one-shot per forward).
    sc_logits_ptr_ = nullptr;

    return gemma4_denoise_batch(backend_, w_, cache_, embed_.data(),
                                canvas.data(), n,
                                /*n_prompt=*/P,
                                sc_logits, sc_use_, sc_temp_inv_,
                                sc_embT, out_logits);
}

void DiffusionGemmaGraph::reset() {
    cache_.cur_pos = 0;
    prefix_len_    = 0;
    sc_logits_ptr_ = nullptr;
    sc_use_        = 0.0f;
    sc_temp_inv_   = 1.0f;
}

}  // namespace dflash::common
