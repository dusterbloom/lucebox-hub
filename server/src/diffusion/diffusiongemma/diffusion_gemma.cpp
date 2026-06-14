// DiffusionGemmaGraph implementation. See diffusion_gemma.h.

#include "diffusion_gemma.h"

#include "dflash27b.h"   // dflash27b_last_error
#include "ggml-cuda.h"   // ggml_backend_cuda_init

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <thread>
#include <vector>

#ifdef DFLASH27B_BACKEND_CUDA
#include <cuda_runtime.h>
#endif

namespace dflash::common {

DiffusionGemmaGraph::DiffusionGemmaGraph(const DiffusionGemmaConfig & cfg) : cfg_(cfg) {}

DiffusionGemmaGraph::~DiffusionGemmaGraph() {
    if (sc_embT_buf_) { ggml_backend_buffer_free(sc_embT_buf_); sc_embT_buf_ = nullptr; }
    if (sc_embT_ctx_) { ggml_free(sc_embT_ctx_);                sc_embT_ctx_ = nullptr; }
    sc_embT_ = nullptr;
#ifdef DFLASH27B_BACKEND_CUDA
    if (sc_dev_buf_a_) { ggml_backend_buffer_free(sc_dev_buf_a_); sc_dev_buf_a_ = nullptr; }
    if (sc_dev_ctx_a_) { ggml_free(sc_dev_ctx_a_);               sc_dev_ctx_a_ = nullptr; }
    sc_dev_ten_a_ = nullptr;
    if (sc_dev_buf_b_) { ggml_backend_buffer_free(sc_dev_buf_b_); sc_dev_buf_b_ = nullptr; }
    if (sc_dev_ctx_b_) { ggml_free(sc_dev_ctx_b_);               sc_dev_ctx_b_ = nullptr; }
    sc_dev_ten_b_ = nullptr;
    if (u_dev_buf_) { ggml_backend_buffer_free(u_dev_buf_); u_dev_buf_ = nullptr; }
    if (u_dev_ctx_) { ggml_free(u_dev_ctx_);               u_dev_ctx_ = nullptr; }
    u_dev_ten_ = nullptr;
#endif
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
    prefix_len_    = (int)prompt.size();
    out_prefix_len = prefix_len_;
    cache_.cur_pos = 0;
    prompt_cached_ = false;

    if (!loaded_) return false;

    // L0: prefix-KV cache.  Enabled unless DG_NO_L0_CACHE=1 is set.
    static const bool s_use_l0 = (std::getenv("DG_NO_L0_CACHE") == nullptr);
    if (!s_use_l0 || prefix_len_ == 0) return true;

    const int P = prefix_len_;
    const int hidden = w_.n_embd;

    // Embed the prompt tokens and scale by sqrt(n_embd) (same as forward_block).
    std::vector<float> prompt_embed((size_t)P * hidden);
    if (!w_.embedder.embed(prompt.data(), P, prompt_embed.data())) {
        std::fprintf(stderr, "[diffusiongemma] prepare: embed failed (P=%d)\n", P);
        return false;
    }
    const float scale = std::sqrt((float)hidden);
    for (float & v : prompt_embed) v *= scale;

    // Prefill prompt KV into the cache.
    if (!gemma4_prefill_prompt_for_denoise(backend_, w_, cache_,
                                            prompt_embed.data(),
                                            prompt.data(), P)) {
        std::fprintf(stderr, "[diffusiongemma] prepare: prompt prefill failed\n");
        return false;
    }

    prompt_cached_ = true;
    std::fprintf(stderr, "[diffusiongemma] L0: prompt KV cached (%d tokens)\n", P);
    std::fflush(stderr);
    return true;
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

    if (prompt_cached_) {
        return gemma4_denoise_canvas(backend_, w_, cache_, embed_.data(),
                                     canvas.data(), n,
                                     /*n_prompt=*/P,
                                     sc_logits, sc_use_, sc_temp_inv_,
                                     sc_embT, out_logits);
    }
    return gemma4_denoise_batch(backend_, w_, cache_, embed_.data(),
                                canvas.data(), n,
                                /*n_prompt=*/P,
                                sc_logits, sc_use_, sc_temp_inv_,
                                sc_embT, out_logits);
}

void DiffusionGemmaGraph::reset() {
    cache_.cur_pos = 0;
    prefix_len_    = 0;
    prompt_cached_ = false;
    sc_logits_ptr_ = nullptr;
    sc_use_        = 0.0f;
    sc_temp_inv_   = 1.0f;
#ifdef DFLASH27B_BACKEND_CUDA
    sc_dev_a_is_cur_ = true;  // reset double-buffer state
#endif
}

#ifdef DFLASH27B_BACKEND_CUDA
// GPU-accelerated forward + sample. Overrides the default CPU fallback in
// DiffusionModelGraph. Keeps logits device-resident; copies only ~3 KB to host.
bool DiffusionGemmaGraph::forward_block_dev(
    const std::vector<int32_t> & canvas,
    int block_begin, int block_len,
    bool /*bidirectional*/,
    const std::vector<float> & u_host,
    float temp_inv,
    DevSampleResult & out)
{
    if (!loaded_) return false;
    const int n = (int)canvas.size();
    if (block_begin < 0 || block_len <= 0 || block_begin + block_len > n) return false;

    const int P = block_begin;
    const int C = block_len;

    // ── Embed (same as forward_block) ────────────────────────────────
    const int hidden = w_.n_embd;
    embed_.resize((size_t)n * hidden);
    if (!w_.embedder.embed(canvas.data(), n, embed_.data())) {
        std::fprintf(stderr, "[diffusiongemma dev] embed failed (n=%d)\n", n);
        return false;
    }
    const float scale = std::sqrt((float)hidden);
    for (size_t i = 0; i < embed_.size(); ++i) embed_[i] *= scale;

    // ── Build sc_embT lazily ──────────────────────────────────────────
    ggml_tensor * sc_embT = nullptr;
    if (sc_use_ != 0.0f && w_.sc_pre_norm) {
        if (!ensure_sc_embT()) {
            std::fprintf(stderr, "[diffusiongemma dev] sc_embT build failed\n");
        } else {
            sc_embT = sc_embT_;
        }
    }

    // ── Ensure device SC buffers allocated (via ggml, not raw cudaMalloc) ───
    // Using ggml alloc keeps these in the same VMM pool as weights/KV cache,
    // preventing address-space conflicts on sm_86 RTX 3090.
    ggml_backend_buffer_type_t buft = ggml_backend_get_default_buffer_type(backend_);
    const int64_t sc_n_elem = (int64_t)w_.n_vocab * C;

    auto alloc_dev_tensor = [&](ggml_context *& ctx, ggml_backend_buffer_t & buf,
                                 ggml_tensor *& ten, int64_t n_elem, const char * label) -> bool {
        if (buf) { ggml_backend_buffer_free(buf); buf = nullptr; }
        if (ctx) { ggml_free(ctx); ctx = nullptr; }
        ten = nullptr;
        ggml_init_params ip{ ggml_tensor_overhead() * 2, nullptr, /*no_alloc=*/true };
        ctx = ggml_init(ip);
        if (!ctx) {
            std::fprintf(stderr, "[diffusiongemma dev] ggml_init failed for %s\n", label);
            return false;
        }
        ten = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, n_elem);
        if (!ten) {
            ggml_free(ctx); ctx = nullptr;
            std::fprintf(stderr, "[diffusiongemma dev] tensor alloc failed for %s\n", label);
            return false;
        }
        buf = ggml_backend_alloc_ctx_tensors_from_buft(ctx, buft);
        if (!buf) {
            ggml_free(ctx); ctx = nullptr; ten = nullptr;
            std::fprintf(stderr, "[diffusiongemma dev] buf alloc failed for %s\n", label);
            return false;
        }
        return true;
    };

    if (sc_dev_C_ != C) {
        if (!alloc_dev_tensor(sc_dev_ctx_a_, sc_dev_buf_a_, sc_dev_ten_a_, sc_n_elem, "sc_a") ||
            !alloc_dev_tensor(sc_dev_ctx_b_, sc_dev_buf_b_, sc_dev_ten_b_, sc_n_elem, "sc_b")) {
            std::fprintf(stderr, "[diffusiongemma dev] SC device buffer alloc failed (C=%d)\n", C);
            return false;
        }
        sc_dev_C_        = C;
        sc_dev_a_is_cur_ = true;
    }
    if (u_dev_C_ != C) {
        if (!alloc_dev_tensor(u_dev_ctx_, u_dev_buf_, u_dev_ten_, (int64_t)C, "u_dev")) {
            std::fprintf(stderr, "[diffusiongemma dev] u_dev alloc failed (C=%d)\n", C);
            return false;
        }
        u_dev_C_ = C;
    }

    // ── Upload per-step uniform randoms ──────────────────────────────
    ggml_backend_tensor_set(u_dev_ten_, u_host.data(), 0, (size_t)C * sizeof(float));

    // ── Determine which buffer holds prev-step SC (input) / this step (output) ─
    float * sc_in  = (sc_use_ == 0.0f) ? nullptr
                   : (sc_dev_a_is_cur_ ? (float*)sc_dev_ten_a_->data : (float*)sc_dev_ten_b_->data);
    float * sc_out = sc_dev_a_is_cur_ ? (float*)sc_dev_ten_b_->data : (float*)sc_dev_ten_a_->data;
    // Flip for next step.
    sc_dev_a_is_cur_ = !sc_dev_a_is_cur_;

    // ── GPU-mode forward ──────────────────────────────────────────────
    DenoiseBatchGpuMode mode;
    mode.sc_dev_in   = sc_in;
    mode.sc_dev_out  = sc_out;
    mode.u_dev       = (const float*)u_dev_ten_->data;
    mode.temp_inv    = temp_inv;
    mode.out_sampled = &out.sampled;
    mode.out_entropy = &out.entropy;
    mode.out_argmax  = &out.argmax;

    // sc_logits_ptr_ is not needed in GPU mode (SC is device-resident).
    sc_logits_ptr_ = nullptr;

    // ── Per-step timing (reported every step for measurement) ─────────
    using clock_t = std::chrono::steady_clock;
    const auto t0 = clock_t::now();

    std::vector<float> dummy_logits;  // not populated in GPU mode
    bool ok;
    if (prompt_cached_) {
        ok = gemma4_denoise_canvas(backend_, w_, cache_, embed_.data(),
                                   canvas.data(), n,
                                   /*n_prompt=*/P,
                                   /*sc_logits=*/nullptr,
                                   sc_use_, sc_temp_inv_,
                                   sc_embT, dummy_logits,
                                   &mode);
    } else {
        ok = gemma4_denoise_batch(backend_, w_, cache_, embed_.data(),
                                  canvas.data(), n,
                                  /*n_prompt=*/P,
                                  /*sc_logits=*/nullptr,
                                  sc_use_, sc_temp_inv_,
                                  sc_embT, dummy_logits,
                                  &mode);
    }

    const auto t1 = clock_t::now();
    const double ms_step = std::chrono::duration<double, std::milli>(t1 - t0).count();
    // decode_tps: C tokens per (ms_step/1000) seconds
    const double tps = (ms_step > 0) ? (C * 1000.0 / ms_step) : 0.0;
    std::fprintf(stderr, "[dg-timing] step ms=%.1f  tok/s=%.1f  C=%d  sc=%s\n",
                 ms_step, tps, C, sc_in ? "on" : "off");
    std::fflush(stderr);

    // out.logits intentionally left empty (logits stay device-resident).
    return ok;
}
#endif  // DFLASH27B_BACKEND_CUDA

}  // namespace dflash::common
