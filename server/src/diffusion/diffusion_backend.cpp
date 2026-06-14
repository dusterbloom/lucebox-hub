// DiffusionBackend implementation. See diffusion_backend.h.

#include "diffusion_backend.h"
#include "diffusion_decoder.h"
#include "diffusiongemma/diffusion_gemma.h"
#include "common/snapshot_backend.h"

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <utility>

namespace dflash::common {

namespace {

DiffusionGemmaGraph * as_diffusion_gemma(DiffusionModelGraph * model) {
    return dynamic_cast<DiffusionGemmaGraph *>(model);
}

DiffusionStream make_stream(const DaemonIO & io) {
    DiffusionStream stream;
    stream.on_token = [&io](int32_t tok) -> bool {
        io.emit(tok);
        return !io.cancelled;
    };
    return stream;
}

void copy_snapshot_to_cache(const Gemma4Snapshot & snap, Gemma4Cache & cache) {
    for (int il = 0; il < cache.n_layer; ++il) {
        if (!cache.k[il] || !cache.v[il] ||
            il >= (int)snap.k_snap.size() || !snap.k_snap[il] || !snap.v_snap[il]) {
            continue;
        }

        ggml_tensor * ck = cache.k[il];
        const int D          = (int)ck->ne[0];
        const int Hk         = (int)ck->ne[2];
        const int cache_len  = (int)ck->ne[1];
        const int save_pos   = (int)snap.k_snap[il]->ne[1];
        const size_t elem_sz = ggml_element_size(ck);
        const size_t head_bytes_src = (size_t)D * save_pos * elem_sz;
        const size_t head_bytes_dst = (size_t)D * cache_len * elem_sz;

        for (int h = 0; h < Hk; ++h) {
            ggml_backend_tensor_set(cache.k[il],
                (const char *)snap.k_snap[il]->data + h * head_bytes_src,
                h * head_bytes_dst, head_bytes_src);
            ggml_backend_tensor_set(cache.v[il],
                (const char *)snap.v_snap[il]->data + h * head_bytes_src,
                h * head_bytes_dst, head_bytes_src);
        }
    }
}

}  // namespace

DiffusionBackend::DiffusionBackend(std::unique_ptr<DiffusionModelGraph> model,
                                   DiffusionConfig cfg,
                                   std::string arch_label)
    : model_(std::move(model)), cfg_(cfg), arch_label_(std::move(arch_label)) {
    if (auto * dg = as_diffusion_gemma(model_.get())) {
        snap_compute_backend_ = dg->backend_for_snapshot();
        if (snap_compute_backend_) {
            snap_backend_ = create_snapshot_backend(snap_compute_backend_);
            if (!snap_backend_) {
                std::fprintf(stderr, "[diffusion] snapshot backend init failed\n");
            }
        }
    }
}

DiffusionBackend::~DiffusionBackend() { shutdown(); }

void DiffusionBackend::print_ready_banner() const {
    const char * scheme = (cfg_.noise_scheme == DiffusionNoise::UniformState)
                              ? "uniform-state" : "masked";
    std::printf("[%s-diffusion-daemon] ready (block=%d steps=%d noise=%s)\n",
                arch_label_.c_str(), cfg_.block_size,
                cfg_.n_steps > 0 ? cfg_.n_steps : cfg_.block_size, scheme);
}

bool DiffusionBackend::park(const std::string & /*what*/) { return true; }
bool DiffusionBackend::unpark(const std::string & /*what*/) { return true; }

GenerateResult DiffusionBackend::generate_impl(const GenerateRequest & req,
                                               const DaemonIO & io) {
    GenerateResult r;
    if (!model_) { r.error = "no_model"; return r; }

    DaemonIO out_io = io.with_token_callback(req.on_token);
    DiffusionStream stream = make_stream(out_io);

    using clock = std::chrono::steady_clock;
    const auto t_prefill0 = clock::now();

    int prefix_len = 0;
    bool prepared = false;

    if (req.snap_slot >= 0 && req.snap_pos > 0 &&
        req.snap_pos < (int)req.prompt.size()) {
        if (auto * dg = as_diffusion_gemma(model_.get())) {
            std::vector<int32_t> prefix(req.prompt.begin(),
                                        req.prompt.begin() + req.snap_pos);
            int snap_prefix_len = 0;
            if (model_->prepare(prefix, snap_prefix_len) &&
                snap_prefix_len == req.snap_pos &&
                snapshot_save(req.snap_slot) &&
                dg->prepare_delta_from_cache(req.prompt, req.snap_pos, prefix_len)) {
                std::fprintf(stderr,
                    "[diffusion] inline-snap slot=%d cur_pos=%d\n",
                    req.snap_slot, req.snap_pos);
                prepared = true;
            }
        }
    }

    if (!prepared) {
        if (!model_->prepare(req.prompt, prefix_len)) {
            r.error = "prepare";
            return r;
        }
        if (prefix_len < 0 || prefix_len > (int)req.prompt.size()) {
            prefix_len = (int)req.prompt.size();
        }

        if (req.snap_slot >= 0 && req.snap_pos > 0) {
            if (auto * dg = as_diffusion_gemma(model_.get())) {
                Gemma4Cache & cache = dg->cache_for_snapshot();
                const int saved_pos = cache.cur_pos;
                if (req.snap_pos <= saved_pos) {
                    cache.cur_pos = req.snap_pos;
                    if (snapshot_save(req.snap_slot)) {
                        std::fprintf(stderr,
                            "[diffusion] inline-snap slot=%d cur_pos=%d\n",
                            req.snap_slot, req.snap_pos);
                    }
                    cache.cur_pos = saved_pos;
                }
            }
        }
    }

    r.prefill_s = std::chrono::duration<double>(clock::now() - t_prefill0).count();

    const auto t_decode0 = clock::now();
    DiffusionDecodeResult d = run_diffusion_generate(
        *model_, req.prompt, req.n_gen, cfg_, req.sampler, req.do_sample,
        stream, prefix_len);
    const auto t_decode1 = clock::now();

    r.ok        = d.ok;
    r.error     = d.error;
    r.tokens    = std::move(d.tokens);
    r.decode_s  = std::chrono::duration<double>(t_decode1 - t_decode0).count();
    return r;
}

// ── Prefix snapshots ────────────────────────────────────────────────────
bool DiffusionBackend::snapshot_save(int slot) {
    if (slot < 0 || slot >= PREFIX_SLOTS) return false;
    auto * dg = as_diffusion_gemma(model_.get());
    if (!dg) return false;

    Gemma4Cache & cache = dg->cache_for_snapshot();
    if (!snap_backend_) {
        snap_compute_backend_ = dg->backend_for_snapshot();
        if (!snap_compute_backend_) return false;
        snap_backend_ = create_snapshot_backend(snap_compute_backend_);
        if (!snap_backend_) return false;
    }

    auto & snap = snapshots_[slot];
    const int n_layer = cache.n_layer;
    const int snap_pos = cache.cur_pos;
    if (snap_pos <= 0) return false;

    const bool needs_alloc = (snap.ctx == nullptr) || (snap.cur_pos != snap_pos);
    if (needs_alloc) {
        free_gemma4_snapshot(snap);

        ggml_init_params ip{};
        ip.mem_size = ggml_tensor_overhead() * (size_t)(n_layer * 2 + 4) + 4096;
        ip.no_alloc = true;
        snap.ctx = ggml_init(ip);
        if (!snap.ctx) return false;

        snap.k_snap.resize(n_layer, nullptr);
        snap.v_snap.resize(n_layer, nullptr);
        for (int il = 0; il < n_layer; ++il) {
            if (cache.k[il] && cache.v[il]) {
                ggml_tensor * ck = cache.k[il];
                const int cache_len = (int)ck->ne[1];
                const int save_pos = std::min(snap_pos, cache_len);
                snap.k_snap[il] = ggml_new_tensor_3d(snap.ctx, ck->type,
                                                      ck->ne[0], save_pos, ck->ne[2]);
                snap.v_snap[il] = ggml_new_tensor_3d(snap.ctx, ck->type,
                                                      ck->ne[0], save_pos, ck->ne[2]);
            }
        }
        snap.feat_snap = nullptr;
        snap.feat_cap  = 0;

        snap.buf = ggml_backend_alloc_ctx_tensors(snap.ctx, snap_backend_);
        if (!snap.buf) {
            ggml_free(snap.ctx); snap.ctx = nullptr;
            snap.k_snap.clear(); snap.v_snap.clear();
            snap.feat_snap = nullptr;
            return false;
        }
    }

    for (int il = 0; il < n_layer; ++il) {
        if (!cache.k[il] || !cache.v[il] || !snap.k_snap[il] || !snap.v_snap[il]) {
            continue;
        }

        ggml_tensor * ck = cache.k[il];
        const int D          = (int)ck->ne[0];
        const int Hk         = (int)ck->ne[2];
        const int cache_len  = (int)ck->ne[1];
        const int save_pos   = std::min(snap_pos, cache_len);
        const size_t elem_sz = ggml_element_size(ck);
        const size_t head_bytes_src = (size_t)D * cache_len * elem_sz;
        const size_t head_bytes_dst = (size_t)D * save_pos * elem_sz;

        for (int h = 0; h < Hk; ++h) {
            ggml_backend_tensor_get(cache.k[il],
                (char *)snap.k_snap[il]->data + h * head_bytes_dst,
                h * head_bytes_src, head_bytes_dst);
            ggml_backend_tensor_get(cache.v[il],
                (char *)snap.v_snap[il]->data + h * head_bytes_dst,
                h * head_bytes_src, head_bytes_dst);
        }
    }
    snap.cur_pos = snap_pos;
    snap.last_tok = cache.last_tok;
    std::fprintf(stderr, "[diffusion] snapshot saved slot=%d pos=%d\n",
                 slot, snap.cur_pos);
    return true;
}

void DiffusionBackend::snapshot_free(int slot) {
    if (slot < 0 || slot >= PREFIX_SLOTS) return;
    free_gemma4_snapshot(snapshots_[slot]);
}

bool DiffusionBackend::snapshot_used(int slot) const {
    return slot >= 0 && slot < PREFIX_SLOTS && snapshots_[slot].ctx != nullptr;
}

int DiffusionBackend::snapshot_cur_pos(int slot) const {
    if (slot < 0 || slot >= PREFIX_SLOTS || !snapshots_[slot].ctx) return 0;
    return snapshots_[slot].cur_pos;
}

GenerateResult DiffusionBackend::restore_and_generate_impl(
        int slot, const GenerateRequest & req, const DaemonIO & io) {
    GenerateResult r;
    if (!model_) { r.error = "no_model"; return r; }

    DaemonIO out_io = io.with_token_callback(req.on_token);
    if (slot < 0 || slot >= PREFIX_SLOTS || !snapshots_[slot].ctx) {
        r.error = "bad slot";
        out_io.emit(-1);
        return r;
    }

    auto * dg = as_diffusion_gemma(model_.get());
    if (!dg) {
        r.error = "restore_unsupported";
        out_io.emit(-1);
        return r;
    }

    const Gemma4Snapshot & snap = snapshots_[slot];
    Gemma4Cache & cache = dg->cache_for_snapshot();
    copy_snapshot_to_cache(snap, cache);
    cache.cur_pos = snap.cur_pos;
    cache.last_tok = snap.last_tok;
    dg->mark_prompt_cache_restored(snap.cur_pos);

    const int prompt_len = (int)req.prompt.size();
    if (prompt_len < snap.cur_pos) {
        std::fprintf(stderr,
            "[diffusion] snapshot longer than prompt (snap=%d > prompt=%d) - "
            "fresh prefill fallback\n", snap.cur_pos, prompt_len);
        model_->reset();
        return generate_impl(req, io);
    }

    using clock = std::chrono::steady_clock;
    const auto t_prefill0 = clock::now();

    int prefix_len = snap.cur_pos;
    if (!dg->prepare_delta_from_cache(req.prompt, snap.cur_pos, prefix_len)) {
        r.error = "prefill";
        return r;
    }

    if (req.snap_slot >= 0 && req.snap_pos > 0) {
        const int saved_pos = cache.cur_pos;
        if (req.snap_pos <= saved_pos) {
            cache.cur_pos = req.snap_pos;
            if (snapshot_save(req.snap_slot)) {
                std::fprintf(stderr,
                    "[diffusion] inline-snap slot=%d cur_pos=%d\n",
                    req.snap_slot, req.snap_pos);
            }
            cache.cur_pos = saved_pos;
        }
    }

    r.prefill_s = std::chrono::duration<double>(clock::now() - t_prefill0).count();

    DiffusionStream stream = make_stream(out_io);
    const auto t_decode0 = clock::now();
    DiffusionDecodeResult d = run_diffusion_generate(
        *model_, req.prompt, req.n_gen, cfg_, req.sampler, req.do_sample,
        stream, prefix_len);
    const auto t_decode1 = clock::now();

    r.ok        = d.ok;
    r.error     = d.error;
    r.tokens    = std::move(d.tokens);
    r.decode_s  = std::chrono::duration<double>(t_decode1 - t_decode0).count();
    return r;
}

bool DiffusionBackend::handle_compress(const std::string & /*line*/,
                                       const DaemonIO & io) {
    std::fprintf(stderr, "[diffusion] pflash compress is not supported\n");
    io.emit(-1);
    return false;
}

void DiffusionBackend::shutdown() {
    for (int i = 0; i < PREFIX_SLOTS; ++i) snapshot_free(i);
    free_snapshot_backend(snap_backend_, snap_compute_backend_);
    snap_backend_ = nullptr;
    snap_compute_backend_ = nullptr;
    if (model_) model_->reset();
    model_.reset();
}

}  // namespace dflash::common
