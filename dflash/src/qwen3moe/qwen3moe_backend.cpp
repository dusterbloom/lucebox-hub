// Qwen3MoeBackend implementation — Phase A.3 (runtime: prefill/decode/generate/snapshots).
//
// Ported verbatim from dflash/src/qwen3/qwen3_backend.cpp with type-substitution:
//   Qwen3DrafterWeights  → Qwen3MoeWeights
//   Qwen3Cache           → Qwen3MoeCache
//   Qwen3Snapshot        → Qwen3MoeSnapshot
//   free_qwen3_drafter_model → free_qwen3moe_weights
//
// do_step() is implemented in qwen3moe_graph.cpp (MoE graph builder).
// park/unpark reload the GGUF, matching Qwen3Backend exactly.

#include "qwen3moe_backend.h"
#include "common/sampler.h"
#include "common/dflash_draft_graph.h"   // build_draft_step
#include "dflash27b.h"                   // dflash27b_last_error

#include "ggml-cuda.h"
#include "ggml-alloc.h"

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cmath>
#include <utility>
#include <vector>

namespace dflash::common {

// ── Construction / destruction ─────────────────────────────────────────────

Qwen3MoeBackend::Qwen3MoeBackend(const Qwen3MoeBackendConfig & cfg) : cfg_(cfg) {}

Qwen3MoeBackend::~Qwen3MoeBackend() { shutdown(); }

// ── init ───────────────────────────────────────────────────────────────────

bool Qwen3MoeBackend::init() {
    backend_ = ggml_backend_cuda_init(cfg_.device.gpu);
    if (!backend_) {
        std::fprintf(stderr, "[qwen3moe] CUDA init failed for GPU %d\n",
                     cfg_.device.gpu);
        return false;
    }

    if (!load_qwen3moe_gguf(cfg_.model_path ? cfg_.model_path : "",
                            backend_, w_)) {
        std::fprintf(stderr, "[qwen3moe] model load failed\n");
        return false;
    }
    std::printf("[qwen3moe] loaded %s (%d layers, hidden=%d, experts=%d/%d, vocab=%d)\n",
                cfg_.model_path, w_.n_layer, w_.n_embd,
                w_.n_expert_used, w_.n_expert, w_.n_vocab);

    if (!create_qwen3moe_cache(backend_, w_, cfg_.device.max_ctx, cache_)) {
        std::fprintf(stderr, "[qwen3moe] cache creation failed\n");
        return false;
    }
    std::printf("[qwen3moe] cache allocated (max_ctx=%d)\n", cfg_.device.max_ctx);

    if (!w_.tok_embd) {
        std::fprintf(stderr, "[qwen3moe] no token embedding tensor\n");
        return false;
    }

    // ── Drafter (DFlash spec-decode) ─────────────────────────────────────────
    if (cfg_.draft_path) {
        // GPU split: use a separate CUDA backend if draft_gpu differs from target
        split_gpus_ = (cfg_.draft_gpu >= 0 && cfg_.draft_gpu != cfg_.device.gpu);
        draft_backend_ = backend_;  // default: share target's CUDA backend
        if (split_gpus_) {
            draft_backend_ = ggml_backend_cuda_init(cfg_.draft_gpu);
            if (!draft_backend_) {
                std::fprintf(stderr, "[qwen3moe] draft CUDA init failed (gpu=%d)\n",
                             cfg_.draft_gpu);
                return false;
            }
        }

        // Load DFlash draft GGUF. Pass nullptr for the target parameter since
        // Qwen3MoeWeights is not TargetWeights; mask_token_id comes from the GGUF.
        if (!load_draft_gguf(cfg_.draft_path, draft_backend_, dw_, nullptr)) {
            std::fprintf(stderr, "[qwen3moe] draft load failed: %s\n",
                         dflash27b_last_error());
            return false;
        }
        // Override mask_token_id: load_draft_gguf uses a hardcoded 27B default
        // (248070) which is out-of-bounds for Qwen3's 151936-token vocab.
        // The correct mask for Qwen3-Coder-30B-A3B-DFlash is 151669 (matches
        // the dflash_config in the source HF repo).
        if (dw_.mask_token_id < 0 || dw_.mask_token_id >= w_.n_vocab) {
            dw_.mask_token_id = 151669;
        }
        std::printf("[qwen3moe] draft loaded: %d layers, block_size=%d, "
                    "n_target_layers=%d, mask_token=%d\n",
                    dw_.n_layer, dw_.block_size, dw_.n_target_layers,
                    dw_.mask_token_id);

        // Feature mirror (cross-GPU or F32-conversion buffer for spec decode)
        const int mirror_cap = std::min(cfg_.draft_ctx_max, cfg_.device.max_ctx);
        const int draft_gpu  = (cfg_.draft_gpu >= 0) ? cfg_.draft_gpu : cfg_.device.gpu;
        if (!draft_feature_mirror_init(feature_mirror_, draft_backend_,
                                       draft_gpu, cfg_.device.gpu,
                                       mirror_cap, dw_.n_target_layers,
                                       w_.n_embd)) {
            std::fprintf(stderr,
                         "[qwen3moe] feature mirror init failed (cap=%d, "
                         "n_target=%d, hidden=%d) — spec decode unavailable\n",
                         mirror_cap, dw_.n_target_layers, w_.n_embd);
            // Non-fatal — AR decode still works.
        } else {
            std::printf("[qwen3moe] feature mirror init: cap=%d, n_target=%d, hidden=%d\n",
                        mirror_cap, dw_.n_target_layers, w_.n_embd);
        }
    }

    std::fflush(stdout);
    return true;
}

// ── Ready banner ───────────────────────────────────────────────────────────

void Qwen3MoeBackend::print_ready_banner() const {
    std::printf("[qwen3moe-daemon] ready (layers=%d hidden=%d vocab=%d max_ctx=%d)\n",
                w_.n_layer, w_.n_embd, w_.n_vocab, cfg_.device.max_ctx);
    std::fflush(stdout);
}

// ── Park / unpark ──────────────────────────────────────────────────────────

bool Qwen3MoeBackend::park(const std::string & what) {
    if (what == "target" || what == "all") {
        if (!parked_) {
            if (w_.buf) {
                ggml_backend_buffer_free(w_.buf);
                w_.buf = nullptr;
            }
            parked_ = true;
            std::printf("[qwen3moe] target parked\n");
            std::fflush(stdout);
        }
        return true;
    }
    return false;
}

bool Qwen3MoeBackend::unpark(const std::string & what) {
    if (what == "target" || what == "all") {
        if (parked_) {
            Qwen3MoeWeights w_new;
            if (!load_qwen3moe_gguf(cfg_.model_path ? cfg_.model_path : "",
                                    backend_, w_new)) {
                std::fprintf(stderr, "[qwen3moe] unpark reload failed\n");
                return false;
            }
            w_ = w_new;
            parked_ = false;
            std::printf("[qwen3moe] target unparked\n");
            std::fflush(stdout);
        }
        return true;
    }
    return false;
}

// ── do_step is in qwen3moe_graph.cpp ──────────────────────────────────────

// ── Embed helper ───────────────────────────────────────────────────────────
// Build a small get_rows graph, run it, copy result to embed_buf.

static bool embed_tokens(ggml_backend_t backend,
                          ggml_tensor *  tok_embd,
                          const int32_t * ids,
                          int             n,
                          int             hidden,
                          float *         embed_buf) {
    ggml_init_params ip{};
    ip.mem_size = ggml_tensor_overhead() * 8 + ggml_graph_overhead() + 16 * 1024;
    ip.no_alloc = true;
    ggml_context * ectx = ggml_init(ip);
    if (!ectx) return false;

    ggml_tensor * id_t = ggml_new_tensor_1d(ectx, GGML_TYPE_I32, n);
    ggml_set_input(id_t);
    ggml_tensor * emb  = ggml_get_rows(ectx, tok_embd, id_t);
    ggml_tensor * out  = ggml_new_tensor_2d(ectx, GGML_TYPE_F32, hidden, n);
    ggml_tensor * cpy  = ggml_cpy(ectx, emb, out);
    ggml_set_output(cpy);
    ggml_cgraph * gf = ggml_new_graph(ectx);
    ggml_build_forward_expand(gf, cpy);

    // Static gallocr — reused across calls; saves cudaMalloc/cudaFree per embed.
    static ggml_gallocr_t galloc = nullptr;
    if (!galloc) galloc = ggml_gallocr_new(
        ggml_backend_get_default_buffer_type(backend));
    if (!ggml_gallocr_alloc_graph(galloc, gf)) {
        ggml_free(ectx);
        return false;
    }
    ggml_backend_tensor_set(id_t, ids, 0, sizeof(int32_t) * n);
    ggml_backend_graph_compute(backend, gf);
    ggml_backend_tensor_get(cpy, embed_buf, 0, sizeof(float) * (size_t)hidden * n);
    ggml_free(ectx);
    return true;
}

// ── Prefill ────────────────────────────────────────────────────────────────

int Qwen3MoeBackend::do_prefill(const std::vector<int32_t> & tokens,
                                 const DaemonIO &             io,
                                 int                          kv_offset) {
    const int hidden = w_.n_embd;
    const int total  = (int)tokens.size();
    const int chunk  = std::max(1, cfg_.chunk);
    int committed    = 0;

    std::vector<float> embed_buf((size_t)chunk * hidden);

    for (int start = 0; start < total; start += chunk) {
        const int n = std::min(chunk, total - start);

        if (!embed_tokens(backend_, w_.tok_embd,
                          tokens.data() + start, n, hidden, embed_buf.data())) {
            return -1;
        }

        std::vector<float> logits;
        if (!do_step(embed_buf.data(), n, kv_offset + start, logits)) {
            return -1;
        }
        committed = kv_offset + start + n;
        cache_.cur_pos = committed;
        last_logits_ = std::move(logits);
    }

    return committed;
}

// ── Decode ─────────────────────────────────────────────────────────────────

bool Qwen3MoeBackend::do_decode(int                    committed,
                                 int                    n_gen,
                                 std::vector<int32_t> & out_tokens,
                                 const DaemonIO &       io) {
    const int hidden = w_.n_embd;
    const int vocab  = w_.n_vocab;
    std::vector<float> logits;
    std::vector<float> embed_buf(hidden);

    for (int i = 0; i < n_gen; ++i) {
        // First iteration uses prefill logits (already in last_logits_)
        if (i == 0) {
            if (last_logits_.empty()) return false;
            logits = std::move(last_logits_);
        }

        // Sample next token
        int32_t next;
        if (sampler_.needs_logit_processing()) {
            next = sample_logits(logits.data(), vocab, sampler_,
                                 out_tokens, sampler_rng_);
        } else {
            next = 0;
            float best = logits[0];
            for (int j = 1; j < vocab; ++j) {
                if (logits[j] > best) { best = logits[j]; next = j; }
            }
        }

        out_tokens.push_back(next);
        io.emit(next);
        committed++;
        cache_.cur_pos = committed;
        if (io.cancelled) break;

        // EOS check (Qwen tokenizer)
        if (next == 151643 || next == 151645) break;

        // Last iteration — don't need logits for another step
        if (i == n_gen - 1) break;

        // Embed and step to get logits for next iteration
        if (!embed_tokens(backend_, w_.tok_embd, &next, 1, hidden, embed_buf.data())) {
            return false;
        }
        if (!do_step(embed_buf.data(), 1, committed, logits)) {
            return false;
        }
    }

    return true;
}

// ── Generate ───────────────────────────────────────────────────────────────

GenerateResult Qwen3MoeBackend::generate(const GenerateRequest & req,
                                          const DaemonIO &        io) {
    GenerateResult result;
    DaemonIO out_io = io.with_token_callback(req.on_token);
    sampler_ = req.sampler;
    if (req.do_sample && sampler_.seed != 0) {
        sampler_rng_.seed(sampler_.seed);
    }

    cache_.cur_pos = 0;

    // Prefill
    const int committed = do_prefill(req.prompt, out_io);
    if (committed < 0) {
        result.error = "prefill";
        return result;
    }

    // BUG3 FIX: populate feature ring by re-running the target forward with
    // capture for all prefill positions.  do_prefill calls do_step() which has
    // no capture mechanism, so the feature ring was still all-zeros — draft
    // predictions were noise, causing ~7% acceptance.  verify_batch writes KV
    // at the same positions (idempotent) and populates feature_mirror_.
    if (supports_dflash_spec_decode() && !sampler_.needs_logit_processing()) {
        DFlashTarget * tgt = dflash_target();
        if (tgt) {
            const int saved_pos = cache_.cur_pos;
            const int prefill_chunk = std::max(1,
                cfg_.chunk > 0 ? cfg_.chunk : 512);
            const int prompt_len = (int)req.prompt.size();
            bool capture_ok = true;
            for (int start = 0; start < prompt_len && capture_ok; start += prefill_chunk) {
                const int n = std::min(prefill_chunk, prompt_len - start);
                std::vector<int32_t> chunk(req.prompt.begin() + start,
                                           req.prompt.begin() + start + n);
                int dummy_last = 0;
                if (!tgt->verify_batch(chunk, /*base_pos=*/start,
                                       dummy_last, /*all_argmax=*/nullptr)) {
                    std::fprintf(stderr,
                        "[qwen3moe] prefill feature capture failed "
                        "(start=%d n=%d) — spec decode may degrade\n",
                        start, n);
                    capture_ok = false;
                }
            }
            cache_.cur_pos = saved_pos;
        }
    }

    // Inline snapshot
    if (req.snap_slot >= 0 && req.snap_pos > 0 && req.snap_pos <= committed) {
        cache_.cur_pos = req.snap_pos;
        if (snapshot_save(req.snap_slot)) {
            std::printf("[snap] inline slot=%d cur_pos=%d\n",
                        req.snap_slot, req.snap_pos);
            std::fflush(stdout);
        }
        cache_.cur_pos = committed;
    }

    if (req.n_gen <= 0) {
        out_io.emit(-1);
        result.ok = true;
        return result;
    }

    // Get logits for first generated token: re-step at committed-1 position
    const int hidden = w_.n_embd;
    const int vocab  = w_.n_vocab;
    std::vector<float> logits;
    std::vector<float> embed_buf(hidden);

    int32_t last_tok = req.prompt.back();
    if (!embed_tokens(backend_, w_.tok_embd, &last_tok, 1, hidden, embed_buf.data())) {
        result.error = "embed alloc";
        return result;
    }
    if (!do_step(embed_buf.data(), 1, committed - 1, logits)) {
        result.error = "first logits";
        return result;
    }

    // Compute first-token argmax (used by both paths below).
    int32_t first;
    if (sampler_.needs_logit_processing()) {
        first = sample_logits(logits.data(), vocab, sampler_,
                              result.tokens, sampler_rng_);
    } else {
        first = 0;
        float best = logits[0];
        for (int j = 1; j < vocab; ++j) {
            if (logits[j] > best) { best = logits[j]; first = j; }
        }
    }

    // BUG2 FIX: for spec decode, do NOT pre-emit `first`; let do_spec_decode
    // emit it as replay_tok[0] (mirrors qwen35_backend behavior).  For the AR
    // path, emit it here as before.
    int cur_committed = committed;
    if (req.n_gen > 0 && supports_dflash_spec_decode() && !sampler_.needs_logit_processing()) {
        // Spec-decode path: seed with `first`, emit all n_gen tokens inside loop.
        last_prefill_tok_ = first;
        if (!do_spec_decode(cur_committed, req.n_gen, result.tokens, out_io,
                            req.hint_tokens)) {
            result.error = "spec decode";
            out_io.emit(-1);
            return result;
        }
    } else {
        // AR path: emit first token, then continue AR.
        result.tokens.push_back(first);
        out_io.emit(first);

        if (out_io.cancelled) {
            out_io.emit(-1);
            result.ok = true;
            return result;
        }
        if (first == 151643 || first == 151645) {
            out_io.emit(-1);
            result.ok = true;
            return result;
        }

        if (req.n_gen > 1) {
            // AR fallback: step `first` to get next logits, then AR loop.
            if (!embed_tokens(backend_, w_.tok_embd, &first, 1, hidden, embed_buf.data())) {
                result.error = "embed2 alloc";
                return result;
            }
            if (!do_step(embed_buf.data(), 1, cur_committed, last_logits_)) {
                result.error = "decode logits";
                return result;
            }
            cur_committed++;
            cache_.cur_pos = cur_committed;

            if (!do_decode(cur_committed, req.n_gen - 1, result.tokens, out_io)) {
                result.error = "decode";
                return result;
            }
        }
    }

    out_io.emit(-1);
    result.ok = true;
    return result;
}

// ── Restore + generate ─────────────────────────────────────────────────────

GenerateResult Qwen3MoeBackend::restore_and_generate(int                     slot,
                                                      const GenerateRequest & req,
                                                      const DaemonIO &        io) {
    GenerateResult result;
    DaemonIO out_io = io.with_token_callback(req.on_token);

    if (slot < 0 || slot >= PREFIX_SLOTS || !snapshots_[slot].ctx) {
        result.error = "bad slot";
        out_io.emit(-1);
        return result;
    }

    // Restore KV cache from snapshot
    const auto & snap = snapshots_[slot];
    for (int il = 0; il < cache_.n_layer; ++il) {
        ggml_backend_tensor_copy(snap.k_snap[il], cache_.k[il]);
        ggml_backend_tensor_copy(snap.v_snap[il], cache_.v[il]);
    }
    cache_.cur_pos = snap.cur_pos;
    const int prefix_len = snap.cur_pos;

    sampler_ = req.sampler;
    if (req.do_sample && sampler_.seed != 0) {
        sampler_rng_.seed(sampler_.seed);
    }

    // Prefill only tokens after the restored prefix
    if (prefix_len < (int)req.prompt.size()) {
        std::vector<int32_t> remaining(req.prompt.begin() + prefix_len,
                                        req.prompt.end());
        const int committed = do_prefill(remaining, out_io, prefix_len);
        if (committed < 0) {
            result.error = "prefill after restore";
            return result;
        }
    }

    const int total_committed = (int)req.prompt.size();
    cache_.cur_pos = total_committed;

    // BUG3 FIX (mirror): populate feature ring for the full prompt.
    // The snapshot only restores KV layers; the feature ring is not snapshotted.
    // Re-run verify_batch over the full prompt to fill feature_mirror_ before
    // the first spec-decode round.
    if (supports_dflash_spec_decode() && !sampler_.needs_logit_processing()) {
        DFlashTarget * tgt = dflash_target();
        if (tgt) {
            const int saved_pos = cache_.cur_pos;
            const int prefill_chunk = std::max(1,
                cfg_.chunk > 0 ? cfg_.chunk : 512);
            const int prompt_len = (int)req.prompt.size();
            bool capture_ok = true;
            for (int start = 0; start < prompt_len && capture_ok; start += prefill_chunk) {
                const int n = std::min(prefill_chunk, prompt_len - start);
                std::vector<int32_t> chunk(req.prompt.begin() + start,
                                           req.prompt.begin() + start + n);
                int dummy_last = 0;
                if (!tgt->verify_batch(chunk, /*base_pos=*/start,
                                       dummy_last, /*all_argmax=*/nullptr)) {
                    std::fprintf(stderr,
                        "[qwen3moe] restore: prefill feature capture failed "
                        "(start=%d n=%d) — spec decode may degrade\n",
                        start, n);
                    capture_ok = false;
                }
            }
            cache_.cur_pos = saved_pos;
        }
    }

    if (req.snap_slot >= 0 && req.snap_pos > 0 && req.snap_pos <= total_committed) {
        cache_.cur_pos = req.snap_pos;
        if (snapshot_save(req.snap_slot)) {
            std::printf("[snap] inline slot=%d cur_pos=%d\n",
                        req.snap_slot, req.snap_pos);
            std::fflush(stdout);
        }
        cache_.cur_pos = total_committed;
    }

    if (req.n_gen <= 0) {
        out_io.emit(-1);
        result.ok = true;
        return result;
    }

    const int hidden = w_.n_embd;
    const int vocab  = w_.n_vocab;
    std::vector<float> logits;
    std::vector<float> embed_buf(hidden);

    int32_t last_tok = req.prompt.back();
    if (!embed_tokens(backend_, w_.tok_embd, &last_tok, 1, hidden, embed_buf.data())) {
        result.error = "embed alloc";
        return result;
    }
    if (!do_step(embed_buf.data(), 1, total_committed - 1, logits)) {
        result.error = "first logits";
        return result;
    }

    int32_t first;
    if (sampler_.needs_logit_processing()) {
        first = sample_logits(logits.data(), vocab, sampler_,
                              result.tokens, sampler_rng_);
    } else {
        first = 0;
        float best = logits[0];
        for (int j = 1; j < vocab; ++j) {
            if (logits[j] > best) { best = logits[j]; first = j; }
        }
    }

    // BUG2 FIX (mirror): same as generate() — for spec decode don't pre-emit first.
    int cur_committed = total_committed;
    if (req.n_gen > 0 && supports_dflash_spec_decode() && !sampler_.needs_logit_processing()) {
        last_prefill_tok_ = first;
        if (!do_spec_decode(cur_committed, req.n_gen, result.tokens, out_io,
                            req.hint_tokens)) {
            result.error = "spec decode";
            out_io.emit(-1);
            return result;
        }
    } else {
        result.tokens.push_back(first);
        out_io.emit(first);

        if (out_io.cancelled) {
            out_io.emit(-1);
            result.ok = true;
            return result;
        }
        if (first == 151643 || first == 151645) {
            out_io.emit(-1);
            result.ok = true;
            return result;
        }

        if (req.n_gen > 1) {
            if (!embed_tokens(backend_, w_.tok_embd, &first, 1, hidden, embed_buf.data())) {
                result.error = "embed2 alloc";
                return result;
            }
            if (!do_step(embed_buf.data(), 1, cur_committed, last_logits_)) {
                result.error = "decode logits";
                return result;
            }
            cur_committed++;
            cache_.cur_pos = cur_committed;

            if (!do_decode(cur_committed, req.n_gen - 1, result.tokens, out_io)) {
                result.error = "decode";
                return result;
            }
        }
    }

    out_io.emit(-1);
    result.ok = true;
    return result;
}

// ── Snapshots ──────────────────────────────────────────────────────────────

bool Qwen3MoeBackend::snapshot_save(int slot) {
    if (slot < 0 || slot >= PREFIX_SLOTS) return false;
    snapshot_free(slot);

    auto & snap = snapshots_[slot];
    const int n_layer = cache_.n_layer;

    ggml_init_params ip{};
    ip.mem_size = ggml_tensor_overhead() * (size_t)(n_layer * 2 + 4) + 4096;
    ip.no_alloc = true;
    snap.ctx = ggml_init(ip);
    if (!snap.ctx) return false;

    snap.k_snap.resize(n_layer);
    snap.v_snap.resize(n_layer);
    for (int il = 0; il < n_layer; ++il) {
        snap.k_snap[il] = ggml_dup_tensor(snap.ctx, cache_.k[il]);
        snap.v_snap[il] = ggml_dup_tensor(snap.ctx, cache_.v[il]);
        char name[64];
        std::snprintf(name, sizeof(name), "snap_k_%d", il);
        ggml_set_name(snap.k_snap[il], name);
        std::snprintf(name, sizeof(name), "snap_v_%d", il);
        ggml_set_name(snap.v_snap[il], name);
    }

    snap.buf = ggml_backend_alloc_ctx_tensors(snap.ctx, backend_);
    if (!snap.buf) {
        ggml_free(snap.ctx);
        snap.ctx = nullptr;
        return false;
    }

    for (int il = 0; il < n_layer; ++il) {
        ggml_backend_tensor_copy(cache_.k[il], snap.k_snap[il]);
        ggml_backend_tensor_copy(cache_.v[il], snap.v_snap[il]);
    }
    snap.cur_pos = cache_.cur_pos;

    std::printf("[qwen3moe] snapshot saved slot=%d pos=%d\n", slot, snap.cur_pos);
    std::fflush(stdout);
    return true;
}

void Qwen3MoeBackend::snapshot_free(int slot) {
    if (slot < 0 || slot >= PREFIX_SLOTS) return;
    free_qwen3moe_snapshot(snapshots_[slot]);
}

bool Qwen3MoeBackend::snapshot_used(int slot) const {
    if (slot < 0 || slot >= PREFIX_SLOTS) return false;
    return snapshots_[slot].ctx != nullptr;
}

int Qwen3MoeBackend::snapshot_cur_pos(int slot) const {
    if (slot < 0 || slot >= PREFIX_SLOTS) return 0;
    return snapshots_[slot].cur_pos;
}

// ── DFlash target accessor ─────────────────────────────────────────────────

DFlashTarget * Qwen3MoeBackend::dflash_target() {
    if (!dflash_target_ && cfg_.draft_path && feature_mirror_.target_feat) {
        dflash_target_ = std::make_unique<Qwen3MoeDFlashTarget>(
            w_, cache_, backend_, feature_mirror_,
            cfg_.fa_window, cfg_.kq_stride_pad,
            dw_.n_target_layers, dw_.mask_token_id);
    }
    return dflash_target_.get();
}

// ── DFlash speculative decode ──────────────────────────────────────────────
// Mirrors Qwen35Backend::do_spec_decode: draft → verify → accept → replay.

bool Qwen3MoeBackend::do_spec_decode(int                    committed,
                                      int                    n_gen,
                                      std::vector<int32_t> & out_tokens,
                                      const DaemonIO &       io,
                                      const std::vector<int32_t> * hint_tokens) {
    DFlashTarget * target = dflash_target();
    if (!target) return false;

    const int hidden = w_.n_embd;
    const int q_len  = dw_.block_size;

    // BUG2 FIX: seed from last_prefill_tok_ (set by generate() / restore_and_generate()
    // before calling us). This mirrors qwen35_backend where cache_.last_tok holds the
    // prefill's final argmax. We do NOT pre-emit this token; do_spec_decode emits it as
    // replay_tok[0] in the first accepted batch, so there is no double-emission.
    int32_t last_tok = last_prefill_tok_;

    StepGraph draft_sg;

    std::vector<float>   noise_embed((size_t)hidden * q_len);
    std::vector<int32_t> noise_ids(q_len);
    std::vector<int32_t> draft_tok(q_len);
    std::vector<int32_t> target_tok(q_len);
    std::vector<int32_t> pos_q(q_len);
    std::vector<int32_t> pos_k;
    std::vector<float>   local_hidden;

    int n_generated   = 0;
    int n_draft_steps = 0;
    int n_accept_sum  = 0;
    int n_hint_proposed = 0;
    int n_hint_accepted = 0;

    auto t_dec0 = std::chrono::steady_clock::now();

    while (n_generated < n_gen) {
        const int need_commit_budget = n_gen - n_generated;

        // 1. Build noise embeddings: [last_tok, mask, mask, ...]
        noise_ids[0] = last_tok;
        for (int i = 1; i < q_len; i++) noise_ids[i] = target->mask_token_id();
        if (!target->embed_tokens(noise_ids.data(), q_len, noise_embed.data())) {
            std::fprintf(stderr, "[qwen3moe-spec] noise embed failed\n");
            step_graph_destroy(draft_sg);
            return false;
        }

        // 2. Draft compute
        constexpr int DRAFT_CTX_MAX_DEFAULT = 2048;
        const int ring_cap = feature_mirror_.cap;
        const int draft_ctx = std::min(committed,
            std::min(ring_cap, std::max(DRAFT_CTX_MAX_DEFAULT, cfg_.draft_ctx_max)));
        const int draft_start = committed - draft_ctx;
        int mirror_slot0 = 0;
        const bool use_mirror_view =
            draft_feature_mirror_can_view(feature_mirror_, committed, draft_ctx, mirror_slot0);

        if (!build_draft_step(draft_sg, dw_, /*lm_head=*/nullptr, draft_backend_,
                              draft_ctx, use_mirror_view ? &feature_mirror_ : nullptr,
                              committed,
                              std::min(ring_cap, std::max(DRAFT_CTX_MAX_DEFAULT, cfg_.draft_ctx_max)))) {
            std::fprintf(stderr, "[qwen3moe-spec] draft build failed\n");
            step_graph_destroy(draft_sg);
            return false;
        }
        if (!use_mirror_view &&
            !copy_feature_ring_range_to_tensor(feature_mirror_, draft_sg.target_hidden_cat,
                                               draft_start, draft_ctx)) {
            std::fprintf(stderr, "[qwen3moe-spec] feature copy failed\n");
            step_graph_destroy(draft_sg);
            return false;
        }
        ggml_backend_tensor_set(draft_sg.inp_embed, noise_embed.data(), 0,
                                sizeof(float) * noise_embed.size());
        pos_k.resize((size_t)draft_ctx + q_len);
        for (int i = 0; i < q_len; i++) pos_q[i] = draft_ctx + i;
        for (int i = 0; i < draft_ctx + q_len; i++) pos_k[i] = i;
        ggml_backend_tensor_set(draft_sg.positions, pos_q.data(), 0,
                                sizeof(int32_t) * pos_q.size());
        ggml_backend_tensor_set(draft_sg.positions_k, pos_k.data(), 0,
                                sizeof(int32_t) * pos_k.size());

        auto st = ggml_backend_graph_compute(draft_backend_, draft_sg.gf);
        if (st != GGML_STATUS_SUCCESS) {
            std::fprintf(stderr, "[qwen3moe-spec] draft compute failed (%d)\n", (int)st);
            step_graph_destroy(draft_sg);
            return false;
        }

        local_hidden.resize((size_t)hidden * q_len);
        ggml_backend_tensor_get(draft_sg.hidden_states, local_hidden.data(), 0,
                                sizeof(float) * local_hidden.size());

        // 3. Project draft hidden → token IDs via target LM head
        if (!target->project_hidden_to_tokens(local_hidden.data(), q_len, draft_tok)) {
            std::fprintf(stderr, "[qwen3moe-spec] projection failed\n");
            step_graph_destroy(draft_sg);
            return false;
        }
        draft_tok[0] = last_tok;

        // 3b. Tool call hint injection
        int hint_fill = 0;
        if (hint_tokens && n_generated < (int)hint_tokens->size()) {
            const int hint_avail = (int)hint_tokens->size() - n_generated;
            hint_fill = std::min(hint_avail, q_len - 1);
            for (int i = 0; i < hint_fill; i++) {
                draft_tok[1 + i] = (*hint_tokens)[n_generated + i];
            }
        }

        // 4. Snapshot KV, verify
        if (!target->snapshot_kv()) {
            std::fprintf(stderr, "[qwen3moe-spec] snapshot_kv failed\n");
            step_graph_destroy(draft_sg);
            return false;
        }

        int verify_last_tok = -1;
        if (!target->verify_batch(draft_tok, committed, verify_last_tok, &target_tok)) {
            std::fprintf(stderr, "[qwen3moe-spec] verify failed\n");
            target->restore_kv();
            step_graph_destroy(draft_sg);
            return false;
        }

        // 5. Accept longest prefix
        int accept_n = 1;
        for (int i = 0; i < q_len - 1; i++) {
            if (draft_tok[i + 1] == target_tok[i]) accept_n++;
            else break;
        }
        if (hint_fill > 0) {
            n_hint_proposed += hint_fill;
            n_hint_accepted += std::min(hint_fill, accept_n - 1);
        }
        int bonus_tok = (accept_n < q_len) ? target_tok[accept_n - 1] : -1;
        int commit_n  = accept_n + (bonus_tok >= 0 ? 1 : 0);
        if (commit_n > need_commit_budget) {
            commit_n = need_commit_budget;
            if (commit_n <= accept_n) bonus_tok = -1;
        }

        // 6. Replay: restore KV, re-run accepted tokens
        if (!target->restore_kv()) {
            std::fprintf(stderr, "[qwen3moe-spec] restore_kv failed\n");
            step_graph_destroy(draft_sg);
            return false;
        }

        std::vector<int32_t> replay_tok((size_t)commit_n);
        for (int i = 0; i < commit_n; i++) {
            replay_tok[i] = (i < accept_n) ? draft_tok[i] : bonus_tok;
        }
        int replay_last_tok = -1;
        if (!target->verify_batch(replay_tok, committed, replay_last_tok, nullptr)) {
            std::fprintf(stderr, "[qwen3moe-spec] replay failed\n");
            step_graph_destroy(draft_sg);
            return false;
        }
        last_tok = replay_last_tok;

        // 7. Feature ring is updated by verify_batch (via copy_capture_slice_to_draft_ring).
        // No separate sync needed here — the replay verify_batch already wrote the
        // accepted-token captures into the mirror.

        // 8. Emit committed tokens
        bool hit_eos = false;
        int emitted  = 0;
        for (int i = 0; i < commit_n; i++) {
            out_tokens.push_back(replay_tok[i]);
            io.emit(replay_tok[i]);
            emitted++;
            if (io.cancelled) break;
            if (replay_tok[i] == 151643 || replay_tok[i] == 151645) {
                hit_eos = true; break;
            }
        }
        committed      += emitted;
        cache_.cur_pos  = committed;
        n_generated    += emitted;
        n_accept_sum   += std::min(accept_n, emitted);
        n_draft_steps++;
        if (io.cancelled || hit_eos) break;
    }

    step_graph_destroy(draft_sg);

    auto t_dec1 = std::chrono::steady_clock::now();
    const double decode_s = std::chrono::duration<double>(t_dec1 - t_dec0).count();
    const int total_draft_pos = std::max(1, n_draft_steps * q_len);
    const double accept_pct = 100.0 * (double)n_accept_sum / (double)total_draft_pos;
    std::fprintf(stderr,
        "[qwen3moe-spec] tokens=%d time=%.3f s speed=%.2f tok/s "
        "steps=%d accepted=%d/%d (%.1f%%) avg_commit=%.2f\n",
        n_generated, decode_s,
        n_generated > 0 ? n_generated / decode_s : 0.0,
        n_draft_steps, n_accept_sum, total_draft_pos, accept_pct,
        n_draft_steps > 0 ? (double)n_generated / (double)n_draft_steps : 0.0);
    if (n_hint_proposed > 0) {
        std::fprintf(stderr, "[qwen3moe-spec] hint: %d/%d accepted (%.1f%%)\n",
                     n_hint_accepted, n_hint_proposed,
                     100.0 * (double)n_hint_accepted / (double)n_hint_proposed);
    }

    return true;
}

// ── Compress (Phase C — stub) ──────────────────────────────────────────────

bool Qwen3MoeBackend::handle_compress(const std::string & /*line*/,
                                       const DaemonIO &    /*io*/) {
    return false;
}
void Qwen3MoeBackend::free_drafter() {}

// ── try_handle_command ─────────────────────────────────────────────────────

bool Qwen3MoeBackend::try_handle_command(const std::string & /*line*/,
                                          const DaemonIO &    /*io*/) {
    return false;
}

// ── Shutdown ───────────────────────────────────────────────────────────────

void Qwen3MoeBackend::shutdown() {
    // Drafter teardown (mirrors qwen35_backend.cpp shutdown order)
    step_graph_destroy(draft_sg_);
    draft_feature_mirror_free(feature_mirror_);
    free_draft_weights(dw_);
    if (split_gpus_ && draft_backend_ && draft_backend_ != backend_) {
        ggml_backend_free(draft_backend_);
        draft_backend_ = nullptr;
        split_gpus_ = false;
    }

    for (int s = 0; s < PREFIX_SLOTS; ++s) {
        free_qwen3moe_snapshot(snapshots_[s]);
    }
    free_qwen3moe_cache(cache_);
    if (!parked_) {
        free_qwen3moe_weights(w_);
    }
    if (backend_) {
        ggml_backend_free(backend_);
        backend_ = nullptr;
    }
}

}  // namespace dflash::common
