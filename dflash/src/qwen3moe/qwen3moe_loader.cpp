// Qwen3MoeBackend GGUF loader.
//
// Implements load_qwen3moe_gguf() and create_qwen3moe_cache().
//
// Loader pattern: Gemma4 (no_alloc=true, ctx=&meta_ctx,
// ggml_backend_alloc_ctx_tensors, dtype-agnostic copy loop).
// DO NOT use the dense qwen3_loader.cpp pattern (hardcodes BF16 and allocates
// host RAM for the whole model with no_alloc=false).
//
// Cache pattern: adapted directly from qwen3_backend.cpp::create_qwen3_cache.

#include "qwen3moe_internal.h"
#include "internal.h"

#include <cinttypes>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <utility>
#include <vector>

#if !defined(_WIN32)
#include <cerrno>
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>
#endif

namespace dflash::common {

namespace {

// ── Small helpers ──────────────────────────────────────────────────────────

uint32_t qmoe_get_u32(gguf_context * g, const char * key, uint32_t def) {
    int64_t id = gguf_find_key(g, key);
    if (id < 0) return def;
    if (gguf_get_kv_type(g, id) == GGUF_TYPE_ARRAY) {
        if (gguf_get_arr_n(g, id) == 0) return def;
        return ((const uint32_t *)gguf_get_arr_data(g, id))[0];
    }
    return gguf_get_val_u32(g, id);
}

float qmoe_get_f32(gguf_context * g, const char * key, float def) {
    int64_t id = gguf_find_key(g, key);
    if (id < 0) return def;
    if (gguf_get_kv_type(g, id) == GGUF_TYPE_ARRAY) {
        if (gguf_get_arr_n(g, id) == 0) return def;
        return ((const float *)gguf_get_arr_data(g, id))[0];
    }
    return gguf_get_val_f32(g, id);
}

}  // namespace

// ─────────────────────────────────────────────────────────────────────────────
// load_qwen3moe_gguf
// ─────────────────────────────────────────────────────────────────────────────

bool load_qwen3moe_gguf(const std::string & path,
                        ggml_backend_t      backend,
                        Qwen3MoeWeights &   out) {
    // ── Step 1: open meta context (no_alloc, dtype-agnostic) ───────────────
    ggml_context * meta_ctx = nullptr;
    gguf_init_params gip{};
    gip.no_alloc = true;
    gip.ctx      = &meta_ctx;
    gguf_context * gctx = gguf_init_from_file(path.c_str(), gip);
    if (!gctx) {
        set_last_error("[qwen3moe-loader] gguf_init_from_file failed: " + path);
        return false;
    }

    // ── Step 2: validate arch ───────────────────────────────────────────────
    {
        int64_t aid = gguf_find_key(gctx, "general.architecture");
        if (aid < 0) {
            set_last_error("[qwen3moe-loader] missing general.architecture");
            gguf_free(gctx);
            return false;
        }
        const char * arch = gguf_get_val_str(gctx, aid);
        if (std::string(arch) != "qwen3moe") {
            set_last_error(std::string("[qwen3moe-loader] unexpected arch: ")
                           + arch + " (expected qwen3moe)");
            gguf_free(gctx);
            return false;
        }
    }

    // ── Step 3: read metadata ───────────────────────────────────────────────
    const uint32_t n_layer       = qmoe_get_u32(gctx, "qwen3moe.block_count",              0);
    const uint32_t n_embd        = qmoe_get_u32(gctx, "qwen3moe.embedding_length",         0);
    const uint32_t n_head        = qmoe_get_u32(gctx, "qwen3moe.attention.head_count",     0);
    const uint32_t n_head_kv     = qmoe_get_u32(gctx, "qwen3moe.attention.head_count_kv",  0);
    const uint32_t n_expert      = qmoe_get_u32(gctx, "qwen3moe.expert_count",             0);
    const uint32_t n_expert_used = qmoe_get_u32(gctx, "qwen3moe.expert_used_count",        0);
    const uint32_t n_ff_exp      = qmoe_get_u32(gctx, "qwen3moe.expert_feed_forward_length", 0);
    const uint32_t n_ctx_max     = qmoe_get_u32(gctx, "qwen3moe.context_length",           0);
    const float    rope_theta    = qmoe_get_f32(gctx, "qwen3moe.rope.freq_base",           1e7f);
    const float    norm_eps      = qmoe_get_f32(gctx, "qwen3moe.attention.layer_norm_rms_epsilon", 1e-6f);

    // head_dim: NO default — abort if missing (see Gotcha A in spec)
    uint32_t head_dim = 0;
    {
        int64_t id = gguf_find_key(gctx, "qwen3moe.attention.key_length");
        if (id < 0) {
            set_last_error("[qwen3moe-loader] missing required key qwen3moe.attention.key_length");
            gguf_free(gctx);
            return false;
        }
        head_dim = gguf_get_val_u32(gctx, id);
    }

    // value_length — log only (sanity check vs key_length)
    {
        uint32_t val_len = qmoe_get_u32(gctx, "qwen3moe.attention.value_length", 0);
        if (val_len != 0 && val_len != head_dim) {
            std::fprintf(stderr,
                "[qwen3moe-loader] WARNING: value_length=%u != key_length=%u\n",
                val_len, head_dim);
        }
    }

    // Dense FFN (absent in all-MoE arch) — log only
    {
        uint32_t n_ff = qmoe_get_u32(gctx, "qwen3moe.feed_forward_length", 0);
        if (n_ff != 0) {
            std::fprintf(stderr,
                "[qwen3moe-loader] note: feed_forward_length=%u (dense FFN absent in MoE arch)\n",
                n_ff);
        }
    }

    // Tokenizer IDs — log only
    {
        uint32_t bos = qmoe_get_u32(gctx, "tokenizer.ggml.bos_token_id", 0xFFFFFFFFu);
        uint32_t eos = qmoe_get_u32(gctx, "tokenizer.ggml.eos_token_id", 0xFFFFFFFFu);
        if (bos != 0xFFFFFFFFu)
            std::fprintf(stderr, "[qwen3moe-loader] bos_token_id=%u\n", bos);
        if (eos != 0xFFFFFFFFu)
            std::fprintf(stderr, "[qwen3moe-loader] eos_token_id=%u\n", eos);
    }

    // ── Step 4: derive n_vocab — prefer metadata, fall back via axis match ──
    // Some Coder ggufs store tok_embd as [vocab, hidden] (transposed) so we
    // can't blindly read ne[1]. Pick the axis that does NOT equal n_embd.
    uint32_t n_vocab = qmoe_get_u32(gctx, "qwen3moe.vocab_size", 0);
    if (n_vocab == 0) {
        ggml_tensor * tok = ggml_get_tensor(meta_ctx, "token_embd.weight");
        if (tok) {
            if ((uint32_t)tok->ne[0] == n_embd) {
                n_vocab = (uint32_t)tok->ne[1];
            } else if ((uint32_t)tok->ne[1] == n_embd) {
                n_vocab = (uint32_t)tok->ne[0];
                std::fprintf(stderr,
                    "[qwen3moe-loader] tok_embd is transposed [vocab=%lld, hidden=%lld]; "
                    "n_vocab=%u (will need physical transpose at load)\n",
                    (long long)tok->ne[0], (long long)tok->ne[1], n_vocab);
            } else {
                std::fprintf(stderr,
                    "[qwen3moe-loader] tok_embd shape [%lld,%lld] does not match "
                    "n_embd=%u on either axis; set qwen3moe.vocab_size in GGUF metadata\n",
                    (long long)tok->ne[0], (long long)tok->ne[1], n_embd);
            }
        }
    }

    // ── Step 5: zero-guard essential hparams ───────────────────────────────
    if (n_layer == 0 || n_embd == 0 || n_head == 0 || n_head_kv == 0
            || head_dim == 0 || n_expert == 0 || n_ff_exp == 0 || n_vocab == 0) {
        set_last_error("[qwen3moe-loader] missing essential hparams"
                       " (n_layer=" + std::to_string(n_layer)
                       + " n_embd=" + std::to_string(n_embd)
                       + " n_head=" + std::to_string(n_head)
                       + " n_head_kv=" + std::to_string(n_head_kv)
                       + " head_dim=" + std::to_string(head_dim)
                       + " n_expert=" + std::to_string(n_expert)
                       + " n_ff_exp=" + std::to_string(n_ff_exp)
                       + " n_vocab=" + std::to_string(n_vocab) + ")");
        gguf_free(gctx);
        return false;
    }

    // ── Step 6: populate out.* fields ──────────────────────────────────────
    out.n_layer        = (int)n_layer;
    out.n_embd         = (int)n_embd;
    out.n_head         = (int)n_head;
    out.n_head_kv      = (int)n_head_kv;
    out.head_dim       = (int)head_dim;
    out.n_expert       = (int)n_expert;
    out.n_expert_used  = (int)n_expert_used;
    out.n_ff_exp       = (int)n_ff_exp;
    out.n_vocab        = (int)n_vocab;
    out.n_ctx_max      = (int)(n_ctx_max > 0 ? n_ctx_max : 262144);
    out.rope_theta     = rope_theta;
    out.norm_eps       = norm_eps;
    out.norm_topk_prob = true;  // Qwen3-MoE default
    out.backend        = backend;

    // ── Step 7: mmap the GGUF ──────────────────────────────────────────────
    int fd = ::open(path.c_str(), O_RDONLY);
    if (fd < 0) {
        set_last_error(std::string("[qwen3moe-loader] open failed: ") + strerror(errno)
                       + " path=" + path);
        gguf_free(gctx);
        return false;
    }
    struct stat st;
    if (::fstat(fd, &st) < 0) {
        set_last_error("[qwen3moe-loader] fstat failed");
        ::close(fd);
        gguf_free(gctx);
        return false;
    }
    void * mm = ::mmap(nullptr, (size_t)st.st_size, PROT_READ, MAP_PRIVATE, fd, 0);
    ::close(fd);
    if (mm == MAP_FAILED) {
        set_last_error("[qwen3moe-loader] mmap failed");
        gguf_free(gctx);
        return false;
    }

    // ── Step 8: allocate backend buffer + copy all tensors ─────────────────
    out.ctx = meta_ctx;
    out.buf = ggml_backend_alloc_ctx_tensors(meta_ctx, backend);
    if (!out.buf) {
        set_last_error("[qwen3moe-loader] ggml_backend_alloc_ctx_tensors failed");
        ::munmap(mm, (size_t)st.st_size);
        gguf_free(gctx);
        out.ctx = nullptr;
        return false;
    }

    // ── Step 9: dtype-agnostic copy loop (Gemma4 pattern) ──────────────────
    const size_t data_offset = gguf_get_data_offset(gctx);
    const int n_tensors = gguf_get_n_tensors(gctx);
    for (int i = 0; i < n_tensors; ++i) {
        const char * name = gguf_get_tensor_name(gctx, i);
        ggml_tensor * t = ggml_get_tensor(meta_ctx, name);
        if (!t) continue;
        size_t off = data_offset + gguf_get_tensor_offset(gctx, i);
        ggml_backend_tensor_set(t, (const char *)mm + off, 0, ggml_nbytes(t));
    }

    // Done with mmap — data is on the backend now.
    ::munmap(mm, (size_t)st.st_size);

    // ── Step 10: assign struct pointers ────────────────────────────────────

    // Top-level
    out.tok_embd = ggml_get_tensor(meta_ctx, "token_embd.weight");
    out.out_norm = ggml_get_tensor(meta_ctx, "output_norm.weight");
    out.output   = ggml_get_tensor(meta_ctx, "output.weight");

    if (!out.tok_embd || !out.out_norm || !out.output) {
        set_last_error("[qwen3moe-loader] missing top-level tensor"
            + std::string(!out.tok_embd ? " token_embd.weight" : "")
            + std::string(!out.out_norm ? " output_norm.weight" : "")
            + std::string(!out.output   ? " output.weight"      : ""));
        gguf_free(gctx);
        return false;
    }

    // Note: gguf-compat fixups for transposed [vocab,hidden] tok_embd and for
    // k-quant tok_embd that ggml-cuda's get_rows doesn't support are NOT done
    // here yet — the dequant+swap path was attempted but produced incorrect
    // embeddings (causes immediate EOS sampling on Coder UD-IQ2_XXS and
    // Coder-Instruct-IQ2_XXS-tokembd-F16). Tracked as open follow-ups; the
    // vocab-axis detection above is the safe partial fix.

    // Per-layer
    out.layers.resize(n_layer);
    char buf[256];
    bool ok = true;
    for (uint32_t il = 0; il < n_layer; ++il) {
        auto & L = out.layers[il];

        auto get = [&](const char * suffix) -> ggml_tensor * {
            std::snprintf(buf, sizeof(buf), "blk.%u.%s", il, suffix);
            ggml_tensor * t = ggml_get_tensor(meta_ctx, buf);
            if (!t) {
                std::fprintf(stderr,
                    "[qwen3moe-loader] missing tensor layer %u: %s\n", il, buf);
                ok = false;
            }
            return t;
        };

        // Attention
        L.attn_norm = get("attn_norm.weight");
        L.wq        = get("attn_q.weight");
        L.wk        = get("attn_k.weight");
        L.wv        = get("attn_v.weight");
        L.wo        = get("attn_output.weight");
        L.q_norm    = get("attn_q_norm.weight");
        L.k_norm    = get("attn_k_norm.weight");

        // MoE FFN
        L.ffn_norm      = get("ffn_norm.weight");
        L.ffn_gate_inp  = get("ffn_gate_inp.weight");
        L.ffn_gate_exps = get("ffn_gate_exps.weight");
        L.ffn_up_exps   = get("ffn_up_exps.weight");
        L.ffn_down_exps = get("ffn_down_exps.weight");
    }

    if (!ok) {
        set_last_error("[qwen3moe-loader] one or more per-layer tensors are missing");
        gguf_free(gctx);
        return false;
    }

    // ── Step 11: release gguf context (meta_ctx owned by out.ctx) ──────────
    gguf_free(gctx);

    // ── Step 12: banner ─────────────────────────────────────────────────────
    std::printf("[qwen3moe] loaded %d layers, hidden=%d, experts=%d/%d, n_ff_exp=%d\n",
                out.n_layer, out.n_embd, out.n_expert_used, out.n_expert, out.n_ff_exp);
    std::printf("[qwen3moe] n_head=%d, n_head_kv=%d, head_dim=%d, vocab=%d, rope_theta=%g\n",
                out.n_head, out.n_head_kv, out.head_dim, out.n_vocab, out.rope_theta);
    std::fflush(stdout);

    return true;
}

// ─────────────────────────────────────────────────────────────────────────────
// free_qwen3moe_weights
// ─────────────────────────────────────────────────────────────────────────────

void free_qwen3moe_weights(Qwen3MoeWeights & w) {
    if (w.buf) { ggml_backend_buffer_free(w.buf); w.buf = nullptr; }
    if (w.ctx) { ggml_free(w.ctx);                w.ctx = nullptr; }
    w.layers.clear();
}

// ─────────────────────────────────────────────────────────────────────────────
// create_qwen3moe_cache — BF16/F16 KV cache, [head_dim, n_head_kv, max_ctx]
// ─────────────────────────────────────────────────────────────────────────────
//
// Layout note (Phase 2): positions are the OUTER dim so each kv slot is a
// D*Hk-element contiguous block. This makes the cache reshapeable to
// [D*Hk, max_ctx] for ggml_set_rows writes — required so per-step K/V writes
// don't bake kv_start into the graph topology. For attention reads we view
// as [D, Hk, kv_len] and permute(0,2,1,3) to feed flash_attn_ext, which
// expects [D, kv_len, Hk] (the old static layout).
bool create_qwen3moe_cache(ggml_backend_t         backend,
                           const Qwen3MoeWeights & w,
                           int                     max_ctx,
                           Qwen3MoeCache &         out) {
    const int n_layer = w.n_layer;
    const int D       = w.head_dim;
    const int Hk      = w.n_head_kv;

    ggml_init_params ip{};
    ip.mem_size = ggml_tensor_overhead() * (size_t)(n_layer * 2 + 4) + 4096;
    ip.no_alloc = true;
    out.ctx = ggml_init(ip);
    if (!out.ctx) {
        set_last_error("[qwen3moe-cache] ggml_init failed");
        return false;
    }

    // BF16 where WMMA flash-prefill is available, else F16.
    const ggml_type half_type =
#ifdef DFLASH27B_HAVE_CUDA_WMMA_FLASHPREFILL
        GGML_TYPE_BF16;
#else
        GGML_TYPE_F16;
#endif

    out.k.resize(n_layer);
    out.v.resize(n_layer);
    for (int il = 0; il < n_layer; ++il) {
        out.k[il] = ggml_new_tensor_3d(out.ctx, half_type, D, Hk, max_ctx);
        out.v[il] = ggml_new_tensor_3d(out.ctx, half_type, D, Hk, max_ctx);
    }

    out.buf = ggml_backend_alloc_ctx_tensors(out.ctx, backend);
    if (!out.buf) {
        set_last_error("[qwen3moe-cache] ggml_backend_alloc_ctx_tensors failed");
        ggml_free(out.ctx);
        out.ctx = nullptr;
        return false;
    }

    out.cur_pos = 0;
    out.max_ctx = max_ctx;
    out.n_layer = n_layer;
    return true;
}

// ─────────────────────────────────────────────────────────────────────────────
// free helpers (cleanup only — already implemented in scaffold, preserved here)
// ─────────────────────────────────────────────────────────────────────────────

void free_qwen3moe_cache(Qwen3MoeCache & c) {
    if (c.buf) { ggml_backend_buffer_free(c.buf); c.buf = nullptr; }
    if (c.ctx) { ggml_free(c.ctx);                c.ctx = nullptr; }
    c.k.clear();
    c.v.clear();
    c.cur_pos = 0;
}

void free_qwen3moe_snapshot(Qwen3MoeSnapshot & s) {
    if (s.buf) { ggml_backend_buffer_free(s.buf); s.buf = nullptr; }
    if (s.ctx) { ggml_free(s.ctx);                s.ctx = nullptr; }
    s.k_snap.clear();
    s.v_snap.clear();
    s.cur_pos = 0;
    s.last_tok = -1;
}

}  // namespace dflash::common
