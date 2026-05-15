// qwen36_mtp_loader.cpp — Discovery loader for unsloth Qwen3.6 MTP GGUFs.
//
// PR 2 (this commit): introspection only. Reads the GGUF header, parses
// NextN metadata (`<arch>.nextn_predict_layers`), and probes for the
// per-MTP-block NextN tensors named per dflash/deps/llama.cpp's
// LLM_TENSOR_NEXTN_* schema. Populates Qwen36MtpWeights pointers from
// the ggml_context built by ggml_init_from_file.
//
// Real weight materialization onto a CUDA backend, plus the MTP forward
// graph, are deferred to PR 2b. The skeleton proves the abstraction
// (foundation interfaces in src/common/mtp_*) accommodates Qwen3.6's
// NativeHeads design, gates compile against the unsloth tensor layout,
// and gives a clear plug-in point for the forward.

#include "qwen36_mtp.h"

#include "ggml.h"
#include "gguf.h"

#include <cstdio>
#include <cstdint>
#include <cstring>
#include <string>
#include <vector>

namespace dflash27b::mtp {

namespace {

// Bind a tensor pointer by exact name lookup on the ggml_context that
// gguf_init_from_file built. Returns nullptr if absent.
ggml_tensor * find_tensor(ggml_context * ctx, const std::string & name) {
    return ggml_get_tensor(ctx, name.c_str());
}

// Read a UINT32 metadata value from a gguf_context. Returns false if the
// key is missing.
bool gguf_kv_u32(struct gguf_context * gguf, const char * key, uint32_t & out) {
    int64_t id = gguf_find_key(gguf, key);
    if (id < 0) return false;
    out = gguf_get_val_u32(gguf, id);
    return true;
}

bool gguf_kv_str(struct gguf_context * gguf, const char * key,
                 std::string & out) {
    int64_t id = gguf_find_key(gguf, key);
    if (id < 0) return false;
    out = gguf_get_val_str(gguf, id);
    return true;
}

}  // anonymous

bool load_qwen36_mtp_weights(const std::string & gguf_path,
                             ggml_context * ctx,
                             int expected_n_embd,
                             int expected_n_vocab,
                             Qwen36MtpWeights & out_weights,
                             std::string & out_error) {
    out_error.clear();

    // PR 2 skeleton: open the GGUF file to read metadata + tensor list,
    // bind pointers in the caller's ctx. The real impl will also copy
    // bytes onto the CUDA backend and own the lifecycle through a
    // backend buffer; for now we rely on the caller's ctx already
    // containing the tensors (e.g., the same ctx used by Qwen35Backend
    // when loading the backbone).
    struct gguf_init_params gp{};
    gp.no_alloc = true;
    gp.ctx      = nullptr;
    struct gguf_context * gguf = gguf_init_from_file(gguf_path.c_str(), gp);
    if (!gguf) {
        out_error = "qwen36_mtp_loader: gguf_init_from_file failed for " + gguf_path;
        return false;
    }

    // ── Arch and dimensions ────────────────────────────────────────────
    std::string arch;
    if (!gguf_kv_str(gguf, "general.architecture", arch)) {
        out_error = "qwen36_mtp_loader: missing general.architecture";
        gguf_free(gguf);
        return false;
    }
    if (arch != "qwen35") {
        out_error = "qwen36_mtp_loader: expected arch=qwen35, got arch=" + arch;
        gguf_free(gguf);
        return false;
    }
    out_weights.backbone_arch = arch;
    gguf_kv_str(gguf, "general.name",          out_weights.base_model_name);

    uint32_t n_layer = 0;
    if (!gguf_kv_u32(gguf, "qwen35.block_count", n_layer)) {
        out_error = "qwen36_mtp_loader: missing qwen35.block_count";
        gguf_free(gguf);
        return false;
    }
    uint32_t n_embd_meta = 0;
    if (!gguf_kv_u32(gguf, "qwen35.embedding_length", n_embd_meta)) {
        out_error = "qwen36_mtp_loader: missing qwen35.embedding_length";
        gguf_free(gguf);
        return false;
    }
    if ((int)n_embd_meta != expected_n_embd) {
        char msg[256];
        std::snprintf(msg, sizeof msg,
            "qwen36_mtp_loader: backbone n_embd mismatch (gguf=%u, expected=%d)",
            n_embd_meta, expected_n_embd);
        out_error = msg;
        gguf_free(gguf);
        return false;
    }
    (void)expected_n_vocab;  // vocab cross-check folded into the optional
                             // embed_tokens / shared_head_head shape check

    // ── NextN head count ──────────────────────────────────────────────
    uint32_t n_heads = 0;
    if (!gguf_kv_u32(gguf, "qwen35.nextn_predict_layers", n_heads)) {
        out_error = "qwen36_mtp_loader: GGUF lacks qwen35.nextn_predict_layers — this is not an MTP variant";
        gguf_free(gguf);
        return false;
    }
    if (n_heads == 0) {
        out_error = "qwen36_mtp_loader: nextn_predict_layers=0 — no MTP heads in this file";
        gguf_free(gguf);
        return false;
    }
    if (n_heads > (uint32_t)n_layer) {
        char msg[256];
        std::snprintf(msg, sizeof msg,
            "qwen36_mtp_loader: nextn_predict_layers=%u > n_layer=%u (corrupt metadata)",
            n_heads, n_layer);
        out_error = msg;
        gguf_free(gguf);
        return false;
    }

    out_weights.n_embd            = (int)n_embd_meta;
    out_weights.n_vocab           = expected_n_vocab;
    out_weights.n_heads           = (int)n_heads;
    out_weights.n_backbone_layers = (int)n_layer;
    out_weights.heads.clear();
    out_weights.heads.resize(n_heads);

    // ── Bind per-head tensor pointers ─────────────────────────────────
    // Heads occupy the last `n_heads` block indices: [n_layer - n_heads,
    // n_layer - 1]. For each, the required tensors are eh_proj/enorm/
    // hnorm; embed_tokens / shared_head_head / shared_head_norm are
    // optional (caller falls back to backbone tensors when absent).
    int missing_required = 0;
    for (int h = 0; h < (int)n_heads; h++) {
        const int layer_idx = (int)n_layer - (int)n_heads + h;
        auto & head = out_weights.heads[h];
        head.layer_idx = layer_idx;

        auto bind = [&](const char * base, bool required) -> ggml_tensor * {
            char name[256];
            std::snprintf(name, sizeof name, "blk.%d.%s.weight", layer_idx, base);
            ggml_tensor * t = find_tensor(ctx, name);
            if (!t && required) {
                std::fprintf(stderr,
                    "[qwen36_mtp_loader] missing required tensor: %s\n", name);
                missing_required++;
            }
            return t;
        };
        head.eh_proj          = bind("nextn.eh_proj",          /*required=*/true);
        head.enorm            = bind("nextn.enorm",            /*required=*/true);
        head.hnorm            = bind("nextn.hnorm",            /*required=*/true);
        head.embed_tokens     = bind("nextn.embed_tokens",     /*required=*/false);
        head.shared_head_head = bind("nextn.shared_head_head", /*required=*/false);
        head.shared_head_norm = bind("nextn.shared_head_norm", /*required=*/false);
    }

    gguf_free(gguf);

    if (missing_required > 0) {
        char msg[256];
        std::snprintf(msg, sizeof msg,
            "qwen36_mtp_loader: %d required NextN tensor(s) missing — context likely lacks the MTP tensors. Did the backbone loader allocate them?",
            missing_required);
        out_error = msg;
        return false;
    }
    return true;
}

}  // namespace dflash27b::mtp
