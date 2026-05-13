// Loaders for the Gemma4 DFlash draft model weights.
// Graph builders (build_draft_kv_prefill_graph, build_gemma4_draft_graph)
// land in the dflash-runtime PR.
//
// Architecture:
//   - 5-layer block-diffusion draft (4 SWA + 1 full attention)
//   - 6 captured target layers
//   - FC input = 6 * target_hidden (target_hidden = 4096 for all Gemma4 variants),
//     giving FC width = 24576
//   - Logit softcapping: tanh(logits / cap) * cap, cap = 30.0
//   - Tied lm_head: uses tok_embd transposed (or a provided lm_head weight)
//   - Vocab = 262144
//   - Draft has its own lm_head + softcap
//
// Safetensors tensor naming (actual file, no model. prefix):
//   fc.weight                                           → fc
//   hidden_norm.weight                                  → hidden_norm
//   norm.weight                                         → out_norm
//   layers.{i}.self_attn.q_proj.weight                  → wq
//   layers.{i}.self_attn.k_proj.weight                  → wk
//   layers.{i}.self_attn.v_proj.weight                  → wv
//   layers.{i}.self_attn.o_proj.weight                  → wo
//   layers.{i}.self_attn.q_norm.weight                  → q_norm
//   layers.{i}.self_attn.k_norm.weight                  → k_norm
//   layers.{i}.input_layernorm.weight                   → attn_norm
//   layers.{i}.post_attention_layernorm.weight          → ffn_norm
//   layers.{i}.mlp.gate_proj.weight                     → w_gate
//   layers.{i}.mlp.up_proj.weight                       → w_up
//   layers.{i}.mlp.down_proj.weight                     → w_down
//   (no embed_tokens — tok_embd is injected from the target at runtime)

#include "internal.h"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fcntl.h>
#include <string>
#include <unordered_map>
#include <vector>

#if defined(_WIN32)
#  if !defined(NOMINMAX)
#    define NOMINMAX
#  endif
#  if !defined(WIN32_LEAN_AND_MEAN)
#    define WIN32_LEAN_AND_MEAN
#  endif
#  include <windows.h>
#else
#  include <cerrno>
#  include <sys/mman.h>
#  include <sys/stat.h>
#  include <unistd.h>
#endif

namespace dflash27b {

// ─── Safetensors loader ───────────────────────────────────────────────────

namespace {

struct GStEntry {
    std::string          dtype;
    std::vector<int64_t> shape;
    uint64_t             data_start;
    uint64_t             data_end;
};

using GStMap = std::unordered_map<std::string, GStEntry>;

// Minimal safetensors JSON header parser (same algorithm as safetensors_draft.cpp).
static bool parse_gst_header(const char * h, size_t hlen, GStMap & out) {
    auto skip_ws = [&](size_t & i) {
        while (i < hlen && (h[i] == ' ' || h[i] == '\t' ||
                            h[i] == '\n' || h[i] == '\r')) i++;
    };
    size_t i = 0;
    skip_ws(i);
    if (i >= hlen || h[i] != '{') return false;
    i++;
    while (i < hlen) {
        skip_ws(i);
        if (i >= hlen) return false;
        if (h[i] == '}') { i++; break; }
        if (h[i] == ',') { i++; skip_ws(i); }
        if (i >= hlen || h[i] != '"') return false;
        i++;
        size_t name_start = i;
        while (i < hlen && h[i] != '"') i++;
        if (i >= hlen) return false;
        std::string name(h + name_start, i - name_start);
        i++;
        skip_ws(i);
        if (i >= hlen || h[i] != ':') return false;
        i++;
        skip_ws(i);
        if (i >= hlen || h[i] != '{') return false;
        size_t obj_start = i;
        int depth = 0;
        size_t obj_end = i;
        for (; obj_end < hlen; obj_end++) {
            if      (h[obj_end] == '{') depth++;
            else if (h[obj_end] == '}') { if (--depth == 0) { obj_end++; break; } }
        }
        if (depth != 0) return false;
        if (name == "__metadata__") { i = obj_end; continue; }

        std::string obj(h + obj_start, obj_end - obj_start);
        GStEntry e;
        {
            auto k = obj.find("\"dtype\":\"");
            if (k == std::string::npos) return false;
            auto vs = k + 9;
            auto ve = obj.find('"', vs);
            if (ve == std::string::npos) return false;
            e.dtype = obj.substr(vs, ve - vs);
        }
        {
            auto k = obj.find("\"shape\":[");
            if (k == std::string::npos) return false;
            auto vs = k + 9;
            auto ve = obj.find(']', vs);
            if (ve == std::string::npos) return false;
            const char * p  = obj.c_str() + vs;
            const char * pe = obj.c_str() + ve;
            while (p < pe) {
                char * end = nullptr;
                long long v = std::strtoll(p, &end, 10);
                if (end == p) break;
                e.shape.push_back((int64_t)v);
                p = end;
                while (p < pe && (*p == ',' || *p == ' ')) p++;
            }
        }
        {
            auto k = obj.find("\"data_offsets\":[");
            if (k == std::string::npos) return false;
            auto vs = k + 16;
            auto ve = obj.find(']', vs);
            if (ve == std::string::npos) return false;
            unsigned long long s = 0, ed = 0;
            if (std::sscanf(obj.c_str() + vs, "%llu , %llu", &s, &ed) != 2)
                if (std::sscanf(obj.c_str() + vs, "%llu,%llu", &s, &ed) != 2) return false;
            e.data_start = s;
            e.data_end   = ed;
        }
        out.emplace(std::move(name), std::move(e));
        i = obj_end;
    }
    return true;
}

static ggml_type gst_dtype_to_ggml(const std::string & dt) {
    if (dt == "BF16") return GGML_TYPE_BF16;
    if (dt == "F16")  return GGML_TYPE_F16;
    if (dt == "F32")  return GGML_TYPE_F32;
    return GGML_TYPE_COUNT;
}

struct GMmap {
    void * addr = nullptr;
    size_t len  = 0;
#if defined(_WIN32)
    HANDLE hFile = INVALID_HANDLE_VALUE;
    HANDLE hMap  = nullptr;
#else
    int fd = -1;
#endif

    bool open_ro(const std::string & path, std::string & err) {
#if defined(_WIN32)
        hFile = CreateFileA(path.c_str(), GENERIC_READ, FILE_SHARE_READ,
                            nullptr, OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL, nullptr);
        if (hFile == INVALID_HANDLE_VALUE) {
            err = "CreateFileA: " + path + ": error " + std::to_string(GetLastError());
            return false;
        }
        LARGE_INTEGER sz;
        if (!GetFileSizeEx(hFile, &sz)) {
            err = "GetFileSizeEx failed"; return false;
        }
        len = (size_t)sz.QuadPart;
        hMap = CreateFileMappingA(hFile, nullptr, PAGE_READONLY, 0, 0, nullptr);
        if (!hMap) { err = "CreateFileMappingA failed"; return false; }
        addr = MapViewOfFile(hMap, FILE_MAP_READ, 0, 0, 0);
        if (!addr) { err = "MapViewOfFile failed"; return false; }
#else
        fd = ::open(path.c_str(), O_RDONLY);
        if (fd < 0) { err = "open: " + path + ": " + std::strerror(errno); return false; }
        struct stat st;
        if (::fstat(fd, &st) < 0) { err = "fstat: " + std::string(std::strerror(errno)); return false; }
        len  = (size_t)st.st_size;
        addr = ::mmap(nullptr, len, PROT_READ, MAP_PRIVATE, fd, 0);
        if (addr == MAP_FAILED) {
            err  = "mmap: " + std::string(std::strerror(errno));
            addr = nullptr; return false;
        }
#endif
        return true;
    }

    ~GMmap() {
#if defined(_WIN32)
        if (addr)                             UnmapViewOfFile(addr);
        if (hMap)                             CloseHandle(hMap);
        if (hFile != INVALID_HANDLE_VALUE)    CloseHandle(hFile);
#else
        if (addr) ::munmap(addr, len);
        if (fd >= 0) ::close(fd);
#endif
    }
};

// Allocate one ggml tensor for a safetensors entry.
// HF row-major [out, in] → ggml ne[0]=in, ne[1]=out (byte layout identical).
// norm weights are kept as F32 (ggml CUDA elementwise ops require non-BF16 src1).
// Projection weights stay BF16 (Ampere+) or are converted to F16 (Turing).
static ggml_tensor * galloc_tensor(
    ggml_context *               gctx,
    const GStMap &               st,
    const std::string &          name,
    const std::vector<int64_t> & expected_shape,
    ggml_type                    gt_override = GGML_TYPE_COUNT)
{
    auto it = st.find(name);
    if (it == st.end()) {
        set_last_error("gemma4 safetensors: missing tensor '" + name + "'");
        return nullptr;
    }
    const GStEntry & e = it->second;
    if (e.dtype != "BF16") {
        set_last_error("gemma4 safetensors: '" + name + "' dtype=" + e.dtype +
                       " expected BF16");
        return nullptr;
    }
    if (e.shape.size() != expected_shape.size()) {
        set_last_error("gemma4 safetensors: '" + name + "' ndim mismatch");
        return nullptr;
    }
    for (size_t k = 0; k < expected_shape.size(); k++) {
        if (e.shape[k] != expected_shape[k]) {
            char buf[256];
            std::snprintf(buf, sizeof(buf),
                "gemma4 safetensors: '%s' shape[%zu]=%lld expected %lld",
                name.c_str(), k, (long long)e.shape[k], (long long)expected_shape[k]);
            set_last_error(buf);
            return nullptr;
        }
    }
    ggml_type gt = (gt_override == GGML_TYPE_COUNT) ? GGML_TYPE_BF16 : gt_override;
    ggml_tensor * t = nullptr;
    if (expected_shape.size() == 1) {
        t = ggml_new_tensor_1d(gctx, gt, expected_shape[0]);
    } else if (expected_shape.size() == 2) {
        // [out, in] → ne[0]=in, ne[1]=out
        t = ggml_new_tensor_2d(gctx, gt, expected_shape[1], expected_shape[0]);
    } else {
        set_last_error("gemma4 safetensors: unexpected ndim > 2 for '" + name + "'");
        return nullptr;
    }
    ggml_set_name(t, name.c_str());
    return t;
}

static void g_bf16_to_f32(const uint16_t * src, float * dst, size_t n) {
    for (size_t i = 0; i < n; i++) {
        uint32_t bits = ((uint32_t)src[i]) << 16;
        std::memcpy(&dst[i], &bits, 4);
    }
}

static void g_bf16_to_f16(const uint16_t * src, uint16_t * dst, size_t n) {
    for (size_t i = 0; i < n; i++) {
        uint32_t bits = ((uint32_t)src[i]) << 16;
        float f;
        std::memcpy(&f, &bits, 4);
        uint32_t u;
        std::memcpy(&u, &f, 4);
        uint32_t sign = (u >> 16) & 0x8000;
        int32_t  exp  = ((u >> 23) & 0xFF) - 127 + 15;
        uint32_t mant = (u >> 13) & 0x03FF;
        if      (exp <= 0)  dst[i] = (uint16_t)sign;
        else if (exp >= 31) dst[i] = (uint16_t)(sign | 0x7C00);
        else                dst[i] = (uint16_t)(sign | (exp << 10) | mant);
    }
}

static bool g_cuda_has_native_bf16() {
    const char * env = std::getenv("DFLASH27B_DRAFT_FP16");
    if (env && std::atoi(env) != 0) return false;
#if defined(DFLASH27B_MIN_SM) && DFLASH27B_MIN_SM < 80
    return false;
#else
    return true;
#endif
}

static uint32_t get_u32_or(const gguf_context * g, const char * key, uint32_t fallback) {
    int64_t id = gguf_find_key(g, key);
    if (id < 0) return fallback;
    return gguf_get_val_u32(g, id);
}

static float get_f32_or(const gguf_context * g, const char * key, float fallback) {
    int64_t id = gguf_find_key(g, key);
    if (id < 0) return fallback;
    return gguf_get_val_f32(g, id);
}

} // anonymous namespace

// ─── Public loader ────────────────────────────────────────────────────────

// Load Gemma4 DFlash draft weights from a directory containing one or more
// safetensors shards.  We look for files named:
//   model.safetensors           (single-shard)
//   model-00001-of-NNNNN.safetensors  (multi-shard, first shard only for now)
//
// In practice the z-lab Gemma4 draft is small enough to fit in a single shard.
bool load_gemma4_draft_safetensors(const std::string & dir_path,
                                    ggml_backend_t       backend,
                                    GemmaDraftWeights &  out)
{
    // ── 1. Find the shard file ────────────────────────────────────────
    // Try the canonical single-shard name first.
    std::string path = dir_path + "/model.safetensors";
    {
        // Quick existence check without mmap
        int fd_check = ::open(path.c_str(), O_RDONLY);
        if (fd_check < 0) {
            // Fall back to first numbered shard
            path = dir_path + "/model-00001-of-00001.safetensors";
            fd_check = ::open(path.c_str(), O_RDONLY);
            if (fd_check < 0) {
                set_last_error("gemma4 draft: no safetensors file found in " + dir_path);
                return false;
            }
        }
        ::close(fd_check);
    }

    // ── 2. Open + mmap ───────────────────────────────────────────────
    GMmap mm;
    std::string err;
    if (!mm.open_ro(path, err)) { set_last_error(err); return false; }
    if (mm.len < 8) { set_last_error("gemma4 draft: safetensors file too small"); return false; }

    // ── 3. Parse header ──────────────────────────────────────────────
    uint64_t header_len = 0;
    std::memcpy(&header_len, mm.addr, 8);
    if (header_len == 0 || 8 + header_len > mm.len) {
        set_last_error("gemma4 draft: bad safetensors header length");
        return false;
    }
    const char * header_ptr = (const char *)mm.addr + 8;
    GStMap st;
    if (!parse_gst_header(header_ptr, (size_t)header_len, st)) {
        set_last_error("gemma4 draft: safetensors JSON parse failed");
        return false;
    }
    const uint8_t * blob      = (const uint8_t *)mm.addr + 8 + header_len;
    const size_t    blob_size = mm.len - 8 - (size_t)header_len;

    // ── 4. Infer draft dimensions from FC weight shape ───────────────
    //   fc: [n_vocab_or_target_feat_in, draft_hidden]
    //   The FC input is 6*target_hidden; FC output is draft_hidden.
    //   HF shape in safetensors: [draft_hidden, 6*target_hidden]
    {
        auto it = st.find("fc.weight");
        if (it == st.end()) {
            set_last_error("gemma4 draft: fc.weight not found");
            return false;
        }
        const GStEntry & e = it->second;
        if (e.shape.size() != 2) {
            set_last_error("gemma4 draft: model.fc.weight expected 2D");
            return false;
        }
        // HF stores as [out_features, in_features] = [draft_hidden, 6*target_hidden]
        out.n_embd        = (int)e.shape[0];
        int fc_in         = (int)e.shape[1];
        out.target_hidden = fc_in / GEMMA4_DRAFT_N_TARGET_LAYERS;
        if (fc_in % GEMMA4_DRAFT_N_TARGET_LAYERS != 0) {
            char buf[128];
            std::snprintf(buf, sizeof(buf),
                "gemma4 draft: FC input %d not divisible by n_target_layers %d",
                fc_in, GEMMA4_DRAFT_N_TARGET_LAYERS);
            set_last_error(buf);
            return false;
        }
    }

    // Infer n_head / n_head_kv / n_ff from layer 0 weight shapes
    {
        auto iq = st.find("layers.0.self_attn.q_proj.weight");
        auto ik = st.find("layers.0.self_attn.k_proj.weight");
        auto ig = st.find("layers.0.mlp.gate_proj.weight");
        if (iq == st.end() || ik == st.end() || ig == st.end()) {
            set_last_error("gemma4 draft: missing required layer-0 weight tensors");
            return false;
        }
        // q_proj HF shape: [q_dim, n_embd] where q_dim = n_head * head_dim
        int q_dim = (int)iq->second.shape[0];
        int kv_dim = (int)ik->second.shape[0];
        out.n_head    = q_dim  / out.head_dim;
        out.n_head_kv = kv_dim / out.head_dim;
        out.n_ff      = (int)ig->second.shape[0];
        // Also set layer_is_swa: layers [0..n_layer-2] are SWA, last is full
        out.layer_is_swa.assign((size_t)out.n_layer, true);
        out.layer_is_swa[(size_t)(out.n_layer - 1)] = false;
    }

    const int64_t HIDDEN  = out.n_embd;
    const int64_t Q_DIM   = (int64_t)out.n_head    * out.head_dim;
    const int64_t KV_DIM  = (int64_t)out.n_head_kv * out.head_dim;
    const int64_t INTER   = out.n_ff;
    const int64_t HD      = out.head_dim;
    const int64_t FC_IN   = (int64_t)GEMMA4_DRAFT_N_TARGET_LAYERS * out.target_hidden;
    // VOCAB not used here; tok_embd is injected at runtime from the target model.

    // ── 5. Allocate ggml context ─────────────────────────────────────
    //   tensors: fc, hidden_norm, out_norm = 3 top-level (tok_embd injected at runtime)
    //            11 tensors × 5 layers = 55
    //   total = 58 + headroom
    const int n_tensors = 3 + 11 * out.n_layer + 8;
    ggml_init_params ip{};
    ip.mem_size   = (size_t)n_tensors * ggml_tensor_overhead();
    ip.mem_buffer = nullptr;
    ip.no_alloc   = true;
    out.ctx = ggml_init(ip);
    if (!out.ctx) { set_last_error("gemma4 draft: ggml_init failed"); return false; }
    out.backend = backend;
    out.layers.assign((size_t)out.n_layer, GemmaDraftLayer{});

    const ggml_type NORM_GT = GGML_TYPE_F32;
    const bool      nbf16   = g_cuda_has_native_bf16();
    const ggml_type PROJ_GT = nbf16 ? GGML_TYPE_COUNT : GGML_TYPE_F16;

    // ── 6. Create named tensors ──────────────────────────────────────
    out.fc          = galloc_tensor(out.ctx, st, "fc.weight",          {HIDDEN, FC_IN}, PROJ_GT);
    out.hidden_norm = galloc_tensor(out.ctx, st, "hidden_norm.weight", {HIDDEN},        NORM_GT);
    out.out_norm    = galloc_tensor(out.ctx, st, "norm.weight",        {HIDDEN},        NORM_GT);
    // tok_embd is not present in the draft safetensors; the draft shares
    // the target model's token embedding which is injected at runtime.
    out.tok_embd    = nullptr;
    if (!out.fc || !out.hidden_norm || !out.out_norm) return false;

    for (int il = 0; il < out.n_layer; il++) {
        char pfx[64];
        std::snprintf(pfx, sizeof(pfx), "layers.%d.", il);
        std::string p = pfx;
        GemmaDraftLayer & L = out.layers[(size_t)il];

        L.attn_norm = galloc_tensor(out.ctx, st, p + "input_layernorm.weight",          {HIDDEN},       NORM_GT);
        L.ffn_norm  = galloc_tensor(out.ctx, st, p + "post_attention_layernorm.weight", {HIDDEN},       NORM_GT);
        L.wq        = galloc_tensor(out.ctx, st, p + "self_attn.q_proj.weight",         {Q_DIM,  HIDDEN}, PROJ_GT);
        L.wk        = galloc_tensor(out.ctx, st, p + "self_attn.k_proj.weight",         {KV_DIM, HIDDEN}, PROJ_GT);
        L.wv        = galloc_tensor(out.ctx, st, p + "self_attn.v_proj.weight",         {KV_DIM, HIDDEN}, PROJ_GT);
        L.wo        = galloc_tensor(out.ctx, st, p + "self_attn.o_proj.weight",         {HIDDEN, Q_DIM},  PROJ_GT);
        L.q_norm    = galloc_tensor(out.ctx, st, p + "self_attn.q_norm.weight",         {HD},             NORM_GT);
        L.k_norm    = galloc_tensor(out.ctx, st, p + "self_attn.k_norm.weight",         {HD},             NORM_GT);
        L.w_gate    = galloc_tensor(out.ctx, st, p + "mlp.gate_proj.weight",            {INTER,  HIDDEN}, PROJ_GT);
        L.w_up      = galloc_tensor(out.ctx, st, p + "mlp.up_proj.weight",              {INTER,  HIDDEN}, PROJ_GT);
        L.w_down    = galloc_tensor(out.ctx, st, p + "mlp.down_proj.weight",            {HIDDEN, INTER},  PROJ_GT);

        if (!L.attn_norm || !L.ffn_norm || !L.wq || !L.wk || !L.wv || !L.wo ||
            !L.q_norm || !L.k_norm || !L.w_gate || !L.w_up || !L.w_down) {
            return false;
        }
    }

    // ── 7. Allocate backend buffer and upload bytes ──────────────────
    out.buf = ggml_backend_alloc_ctx_tensors(out.ctx, backend);
    if (!out.buf) {
        set_last_error("gemma4 draft: ggml_backend_alloc_ctx_tensors failed");
        return false;
    }

    std::vector<float>    scratch_f32;
    std::vector<uint16_t> scratch_f16;

    for (ggml_tensor * t = ggml_get_first_tensor(out.ctx); t != nullptr;
         t = ggml_get_next_tensor(out.ctx, t))
    {
        const char * name = ggml_get_name(t);
        auto it = st.find(name);
        if (it == st.end()) {
            set_last_error(std::string("gemma4 draft post-alloc: '") +
                           name + "' vanished from header");
            return false;
        }
        const GStEntry & e = it->second;
        if (e.data_end > (uint64_t)blob_size) {
            set_last_error(std::string("gemma4 draft: data offset out of bounds for '") +
                           name + "'");
            return false;
        }
        const size_t src_bytes = (size_t)(e.data_end - e.data_start);
        const size_t dst_bytes = ggml_nbytes(t);
        const bool same = (t->type == gst_dtype_to_ggml(e.dtype));

        if (same) {
            if (src_bytes != dst_bytes) {
                char buf[256];
                std::snprintf(buf, sizeof(buf),
                    "gemma4 draft: byte mismatch for '%s': blob=%zu ggml=%zu",
                    name, src_bytes, dst_bytes);
                set_last_error(buf);
                return false;
            }
            ggml_backend_tensor_set(t, blob + e.data_start, 0, dst_bytes);
        } else if (e.dtype == "BF16" && t->type == GGML_TYPE_F32) {
            const size_t n = ggml_nelements(t);
            if (src_bytes != n * 2 || dst_bytes != n * 4) {
                set_last_error(std::string("gemma4 draft: BF16->F32 size mismatch for '") + name + "'");
                return false;
            }
            scratch_f32.resize(n);
            g_bf16_to_f32((const uint16_t *)(blob + e.data_start),
                          scratch_f32.data(), n);
            ggml_backend_tensor_set(t, scratch_f32.data(), 0, dst_bytes);
        } else if (e.dtype == "BF16" && t->type == GGML_TYPE_F16) {
            const size_t n = ggml_nelements(t);
            if (src_bytes != n * 2 || dst_bytes != n * 2) {
                set_last_error(std::string("gemma4 draft: BF16->F16 size mismatch for '") + name + "'");
                return false;
            }
            scratch_f16.resize(n);
            g_bf16_to_f16((const uint16_t *)(blob + e.data_start),
                          scratch_f16.data(), n);
            ggml_backend_tensor_set(t, scratch_f16.data(), 0, dst_bytes);
        } else {
            set_last_error(std::string("gemma4 draft: unsupported dtype conversion for '") +
                           name + "': " + e.dtype + " -> " + ggml_type_name(t->type));
            return false;
        }
    }

    std::fprintf(stderr,
        "[gemma4 draft] loaded: n_layer=%d n_head=%d n_kv=%d "
        "n_embd=%d n_ff=%d head_dim=%d target_hidden=%d vocab=%d\n",
        out.n_layer, out.n_head, out.n_head_kv,
        out.n_embd, out.n_ff, out.head_dim, out.target_hidden, out.n_vocab);
    std::fflush(stderr);

    return true;
}

bool load_gemma4_draft_gguf(const std::string & path,
                            ggml_backend_t       backend,
                            GemmaDraftWeights &  out)
{
    // ── 1. Parse metadata + create ggml_context with tensor descriptors ──
    ggml_context * meta_ctx = nullptr;
    gguf_init_params gip{};
    gip.no_alloc = true;
    gip.ctx      = &meta_ctx;
    gguf_context * gctx = gguf_init_from_file(path.c_str(), gip);
    if (!gctx) {
        set_last_error("gguf_init_from_file failed: " + path);
        return false;
    }

    // Validate arch
    {
        int64_t arch_id = gguf_find_key(gctx, "general.architecture");
        if (arch_id < 0) {
            set_last_error("gemma4 draft GGUF: missing general.architecture");
            gguf_free(gctx);
            return false;
        }
        const char * arch = gguf_get_val_str(gctx, arch_id);
        if (std::string(arch) != "gemma4-dflash-draft") {
            set_last_error(std::string("gemma4 draft GGUF: unexpected arch: ") + arch +
                           " (expected gemma4-dflash-draft)");
            gguf_free(gctx);
            return false;
        }
    }

    // Read dimensions from GGUF metadata
    int64_t arch_id2 = gguf_find_key(gctx, "general.architecture");
    const char * A   = gguf_get_val_str(gctx, arch_id2);
    char key[256];

    auto read_u32 = [&](const char * suffix, uint32_t fallback) -> uint32_t {
        std::snprintf(key, sizeof(key), "%s.%s", A, suffix);
        return get_u32_or(gctx, key, fallback);
    };
    auto read_f32 = [&](const char * suffix, float fallback) -> float {
        std::snprintf(key, sizeof(key), "%s.%s", A, suffix);
        return get_f32_or(gctx, key, fallback);
    };

    const uint32_t n_embd       = read_u32("embedding_length",        0);
    const uint32_t n_layer      = read_u32("block_count",             0);
    const uint32_t n_ff         = read_u32("feed_forward_length",     0);
    const uint32_t n_head       = read_u32("attention.head_count",    0);
    const uint32_t n_head_kv    = read_u32("attention.head_count_kv", 0);
    const uint32_t head_dim     = read_u32("attention.key_length",    0);
    const uint32_t block_sz     = read_u32("dflash.block_size",       0);
    const uint32_t n_tgt_lay    = read_u32("dflash.n_target_layers",  0);
    const uint32_t target_hid   = read_u32("dflash.target_hidden",    0);
    const uint32_t mask_tok_id  = read_u32("dflash.mask_token_id",    GEMMA4_31B_DRAFT_MASK_TOKEN_ID);
    const uint32_t sliding_win  = read_u32("dflash.sliding_window",   2048);
    const float    logit_cap    = read_f32("dflash.logit_softcap",    GEMMA4_LOGIT_SOFTCAP);
    const float    rope_theta   = read_f32("rope.freq_base",          GEMMA4_ROPE_THETA);

    if (n_embd == 0 || n_layer == 0 || n_ff == 0 || n_head == 0 ||
        n_head_kv == 0 || head_dim == 0) {
        char buf[256];
        std::snprintf(buf, sizeof(buf),
            "gemma4 draft GGUF: missing hparams: n_embd=%u n_layer=%u n_ff=%u "
            "n_head=%u n_head_kv=%u head_dim=%u",
            n_embd, n_layer, n_ff, n_head, n_head_kv, head_dim);
        set_last_error(buf);
        gguf_free(gctx);
        return false;
    }

    // Validate block_size and n_target_layers match compiled constants
    if (block_sz != (uint32_t)GEMMA4_DRAFT_BLOCK_SIZE ||
        n_tgt_lay != (uint32_t)GEMMA4_DRAFT_N_TARGET_LAYERS) {
        char buf[256];
        std::snprintf(buf, sizeof(buf),
            "gemma4 draft GGUF: dflash.block_size=%u (expected %d), "
            "dflash.n_target_layers=%u (expected %d)",
            block_sz, GEMMA4_DRAFT_BLOCK_SIZE,
            n_tgt_lay, GEMMA4_DRAFT_N_TARGET_LAYERS);
        set_last_error(buf);
        gguf_free(gctx);
        return false;
    }

    // Sanity-check upper bounds
    constexpr uint32_t MAX_LAYERS  = 1024;
    constexpr uint32_t MAX_EMBD    = 1u << 17;
    constexpr uint32_t MAX_FF      = 1u << 19;
    constexpr uint32_t MAX_HEADS   = 1024;
    constexpr uint32_t MAX_HEADDIM = 1024;
    if (n_layer   > MAX_LAYERS  || n_embd    > MAX_EMBD  ||
        n_ff      > MAX_FF      || n_head    > MAX_HEADS ||
        n_head_kv > MAX_HEADS   || head_dim  > MAX_HEADDIM ||
        n_head_kv > n_head      || (n_head % n_head_kv) != 0) {
        char buf[320];
        std::snprintf(buf, sizeof(buf),
            "gemma4 draft GGUF: hparams out of range: n_embd=%u n_layer=%u n_ff=%u "
            "n_head=%u n_head_kv=%u head_dim=%u",
            n_embd, n_layer, n_ff, n_head, n_head_kv, head_dim);
        set_last_error(buf);
        gguf_free(gctx);
        return false;
    }

    // ── 2. Populate GemmaDraftWeights scalars ────────────────────────────
    out.ctx           = meta_ctx;
    out.backend       = backend;
    out.n_layer       = (int)n_layer;
    out.n_head        = (int)n_head;
    out.n_head_kv     = (int)n_head_kv;
    out.head_dim      = (int)head_dim;
    out.n_embd        = (int)n_embd;
    out.n_ff          = (int)n_ff;
    out.block_size    = (int)block_sz;
    out.n_target_layers = (int)n_tgt_lay;
    out.target_hidden = (int)target_hid;
    out.mask_token_id = (int)mask_tok_id;
    out.sliding_window = (int)sliding_win;
    out.logit_softcap = logit_cap;
    out.rope_theta    = rope_theta;

    // layers [0..n_layer-2] are SWA, last layer is full attention
    out.layer_is_swa.assign((size_t)n_layer, true);
    out.layer_is_swa[(size_t)(n_layer - 1)] = false;

    out.layers.assign((size_t)n_layer, GemmaDraftLayer{});

    // tok_embd is injected at runtime from the target model (same as safetensors path)
    out.tok_embd = nullptr;

    // ── 3. Wire tensor pointers ──────────────────────────────────────────
    auto g = [&](const char * name) -> ggml_tensor * {
        return ggml_get_tensor(meta_ctx, name);
    };

    out.fc          = g("dflash.fc.weight");
    out.hidden_norm = g("dflash.hidden_norm.weight");
    out.out_norm    = g("output_norm.weight");

    if (!out.fc || !out.hidden_norm || !out.out_norm) {
        set_last_error("gemma4 draft GGUF: missing top-level tensors "
                       "(dflash.fc.weight / dflash.hidden_norm.weight / output_norm.weight)");
        gguf_free(gctx);
        return false;
    }

    for (int il = 0; il < out.n_layer; il++) {
        char name[128];
        auto fnd = [&](const char * suffix) -> ggml_tensor * {
            std::snprintf(name, sizeof(name), "blk.%d.%s", il, suffix);
            return ggml_get_tensor(meta_ctx, name);
        };
        GemmaDraftLayer & L = out.layers[il];
        L.attn_norm = fnd("attn_norm.weight");
        L.ffn_norm  = fnd("ffn_norm.weight");
        L.wq        = fnd("attn_q.weight");
        L.wk        = fnd("attn_k.weight");
        L.wv        = fnd("attn_v.weight");
        L.wo        = fnd("attn_output.weight");
        L.q_norm    = fnd("attn_q_norm.weight");
        L.k_norm    = fnd("attn_k_norm.weight");
        L.w_gate    = fnd("ffn_gate.weight");
        L.w_up      = fnd("ffn_up.weight");
        L.w_down    = fnd("ffn_down.weight");
        if (!L.attn_norm || !L.ffn_norm || !L.wq || !L.wk || !L.wv || !L.wo ||
            !L.q_norm || !L.k_norm || !L.w_gate || !L.w_up || !L.w_down) {
            char b[128];
            std::snprintf(b, sizeof(b),
                "gemma4 draft GGUF: layer %d missing tensors", il);
            set_last_error(b);
            gguf_free(gctx);
            return false;
        }
    }

    // ── 4. Allocate backend buffer for all tensors ───────────────────────
    out.buf = ggml_backend_alloc_ctx_tensors(meta_ctx, backend);
    if (!out.buf) {
        set_last_error("gemma4 draft GGUF: ggml_backend_alloc_ctx_tensors failed");
        gguf_free(gctx);
        return false;
    }

    // ── 5. mmap file and copy tensor bytes to backend ────────────────────
    std::string err;
    GMmap mm;
    if (!mm.open_ro(path, err)) { set_last_error(err); gguf_free(gctx); return false; }
    const size_t  data_start = gguf_get_data_offset(gctx);
    const int64_t n_tensors  = gguf_get_n_tensors(gctx);

    size_t total = 0;
    for (int64_t tid = 0; tid < n_tensors; tid++) {
        const char * tname = gguf_get_tensor_name(gctx, tid);
        ggml_tensor * t = ggml_get_tensor(meta_ctx, tname);
        if (!t) continue;
        const size_t off = data_start + gguf_get_tensor_offset(gctx, tid);
        const size_t sz  = gguf_get_tensor_size(gctx, tid);
        if (off + sz > mm.len) {
            set_last_error(std::string("gemma4 draft GGUF: tensor '") +
                           tname + "' overflows file");
            gguf_free(gctx);
            return false;
        }
        ggml_backend_tensor_set(t, (const uint8_t *)mm.addr + off, 0, sz);
        total += sz;
    }

    gguf_free(gctx);

    std::fprintf(stderr,
        "[gemma4 draft GGUF] loaded: n_layer=%d n_head=%d n_kv=%d "
        "n_embd=%d n_ff=%d head_dim=%d target_hidden=%d  (%.2f GiB on GPU)\n",
        out.n_layer, out.n_head, out.n_head_kv,
        out.n_embd, out.n_ff, out.head_dim, out.target_hidden,
        total / (1024.0 * 1024.0 * 1024.0));
    std::fflush(stderr);

    return true;
}

void free_gemma4_draft_weights(GemmaDraftWeights & w) {
    if (w.buf) { ggml_backend_buffer_free(w.buf); w.buf = nullptr; }
    if (w.ctx) { ggml_free(w.ctx);                w.ctx = nullptr; }
    w.layers.clear();
    w.layer_is_swa.clear();
    w.fc          = nullptr;
    w.hidden_norm = nullptr;
    w.out_norm    = nullptr;
    w.tok_embd    = nullptr;
}

} // namespace dflash27b
