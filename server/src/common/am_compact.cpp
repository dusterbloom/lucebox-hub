#include "common/am_compact.h"

#include "ggml-alloc.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <vector>

namespace luce::am {

namespace {

// strtod / strtol over the whole string; false on garbage or trailing characters.
bool parse_double(const char * s, double & out) {
    char * end = nullptr;
    out = std::strtod(s, &end);
    return end != s && *end == '\0';
}
bool parse_nonneg_int(const char * s, int & out) {
    char * end = nullptr;
    const long v = std::strtol(s, &end, 10);
    if (end == s || *end != '\0' || v < 0 || v > 1000000) return false;
    out = (int) v;
    return true;
}
bool empty(const char * s) { return !s || !*s; }

double ms_since(std::chrono::steady_clock::time_point t0) {
    return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
}

}  // namespace

AutoConfig auto_config_from_strings(const char * ratio, const char * prefix_blocks,
                                    const char * recent_blocks, std::string * err) {
    AutoConfig c;
    auto fail = [&](const std::string & m) {
        if (err) *err = m;
        return AutoConfig{};
    };
    if (!empty(ratio)) {
        double r = 0;
        if (!parse_double(ratio, r) || !std::isfinite(r) || r < 0.0)
            return fail(std::string("--kvflash-am-ratio expects a number >= 0, got '") + ratio + "'");
        const double supported = (double) kAutoSlots / (double) kAutoBlockTokens;
        if (r != 0.0 && std::fabs(r - supported) > 1e-6)
            return fail(std::string("--kvflash-am-ratio ") + ratio + " is not supported: a compacted block is one " +
                        std::to_string(kAutoSlots) + "-row chunk standing for " + std::to_string(kAutoBlockTokens) +
                        " tokens, so the only ratio is 0.10");
        c.ratio = r;
    }
    if (!empty(prefix_blocks) && !parse_nonneg_int(prefix_blocks, c.prefix_blocks))
        return fail(std::string("--kvflash-am-prefix-blocks expects an integer >= 0, got '") + prefix_blocks + "'");
    if (!empty(recent_blocks) && !parse_nonneg_int(recent_blocks, c.recent_blocks))
        return fail(std::string("--kvflash-am-recent-blocks expects an integer >= 0, got '") + recent_blocks + "'");
    return c;
}

bool auto_active(const AutoConfig & cfg, bool kvflash_active, bool arch_supported) {
    return cfg.enabled() && kvflash_active && arch_supported;
}

bool should_compact(const AutoConfig & cfg, const SealInfo & s) {
    if (!cfg.enabled() || s.block < 0) return false;
    const int64_t block_end = ((int64_t) s.block + 1) * kAutoBlockTokens;
    if (block_end > s.prompt_len) return false;                 // not a sealed block of this prompt
    if (s.block < cfg.prefix_blocks) return false;              // exact prefix
    const int n_sealed = s.prompt_len / kAutoBlockTokens;
    if (cfg.recent_blocks > 0 && s.block >= n_sealed - cfg.recent_blocks) return false;   // exact recent window
    if (s.exact_tail_tokens > 0 && block_end > (int64_t) s.prompt_len - s.exact_tail_tokens) return false;
    return !s.pinned;
}

int64_t auto_reserve_bytes(int64_t ring_bytes, int head_dim, int n_head_kv) {
    const int64_t block_f32 = (int64_t) kAutoBlockTokens * head_dim * n_head_kv * 4;   // K or V of one layer
    constexpr int64_t kFitScratch = 32LL * 1024 * 1024;   // score + Gram graphs, one layer at a time
    return ring_bytes + 2 * block_f32 + kFitScratch;
}

bool compact_block(ggml_backend_t backend, luce_am::GramAccumulator & acc, ggml_tensor * q, ggml_tensor * k,
                   ggml_tensor * v, int slots, const luce_am::NnlsParams & params, BlockFit & out, std::string * err) {
    auto fail = [&](const std::string & m) {
        if (err) *err = m;
        return false;
    };
    if (!backend || !q || !k || !v) return fail("null argument");
    const int d = (int) q->ne[0], n_tokens = (int) q->ne[1], n_head = (int) q->ne[2];
    const int n_keys = (int) k->ne[1], n_kv = (int) k->ne[2];
    if (slots < 1 || slots > n_keys) return fail("slots must be in [1, n_keys]");
    if (k->ne[0] != d || v->ne[0] != d || v->ne[1] != n_keys || v->ne[2] != n_kv) return fail("q, k, v shapes disagree");
    out = BlockFit{};
    const auto t0 = std::chrono::steady_clock::now();
    const int chunk_rows = std::max(n_head, n_head * 128);   // 128 query rows of every head per softmax chunk
    if (!select_keys(backend, q, k, slots, chunk_rows, out.sel, err)) return false;
    sort_sel_ascending(out.sel, n_kv, slots);
    out.ms_select = ms_since(t0);

    luce_am::BlockShape shape;
    shape.head_dim = d;
    shape.n_tokens = n_tokens;
    shape.n_head = n_head;
    shape.n_head_kv = n_kv;
    shape.n_keys = n_keys;
    shape.t = slots;
    shape.scale = 1.0f / std::sqrt((float) d);
    const auto t1 = std::chrono::steady_clock::now();
    luce_am::BlockTensors bt;
    bt.q = q;
    bt.k = k;
    bt.v = v;
    bt.sel = out.sel.data();
    if (!luce_am::fit_block(acc, backend, shape, bt, params, out.fit, err)) {
        out.sel.clear();
        return false;
    }
    out.ms_fit = ms_since(t1);
    return true;
}

bool compact_block_host(ggml_backend_t backend, luce_am::GramAccumulator & acc, int head_dim, int n_tokens, int n_head,
                        int n_head_kv, int n_keys, int slots, const ggml_fp16_t * q, const float * k, const float * v,
                        const luce_am::NnlsParams & params, BlockFit & out, std::string * err) {
    ggml_init_params ip{};
    ip.mem_size = 8 * ggml_tensor_overhead();
    ip.no_alloc = true;
    std::unique_ptr<ggml_context, void (*)(ggml_context *)> ctx(ggml_init(ip), ggml_free);
    if (!ctx) {
        if (err) *err = "ggml_init failed";
        return false;
    }
    ggml_tensor * tq = ggml_new_tensor_3d(ctx.get(), GGML_TYPE_F16, head_dim, n_tokens, n_head);
    ggml_tensor * tk = ggml_new_tensor_3d(ctx.get(), GGML_TYPE_F32, head_dim, n_keys, n_head_kv);
    ggml_tensor * tv = ggml_new_tensor_3d(ctx.get(), GGML_TYPE_F32, head_dim, n_keys, n_head_kv);
    std::unique_ptr<ggml_backend_buffer, void (*)(ggml_backend_buffer_t)> buf(
        ggml_backend_alloc_ctx_tensors(ctx.get(), backend), ggml_backend_buffer_free);
    if (!buf) {
        if (err) *err = "device allocation failed";
        return false;
    }
    ggml_backend_tensor_set(tq, q, 0, ggml_nbytes(tq));
    ggml_backend_tensor_set(tk, k, 0, ggml_nbytes(tk));
    ggml_backend_tensor_set(tv, v, 0, ggml_nbytes(tv));
    return compact_block(backend, acc, tq, tk, tv, slots, params, out, err);
}

void sort_sel_ascending(std::vector<int32_t> & sel, int n_head_kv, int slots) {
    for (int h = 0; h < n_head_kv; h++)
        std::sort(sel.begin() + (size_t) h * slots, sel.begin() + (size_t) (h + 1) * slots);
}

void gather_rows(const uint8_t * rows, size_t row_bytes, const int32_t * sel, int slots, uint8_t * out) {
    for (int s = 0; s < slots; s++)
        std::memcpy(out + (size_t) s * row_bytes, rows + (size_t) sel[s] * row_bytes, row_bytes);
}


bool spec_decode_allowed(bool draft_ready, bool compacted) { return draft_ready && !compacted; }

bool restore_needs_fresh_prefill(bool auto_on, int snap_pos, int prompt_len) {
    return auto_on && prompt_len > snap_pos;
}

bool kv_bias_wanted(bool kvflash_active, bool paged, int n_slots, const AutoConfig & cfg, bool test_hook) {
    return kvflash_active && !paged && n_slots == 1 && (cfg.enabled() || test_hook);
}

// 10 chunks of the block, the 64-row sink chunk, and 5 chunks of slack for the protected tail.
int auto_min_pool_tokens() { return kAutoBlockTokens + 6 * kAutoSlots; }
bool pool_fits_block(int pool_tokens) { return pool_tokens >= auto_min_pool_tokens(); }

bool rows_finite(ggml_type type, const uint8_t * rows, int64_t n_rows, int d) {
    const ggml_type_traits * tr = ggml_get_type_traits(type);
    std::vector<float> f((size_t) n_rows * d);
    tr->to_float(rows, f.data(), n_rows * d);
    for (float x : f) if (!std::isfinite(x)) return false;
    return true;
}

}  // namespace luce::am
