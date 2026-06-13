// test_kvflash — verifies KVFlash, the bounded-resident-pool KV cache
// (kvflash_pager.h).
//
// Runs against one loaded qwen35 target:
//
//   A  baseline: cache at LOGICAL context (default 131072), maskless decode
//      (production AR path shape). Reference tokens + baseline KV memory.
//   B  relocation proof: small pool, chunks at SHUFFLED physical blocks,
//      explicit pool slot mask, teacher-forced replay of A. Argmax must
//      track A (position-independence + mask exactness).
//   C  paging proof: pool ≪ prompt+gen, live eviction, bit-exact
//      page_out/page_in roundtrip, KV bytes vs A.
//   D  reselect/recall: evicted chunk recalled via score_hook + reselect()
//      (the FlashMemory τ-step lookahead machinery); decode continues.
//   E  performance profile: decode ms/step vs FA span — baseline at
//      8K/32K/128K vs pool 1K/4K at 128K-logical — plus page-event and
//      mask-refill microbenchmarks.
//   G  prefix-cache compose: serialize/deserialize is byte-lossless across
//      a save->restore at a different pool size / block order (G1), and a
//      restored, relocated prefix continues coherently — argmax tracks A
//      within the relocation bound (G2). Pins KVFlash + prefix snapshots.
//
// Usage:
//   test_kvflash <qwen35.gguf> [--logical-ctx=N] [--pool-b=N] [--pool-c=N]
//                 [--prompt=N] [--gen=N] [--skip-profile] [--no-mask]
//   modes: (default) verification suite A-F | --niah | --niah256 | --longab

#include "dflash27b.h"
#include "internal.h"
#include "kvflash_pager.h"
#include "kvflash_qk.h"
#include "attn_masks.h"
#include "qwen3_drafter.h"
#include "qwen3_kvflash_scorer.h"

#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-cuda.h"

#include <cinttypes>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

using namespace dflash::common;

namespace {

double now_ms() {
    return std::chrono::duration<double, std::milli>(
        std::chrono::steady_clock::now().time_since_epoch()).count();
}

size_t kv_cache_bytes(const TargetCache & c) {
    size_t n = 0;
    for (auto * t : c.attn_k) if (t) n += ggml_nbytes(t);
    for (auto * t : c.attn_v) if (t) n += ggml_nbytes(t);
    return n;
}

size_t vram_used_now() {
    size_t free_b = 0, total_b = 0;
    ggml_backend_cuda_get_device_memory(0, &free_b, &total_b);
    return total_b - free_b;
}

// Single-token stepper over build_qwen35_graph with explicit control of:
//   * kv_write_rows  — physical pool slot for the KV append
//   * positions      — logical position (M-RoPE)
//   * span           — FA window length (kv_start = span-1 in graph terms)
//   * attn_mask      — optional [align32(span_padded), 32] f16 slot mask
//
// The graph arena and gallocr persist across rebuilds (same trick as
// build_target_step) so identical topology lands at identical addresses
// and the ggml-cuda CUDA-graph cache can replay decode steps.
struct Stepper {
    ggml_context *  ctx = nullptr;
    ggml_cgraph *   gf  = nullptr;
    ggml_gallocr_t  alloc = nullptr;
    ggml_tensor *   inp_embed = nullptr;
    ggml_tensor *   positions = nullptr;
    ggml_tensor *   attn_mask = nullptr;
    ggml_tensor *   kv_write_rows = nullptr;
    ggml_tensor *   logits = nullptr;
    ggml_tensor *   argmax_tokens = nullptr;

    const TargetWeights * w = nullptr;
    TargetCache * cache = nullptr;
    ggml_backend_t backend = nullptr;
    int span = 0;
    bool with_mask = false;
    bool q_capture = false;   // write last token's per-layer post-RoPE Q to cache.q_cap

    std::vector<uint8_t> arena;
    std::vector<float> embed_buf;
    std::vector<uint16_t> mask_buf;
    uint64_t mask_epoch = (uint64_t)-1;
    double mask_fill_ms_total = 0.0;
    int mask_fills = 0;

    bool init(const TargetWeights & tw, TargetCache & tc, ggml_backend_t be,
              int span_, bool with_mask_, bool q_capture_ = false) {
        w = &tw; cache = &tc; backend = be;
        span = span_; with_mask = with_mask_; q_capture = q_capture_;
        embed_buf.resize(tw.n_embd);
        arena.resize((size_t)512 * 1024 * 1024);
        return build();
    }

    bool build() {
        if (ctx) { ggml_free(ctx); ctx = nullptr; }
        ggml_init_params ip{};
        ip.mem_size   = arena.size();
        ip.mem_buffer = arena.data();
        ip.no_alloc   = true;
        ctx = ggml_init(ip);
        if (!ctx) return false;

        const int hidden = w->n_embd;
        inp_embed = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, hidden, 1, 1);
        ggml_set_input(inp_embed);
        positions = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, 4);
        ggml_set_input(positions);
        kv_write_rows = ggml_new_tensor_2d(ctx, GGML_TYPE_I64, 1, w->n_head_kv);
        ggml_set_input(kv_write_rows);

        attn_mask = nullptr;
        if (with_mask) {
            // FA span is padded to 256 on the step-invariant path; the mask
            // kv dim must cover it.
            const int span_padded = std::min(((span + 255) / 256) * 256,
                                             (int)cache->attn_k[0]->ne[1]);
            attn_mask = ggml_new_tensor_2d(ctx, GGML_TYPE_F16,
                                           align_up(span_padded, KQ_MASK_PAD),
                                           align_up(1, KQ_MASK_PAD));
            ggml_set_input(attn_mask);
            mask_buf.assign((size_t)attn_mask->ne[0] * attn_mask->ne[1], F16_NEG_INF);
            mask_epoch = (uint64_t)-1;
        }

        gf = ggml_new_graph_custom(ctx, 16384, false);

        QwenGraphInputs gi{};
        gi.inp_embed      = inp_embed;
        gi.positions      = positions;
        gi.attn_mask      = attn_mask;
        gi.n_tokens       = 1;
        gi.kv_start       = span - 1;
        gi.capture_layers = false;
        gi.kv_write_rows  = kv_write_rows;
        gi.q_capture      = q_capture;

        QwenGraphOutputs go = build_qwen35_graph(ctx, gf, *w, *cache, gi);
        if (!go.logits) return false;
        logits = go.logits;
        ggml_set_output(logits);
        argmax_tokens = ggml_argmax(ctx, logits);
        ggml_set_output(argmax_tokens);
        ggml_build_forward_expand(gf, argmax_tokens);

        if (!alloc) alloc = ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend));
        return ggml_gallocr_alloc_graph(alloc, gf);
    }

    void refresh_mask(const KvFlashPager & pager) {
        if (!attn_mask) return;
        const double t0 = now_ms();
        if (pager.epoch() != mask_epoch) {
            // Host-side rebuild only on residency change.
            std::fill(mask_buf.begin(), mask_buf.end(), F16_NEG_INF);
            pager.fill_slot_mask(mask_buf.data());
            mask_epoch = pager.epoch();
            mask_fills++;
        }
        // Upload EVERY step: the compute-buffer region backing this input
        // tensor is reused by graph execution, so a stale upload reads as
        // garbage (NaN logits) on the next step. Production prefill
        // re-uploads its mask before every compute for the same reason.
        ggml_backend_tensor_set(attn_mask, mask_buf.data(), 0,
                                mask_buf.size() * sizeof(uint16_t));
        mask_fill_ms_total += now_ms() - t0;
    }

    int32_t step(int32_t tok, int pos, int phys_slot) {
        if (!w->embedder.embed(&tok, 1, embed_buf.data())) {
            std::fprintf(stderr, "embed failed: tok=%d pos=%d (NaN logits upstream?)\n", tok, pos);
            std::exit(1);
        }
        ggml_backend_tensor_set(inp_embed, embed_buf.data(), 0,
                                sizeof(float) * embed_buf.size());
        int32_t p4[4] = { pos, pos, pos, 0 };
        ggml_backend_tensor_set(positions, p4, 0, sizeof(int32_t) * 4);
        std::vector<int64_t> rows(w->n_head_kv, (int64_t)phys_slot);
        ggml_backend_tensor_set(kv_write_rows, rows.data(), 0,
                                sizeof(int64_t) * rows.size());
        if (ggml_backend_graph_compute(backend, gf) != GGML_STATUS_SUCCESS) {
            std::fprintf(stderr, "graph_compute failed pos=%d\n", pos);
            std::exit(1);
        }
        int32_t next = 0;
        ggml_backend_tensor_get(argmax_tokens, &next, 0, sizeof(int32_t));
        return next;
    }

    void destroy() {
        if (alloc) { ggml_gallocr_free(alloc); alloc = nullptr; }
        if (ctx)   { ggml_free(ctx); ctx = nullptr; }
    }
};

std::vector<int32_t> make_prompt(int n, int vocab) {
    std::vector<int32_t> p(n);
    uint64_t s = 0x9E3779B97F4A7C15ull;
    // Cap below the drafter vocab too (Qwen3-0.6B ~151K) so the same ids
    // are scoreable by the indexer in run F.
    const int cap = std::min(vocab, 100000);
    for (int i = 0; i < n; i++) {
        s = s * 6364136223846793005ull + 1442695040888963407ull;
        p[i] = (int32_t)(1000 + (s >> 33) % (uint64_t)(cap / 2));
    }
    return p;
}

// Pooled chunked prefill: 64-token (one pager chunk) batched forwards with
// slot-mapped set_rows writes and a resident+causal mask. This is the
// prompt > pool path: prefill evicts like decode does. Graph is built once
// (fixed topology) and reused for every chunk.
struct BatchStepper {
    ggml_context *  ctx = nullptr;
    ggml_cgraph *   gf  = nullptr;
    ggml_gallocr_t  alloc = nullptr;
    ggml_tensor *   inp_embed = nullptr;
    ggml_tensor *   positions = nullptr;
    ggml_tensor *   attn_mask = nullptr;
    ggml_tensor *   kv_write_rows = nullptr;
    ggml_tensor *   logits = nullptr;
    ggml_tensor *   argmax_tokens = nullptr;

    const TargetWeights * w = nullptr;
    TargetCache * cache = nullptr;
    ggml_backend_t backend = nullptr;
    int pool = 0;
    static constexpr int NB = 64;          // tokens per chunk

    std::vector<uint8_t> arena;
    std::vector<float> embed_buf;
    std::vector<uint16_t> mask_buf;

    bool init(const TargetWeights & tw, TargetCache & tc, ggml_backend_t be, int pool_) {
        w = &tw; cache = &tc; backend = be; pool = pool_;
        embed_buf.resize((size_t)tw.n_embd * NB);
        arena.resize((size_t)512 * 1024 * 1024);

        ggml_init_params ip{};
        ip.mem_size   = arena.size();
        ip.mem_buffer = arena.data();
        ip.no_alloc   = true;
        ctx = ggml_init(ip);
        if (!ctx) return false;

        inp_embed = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, tw.n_embd, NB, 1);
        ggml_set_input(inp_embed);
        positions = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, 4 * NB);
        ggml_set_input(positions);
        kv_write_rows = ggml_new_tensor_2d(ctx, GGML_TYPE_I64, NB, tw.n_head_kv);
        ggml_set_input(kv_write_rows);
        attn_mask = ggml_new_tensor_2d(ctx, GGML_TYPE_F16,
                                       align_up(pool, KQ_MASK_PAD),
                                       align_up(NB, KQ_MASK_PAD));
        ggml_set_input(attn_mask);
        mask_buf.assign((size_t)attn_mask->ne[0] * attn_mask->ne[1], F16_NEG_INF);

        gf = ggml_new_graph_custom(ctx, 16384, false);
        QwenGraphInputs gi{};
        gi.inp_embed   = inp_embed;
        gi.positions   = positions;
        gi.attn_mask   = attn_mask;
        gi.n_tokens    = NB;
        gi.kv_start    = pool - NB;      // span = whole pool
        gi.kv_write_rows = kv_write_rows;
        gi.last_token_logits_only = true;
        QwenGraphOutputs go = build_qwen35_graph(ctx, gf, *w, *cache, gi);
        if (!go.logits) return false;
        logits = go.logits;
        ggml_set_output(logits);
        argmax_tokens = ggml_argmax(ctx, logits);
        ggml_set_output(argmax_tokens);
        ggml_build_forward_expand(gf, argmax_tokens);
        alloc = ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend));
        return ggml_gallocr_alloc_graph(alloc, gf);
    }

    // One 64-token chunk at logical [pos_base, pos_base+64). Allocates the
    // chunk's block (evicting if needed), writes slot-mapped, masks
    // resident slots + causal-within-chunk. Returns last-token argmax.
    int32_t step_chunk(const int32_t * toks, int pos_base, KvFlashPager & pager) {
        int slots[NB];
        for (int i = 0; i < NB; i++) slots[i] = pager.slot_for(pos_base + i);

        if (!w->embedder.embed(toks, NB, embed_buf.data())) {
            std::fprintf(stderr, "batch embed failed @%d\n", pos_base);
            std::exit(1);
        }
        ggml_backend_tensor_set(inp_embed, embed_buf.data(), 0,
                                sizeof(float) * embed_buf.size());
        std::vector<int32_t> p4((size_t)4 * NB);
        for (int i = 0; i < NB; i++) {
            p4[4 * i + 0] = p4[4 * i + 1] = p4[4 * i + 2] = pos_base + i;
            p4[4 * i + 3] = 0;
        }
        ggml_backend_tensor_set(positions, p4.data(), 0, sizeof(int32_t) * p4.size());
        // [n_tokens, n_head_kv] ne0-major: (token i, head h) at i + h*NB.
        std::vector<int64_t> rows((size_t)NB * w->n_head_kv);
        for (int h = 0; h < w->n_head_kv; h++) {
            for (int i = 0; i < NB; i++) {
                rows[(size_t)h * NB + i] = slots[i];
            }
        }
        ggml_backend_tensor_set(kv_write_rows, rows.data(), 0,
                                sizeof(int64_t) * rows.size());

        // Mask: per q row, resident slots (excluding this chunk) attendable,
        // this chunk's slots causal. Rebuilt + uploaded per chunk.
        const size_t kvd = (size_t)attn_mask->ne[0];
        std::fill(mask_buf.begin(), mask_buf.end(), F16_NEG_INF);
        pager.fill_slot_mask(mask_buf.data());                    // row 0 base
        const int this_block = slots[0] - slots[0] % NB;
        for (int i = 0; i < NB; i++) mask_buf[(size_t)this_block + i] = F16_NEG_INF;
        for (int q = 1; q < NB; q++) {
            std::memcpy(mask_buf.data() + (size_t)q * kvd, mask_buf.data(), kvd * 2);
        }
        for (int q = 0; q < NB; q++) {
            for (int i = 0; i <= q; i++) {
                mask_buf[(size_t)q * kvd + slots[i]] = F16_ZERO;
            }
        }
        ggml_backend_tensor_set(attn_mask, mask_buf.data(), 0,
                                mask_buf.size() * sizeof(uint16_t));

        if (ggml_backend_graph_compute(backend, gf) != GGML_STATUS_SUCCESS) {
            std::fprintf(stderr, "batch compute failed @%d\n", pos_base);
            std::exit(1);
        }
        int32_t last = 0;
        ggml_backend_tensor_get(argmax_tokens, &last, 0, sizeof(int32_t));
        return last;
    }

    void destroy() {
        if (alloc) { ggml_gallocr_free(alloc); alloc = nullptr; }
        if (ctx)   { ggml_free(ctx); ctx = nullptr; }
    }
};


int arg_int(int argc, char ** argv, const char * key, int defv) {
    const size_t kl = std::strlen(key);
    for (int i = 2; i < argc; i++) {
        if (std::strncmp(argv[i], key, kl) == 0 && argv[i][kl] == '=') {
            return std::atoi(argv[i] + kl + 1);
        }
    }
    return defv;
}

bool arg_flag(int argc, char ** argv, const char * key) {
    for (int i = 2; i < argc; i++) if (std::strcmp(argv[i], key) == 0) return true;
    return false;
}

const char * arg_str(int argc, char ** argv, const char * key, const char * defv) {
    const size_t kl = std::strlen(key);
    for (int i = 2; i < argc; i++) {
        if (std::strncmp(argv[i], key, kl) == 0 && argv[i][kl] == '=') {
            return argv[i] + kl + 1;
        }
    }
    return defv;
}

// Real-content filler: raw little-endian i32 token stream. Ids are folded
// below 100000 (drafter-scoreable, same fold as KvFlashDrafterScorer /
// make_prompt). If the file is shorter than n, the remainder is synthetic.
std::vector<int32_t> load_token_stream(const char * path, int n, int vocab) {
    std::vector<int32_t> out;
    if (std::FILE * f = std::fopen(path, "rb")) {
        std::fseek(f, 0, SEEK_END);
        const long bytes = std::ftell(f);
        std::fseek(f, 0, SEEK_SET);
        out.resize(std::min<long>(bytes / 4, n));
        if (!out.empty() && std::fread(out.data(), 4, out.size(), f) != out.size()) {
            out.clear();
        }
        std::fclose(f);
    }
    const size_t real_n = out.size();
    for (auto & t : out) {
        if (t < 0 || t >= 100000) t = 1000 + (t < 0 ? -t : t) % 99000;
    }
    if ((int)out.size() < n) {
        auto pad = make_prompt(n - (int)out.size(), vocab);
        out.insert(out.end(), pad.begin(), pad.end());
    }
    std::printf("[qkbench] filler: %zu real tokens from %s + %d synthetic\n",
                real_n, path, n - (int)real_n);
    return out;
}

struct StepTimes {
    double p50 = 0, p95 = 0, mean = 0;
};

StepTimes summarize(std::vector<double> & ms) {
    StepTimes r;
    if (ms.empty()) return r;
    std::sort(ms.begin(), ms.end());
    r.p50 = ms[ms.size() / 2];
    r.p95 = ms[(size_t)(ms.size() * 0.95)];
    for (double v : ms) r.mean += v;
    r.mean /= ms.size();
    return r;
}

} // namespace

int main(int argc, char ** argv) {
    if (argc < 2) {
        std::fprintf(stderr, "usage: %s <qwen35.gguf> [--logical-ctx=N] [--pool-b=N] "
                             "[--pool-c=N] [--prompt=N] [--gen=N] [--skip-profile]\n", argv[0]);
        return 2;
    }
    const int logical_ctx = arg_int(argc, argv, "--logical-ctx", 131072);
    const int pool_b      = arg_int(argc, argv, "--pool-b", 2048);
    const int pool_c      = arg_int(argc, argv, "--pool-c", 1024);
    const int n_prompt    = arg_int(argc, argv, "--prompt", 512);
    const int n_gen       = arg_int(argc, argv, "--gen", 1200);
    const bool skip_prof  = arg_flag(argc, argv, "--skip-profile");
    // Explicit pool slot mask: exact exclusion of non-resident slots.
    // ON by default (requires the per-step re-upload in refresh_mask: the
    // mask input's compute-buffer region is clobbered by graph execution).
    // --no-mask falls back to the zero-row approximation production's
    // padded span uses.
    const bool use_mask   = !arg_flag(argc, argv, "--no-mask");
    const int total       = n_prompt + n_gen;
    if (total > pool_b) {
        std::fprintf(stderr, "config error: prompt+gen (%d) must fit pool-b (%d)\n", total, pool_b);
        return 2;
    }

    ggml_backend_t backend = ggml_backend_cuda_init(0);
    if (!backend) { std::fprintf(stderr, "cuda init failed\n"); return 1; }
    const size_t vram0 = vram_used_now();

    TargetWeights w;
    if (!load_target_gguf(argv[1], backend, w)) {
        std::fprintf(stderr, "load: %s\n", dflash27b_last_error());
        return 1;
    }
    std::printf("[load] weights ok, vram_used=%.1f MiB\n",
                (vram_used_now() - vram0) / 1048576.0);

    // ── --qkbench: LRU vs drafter vs target-QK residency policies ───
    // Adopt-vs-build decision bench (optimizations/msa-sm86/PLAN.md pivot):
    // per cell {L, pool, arm} measure prefill wall, one-shot rescore cost,
    // decode tok/s (240-token free run), needle /16 (longab methodology,
    // depth 0.25) and a 4-fact cold-history retrieval probe with decoys
    // (exact-token recall /16 each + whether the gold chunks paged in).
    // Filler is a REAL token stream (--qk-tokens) padded synthetically.
    // Run ONE cell per process (--qk-L/--qk-pool/--qk-arm): the CUDA VMM
    // pool grows monotonically across large-cache configs.
    if (arg_flag(argc, argv, "--qkbench")) {
        const int  L    = arg_int(argc, argv, "--qk-L", 65536);
        const int  pool = arg_int(argc, argv, "--qk-pool", 4096);
        const char * arm = arg_str(argc, argv, "--qk-arm", "qk");   // lru|drafter|qk
        const char * tok_path = arg_str(argc, argv, "--qk-tokens",
            "/tmp/lsa-msa-64k/goldgate_xl.turn10.65536.tokens.i32");
        const bool is_drafter = std::strcmp(arm, "drafter") == 0;
        const bool is_qk      = std::strcmp(arm, "qk") == 0;

        DrafterContext dctx;
        KvFlashDrafterScorer dscorer(&dctx);
        if (is_drafter) {
            const char * dpath = arg_str(argc, argv, "--qk-drafter",
                "/opt/lucebox/models/drafter/Qwen3-0.6B-BF16.gguf");
            if (!load_drafter(dpath, 0, dctx)) {
                std::fprintf(stderr, "drafter load failed\n");
                return 1;
            }
            // Reserve drafter compute buffers before cache churn fragments
            // the CUDA pool (same warmup as --niah).
            std::vector<int32_t> warm(33024, 1234);
            std::vector<float> tmp;
            dscorer.score_chunks(warm, 64, tmp);
        }

        auto prompt = load_token_stream(tok_path, L, w.n_vocab);

        // Needle (longab methodology): 48 seeded tokens at depth 0.25.
        std::vector<int32_t> needle(48);
        {
            uint64_t ns = 0xDEADBEEFCAFEull;
            for (int i = 0; i < 48; i++) {
                ns = ns * 6364136223846793005ull + 1442695040888963407ull;
                needle[i] = (int32_t)(1000 + (ns >> 33) % 49000);
            }
            const int npos = ((int)(0.25 * (L - 512)) / 32) * 32;
            for (int i = 0; i < 48; i++) prompt[npos + i] = needle[i];
        }

        // 4 facts + 4 decoys. A decoy shares the fact's first 16 tokens and
        // diverges after — retrieval must discriminate continuations, not
        // just prefix-match. Gold = the fact's chunks; query = first 32.
        constexpr int NF = 4;
        const double fact_depth[NF]  = {0.10, 0.40, 0.60, 0.85};
        const double decoy_depth[NF] = {0.17, 0.47, 0.67, 0.92};
        std::vector<std::vector<int32_t>> fact(NF), decoy(NF);
        int fact_pos[NF], decoy_pos[NF];
        for (int fidx = 0; fidx < NF; fidx++) {
            uint64_t s = 0xF00DF00DULL + (uint64_t)fidx * 0x9E3779B97F4A7C15ull;
            fact[fidx].resize(48);
            decoy[fidx].resize(48);
            for (int i = 0; i < 48; i++) {
                s = s * 6364136223846793005ull + 1442695040888963407ull;
                fact[fidx][i] = (int32_t)(1000 + (s >> 33) % 49000);
            }
            for (int i = 0; i < 16; i++) decoy[fidx][i] = fact[fidx][i];
            for (int i = 16; i < 48; i++) {
                s = s * 6364136223846793005ull + 1442695040888963407ull;
                decoy[fidx][i] = (int32_t)(1000 + (s >> 33) % 49000);
            }
            fact_pos[fidx]  = ((int)(fact_depth[fidx]  * (L - 512)) / 32) * 32;
            decoy_pos[fidx] = ((int)(decoy_depth[fidx] * (L - 512)) / 32) * 32;
            for (int i = 0; i < 48; i++) prompt[fact_pos[fidx]  + i] = fact[fidx][i];
            for (int i = 0; i < 48; i++) prompt[decoy_pos[fidx] + i] = decoy[fidx][i];
        }

        // ── Cache + pager + (qk) key pool ────────────────────────────
        TargetCache cache;
        if (!create_target_cache(w, pool, 0, backend, cache, true)) return 1;
        KvFlashPager pager;
        KvFlashConfig pc; pc.pool_tokens = pool;
        if (!pager.attach(pc, cache.attn_k, cache.attn_v)) return 1;

        KvFlashQkPool qkpool;
        KvFlashQkDims qd;
        qd.n_layers = (int)cache.attn_k.size();
        qd.n_q_heads = w.n_head; qd.n_kv_heads = w.n_head_kv;
        qd.head_dim = w.n_embd_head_k;
        qkpool.reset(qd);
        KvFlashTargetQkScorer qscorer(&qkpool);

        // ── Prefill (pooled, 64-token chunks); qk pools keys at seal ──
        double t0 = now_ms();
        BatchStepper bs;
        if (!bs.init(w, cache, backend, pool)) return 1;
        for (int p = 0; p < L; p += 64) {
            bs.step_chunk(prompt.data() + p, p, pager);
            if (is_qk) {
                const int c = p / 64;
                if (!qkpool.pool_chunk(cache.attn_k, pager.block_of(c), 64, c)) {
                    std::fprintf(stderr, "[qkbench] pool_chunk failed @chunk %d\n", c);
                    return 1;
                }
            }
        }
        bs.destroy();
        const double prefill_s = (now_ms() - t0) / 1000.0;

        Stepper st;
        if (!st.init(w, cache, backend, pool, /*with_mask=*/true, /*q_capture=*/is_qk)) return 1;
        std::vector<int32_t> hist = prompt;   // drafter arm history
        int cur = L;
        std::vector<float> scores;
        double rescore_s_total = 0.0;
        int    rescore_n = 0;

        auto feed = [&](int32_t tok) -> int32_t {
            const int slot = pager.slot_for(cur);
            st.refresh_mask(pager);
            const int32_t next = st.step(tok, cur, slot);
            hist.push_back(tok);
            cur++;
            // keep qk pooled keys current as decode seals chunks
            if (is_qk && cur % 64 == 0) {
                const int c = cur / 64 - 1;
                if (!qkpool.has(c)) qkpool.pool_chunk(cache.attn_k, pager.block_of(c), 64, c);
            }
            return next;
        };

        // Arm-specific rescore + reselect (timed). LRU arm: no-op.
        auto reselect_arm = [&]() -> double {
            const double r0 = now_ms();
            bool ok = false;
            if (is_drafter) {
                ok = dscorer.score_chunks(hist, pc.chunk_tokens, scores);
            } else if (is_qk) {
                std::vector<float> q((size_t)ggml_nelements(cache.q_cap));
                ggml_backend_tensor_get(cache.q_cap, q.data(), 0, q.size() * sizeof(float));
                qscorer.set_query(q.data(), q.size());
                ok = qscorer.score_chunks(hist, pc.chunk_tokens, scores);
            }
            if (ok) {   // lru arm: intentional no-op (ok stays false)
                pager.score_hook = [&scores](int c) {
                    return c < (int)scores.size() ? scores[c] : 1e30f;
                };
                pager.reselect();
                pager.score_hook = nullptr;
            }
            const double r_s = (now_ms() - r0) / 1000.0;
            rescore_s_total += r_s;
            rescore_n++;
            return r_s;
        };

        auto chunks_resident = [&](int pos0, int n_tok) {
            int res = 0, tot = 0;
            for (int c = pos0 / 64; c <= (pos0 + n_tok - 1) / 64; c++) {
                tot++;
                if (pager.is_resident(c)) res++;
            }
            return std::pair<int,int>(res, tot);
        };

        // ── Needle probe (their /16) ──────────────────────────────────
        int32_t next = -1;
        for (int i = 0; i < 32; i++) next = feed(needle[i]);
        const double needle_rescore_s = reselect_arm();
        int needle_match = 0;
        for (int i = 0; i < 16; i++) {
            if (next == needle[32 + i]) needle_match++;
            next = feed(needle[32 + i]);
        }
        const auto needle_res = chunks_resident(((int)(0.25 * (L - 512)) / 32) * 32, 48);

        // ── Timed free run (decode tok/s) ─────────────────────────────
        t0 = now_ms();
        for (int i = 0; i < 240; i++) next = feed(next);
        const double tok_s = 240.0 / ((now_ms() - t0) / 1000.0);

        // ── Fact probes ───────────────────────────────────────────────
        int fact_match_total = 0, gold_res_total = 0, gold_tot_total = 0;
        int fact_match[NF];
        for (int fidx = 0; fidx < NF; fidx++) {
            for (int i = 0; i < 32; i++) next = feed(fact[fidx][i]);
            reselect_arm();
            const auto gr = chunks_resident(fact_pos[fidx], 48);
            gold_res_total += gr.first;
            gold_tot_total += gr.second;
            fact_match[fidx] = 0;
            for (int i = 0; i < 16; i++) {
                if (next == fact[fidx][32 + i]) fact_match[fidx]++;
                next = feed(fact[fidx][32 + i]);
            }
            fact_match_total += fact_match[fidx];
        }

        const auto & ps = pager.stats();
        std::printf("\n[qkbench] L=%d pool=%d arm=%s\n", L, pool, arm);
        std::printf("QKBENCH %-7d %-6d %-8s prefill_s=%-8.1f needle_rescore_s=%-7.2f "
                    "rescore_mean_s=%-7.2f dec_tok_s=%-6.1f needle=%d/16 "
                    "needle_gold_res=%d/%d facts=%d/64 [%d,%d,%d,%d] "
                    "gold_res=%d/%d page_outs=%" PRId64 " page_ins=%" PRId64 "\n",
                    L, pool, arm, prefill_s, needle_rescore_s,
                    rescore_n ? rescore_s_total / rescore_n : 0.0, tok_s,
                    needle_match, needle_res.first, needle_res.second,
                    fact_match_total,
                    fact_match[0], fact_match[1], fact_match[2], fact_match[3],
                    gold_res_total, gold_tot_total,
                    ps.page_outs, ps.page_ins);
        std::fflush(stdout);

        st.destroy();
        free_target_cache(cache);
        if (dctx.loaded) free_drafter(dctx);
        free_target_weights(w);
        ggml_backend_free(backend);
        return 0;
    }

    // ── --longab: end-to-end long-prompt A/B (speed + accuracy) ─────
    // For L in {32K, 64K, 128K}: full-cache baseline vs pool-4096 with
    // drafter reselect. Measures prefill time, decode tok/s over a
    // 240-token free run, and needle recall (depth 0.25, outside both
    // the sinks and the LRU window).
    if (arg_flag(argc, argv, "--longab")) {
        // Drafter loads lazily, pool mode only: the full-cache baseline at
        // 256K needs every byte (weights 15.3 GiB + KV 4.6 GiB).
        DrafterContext dctx;
        KvFlashDrafterScorer scorer(&dctx);
        // Single-config mode (one process per config: the CUDA VMM pool
        // grows monotonically across large-cache configs and aborts).
        const int only_L = arg_int(argc, argv, "--longab-L", 0);
        const int only_mode = arg_int(argc, argv, "--longab-mode", -1); // 0=full 1=pool
        std::printf("\n%-7s %-10s %-9s %-9s %-9s %-9s %s\n",
                    "L", "mode", "prefill_s", "rescore_s", "dec_tok/s", "needle", "kv_vram");
        for (int L : { 32768, 65536, 131072, 262144 }) {
            if (only_L > 0 && L != only_L) continue;
            for (int mode = 0; mode < 2; mode++) {           // 0=baseline 1=pool
                if (only_mode >= 0 && mode != only_mode) continue;
                if (mode == 1 && !dctx.loaded &&
                    !load_drafter("/opt/lucebox/models/drafter/Qwen3-0.6B-BF16.gguf", 0, dctx)) {
                    std::fprintf(stderr, "drafter load failed\n");
                    return 1;
                }
                const int pool = mode == 0 ? L : 4096;
                auto prompt = make_prompt(L, w.n_vocab);
                std::vector<int32_t> needle(48);
                uint64_t ns = 0xDEADBEEFCAFEull;
                for (int i = 0; i < 48; i++) {
                    ns = ns * 6364136223846793005ull + 1442695040888963407ull;
                    needle[i] = (int32_t)(1000 + (ns >> 33) % 49000);
                }
                const int npos = ((int)(0.25 * (L - 512)) / 32) * 32;
                for (int i = 0; i < 48; i++) prompt[npos + i] = needle[i];

                TargetCache cache;
                if (!create_target_cache(w, pool, 0, backend, cache, true)) return 1;
                const double kv_mib = kv_cache_bytes(cache) / 1048576.0;
                KvFlashPager pager;
                KvFlashConfig pc; pc.pool_tokens = pool;
                if (!pager.attach(pc, cache.attn_k, cache.attn_v)) return 1;

                double t0 = now_ms();
                BatchStepper bs;
                if (!bs.init(w, cache, backend, pool)) return 1;
                for (int p = 0; p < L; p += 64) bs.step_chunk(prompt.data() + p, p, pager);
                bs.destroy();
                const double prefill_s = (now_ms() - t0) / 1000.0;

                Stepper st;
                if (!st.init(w, cache, backend, pool, mode == 1)) return 1;
                int32_t next = -1;
                for (int i = 0; i < 32; i++) {
                    const int slot = pager.slot_for(L + i);
                    st.refresh_mask(pager);
                    next = st.step(needle[i], L + i, slot);
                }
                double rescore_s = 0;
                if (mode == 1) {
                    std::vector<int32_t> hist = prompt;
                    hist.insert(hist.end(), needle.begin(), needle.begin() + 32);
                    std::vector<float> scores;
                    t0 = now_ms();
                    if (scorer.score_chunks(hist, pc.chunk_tokens, scores)) {
                        pager.score_hook = [&scores](int c) {
                            return c < (int)scores.size() ? scores[c] : 1e30f;
                        };
                        pager.reselect();
                        pager.score_hook = nullptr;
                    }
                    rescore_s = (now_ms() - t0) / 1000.0;
                }
                int match = 0;
                for (int i = 0; i < 16; i++) {
                    if (next == needle[32 + i]) match++;
                    const int pos = L + 32 + i;
                    const int slot = pager.slot_for(pos);
                    st.refresh_mask(pager);
                    next = st.step(needle[32 + i], pos, slot);
                }
                t0 = now_ms();
                for (int i = 0; i < 240; i++) {              // timed free run
                    const int pos = L + 48 + i;
                    const int slot = pager.slot_for(pos);
                    st.refresh_mask(pager);
                    next = st.step(next, pos, slot);
                }
                const double tok_s = 240.0 / ((now_ms() - t0) / 1000.0);
                std::printf("%-7d %-10s %-9.1f %-9.1f %-9.1f %d/16   %.0f MiB\n",
                            L, mode == 0 ? "full" : "pool4096",
                            prefill_s, rescore_s, tok_s, match, kv_mib);
                std::fflush(stdout);
                st.destroy();
                free_target_cache(cache);
            }
        }
        if (dctx.loaded) free_drafter(dctx);
        free_target_weights(w);
        ggml_backend_free(backend);
        return 0;
    }

    // ── --niah256: native-max-context probe (262144 logical) ────────
    // Pooled configs only: the fixed-span harness makes a full-pool
    // control prefill take hours at 256K. The LRU row with the needle
    // inside the recency window is the induction control (distance-free).
    if (arg_flag(argc, argv, "--niah256")) {
        DrafterContext dctx;
        if (!load_drafter("/opt/lucebox/models/drafter/Qwen3-0.6B-BF16.gguf", 0, dctx)) {
            std::fprintf(stderr, "drafter load failed\n");
            return 1;
        }
        KvFlashDrafterScorer scorer(&dctx);
        const int L = 262144, pool = 16384;          // 6.25% residency
        struct Cfg { const char * policy; double depth; };
        const Cfg cfgs[] = {
            {"lru",     0.97},   // in-window: induction control at 256K
            {"lru",     0.50},
            {"drafter", 0.10},
            {"drafter", 0.50},
            {"drafter", 0.90},
        };
        std::printf("\n%-7s %-6s %-8s %-6s %s\n", "L", "pool", "policy", "depth", "match/16");
        for (const Cfg & cfg : cfgs) {
            auto prompt = make_prompt(L, w.n_vocab);
            std::vector<int32_t> needle(48);
            uint64_t ns = 0xDEADBEEFCAFEull;
            for (int i = 0; i < 48; i++) {
                ns = ns * 6364136223846793005ull + 1442695040888963407ull;
                needle[i] = (int32_t)(1000 + (ns >> 33) % 49000);
            }
            const int npos = ((int)(cfg.depth * (L - 512)) / 32) * 32;
            for (int i = 0; i < 48; i++) prompt[npos + i] = needle[i];

            TargetCache cache;
            if (!create_target_cache(w, pool, 0, backend, cache, true)) return 1;
            KvFlashPager pager;
            KvFlashConfig pc; pc.pool_tokens = pool;
            if (!pager.attach(pc, cache.attn_k, cache.attn_v)) return 1;

            const double t0 = now_ms();
            BatchStepper bs;
            if (!bs.init(w, cache, backend, pool)) return 1;
            for (int p = 0; p < L; p += 64) bs.step_chunk(prompt.data() + p, p, pager);
            bs.destroy();
            std::printf("[256k] prefill %.1f s, host backing %.2f GiB\n",
                        (now_ms() - t0) / 1000.0,
                        pager.stats().host_bytes / 1073741824.0);

            Stepper st;
            if (!st.init(w, cache, backend, pool, true)) return 1;
            int32_t next = -1;
            for (int i = 0; i < 32; i++) {
                const int slot = pager.slot_for(L + i);
                st.refresh_mask(pager);
                next = st.step(needle[i], L + i, slot);
            }
            if (std::strcmp(cfg.policy, "drafter") == 0) {
                std::vector<int32_t> hist = prompt;
                hist.insert(hist.end(), needle.begin(), needle.begin() + 32);
                std::vector<float> scores;
                const double r0 = now_ms();
                if (!scorer.score_chunks(hist, pc.chunk_tokens, scores)) {
                    std::printf("[256k] WARN rescore failed\n");
                } else {
                    std::printf("[256k] rescore %.1f s\n", (now_ms() - r0) / 1000.0);
                    pager.score_hook = [&scores](int c) {
                        return c < (int)scores.size() ? scores[c] : 1e30f;
                    };
                    pager.reselect();
                    pager.score_hook = nullptr;
                }
            }
            int match = 0;
            for (int i = 0; i < 16; i++) {
                if (next == needle[32 + i]) match++;
                const int pos = L + 32 + i;
                const int slot = pager.slot_for(pos);
                st.refresh_mask(pager);
                next = st.step(needle[32 + i], pos, slot);
            }
            std::printf("%-7d %-6d %-8s %-6.2f %d/16\n", L, pool, cfg.policy, cfg.depth, match);
            std::fflush(stdout);
            st.destroy();
            free_target_cache(cache);
        }
        free_drafter(dctx);
        free_target_weights(w);
        ggml_backend_free(backend);
        return 0;
    }

    if (arg_flag(argc, argv, "--niah")) {
        DrafterContext dctx;
        const bool have_drafter =
            load_drafter("/opt/lucebox/models/drafter/Qwen3-0.6B-BF16.gguf", 0, dctx);
        if (!have_drafter) std::printf("[niah] drafter unavailable, skipping drafter policy\n");
        KvFlashDrafterScorer scorer(&dctx);
        if (have_drafter) {
            // Reserve the drafter's compute buffers at max context NOW,
            // before target-side cache churn fragments the CUDA pool.
            // Without this, 32K rescores OOM late in the sweep and the
            // drafter policy silently degrades to LRU.
            std::vector<int32_t> warm(33024, 1234);
            std::vector<float> tmp;
            scorer.score_chunks(warm, 64, tmp);
        }

        const int Ls[] = { 8192, 32768 };
        const double depths[] = { 0.10, 0.50, 0.90 };
        std::printf("\n%-7s %-6s %-8s %-6s %s\n", "L", "pool", "policy", "depth", "match/16");
        for (int L : Ls) {
            const int pools[] = { L, L / 4, ((L / 10) / 256) * 256 };
            for (int pi = 0; pi < 3; pi++) {
                const int pool = pools[pi];
                const char * policies[] = { "lru", "drafter" };
                const int n_pol = (pi == 0) ? 1 : (have_drafter ? 2 : 1); // full pool: control only
                for (int pol = 0; pol < n_pol; pol++) {
                    for (double depth : depths) {
                        // Needle: 48 unique-as-a-sequence tokens from the
                        // filler id range (matched embedding statistics).
                        // Query = first 32 (longer match = stronger
                        // induction), score the last 16.
                        auto prompt = make_prompt(L, w.n_vocab);
                        std::vector<int32_t> needle(48);
                        uint64_t ns = 0xDEADBEEFCAFEull;
                        for (int i = 0; i < 48; i++) {
                            ns = ns * 6364136223846793005ull + 1442695040888963407ull;
                            needle[i] = (int32_t)(1000 + (ns >> 33) % 49000);
                        }
                        const int npos = ((int)(depth * (L - 512)) / 32) * 32;
                        for (int i = 0; i < 48; i++) prompt[npos + i] = needle[i];

                        TargetCache cache;
                        if (!create_target_cache(w, pool, 0, backend, cache, true)) return 1;
                        KvFlashPager pager;
                        KvFlashConfig pc; pc.pool_tokens = pool;
                        if (!pager.attach(pc, cache.attn_k, cache.attn_v)) return 1;

                        BatchStepper bs;
                        if (!bs.init(w, cache, backend, pool)) return 1;
                        for (int p = 0; p < L; p += 64) {
                            bs.step_chunk(prompt.data() + p, p, pager);
                        }
                        bs.destroy();

                        Stepper st;
                        if (!st.init(w, cache, backend, pool, true)) return 1;
                        int32_t next = -1;
                        for (int i = 0; i < 32; i++) {       // query: needle prefix
                            const int slot = pager.slot_for(L + i);
                            st.refresh_mask(pager);
                            next = st.step(needle[i], L + i, slot);
                        }
                        if (pol == 1) {                       // drafter reselect
                            std::vector<int32_t> hist = prompt;
                            hist.insert(hist.end(), needle.begin(), needle.begin() + 32);
                            std::vector<float> scores;
                            if (!scorer.score_chunks(hist, pc.chunk_tokens, scores)) {
                                std::printf("[niah] WARN: rescore failed (L=%d pool=%d)\n", L, pool);
                            } else {
                                pager.score_hook = [&scores](int c) {
                                    return c < (int)scores.size() ? scores[c] : 1e30f;
                                };
                                pager.reselect();
                                pager.score_hook = nullptr;
                            }
                        }
                        int match = 0;
                        for (int i = 0; i < 16; i++) {        // continuation
                            if (next == needle[32 + i]) match++;
                            // Teacher-force ground truth: one miss must not
                            // cascade; we measure per-position retrieval.
                            const int pos = L + 32 + i;
                            const int slot = pager.slot_for(pos);
                            st.refresh_mask(pager);
                            next = st.step(needle[32 + i], pos, slot);
                        }
                        std::printf("%-7d %-6d %-8s %-6.2f %d/16\n",
                                    L, pool, pi == 0 ? "full" : policies[pol],
                                    depth, match);
                        std::fflush(stdout);
                        st.destroy();
                        free_target_cache(cache);
                    }
                }
            }
        }
        if (have_drafter) free_drafter(dctx);
        free_target_weights(w);
        ggml_backend_free(backend);
        return 0;
    }

    const auto prompt = make_prompt(n_prompt, w.n_vocab);
    std::vector<int32_t> tokens_a;
    size_t mem_a_kv = 0, mem_a_buf = 0, mem_a_vram = 0;
    size_t mem_c_kv = 0, mem_c_buf = 0, mem_c_vram = 0;
    int hard_failures = 0;

    // ── Run A: baseline at logical context, maskless ────────────────
    {
        const size_t v_before = vram_used_now();
        TargetCache cache;
        if (!create_target_cache(w, logical_ctx, 0, backend, cache, /*prefill_only=*/true)) {
            std::fprintf(stderr, "cache A: %s\n", dflash27b_last_error());
            return 1;
        }
        mem_a_kv = kv_cache_bytes(cache);
        mem_a_buf = ggml_backend_buffer_get_size(cache.base_buf);
        mem_a_vram = vram_used_now() - v_before;
        std::printf("[A] logical_ctx=%d  kv=%.1f MiB  base_buf=%.1f MiB  vram_delta=%.1f MiB\n",
                    logical_ctx, mem_a_kv / 1048576.0, mem_a_buf / 1048576.0,
                    mem_a_vram / 1048576.0);

        Stepper st;
        int32_t next = -1;
        const double t0 = now_ms();
        for (int pos = 0; pos < total; pos++) {
            // Production-like growing span: rebuild only when the padded
            // span crosses a 256 boundary (mirrors do_ar_decode topology).
            const int want_span = pos + 1;
            if (!st.ctx || ((want_span + 255) / 256) != ((st.span + 255) / 256)) {
                st.span = want_span;
                if (!st.ctx) { if (!st.init(w, cache, backend, want_span, false)) return 1; }
                else if (!st.build()) return 1;
            }
            const int32_t tok = pos < n_prompt ? prompt[pos]
                              : (tokens_a.push_back(next), next);
            next = st.step(tok, pos, pos);
            cache.cur_pos = pos + 1;
        }
        tokens_a.push_back(next);
        std::printf("[A] decoded %zu tokens, %.1f tok/s overall\n",
                    tokens_a.size(), total / ((now_ms() - t0) / 1000.0));
        st.destroy();
        free_target_cache(cache);
    }

    // ── Run B: relocation + mask exactness, teacher-forced ──────────
    {
        TargetCache cache;
        if (!create_target_cache(w, pool_b, 0, backend, cache, /*prefill_only=*/true)) {
            std::fprintf(stderr, "cache B: %s\n", dflash27b_last_error());
            return 1;
        }
        KvFlashPager pager;
        KvFlashConfig pc;
        pc.pool_tokens = pool_b;
        if (!pager.attach(pc, cache.attn_k, cache.attn_v)) return 1;
        const int nb = pool_b / pc.chunk_tokens;
        std::vector<int> order(nb);
        for (int i = 0; i < nb; i++) order[i] = i;
        uint64_t s = 12345;
        for (int i = nb - 1; i > 0; i--) {
            s = s * 6364136223846793005ull + 1442695040888963407ull;
            const int j = (int)((s >> 33) % (uint64_t)(i + 1));
            std::swap(order[i], order[j]);
        }
        pager.set_block_order(order);

        Stepper st;
        if (!st.init(w, cache, backend, pool_b, use_mask)) return 1;
        int mismatches = 0, first_mismatch = -1;
        for (int pos = 0; pos < total; pos++) {
            const int32_t tok = pos < n_prompt ? prompt[pos] : tokens_a[pos - n_prompt];
            const int slot = pager.slot_for(pos);
            st.refresh_mask(pager);
            const int32_t next = st.step(tok, pos, slot);
            const int ref_idx = pos - n_prompt + 1;
            if (pos >= n_prompt - 1 && ref_idx < (int)tokens_a.size()) {
                if (next != tokens_a[ref_idx]) {
                    mismatches++;
                    if (first_mismatch < 0) first_mismatch = pos;
                }
            }
        }
        const double rate = 100.0 * mismatches / (n_gen + 1);
        std::printf("[B] shuffled+masked, pool=%d: %d/%d argmax mismatches (%.2f%%), first at pos %d; "
                    "mask refills=%d avg=%.3f ms\n",
                    pool_b, mismatches, n_gen + 1, rate, first_mismatch,
                    st.mask_fills, st.mask_fills ? st.mask_fill_ms_total / st.mask_fills : 0.0);
        // Gate at 2%: the flip sources are the maskless zero-row softmax
        // mass plus run-to-run fattn nondeterminism; both measured ~1%
        // (10-14 flips/1201 across runs), so a 1% gate flaps on noise.
        std::printf("%s relocation equivalence (threshold 2%%)\n", rate <= 2.0 ? "PASS" : "FAIL");
        if (rate > 2.0) hard_failures++;
        st.destroy();
        free_target_cache(cache);
    }

    // ── Run G: prefix-cache equivalence (KVFlash + snapshot compose) ────
    // G1 proves serialize/deserialize is byte-lossless including host-backed
    // chunks; G2 proves the model continues coherently from a restored,
    // physically RELOCATED prefix (KV carried as the pager blob, recurrent
    // state copied as the real restore_target_cache does).
    {
        // G1: byte-exact round-trip through a pool smaller than the prompt,
        // so the blob spans both resident AND host-backed chunks.
        const int pool_g1 = 7 * 64;   // 1 sink + 4 tail + 2 evictable blocks
        {
            TargetCache cs;
            if (!create_target_cache(w, pool_g1, 0, backend, cs, /*prefill_only=*/true)) {
                std::fprintf(stderr, "cache G1 src: %s\n", dflash27b_last_error()); return 1;
            }
            KvFlashPager ps; KvFlashConfig pc; pc.pool_tokens = pool_g1;
            if (!ps.attach(pc, cs.attn_k, cs.attn_v)) return 1;
            Stepper st; if (!st.init(w, cs, backend, pool_g1, use_mask)) return 1;
            for (int pos = 0; pos < n_prompt; pos++) {
                const int slot = ps.slot_for(pos);
                st.refresh_mask(ps);
                (void)st.step(prompt[pos], pos, slot);
                cs.cur_pos = pos + 1;
            }
            std::vector<uint8_t> blob1; ps.serialize(blob1);
            const int64_t src_pageouts = ps.stats().page_outs;
            st.destroy();

            TargetCache cd;
            if (!create_target_cache(w, pool_g1, 0, backend, cd, /*prefill_only=*/true)) {
                std::fprintf(stderr, "cache G1 dst: %s\n", dflash27b_last_error()); return 1;
            }
            KvFlashPager pd; if (!pd.attach(pc, cd.attn_k, cd.attn_v)) return 1;
            const bool ok = pd.deserialize(blob1, n_prompt);
            std::vector<uint8_t> blob2; if (ok) pd.serialize(blob2);
            const bool exact = ok && blob1.size() == blob2.size() &&
                               std::memcmp(blob1.data(), blob2.data(), blob1.size()) == 0;
            std::printf("[G1] pool=%d prompt=%d: src page_outs=%" PRId64
                        ", blob=%zu B, deserialize=%s, reserialize match=%s\n",
                        pool_g1, n_prompt, src_pageouts, blob1.size(),
                        ok ? "ok" : "FAIL", exact ? "yes" : "no");
            std::printf("%s serialize/deserialize byte-lossless (incl. host-backed chunks)\n",
                        (exact && src_pageouts > 0) ? "PASS" : "FAIL");
            if (!(exact && src_pageouts > 0)) hard_failures++;
            free_target_cache(cd);
            free_target_cache(cs);
        }

        // G1b: chunk-UNALIGNED prefix round-trip + resume slot allocation,
        // pager-level (no model forward). Pins the prefix-skip resume edges:
        // a snapshot taken mid-chunk and a suffix prefilled after restore with
        // live eviction must not corrupt the page table or fail allocation.
        {
            const int pool = 8 * 64;
            TargetCache cs, cd;
            if (!create_target_cache(w, pool, 0, backend, cs, /*prefill_only=*/true) ||
                !create_target_cache(w, pool, 0, backend, cd, /*prefill_only=*/true)) {
                std::fprintf(stderr, "cache G1b: %s\n", dflash27b_last_error()); return 1;
            }
            KvFlashPager ps; KvFlashConfig pc; pc.pool_tokens = pool;
            if (!ps.attach(pc, cs.attn_k, cs.attn_v)) return 1;
            const int Nu = 9 * 64 + 30;   // unaligned: last chunk is partial
            bool prefill_ok = true;
            for (int p = 0; p < Nu; p++) if (ps.slot_for(p) < 0) { prefill_ok = false; break; }
            std::vector<uint8_t> b1; ps.serialize(b1);
            const int64_t po = ps.stats().page_outs;

            KvFlashPager pd; if (!pd.attach(pc, cd.attn_k, cd.attn_v)) return 1;
            const int nb = pool / pc.chunk_tokens;
            std::vector<int> ord(nb);
            for (int i = 0; i < nb; i++) ord[i] = i;
            std::swap(ord[0], ord[4]); std::swap(ord[2], ord[7]);
            pd.set_block_order(ord);
            const bool ok = pd.deserialize(b1, Nu);
            std::vector<uint8_t> b2; if (ok) pd.serialize(b2);
            const bool exact = ok && b1 == b2;
            bool resume_ok = true;          // suffix prefill with live eviction
            for (int p = Nu; p < Nu + 200; p++) if (pd.slot_for(p) < 0) { resume_ok = false; break; }
            const bool floor = pd.is_resident(0) && pd.is_resident((Nu + 199) / 64);
            std::printf("[G1b] unaligned N=%d (tail=%d) page_outs=%" PRId64
                        ", reserialize-exact=%s, resume+200=%s, floor=%d\n",
                        Nu, Nu % 64, po, exact ? "yes" : "no",
                        resume_ok ? "ok" : "FAIL", floor);
            std::printf("%s unaligned-prefix restore + resume (pager-level)\n",
                        (prefill_ok && exact && resume_ok && floor && po > 0) ? "PASS" : "FAIL");
            if (!(prefill_ok && exact && resume_ok && floor && po > 0)) hard_failures++;
            free_target_cache(cd);
            free_target_cache(cs);
        }

        // G2: continuation equivalence — restore a relocated prefix and
        // teacher-force the baseline; argmax must track A within run B's bound.
        {
            const int pool_g2 = (((total + 63) / 64) + 2) * 64;   // >= total, no eviction
            TargetCache cs, cd;
            if (!create_target_cache(w, pool_g2, 0, backend, cs, /*prefill_only=*/true) ||
                !create_target_cache(w, pool_g2, 0, backend, cd, /*prefill_only=*/true)) {
                std::fprintf(stderr, "cache G2: %s\n", dflash27b_last_error()); return 1;
            }
            KvFlashPager ps; KvFlashConfig pc; pc.pool_tokens = pool_g2;
            if (!ps.attach(pc, cs.attn_k, cs.attn_v)) return 1;
            // Shuffle the source placement so save->restore is a real
            // relocation (source shuffled -> logical blob -> dest packed).
            const int nb = pool_g2 / pc.chunk_tokens;
            std::vector<int> order(nb);
            for (int i = 0; i < nb; i++) order[i] = i;
            uint64_t s = 0x9E3779B97F4A7C15ull;
            for (int i = nb - 1; i > 0; i--) {
                s = s * 6364136223846793005ull + 1442695040888963407ull;
                std::swap(order[i], order[(int)((s >> 33) % (uint64_t)(i + 1))]);
            }
            ps.set_block_order(order);

            Stepper sts; if (!sts.init(w, cs, backend, pool_g2, use_mask)) return 1;
            int32_t seed = -1;
            for (int pos = 0; pos < n_prompt; pos++) {
                const int slot = ps.slot_for(pos);
                sts.refresh_mask(ps);
                seed = sts.step(prompt[pos], pos, slot);
                cs.cur_pos = pos + 1;
            }
            std::vector<uint8_t> blob; ps.serialize(blob);
            // Recurrent (delta-net) state is orthogonal to the pager; the real
            // restore_target_cache copies it, so mirror that to isolate the
            // KV-restore path under test.
            for (size_t i = 0; i < cs.ssm_state.size(); i++) {
                if (cs.ssm_state[i]  && cd.ssm_state[i])  ggml_backend_tensor_copy(cs.ssm_state[i],  cd.ssm_state[i]);
                if (cs.conv_state[i] && cd.conv_state[i]) ggml_backend_tensor_copy(cs.conv_state[i], cd.conv_state[i]);
            }
            if (cs.target_feat && cd.target_feat) ggml_backend_tensor_copy(cs.target_feat, cd.target_feat);
            sts.destroy();

            KvFlashPager pd; if (!pd.attach(pc, cd.attn_k, cd.attn_v)) return 1;
            if (!pd.deserialize(blob, n_prompt)) {
                std::printf("FAIL G2 deserialize (blob/geometry mismatch)\n");
                hard_failures++;
            } else {
                cd.cur_pos = n_prompt;
                Stepper st; if (!st.init(w, cd, backend, pool_g2, use_mask)) return 1;
                int mismatches = 0, first = -1;
                for (int pos = n_prompt; pos < total; pos++) {
                    const int gi = pos - n_prompt;
                    const int slot = pd.slot_for(pos);
                    st.refresh_mask(pd);
                    const int32_t out = st.step(tokens_a[gi], pos, slot);
                    cd.cur_pos = pos + 1;
                    const int ref = gi + 1;
                    if (ref < (int)tokens_a.size() && out != tokens_a[ref]) {
                        mismatches++; if (first < 0) first = pos;
                    }
                }
                const double rate = 100.0 * mismatches / n_gen;
                std::printf("[G2] pool=%d restore@%d (seed=%d, A[0]=%d): %d/%d argmax "
                            "mismatches (%.2f%%), first at %d\n",
                            pool_g2, n_prompt, seed, tokens_a.empty() ? -1 : tokens_a[0],
                            mismatches, n_gen, rate, first);
                std::printf("%s restored-prefix continuation tracks baseline (threshold 2%%)\n",
                            rate <= 2.0 ? "PASS" : "FAIL");
                if (rate > 2.0) hard_failures++;
                st.destroy();
            }
            free_target_cache(cd);
            free_target_cache(cs);
        }
    }

    // ── Run C: live paging + roundtrip; D: reselect recall ──────────
    {
        const size_t v_before = vram_used_now();
        TargetCache cache;
        if (!create_target_cache(w, pool_c, 0, backend, cache, /*prefill_only=*/true)) {
            std::fprintf(stderr, "cache C: %s\n", dflash27b_last_error());
            return 1;
        }
        mem_c_kv = kv_cache_bytes(cache);
        mem_c_buf = ggml_backend_buffer_get_size(cache.base_buf);
        mem_c_vram = vram_used_now() - v_before;

        KvFlashPager pager;
        KvFlashConfig pc;
        pc.pool_tokens = pool_c;
        if (!pager.attach(pc, cache.attn_k, cache.attn_v)) return 1;

        Stepper st;
        if (!st.init(w, cache, backend, pool_c, use_mask)) return 1;
        int32_t next = -1;
        for (int pos = 0; pos < n_prompt; pos++) {
            const int slot = pager.slot_for(pos);
            st.refresh_mask(pager);
            next = st.step(prompt[pos], pos, slot);
            cache.cur_pos = pos + 1;
        }
        {   // bit-exact roundtrip on chunk 2
            ggml_tensor * t = cache.attn_k[0];
            const size_t seg = (size_t)pc.chunk_tokens * t->nb[1];
            std::vector<uint8_t> before(seg), after(seg);
            ggml_backend_tensor_get(t, before.data(),
                (size_t)pager.block_of(2) * pc.chunk_tokens * t->nb[1], seg);
            if (!pager.page_out(2) || !pager.page_in(2)) {
                std::fprintf(stderr, "roundtrip paging failed\n"); return 1;
            }
            ggml_backend_tensor_get(t, after.data(),
                (size_t)pager.block_of(2) * pc.chunk_tokens * t->nb[1], seg);
            const bool exact = std::memcmp(before.data(), after.data(), seg) == 0;
            std::printf("%s page_out/page_in roundtrip bit-exact (chunk 2 -> block %d)\n",
                        exact ? "PASS" : "FAIL", pager.block_of(2));
            if (!exact) hard_failures++;
        }

        std::vector<int32_t> tokens_c;
        const double t0 = now_ms();
        for (int pos = n_prompt; pos < total; pos++) {
            tokens_c.push_back(next);
            const int slot = pager.slot_for(pos);
            st.refresh_mask(pager);
            next = st.step(next, pos, slot);
            cache.cur_pos = pos + 1;
        }
        tokens_c.push_back(next);
        const double secs = (now_ms() - t0) / 1000.0;
        int agree = 0;
        while (agree < (int)tokens_c.size() && agree < (int)tokens_a.size() &&
               tokens_c[agree] == tokens_a[agree]) agree++;
        const auto & ps = pager.stats();
        std::printf("[C] pool=%d masked: decode %.1f tok/s, page_outs=%" PRId64
                    " page_ins=%" PRId64 " host=%.1f MiB; baseline agreement %d tokens\n",
                    pool_c, n_gen / secs, ps.page_outs, ps.page_ins,
                    ps.host_bytes / 1048576.0, agree);
        std::printf("PASS paged decode with eviction (%d evictions)\n", (int)ps.page_outs);

        // ── Run D: τ-style reselect recall ──────────────────────────
        {
            int victim = -1;   // earliest paged-out, non-sink chunk
            for (int c = pc.sink_chunks; c < pager.n_chunks(); c++) {
                if (!pager.is_resident(c)) { victim = c; break; }
            }
            if (victim < 0) {
                std::printf("FAIL reselect demo: no paged-out chunk found\n");
                hard_failures++;
            } else {
                // Score injection: the victim becomes the hottest chunk —
                // stands in for a drafter rescore flagging recalled context.
                pager.score_hook = [&](int c) { return c == victim ? 2.0f : 1.0f / (1 + c); };
                const double r0 = now_ms();
                const int events = pager.reselect();
                const double r_ms = now_ms() - r0;
                const bool back = pager.is_resident(victim);
                std::printf("%s reselect recalled chunk %d (%d page events, %.2f ms)\n",
                            back ? "PASS" : "FAIL", victim, events, r_ms);
                if (!back) hard_failures++;
                // decode must continue cleanly after the residency change
                pager.score_hook = nullptr;
                for (int pos = total; pos < total + 64; pos++) {
                    const int slot = pager.slot_for(pos);
                    st.refresh_mask(pager);
                    next = st.step(next, pos, slot);
                }
                std::printf("PASS decode continues after reselect (64 tokens)\n");
            }
        }
        st.destroy();
        free_target_cache(cache);
    }

    // ── Run F: full LSA loop — drafter as Memory Indexer ────────────
    // Prompt LARGER than the pool, so prefill itself evicts; then the
    // FlashMemory inference paradigm end to end: every τ=64 decoded
    // tokens the drafter rescores the whole sequence (tail attention =
    // indexer query), score_hook gets the fresh chunk scores, and
    // reselect() repages the pool. PASS requires at least one genuine
    // drafter-driven recall of a chunk evicted earlier.
    {
        const char * drafter_path = "/opt/lucebox/models/drafter/Qwen3-0.6B-BF16.gguf";
        DrafterContext dctx;
        if (!load_drafter(drafter_path, 0, dctx)) {
            std::printf("FAIL indexer run: drafter load failed (%s)\n", dflash27b_last_error());
            hard_failures++;
        } else {
            const int n_prompt_f = 2048, n_gen_f = 768, pool_f = 1024, tau = 64;
            const auto prompt_f = make_prompt(n_prompt_f, w.n_vocab);
            TargetCache cache;
            if (!create_target_cache(w, pool_f, 0, backend, cache, true)) return 1;
            KvFlashPager pager;
            KvFlashConfig pc;
            pc.pool_tokens = pool_f;
            if (!pager.attach(pc, cache.attn_k, cache.attn_v)) return 1;
            KvFlashDrafterScorer scorer(&dctx);   // the production indexer plugin

            Stepper st;
            if (!st.init(w, cache, backend, pool_f, use_mask)) return 1;
            std::vector<int32_t> all_ids = prompt_f;
            int32_t next = -1;
            for (int pos = 0; pos < n_prompt_f; pos++) {
                const int slot = pager.slot_for(pos);
                st.refresh_mask(pager);
                next = st.step(prompt_f[pos], pos, slot);
            }
            const int64_t prefill_evictions = pager.stats().page_outs;

            std::vector<double> rescore_ms, reselect_ms;
            int64_t recalls = 0;
            std::vector<float> scores;
            const double t0 = now_ms();
            for (int g = 0; g < n_gen_f; g++) {
                const int pos = n_prompt_f + g;
                if (g % tau == 0) {
                    double r0 = now_ms();
                    if (!scorer.score_chunks(all_ids, pc.chunk_tokens, scores)) {
                        std::fprintf(stderr, "scorer failed\n");
                        std::exit(1);
                    }
                    rescore_ms.push_back(now_ms() - r0);
                    pager.score_hook = [&scores](int c) {
                        return c < (int)scores.size() ? scores[c] : 1e30f;
                    };
                    r0 = now_ms();
                    const int64_t ins_before = pager.stats().page_ins;
                    pager.reselect();
                    reselect_ms.push_back(now_ms() - r0);
                    recalls += pager.stats().page_ins - ins_before;
                }
                const int slot = pager.slot_for(pos);
                st.refresh_mask(pager);
                next = st.step(next, pos, slot);
                all_ids.push_back(next);
            }
            const double secs = (now_ms() - t0) / 1000.0;
            StepTimes rs = summarize(rescore_ms), rsel = summarize(reselect_ms);
            const auto & ps = pager.stats();
            std::printf("[F] LSA loop: prompt=%d pool=%d gen=%d tau=%d -> %.1f tok/s "
                        "(prefill evicted %" PRId64 ")\n",
                        n_prompt_f, pool_f, n_gen_f, tau, n_gen_f / secs, prefill_evictions);
            std::printf("[F] indexer rescore p50=%.1f ms (full 0.6B re-prefill, %zu calls); "
                        "reselect p50=%.2f ms; drafter-driven recalls=%" PRId64
                        "; total page_outs=%" PRId64 " page_ins=%" PRId64 "\n",
                        rs.p50, rescore_ms.size(), rsel.p50, recalls,
                        ps.page_outs, ps.page_ins);
            std::printf("%s LSA loop: drafter-driven recall of evicted context (recalls >= 1)\n",
                        recalls >= 1 ? "PASS" : "FAIL");
            if (recalls < 1) hard_failures++;
            st.destroy();
            free_target_cache(cache);
            free_drafter(dctx);
        }
    }

    // ── Run E: performance profile ──────────────────────────────────
    if (!skip_prof) {
        std::printf("\n=== DECODE PROFILE (64 timed steps each, junk KV, span = FA window) ===\n");
        auto profile = [&](const char * tag, int alloc_ctx, int span, bool masked,
                           KvFlashPager * pager, int pos_base) {
            TargetCache cache;
            if (!create_target_cache(w, alloc_ctx, 0, backend, cache, true)) {
                std::fprintf(stderr, "cache E(%s): %s\n", tag, dflash27b_last_error());
                std::exit(1);
            }
            KvFlashPager local;
            if (masked && !pager) {
                KvFlashConfig pc; pc.pool_tokens = alloc_ctx;
                local.attach(pc, cache.attn_k, cache.attn_v);
                // mark whole pool resident so the mask is all-zero (worst
                // case mask read, no -inf shortcut)
                for (int p = 0; p < alloc_ctx; p += 64) local.slot_for(p);
                pager = &local;
            }
            Stepper st;
            if (!st.init(w, cache, backend, span, masked)) std::exit(1);
            // warmup 8, then time 64 (refresh included: it is part of the
            // real per-step cost in masked mode)
            int32_t tok = 1000;
            for (int i = 0; i < 8; i++) {
                if (masked) st.refresh_mask(*pager);
                tok = st.step(tok, pos_base + i, (i * 64) % alloc_ctx);
            }
            std::vector<double> ms;
            for (int i = 0; i < 64; i++) {
                const double t0 = now_ms();
                if (masked) st.refresh_mask(*pager);
                tok = st.step(tok, pos_base + 8 + i, (8 * 64 + i) % alloc_ctx);
                ms.push_back(now_ms() - t0);
            }
            const StepTimes r = summarize(ms);
            std::printf("%-28s span=%6d  p50=%7.2f ms  p95=%7.2f ms  mean=%7.2f ms  (%5.1f tok/s)\n",
                        tag, span, r.p50, r.p95, r.mean, 1000.0 / r.mean);
            st.destroy();
            free_target_cache(cache);
        };
        profile("baseline 8K",   8192,   8192,   false, nullptr, 8192 - 72);
        profile("baseline 32K",  32768,  32768,  false, nullptr, 32768 - 72);
        profile("baseline 128K", 131072, 131072, false, nullptr, 131072 - 72);
        profile("pool 1K masked (128K logical)", 1024, 1024, true,  nullptr, 130000);
        profile("pool 1K maskless",              1024, 1024, false, nullptr, 130000);
        profile("pool 4K masked (128K logical)", 4096, 4096, true,  nullptr, 130000);

        // Page-event microbench on a small pool.
        {
            TargetCache cache;
            if (!create_target_cache(w, 1024, 0, backend, cache, true)) std::exit(1);
            KvFlashPager pager;
            KvFlashConfig pc; pc.pool_tokens = 1024;
            pager.attach(pc, cache.attn_k, cache.attn_v);
            for (int p = 0; p < 1024; p += 64) pager.slot_for(p);
            std::vector<double> out_ms, in_ms;
            for (int rep = 0; rep < 32; rep++) {
                const int c = 2 + (rep % 8);
                double t0 = now_ms();
                pager.page_out(c);
                out_ms.push_back(now_ms() - t0);
                t0 = now_ms();
                pager.page_in(c);
                in_ms.push_back(now_ms() - t0);
            }
            const StepTimes o = summarize(out_ms), i = summarize(in_ms);
            std::printf("page_out: p50=%.2f ms  p95=%.2f ms   page_in: p50=%.2f ms  p95=%.2f ms  (per 64-token chunk, %zu KiB)\n",
                        o.p50, o.p95, i.p50, i.p95,
                        (size_t)(pager.stats().host_bytes / std::max<int64_t>(1, 8) / 1024));
            free_target_cache(cache);
        }
    }

    // ── Run H: pinned protected class ──────────────────────────────
    // Pool geometry: pool=8*64=512 tokens, sink_chunks=1, tail_window_chunks=4.
    // max_pins = 8 - 1 - 4 - 1 = 2 (two non-sink, non-tail marginal slots).
    {
        const int chunk      = 64;
        const int pool_h     = 8 * chunk;
        const int sink_h     = 1;
        const int tail_h     = 4;
        // max_pins = n_blocks - sink - tail - 1 = 8-1-4-1 = 2
        const int max_pins_h = 8 - sink_h - tail_h - 1;  // = 2

        // H1 — pin survives full pool-pressure eviction
        {
            TargetCache ch1;
            if (!create_target_cache(w, pool_h, 0, backend, ch1, /*prefill_only=*/true)) {
                std::fprintf(stderr, "cache H1: %s\n", dflash27b_last_error()); return 1;
            }
            KvFlashPager ph1;
            KvFlashConfig pc_h1;
            pc_h1.pool_tokens        = pool_h;
            pc_h1.sink_chunks        = sink_h;
            pc_h1.tail_window_chunks = tail_h;
            if (!ph1.attach(pc_h1, ch1.attn_k, ch1.attn_v)) return 1;

            // materialize chunks 0..7 (fills the pool)
            for (int p = 0; p < pool_h; p++) ph1.slot_for(p);
            // pin chunk 2 (non-sink, non-tail at this point)
            ph1.pin_chunk(2);
            const bool is_pinned_before = ph1.is_pinned(2);
            const bool res_before = ph1.is_resident(2);

            // drive past chunk 11 to force eviction — pool=8 blocks, cur_chunk
            // will be >7 so tail window covers last 4, sink covers 0, victim
            // pool must evict something. Chunks 1..3 are the mid-evictable ones.
            for (int p = pool_h; p < 12 * chunk; p++) ph1.slot_for(p);

            const bool still_pinned  = ph1.is_pinned(2);
            const bool still_resident = ph1.is_resident(2);
            const bool evictions_ran  = ph1.stats().page_outs >= 1;
            // chunk 3 is NOT pinned → should have been a victim
            const bool c3_evicted = !ph1.is_resident(3);
            std::printf("[H1] pinned_before=%d res_before=%d still_pinned=%d "
                        "still_resident=%d evictions=%lld c3_evicted=%d\n",
                        (int)is_pinned_before, (int)res_before, (int)still_pinned,
                        (int)still_resident, (long long)ph1.stats().page_outs,
                        (int)c3_evicted);
            const bool h1_pass = is_pinned_before && res_before &&
                                 still_pinned && still_resident &&
                                 evictions_ran && c3_evicted;
            std::printf("%s H1: pin survives full pool-pressure eviction\n",
                        h1_pass ? "PASS" : "FAIL");
            if (!h1_pass) hard_failures++;
            free_target_cache(ch1);
        }

        // H2 — pin survives adversarial reselect()
        {
            TargetCache ch2;
            if (!create_target_cache(w, pool_h, 0, backend, ch2, /*prefill_only=*/true)) {
                std::fprintf(stderr, "cache H2: %s\n", dflash27b_last_error()); return 1;
            }
            KvFlashPager ph2;
            KvFlashConfig pc_h2;
            pc_h2.pool_tokens        = pool_h;
            pc_h2.sink_chunks        = sink_h;
            pc_h2.tail_window_chunks = tail_h;
            if (!ph2.attach(pc_h2, ch2.attn_k, ch2.attn_v)) return 1;

            // fill pool fully so reselect has something to evict
            for (int p = 0; p < pool_h; p++) ph2.slot_for(p);
            ph2.pin_chunk(2);
            // score_hook gives chunk 2 the worst possible score
            ph2.score_hook = [](int c) { return c == 2 ? -1e30f : 1.0f; };
            ph2.reselect();
            ph2.score_hook = nullptr;
            const bool h2_pass = ph2.is_resident(2);
            std::printf("%s H2: pin survives adversarial reselect() (is_resident=%d)\n",
                        h2_pass ? "PASS" : "FAIL", (int)h2_pass);
            if (!h2_pass) hard_failures++;
            free_target_cache(ch2);
        }

        // H3 — pin survives serialize/deserialize via export/import (NOT in KV blob)
        {
            TargetCache cs3, cd3;
            if (!create_target_cache(w, pool_h, 0, backend, cs3, /*prefill_only=*/true) ||
                !create_target_cache(w, pool_h, 0, backend, cd3, /*prefill_only=*/true)) {
                std::fprintf(stderr, "cache H3: %s\n", dflash27b_last_error()); return 1;
            }
            KvFlashPager ps3;
            KvFlashConfig pc_h3;
            pc_h3.pool_tokens        = pool_h;
            pc_h3.sink_chunks        = sink_h;
            pc_h3.tail_window_chunks = tail_h;
            if (!ps3.attach(pc_h3, cs3.attn_k, cs3.attn_v)) return 1;

            const int Nu3 = 11 * chunk + 30;  // unaligned
            for (int p = 0; p < Nu3; p++) ps3.slot_for(p);
            ps3.pin_chunk(5);
            std::vector<uint8_t> blob3;
            ps3.serialize(blob3);
            std::vector<uint8_t> pins3;
            ps3.export_pins(pins3);

            // fresh pager on a new cache
            KvFlashPager pd3;
            if (!pd3.attach(pc_h3, cd3.attn_k, cd3.attn_v)) return 1;
            const bool deser_ok = pd3.deserialize(blob3, Nu3);
            // pins must NOT be in the KV blob
            const bool pin_absent = !pd3.is_pinned(5);
            // import pins then drive suffix eviction
            pd3.import_pins(pins3);
            bool slot_ok = true;
            for (int p = Nu3; p < Nu3 + 300; p++) {
                if (pd3.slot_for(p) < 0) { slot_ok = false; break; }
            }
            const bool h3_pass = deser_ok && pin_absent && slot_ok &&
                                 pd3.is_resident(5) && pd3.is_pinned(5);
            std::printf("[H3] deser=%d pin_absent_before_import=%d slot_ok=%d "
                        "resident_after=%d pinned_after=%d\n",
                        (int)deser_ok, (int)pin_absent, (int)slot_ok,
                        (int)pd3.is_resident(5), (int)pd3.is_pinned(5));
            std::printf("%s H3: pin survives export/import (not in KV blob)\n",
                        h3_pass ? "PASS" : "FAIL");
            if (!h3_pass) hard_failures++;
            free_target_cache(cs3);
            free_target_cache(cd3);
        }

        // H4 — deadlock guard: max_pins=2, third pin refused, no deadlock
        {
            TargetCache ch4;
            if (!create_target_cache(w, pool_h, 0, backend, ch4, /*prefill_only=*/true)) {
                std::fprintf(stderr, "cache H4: %s\n", dflash27b_last_error()); return 1;
            }
            KvFlashPager ph4;
            KvFlashConfig pc_h4;
            pc_h4.pool_tokens        = pool_h;
            pc_h4.sink_chunks        = sink_h;
            pc_h4.tail_window_chunks = tail_h;
            if (!ph4.attach(pc_h4, ch4.attn_k, ch4.attn_v)) return 1;

            // materialize chunks 0..7
            for (int p = 0; p < pool_h; p++) ph4.slot_for(p);

            // At this point cur_chunk_=7; tail_window protects [8-1-4..7]=[3..7].
            // sink protects chunk 0. Chunks 1,2 are the marginal evictable ones.
            ph4.pin_chunk(1);
            ph4.pin_chunk(2);
            ph4.pin_chunk(3);  // 3 is tail-protected, so marginal count=2 → refused? No —
                               // 3 is in tail window → n_pinned_marginal skips it → budget NOT consumed
                               // So pin_chunk(3) of a tail-protected chunk succeeds but is free.
                               // Then pin_chunk(4) would be the 3rd marginal → refused.
            // Let's push far enough that 3 is no longer tail-protected,
            // then try pinning a 3rd marginal chunk.
            // After filling past chunk 11: cur_chunk_=11, tail=[7..10], sink=[0]
            // marginal = pinned chunks NOT in sink/tail
            // Let's get to cur_chunk_ >= 9 first: fill to chunk 9
            for (int p = pool_h; p < 9 * chunk; p++) ph4.slot_for(p);
            // Now cur_chunk_=8 (pos 8*64..8*64+63 → c=8), tail_window=[8-4..7]?
            // Wait: cur_chunk_ = 8, tail=[cur_chunk_-1-tail_h .. cur_chunk_-1] = [3..7]
            // So chunks 1,2 are marginal. chunk 3 is in tail → not marginal.
            // Pins on 1 and 2: n_pinned_marginal = 2 = max_pins → pin of chunk 3 would
            // be marginal too once it leaves tail. Let's extend further.
            for (int p = 9 * chunk; p < 12 * chunk; p++) ph4.slot_for(p);
            // cur_chunk_=11, tail=[11-1-4..10]=[6..10]
            // Chunks 1,2 pinned AND not in sink/tail → marginal=2 = max_pins
            // chunk 3 no longer in tail → pinning it would make marginal=3 → refused
            ph4.pin_chunk(3);  // this should now be refused
            const bool c1_pinned = ph4.is_pinned(1);
            const bool c2_pinned = ph4.is_pinned(2);
            const bool c3_refused = !ph4.is_pinned(3);

            // drive even more tokens — no slot_for should return -1
            bool no_deadlock = true;
            for (int p = 12 * chunk; p < 20 * chunk; p++) {
                if (ph4.slot_for(p) < 0) { no_deadlock = false; break; }
            }
            const bool c1_still = ph4.is_resident(1);
            const bool c2_still = ph4.is_resident(2);
            std::printf("[H4] c1_pinned=%d c2_pinned=%d c3_refused=%d "
                        "no_deadlock=%d c1_resident=%d c2_resident=%d max_pins=%d\n",
                        (int)c1_pinned, (int)c2_pinned, (int)c3_refused,
                        (int)no_deadlock, (int)c1_still, (int)c2_still, max_pins_h);
            const bool h4_pass = c1_pinned && c2_pinned && c3_refused &&
                                 no_deadlock && c1_still && c2_still;
            std::printf("%s H4: deadlock guard + pin budget enforced\n",
                        h4_pass ? "PASS" : "FAIL");
            if (!h4_pass) hard_failures++;
            free_target_cache(ch4);
        }

        // H5 — bounds + idempotency
        {
            TargetCache ch5;
            if (!create_target_cache(w, pool_h, 0, backend, ch5, /*prefill_only=*/true)) {
                std::fprintf(stderr, "cache H5: %s\n", dflash27b_last_error()); return 1;
            }
            KvFlashPager ph5;
            KvFlashConfig pc_h5;
            pc_h5.pool_tokens        = pool_h;
            pc_h5.sink_chunks        = sink_h;
            pc_h5.tail_window_chunks = tail_h;
            if (!ph5.attach(pc_h5, ch5.attn_k, ch5.attn_v)) return 1;

            for (int p = 0; p < pool_h; p++) ph5.slot_for(p);

            // OOB: no crash
            const bool oob_safe = !ph5.is_pinned(9999);

            // idempotent: double-pin of chunk 2 consumes budget once
            ph5.pin_chunk(2);
            ph5.pin_chunk(2);  // second call — idempotent, no budget consumed twice
            // budget should still allow one more marginal pin
            // At cur_chunk_=7, tail=[7-1-4..6]=[2..6], so chunk 2 IS in tail → marginal=0
            // Need to extend so chunk 2 is out of tail window
            for (int p = pool_h; p < 12 * chunk; p++) ph5.slot_for(p);
            // cur_chunk_=11, tail=[6..10], chunk 2 now marginal, pinned → marginal=1
            // pin chunk 3 (also marginal) — should succeed since budget=2 and we only used 1
            ph5.pin_chunk(3);
            const bool c2_pinned = ph5.is_pinned(2);
            const bool c3_also_pinned = ph5.is_pinned(3);

            std::printf("[H5] oob_safe=%d c2_pinned=%d c3_also_pinned=%d\n",
                        (int)oob_safe, (int)c2_pinned, (int)c3_also_pinned);
            const bool h5_pass = oob_safe && c2_pinned && c3_also_pinned;
            std::printf("%s H5: bounds safe + idempotency preserves pin budget\n",
                        h5_pass ? "PASS" : "FAIL");
            if (!h5_pass) hard_failures++;
            free_target_cache(ch5);
        }

        // ── Pure-function test: kvflash_chunk_should_pin ──────────────
        {
            std::vector<uint8_t> needle(10, 0);
            needle[3] = 1; needle[7] = 1;

            // header window: pin_header_chunks=2 → c<2 pinned
            const bool hdr0 = kvflash_chunk_should_pin(0, 2, false, {});
            const bool hdr1 = kvflash_chunk_should_pin(1, 2, false, {});
            const bool hdr2 = !kvflash_chunk_should_pin(2, 2, false, {});  // boundary: c==2 not pinned

            // needle bitmask
            const bool ndl3 = kvflash_chunk_should_pin(3, 0, true, needle);
            const bool ndl4 = !kvflash_chunk_should_pin(4, 0, true, needle);
            const bool ndl7 = kvflash_chunk_should_pin(7, 0, true, needle);

            // empty bitmask + no header → no pin
            const bool none = !kvflash_chunk_should_pin(5, 0, true, {});

            std::printf("[H-pure] hdr0=%d hdr1=%d hdr2_boundary=%d "
                        "ndl3=%d ndl4=%d ndl7=%d none=%d\n",
                        (int)hdr0, (int)hdr1, (int)hdr2,
                        (int)ndl3, (int)ndl4, (int)ndl7, (int)none);
            const bool pure_pass = hdr0 && hdr1 && hdr2 && ndl3 && ndl4 && ndl7 && none;
            std::printf("%s H-pure: kvflash_chunk_should_pin pure function\n",
                        pure_pass ? "PASS" : "FAIL");
            if (!pure_pass) hard_failures++;
        }
    }

    // ── Memory verdict ──────────────────────────────────────────────
    const double red_kv = 100.0 * (1.0 - (double)mem_c_kv / (double)mem_a_kv);
    std::printf("\n=== KV MEMORY ===\n");
    std::printf("baseline  (ctx %6d): kv=%9.1f MiB  base_buf=%9.1f MiB  vram_delta=%9.1f MiB\n",
                logical_ctx, mem_a_kv / 1048576.0, mem_a_buf / 1048576.0, mem_a_vram / 1048576.0);
    std::printf("pooled    (pool %5d): kv=%9.1f MiB  base_buf=%9.1f MiB  vram_delta=%9.1f MiB\n",
                pool_c, mem_c_kv / 1048576.0, mem_c_buf / 1048576.0, mem_c_vram / 1048576.0);
    std::printf("attn-KV reduction: %.1f%%\n", red_kv);
    std::printf("%s KV memory reduction >= 90%%\n", red_kv >= 90.0 ? "PASS" : "FAIL");
    if (red_kv < 90.0) hard_failures++;

    free_target_weights(w);
    ggml_backend_free(backend);
    std::printf("\n%s (%d hard failures)\n", hard_failures == 0 ? "ALL PASS" : "FAILED", hard_failures);
    return hard_failures == 0 ? 0 : 1;
}
