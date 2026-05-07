// Gemma4 DFlash speculative decoding end-to-end test / benchmark driver.
//
// Pipeline:
//   1. Load target (Gemma4-31B or 26B-A4B GGUF) + draft (z-lab Gemma4-DFlash
//      safetensors directory).
//   2. Prefill: single-token autoregressive decode over prompt tokens,
//      capture_layers=true so target_feat gets populated for every prompt pos.
//   3. Decode loop (until n_predict):
//      a. [target-only path, always active]
//         Run target forward for last committed token → logits → sample next.
//      b. [speculative path, active when draft is loaded] — TODO
//         i.  Get target_feat from cache.
//         ii. Run draft model to propose a block of tokens (DDTree).
//         iii. Verify proposals against target in one batched forward.
//         iv. Accept longest verified prefix + bonus token, advance cache.
//   4. Print generated text and timing stats.
//
// Usage:
//   test_gemma4_dflash --model <gemma4.gguf> [--draft <dir>]
//                      [--prompt <text>] [--n-predict <N>]
//                      [--ctx-size <N>] [--kv-k <type>] [--kv-v <type>]
//                      [--seed <N>] [--temp <F>] [--top-k <N>] [--top-p <F>]
//                      [--budget <N>] [--gpu <N>] [--bench]

#include "internal.h"
#include "dflash27b.h"
#include "gemma4.h"
#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-cuda.h"
#include <cuda_runtime_api.h>

#ifdef _WIN32
#define setenv(name, value, overwrite) _putenv_s(name, value)
#define unsetenv(name) _putenv_s(name, "")
#endif

#include <algorithm>
#include <chrono>
#include <cinttypes>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <unordered_set>
#include <vector>
#include <random>

using namespace dflash27b;

// ─── Utilities ────────────────────────────────────────────────────────────

static constexpr int    KQ_MASK_PAD  = 32;
static constexpr uint16_t F16_ZERO   = 0x0000;
static constexpr uint16_t F16_NEG_INF = 0xFC00;

static int g_kq_stride_pad = KQ_MASK_PAD;

static int align_up(int x, int a) { return ((x + a - 1) / a) * a; }

static int argmax_f32(const float * x, int n) {
    int best = 0;
    float bv = x[0];
    for (int i = 1; i < n; i++) {
        if (x[i] > bv) { bv = x[i]; best = i; }
    }
    return best;
}

// ─── Sampler ──────────────────────────────────────────────────────────────

struct SamplerCfg {
    float    temp       = 0.0f;
    float    top_p      = 1.0f;
    int      top_k      = 0;
    float    rep_pen    = 1.0f;
    int      rep_window = 256;
    uint64_t seed       = 0;
};

static int sample_logits(const float * logits_in,
                         int vocab,
                         const SamplerCfg & cfg,
                         const std::vector<int32_t> & history,
                         std::mt19937_64 & rng) {
    if (cfg.temp <= 0.0f) {
        return argmax_f32(logits_in, vocab);
    }

    std::vector<std::pair<float, int>> cand(vocab);
    for (int i = 0; i < vocab; i++) cand[i] = {logits_in[i], i};

    if (cfg.rep_pen > 1.0f && !history.empty()) {
        const int win = std::min((int)history.size(), cfg.rep_window);
        const int from = (int)history.size() - win;
        std::unordered_set<int> seen;
        for (int i = from; i < (int)history.size(); i++) seen.insert(history[i]);
        for (auto & c : cand) {
            if (seen.count(c.second)) {
                c.first = (c.first > 0.0f) ? c.first / cfg.rep_pen
                                           : c.first * cfg.rep_pen;
            }
        }
    }

    if (cfg.top_k > 0 && cfg.top_k < vocab) {
        std::partial_sort(cand.begin(), cand.begin() + cfg.top_k, cand.end(),
                          [](const auto & a, const auto & b) { return a.first > b.first; });
        cand.resize(cfg.top_k);
    } else {
        std::sort(cand.begin(), cand.end(),
                  [](const auto & a, const auto & b) { return a.first > b.first; });
    }

    const float inv_t = 1.0f / std::max(1e-3f, cfg.temp);
    float maxv = cand.front().first * inv_t;
    double Z = 0.0;
    std::vector<float> probs(cand.size());
    for (size_t i = 0; i < cand.size(); i++) {
        probs[i] = std::exp(cand[i].first * inv_t - maxv);
        Z += probs[i];
    }
    for (auto & p : probs) p = (float)(p / Z);

    if (cfg.top_p > 0.0f && cfg.top_p < 1.0f) {
        double cum = 0.0;
        size_t cut = probs.size();
        for (size_t i = 0; i < probs.size(); i++) {
            cum += probs[i];
            if (cum >= cfg.top_p) { cut = i + 1; break; }
        }
        probs.resize(cut);
        cand.resize(cut);
        double zz = 0.0;
        for (auto p : probs) zz += p;
        for (auto & p : probs) p = (float)(p / zz);
    }

    std::uniform_real_distribution<double> u(0.0, 1.0);
    double r = u(rng);
    double acc = 0.0;
    for (size_t i = 0; i < probs.size(); i++) {
        acc += probs[i];
        if (r <= acc) return cand[i].second;
    }
    return cand.back().second;
}

// ─── Causal mask builder ──────────────────────────────────────────────────

static void build_causal_mask(std::vector<uint16_t> & out,
                              int kv_len, int n_tokens, int kv_start) {
    const int kv_pad = align_up(kv_len, g_kq_stride_pad);
    const int q_pad  = align_up(n_tokens, KQ_MASK_PAD);
    out.assign((size_t)kv_pad * q_pad, F16_NEG_INF);
    for (int q = 0; q < n_tokens; q++) {
        const int abs_q = kv_start + q;
        for (int k = 0; k <= abs_q && k < kv_len; k++) {
            out[(size_t)q * kv_pad + k] = F16_ZERO;
        }
    }
}

// ─── Per-step graph state (rebuilt each forward pass since kv_len varies) ─

struct StepGraph {
    ggml_context   * ctx        = nullptr;
    ggml_cgraph    * gf         = nullptr;
    ggml_gallocr_t   alloc      = nullptr;
    ggml_tensor    * inp_embed  = nullptr;
    ggml_tensor    * positions  = nullptr;
    ggml_tensor    * attn_mask  = nullptr;
    ggml_tensor    * logits     = nullptr;
};

static void step_graph_free(StepGraph & sg) {
    if (sg.ctx) { ggml_free(sg.ctx); sg.ctx = nullptr; }
    sg.gf        = nullptr;
    sg.inp_embed = nullptr;
    sg.positions = nullptr;
    sg.attn_mask = nullptr;
    sg.logits    = nullptr;
}

static void step_graph_destroy(StepGraph & sg) {
    if (sg.alloc) { ggml_gallocr_free(sg.alloc); sg.alloc = nullptr; }
    step_graph_free(sg);
}

// Build a single-step target forward graph.
//   n_tokens  - number of tokens in this forward (1 for decode, >1 for prefill)
//   kv_start  - index of the first new token in the KV cache
//   with_mask - whether to allocate an attention-mask input (required for n_tokens > 1)
//   capture   - whether to write captured layer features to cache.target_feat
static bool build_gemma4_step(StepGraph & sg,
                              const GemmaTargetWeights & w,
                              GemmaTargetCache & cache,
                              ggml_backend_t backend,
                              int kv_start,
                              int n_tokens,
                              bool with_mask,
                              bool capture) {
    step_graph_free(sg);

    ggml_init_params ip{};
    ip.mem_size   = 512 * 1024 * 1024;
    ip.mem_buffer = nullptr;
    ip.no_alloc   = true;
    sg.ctx = ggml_init(ip);
    if (!sg.ctx) return false;

    sg.inp_embed = ggml_new_tensor_3d(sg.ctx, GGML_TYPE_F32, w.n_embd, n_tokens, 1);
    ggml_set_name(sg.inp_embed, "inp_embed");
    ggml_set_input(sg.inp_embed);

    sg.positions = ggml_new_tensor_1d(sg.ctx, GGML_TYPE_I32, n_tokens);
    ggml_set_name(sg.positions, "positions");
    ggml_set_input(sg.positions);

    if (with_mask) {
        const int kv_len = kv_start + n_tokens;
        const int kv_pad = align_up(kv_len, g_kq_stride_pad);
        const int q_pad  = align_up(n_tokens, KQ_MASK_PAD);
        sg.attn_mask = ggml_new_tensor_2d(sg.ctx, GGML_TYPE_F16, kv_pad, q_pad);
        ggml_set_name(sg.attn_mask, "attn_mask");
        ggml_set_input(sg.attn_mask);
    }

    sg.gf = ggml_new_graph_custom(sg.ctx, 16384, false);

    GemmaGraphInputs gi{};
    gi.inp_embed      = sg.inp_embed;
    gi.positions      = sg.positions;
    gi.attn_mask      = sg.attn_mask;
    gi.n_tokens       = n_tokens;
    gi.kv_start       = kv_start;
    gi.capture_layers = capture;

    GemmaGraphOutputs go = build_gemma4_graph(sg.ctx, sg.gf, w, cache, gi);
    if (!go.logits) return false;
    sg.logits = go.logits;
    ggml_set_output(sg.logits);
    ggml_build_forward_expand(sg.gf, sg.logits);

    if (!sg.alloc) {
        sg.alloc = ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend));
    }
    return ggml_gallocr_alloc_graph(sg.alloc, sg.gf);
}

// ─── Embed one token into the inp_embed input tensor ─────────────────────

static bool embed_token(const GemmaTargetWeights & w,
                        int32_t token_id,
                        ggml_tensor * inp_embed,
                        ggml_backend_t backend) {
    const int hidden = w.n_embd;
    std::vector<float> emb(hidden);
    if (!w.embedder.embed(&token_id, 1, emb.data())) {
        std::fprintf(stderr, "[embed] failed for token %d\n", token_id);
        return false;
    }
    // inp_embed shape: [hidden, 1, 1]
    ggml_backend_tensor_set(inp_embed, emb.data(), 0, sizeof(float) * hidden);
    (void)backend;
    return true;
}

// Embed a batch of tokens (for chunked prefill).
static bool embed_tokens_batch(const GemmaTargetWeights & w,
                               const int32_t * ids,
                               int n,
                               ggml_tensor * inp_embed,
                               ggml_backend_t backend) {
    const int hidden = w.n_embd;
    std::vector<float> emb((size_t)hidden * n);
    if (!w.embedder.embed(ids, n, emb.data())) {
        std::fprintf(stderr, "[embed_batch] failed for %d tokens\n", n);
        return false;
    }
    ggml_backend_tensor_set(inp_embed, emb.data(), 0, sizeof(float) * hidden * n);
    (void)backend;
    return true;
}

// ─── EOS check ───────────────────────────────────────────────────────────

#define IS_EOS_TOK(tok, w) \
    (((w).eos_chat_id >= 0 && (tok) == (w).eos_chat_id) || \
     ((w).eos_id      >= 0 && (tok) == (w).eos_id))

// ─── KV type resolution helper ───────────────────────────────────────────

static ggml_type kv_type_from_string(const std::string & s) {
    if (s == "f16")   return GGML_TYPE_F16;
    if (s == "q8_0")  return GGML_TYPE_Q8_0;
    if (s == "q4_0")  return GGML_TYPE_Q4_0;
    if (s == "tq3_0") return GGML_TYPE_TQ3_0;
    return GGML_TYPE_Q8_0;  // default
}

// ─── Nanosecond wall clock ────────────────────────────────────────────────

static double now_ms() {
    return std::chrono::duration<double, std::milli>(
        std::chrono::steady_clock::now().time_since_epoch()).count();
}

// ─── Minimal tokenizer stub ──────────────────────────────────────────────
//
// A proper tokenizer (SentencePiece / tiktoken) requires linking to an
// external library. For benchmarking purposes we provide two modes:
//
//   1. Pre-tokenised input via --tokens <id0,id1,...>
//      Pass comma-separated integer token IDs directly. This is the
//      recommended path for reproducible benchmarks.
//
//   2. Byte-fallback: each byte of the --prompt string becomes one token.
//      This is NOT linguistically valid but lets the driver run without any
//      tokenizer library. Override with --tokens for real evaluation.

static std::vector<int32_t> tokenize_byte_fallback(const std::string & text) {
    std::vector<int32_t> ids;
    ids.reserve(text.size());
    for (unsigned char c : text) {
        ids.push_back((int32_t)c);
    }
    return ids;
}

static std::vector<int32_t> parse_token_ids(const std::string & s) {
    std::vector<int32_t> ids;
    const char * p = s.c_str();
    while (*p) {
        char * end = nullptr;
        long v = std::strtol(p, &end, 10);
        if (end == p) break;
        ids.push_back((int32_t)v);
        if (*end == '\0') break;
        if (*end == ',') { p = end + 1; continue; }
        break;
    }
    return ids;
}

// ─── Main ─────────────────────────────────────────────────────────────────

static void print_usage(const char * prog) {
    std::fprintf(stderr,
        "usage: %s --model <gemma4.gguf> [options]\n"
        "\n"
        "Options:\n"
        "  --model  <path>   path to Gemma4 GGUF (target, required)\n"
        "  --draft  <dir>    path to z-lab DFlash safetensors directory (optional)\n"
        "  --prompt <text>   input prompt text (default: \"Hello, world!\")\n"
        "  --tokens <ids>    comma-separated prompt token IDs (overrides --prompt)\n"
        "  --n-predict <N>   max tokens to generate (default: 128)\n"
        "  --ctx-size  <N>   max context size (default: 4096)\n"
        "  --kv-k <type>     KV cache K type: f16/q8_0/q4_0/tq3_0 (default: q8_0)\n"
        "  --kv-v <type>     KV cache V type: f16/q8_0/q4_0/tq3_0 (default: q8_0)\n"
        "  --seed <N>        RNG seed (default: 0)\n"
        "  --temp <F>        temperature, 0 = greedy (default: 0.0)\n"
        "  --top-k <N>       top-k sampling, 0 = disabled (default: 0)\n"
        "  --top-p <F>       nucleus sampling (default: 1.0)\n"
        "  --budget <N>      DDTree budget for speculative decoding (default: 22)\n"
        "  --gpu <N>         CUDA device index (default: 0)\n"
        "  --bench           benchmark mode: repeat generation, report statistics\n"
        "  --fa-window <N>   sliding attention window for full layers (0 = full, default: 0)\n"
        "\n",
        prog);
}

int main(int argc, char ** argv) {
    if (argc < 2) {
        print_usage(argv[0]);
        return 2;
    }

    // ── Parse CLI arguments ───────────────────────────────────────────────
    std::string  model_path;
    std::string  draft_path;
    std::string  prompt_text  = "Hello, world!";
    std::string  token_ids_str;
    int          n_predict    = 128;
    int          ctx_size     = 4096;
    std::string  kv_k_str     = "q8_0";
    std::string  kv_v_str     = "q8_0";
    int          gpu           = 0;
    int          ddtree_budget = 22;
    bool         bench_mode   = false;
    int          fa_window    = 0;
    SamplerCfg   sampler;

    for (int i = 1; i < argc; i++) {
        auto require_next = [&](const char * flag) -> const char * {
            if (i + 1 >= argc) {
                std::fprintf(stderr, "error: %s requires an argument\n", flag);
                std::exit(2);
            }
            return argv[++i];
        };

        if      (std::strcmp(argv[i], "--model")     == 0) model_path    = require_next("--model");
        else if (std::strcmp(argv[i], "--draft")     == 0) draft_path    = require_next("--draft");
        else if (std::strcmp(argv[i], "--prompt")    == 0) prompt_text   = require_next("--prompt");
        else if (std::strcmp(argv[i], "--tokens")    == 0) token_ids_str = require_next("--tokens");
        else if (std::strcmp(argv[i], "--n-predict") == 0) n_predict     = std::atoi(require_next("--n-predict"));
        else if (std::strcmp(argv[i], "--ctx-size")  == 0) ctx_size      = std::atoi(require_next("--ctx-size"));
        else if (std::strcmp(argv[i], "--kv-k")      == 0) kv_k_str      = require_next("--kv-k");
        else if (std::strcmp(argv[i], "--kv-v")      == 0) kv_v_str      = require_next("--kv-v");
        else if (std::strcmp(argv[i], "--seed")      == 0) sampler.seed  = (uint64_t)std::atoll(require_next("--seed"));
        else if (std::strcmp(argv[i], "--temp")      == 0) sampler.temp  = (float)std::atof(require_next("--temp"));
        else if (std::strcmp(argv[i], "--top-k")     == 0) sampler.top_k = std::atoi(require_next("--top-k"));
        else if (std::strcmp(argv[i], "--top-p")     == 0) sampler.top_p = (float)std::atof(require_next("--top-p"));
        else if (std::strcmp(argv[i], "--budget")    == 0) ddtree_budget = std::atoi(require_next("--budget"));
        else if (std::strcmp(argv[i], "--gpu")       == 0) gpu           = std::atoi(require_next("--gpu"));
        else if (std::strcmp(argv[i], "--fa-window") == 0) fa_window     = std::atoi(require_next("--fa-window"));
        else if (std::strcmp(argv[i], "--bench")     == 0) bench_mode    = true;
        else if (std::strcmp(argv[i], "--help")      == 0 ||
                 std::strcmp(argv[i], "-h")          == 0) {
            print_usage(argv[0]);
            return 0;
        } else {
            std::fprintf(stderr, "warning: unknown argument: %s\n", argv[i]);
        }
    }

    if (model_path.empty()) {
        std::fprintf(stderr, "error: --model is required\n");
        print_usage(argv[0]);
        return 2;
    }

    // ── KV type env vars (consumed by create_gemma4_cache → resolve_kv_types) ─
    setenv("DFLASH27B_KV_K", kv_k_str.c_str(), 1);
    setenv("DFLASH27B_KV_V", kv_v_str.c_str(), 1);

    // TurboQuant / TQ3 FA kernels require kv_len aligned to 256.
    if (kv_k_str == "tq3_0" || kv_v_str == "tq3_0") {
        g_kq_stride_pad = 256;
    }

    // ── CUDA device validation ────────────────────────────────────────────
    int cuda_device_count = 0;
    cudaGetDeviceCount(&cuda_device_count);
    if (gpu >= cuda_device_count) {
        std::fprintf(stderr, "error: --gpu %d out of range (device_count=%d)\n",
                     gpu, cuda_device_count);
        return 2;
    }
    cudaSetDevice(gpu);

    std::printf("[cfg] model=%s draft=%s gpu=%d ctx=%d n_predict=%d kv_k=%s kv_v=%s "
                "temp=%.2f top_k=%d top_p=%.2f budget=%d bench=%d fa_window=%d\n",
                model_path.c_str(),
                draft_path.empty() ? "(none)" : draft_path.c_str(),
                gpu, ctx_size, n_predict,
                kv_k_str.c_str(), kv_v_str.c_str(),
                sampler.temp, sampler.top_k, sampler.top_p,
                ddtree_budget, (int)bench_mode, fa_window);

    // ── Backend init ──────────────────────────────────────────────────────
    ggml_backend_t backend = ggml_backend_cuda_init(gpu);
    if (!backend) {
        std::fprintf(stderr, "error: ggml_backend_cuda_init(%d) failed\n", gpu);
        return 1;
    }

    // ── Load target weights ───────────────────────────────────────────────
    GemmaTargetWeights w;
    {
        double t0 = now_ms();
        if (!load_gemma4_target_gguf(model_path, backend, w)) {
            std::fprintf(stderr, "load_gemma4_target_gguf: %s\n", dflash27b_last_error());
            return 1;
        }
        double t1 = now_ms();
        std::printf("[target] loaded %d layers, n_embd=%d, vocab=%d  (%.1f ms)\n",
                    w.n_layer, w.n_embd, w.n_vocab, t1 - t0);
    }

    // ── Load draft weights (optional) ────────────────────────────────────
    // The GemmaDraftWeights struct is defined file-locally in gemma4_dflash_graph.cpp;
    // we forward-declare the loader here via the internal linkage it provides.
    // For now the driver supports target-only mode; draft integration is a TODO.
    const bool have_draft = !draft_path.empty();
    if (have_draft) {
        std::printf("[draft] TODO: load_gemma4_draft_safetensors(\"%s\") — "
                    "draft integration pending\n",
                    draft_path.c_str());
        std::printf("[draft] Running in target-only mode for this build.\n");
    }

    // ── Create KV cache ───────────────────────────────────────────────────
    GemmaTargetCache cache;
    {
        double t0 = now_ms();
        if (!create_gemma4_cache(w, ctx_size, backend, cache)) {
            std::fprintf(stderr, "create_gemma4_cache: %s\n", dflash27b_last_error());
            return 1;
        }
        double t1 = now_ms();
        std::printf("[cache] created max_ctx=%d, kv_layers=%zu  (%.1f ms)\n",
                    cache.max_ctx, cache.attn_k.size(), t1 - t0);
    }

    // ── Tokenize prompt ───────────────────────────────────────────────────
    std::vector<int32_t> prompt_ids;
    if (!token_ids_str.empty()) {
        prompt_ids = parse_token_ids(token_ids_str);
        if (prompt_ids.empty()) {
            std::fprintf(stderr, "error: --tokens produced no valid token IDs\n");
            return 2;
        }
        std::printf("[tokens] using %zu pre-tokenised IDs from --tokens\n",
                    prompt_ids.size());
    } else {
        prompt_ids = tokenize_byte_fallback(prompt_text);
        std::printf("[tokens] byte-fallback tokenisation: %zu tokens "
                    "(pass --tokens <ids> for real tokenisation)\n",
                    prompt_ids.size());
    }

    if ((int)prompt_ids.size() >= ctx_size) {
        std::fprintf(stderr, "error: prompt (%zu tokens) >= ctx_size (%d)\n",
                     prompt_ids.size(), ctx_size);
        return 2;
    }

    // ── RNG ───────────────────────────────────────────────────────────────
    std::mt19937_64 rng(sampler.seed);

    // ── Benchmark loop outer container ────────────────────────────────────
    const int bench_runs = bench_mode ? 3 : 1;
    std::vector<double> bench_tok_per_sec;

    // Declared here (main scope) so step_graph_destroy(sg) in cleanup is valid.
    StepGraph sg;

    for (int bench_iter = 0; bench_iter < bench_runs; bench_iter++) {

        if (bench_runs > 1) {
            reset_gemma4_cache(cache);
            std::printf("[bench] run %d/%d\n", bench_iter + 1, bench_runs);
        }

        // ── Prefill ───────────────────────────────────────────────────────
        //
        // We run each prompt token through the target one at a time.
        // A batched prefill would be faster; this simpler loop is enough for
        // correctness testing and matches the decode-loop pattern.
        //
        // For each prompt token t at position p:
        //   1. Embed token t → inp_embed
        //   2. Set positions[0] = p
        //   3. Build forward graph (with causal mask for p > 0)
        //   4. Compute graph → logits (discarded during prefill; only KV + target_feat matter)

        std::printf("[prefill] %zu tokens ...\n", prompt_ids.size());
        double prefill_t0 = now_ms();
        int last_logit_tok = -1;

        for (int pi = 0; pi < (int)prompt_ids.size(); pi++) {
            const int32_t tok = prompt_ids[pi];
            const int     pos = pi;
            const bool    need_mask = (pi > 0);
            const int     kv_start  = pos;

            if (!build_gemma4_step(sg, w, cache, backend,
                                   kv_start, /*n_tokens=*/1,
                                   need_mask, /*capture=*/true)) {
                std::fprintf(stderr, "prefill build failed at token %d\n", pi);
                return 1;
            }

            if (!embed_token(w, tok, sg.inp_embed, backend)) return 1;

            // positions: single i32
            int32_t pos_val = pos;
            ggml_backend_tensor_set(sg.positions, &pos_val, 0, sizeof(int32_t));

            // Causal mask for n_tokens=1 at position pos: attend all [0..pos].
            if (sg.attn_mask) {
                const int kv_len  = kv_start + 1;
                std::vector<uint16_t> mask_buf;
                build_causal_mask(mask_buf, kv_len, 1, kv_start);
                ggml_backend_tensor_set(sg.attn_mask, mask_buf.data(), 0,
                                        sizeof(uint16_t) * mask_buf.size());
            }

            auto st = ggml_backend_graph_compute(backend, sg.gf);
            if (st != GGML_STATUS_SUCCESS) {
                std::fprintf(stderr, "prefill compute failed at token %d\n", pi);
                return 1;
            }

            cache.cur_pos = pos + 1;

            // Read last token's logits for the generation seed
            if (pi == (int)prompt_ids.size() - 1) {
                const int vocab = w.n_vocab;
                std::vector<float> logits_cpu(vocab);
                ggml_backend_tensor_get(sg.logits, logits_cpu.data(), 0,
                                        sizeof(float) * vocab);
                last_logit_tok = sample_logits(logits_cpu.data(), vocab,
                                               sampler, prompt_ids, rng);
                cache.last_tok = last_logit_tok;
            }

            step_graph_free(sg);
        }

        double prefill_t1 = now_ms();
        std::printf("[prefill] done in %.1f ms  (last sampled token: %d)\n",
                    prefill_t1 - prefill_t0, last_logit_tok);

        // ── Decode loop ───────────────────────────────────────────────────
        //
        // Target-only autoregressive path.
        // Each iteration:
        //   1. Feed `last_tok` through the target at position `committed`.
        //   2. Sample the next token from logits.
        //   3. Append to generated sequence.
        //   4. Stop if EOS or n_predict reached.
        //
        // TODO: When a draft model is loaded, replace this with the speculative
        // decoding loop:
        //   a. Sync target_feat to the draft feature mirror.
        //   b. Build noise block: [last_tok, MASK * (block_size-1)].
        //   c. Run draft forward → draft logits.
        //   d. Build DDTree from top-K distributions (budget = ddtree_budget).
        //   e. Run tree-verify batched target forward with ancestor-only mask.
        //   f. Walk tree accepting longest prefix + bonus token.
        //   g. Rollback SSM/conv state to accepted position.
        //   h. Advance committed, last_tok.

        std::vector<int32_t> generated;
        generated.reserve(n_predict);
        std::vector<int32_t> history(prompt_ids);

        int committed = cache.cur_pos;
        int32_t cur_tok = last_logit_tok;

        double decode_t0 = now_ms();
        double first_token_ms = -1.0;

        while ((int)generated.size() < n_predict) {

            if (IS_EOS_TOK(cur_tok, w)) {
                std::printf("\n[decode] EOS token %d at step %zu\n",
                            cur_tok, generated.size());
                break;
            }

            if (committed >= ctx_size - 1) {
                std::printf("\n[decode] context full at step %zu\n",
                            generated.size());
                break;
            }

            // Build single-token decode graph
            if (!build_gemma4_step(sg, w, cache, backend,
                                   committed, /*n_tokens=*/1,
                                   /*with_mask=*/false,
                                   /*capture=*/have_draft)) {
                std::fprintf(stderr, "[decode] build failed at step %zu\n",
                             generated.size());
                return 1;
            }

            if (!embed_token(w, cur_tok, sg.inp_embed, backend)) return 1;

            int32_t pos_val = committed;
            ggml_backend_tensor_set(sg.positions, &pos_val, 0, sizeof(int32_t));

            double step_t0 = now_ms();
            auto st = ggml_backend_graph_compute(backend, sg.gf);
            double step_t1 = now_ms();

            if (st != GGML_STATUS_SUCCESS) {
                std::fprintf(stderr, "[decode] compute failed at step %zu\n",
                             generated.size());
                return 1;
            }

            committed++;
            cache.cur_pos = committed;

            // Fetch logits and sample
            const int vocab = w.n_vocab;
            std::vector<float> logits_cpu(vocab);
            ggml_backend_tensor_get(sg.logits, logits_cpu.data(), 0,
                                    sizeof(float) * vocab);

            const int32_t next_tok = (int32_t)sample_logits(
                logits_cpu.data(), vocab, sampler, history, rng);

            generated.push_back(cur_tok);
            history.push_back(cur_tok);

            if (first_token_ms < 0.0 && !generated.empty()) {
                first_token_ms = step_t1 - step_t0;
            }

            // Print token id (a proper decoder would map id -> string here)
            std::printf("%d ", cur_tok);
            std::fflush(stdout);

            cur_tok = next_tok;
            cache.last_tok = cur_tok;

            step_graph_free(sg);

            // TODO (speculative path): when have_draft, run draft + DDTree here
            // instead of the single-token autoregressive step above.
            (void)ddtree_budget;
            (void)fa_window;
        }

        double decode_t1 = now_ms();
        const double decode_ms = decode_t1 - decode_t0;
        const int    n_gen     = (int)generated.size();
        const double tps       = (decode_ms > 0.0 && n_gen > 0)
                                     ? n_gen / (decode_ms / 1000.0)
                                     : 0.0;

        bench_tok_per_sec.push_back(tps);

        std::printf("\n");
        std::printf("[stats] generated=%d  decode_ms=%.1f  tok/s=%.2f  "
                    "first_tok_ms=%.2f\n",
                    n_gen, decode_ms, tps, first_token_ms);
        std::printf("[stats] prefill=%zu tokens  context_used=%d/%d\n",
                    prompt_ids.size(), committed, ctx_size);

        // ── Memory stats ──────────────────────────────────────────────────
        {
            size_t free_bytes = 0, total_bytes = 0;
            cudaMemGetInfo(&free_bytes, &total_bytes);
            const double used_gb  = (total_bytes - free_bytes) / (1024.0 * 1024.0 * 1024.0);
            const double total_gb = total_bytes / (1024.0 * 1024.0 * 1024.0);
            std::printf("[mem]  VRAM used=%.2f GB  total=%.2f GB\n",
                        used_gb, total_gb);
        }

    } // bench loop

    // ── Benchmark summary ─────────────────────────────────────────────────
    if (bench_mode && bench_tok_per_sec.size() > 1) {
        std::sort(bench_tok_per_sec.begin(), bench_tok_per_sec.end());
        const double median = bench_tok_per_sec[bench_tok_per_sec.size() / 2];
        const double best   = bench_tok_per_sec.back();
        std::printf("\n[bench] median=%.2f tok/s  best=%.2f tok/s  runs=%zu\n",
                    median, best, bench_tok_per_sec.size());
    }

    // ── Cleanup ───────────────────────────────────────────────────────────
    step_graph_destroy(sg);
    free_gemma4_cache(cache);
    free_gemma4_target_weights(w);
    ggml_backend_free(backend);

    return 0;
}
