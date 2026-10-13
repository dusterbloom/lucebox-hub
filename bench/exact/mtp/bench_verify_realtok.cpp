// Minimal MTP verify-cost driver (real consecutive tokens, apples-to-apples
// with bench_verify_main.cpp's workload on origin/main).
//
// bench_verify.cpp (commit 179ea1b7) fed the SAME token to every row of a
// verify call (`std::vector<int32_t> toks(k, token)`), so k>1 rows never
// routed through distinct experts the way a real MTP draft window does.
// This file is bench_verify.cpp's own API (our branch's qwen4exp_forward /
// create_qwen4exp_cache signatures -- note the extra kv_type + reference
// params vs origin/main) with the same real-consecutive-token fix and
// rollback/advance logic bench_verify_main.cpp uses against main's API:
// each row gets a distinct, consecutive id from the ids file (wrapping once
// exhausted), and every verify forward is followed by
// qwen4exp_verify_rollback(..., retained=k) to mirror qwen4exp_mtp_step's
// real call pattern (full-acceptance) and keep the cache's ple_prev /
// indexer_blocks state valid for the next call's row_spans.
//
//   bench_verify_realtok MODEL MTP_SIDECAR IDS reps ks_csv max_ctx
#include "qwen4exp_internal.h"
#include "qwen4exp_graph.h"
#include "qwen4exp_cache.h"
#include "qwen4exp_mtp.h"
#include "ggml-cuda.h"
#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>
using namespace luce::common;
static std::vector<int32_t> read_ids(const char * path) {
    std::ifstream f(path); std::vector<int32_t> v; int t;
    while (f >> t) v.push_back(t);
    if (!f.eof() || v.empty()) { std::fprintf(stderr, "bad ids %s\n", path); std::exit(2); }
    return v;
}
int main(int argc, char ** argv) {
    if (argc != 7) { std::fprintf(stderr, "usage: MODEL SIDECAR IDS reps ks_csv max_ctx\n"); return 2; }
    const auto ids = read_ids(argv[3]);
    const int reps = std::stoi(argv[4]), max_ctx = std::stoi(argv[6]);
    std::vector<int> ks; { std::stringstream s(argv[5]); std::string p; while (std::getline(s, p, ',')) ks.push_back(std::stoi(p)); }
    const int mtp_draft = QWEN4EXP_MTP_MAX_DRAFT;
    for (int k : ks) {
        if (k > mtp_draft + 1) { std::fprintf(stderr, "width %d exceeds mtp_draft+1=%d\n", k, mtp_draft + 1); return 2; }
    }
    constexpr int WARMUP = 3;
    ggml_backend_t backend = ggml_backend_cuda_init(0);
    Qwen4ExpWeights w;
    if (!backend || !load_qwen4exp_gguf(argv[1], backend, w, argv[2], false, QWEN4EXP_MTP_VOCAB)) return 1;
    Qwen4ExpCache cache;
    if (!create_qwen4exp_cache(backend, w, max_ctx, GGML_TYPE_F16, cache, /*mtp=*/true, mtp_draft, false)) return 1;
    if (!qwen4exp_verify_supported(cache)) { std::fprintf(stderr, "verify unsupported\n"); return 1; }
    std::vector<float> logits;
    const int prefill = (int) ids.size() - 1;
    if (prefill <= 0) { std::fprintf(stderr, "ids file too short\n"); return 2; }
    if (!qwen4exp_forward(backend, w, cache, ids.data(), prefill, 0, logits).ok) return 1;
    int pos = prefill;
    size_t idx = (size_t) prefill;  // cursor into `ids`, wraps once exhausted
    const size_t n_ids = ids.size();
    auto now = [] { return std::chrono::steady_clock::now(); };
    for (int k : ks) {
        const bool verify = k > 1;
        std::vector<int32_t> toks(k);
        std::vector<double> ms_reps;
        for (int rep = 0; rep < reps + WARMUP; ++rep) {
            if (pos + k >= max_ctx) { std::fprintf(stderr, "max_ctx exhausted\n"); return 2; }
            // real, distinct, consecutive ids from the corpus (not the same
            // token repeated k times), so rows route through different
            // experts like a real draft window.
            for (int i = 0; i < k; ++i) toks[(size_t) i] = ids[(idx + (size_t) i) % n_ids];
            const auto t0 = now();
            if (!qwen4exp_forward(backend, w, cache, toks.data(), k, pos, logits, nullptr, verify).ok) return 1;
            // Mirror qwen4exp_mtp_step's real call pattern: every verify
            // forward is followed by qwen4exp_verify_rollback(retained=k),
            // i.e. "accept every drafted row", keeping cache.ple_prev /
            // indexer_blocks valid for the next call's row_spans.
            if (verify && !qwen4exp_verify_rollback(backend, w, cache, pos, k)) return 1;
            const double ms = std::chrono::duration<double, std::milli>(now() - t0).count();
            if (rep == reps + WARMUP - 1 && std::getenv("LUCE_PRINT_ARGMAX")) {
                // Cheap determinism check across A/B/C arms: sum of per-row
                // argmax token ids on the last rep of this width.
                const int n_vocab = (int) (logits.size() / (size_t) k);
                long long argmax_sum = 0;
                for (int row = 0; row < k; ++row) {
                    const float * row_logits = logits.data() + (size_t) row * n_vocab;
                    int best = 0;
                    for (int v = 1; v < n_vocab; ++v) {
                        if (row_logits[v] > row_logits[best]) best = v;
                    }
                    argmax_sum += best;
                }
                std::printf("[argmax] k=%d argmax_sum=%lld\n", k, argmax_sum);
            }
            pos += k;
            idx = (idx + (size_t) k) % n_ids;
            if (rep >= WARMUP) ms_reps.push_back(ms);
        }
        std::sort(ms_reps.begin(), ms_reps.end());
        const char * tag = verify ? "verify" : "decode";
        std::printf("[%s] k=%d median_ms=%.4f n=%zu pos=%d\n", tag, k, ms_reps[ms_reps.size() / 2], ms_reps.size(), pos);
        std::fflush(stdout);
    }
    free_qwen4exp_cache(cache);
    free_qwen4exp_weights(w);
    ggml_backend_free(backend);
    return 0;
}
