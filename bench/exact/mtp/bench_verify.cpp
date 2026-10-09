// Minimal MTP verify-cost driver: times qwen4exp_forward at fixed widths k
// (k=1 plain decode reference, k>1 verify=true) on a stable 8K-ish context.
//   bench_verify MODEL MTP_SIDECAR IDS reps ks_csv max_ctx
#include "qwen4exp_internal.h"
#include "qwen4exp_graph.h"
#include "qwen4exp_cache.h"
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
    const auto prompt = read_ids(argv[3]);
    const int reps = std::stoi(argv[4]), max_ctx = std::stoi(argv[6]);
    std::vector<int> ks; { std::stringstream s(argv[5]); std::string p; while (std::getline(s, p, ',')) ks.push_back(std::stoi(p)); }
    ggml_backend_t backend = ggml_backend_cuda_init(0);
    Qwen4ExpWeights w;
    if (!backend || !load_qwen4exp_gguf(argv[1], backend, w, argv[2], false, QWEN4EXP_MTP_VOCAB)) return 1;
    Qwen4ExpCache cache;
    if (!create_qwen4exp_cache(backend, w, max_ctx, GGML_TYPE_F16, cache, /*mtp=*/true, /*mtp_draft=*/7, false)) return 1;
    if (!qwen4exp_verify_supported(cache)) { std::fprintf(stderr, "verify unsupported\n"); return 1; }
    std::vector<float> logits;
    const int prefill = (int) prompt.size() - 1;
    if (!qwen4exp_forward(backend, w, cache, prompt.data(), prefill, 0, logits).ok) return 1;
    int pos = prefill;
    const int32_t token = prompt[prefill];
    auto now = [] { return std::chrono::steady_clock::now(); };
    for (int k : ks) {
        const bool verify = k > 1;
        std::vector<int32_t> toks(k, token);
        std::vector<double> ms_reps;
        for (int rep = 0; rep < reps + 3; ++rep) {
            if (pos + k >= max_ctx) { std::fprintf(stderr, "max_ctx exhausted\n"); return 2; }
            const auto t0 = now();
            if (!qwen4exp_forward(backend, w, cache, toks.data(), k, pos, logits, nullptr, verify).ok) return 1;
            const double ms = std::chrono::duration<double, std::milli>(now() - t0).count();
            pos += k;
            if (rep >= 3) ms_reps.push_back(ms);
        }
        std::sort(ms_reps.begin(), ms_reps.end());
        std::printf("[verify] k=%d median_ms=%.4f n=%zu pos=%d\n", k, ms_reps[ms_reps.size() / 2], ms_reps.size(), pos);
        std::fflush(stdout);
    }
    free_qwen4exp_cache(cache);
    free_qwen4exp_weights(w);
    ggml_backend_free(backend);
    return 0;
}
