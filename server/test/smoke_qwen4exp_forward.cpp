// Smoke test for the Qwen3.8-Flash-Next (`qwen4exp`) forward path.
//
// Loads shard 1 (shard 2, the lazy PLE table, is discovered from the shard-1
// filename), builds the KV + delta-net cache, runs a prefill chunk and one
// decode step, and checks the logits are finite and the expected size. Mirrors
// smoke_qwen3_forward.cpp; no daemon involved.
//
// Usage:
//   smoke_qwen4exp_forward <shard1.gguf> [seq_len=16]

#include "qwen4exp_internal.h"
#include "qwen4exp_graph.h"
#include "qwen4exp_cache.h"

#include "ggml-cuda.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>
#include <fstream>

using namespace luce::common;

namespace {

bool all_finite(const std::vector<float> & v) {
    for (const float f : v) {
        if (!std::isfinite(f)) {
            return false;
        }
    }
    return true;
}

int argmax(const std::vector<float> & v) {
    int best = 0;
    for (size_t i = 1; i < v.size(); ++i) {
        if (v[i] > v[best]) {
            best = (int) i;
        }
    }
    return best;
}

// Compare only authoritative cache rows; reset deliberately leaves unused K/V intact.
bool same_cache(const Qwen4ExpCache & a, const Qwen4ExpCache & b, int tokens) {
    if (a.indexer_blocks != b.indexer_blocks || a.ple_prev != b.ple_prev) return false;
    auto tensors_equal = [](const std::vector<ggml_tensor *> & x,
                            const std::vector<ggml_tensor *> & y, int rows) {
        if (x.size() != y.size()) return false;
        for (size_t i = 0; i < x.size(); ++i) {
            if (!x[i] || !y[i]) { if (x[i] != y[i]) return false; else continue; }
            const size_t bytes = rows < 0 ? ggml_nbytes(x[i]) : (size_t) rows * x[i]->nb[1];
            std::vector<char> xb(bytes), yb(bytes);
            for (int64_t h = 0; h < (rows < 0 ? 1 : x[i]->ne[2]); ++h) {
                ggml_backend_tensor_get(x[i], xb.data(), h * x[i]->nb[2], bytes);
                ggml_backend_tensor_get(y[i], yb.data(), h * y[i]->nb[2], bytes);
                if (xb != yb) return false;
            }
        }
        return true;
    };
    return tensors_equal(a.attn_k, b.attn_k, tokens) && tensors_equal(a.attn_v, b.attn_v, tokens) &&
        tensors_equal(a.indexer_raw, b.indexer_raw, tokens) &&
        tensors_equal(a.indexer_k, b.indexer_k, a.indexer_blocks) &&
        tensors_equal(a.ssm_state, b.ssm_state, -1) && tensors_equal(a.conv_state, b.conv_state, -1) &&
        tensors_equal(a.ple_conv_state, b.ple_conv_state, -1);
}

}  // namespace

int main(int argc, char ** argv) {
    if (argc < 2) {
        std::fprintf(stderr, "usage: %s <shard1.gguf> [seq_len=16]\n", argv[0]);
        return 2;
    }
    const std::string path = argv[1];
    const int S = (argc >= 3) ? std::atoi(argv[2]) : 16;
    if (S < 1) {
        std::fprintf(stderr, "[smoke] seq_len must be >= 1\n");
        return 2;
    }

    ggml_backend_t backend = ggml_backend_cuda_init(0);
    if (!backend) {
        std::fprintf(stderr, "[smoke] no GPU backend available\n");
        return 77;
    }

    Qwen4ExpWeights w;
    auto t_load0 = std::chrono::steady_clock::now();
    if (!load_qwen4exp_gguf(path, backend, w)) {
        std::fprintf(stderr, "[smoke] load_qwen4exp_gguf failed\n");
        ggml_backend_free(backend);
        return 1;
    }
    auto t_load1 = std::chrono::steady_clock::now();
    std::printf("[smoke] load %.2fs layers=%d vocab=%d shard2=%s\n",
        std::chrono::duration<double>(t_load1 - t_load0).count(),
        w.n_layer, w.n_vocab, w.ple_reader.available() ? "yes" : "no");

    const char * tg_env = getenv("QWEN4EXP_SMOKE_TG");
    const int n_gen = tg_env ? std::max(1, std::atoi(tg_env)) : 1;
    Qwen4ExpCache cache;
    if (!create_qwen4exp_cache(backend, w, S + std::max(4, n_gen), GGML_TYPE_F16, cache)) {
        std::fprintf(stderr, "[smoke] create_qwen4exp_cache failed\n");
        free_qwen4exp_weights(w);
        ggml_backend_free(backend);
        return 1;
    }

    std::vector<int32_t> tokens((size_t) S);
    for (int i = 0; i < S; ++i) {
        tokens[(size_t) i] = (int32_t) ((i * 7919 + 13) % w.n_vocab);
    }

    if (const char * file = getenv("QWEN4EXP_TOKEN_FILE")) {
        std::ifstream input(file);
        for (int i = 0; i < S; ++i) {
            if (!(input >> tokens[i]) || tokens[i] < 0 || tokens[i] >= w.n_vocab) {
                std::fprintf(stderr, "invalid QWEN4EXP_TOKEN_FILE\n");
                return 2;
            }
        }
        int extra;
        if (input >> extra) { std::fprintf(stderr, "too many input tokens\n"); return 2; }
    }
    std::vector<float> logits;
    auto t0 = std::chrono::steady_clock::now();
    const Qwen4ExpForwardResult pre = qwen4exp_forward(backend, w, cache, tokens.data(), S, 0, logits);
    auto t1 = std::chrono::steady_clock::now();

    int rc = 0;
    if (!pre.ok || logits.size() != (size_t) w.n_vocab || !all_finite(logits)) {
        std::fprintf(stderr, "[smoke] prefill FAILED ok=%d logits=%zu finite=%d\n",
            (int) pre.ok, logits.size(), (int) all_finite(logits));
        rc = 1;
    } else {
        std::printf("[smoke] prefill OK %.3fs T=%d argmax=%d\n",
            std::chrono::duration<double>(t1 - t0).count(), S, argmax(logits));
    }

    if (rc == 0) {
        auto d0 = std::chrono::steady_clock::now();
        for (int i = 0; i < n_gen; ++i) {
            const int32_t next = (int32_t) argmax(logits);
            const auto dec = qwen4exp_forward(backend, w, cache, &next, 1, S + i, logits);
            if (!dec.ok || logits.size() != (size_t) w.n_vocab || !all_finite(logits)) {
                std::fprintf(stderr, "[smoke] decode FAILED pos=%d\n", S + i);
                rc = 1;
                break;
            }
        }
        const double seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - d0).count();
        std::printf("[smoke] tg%d ok=%d %.3fs %.3f tok/s argmax=%d\n",
            n_gen, rc == 0, seconds, n_gen / seconds, argmax(logits));
    }

    // Optional exact differential: identical prefill, then every T=1 step with
    // stable QSA enabled vs the original per-step rebuild (including cache bits).
    if (const char * check = getenv("QWEN4EXP_SMOKE_STABLE"); rc == 0 && check) {
        const int n = std::atoi(check), start = S - n;
        Qwen4ExpCache reference;
        bool ok = n > 0 && n < S && create_qwen4exp_cache(backend, w, cache.max_ctx, GGML_TYPE_F16, reference);
        std::vector<float> expected, actual;
        reset_qwen4exp_state(backend, cache);
        ok = ok && qwen4exp_forward(backend, w, cache, tokens.data(), start, 0, actual).ok &&
                   qwen4exp_forward(backend, w, reference, tokens.data(), start, 0, expected).ok;
        int replays = 0;
        for (int p = start; ok && p < S; ++p) {
            ok = qwen4exp_forward(backend, w, reference, &tokens[p], 1, p, expected, true).ok;
            const auto & ws = cache.decode_workspace;
            const bool replay = ws.gf && ws.qsa_blocks > 0 && ws.next_pos == p &&
                                cache.indexer_blocks == p / 4 && p + 1 <= ws.kv_bucket;
            const uint64_t builds = ws.builds, prior_replays = ws.replays;
            ok = ok && qwen4exp_forward(backend, w, cache, &tokens[p], 1, p, actual).ok;
            ok = ok && actual.size() == expected.size() &&
                 std::memcmp(actual.data(), expected.data(), actual.size() * sizeof(float)) == 0 &&
                 same_cache(cache, reference, p + 1) &&
                 (!replay || (ws.builds == builds && ws.replays == prior_replays + 1));
            replays += replay;
            if (!ok) std::fprintf(stderr, "[smoke] stable/rebuild mismatch at pos=%d\n", p);
        }
        free_qwen4exp_cache(reference);
        std::printf("[smoke] stable/rebuild bits ok=%d QSA replays=%d\n", (int) ok, replays);
        if (!ok || replays == 0) rc = 1;
    }

    // QWEN4EXP_SMOKE_SPLIT=N[:c]: the same S tokens as one prefill vs a prefill of S-N plus the last N tokens in
    // chunks of c (default 1, i.e. decode) must give the same last-position distribution. QSA selection is
    // chunk-invariant, so past the block budget this checks decode (c=1) against prefill; c in 9..127 runs dense
    // attention there and shows how far a wrong selection drifts.
    if (const char * split_env = getenv("QWEN4EXP_SMOKE_SPLIT"); rc == 0 && split_env) {
        const int N = std::atoi(split_env);
        const char * colon = std::strchr(split_env, ':');
        const int step = colon ? std::max(1, std::atoi(colon + 1)) : 1;
        std::vector<float> full, split;
        reset_qwen4exp_state(backend, cache);
        bool ok = N > 0 && N < S && qwen4exp_forward(backend, w, cache, tokens.data(), S, 0, full).ok;
        reset_qwen4exp_state(backend, cache);
        ok = ok && qwen4exp_forward(backend, w, cache, tokens.data(), S - N, 0, split).ok;
        for (int i = S - N; ok && i < S; i += step) {
            const int n = std::min(step, S - i);
            ok = qwen4exp_forward(backend, w, cache, &tokens[i], n, i, split).ok;
        }
        float max_diff = 0.0f;
        double kl = 0.0;
        int overlap = 0;
        if (ok) {
            auto softmax = [](const std::vector<float> & v) {
                const float mx = *std::max_element(v.begin(), v.end());
                std::vector<double> p(v.size());
                double z = 0.0;
                for (size_t i = 0; i < v.size(); ++i) z += (p[i] = std::exp((double) v[i] - mx));
                for (double & x : p) x /= z;
                return p;
            };
            auto top10 = [](const std::vector<float> & v) {
                std::vector<int> idx(v.size());
                for (size_t i = 0; i < v.size(); ++i) idx[i] = (int) i;
                std::partial_sort(idx.begin(), idx.begin() + 10, idx.end(), [&](int a, int b) { return v[a] > v[b]; });
                idx.resize(10);
                return idx;
            };
            const std::vector<double> pf = softmax(full), ps = softmax(split);
            for (size_t i = 0; i < full.size(); ++i) {
                max_diff = std::max(max_diff, std::fabs(full[i] - split[i]));
                if (pf[i] > 0.0) kl += pf[i] * std::log(pf[i] / std::max(ps[i], 1e-300));
            }
            const std::vector<int> tf = top10(full), ts = top10(split);
            for (int a : tf) overlap += (int) std::count(ts.begin(), ts.end(), a);
        }
        std::printf("[smoke] split S=%d N=%d chunk=%d ok=%d argmax %d vs %d max_abs_diff=%.4f kl=%.6f top10_overlap=%d\n",
            S, N, step, (int) ok, ok ? argmax(full) : -1, ok ? argmax(split) : -1, max_diff, kl, overlap);
        if (!ok || argmax(full) != argmax(split)) rc = 1;
    }

    free_qwen4exp_cache(cache);
    free_qwen4exp_weights(w);
    ggml_backend_free(backend);

    std::printf("[smoke] %s\n", rc == 0 ? "PASS" : "FAIL");
    return rc;
}
