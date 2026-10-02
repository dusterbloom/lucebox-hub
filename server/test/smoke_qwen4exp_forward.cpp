// Smoke test for the Qwen3.8-Flash-Next (`qwen4exp`) forward path.
//
// Loads shard 1 (shard 2, the lazy PLE table, is discovered from the shard-1
// filename), builds the KV + delta-net cache, runs a prefill chunk and one
// decode step, and checks the logits are finite and the expected size. Mirrors
// smoke_qwen3_forward.cpp; no daemon involved.
//
// Usage:
//   smoke_qwen4exp_forward <shard1.gguf> [seq_len=16]
//
// QWEN4EXP_SMOKE_MTP=N (needs the MTP sidecar): greedy-decodes N tokens after the seq_len-token prompt with plain
// decode and with MTP speculation and fails unless both give the same tokens from bit-identical logits. Use a
// seq_len past 2052 to cover QSA decode, e.g.
//   QWEN4EXP_SMOKE_MTP=128 smoke_qwen4exp_forward <shard1.gguf> 2200

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

uint64_t hash_row(const float * row, size_t n) {   // FNV-1a over the bytes: equal iff bit-identical (in practice)
    uint64_t h = 1469598103934665603ull;
    const unsigned char * p = (const unsigned char *) row;
    for (size_t i = 0; i < n * sizeof(float); ++i) h = (h ^ p[i]) * 1099511628211ull;
    return h;
}

// QWEN4EXP_SMOKE_MTP: greedy-decode n_gen tokens after `prompt` twice -- plain T=1 steps, then MTP speculation
// (draft, two-token verify, rollback on reject) -- and require the same tokens from bit-identical logits at every
// position. Prefill runs in chunks of QWEN4EXP_SMOKE_CHUNK (default: the whole prompt).
int run_mtp_check(ggml_backend_t backend, const Qwen4ExpWeights & w, const std::vector<int32_t> & prompt, int n_gen) {
    Qwen4ExpCache cache;
    const int S = (int) prompt.size();
    if (!w.mtp_eh_proj || n_gen < 1 ||
        !create_qwen4exp_cache(backend, w, S + n_gen + 4, GGML_TYPE_F16, cache, /*mtp=*/true) ||
        !qwen4exp_verify_supported(cache)) {
        std::fprintf(stderr, "[smoke] mtp: needs a sidecar (QWEN4EXP_MTP or <repo>/MTP/mtp-*.gguf) and the default graph\n");
        free_qwen4exp_cache(cache);
        return 1;
    }
    const char * chunk_env = getenv("QWEN4EXP_SMOKE_CHUNK");
    const int chunk = chunk_env ? std::max(1, std::atoi(chunk_env)) : S;
    const size_t V = (size_t) w.n_vocab, hd = (size_t) w.n_embd * w.n_hc;
    auto top = [&](const float * row) { return (int32_t) (std::max_element(row, row + V) - row); };
    std::vector<float> logits, hidden, mtp_h, mtp_logits;
    std::vector<int32_t> mtp_tok;
    int mtp_pos = 0;

    // Prefill; with `mtp` also the draft layer's catch-up over the prompt (as Qwen4ExpBackend::generate_impl).
    auto prefill = [&](bool mtp) {
        reset_qwen4exp_state(backend, cache);
        mtp_h.clear();
        mtp_pos = 0;
        for (int pos = 0; pos < S; pos += chunk) {
            const int n = std::min(chunk, S - pos);
            if (!qwen4exp_forward(backend, w, cache, prompt.data() + pos, n, pos, logits, mtp ? &hidden : nullptr).ok) {
                return false;
            }
            if (!mtp) continue;
            mtp_tok.assign(prompt.begin() + pos + (pos == 0 ? 1 : 0), prompt.begin() + pos + n);
            mtp_h.insert(mtp_h.end(), hidden.begin(), hidden.end());
            const int n_pairs = (int) mtp_tok.size();
            if (n_pairs > 0 && !qwen4exp_mtp_forward(backend, w, cache, mtp_tok.data(), mtp_h.data(), n_pairs, mtp_pos,
                                                     mtp_logits)) {
                return false;
            }
            mtp_h.erase(mtp_h.begin(), mtp_h.begin() + (std::ptrdiff_t) ((size_t) n_pairs * hd));
            mtp_pos += n_pairs;
        }
        return true;
    };

    std::vector<int32_t> ref, out;
    std::vector<uint64_t> ref_hash, out_hash;   // hash of the logits each token was taken from
    bool ok = prefill(false);
    if (ok) {
        int32_t next = top(logits.data());
        ref.push_back(next);
        ref_hash.push_back(hash_row(logits.data(), V));
        for (int pos = S; ok && (int) ref.size() < n_gen; ++pos) {
            ok = qwen4exp_forward(backend, w, cache, &next, 1, pos, logits).ok;
            next = top(logits.data());
            ref.push_back(next);
            ref_hash.push_back(hash_row(logits.data(), V));
        }
    }

    long long drafts = 0, accepted = 0;
    const auto t0 = std::chrono::steady_clock::now();
    ok = ok && prefill(true);
    if (ok) {
        int32_t next = top(logits.data());
        out.push_back(next);
        out_hash.push_back(hash_row(logits.data(), V));
        mtp_tok.assign(1, next);
        int pos = S;
        while (ok && (int) out.size() < n_gen) {
            const bool verify = n_gen - (int) out.size() >= 2;
            int32_t draft = -1;
            if (verify) {
                ok = qwen4exp_mtp_forward(backend, w, cache, mtp_tok.data(), mtp_h.data(), (int) mtp_tok.size(),
                                          mtp_pos, mtp_logits);
                draft = top(mtp_logits.data());
            }
            const int32_t in[2] = { next, draft };
            ok = ok && qwen4exp_forward(backend, w, cache, in, verify ? 2 : 1, pos, logits,
                                        verify ? &hidden : nullptr, verify).ok;
            if (!ok) break;
            const int32_t tok = top(logits.data());
            out.push_back(tok);
            out_hash.push_back(hash_row(logits.data(), V));
            if (!verify) {
                ++pos;
                next = tok;
                continue;
            }
            ++drafts;
            mtp_h.assign(hidden.begin(), hidden.begin() + (std::ptrdiff_t) hd);
            mtp_tok.assign(1, tok);
            mtp_pos = pos;
            if (tok == draft) {
                ++accepted;
                pos += 2;
                if ((int) out.size() < n_gen) {
                    next = top(logits.data() + V);
                    out.push_back(next);
                    out_hash.push_back(hash_row(logits.data() + V, V));
                    mtp_h.insert(mtp_h.end(), hidden.begin() + (std::ptrdiff_t) hd, hidden.end());
                    mtp_tok.push_back(next);
                }
            } else {
                ok = qwen4exp_verify_rollback(backend, w, cache, pos);
                ++pos;
                next = tok;
            }
        }
    }
    const double spec_s = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    free_qwen4exp_cache(cache);

    int first_token = -1, first_logits = -1;
    for (size_t i = 0; ok && i < ref.size() && i < out.size(); ++i) {
        if (first_token < 0 && ref[i] != out[i]) first_token = (int) i;
        if (first_logits < 0 && ref_hash[i] != out_hash[i]) first_logits = (int) i;
    }
    const bool same = ok && ref.size() == out.size() && first_token < 0 && first_logits < 0;
    std::printf("[smoke] mtp S=%d n_gen=%d ok=%d tokens_identical=%d logits_identical=%d first_token_diff=%d "
                "first_logits_diff=%d drafts=%lld accepted=%lld rate=%.3f spec_s=%.2f\n",
                S, n_gen, (int) ok, (int) (ok && first_token < 0), (int) (ok && first_logits < 0), first_token,
                first_logits, drafts, accepted, drafts ? (double) accepted / (double) drafts : 0.0, spec_s);
    return same ? 0 : 1;
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
            ok = qwen4exp_forward(backend, w, reference, &tokens[p], 1, p, expected, nullptr, false, true).ok;
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

    if (const char * mtp_env = getenv("QWEN4EXP_SMOKE_MTP"); rc == 0 && mtp_env) {
        rc = run_mtp_check(backend, w, tokens, std::atoi(mtp_env));
    }

    free_qwen4exp_cache(cache);
    free_qwen4exp_weights(w);
    ggml_backend_free(backend);

    std::printf("[smoke] %s\n", rc == 0 ? "PASS" : "FAIL");
    return rc;
}
