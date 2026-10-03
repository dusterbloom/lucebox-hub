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
//   QWEN4EXP_MTP_DRAFT=4 QWEN4EXP_SMOKE_MTP=128 smoke_qwen4exp_forward <shard1.gguf> 2200
// Also forces every retained prefix at all four block alignments and compares
// cache bytes plus replacement-token logits, including QSA block recompletion.

#include "qwen4exp_internal.h"
#include "qwen4exp_graph.h"
#include "qwen4exp_cache.h"

#include "ggml-cuda.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
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

bool same_bits(const std::vector<float> & a, const std::vector<float> & b) {
    return a.size() == b.size() &&
        std::memcmp(a.data(), b.data(), a.size() * sizeof(float)) == 0;
}

uint64_t logits_hash(const std::vector<float> & x) {
    uint64_t h = 1469598103934665603ull;
    const uint8_t * p = reinterpret_cast<const uint8_t *>(x.data());
    for (size_t i = 0; i < x.size() * sizeof(float); ++i) { h ^= p[i]; h *= 1099511628211ull; }
    return h;
}

// QWEN4EXP_SMOKE_MTP: greedy-decode n_gen tokens after `prompt` twice -- plain T=1 steps, then MTP speculation
// (k chained drafts, k+1-token verify, rollback on reject) -- and require the same tokens from bit-identical logits at every
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
    bool prefill_kv_identical = true;

    // Full MTP is the oracle: compare the actual F16 K/V bytes for every prompt
    // pair, including the pair carried across a trunk chunk boundary.
    auto mtp_kv_bytes = [&](int pos, int n) {
        std::vector<char> bytes;
        for (ggml_tensor * t : {cache.mtp_k, cache.mtp_v}) {
            const size_t size = (size_t) n * t->nb[1];
            for (int64_t h = 0; h < t->ne[2]; ++h) {
                const size_t start = bytes.size();
                bytes.resize(start + size);
                ggml_backend_tensor_get(t, bytes.data() + start, h * t->nb[2] + (size_t) pos * t->nb[1], size);
            }
        }
        return bytes;
    };

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
            if (n_pairs > 0) {
                const auto expected = mtp_kv_bytes(mtp_pos, n_pairs);
                // Poison the destination so a missing/partial write cannot pass on stale oracle bytes.
                for (ggml_tensor * t : {cache.mtp_k, cache.mtp_v}) {
                    for (int64_t h = 0; h < t->ne[2]; ++h) {
                        ggml_backend_tensor_memset(t, 0xa5, h * t->nb[2] + (size_t) mtp_pos * t->nb[1],
                                                   (size_t) n_pairs * t->nb[1]);
                    }
                }
                if (!qwen4exp_mtp_forward(backend, w, cache, mtp_tok.data(), mtp_h.data(), n_pairs, mtp_pos,
                                          mtp_logits, nullptr, /*kv_only=*/true)) return false;
                prefill_kv_identical = expected == mtp_kv_bytes(mtp_pos, n_pairs) && mtp_logits.empty();
                if (!prefill_kv_identical) return false;
            }
            mtp_h.erase(mtp_h.begin(), mtp_h.begin() + (std::ptrdiff_t) ((size_t) n_pairs * hd));
            mtp_pos += n_pairs;
        }
        return true;
    };

    std::vector<int32_t> ref, out;
    std::vector<float> ref_rows;
    int first_logits = -1;
    bool ok = prefill(false);
    if (ok) {
        int32_t next = top(logits.data());
        ref.push_back(next);
        ref_rows.insert(ref_rows.end(), logits.begin(), logits.end());
        for (int pos = S; ok && (int) ref.size() < n_gen; ++pos) {
            ok = qwen4exp_forward(backend, w, cache, &next, 1, pos, logits).ok && all_finite(logits);
            next = top(logits.data());
            ref.push_back(next);
            ref_rows.insert(ref_rows.end(), logits.begin(), logits.end());
        }
    }

    long long drafts = 0, accepted = 0, steps = 0;
    const int configured_k = cache.mtp_draft;
    auto emit = [&](const float * row) {
        const size_t index = out.size();
        out.push_back(top(row));
        if (first_logits < 0 && (ref_rows.size() < (index + 1) * V ||
            std::memcmp(row, ref_rows.data() + index * V, V * sizeof(float)) != 0)) first_logits = (int) index;
    };
    const auto t0 = std::chrono::steady_clock::now();
    ok = ok && prefill(true);
    if (ok) {
        int32_t next = top(logits.data());
        emit(logits.data());
        mtp_tok.assign(1, next);
        int pos = S;
        std::vector<int32_t> draft_tokens;
        while (ok && (int) out.size() < n_gen) {
            const int k = std::min(configured_k, n_gen - (int) out.size() - 1);
            const bool verify = k > 0;
            if (verify) ok = qwen4exp_mtp_draft(backend, w, cache, mtp_tok.data(), mtp_h.data(),
                                                (int) mtp_tok.size(), mtp_pos, k, draft_tokens);
            std::array<int32_t, QWEN4EXP_MTP_MAX_VERIFY> in{}, samples{};
            in[0] = next;
            if (verify) std::copy(draft_tokens.begin(), draft_tokens.end(), in.begin() + 1);
            ok = ok && qwen4exp_forward(backend, w, cache, in.data(), k + 1, pos, logits,
                                        verify ? &hidden : nullptr, verify).ok;
            if (!ok || !all_finite(logits)) { ok = false; break; }
            ++steps;
            drafts += k;
            for (int i = 0; i <= k; ++i) samples[i] = top(logits.data() + (size_t) i * V);
            const auto decision = qwen4exp_mtp_accept(draft_tokens.data(), k, samples.data(), k + 1);
            accepted += decision.n_accepted;
            const int retained = decision.n_emitted;
            for (int i = 0; i < retained; ++i) emit(logits.data() + (size_t) i * V);
            next = decision.emitted[retained - 1];
            if (verify) {
                mtp_h.assign(hidden.begin(), hidden.begin() + (std::ptrdiff_t) ((size_t) retained * hd));
                mtp_tok.assign(decision.emitted.begin(), decision.emitted.begin() + retained);
                mtp_pos = pos;
                ok = qwen4exp_verify_rollback(backend, w, cache, pos, retained);
            }
            pos += retained;
        }
    }
    const double spec_s = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    std::printf("[smoke] mtp prefill S=%d chunk=%d kv_identical=%d\n", S, chunk, (int) (ok && prefill_kv_identical));
    free_qwen4exp_cache(cache);

    int first_token = -1;
    for (size_t i = 0; ok && i < ref.size() && i < out.size(); ++i) {
        if (first_token < 0 && ref[i] != out[i]) first_token = (int) i;
    }
    const bool same = ok && ref.size() == out.size() && first_token < 0 && first_logits < 0;
    std::printf("[smoke] mtp S=%d k=%d n_gen=%d ok=%d tokens_identical=%d logits_identical=%d first_token_diff=%d "
                "first_logits_diff=%d drafts=%lld accepted=%lld rate=%.3f tokens_per_step=%.3f spec_s=%.2f\n",
                S, configured_k, n_gen, (int) ok, (int) (ok && ref.size() == out.size() && first_token < 0),
                (int) (ok && ref.size() == out.size() && first_logits < 0), first_token,
                first_logits, drafts, accepted, drafts ? (double) accepted / (double) drafts : 0.0,
                steps ? (double) (out.size() - 1) / (double) steps : 0.0, spec_s);
    return same ? 0 : 1;
}

// Compare only authoritative cache rows; reset deliberately leaves unused K/V intact.
bool same_cache(const Qwen4ExpCache & a, const Qwen4ExpCache & b, int tokens, bool same_pooled_count = true) {
    if ((same_pooled_count && a.indexer_blocks != b.indexer_blocks) || a.ple_prev != b.ple_prev) return false;
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
        tensors_equal(a.indexer_k, b.indexer_k, std::min(a.indexer_blocks, b.indexer_blocks)) &&
        tensors_equal(a.ssm_state, b.ssm_state, -1) && tensors_equal(a.conv_state, b.conv_state, -1) &&
        tensors_equal(a.ple_conv_state, b.ple_conv_state, -1);
}

// Exercise every retained prefix and every block alignment, independent of the
// model's natural acceptance rate. After truncation replace the first rejected
// input and finish its block; compare authoritative state and logits with AR.
int run_mtp_rollback_check(ggml_backend_t backend, const Qwen4ExpWeights & w,
                           const std::vector<int32_t> & prompt) {
    Qwen4ExpCache cache, reference;
    const int S = (int) prompt.size();
    const int capacity = S + 512;
    bool ok = create_qwen4exp_cache(backend, w, capacity, GGML_TYPE_F16, cache, true) &&
              create_qwen4exp_cache(backend, w, capacity, GGML_TYPE_F16, reference);
    const int k = cache.mtp_draft;
    std::vector<float> actual, expected, verified;
    ok = ok && qwen4exp_forward(backend, w, cache, prompt.data(), S, 0, actual).ok &&
               qwen4exp_forward(backend, w, reference, prompt.data(), S, 0, expected).ok;
    int pos = S, cases = 0;
    auto advance = [&](int32_t token) {
        const bool same = qwen4exp_forward(backend, w, cache, &token, 1, pos, actual).ok &&
                          qwen4exp_forward(backend, w, reference, &token, 1, pos, expected).ok &&
            actual.size() == expected.size() && all_finite(actual) &&
            std::memcmp(actual.data(), expected.data(), actual.size() * sizeof(float)) == 0 &&
            cache.cur_pos == pos + 1 && cache.indexer_blocks <= (pos + 1) / 4 &&
            (cache.indexer_blocks == reference.indexer_blocks || reference.indexer_blocks == 0) &&
            same_cache(cache, reference, pos + 1, false);
        ++pos;
        return same;
    };
    for (int retained = 1; ok && retained <= k + 1; ++retained) {
        for (int alignment = 0; ok && alignment < 4; ++alignment) {
            while (ok && pos % 4 != alignment) ok = advance((pos * 7919 + 13) % w.n_vocab);
            std::array<int32_t, QWEN4EXP_MTP_MAX_VERIFY> in{};
            for (int i = 0; i <= k; ++i) in[i] = ((pos + i) * 7919 + 13) % w.n_vocab;
            ok = ok && qwen4exp_forward(backend, w, cache, in.data(), k + 1, pos, verified, nullptr, true).ok;
            for (int i = 0; ok && i < retained; ++i) {
                ok = qwen4exp_forward(backend, w, reference, &in[i], 1, pos + i, expected).ok &&
                     all_finite(expected) &&
                     std::memcmp(verified.data() + (size_t) i * w.n_vocab, expected.data(),
                                 expected.size() * sizeof(float)) == 0;
            }
            ok = ok && qwen4exp_verify_rollback(backend, w, cache, pos, retained) &&
                 cache.cur_pos == pos + retained && cache.indexer_blocks <= (pos + retained) / 4 &&
                 // Verify may bootstrap QSA ahead of a still-dense reference.
                 (cache.indexer_blocks == reference.indexer_blocks || reference.indexer_blocks == 0) &&
                 same_cache(cache, reference, pos + retained, false);
            pos += retained;
            // Four replacements guarantee that a rejected block completion is
            // recomputed, even when the retained position is block-aligned.
            for (int i = 0; ok && i < 4; ++i) ok = advance((pos * 7919 + 14) % w.n_vocab);
            if (ok) ++cases;
            else std::fprintf(stderr, "[smoke] rollback mismatch k=%d retained=%d alignment=%d pos=%d\n",
                              k, retained, alignment, pos);
        }
    }
    std::printf("[smoke] mtp rollback S=%d k=%d cases=%d/%d cache_and_logits_identical=%d\n",
                S, k, cases, 4 * (k + 1), (int) ok);
    free_qwen4exp_cache(cache);
    free_qwen4exp_cache(reference);
    return ok ? 0 : 1;
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
        const int decode_rc = run_mtp_check(backend, w, tokens, std::atoi(mtp_env));
        // Independent caches: report rollback even when natural drafting differs.
        const int rollback_rc = run_mtp_rollback_check(backend, w, tokens);
        rc = decode_rc || rollback_rc;
    }

    // A reset/reuse must match a freshly allocated cache for the same prefix.
    if (rc == 0) {
        Qwen4ExpCache fresh;
        if (!create_qwen4exp_cache(backend, w, S + 4, GGML_TYPE_F16, fresh)) {
            std::fprintf(stderr, "[smoke] fresh cache creation failed\n");
            rc = 1;
        } else {
            reset_qwen4exp_state(backend, cache);
            reset_qwen4exp_state(backend, fresh);
            std::vector<float> tmp, reused_logits, fresh_logits;
            const bool p1 = qwen4exp_forward(backend, w, cache, tokens.data(), S, 0, tmp).ok;
            const bool p2 = qwen4exp_forward(backend, w, fresh, tokens.data(), S, 0, tmp).ok;
            const int32_t reuse_token = 77 % w.n_vocab;
            const Qwen4ExpForwardResult rr = qwen4exp_forward(backend, w, cache, &reuse_token, 1, S, reused_logits);
            const Qwen4ExpForwardResult fr = qwen4exp_forward(backend, w, fresh, &reuse_token, 1, S, fresh_logits);
            if (!p1 || !p2 || !rr.ok || !fr.ok || !same_bits(reused_logits, fresh_logits)) {
                std::fprintf(stderr, "[smoke] cancel-reset-reuse FAILED\n");
                rc = 1;
            } else {
                std::printf("[smoke] cancel-reset-reuse OK reused_hash=%016llx fresh_cache_hash=%016llx\n",
                    (unsigned long long) logits_hash(reused_logits),
                    (unsigned long long) logits_hash(fresh_logits));
            }
            free_qwen4exp_cache(fresh);
        }
    }

    free_qwen4exp_cache(cache);
    free_qwen4exp_weights(w);
    ggml_backend_free(backend);

    std::printf("[smoke] %s\n", rc == 0 ? "PASS" : "FAIL");
    return rc;
}
