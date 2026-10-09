// Minimal MTP verify-cost driver against Luce-Org/lucebox origin/main's
// qwen4exp API (commit 49c6b015 at write time). Companion to bench_verify.cpp
// (our own branch's API, commit 179ea1b7); see VERIFY-COST.md for background
// and the workload flaw this file fixes (uniform-token verify rows never
// routing through real experts).
//
//   bench_verify_main MODEL MTP_SIDECAR IDS reps widths_csv prefill_len
//
// widths_csv is a list of n_tokens values passed straight to qwen4exp_forward
// (k=1 -> plain T=1 decode reference, verify=false; k>1 -> verify=true, same
// meaning "k" had in bench_verify.cpp). max_ctx is not a CLI arg on main's
// API (create_qwen4exp_cache dropped the ggml_type kv-type param, not the
// max_ctx param) -- it is computed from prefill_len + the token budget every
// width/rep will consume, plus slack; see max_ctx below.
//
// ─── Build note (NOT compiled on this box; GPU box is offline) ────────────
// This file was written by reading origin/main's headers only. From a main
// build dir (server/CMakeLists.txt, cuda or hip arm), the minimal manual
// link line mirrors smoke_qwen4exp_forward's:
//   target_include_directories: ${LUCE_SRC_INCLUDE_DIRS} i.e.
//     server/src, server/src/common, server/src/qwen4exp, server/src/server
//     (see server/CMakeLists.txt:525-537), plus
//     deps/llama.cpp/ggml/include and deps/llama.cpp/ggml/src for ggml.h /
//     ggml-cuda.h / ggml-backend.h / ggml-alloc.h.
//   target_link_libraries: luce_common ggml ${LUCE_GGML_BACKEND_TARGET}
//     (ggml-cuda or ggml-hip; see server/CMakeLists.txt:1942, the
//     smoke_qwen4exp_forward target) plus ggml-base, and CUDA::cudart or
//     hip::host per LUCE_GPU_BACKEND (server/CMakeLists.txt:1045-1048).
//   libluce_common.a is produced by server/CMakeLists.txt:540
//     (add_library(luce_common STATIC ...)), which bundles
//     qwen4exp_loader.cpp / qwen4exp_cache.cpp / qwen4exp_graph.cpp /
//     qwen4exp_seq_engine.cpp / qwen4exp_backend.cpp (lines 548-552).
// Uncertain: whether a standalone TU outside the CMake tree needs any of the
// backend-specific compile definitions set at server/CMakeLists.txt:728-748
// (e.g. GGML_USE_HIP, LUCE_BACKEND_HIP) -- those are set PRIVATE on
// luce_common itself, not propagated to consumers, so a hand-linked bench
// should not need them, but this is unverified without a build.
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
    std::ifstream f(path);
    std::vector<int32_t> v;
    int t;
    while (f >> t) v.push_back(t);
    if (!f.eof() || v.empty()) { std::fprintf(stderr, "bad ids %s\n", path); std::exit(2); }
    return v;
}

int main(int argc, char ** argv) {
    if (argc != 7) {
        std::fprintf(stderr, "usage: MODEL SIDECAR IDS reps widths_csv prefill_len\n");
        return 2;
    }
    const std::vector<int32_t> ids = read_ids(argv[3]);
    const int reps = std::stoi(argv[4]);
    const int prefill_len = std::stoi(argv[6]);
    std::vector<int> ks;
    {
        std::stringstream s(argv[5]);
        std::string p;
        while (std::getline(s, p, ',')) ks.push_back(std::stoi(p));
    }
    if (prefill_len <= 0 || (int) ids.size() < prefill_len) {
        std::fprintf(stderr, "ids file too short for prefill_len=%d (have %zu)\n", prefill_len, ids.size());
        return 2;
    }

    // qwen4exp's retained verify graph is keyed by n_tokens (array index, see
    // qwen4exp_graph.cpp: cache.verify_workspace[T]), sized
    // QWEN4EXP_MTP_MAX_VERIFY + 1 = mtp_draft + 2 slots (qwen4exp_backend.cpp:
    // QWEN4EXP_MTP_MAX_DRAFT = 7, QWEN4EXP_MTP_MAX_VERIFY = 8). Keep mtp_draft
    // at the default 7 so every width up to 8 has a slot; forward_impl rejects
    // verify calls with n_tokens > cache.mtp_draft + 1 (qwen4exp_graph.cpp:2091).
    const int mtp_draft = QWEN4EXP_MTP_MAX_DRAFT;
    for (int k : ks) {
        if (k > mtp_draft + 1) {
            std::fprintf(stderr, "width %d exceeds mtp_draft+1=%d\n", k, mtp_draft + 1);
            return 2;
        }
    }

    constexpr int WARMUP = 3;  // >= the 2 calls ggml-cuda's graph-replay warmup needs
                                // (ggml_cuda_graph_update_required / capture-then-replay).

    // No max_ctx CLI arg on main's create_qwen4exp_cache (the ggml_type kv-type
    // param was dropped, not max_ctx -- see qwen4exp_cache.h). Size it from the
    // actual token budget every width/rep will advance pos by, plus slack so a
    // stable-graph KV-bucket step (256-token windows, qwen4exp_stable_kv_span)
    // does not get crossed mid-measurement for the widest k.
    int64_t budget = prefill_len;
    for (int k : ks) budget += (int64_t) (reps + WARMUP) * k;
    budget += 4096;
    const int max_ctx = (int) budget;

    ggml_backend_t backend = ggml_backend_cuda_init(0);
    Qwen4ExpWeights w;
    // load_qwen4exp_gguf(path, backend, out, mtp_override, mtp_vocab): main
    // dropped the `reference` bool our branch's qwen4exp_internal.h still has
    // (5 params now, not 6) -- qwen4exp_internal.h:260-264 on origin/main.
    if (!backend || !load_qwen4exp_gguf(argv[1], backend, w, argv[2], QWEN4EXP_MTP_VOCAB)) return 1;

    Qwen4ExpCache cache;
    // create_qwen4exp_cache(backend, w, max_ctx, cache, mtp, mtp_draft,
    // slot_states, state_slot): main dropped the ggml_type kv_type param (now
    // hardcoded GGML_TYPE_F16 internally, qwen4exp_cache.cpp:62) and the
    // `reference` bool, and added slot_states/state_slot for the shared
    // concurrency-slot recurrent state (unused here: single sequence, no
    // sharing) -- qwen4exp_cache.h:201-204 on origin/main.
    if (!create_qwen4exp_cache(backend, w, max_ctx, cache, /*mtp=*/true, mtp_draft,
                               /*slot_states=*/nullptr, /*state_slot=*/-1)) return 1;
    if (!qwen4exp_verify_supported(cache)) { std::fprintf(stderr, "verify unsupported\n"); return 1; }

    std::vector<float> logits;
    // Prefill: plain forward, verify=false, no MTP hidden capture -- matches
    // bench_verify.cpp's prefill call. qwen4exp_forward's trailing params
    // (out_hidden, verify, mtp_prefill, inputs) default to null/false/false/
    // null, same defaults as the old API (qwen4exp_graph.h:105-115).
    if (!qwen4exp_forward(backend, w, cache, ids.data(), prefill_len, 0, logits).ok) return 1;

    int pos = prefill_len;
    size_t idx = (size_t) prefill_len;  // cursor into `ids`, wraps once exhausted
    const size_t n_ids = ids.size();
    auto now = [] { return std::chrono::steady_clock::now(); };

    for (int k : ks) {
        const bool verify = k > 1;
        std::vector<int32_t> toks(k);
        std::vector<double> ms_reps;
        for (int rep = 0; rep < reps + WARMUP; ++rep) {
            if (pos + k >= max_ctx) { std::fprintf(stderr, "max_ctx exhausted\n"); return 2; }
            // Fix for the uniform-token flaw: each row gets a real, distinct,
            // consecutive id from the corpus (wrapping into `ids` once the
            // cursor runs past prefill_len + everything consumed so far), so
            // rows route through different experts like real drafts instead
            // of all hitting the same expert set as a single repeated token.
            for (int i = 0; i < k; ++i) toks[(size_t) i] = ids[(idx + (size_t) i) % n_ids];
            const auto t0 = now();
            if (!qwen4exp_forward(backend, w, cache, toks.data(), k, pos, logits, nullptr, verify).ok) return 1;
            // Mirror qwen4exp_mtp_step's real call pattern (qwen4exp_graph.cpp:
            // 2696-2719): every verify forward is followed by
            // qwen4exp_verify_rollback. forward_impl already advances
            // cache.cur_pos to pos+k unconditionally for a verify call
            // (qwen4exp_graph.cpp:2189), so skipping rollback would leave
            // cur_pos correct but cache.ple_prev / indexer_blocks at their
            // last-row values without going through the retained-row device
            // copy path -- calling rollback with retained=k ("accept every
            // drafted row") keeps the cache in the same state a real
            // full-acceptance step would leave it in, so the next call's
            // row_spans (and therefore its retained verify graph) stay valid.
            if (verify && !qwen4exp_verify_rollback(backend, w, cache, pos, k)) return 1;
            const double ms = std::chrono::duration<double, std::milli>(now() - t0).count();
            pos += k;
            idx = (idx + (size_t) k) % n_ids;
            if (rep >= WARMUP) ms_reps.push_back(ms);
        }
        std::sort(ms_reps.begin(), ms_reps.end());
        const char * tag = verify ? "verify" : "decode";  // k=1 prints the T=1 decode reference
        std::printf("[%s] k=%d median_ms=%.4f n=%zu pos=%d\n",
                     tag, k, ms_reps[ms_reps.size() / 2], ms_reps.size(), pos);
        // Graph-replay detection without GGML_CUDA_GRAPH_STATS=1: both
        // Qwen4ExpDecodeWorkspace::builds and ::replays are plain public
        // counters already bumped by forward_impl (builds on a graph
        // (re)build, replays on a retained-graph replay -- qwen4exp_cache.h:
        // 65-66, incremented at qwen4exp_graph.cpp:2190 and further down on
        // the build path). verify_workspace is indexed by n_tokens.
        const Qwen4ExpDecodeWorkspace & ws = verify ? cache.verify_workspace[(size_t) k] : cache.decode_workspace;
        std::printf("[%s] k=%d builds=%llu replays=%llu\n", tag, k,
                     (unsigned long long) ws.builds, (unsigned long long) ws.replays);
        std::fflush(stdout);
    }

    free_qwen4exp_cache(cache);
    free_qwen4exp_weights(w);
    ggml_backend_free(backend);
    return 0;
}
