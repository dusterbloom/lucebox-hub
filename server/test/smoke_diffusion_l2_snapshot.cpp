// Smoke test: DiffusionGemma L2' inter-block snapshot.
//
// Two gate checks:
//   1. single-block invariance: with L2 disabled (cfg.enable_l2_interblock=false)
//      vs L2 enabled, output of a single block is identical — the snapshot save
//      after block 1 does not corrupt the L2-disabled arm, and both arms use
//      the same gemma4_denoise_canvas path. L2' hooks are a strict no-op when
//      there is no next block to restore from.
//
//   2. multi-block generation: generates n_gen >= 2*block_size tokens with L2
//      enabled and verifies it completes without error. The reference is the
//      full-batch path (DG_NO_L0_CACHE=1), which recomputes the full sequence
//      every denoising step (correct but ~n_blocks× slower). We print whether
//      outputs match; they should be equivalent but F16-KV vs F32-batch
//      precision may cause minor divergence, so this is advisory-only.
//
// Usage:
//   smoke_diffusion_l2_snapshot <model.gguf> <prompt.bin>
//       [n_gen=512] [block_size=256] [steps=8] [seed=42]

#include "diffusion_backend.h"
#include "diffusion_decoder.h"
#include "diffusion_types.h"
#include "diffusiongemma/diffusion_gemma.h"

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

using namespace dflash::common;

static std::vector<int32_t> read_i32_bin(const char * path) {
    std::ifstream f(path, std::ios::binary | std::ios::ate);
    if (!f) throw std::runtime_error(std::string("cannot open: ") + path);
    const size_t sz = (size_t)f.tellg();
    if (sz % 4 != 0) throw std::runtime_error("file size not a multiple of 4");
    f.seekg(0);
    std::vector<int32_t> v(sz / 4);
    f.read(reinterpret_cast<char *>(v.data()), (std::streamsize)sz);
    return v;
}

static void print_tokens(const char * label, const std::vector<int32_t> & toks) {
    std::printf("%s", label);
    const int show = std::min((int)toks.size(), 32);
    for (int i = 0; i < show; ++i) std::printf(" %d", toks[i]);
    if ((int)toks.size() > show) std::printf(" ...[%d total]", (int)toks.size());
    std::printf("\n");
}

struct RunResult {
    std::vector<int32_t> tokens;
    double               wall_s = 0.0;
};

static RunResult run_generate(const char * model_path,
                              const std::vector<int32_t> & prompt,
                              int n_gen, int block_size, int steps,
                              uint64_t seed,
                              bool l2_enabled,
                              bool no_l0_cache = false) {
    if (no_l0_cache) setenv("DG_NO_L0_CACHE", "1", 1);
    else             unsetenv("DG_NO_L0_CACHE");

    const int min_ctx = (int)prompt.size() + n_gen;
    const int max_ctx = ((std::max(min_ctx, 512) + 63) / 64) * 64;

    DiffusionGemmaConfig gcfg;
    gcfg.model_path = model_path;
    gcfg.gpu        = 0;
    gcfg.max_ctx    = max_ctx;

    auto graph = std::make_unique<DiffusionGemmaGraph>(gcfg);
    if (!graph->init()) {
        unsetenv("DG_NO_L0_CACHE");
        throw std::runtime_error("model init failed");
    }
    unsetenv("DG_NO_L0_CACHE");

    DiffusionConfig cfg;
    cfg.remasking            = DiffusionRemask::LowConfidence;
    cfg.noise_scheme         = DiffusionNoise::UniformState;
    cfg.block_size           = block_size;
    cfg.n_steps              = steps;
    cfg.seed                 = seed;
    cfg.confidence_threshold = 0.9f;
    cfg.enable_l2_interblock = l2_enabled;

    DiffusionBackend backend(std::move(graph), cfg, "diffusiongemma");
    DaemonIO io;
    io.stream_fd = -1;

    GenerateRequest req;
    req.prompt    = prompt;
    req.n_gen     = n_gen;
    req.do_sample = false;

    using clock = std::chrono::steady_clock;
    const auto t0  = clock::now();
    GenerateResult r = backend.generate(req, io);
    const double wall_s = std::chrono::duration<double>(clock::now() - t0).count();

    if (!r.ok)
        throw std::runtime_error("generation failed: " + r.error);

    RunResult res;
    res.tokens = std::move(r.tokens);
    res.wall_s = wall_s;
    return res;
}

int main(int argc, char ** argv) {
    if (argc < 3) {
        std::fprintf(stderr,
            "usage: %s <model.gguf> <prompt.bin>"
            " [n_gen=512] [block_size=256] [steps=8] [seed=42]\n",
            argv[0]);
        return 2;
    }

    const char *   model_path  = argv[1];
    const char *   prompt_path = argv[2];
    const int      n_gen       = argc > 3 ? std::atoi(argv[3]) : 512;
    const int      block_size  = argc > 4 ? std::atoi(argv[4]) : 256;
    const int      steps       = argc > 5 ? std::atoi(argv[5]) : 8;
    const uint64_t seed        = argc > 6
        ? (uint64_t)std::strtoull(argv[6], nullptr, 10) : 42ULL;

    if (n_gen < 2 * block_size) {
        std::fprintf(stderr,
            "[l2-smoke] n_gen=%d must be >= 2*block_size=%d to exercise 2+ blocks\n",
            n_gen, 2 * block_size);
        return 2;
    }

    std::vector<int32_t> prompt;
    try {
        prompt = read_i32_bin(prompt_path);
    } catch (const std::exception & e) {
        std::fprintf(stderr, "[l2-smoke] cannot load prompt: %s\n", e.what());
        return 1;
    }
    if (prompt.empty()) {
        std::fprintf(stderr, "[l2-smoke] prompt is empty\n");
        return 1;
    }

    std::printf("[l2-smoke] prompt_len=%d n_gen=%d block_size=%d n_blocks=%d steps=%d seed=%llu\n",
                (int)prompt.size(), n_gen, block_size, (n_gen + block_size - 1) / block_size,
                steps, (unsigned long long)seed);

    // ── Gate 1: single-block invariance ──────────────────────────────────────
    // Generate exactly one block with L2-off and L2-on.  Both use L0 prefix KV.
    // L2 save is called after the block but there is no next block, so no restore
    // occurs.  Output must be identical.
    std::printf("\n[l2-smoke] Gate 1: single-block invariance (block_size=%d tokens)\n",
                block_size);
    RunResult sb_off, sb_on;
    try {
        sb_off = run_generate(model_path, prompt, block_size, block_size, steps, seed,
                              /*l2=*/false);
        std::printf("[l2-smoke] 1-blk L2-off: n_tok=%d  wall=%.3fs\n",
                    (int)sb_off.tokens.size(), sb_off.wall_s);
        sb_on  = run_generate(model_path, prompt, block_size, block_size, steps, seed,
                              /*l2=*/true);
        std::printf("[l2-smoke] 1-blk L2-on:  n_tok=%d  wall=%.3fs\n",
                    (int)sb_on.tokens.size(), sb_on.wall_s);
    } catch (const std::exception & e) {
        std::fprintf(stderr, "[l2-smoke] Gate 1 FAIL (run error): %s\n", e.what());
        return 1;
    }
    const bool g1_invariant = (sb_off.tokens == sb_on.tokens);
    std::printf("[l2-smoke] Gate 1 output_invariant=%s\n",
                g1_invariant ? "true" : "false");
    if (!g1_invariant) {
        print_tokens("[l2-smoke] 1-blk L2-off", sb_off.tokens);
        print_tokens("[l2-smoke] 1-blk L2-on ", sb_on.tokens);
        return 1;
    }

    // ── Gate 2: multi-block L2-on generates without crash ────────────────────
    // Also compare against the full-batch reference (DG_NO_L0_CACHE=1).
    // Output equivalence is advisory: F16-KV (L2-on) vs F32-batch (reference)
    // may have minor numerical differences but should match for typical cases.
    std::printf("\n[l2-smoke] Gate 2: multi-block L2-on (%d blocks, n_gen=%d)\n",
                n_gen / block_size, n_gen);
    RunResult mb_l2on, mb_ref;
    try {
        mb_l2on = run_generate(model_path, prompt, n_gen, block_size, steps, seed,
                               /*l2=*/true);
        std::printf("[l2-smoke] multi-blk L2-on:  n_tok=%d  wall=%.3fs\n",
                    (int)mb_l2on.tokens.size(), mb_l2on.wall_s);

        // Full-batch reference (correct slow path; no KV cache bias).
        mb_ref = run_generate(model_path, prompt, n_gen, block_size, steps, seed,
                              /*l2=*/false, /*no_l0_cache=*/true);
        std::printf("[l2-smoke] multi-blk full-batch: n_tok=%d  wall=%.3fs\n",
                    (int)mb_ref.tokens.size(), mb_ref.wall_s);
    } catch (const std::exception & e) {
        std::fprintf(stderr, "[l2-smoke] Gate 2 FAIL (run error): %s\n", e.what());
        return 1;
    }

    const bool g2_match = (mb_l2on.tokens == mb_ref.tokens);
    std::printf("[l2-smoke] Gate 2 output_invariant=%s  (advisory: FP16-cache vs FP32-batch)\n",
                g2_match ? "true" : "false");
    if (!g2_match) {
        print_tokens("[l2-smoke] multi-blk ref  ", mb_ref.tokens);
        print_tokens("[l2-smoke] multi-blk L2-on", mb_l2on.tokens);
        // Non-fatal: F16-KV precision vs F32-batch may cause divergence.
    }

    if (mb_ref.wall_s > 0.0) {
        std::printf("[l2-smoke] L2-on speedup vs full-batch = %.2f×\n",
                    mb_ref.wall_s / mb_l2on.wall_s);
    }

    std::printf("\n[l2-smoke] PASS  Gate1=invariant Gate2=no-crash\n");
    return 0;
}
