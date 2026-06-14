// Smoke test: DiffusionGemma L1 prefix-KV snapshot round-trip.
//
// Usage:
//   smoke_diffusion_l1_snapshot <model.gguf> <prompt.bin> [n_gen=8] [seed=42] [max_steps=2] [snap_len]

#include "diffusion_backend.h"
#include "diffusion_decoder.h"
#include "diffusion_types.h"
#include "diffusiongemma/diffusion_gemma.h"

#include <algorithm>
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
    for (int32_t t : toks) std::printf(" %d", t);
    std::printf("\n");
}

int main(int argc, char ** argv) {
    if (argc < 3) {
        std::fprintf(stderr,
            "usage: %s <model.gguf> <prompt.bin> [n_gen=8] [seed=42] [max_steps=2] [snap_len]\n",
            argv[0]);
        return 2;
    }

    const char * model_path = argv[1];
    const char * prompt_path = argv[2];
    const int n_gen = argc > 3 ? std::atoi(argv[3]) : 8;
    const uint64_t seed = argc > 4 ? (uint64_t)std::strtoull(argv[4], nullptr, 10) : 42ULL;
    const int max_steps = argc > 5 ? std::atoi(argv[5]) : 2;

    std::vector<int32_t> prompt;
    try {
        prompt = read_i32_bin(prompt_path);
    } catch (const std::exception & e) {
        std::fprintf(stderr, "[l1-smoke] cannot load prompt: %s\n", e.what());
        return 1;
    }
    if (prompt.empty()) {
        std::fprintf(stderr, "[l1-smoke] prompt is empty\n");
        return 1;
    }

    int snap_len = argc > 6 ? std::atoi(argv[6]) : std::max(1, (int)prompt.size() / 2);
    snap_len = std::max(1, std::min(snap_len, (int)prompt.size()));
    const int delta_len = (int)prompt.size() - snap_len;

    const int min_ctx = (int)prompt.size() + std::max(0, n_gen);
    const int max_ctx = ((std::max(min_ctx, 512) + 63) / 64) * 64;

    DiffusionGemmaConfig gcfg;
    gcfg.model_path = model_path;
    gcfg.gpu = 0;
    gcfg.max_ctx = max_ctx;

    auto graph = std::make_unique<DiffusionGemmaGraph>(gcfg);
    if (!graph->init()) {
        std::fprintf(stderr, "[l1-smoke] model init failed\n");
        return 1;
    }

    DiffusionConfig cfg;
    cfg.remasking = DiffusionRemask::EntropyBound;
    cfg.noise_scheme = DiffusionNoise::UniformState;
    cfg.block_size = n_gen;
    cfg.seed = seed;
    cfg.eb_max_steps = max_steps;
    cfg.eb_t_min = 0.4f;
    cfg.eb_t_max = 0.8f;
    cfg.eb_entropy_bound = 0.1f;
    cfg.eb_stability_threshold = 1;
    cfg.eb_confidence_threshold = 0.005f;

    DiffusionBackend backend(std::move(graph), cfg, "diffusiongemma");
    DaemonIO io;
    io.stream_fd = -1;

    GenerateRequest snap_req;
    snap_req.prompt.assign(prompt.begin(), prompt.begin() + snap_len);
    snap_req.n_gen = 0;
    snap_req.do_sample = false;
    snap_req.snap_slot = 0;
    snap_req.snap_pos = snap_len;

    GenerateResult snap_res = backend.generate(snap_req, io);
    if (!snap_res.ok || !backend.snapshot_used(0) || backend.snapshot_cur_pos(0) != snap_len) {
        std::fprintf(stderr,
            "[l1-smoke] snapshot save failed ok=%s used=%s cur_pos=%d expected=%d err=%s\n",
            snap_res.ok ? "true" : "false",
            backend.snapshot_used(0) ? "true" : "false",
            backend.snapshot_cur_pos(0), snap_len, snap_res.error.c_str());
        return 1;
    }

    GenerateRequest cold_req;
    cold_req.prompt = prompt;
    cold_req.n_gen = n_gen;
    cold_req.do_sample = false;
    cold_req.snap_slot = 1;
    cold_req.snap_pos = snap_len;

    GenerateResult cold = backend.generate(cold_req, io);
    if (!cold.ok) {
        std::fprintf(stderr, "[l1-smoke] cold generation failed: %s\n", cold.error.c_str());
        return 1;
    }

    GenerateResult restored = backend.restore_and_generate(0, cold_req, io);
    if (!restored.ok) {
        std::fprintf(stderr, "[l1-smoke] restore generation failed: %s\n",
                     restored.error.c_str());
        return 1;
    }

    const bool same = (cold.tokens == restored.tokens);
    std::printf("[l1-smoke] snap_len=%d prompt_len=%d delta_len=%d n_gen=%d seed=%llu steps=%d\n",
                snap_len, (int)prompt.size(), delta_len, n_gen,
                (unsigned long long)seed, max_steps);
    std::printf("[l1-smoke] snapshot_cur_pos=%d\n", backend.snapshot_cur_pos(0));
    std::printf("[l1-smoke] output_invariant=%s\n", same ? "true" : "false");
    print_tokens("[l1-smoke] cold", cold.tokens);
    print_tokens("[l1-smoke] restored", restored.tokens);

    if (!same) return 1;
    if (backend.snapshot_cur_pos(0) != snap_len) return 1;
    return 0;
}
