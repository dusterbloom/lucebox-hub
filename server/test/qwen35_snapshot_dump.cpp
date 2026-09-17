// Paired-state extraction: dump a native Qwen3.5/3.6 target snapshot to the
// LBSNAP01 container for the Qwen->DS4 bridge spike.
//
// The source state at a position is the state after prefilling exactly that
// many source tokens, so the prompt is truncated to `cut` rather than relying
// on a mid-prefill checkpoint hook. Non-paged, single-concurrency, no kvflash:
// those are the modes in which Qwen35Backend prefix snapshots are supported.
//
// Model-gated, not registered in ordinary CI.
//
//   qwen35_snapshot_dump MODEL.gguf PROMPT.txt|TOKENS.bin CUT OUT.lbsnap [GPU]

#include "qwen35/qwen35_backend.h"

#include "lbsnap_writer.h"

#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

using namespace dflash::common;

int main(int argc, char ** argv) {
    if (argc < 5) {
        std::fprintf(stderr,
                     "usage: %s MODEL.gguf PROMPT.txt|TOKENS.bin CUT OUT.lbsnap [GPU]\n",
                     argv[0]);
        return 2;
    }
    const std::string model = argv[1];
    const int cut_arg = std::atoi(argv[3]);
    const std::string out_path = argv[4];
    const int gpu = argc > 5 ? std::atoi(argv[5]) : 0;

    std::vector<int32_t> prompt = lbsnap::read_prompt(argv[2], model);
    if (prompt.empty()) {
        std::fprintf(stderr, "FAIL: no tokens from %s\n", argv[2]);
        return 1;
    }
    if (cut_arg > 0 && cut_arg < (int) prompt.size()) prompt.resize(cut_arg);
    const int cut = (int) prompt.size();

    Qwen35Config config;
    config.target_path = model;
#if defined(DFLASH27B_BACKEND_HIP)
    config.device.backend = PlacementBackend::Hip;
#else
    config.device.backend = PlacementBackend::Cuda;
#endif
    config.device.gpu = gpu;
    config.device.max_ctx = cut + 256;
    config.paged_attention = false;
    config.max_concurrency = 1;
    // Match the measured R9700 Qwen3.8 profile (q8_0 KV); model defaults are
    // not exercised by the supported launch path.
    config.cache_type_k = GGML_TYPE_Q8_0;
    config.cache_type_v = GGML_TYPE_Q8_0;

    Qwen35Backend backend(config);
    std::fprintf(stderr, "[dump] init...\n");
    if (!backend.init()) {
        std::fprintf(stderr, "FAIL: backend init failed\n");
        return 1;
    }
    std::fprintf(stderr, "[dump] init ok, prefill %d tokens...\n", cut);

    GenerateRequest req;
    req.prompt = prompt;
    req.n_gen = 0;
    DaemonIO io;
    if (!backend.generate(req, io).ok()) {
        std::fprintf(stderr, "FAIL: prefill to cut=%d failed\n", cut);
        return 1;
    }
    std::fprintf(stderr, "[dump] prefill ok, snapshot_save...\n");
    if (!backend.snapshot_save(0) || backend.snapshot_cur_pos(0) != cut) {
        std::fprintf(stderr, "FAIL: snapshot at cut=%d failed\n", cut);
        return 1;
    }
    std::fprintf(stderr, "[dump] snapshot ok\n");

    const ModelBackend::SnapshotRef ref = backend.snapshot_ref(0);
    if (!ref.ctx || !ref.buf) {
        std::fprintf(stderr, "FAIL: snapshot_ref(0) empty\n");
        return 1;
    }
    if (lbsnap::write_container(out_path, ref.ctx, ref.cur_pos, cut) < 0) {
        return 1;
    }
    return 0;
}
