// Paired-state extraction: dump a native DS4 snapshot to the LBSNAP01 container
// for the Qwen->DS4 bridge spike (luce_box lucebox_rnd/KV_translator_qwen_ds4).
//
// Prefills a prompt (token IDs, or text tokenized with the model's own vocab)
// and saves a checkpoint at `cut` via the mid-prefill snapshot hook that the
// E0 parity harness validated.
//
// Model-gated, not registered in ordinary CI.
//
//   ds4_snapshot_dump MODEL.gguf PROMPT.txt|TOKENS.bin CUT OUT.lbsnap [GPU]

#include "deepseek4/deepseek4_backend.h"

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

    const std::vector<int32_t> prompt = lbsnap::read_prompt(argv[2], model);
    if (prompt.empty()) {
        std::fprintf(stderr, "FAIL: no tokens from %s\n", argv[2]);
        return 1;
    }

    DeepSeek4BackendConfig config;
    config.model_path = model;
#if defined(DFLASH27B_BACKEND_HIP)
    config.device.backend = PlacementBackend::Hip;
#else
    config.device.backend = PlacementBackend::Cuda;
#endif
    config.device.gpu = gpu;
    config.paged_attention = false;
    config.max_concurrency = 1;
    config.prefill_mode = PrefillAttentionMode::Exact;
    config.chunk = 512;
    config.max_ctx = (int) prompt.size() + 256;

    DeepSeek4Backend backend(config);
    if (!backend.init()) {
        std::fprintf(stderr, "FAIL: backend init failed\n");
        return 1;
    }

    int cut = cut_arg;
    if (cut <= 0 || cut > (int) prompt.size()) cut = (int) prompt.size();

    GenerateRequest req;
    req.prompt = prompt;
    req.n_gen = 0;
    req.snap_slot = 0;
    req.snap_pos = cut;
    DaemonIO io;
    if (!backend.generate(req, io).ok() || !backend.snapshot_used(0) ||
        backend.snapshot_cur_pos(0) != cut) {
        std::fprintf(stderr, "FAIL: checkpoint at cut=%d failed\n", cut);
        return 1;
    }

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
