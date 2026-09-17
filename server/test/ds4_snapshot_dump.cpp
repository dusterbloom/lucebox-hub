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
//   ds4_snapshot_dump MODEL.gguf JOBS.list [GPU]
//
// A JOBS.list file has tab-separated `prompt_path<TAB>cut<TAB>out.lbsnap` lines
// and keeps the model resident across entries.

#include "deepseek4/deepseek4_backend.h"

#include "lbsnap_writer.h"

#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

using namespace dflash::common;

namespace {

bool dump_one(DeepSeek4Backend & backend, const std::string & model,
              const lbsnap::Entry & entry) {
    const std::vector<int32_t> prompt =
        lbsnap::read_prompt(entry.prompt.c_str(), model);
    if (prompt.empty()) {
        std::fprintf(stderr, "FAIL: no tokens from %s\n", entry.prompt.c_str());
        return false;
    }
    int cut = entry.cut;
    if (cut <= 0 || cut > (int) prompt.size()) cut = (int) prompt.size();

    GenerateRequest req;
    req.prompt = prompt;
    req.n_gen = 0;
    req.snap_slot = 0;
    req.snap_pos = cut;
    DaemonIO io;
    if (!backend.generate(req, io).ok() || !backend.snapshot_used(0) ||
        backend.snapshot_cur_pos(0) != cut) {
        std::fprintf(stderr, "FAIL: checkpoint at cut=%d failed (%s)\n", cut,
                     entry.prompt.c_str());
        return false;
    }
    const ModelBackend::SnapshotRef ref = backend.snapshot_ref(0);
    if (!ref.ctx || !ref.buf) {
        std::fprintf(stderr, "FAIL: snapshot_ref(0) empty\n");
        return false;
    }
    const bool ok = lbsnap::write_container(entry.out, ref.ctx, ref.cur_pos, cut) >= 0;
    backend.snapshot_free(0);
    return ok;
}

}  // namespace

int main(int argc, char ** argv) {
    if (argc < 4) {
        std::fprintf(stderr,
                     "usage: %s MODEL.gguf PROMPT.txt|TOKENS.bin CUT OUT.lbsnap [GPU]\n"
                     "       %s MODEL.gguf JOBS.list [GPU]\n",
                     argv[0], argv[0]);
        return 2;
    }
    const std::string model = argv[1];
    const bool batch = lbsnap::is_list_path(argv[2]);

    std::vector<lbsnap::Entry> entries;
    if (batch) {
        entries = lbsnap::read_list(argv[2]);
        if (entries.empty()) {
            std::fprintf(stderr, "FAIL: no entries in %s\n", argv[2]);
            return 1;
        }
    } else {
        if (argc < 5) return 2;
        entries.push_back({argv[2], std::atoi(argv[3]), argv[4]});
    }
    const int gpu = argc > (batch ? 3 : 5) ? std::atoi(argv[batch ? 3 : 5]) : 0;

    // Size the context for the longest prompt across the batch.
    int max_prompt = 0;
    for (const auto & e : entries) {
        FILE * f = std::fopen(e.prompt.c_str(), "rb");
        if (!f) continue;
        const long bytes = std::fseek(f, 0, SEEK_END) == 0 ? std::ftell(f) : 0;
        std::fclose(f);
        const int approx = (int) (bytes / 3) + 64;  // conservative tokens
        if (approx > max_prompt) max_prompt = approx;
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
    config.max_ctx = max_prompt + 512;

    DeepSeek4Backend backend(config);
    if (!backend.init()) {
        std::fprintf(stderr, "FAIL: backend init failed\n");
        return 1;
    }

    int failures = 0;
    for (const auto & e : entries) {
        if (!dump_one(backend, model, e)) ++failures;
    }
    std::fprintf(stderr, "[dump] %zu entries, %d failures\n", entries.size(),
                 failures);
    return failures ? 1 : 0;
}
