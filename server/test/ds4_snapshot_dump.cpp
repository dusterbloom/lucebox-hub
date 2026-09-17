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

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

using namespace dflash::common;

namespace {

bool dump_one(DeepSeek4Backend & backend, const std::vector<int32_t> & tokens,
              int cut_arg, const std::string & out_path) {
    if (tokens.empty()) return false;
    int cut = cut_arg;
    if (cut <= 0 || cut > (int) tokens.size()) cut = (int) tokens.size();

    GenerateRequest req;
    req.prompt = tokens;
    req.n_gen = 0;
    req.snap_slot = 0;
    req.snap_pos = cut;
    DaemonIO io;
    if (!backend.generate(req, io).ok() || !backend.snapshot_used(0) ||
        backend.snapshot_cur_pos(0) != cut) {
        std::fprintf(stderr, "FAIL: checkpoint at cut=%d failed (%s)\n", cut,
                     out_path.c_str());
        return false;
    }
    const ModelBackend::SnapshotRef ref = backend.snapshot_ref(0);
    if (!ref.ctx || !ref.buf) {
        std::fprintf(stderr, "FAIL: snapshot_ref(0) empty\n");
        return false;
    }
    const bool ok = lbsnap::write_container(out_path, ref.ctx, ref.cur_pos, cut) >= 0;
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

    // Tokenize everything up front so the context can be sized from the real
    // maximum. A byte-based estimate under-counts token-dense prompts, which
    // let prefill run past the cache capacity and trip a NULL-buffer assert.
    lbsnap::PromptLoader loader;
    std::vector<std::vector<int32_t>> prompts(entries.size());
    int max_prompt = 0;
    for (size_t i = 0; i < entries.size(); ++i) {
        prompts[i] = loader.load(entries[i].prompt.c_str(), model);
        if (prompts[i].empty()) {
            std::fprintf(stderr, "FAIL: no tokens from %s\n", entries[i].prompt.c_str());
            return 1;
        }
        max_prompt = std::max(max_prompt, (int) prompts[i].size());
    }
    std::fprintf(stderr, "[dump] %zu entries, longest prompt %d tokens\n",
                 entries.size(), max_prompt);

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
    config.max_ctx = max_prompt + 256;

    DeepSeek4Backend backend(config);
    if (!backend.init()) {
        std::fprintf(stderr, "FAIL: backend init failed\n");
        return 1;
    }

    int failures = 0;
    for (size_t i = 0; i < entries.size(); ++i) {
        if (!dump_one(backend, prompts[i], entries[i].cut, entries[i].out)) {
            ++failures;
        }
    }
    std::fprintf(stderr, "[dump] %zu entries, %d failures\n", entries.size(),
                 failures);
    return failures ? 1 : 0;
}
