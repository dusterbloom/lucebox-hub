// Paired-state extraction: dump a native Qwen3.5/3.6/3.8 target snapshot to the
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
//   qwen35_snapshot_dump MODEL.gguf JOBS.list [GPU]
//
// A JOBS.list file has tab-separated `prompt_path<TAB>cut<TAB>out.lbsnap` lines
// and keeps the model resident across entries.
//
// Env: QWEN35_DUMP_MAXCTX (default cut+256), QWEN35_DUMP_CACHE_TYPE (f16|q8_0,
// default q8_0 — the measured R9700 serving profile).

#include "qwen35/qwen35_backend.h"

#include "lbsnap_writer.h"

#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

using namespace dflash::common;

namespace {

ggml_type cache_type_from_env() {
    const char * t = std::getenv("QWEN35_DUMP_CACHE_TYPE");
    if (t && t[0] && std::string(t) == "f16") return GGML_TYPE_F16;
    return GGML_TYPE_Q8_0;
}

bool dump_one(Qwen35Backend & backend, const std::string & model,
              const lbsnap::Entry & entry, int gpu) {
    std::vector<int32_t> prompt = lbsnap::read_prompt(entry.prompt.c_str(), model);
    if (prompt.empty()) {
        std::fprintf(stderr, "FAIL: no tokens from %s\n", entry.prompt.c_str());
        return false;
    }
    if (entry.cut > 0 && entry.cut < (int) prompt.size()) prompt.resize(entry.cut);
    const int cut = (int) prompt.size();

    GenerateRequest req;
    req.prompt = prompt;
    req.n_gen = 0;
    DaemonIO io;
    if (!backend.generate(req, io).ok()) {
        std::fprintf(stderr, "FAIL: prefill to cut=%d failed (%s)\n", cut,
                     entry.prompt.c_str());
        return false;
    }
    if (!backend.snapshot_save(0) || backend.snapshot_cur_pos(0) != cut) {
        std::fprintf(stderr, "FAIL: snapshot at cut=%d failed (%s)\n", cut,
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
    (void) gpu;
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

    Qwen35Config config;
    config.target_path = model;
#if defined(DFLASH27B_BACKEND_HIP)
    config.device.backend = PlacementBackend::Hip;
#else
    config.device.backend = PlacementBackend::Cuda;
#endif
    config.device.gpu = gpu;
    const char * mc = std::getenv("QWEN35_DUMP_MAXCTX");
    config.device.max_ctx = mc && mc[0] ? std::atoi(mc) : 8192;
    config.paged_attention = false;
    config.max_concurrency = 1;
    config.cache_type_k = cache_type_from_env();
    config.cache_type_v = cache_type_from_env();

    Qwen35Backend backend(config);
    if (!backend.init()) {
        std::fprintf(stderr, "FAIL: backend init failed\n");
        return 1;
    }

    int failures = 0;
    for (const auto & e : entries) {
        if (!dump_one(backend, model, e, gpu)) ++failures;
    }
    std::fprintf(stderr, "[dump] %zu entries, %d failures\n", entries.size(),
                 failures);
    return failures ? 1 : 0;
}
