// Qwen3MoeBackend implementation — Phase A scaffolding.
//
// At this stage: init() returns false (loader stub returns false), every
// other method returns a clean failure / no-op. This is enough to wire into
// `backend_factory.cpp` and link the executable. The real loader / step
// graph / decode loop land in the next session — see the TODO comments in
// qwen3moe_loader.cpp and qwen3moe_graph.cpp.

#include "qwen3moe_backend.h"

#include "ggml-cuda.h"

#include <cstdio>
#include <utility>

namespace dflash::common {

Qwen3MoeBackend::Qwen3MoeBackend(const Qwen3MoeBackendConfig & cfg) : cfg_(cfg) {}

Qwen3MoeBackend::~Qwen3MoeBackend() { shutdown(); }

bool Qwen3MoeBackend::init() {
    backend_ = ggml_backend_cuda_init(cfg_.device.gpu);
    if (!backend_) {
        std::fprintf(stderr, "[qwen3moe] CUDA init failed for GPU %d\n",
                     cfg_.device.gpu);
        return false;
    }

    if (!load_qwen3moe_gguf(cfg_.model_path ? cfg_.model_path : "",
                            backend_, w_)) {
        std::fprintf(stderr, "[qwen3moe] model load failed (stub)\n");
        return false;
    }

    if (!create_qwen3moe_cache(backend_, w_, cfg_.device.max_ctx, cache_)) {
        std::fprintf(stderr, "[qwen3moe] cache creation failed (stub)\n");
        return false;
    }

    return true;
}

void Qwen3MoeBackend::print_ready_banner() const {
    std::printf("[qwen3moe-daemon] ready (stub — Phase A scaffolding only)\n");
    std::fflush(stdout);
}

bool Qwen3MoeBackend::park(const std::string & /*what*/)   { return false; }
bool Qwen3MoeBackend::unpark(const std::string & /*what*/) { return false; }

GenerateResult Qwen3MoeBackend::generate(const GenerateRequest & /*req*/,
                                          const DaemonIO &        /*io*/) {
    GenerateResult r;
    r.ok    = false;
    r.error = "qwen3moe backend: generate() not implemented (Phase A stub)";
    return r;
}

bool Qwen3MoeBackend::snapshot_save(int /*slot*/) { return false; }
void Qwen3MoeBackend::snapshot_free(int slot)     {
    if (slot >= 0 && slot < PREFIX_SLOTS) {
        free_qwen3moe_snapshot(snapshots_[slot]);
    }
}
bool Qwen3MoeBackend::snapshot_used(int slot) const {
    return slot >= 0 && slot < PREFIX_SLOTS
           && snapshots_[slot].ctx != nullptr;
}
int  Qwen3MoeBackend::snapshot_cur_pos(int slot) const {
    if (slot < 0 || slot >= PREFIX_SLOTS) return 0;
    return snapshots_[slot].cur_pos;
}

GenerateResult Qwen3MoeBackend::restore_and_generate(int /*slot*/,
                                                      const GenerateRequest & /*req*/,
                                                      const DaemonIO &        /*io*/) {
    GenerateResult r;
    r.ok    = false;
    r.error = "qwen3moe backend: restore_and_generate() not implemented (Phase A stub)";
    return r;
}

bool Qwen3MoeBackend::handle_compress(const std::string & /*line*/,
                                       const DaemonIO &    /*io*/) {
    // pflash compress is Phase C — out of scope for Phase A.
    return false;
}
void Qwen3MoeBackend::free_drafter() {}

bool Qwen3MoeBackend::try_handle_command(const std::string & /*line*/,
                                          const DaemonIO &    /*io*/) {
    return false;
}

void Qwen3MoeBackend::shutdown() {
    for (int s = 0; s < PREFIX_SLOTS; ++s) {
        free_qwen3moe_snapshot(snapshots_[s]);
    }
    free_qwen3moe_cache(cache_);
    free_qwen3moe_weights(w_);
    if (backend_) {
        ggml_backend_free(backend_);
        backend_ = nullptr;
    }
}

// ── Forward-pass primitives: stubs (see qwen3moe_graph.cpp) ───────────────

bool Qwen3MoeBackend::do_step(const float * /*embed*/,
                               int           /*n_tokens*/,
                               int           /*kv_start*/,
                               std::vector<float> & /*out_logits*/) {
    std::fprintf(stderr, "[qwen3moe] do_step not implemented (Phase A stub)\n");
    return false;
}

int Qwen3MoeBackend::do_prefill(const std::vector<int32_t> & /*tokens*/,
                                 const DaemonIO &             /*io*/,
                                 int                          /*kv_offset*/) {
    return -1;
}

bool Qwen3MoeBackend::do_decode(int                             /*committed*/,
                                 int                             /*n_gen*/,
                                 std::vector<int32_t> &          /*out_tokens*/,
                                 const DaemonIO &                /*io*/) {
    return false;
}

}  // namespace dflash::common
