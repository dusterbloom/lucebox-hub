#include "qwen4exp_backend.h"

#include "ggml-cuda.h"

#include <cstdio>
#include <utility>

namespace dflash::common {

Qwen4ExpBackend::Qwen4ExpBackend(Qwen4ExpBackendConfig cfg)
    : cfg_(std::move(cfg)) {}

Qwen4ExpBackend::~Qwen4ExpBackend() {
    shutdown();
}

bool Qwen4ExpBackend::init() {
    if (cfg_.device.is_layer_split()) {
        std::fprintf(stderr, "[qwen4exp] layer split is not supported yet\n");
        return false;
    }
    backend_ = ggml_backend_cuda_init(cfg_.device.gpu);
    if (!backend_) {
        std::fprintf(stderr, "[qwen4exp] backend init failed for GPU %d\n",
                     cfg_.device.gpu);
        return false;
    }
    if (!load_qwen4exp_gguf(cfg_.model_path, backend_, weights_)) {
        std::fprintf(stderr, "[qwen4exp] model load failed: %s\n",
                     dflash27b_last_error());
        return false;
    }
    if (!create_qwen4exp_cache(backend_, weights_, cfg_.device.max_ctx,
                               GGML_TYPE_F16, cache_)) {
        std::fprintf(stderr, "[qwen4exp] cache creation failed\n");
        return false;
    }
    return true;
}

void Qwen4ExpBackend::print_ready_banner() const {
    const Qwen4ExpWeights & w = weights_;
    std::printf(
        "[qwen4exp-daemon] ready layers=%d linear=%d full=%d experts=%d/%d "
        "hc=%d/%d ple=%zu ngram=%d ctx=%d\n",
        w.n_layer,
        w.n_layer - w.n_layer / w.full_attention_interval,
        w.n_layer / w.full_attention_interval,
        w.n_expert_used, w.n_expert, w.n_hc, w.hc_lowrank,
        w.ple_layer_ids.size(), w.ple_ngram_size, cfg_.device.max_ctx);
    std::fflush(stdout);
}

bool Qwen4ExpBackend::park(ParkTarget target) {
    (void) target;
    // Weight residency control lands with the graph; nothing to release yet.
    parked_ = true;
    return true;
}

bool Qwen4ExpBackend::unpark(ParkTarget target) {
    (void) target;
    parked_ = false;
    return true;
}

GenerateResult Qwen4ExpBackend::generate_impl(const GenerateRequest & req,
                                              const DaemonIO & io) {
    (void) req;
    (void) io;
    GenerateResult result;
    if (parked_) {
        result.fail(GenerateErrorCode::ModelParked,
                    "qwen4exp target is parked");
        return result;
    }
    result.fail(GenerateErrorCode::BackendSpecific,
                "qwen4exp forward graph is not implemented yet (Phase 1)");
    return result;
}

bool Qwen4ExpBackend::snapshot_save(int slot) {
    (void) slot;
    return false;
}

void Qwen4ExpBackend::snapshot_free(int slot) {
    (void) slot;
}

bool Qwen4ExpBackend::snapshot_used(int slot) const {
    (void) slot;
    return false;
}

int Qwen4ExpBackend::snapshot_cur_pos(int slot) const {
    (void) slot;
    return 0;
}

GenerateResult Qwen4ExpBackend::restore_and_generate_impl(
        int slot, const GenerateRequest & req, const DaemonIO & io) {
    (void) slot;
    (void) req;
    (void) io;
    GenerateResult result;
    result.fail(GenerateErrorCode::InvalidSnapshotSlot,
                "qwen4exp snapshots are not implemented yet");
    return result;
}

bool Qwen4ExpBackend::handle_compress(const std::string & line,
                                      const DaemonIO & io) {
    (void) line;
    (void) io;
    return false;
}

void Qwen4ExpBackend::free_drafter() {}

void Qwen4ExpBackend::shutdown() {
    free_qwen4exp_cache(cache_);
    free_qwen4exp_weights(weights_);
    if (backend_) {
        ggml_backend_free(backend_);
        backend_ = nullptr;
    }
}

}  // namespace dflash::common
