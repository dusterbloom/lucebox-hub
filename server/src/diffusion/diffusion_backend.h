// DiffusionBackend — a first-class ModelBackend for diffusion (dLLM) families.
//
// Wraps a DiffusionModelGraph + DiffusionConfig and routes generation through
// the model-agnostic denoising loop (diffusion_decoder.cpp). Generation flows
// through the same ModelBackend::generate() path the autoregressive arches use,
// so the daemon loop, native HTTP server and OpenAI/Anthropic SSE layer need no
// diffusion-specific plumbing.
//
// The AR-only surface of ModelBackend mostly does not apply to diffusion:
// pflash compress and DFlash speculative decode are rejected/disabled. Prefix
// KV snapshots are supported for DiffusionGemma prompt reuse.

#pragma once

#include <memory>
#include <string>

#include "common/model_backend.h"
#include "diffusion_types.h"
#include "diffusion_model.h"
#include "gemma4_internal.h"
#include "ggml-backend.h"

namespace dflash::common {

class DiffusionBackend : public ModelBackend {
public:
    DiffusionBackend(std::unique_ptr<DiffusionModelGraph> model,
                     DiffusionConfig cfg,
                     std::string arch_label);
    ~DiffusionBackend() override;

    DiffusionBackend(const DiffusionBackend &)             = delete;
    DiffusionBackend & operator=(const DiffusionBackend &) = delete;

    // ── ModelBackend interface ───────────────────────────────────────
    void print_ready_banner() const override;

    bool park(const std::string & what) override;
    bool unpark(const std::string & what) override;
    bool is_target_parked() const override { return false; }

    GenerateResult generate_impl(const GenerateRequest & req,
                                 const DaemonIO & io) override;

    bool snapshot_save(int slot) override;
    void snapshot_free(int slot) override;
    bool snapshot_used(int slot) const override;
    int  snapshot_cur_pos(int slot) const override;
    GenerateResult restore_and_generate_impl(int slot,
                                             const GenerateRequest & req,
                                             const DaemonIO & io) override;

    SnapshotRef snapshot_ref(int slot) const override;
    bool snapshot_adopt(int slot, ggml_context * ctx,
                        ggml_backend_buffer_t buf, int cur_pos,
                        int32_t last_tok = -1) override;

    bool handle_compress(const std::string & line, const DaemonIO & io) override;
    void free_drafter() override {}

    void shutdown() override;

private:
    std::unique_ptr<DiffusionModelGraph> model_;
    DiffusionConfig                      cfg_;
    std::string                          arch_label_;

    static constexpr int PREFIX_SLOTS = ModelBackend::kMaxSlots;
    ggml_backend_t                      snap_backend_ = nullptr;
    ggml_backend_t                      snap_compute_backend_ = nullptr;
    Gemma4Snapshot                      snapshots_[PREFIX_SLOTS];
};

}  // namespace dflash::common
