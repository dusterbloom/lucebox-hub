// DiffusionBackend — a first-class ModelBackend for diffusion (dLLM) families.
//
// Wraps a DiffusionModelGraph + DiffusionConfig and routes generation through
// the model-agnostic denoising loop (diffusion_decoder.cpp). Generation flows
// through the same ModelBackend::generate() path the autoregressive arches use,
// so the daemon loop, native HTTP server and OpenAI/Anthropic SSE layer need no
// diffusion-specific plumbing.
//
// The AR-only surface of ModelBackend (KV snapshots, pflash compress, DFlash
// speculative decode, park/unpark of a draft model) does not apply to the
// diffusion path yet and is stubbed: snapshots report "unused", compress is
// rejected, and supports_dflash_spec_decode() keeps its false default. (Nemotron
// self-speculation can later light up the DFlash hooks — see the module plan.)

#pragma once

#include <memory>
#include <string>

#include "common/model_backend.h"
#include "diffusion_types.h"
#include "diffusion_model.h"

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

    // Snapshots are not supported by the diffusion path (no causal KV reuse).
    bool snapshot_save(int slot) override;
    void snapshot_free(int slot) override;
    bool snapshot_used(int slot) const override;
    int  snapshot_cur_pos(int slot) const override;
    GenerateResult restore_and_generate_impl(int slot,
                                             const GenerateRequest & req,
                                             const DaemonIO & io) override;

    bool handle_compress(const std::string & line, const DaemonIO & io) override;
    void free_drafter() override {}

    void shutdown() override;

private:
    std::unique_ptr<DiffusionModelGraph> model_;
    DiffusionConfig                      cfg_;
    std::string                          arch_label_;
};

}  // namespace dflash::common
