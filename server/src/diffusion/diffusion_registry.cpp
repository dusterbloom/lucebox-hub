// Diffusion family sub-factory implementation. See diffusion_registry.h.

#include "diffusion_registry.h"
#include "diffusion_backend.h"
#include "diffusiongemma/diffusion_gemma.h"

#include <cstdio>
#include <utility>

namespace dflash::common {

std::unique_ptr<DiffusionModelGraph> create_diffusion_model(
        const std::string & family, DiffusionModelArgs & args) {

    if (family == "diffusiongemma" || family == "diffusion-gemma") {
        // DiffusionGemma reuses the gemma4 backbone (loader + weights + cache)
        // and runs the bidirectional denoising forward (gemma4_denoise_batch).
        DiffusionGemmaConfig gcfg;
        gcfg.model_path = args.model_path;
        gcfg.gpu        = args.device.gpu;
        gcfg.max_ctx    = args.max_ctx > 0 ? args.max_ctx : 4096;
        auto g = std::make_unique<DiffusionGemmaGraph>(gcfg);
        if (!g->init()) {
            std::fprintf(stderr, "[diffusion] diffusiongemma init failed\n");
            return nullptr;
        }
        // Set DiffusionGemma family defaults. DiffusionGemma uses uniform-state
        // noise (no dedicated [MASK] token) and EntropyBound remasking (EB decode).
        // The default DiffusionConfig leaves noise_scheme=Masked which triggers
        // "Masked scheme requires a mask token id" at runtime because there is no
        // mask token in the vocab.
        args.cfg.noise_scheme      = DiffusionNoise::UniformState;
        args.cfg.remasking         = DiffusionRemask::EntropyBound;
        // Measured sweet spot: eb_schedule_steps=12 (schedule hits t_min at step 11;
        // entropy-bound early-stop fires at ~9-12 steps on typical prompts),
        // eb_max_steps=16 (model card diffusion.steps=16; default 48 wastes 4×).
        args.cfg.eb_max_steps      = 16;
        args.cfg.eb_schedule_steps = 12;
        return g;
    }
    if (family == "nemotron-diffusion" || family == "nemotron_diffusion") {
        // Phase 3: dense tri-mode backbone loader/graph. Not yet wired.
        std::fprintf(stderr,
            "[diffusion] family 'nemotron-diffusion' loader not yet wired (phase 3)\n");
        return nullptr;
    }

    std::fprintf(stderr, "[diffusion] unknown family '%s'\n", family.c_str());
    return nullptr;
}

std::unique_ptr<ModelBackend> create_diffusion_backend(
        const std::string & family, DiffusionModelArgs & args) {
    auto model = create_diffusion_model(family, args);
    if (!model) return nullptr;
    return std::make_unique<DiffusionBackend>(std::move(model), args.cfg, family);
}

}  // namespace dflash::common
