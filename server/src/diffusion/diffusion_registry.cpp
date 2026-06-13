// Diffusion family sub-factory implementation. See diffusion_registry.h.

#include "diffusion_registry.h"
#include "diffusion_backend.h"

#include <cstdio>
#include <utility>

namespace dflash::common {

std::unique_ptr<DiffusionModelGraph> create_diffusion_model(
        const std::string & family, const DiffusionModelArgs & args) {
    (void)args;

    if (family == "diffusiongemma") {
        // Phase 2: wrap Gemma4Weights (load_gemma4_gguf) with a bidirectional
        // denoising forward over the gemma4 graph. Not yet wired.
        std::fprintf(stderr,
            "[diffusion] family 'diffusiongemma' loader not yet wired (phase 2)\n");
        return nullptr;
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
        const std::string & family, const DiffusionModelArgs & args) {
    auto model = create_diffusion_model(family, args);
    if (!model) return nullptr;
    return std::make_unique<DiffusionBackend>(std::move(model), args.cfg, family);
}

}  // namespace dflash::common
