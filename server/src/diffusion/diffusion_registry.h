// Diffusion family sub-factory.
//
// Mirrors the top-level backend_factory: given a family id detected from GGUF
// metadata, construct the per-family DiffusionModelGraph and wrap it in a
// DiffusionBackend. Adding a new dLLM family is a localized edit here plus the
// family's loader/graph under server/src/diffusion/<family>/.

#pragma once

#include <memory>
#include <string>

#include "common/model_backend.h"
#include "placement/placement_config.h"
#include "diffusion_types.h"
#include "diffusion_model.h"

namespace dflash::common {

struct DiffusionModelArgs {
    const char *    model_path = nullptr;     // target .gguf (or safetensors dir)
    DevicePlacement device;
    int             max_ctx    = 0;           // 0 => model/card default
    // Decode defaults resolved from the model card; the family loader may also
    // fill cfg.mask_token_id from GGUF metadata when the card leaves it unset.
    DiffusionConfig cfg;
};

// Construct the per-family forward graph (e.g. "diffusiongemma",
// "nemotron-diffusion"). Returns nullptr (diagnostic on stderr) for unknown or
// not-yet-wired families.
std::unique_ptr<DiffusionModelGraph> create_diffusion_model(
    const std::string & family, const DiffusionModelArgs & args);

// Build the family graph and wrap it in a DiffusionBackend. Returns nullptr on
// failure.
std::unique_ptr<ModelBackend> create_diffusion_backend(
    const std::string & family, const DiffusionModelArgs & args);

}  // namespace dflash::common
