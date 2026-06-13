// String <-> enum helpers for diffusion config. See diffusion_types.h.
//
// Deliberately ggml-free and dependency-light so the model-card layer and the
// CPU unit tests can both use them. The canonical strings are the enum values
// in share/model_cards/_schema.json's `diffusion` block.

#include "diffusion_types.h"

namespace dflash::common {

const char * to_string(DiffusionRemask r) {
    switch (r) {
        case DiffusionRemask::LowConfidence:     return "low_confidence";
        case DiffusionRemask::Random:            return "random";
        case DiffusionRemask::ParallelThreshold: return "parallel_threshold";
    }
    return "low_confidence";
}

const char * to_string(DiffusionNoise n) {
    switch (n) {
        case DiffusionNoise::Masked:       return "masked";
        case DiffusionNoise::UniformState: return "uniform_state";
    }
    return "masked";
}

bool remask_from_string(const std::string & s, DiffusionRemask & out) {
    if (s == "low_confidence")     { out = DiffusionRemask::LowConfidence;     return true; }
    if (s == "random")             { out = DiffusionRemask::Random;            return true; }
    if (s == "parallel_threshold") { out = DiffusionRemask::ParallelThreshold; return true; }
    return false;
}

bool noise_from_string(const std::string & s, DiffusionNoise & out) {
    if (s == "masked")        { out = DiffusionNoise::Masked;       return true; }
    if (s == "uniform_state") { out = DiffusionNoise::UniformState; return true; }
    return false;
}

}  // namespace dflash::common
