#pragma once

#include "common/moe_hybrid_stream.h"

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace dflash::common {

// Research-only routed boundary used by H16. The exact stream engine remains
// the immutable teacher; implementations may replace one model layer only.
class KimiK3RoutedOutputProvider {
public:
    virtual ~KimiK3RoutedOutputProvider() = default;

    virtual bool handles_layer(int model_layer) const = 0;
    virtual bool evaluate(
        int model_layer,
        int base_pos,
        const MoeStreamExpertSpec & exact_spec,
        const MoeStreamRouteBatch & native_routes,
        MoeHybridStreamEngine & exact_engine,
        std::vector<float> & output,
        std::string * err = nullptr) = 0;
};

// An unset or "exact" DFLASH_KIMI_LAYER1_PROVIDER returns success with a null
// provider. "slabs" and "whole" require the registered H16 runtime artifacts.
bool create_kimi_k3_progressive_provider_from_env(
    ggml_backend_t expert_backend,
    std::unique_ptr<KimiK3RoutedOutputProvider> & out,
    std::string * err = nullptr);

// Deterministic policy helpers kept public for GPU-free regression tests.
std::vector<int32_t> select_kimi_k3_slab_prefix_ids(
    const int32_t * expert_ids,
    const float * router_weights,
    int top_k,
    const float * ordered_importance,
    int expert_count,
    int slabs_per_expert,
    int budget);

std::vector<int32_t> select_kimi_k3_whole_expert_routes(
    const int32_t * expert_ids,
    const float * router_weights,
    int top_k,
    const float * expert_importance,
    int expert_count,
    int budget);

} // namespace dflash::common
