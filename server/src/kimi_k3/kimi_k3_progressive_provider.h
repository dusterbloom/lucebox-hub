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

// H22 uses a strict, provenance-hashable text file rather than a long
// environment string.  The file must contain exactly one "layer budget" row
// for routed layers 1..92.  Budgets are drawn from the preregistered
// progressive ladder 48,72,...,192.  The returned vector is indexed by
// model_layer - 1.
bool parse_kimi_k3_layer_budget_table(
    const std::string & path,
    std::vector<int32_t> & budgets,
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

// Select whole routes first, then retain an equal calibrated prefix from each
// selected expert. Returned IDs use expert * slabs_per_expert + rank.
std::vector<int32_t> select_kimi_k3_route_slab_prefix_ids(
    const int32_t * expert_ids,
    const float * router_weights,
    int top_k,
    const float * expert_importance,
    int expert_count,
    int slabs_per_expert,
    int route_budget,
    int slabs_per_route);

struct KimiK3CalibratedSlabPlan {
    int requested_budget = 0;
    std::vector<int32_t> selected_slab_ids;
    std::vector<int32_t> exact_route_indices;
};

// Experts whose calibrated flag is zero never enter the slab selector.  They
// are returned as native route indices for exact evaluation, and the actual
// selected count may therefore be below the requested nominal budget.
KimiK3CalibratedSlabPlan plan_kimi_k3_calibrated_slabs(
    const int32_t * expert_ids,
    const float * router_weights,
    int top_k,
    const float * ordered_importance,
    const uint8_t * calibrated_experts,
    int expert_count,
    int slabs_per_expert,
    int requested_budget);

// As above, but first chooses whole calibrated routes using the native expert
// importance and then keeps an equal calibrated slab prefix from each.  Routes
// without sufficient calibration are returned for exact evaluation.
KimiK3CalibratedSlabPlan plan_kimi_k3_calibrated_route_prefixes(
    const int32_t * expert_ids,
    const float * router_weights,
    int top_k,
    const float * expert_importance,
    const uint8_t * calibrated_experts,
    int expert_count,
    int slabs_per_expert,
    int route_budget,
    int slabs_per_route);

} // namespace dflash::common
