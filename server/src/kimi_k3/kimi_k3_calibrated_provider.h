#pragma once

#include "kimi_k3_routed_provider.h"

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace dflash::common {

// Creates the exact, single-owner calibrated sidecar provider. An unset or
// "exact" DFLASH_KIMI_LAYER1_PROVIDER intentionally returns a null provider.
// The sole non-null production value is "all-layers-calibrated96".
bool create_kimi_k3_calibrated_provider_from_env(
    ggml_backend_t expert_backend,
    ggml_backend_t destination_backend,
    std::unique_ptr<KimiK3RoutedOutputProvider> & out,
    std::string * error = nullptr);

// Strict 92-row calibrated budget table. The historical environment variable
// remains DFLASH_KIMI_H22_LAYER_BUDGETS for artifact compatibility.
bool parse_kimi_k3_layer_budget_table(
    const std::string & path,
    std::vector<int32_t> & budgets,
    std::string * error = nullptr);

int kimi_k3_effective_slab_budget(int configured_budget,
                                  int request_min_budget);

// Research-only progressive-rescue control. The environment syntax is a
// comma-separated BASE_POS:BUDGET list. Overrides are monotone floors and
// cannot reduce the configured/request budget.
struct KimiK3PositionBudget {
    int32_t base_pos = 0;
    int32_t slab_budget = 0;
};

bool parse_kimi_k3_position_budgets(
    const char * raw,
    std::vector<KimiK3PositionBudget> & overrides,
    std::string * error = nullptr);

int kimi_k3_effective_position_slab_budget(
    int configured_budget,
    int request_min_budget,
    const std::vector<KimiK3PositionBudget> & overrides,
    int base_pos);

// Research-only router-prefix screen. Unset means native top-16; only the
// preregistered exact route counts are accepted.
bool parse_kimi_k3_route_limit(
    const char * raw,
    int & route_limit,
    std::string * error = nullptr);

struct KimiK3CalibratedSlabPlan {
    int requested_budget = 0;
    std::vector<int32_t> selected_slab_ids;
    std::vector<int32_t> exact_route_indices;
};

std::vector<int32_t> select_kimi_k3_slab_prefix_ids(
    const int32_t * expert_ids,
    const float * router_weights,
    int top_k,
    const float * ordered_importance,
    int expert_count,
    int slabs_per_expert,
    int budget);

// Experts without a qualified calibration stay on the exact native path.
KimiK3CalibratedSlabPlan plan_kimi_k3_calibrated_slabs(
    const int32_t * expert_ids,
    const float * router_weights,
    int top_k,
    const float * ordered_importance,
    const uint8_t * calibrated_experts,
    int expert_count,
    int slabs_per_expert,
    int requested_budget);

enum class KimiK3SparseDeliveryPolicy : uint8_t {
    BufferedSlabs,
    DirectSlabs,
    CompactPageable,
    CompactPinned,
    DirectPinnedCompact,
};

enum class KimiK3SparseUpload : uint8_t {
    SlabCopies,
    PageableCompact,
    PinnedCompact,
    PrepackedCompact,
};

KimiK3SparseUpload kimi_k3_sparse_upload_for_call(
    KimiK3SparseDeliveryPolicy delivery,
    bool has_prepacked_payload);

uint16_t kimi_k3_selected_natural_slab_mask(
    const uint16_t * natural_by_rank,
    const uint8_t * selected_by_rank,
    int slab_count);

void kimi_k3_suppress_resident_slab_ranks(
    const uint16_t * natural_by_rank,
    uint16_t missing_mask,
    uint8_t * selected_by_rank,
    int slab_count);

bool kimi_k3_sparse_natural_mask(
    const uint16_t * naturals,
    int slab_count,
    uint16_t * mask);

struct KimiK3CompactWireLayout {
    size_t metadata_bytes = 32;
    size_t gate_offset = 0;
    size_t up_offset = 0;
    size_t down_offset = 0;
    size_t total_bytes = 0;
};

bool kimi_k3_compact_wire_layout(
    int slab_count,
    size_t gate_slab_bytes,
    size_t up_slab_bytes,
    size_t down_slab_bytes,
    KimiK3CompactWireLayout * layout);

} // namespace dflash::common
