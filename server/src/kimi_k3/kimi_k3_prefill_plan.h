#pragma once

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace dflash::common {

struct KimiK3PrefillPhysicalRead {
    int expert = -1;
    uint16_t natural_slab = 0;
    uint64_t record_offset = 0;
    uint64_t record_bytes = 0;
    uint64_t aligned_offset = 0;
    uint64_t aligned_bytes = 0;
};

struct KimiK3PrefillRouteRef {
    int row = -1;
    int native_route = -1;
    int canonical_partition = 0;
    int canonical_order = -1;
    int expert = -1;
    float weight = 0.0f;
    uint16_t natural_slab_mask = 0;
};

struct KimiK3PrefillExpertGroup {
    int expert = -1;
    uint16_t union_natural_slab_mask = 0;
    std::vector<KimiK3PrefillRouteRef> routes;
};

struct KimiK3PrefillLayerPlan {
    int width = 0;
    int top_k = 0;
    size_t requested_slab_records = 0;
    std::vector<KimiK3PrefillPhysicalRead> physical_reads;
    std::vector<KimiK3PrefillExpertGroup> expert_groups;
    // Flattened [row][canonical order] -> flattened native route index.
    std::vector<int> canonical_routes;
};

// Build a deterministic layer-local plan from already-qualified native routes
// and their calibrated natural-slab masks.  The planner owns no files, cache,
// device memory, or model state.
bool plan_kimi_k3_layer_major_prefill(
    int width,
    int top_k,
    int expert_count,
    int slabs_per_expert,
    uint64_t payload_offset,
    uint64_t slab_record_bytes,
    uint64_t io_alignment,
    const int32_t * selected_ids,
    const float * selected_weights,
    const uint8_t * canonical_partitions,
    const uint16_t * natural_slab_masks,
    KimiK3PrefillLayerPlan & plan,
    std::string * error = nullptr);

} // namespace dflash::common
