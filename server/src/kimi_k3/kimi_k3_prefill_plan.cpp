#include "kimi_k3_prefill_plan.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <map>
#include <numeric>
#include <utility>

namespace dflash::common {

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
        std::string * error) {
    plan = KimiK3PrefillLayerPlan{};
    const auto fail = [&](const char * message) {
        if (error) *error = message;
        return false;
    };
    if (width <= 0 || top_k <= 0 || expert_count <= 0 ||
        slabs_per_expert <= 0 || slabs_per_expert > 16 ||
        slab_record_bytes == 0 || io_alignment == 0 ||
        (io_alignment & (io_alignment - 1)) != 0 || !selected_ids ||
        !selected_weights || !canonical_partitions || !natural_slab_masks) {
        return fail("invalid Kimi-K3 layer-major prefill geometry");
    }
    const size_t route_count =
        static_cast<size_t>(width) * static_cast<size_t>(top_k);
    if (route_count / static_cast<size_t>(top_k) !=
        static_cast<size_t>(width)) {
        return fail("Kimi-K3 layer-major route count overflow");
    }
    const uint16_t allowed_mask = slabs_per_expert == 16
        ? std::numeric_limits<uint16_t>::max()
        : static_cast<uint16_t>((1u << slabs_per_expert) - 1u);

    std::vector<int> canonical_order(route_count, -1);
    plan.canonical_routes.reserve(route_count);
    for (int row = 0; row < width; ++row) {
        std::vector<int> routes(static_cast<size_t>(top_k));
        std::iota(routes.begin(), routes.end(), 0);
        const size_t row_offset = static_cast<size_t>(row) * top_k;
        std::stable_sort(routes.begin(), routes.end(), [&](int left, int right) {
            const uint8_t left_partition = canonical_partitions[
                row_offset + static_cast<size_t>(left)];
            const uint8_t right_partition = canonical_partitions[
                row_offset + static_cast<size_t>(right)];
            if (left_partition != right_partition) {
                return left_partition < right_partition;
            }
            return selected_ids[row_offset + static_cast<size_t>(left)] <
                selected_ids[row_offset + static_cast<size_t>(right)];
        });
        for (int order = 0; order < top_k; ++order) {
            const int native_route = routes[static_cast<size_t>(order)];
            const size_t flattened = row_offset +
                static_cast<size_t>(native_route);
            canonical_order[flattened] = order;
            plan.canonical_routes.push_back(static_cast<int>(flattened));
        }
    }

    using SlabKey = std::pair<int, int>;
    std::map<SlabKey, KimiK3PrefillPhysicalRead> unique_reads;
    std::map<int, KimiK3PrefillExpertGroup> grouped_routes;
    for (size_t flattened = 0; flattened < route_count; ++flattened) {
        const int expert = selected_ids[flattened];
        const float weight = selected_weights[flattened];
        const uint8_t partition = canonical_partitions[flattened];
        const uint16_t mask = natural_slab_masks[flattened];
        if (expert < 0 || expert >= expert_count || partition > 1 ||
            !std::isfinite(weight) ||
            (mask & static_cast<uint16_t>(~allowed_mask)) != 0) {
            return fail("invalid Kimi-K3 layer-major route");
        }

        KimiK3PrefillExpertGroup & group = grouped_routes[expert];
        group.expert = expert;
        group.union_natural_slab_mask = static_cast<uint16_t>(
            group.union_natural_slab_mask | mask);
        group.routes.push_back({
            static_cast<int>(flattened / static_cast<size_t>(top_k)),
            static_cast<int>(flattened % static_cast<size_t>(top_k)),
            partition, canonical_order[flattened], expert, weight, mask});

        for (int natural = 0; natural < slabs_per_expert; ++natural) {
            if ((mask & (1u << natural)) == 0) continue;
            ++plan.requested_slab_records;
            const SlabKey key{expert, natural};
            if (unique_reads.find(key) != unique_reads.end()) continue;
            const uint64_t record_index =
                static_cast<uint64_t>(expert) * slabs_per_expert +
                static_cast<uint64_t>(natural);
            if (record_index >
                (std::numeric_limits<uint64_t>::max() - payload_offset) /
                    slab_record_bytes) {
                return fail("Kimi-K3 layer-major read offset overflow");
            }
            const uint64_t record_offset =
                payload_offset + record_index * slab_record_bytes;
            const uint64_t aligned_offset =
                record_offset & ~(io_alignment - 1);
            const uint64_t prefix = record_offset - aligned_offset;
            if (slab_record_bytes >
                std::numeric_limits<uint64_t>::max() - prefix -
                    (io_alignment - 1)) {
                return fail("Kimi-K3 layer-major read length overflow");
            }
            const uint64_t aligned_bytes =
                (prefix + slab_record_bytes + io_alignment - 1) &
                ~(io_alignment - 1);
            unique_reads.emplace(key, KimiK3PrefillPhysicalRead{
                expert, static_cast<uint16_t>(natural), record_offset,
                slab_record_bytes, aligned_offset, aligned_bytes});
        }
    }

    plan.width = width;
    plan.top_k = top_k;
    plan.physical_reads.reserve(unique_reads.size());
    for (const auto & entry : unique_reads) {
        plan.physical_reads.push_back(entry.second);
    }
    std::stable_sort(
        plan.physical_reads.begin(), plan.physical_reads.end(),
        [](const KimiK3PrefillPhysicalRead & left,
           const KimiK3PrefillPhysicalRead & right) {
            if (left.aligned_offset != right.aligned_offset) {
                return left.aligned_offset < right.aligned_offset;
            }
            return left.record_offset < right.record_offset;
        });

    plan.expert_groups.reserve(grouped_routes.size());
    for (auto & entry : grouped_routes) {
        KimiK3PrefillExpertGroup & group = entry.second;
        std::stable_sort(
            group.routes.begin(), group.routes.end(),
            [](const KimiK3PrefillRouteRef & left,
               const KimiK3PrefillRouteRef & right) {
                if (left.row != right.row) return left.row < right.row;
                if (left.canonical_partition != right.canonical_partition) {
                    return left.canonical_partition < right.canonical_partition;
                }
                return left.canonical_order < right.canonical_order;
            });
        plan.expert_groups.push_back(std::move(group));
    }
    return true;
}

} // namespace dflash::common
