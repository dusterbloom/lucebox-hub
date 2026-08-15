#include "kimi_k3/kimi_k3_progressive_provider.h"

#include "ggml.h"

#include <algorithm>
#include <cassert>
#include <cstdlib>
#include <vector>

using namespace dflash::common;

static void require_raw_zero_block_dequantizes_exactly(ggml_type type) {
    constexpr int kElements = 256;
    const size_t row_bytes = ggml_row_size(type, kElements);
    std::vector<uint8_t> encoded(row_bytes, 0);
    std::vector<float> decoded(kElements, 1.0f);
    const ggml_type_traits * traits = ggml_get_type_traits(type);
    assert(traits != nullptr);
    assert(traits->to_float != nullptr);
    traits->to_float(encoded.data(), decoded.data(), kElements);
    assert(std::all_of(decoded.begin(), decoded.end(),
        [](float value) { return value == 0.0f; }));
}

int main() {
    // Sparse scratch initializes every omitted native quant block with zero
    // bytes.  Verify that this is an exact numeric zero in every routed qtype
    // present in the K3 checkpoint rather than assuming a byte convention.
    require_raw_zero_block_dequantizes_exactly(GGML_TYPE_IQ1_S);
    require_raw_zero_block_dequantizes_exactly(GGML_TYPE_IQ2_XXS);

    const int32_t experts[] = {2, 0};
    const float weights[] = {0.5f, 1.0f};
    // Per-expert values are already in progressive-prefix order.
    const float importance[] = {
        10.0f, 9.0f, 1.0f,
        1.0f, 1.0f, 1.0f,
        20.0f, 8.0f, 2.0f,
    };
    const std::vector<int32_t> slabs = select_kimi_k3_slab_prefix_ids(
        experts, weights, 2, importance, 3, 3, 4);
    assert(slabs.size() == 4);
    // Scores: e0/r0=10, e2/r0=10, e0/r1=9, e2/r1=4. Ties use
    // deterministic expert/rank ordering and every expert remains a prefix.
    const std::vector<int32_t> expected_slabs = {0, 6, 1, 7};
    assert(slabs == expected_slabs);
    for (int expert : {0, 2}) {
        std::vector<int> ranks;
        for (int32_t pseudo : slabs) {
            if (pseudo / 3 == expert) ranks.push_back(pseudo % 3);
        }
        std::sort(ranks.begin(), ranks.end());
        for (size_t i = 0; i < ranks.size(); ++i) {
            assert(ranks[i] == static_cast<int>(i));
        }
    }

    const float whole_importance[] = {2.0f, 1.0f, 3.0f};
    const std::vector<int32_t> whole = select_kimi_k3_whole_expert_routes(
        experts, weights, 2, whole_importance, 3, 1);
    assert(whole == std::vector<int32_t>{0});

    const std::vector<int32_t> route_prefix =
        select_kimi_k3_route_slab_prefix_ids(
            experts, weights, 2, whole_importance, 3, 3, 1, 2);
    assert(route_prefix == std::vector<int32_t>({0, 1}));
    assert(select_kimi_k3_route_slab_prefix_ids(
        experts, weights, 2, whole_importance, 3, 3, 1, 4).empty());

    // Nominal 4-slab request with one uncalibrated route: the uncalibrated
    // route is exact and only the three available calibrated slabs are read.
    const uint8_t calibrated[] = {1, 0, 0};
    const KimiK3CalibratedSlabPlan plan = plan_kimi_k3_calibrated_slabs(
        experts, weights, 2, importance, calibrated, 3, 3, 4);
    assert(plan.requested_budget == 4);
    assert(plan.selected_slab_ids == std::vector<int32_t>({0, 1, 2}));
    assert(plan.exact_route_indices == std::vector<int32_t>({0}));

#if defined(_WIN32)
    _putenv_s("DFLASH_KIMI_LAYER1_PROVIDER", "exact");
#else
    setenv("DFLASH_KIMI_LAYER1_PROVIDER", "exact", 1);
#endif
    std::unique_ptr<KimiK3RoutedOutputProvider> provider;
    std::string error;
    assert(create_kimi_k3_progressive_provider_from_env(
        nullptr, provider, &error));
    assert(!provider);
    return 0;
}
