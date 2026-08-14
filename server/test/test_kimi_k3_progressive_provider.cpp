#include "kimi_k3/kimi_k3_progressive_provider.h"

#include <algorithm>
#include <cassert>
#include <cstdlib>
#include <vector>

using namespace dflash::common;

int main() {
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
