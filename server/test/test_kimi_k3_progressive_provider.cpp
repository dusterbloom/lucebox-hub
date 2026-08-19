#include "kimi_k3/kimi_k3_progressive_provider.h"
#include "device_runtime.h"

#include "ggml.h"
#include "ggml-cuda.h"

#include <algorithm>
#include <cassert>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <limits>
#include <vector>

using namespace dflash::common;

#if defined(DFLASH_KIMI_P45_ASYNC_TEST_HOOK)
namespace dflash::common {
bool kimi_k3_run_p45_async_compact_sentinel(
    ggml_backend_t backend, std::string * err);
}
#endif

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
    using Delivery = KimiK3SparseDeliveryPolicy;
    using Upload = KimiK3SparseUpload;
    assert(kimi_k3_sparse_upload_for_call(
        Delivery::BufferedSlabs, false) == Upload::SlabCopies);
    assert(kimi_k3_sparse_upload_for_call(
        Delivery::DirectSlabs, false) == Upload::SlabCopies);
    assert(kimi_k3_sparse_upload_for_call(
        Delivery::CompactPageable, false) == Upload::PageableCompact);
    assert(kimi_k3_sparse_upload_for_call(
        Delivery::CompactPinned, false) == Upload::PinnedCompact);
    // An exact-fallback P27 call has no direct payload and must repack into
    // pinned staging rather than silently degrading to slab-copy uploads.
    assert(kimi_k3_sparse_upload_for_call(
        Delivery::DirectPinnedCompact, false) == Upload::PinnedCompact);
    for (Delivery delivery : {Delivery::BufferedSlabs, Delivery::DirectSlabs,
             Delivery::CompactPageable, Delivery::CompactPinned,
             Delivery::DirectPinnedCompact}) {
        assert(kimi_k3_sparse_upload_for_call(delivery, true) ==
            Upload::PrepackedCompact);
    }

    // P28 matches the physical P27 trace, not every logical route. A
    // calibrated zero-prefix route has no sidecar request, while an exact
    // fallback is retained as a native-expert request.
    assert(!kimi_k3_prefetch_route_has_physical_request(true, 0));
    assert(kimi_k3_prefetch_route_has_physical_request(true, 1));
    assert(kimi_k3_prefetch_route_has_physical_request(false, 0));

    const uint16_t natural_by_rank[] = {4, 1, 9, 2};
    const uint8_t selected_ranks[] = {1, 1, 0, 1};
    assert(kimi_k3_selected_natural_slab_mask(
        natural_by_rank, selected_ranks, 4) ==
        static_cast<uint16_t>((1u << 4) | (1u << 1) | (1u << 2)));
    uint8_t missing_ranks[] = {1, 1, 0, 1};
    kimi_k3_suppress_resident_slab_ranks(
        natural_by_rank, static_cast<uint16_t>(1u << 2),
        missing_ranks, 4);
    assert(std::vector<uint8_t>(missing_ranks, missing_ranks + 4) ==
        std::vector<uint8_t>({0, 0, 0, 1}));
    uint16_t physical_mask = 0;
    const uint16_t physical_naturals[] = {0, 5, 11};
    assert(kimi_k3_sparse_natural_mask(
        physical_naturals, 3, &physical_mask));
    assert(physical_mask ==
        static_cast<uint16_t>((1u << 0) | (1u << 5) | (1u << 11)));
    const uint16_t duplicate_naturals[] = {2, 2};
    assert(!kimi_k3_sparse_natural_mask(
        duplicate_naturals, 2, &physical_mask));
    const uint16_t invalid_naturals[] = {12};
    assert(!kimi_k3_sparse_natural_mask(
        invalid_naturals, 1, &physical_mask));
    assert(!kimi_k3_sparse_natural_mask(nullptr, 1, &physical_mask));
    assert(!kimi_k3_sparse_natural_mask(
        physical_naturals, 0, &physical_mask));
    assert(!kimi_k3_sparse_natural_mask(
        physical_naturals, 13, &physical_mask));

    KimiK3CompactWireLayout compact_layout;
    assert(kimi_k3_compact_wire_layout(3, 10, 20, 30, &compact_layout));
    assert(compact_layout.metadata_bytes == 32);
    assert(compact_layout.gate_offset == 32);
    assert(compact_layout.up_offset == 62);
    assert(compact_layout.down_offset == 122);
    assert(compact_layout.total_bytes == 212);
    assert(!kimi_k3_compact_wire_layout(0, 10, 20, 30, &compact_layout));
    assert(!kimi_k3_compact_wire_layout(13, 10, 20, 30, &compact_layout));
    assert(!kimi_k3_compact_wire_layout(
        12, std::numeric_limits<size_t>::max(), 20, 30,
        &compact_layout));

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

    const KimiK3CalibratedSlabPlan route_plan =
        plan_kimi_k3_calibrated_route_prefixes(
            experts, weights, 2, whole_importance, calibrated, 3, 3, 1, 2);
    assert(route_plan.requested_budget == 2);
    assert(route_plan.selected_slab_ids == std::vector<int32_t>({0, 1}));
    assert(route_plan.exact_route_indices == std::vector<int32_t>({0}));

    std::string error;
    const std::filesystem::path budget_path =
        std::filesystem::temp_directory_path() /
        "kimi_k3_h22_layer_budget_test.txt";
    {
        std::ofstream table(budget_path);
        for (int layer = 1; layer <= 92; ++layer) {
            table << layer << ' ' << (layer == 24 ? 24 : 96) << '\n';
        }
        assert(table.good());
    }
    std::vector<int32_t> layer_budgets;
    assert(parse_kimi_k3_layer_budget_table(
        budget_path.string(), layer_budgets, &error));
    assert(layer_budgets.size() == 92);
    assert(layer_budgets[0] == 96);
    assert(layer_budgets[23] == 24);
    std::filesystem::remove(budget_path);

#if defined(_WIN32)
    _putenv_s("DFLASH_KIMI_LAYER1_PROVIDER", "exact");
    _putenv_s("DFLASH_KIMI_P42_ORDERED_DEVICE_JOIN", "0");
    _putenv_s("DFLASH_KIMI_P45_ASYNC_COMPACT_QUEUE", "0");
    _putenv_s("DFLASH_KIMI_P43B_BALANCED_DUAL_OWNER", "0");
#else
    setenv("DFLASH_KIMI_LAYER1_PROVIDER", "exact", 1);
    setenv("DFLASH_KIMI_P42_ORDERED_DEVICE_JOIN", "0", 1);
    setenv("DFLASH_KIMI_P45_ASYNC_COMPACT_QUEUE", "0", 1);
    setenv("DFLASH_KIMI_P43B_BALANCED_DUAL_OWNER", "0", 1);
#endif
    std::unique_ptr<KimiK3RoutedOutputProvider> provider;
    error.clear();
    assert(create_kimi_k3_progressive_provider_from_env(
        nullptr, nullptr, nullptr, nullptr, provider, &error));
    assert(!provider);
#if defined(_WIN32)
    _putenv_s("DFLASH_KIMI_P42_ORDERED_DEVICE_JOIN", "1");
#else
    setenv("DFLASH_KIMI_P42_ORDERED_DEVICE_JOIN", "1", 1);
#endif
    error.clear();
    if (create_kimi_k3_progressive_provider_from_env(
            nullptr, nullptr, nullptr, nullptr, provider, &error) ||
        error.empty()) {
        std::fprintf(stderr, "P42 exact-provider selection did not fail closed\n");
        return 1;
    }
#if defined(_WIN32)
    _putenv_s("DFLASH_KIMI_P42_ORDERED_DEVICE_JOIN", "0");
    _putenv_s("DFLASH_KIMI_P45_ASYNC_COMPACT_QUEUE", "1");
#else
    setenv("DFLASH_KIMI_P42_ORDERED_DEVICE_JOIN", "0", 1);
    setenv("DFLASH_KIMI_P45_ASYNC_COMPACT_QUEUE", "1", 1);
#endif
    error.clear();
    if (create_kimi_k3_progressive_provider_from_env(
            nullptr, nullptr, nullptr, nullptr, provider, &error) ||
        error != "P45 async compact queue requires P42 ordered join") {
        std::fprintf(stderr, "P45 dependency did not fail closed: %s\n",
            error.c_str());
        return 1;
    }
#if defined(_WIN32)
    _putenv_s("DFLASH_KIMI_P45_ASYNC_COMPACT_QUEUE", "2");
#else
    setenv("DFLASH_KIMI_P45_ASYNC_COMPACT_QUEUE", "2", 1);
#endif
    error.clear();
    if (create_kimi_k3_progressive_provider_from_env(
            nullptr, nullptr, nullptr, nullptr, provider, &error) ||
        error != "DFLASH_KIMI_P45_ASYNC_COMPACT_QUEUE must be 0 or 1") {
        std::fprintf(stderr, "P45 flag validation did not fail closed: %s\n",
            error.c_str());
        return 1;
    }
#if defined(_WIN32)
    _putenv_s("DFLASH_KIMI_P45_ASYNC_COMPACT_QUEUE", "0");
#else
    setenv("DFLASH_KIMI_P45_ASYNC_COMPACT_QUEUE", "0", 1);
#endif
    // P43b is a dual-stream extension of P45, never a standalone mode.
#if defined(_WIN32)
    _putenv_s("DFLASH_KIMI_P43B_BALANCED_DUAL_OWNER", "1");
#else
    setenv("DFLASH_KIMI_P43B_BALANCED_DUAL_OWNER", "1", 1);
#endif
    error.clear();
    if (create_kimi_k3_progressive_provider_from_env(
            nullptr, nullptr, nullptr, nullptr, provider, &error) ||
        error != "P43b balanced dual owner requires P45") {
        std::fprintf(stderr, "P43b dependency did not fail closed: %s\n",
            error.c_str());
        return 1;
    }
#if defined(_WIN32)
    _putenv_s("DFLASH_KIMI_P43B_BALANCED_DUAL_OWNER", "0");
#else
    setenv("DFLASH_KIMI_P43B_BALANCED_DUAL_OWNER", "0", 1);
#endif
#if defined(DFLASH_KIMI_P45_ASYNC_TEST_HOOK)
    int device_count = 0;
    const bool explicit_device = std::getenv("DFLASH_TEST_GPU") != nullptr;
    if (cudaGetDeviceCount(&device_count) != cudaSuccess ||
        device_count == 0) {
        std::fprintf(stderr, "SKIP: no GPU is visible\n");
        return explicit_device ? 1 : 77;
    }
    int device = device_count > 1 ? 1 : 0;
    if (const char * raw_device = std::getenv("DFLASH_TEST_GPU")) {
        device = std::atoi(raw_device);
    }
    if (device < 0 || device >= device_count ||
        cudaSetDevice(device) != cudaSuccess) {
        std::fprintf(stderr, "SKIP: requested GPU %d is unavailable\n", device);
        return explicit_device ? 1 : 77;
    }
    ggml_backend_t backend = ggml_backend_cuda_init(device);
    if (!backend) {
        std::fprintf(stderr, "P45 sentinel backend initialization failed\n");
        return 1;
    }
    error.clear();
    const bool sentinel_ok =
        kimi_k3_run_p45_async_compact_sentinel(backend, &error);
    ggml_backend_free(backend);
    if (!sentinel_ok) {
        std::fprintf(stderr, "P45 async compact sentinel failed: %s\n",
            error.c_str());
        return 1;
    }
#endif
    return 0;
}
