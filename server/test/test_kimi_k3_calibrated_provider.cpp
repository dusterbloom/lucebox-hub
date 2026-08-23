#include "kimi_k3/kimi_k3_calibrated_provider.h"
#include "kimi_k3/kimi_k3_prefill.h"
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
#include <memory>
#include <string>
#include <vector>

using namespace dflash::common;

#if defined(DFLASH_KIMI_P45_ASYNC_TEST_HOOK)
namespace dflash::common {
bool kimi_k3_run_p45_async_compact_sentinel(
    ggml_backend_t backend, std::string * error);
}
#endif

namespace {

void set_env(const char * name, const char * value) {
#if defined(_WIN32)
    _putenv_s(name, value);
#else
    setenv(name, value, 1);
#endif
}

void require_raw_zero_block_dequantizes_exactly(ggml_type type) {
    constexpr int kElements = 256;
    const size_t row_bytes = ggml_row_size(type, kElements);
    std::vector<uint8_t> encoded(row_bytes, 0);
    std::vector<float> decoded(kElements, 1.0f);
    const ggml_type_traits * traits = ggml_get_type_traits(type);
    assert(traits && traits->to_float);
    traits->to_float(encoded.data(), decoded.data(), kElements);
    assert(std::all_of(decoded.begin(), decoded.end(),
        [](float value) { return value == 0.0f; }));
}

} // namespace

int main() {
    const KimiK3PrefillPolicy width_one{1, false};
    const KimiK3PrefillPolicy width_eight{8, true};
    const KimiK3PrefillPolicy width_64{64, true};
    const KimiK3PrefillPolicy width_1024{1024, true};
    assert(width_one.valid());
    assert(width_eight.valid() && width_64.valid() && width_1024.valid());
    assert((!KimiK3PrefillPolicy{1, true}.valid()));
    assert((!KimiK3PrefillPolicy{32, true}.valid()));
    assert(width_eight.next_width(7) == 1);
    assert(width_eight.next_width(8) == 8);
    assert(width_64.next_width(63) == 1);
    assert(width_1024.next_width(1024) == 1024);

    // Exercise cancellation through the existing width-one seam without a
    // model. This pins successful early termination and prevents a client
    // disconnect from being misreported as a prefill failure.
    KimiK3Weights test_weights;
    test_weights.n_vocab = 3;
    KimiK3Cache test_cache;
    MoeHybridStreamEngine test_stream;
    KimiK3PrefillContext test_context{
        reinterpret_cast<ggml_backend_t>(1), test_weights, test_cache,
        test_stream, nullptr};
    const KimiK3PrefillExecutor test_executor(test_context);
    std::vector<float> test_logits(3, 0.0f);
    KimiK3PrefillExecutionResult test_result;
    std::string test_error;
    int forward_calls = 0;
    int logits_calls = 0;
    const auto test_forward = [&](int32_t, int position) {
        assert(position == test_cache.cur_pos);
        ++forward_calls;
        ++test_cache.cur_pos;
        return true;
    };
    assert(test_executor.run(
        {1, 2, 3}, width_one, test_forward,
        [&](const std::vector<float> &) { ++logits_calls; }, []() {},
        [&]() { return forward_calls == 1; }, test_logits, test_result,
        &test_error));
    assert(test_result.cancelled && test_result.forward_calls == 1);
    assert(test_cache.cur_pos == 1 && logits_calls == 1);

    test_cache.cur_pos = 0;
    forward_calls = 0;
    logits_calls = 0;
    test_error.clear();
    assert(test_executor.run(
        {1}, width_one, test_forward,
        [&](const std::vector<float> &) { ++logits_calls; }, []() {},
        []() { return true; }, test_logits, test_result, &test_error));
    assert(test_result.cancelled && test_result.forward_calls == 0);
    assert(test_cache.cur_pos == 0 && forward_calls == 0 && logits_calls == 0);

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
    assert(kimi_k3_sparse_upload_for_call(
        Delivery::DirectPinnedCompact, false) == Upload::PinnedCompact);
    assert(kimi_k3_sparse_upload_for_call(
        Delivery::DirectPinnedCompact, true) == Upload::PrepackedCompact);

    const uint16_t natural_by_rank[] = {4, 1, 9, 2};
    const uint8_t selected[] = {1, 1, 0, 1};
    assert(kimi_k3_selected_natural_slab_mask(
        natural_by_rank, selected, 4) ==
        static_cast<uint16_t>((1u << 4) | (1u << 1) | (1u << 2)));
    uint8_t missing[] = {1, 1, 0, 1};
    kimi_k3_suppress_resident_slab_ranks(
        natural_by_rank, static_cast<uint16_t>(1u << 2), missing, 4);
    assert(std::vector<uint8_t>(missing, missing + 4) ==
        std::vector<uint8_t>({0, 0, 0, 1}));

    uint16_t mask = 0;
    const uint16_t naturals[] = {0, 5, 11};
    const uint16_t duplicate[] = {2, 2};
    assert(kimi_k3_sparse_natural_mask(naturals, 3, &mask));
    assert(mask == static_cast<uint16_t>((1u << 0) | (1u << 5) |
                                         (1u << 11)));
    assert(!kimi_k3_sparse_natural_mask(duplicate, 2, &mask));

    KimiK3CompactWireLayout layout;
    assert(kimi_k3_compact_wire_layout(3, 10, 20, 30, &layout));
    assert(layout.gate_offset == 32);
    assert(layout.up_offset == 62);
    assert(layout.down_offset == 122);
    assert(layout.total_bytes == 212);
    assert(!kimi_k3_compact_wire_layout(
        12, std::numeric_limits<size_t>::max(), 20, 30, &layout));

    // Omitted sparse blocks are initialized with raw zero bytes. Verify the
    // exact numerical contract for both routed qtypes in the K3 checkpoint.
    require_raw_zero_block_dequantizes_exactly(GGML_TYPE_IQ1_S);
    require_raw_zero_block_dequantizes_exactly(GGML_TYPE_IQ2_XXS);

    const int32_t experts[] = {2, 0};
    const float weights[] = {0.5f, 1.0f};
    const float importance[] = {
        10.0f, 9.0f, 1.0f,
        1.0f, 1.0f, 1.0f,
        20.0f, 8.0f, 2.0f,
    };
    const std::vector<int32_t> slabs = select_kimi_k3_slab_prefix_ids(
        experts, weights, 2, importance, 3, 3, 4);
    assert(slabs == std::vector<int32_t>({0, 6, 1, 7}));

    const uint8_t calibrated[] = {1, 0, 0};
    const KimiK3CalibratedSlabPlan plan = plan_kimi_k3_calibrated_slabs(
        experts, weights, 2, importance, calibrated, 3, 3, 4);
    assert(plan.requested_budget == 4);
    assert(plan.selected_slab_ids == std::vector<int32_t>({0, 1, 2}));
    assert(plan.exact_route_indices == std::vector<int32_t>({0}));

    const std::filesystem::path table_path =
        std::filesystem::temp_directory_path() /
        "kimi_k3_calibrated_budget_test.txt";
    {
        std::ofstream table(table_path);
        for (int layer = 1; layer <= 92; ++layer) {
            table << layer << ' ' << (layer == 24 ? 24 : 96) << '\n';
        }
        assert(table.good());
    }
    std::vector<int32_t> budgets;
    std::string error;
    assert(parse_kimi_k3_layer_budget_table(
        table_path.string(), budgets, &error));
    assert(budgets.size() == 92);
    assert(budgets[0] == 96 && budgets[23] == 24);
    std::filesystem::remove(table_path);

    set_env("DFLASH_KIMI_LAYER1_PROVIDER", "exact");
    set_env("DFLASH_KIMI_P42_ORDERED_DEVICE_JOIN", "0");
    set_env("DFLASH_KIMI_P45_ASYNC_COMPACT_QUEUE", "0");
    set_env("DFLASH_KIMI_EXACT_MACRO_UNION", "0");
    set_env("DFLASH_KIMI_EXACT_MACRO_UNION_PREFETCH", "0");
    std::unique_ptr<KimiK3RoutedOutputProvider> provider;
    assert(create_kimi_k3_calibrated_provider_from_env(
        nullptr, nullptr, provider, &error));
    assert(!provider);

    set_env("DFLASH_KIMI_P42_ORDERED_DEVICE_JOIN", "1");
    error.clear();
    assert(!create_kimi_k3_calibrated_provider_from_env(
        nullptr, nullptr, provider, &error));
    assert(!error.empty());

    set_env("DFLASH_KIMI_P42_ORDERED_DEVICE_JOIN", "0");
    set_env("DFLASH_KIMI_EXACT_MACRO_UNION_PREFETCH", "1");
    error.clear();
    assert(!create_kimi_k3_calibrated_provider_from_env(
        nullptr, nullptr, provider, &error));
    assert(error == "macro union prefetch requires exact macro union");
    set_env("DFLASH_KIMI_EXACT_MACRO_UNION_PREFETCH", "0");

    set_env("DFLASH_KIMI_LAYER1_PROVIDER", "all-layers-calibrated96");
    error.clear();
    assert(!create_kimi_k3_calibrated_provider_from_env(
        reinterpret_cast<ggml_backend_t>(1),
        reinterpret_cast<ggml_backend_t>(2), provider, &error));
    assert(error == "calibrated96 requires one expert/core backend");

    set_env("DFLASH_KIMI_P42_ORDERED_DEVICE_JOIN", "0");
    set_env("DFLASH_KIMI_LAYER1_PROVIDER", "all-slabs");
    error.clear();
    assert(!create_kimi_k3_calibrated_provider_from_env(
        nullptr, nullptr, provider, &error));
    assert(!error.empty());

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
