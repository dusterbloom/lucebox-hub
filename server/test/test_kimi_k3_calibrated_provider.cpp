#include "kimi_k3/kimi_k3_calibrated_provider.h"
#include "kimi_k3/kimi_k3_internal.h"
#include "kimi_k3/kimi_k3_prefill.h"
#include "device_runtime.h"

#include "ggml.h"
#include "ggml-cpu.h"
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

void require_snapshot_test(bool condition, const char * message) {
    if (condition) return;
    std::fprintf(stderr, "Kimi-K3 prefix snapshot test failed: %s\n", message);
    std::abort();
}

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

void fill_snapshot_tensor(ggml_tensor * tensor, float base) {
    require_snapshot_test(tensor != nullptr, "missing fixture tensor");
    if (tensor->type == GGML_TYPE_F16) {
        std::vector<ggml_fp16_t> values(ggml_nelements(tensor));
        for (size_t i = 0; i < values.size(); ++i) {
            values[i] = ggml_fp32_to_fp16(base + static_cast<float>(i));
        }
        ggml_backend_tensor_set(
            tensor, values.data(), 0, values.size() * sizeof(ggml_fp16_t));
        return;
    }
    require_snapshot_test(
        tensor->type == GGML_TYPE_F32, "unexpected fixture tensor type");
    std::vector<float> values(ggml_nelements(tensor));
    for (size_t i = 0; i < values.size(); ++i) {
        values[i] = base + static_cast<float>(i);
    }
    ggml_backend_tensor_set(
        tensor, values.data(), 0, values.size() * sizeof(float));
}

bool snapshot_tensor_prefix_equals(
        ggml_tensor * tensor, float base, size_t elements) {
    if (tensor->type == GGML_TYPE_F16) {
        std::vector<ggml_fp16_t> values(elements);
        ggml_backend_tensor_get(
            tensor, values.data(), 0, values.size() * sizeof(ggml_fp16_t));
        for (size_t i = 0; i < values.size(); ++i) {
            if (values[i] != ggml_fp32_to_fp16(
                    base + static_cast<float>(i))) return false;
        }
        return true;
    }
    std::vector<float> values(elements);
    ggml_backend_tensor_get(
        tensor, values.data(), 0, values.size() * sizeof(float));
    for (size_t i = 0; i < values.size(); ++i) {
        if (values[i] != base + static_cast<float>(i)) return false;
    }
    return true;
}

ggml_backend_t require_kimi_k3_prefix_snapshot_roundtrip() {
    ggml_backend_t live_backend = nullptr;
    int device_count = 0;
    const bool hip_available =
        cudaGetDeviceCount(&device_count) == cudaSuccess && device_count > 0;
    if (hip_available) {
        int device = device_count > 1 ? 1 : 0;
        if (const char * raw_device = std::getenv("DFLASH_TEST_GPU")) {
            device = std::atoi(raw_device);
        }
        require_snapshot_test(
            device >= 0 && device < device_count,
            "requested HIP fixture device is unavailable");
        require_snapshot_test(
            cudaSetDevice(device) == cudaSuccess,
            "HIP fixture device selection");
        live_backend = ggml_backend_cuda_init(device);
        require_snapshot_test(
            live_backend != nullptr, "HIP live backend initialization");
        cudaDeviceProp properties{};
        require_snapshot_test(
            cudaGetDeviceProperties(&properties, device) == cudaSuccess,
            "HIP fixture device properties");
        std::fprintf(stderr,
            "Kimi-K3 prefix snapshot fixture: live=HIP device=%d arch=%s snapshot=CPU\n",
            device, properties.gcnArchName);
    } else {
        std::fprintf(stderr,
            "Kimi-K3 prefix snapshot fixture: HIP unavailable; using CPU live cache\n");
        live_backend = ggml_backend_cpu_init();
    }
    require_snapshot_test(
        live_backend != nullptr, "live backend initialization");

    // Production keeps semantic state on gfx1151 while snapshots live on a
    // separate CPU backend. Keep that direction even when the live side falls
    // back to CPU so this never degenerates into a same-buffer/backend copy.
    ggml_backend_t snapshot_backend = ggml_backend_cpu_init();
    require_snapshot_test(
        snapshot_backend != nullptr, "snapshot backend initialization");

    KimiK3Weights weights;
    weights.n_layer = 2;
    weights.n_head = 2;
    weights.kda_head_dim = 4;
    weights.ssm_d_conv = 3;
    weights.kv_lora_rank = 5;
    weights.rope_dim = 3;
    weights.n_vocab = 7;
    weights.layers.resize(2);
    weights.layers[0].recurrent = true;
    weights.layers[1].recurrent = false;

    KimiK3Cache cache;
    require_snapshot_test(
        create_kimi_k3_cache(live_backend, weights, 16, cache),
        "cache creation");
    cache.cur_pos = 5;
    fill_snapshot_tensor(cache.layers[0].conv_state, 10.0f);
    fill_snapshot_tensor(cache.layers[0].ssm_state, 100.0f);
    fill_snapshot_tensor(cache.layers[1].mla_k, 1000.0f);
    std::vector<float> logits(7);
    for (size_t i = 0; i < logits.size(); ++i) {
        logits[i] = 2000.0f + static_cast<float>(i);
    }

    KimiK3PrefixSnapshot snapshot;
    require_snapshot_test(save_kimi_k3_prefix_snapshot(
        weights, cache, snapshot_backend, logits, snapshot), "snapshot save");
    require_snapshot_test(
        ggml_backend_buffer_is_host(snapshot.buf),
        "snapshot must reside in CPU memory");
    require_snapshot_test(
        snapshot.mla_k[1] && snapshot.mla_k[1]->ne[2] == 5,
        "right-sized MLA snapshot");
    require_snapshot_test(snapshot_tensor_prefix_equals(
        snapshot.conv_state[0], 10.0f,
        ggml_nelements(snapshot.conv_state[0])), "saved GPU-to-CPU conv bytes");
    require_snapshot_test(snapshot_tensor_prefix_equals(
        snapshot.ssm_state[0], 100.0f,
        ggml_nelements(snapshot.ssm_state[0])), "saved GPU-to-CPU SSM bytes");
    require_snapshot_test(snapshot_tensor_prefix_equals(
        snapshot.mla_k[1], 1000.0f,
        ggml_nelements(snapshot.mla_k[1])), "saved GPU-to-CPU MLA bytes");
    require_snapshot_test(
        snapshot.final_logits == logits, "final logits roundtrip");

    // A stale layout must fail before touching the live cache.
    fill_snapshot_tensor(cache.layers[0].conv_state, 300.0f);
    fill_snapshot_tensor(cache.layers[0].ssm_state, 400.0f);
    fill_snapshot_tensor(cache.layers[1].mla_k, 600.0f);
    cache.cur_pos = 13;
    snapshot.mla_k[1]->ne[2] = 4;
    require_snapshot_test(
        !restore_kimi_k3_prefix_snapshot(snapshot, cache),
        "malformed layout rejection");
    require_snapshot_test(cache.cur_pos == 13, "atomic rejected position");
    require_snapshot_test(snapshot_tensor_prefix_equals(
        cache.layers[0].conv_state, 300.0f,
        ggml_nelements(cache.layers[0].conv_state)), "atomic rejected conv");
    require_snapshot_test(snapshot_tensor_prefix_equals(
        cache.layers[0].ssm_state, 400.0f,
        ggml_nelements(cache.layers[0].ssm_state)), "atomic rejected SSM");
    require_snapshot_test(snapshot_tensor_prefix_equals(
        cache.layers[1].mla_k, 600.0f,
        static_cast<size_t>(weights.kv_lora_rank + weights.rope_dim) * 5),
        "atomic rejected MLA");
    snapshot.mla_k[1]->ne[2] = 5;

    cache.snapshot_pos = 3;
    cache.replay_base_pos = 3;
    cache.replay_n_tokens = 2;
    cache.snapshot_valid = true;
    cache.replay_valid = true;
    cache.recurrent_state_pristine = true;
    cache.replay_exact_rows = true;
    require_snapshot_test(
        restore_kimi_k3_prefix_snapshot(snapshot, cache), "snapshot restore");
    require_snapshot_test(cache.cur_pos == 5, "restored position");
    require_snapshot_test(
        cache.snapshot_pos == -1 && cache.replay_base_pos == -1 &&
        cache.replay_n_tokens == 0 && !cache.snapshot_valid &&
        !cache.replay_valid && !cache.recurrent_state_pristine &&
        !cache.replay_exact_rows, "replay metadata invalidation");
    require_snapshot_test(snapshot_tensor_prefix_equals(
        cache.layers[0].conv_state, 10.0f,
        ggml_nelements(cache.layers[0].conv_state)), "restored conv");
    require_snapshot_test(snapshot_tensor_prefix_equals(
        cache.layers[0].ssm_state, 100.0f,
        ggml_nelements(cache.layers[0].ssm_state)), "restored SSM");
    require_snapshot_test(snapshot_tensor_prefix_equals(
        cache.layers[1].mla_k, 1000.0f,
        static_cast<size_t>(weights.kv_lora_rank + weights.rope_dim) * 5),
        "restored MLA");

    free_kimi_k3_prefix_snapshot(snapshot);
    free_kimi_k3_cache(cache);
    ggml_backend_free(snapshot_backend);
    if (!hip_available) {
        ggml_backend_free(live_backend);
        return nullptr;
    }
    return live_backend;
}

} // namespace

int main() {
    ggml_backend_t snapshot_fixture_backend =
        require_kimi_k3_prefix_snapshot_roundtrip();
    if (std::getenv("DFLASH_KIMI_SNAPSHOT_TEST_ONLY")) {
        if (snapshot_fixture_backend) {
            ggml_backend_free(snapshot_fixture_backend);
        }
        return 0;
    }

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
    assert(kimi_k3_effective_slab_budget(24, 0) == 24);
    assert(kimi_k3_effective_slab_budget(24, 96) == 96);
    assert(kimi_k3_effective_slab_budget(192, 96) == 192);

    std::vector<KimiK3PositionBudget> position_budgets;
    error.clear();
    assert(parse_kimi_k3_position_budgets(
        "158:96,159:48", position_budgets, &error));
    assert(position_budgets.size() == 2);
    assert(kimi_k3_effective_position_slab_budget(
        24, 0, position_budgets, 157) == 24);
    assert(kimi_k3_effective_position_slab_budget(
        24, 0, position_budgets, 158) == 96);
    assert(kimi_k3_effective_position_slab_budget(
        24, 120, position_budgets, 158) == 120);
    error.clear();
    assert(!parse_kimi_k3_position_budgets(
        "158:96,158:48", position_budgets, &error));
    assert(!error.empty());
    error.clear();
    assert(!parse_kimi_k3_position_budgets(
        "-1:96", position_budgets, &error));
    assert(!error.empty());

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
    ggml_backend_t backend = snapshot_fixture_backend
        ? snapshot_fixture_backend : ggml_backend_cuda_init(device);
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
