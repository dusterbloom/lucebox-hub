#include "kimi_k3/kimi_k3_ordered_join.h"
#include "common/peer_access.h"
#include "device_runtime.h"

#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-cuda.h"
#include "ggml.h"

#include <cstdlib>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <limits>
#include <vector>

using namespace dflash::common;

#define CHECK(condition) do {                                                \
    if (!(condition)) {                                                       \
        std::fprintf(stderr, "CHECK failed at %s:%d: %s\n",                  \
            __FILE__, __LINE__, #condition);                                  \
        return 1;                                                             \
    }                                                                         \
} while (false)

static uint32_t bits(float value) {
    uint32_t result = 0;
    std::memcpy(&result, &value, sizeof(result));
    return result;
}

static float separate_mul_add(float accumulator, float weight, float value) {
    volatile float product = weight * value;
    volatile float result = accumulator + product;
    return result;
}

static int test_batched_peer_publication(int device_count) {
    if (device_count < 2) {
        std::fprintf(stderr, "SKIP: peer publication needs two GPUs\n");
        return 77;
    }
#if defined(_WIN32)
    _putenv_s("GGML_BATCH_PEER_COPIES", "1");
    _putenv_s("GGML_CUDA_BATCH_PEER_COPIES", "");
#else
    setenv("GGML_BATCH_PEER_COPIES", "1", 1);
    unsetenv("GGML_CUDA_BATCH_PEER_COPIES");
#endif
    constexpr int width = 3584;
    constexpr int rows = 16;
    ggml_backend_t source_backend = ggml_backend_cuda_init(0);
    ggml_backend_t destination_backend = ggml_backend_cuda_init(1);
    CHECK(source_backend != nullptr);
    CHECK(destination_backend != nullptr);
    g_peer_access_opt_in = true;
    if (!cross_device_peer_memcpy_ok(0, 1)) {
        std::fprintf(stderr, "SKIP: GPU0/GPU1 peer access is unavailable\n");
        return 77;
    }

    ggml_init_params source_params{};
    source_params.mem_size =
        (2 * rows + 3) * ggml_tensor_overhead() + 1024;
    source_params.no_alloc = true;
    ggml_context * source_context = ggml_init(source_params);
    CHECK(source_context != nullptr);
    ggml_tensor * reusable = ggml_new_tensor_1d(
        source_context, GGML_TYPE_F32, width);
    ggml_tensor * saved = ggml_new_tensor_2d(
        source_context, GGML_TYPE_F32, width, rows);
    std::vector<ggml_tensor *> saved_rows(rows);
    for (int row = 0; row < rows; ++row) {
        saved_rows[static_cast<size_t>(row)] = ggml_view_1d(
            source_context, saved, width,
            static_cast<size_t>(row) * width * sizeof(float));
    }
    ggml_backend_buffer_t source_buffer =
        ggml_backend_alloc_ctx_tensors(source_context, source_backend);
    CHECK(source_buffer != nullptr);

    ggml_init_params destination_params{};
    destination_params.mem_size =
        (rows + 5) * ggml_tensor_overhead() + 1024;
    destination_params.no_alloc = true;
    ggml_context * destination_context = ggml_init(destination_params);
    CHECK(destination_context != nullptr);
    ggml_tensor * published = ggml_new_tensor_2d(
        destination_context, GGML_TYPE_F32, width, rows);
    std::vector<ggml_tensor *> published_rows(rows);
    for (int row = 0; row < rows; ++row) {
        published_rows[static_cast<size_t>(row)] = ggml_view_1d(
            destination_context, published, width,
            static_cast<size_t>(row) * width * sizeof(float));
    }
    ggml_tensor * join_indices = ggml_new_tensor_1d(
        destination_context, GGML_TYPE_I32, rows);
    ggml_tensor * join_weights = ggml_new_tensor_1d(
        destination_context, GGML_TYPE_F32, rows);
    ggml_tensor * join_output = ggml_new_tensor_1d(
        destination_context, GGML_TYPE_F32, width);
    ggml_backend_buffer_t destination_buffer =
        ggml_backend_alloc_ctx_tensors(
            destination_context, destination_backend);
    CHECK(destination_buffer != nullptr);

    std::vector<float> expected(static_cast<size_t>(rows) * width);
    std::vector<float> actual(static_cast<size_t>(rows) * width);
    std::vector<int32_t> indices(rows);
    std::vector<float> weights(rows);
    std::vector<float> joined(width);
    for (int row = 0; row < rows; ++row) {
        indices[static_cast<size_t>(row)] = row;
        weights[static_cast<size_t>(row)] = row % 3 == 0 ? -0.25f : 0.5f;
    }
    for (int iteration = 0; iteration < 2; ++iteration) {
        for (int row = 0; row < rows; ++row) {
            float * values = expected.data() +
                static_cast<size_t>(row) * width;
            for (int dimension = 0; dimension < width; ++dimension) {
                values[dimension] = static_cast<float>(
                    iteration * 100000 + row * -17000 +
                    dimension * (row + 3) - 7000);
            }
            ggml_backend_tensor_set_async(
                source_backend, reusable, values, 0,
                ggml_nbytes(reusable));
            // Reuse the graph-owned producer immediately. Source-stream
            // ordering must preserve each result in its local row.
            ggml_backend_tensor_copy_async(
                source_backend, source_backend, reusable,
                saved_rows[static_cast<size_t>(row)]);
        }
        // Consecutive cross-device publications form one HIP peer batch.
        for (int row = 0; row < rows; ++row) {
            ggml_backend_tensor_copy_async(
                source_backend, destination_backend,
                saved_rows[static_cast<size_t>(row)],
                published_rows[static_cast<size_t>(row)]);
        }
        ggml_backend_synchronize(destination_backend);
        ggml_backend_tensor_get(
            published, actual.data(), 0, ggml_nbytes(published));
        for (size_t index = 0; index < actual.size(); ++index) {
            CHECK(bits(actual[index]) == bits(expected[index]));
        }
        ggml_backend_tensor_set(
            join_indices, indices.data(), 0, ggml_nbytes(join_indices));
        ggml_backend_tensor_set(
            join_weights, weights.data(), 0, ggml_nbytes(join_weights));
        const char * failure = nullptr;
        CHECK(kimi_k3_ordered_join_launch(
            static_cast<const float *>(published->data), rows, width,
            nullptr, 0,
            static_cast<const int32_t *>(join_indices->data),
            static_cast<const float *>(join_weights->data),
            rows, rows, static_cast<float *>(join_output->data), &failure));
        ggml_backend_tensor_get(
            join_output, joined.data(), 0, ggml_nbytes(join_output));
        for (int dimension = 0; dimension < width; ++dimension) {
            float teacher = 0.0f;
            for (int row = 0; row < rows; ++row) {
                teacher = separate_mul_add(
                    teacher, weights[static_cast<size_t>(row)],
                    expected[static_cast<size_t>(row) * width + dimension]);
            }
            CHECK(bits(joined[static_cast<size_t>(dimension)]) ==
                bits(teacher));
        }
    }
    ggml_backend_synchronize(source_backend);
    ggml_backend_synchronize(destination_backend);
    ggml_backend_buffer_free(destination_buffer);
    ggml_backend_buffer_free(source_buffer);
    ggml_free(destination_context);
    ggml_free(source_context);
    ggml_backend_free(destination_backend);
    ggml_backend_free(source_backend);
    return 0;
}

int main() {
    constexpr int width = 3584;
    constexpr int row_count = 16 * 13;
    std::vector<float> rows(static_cast<size_t>(row_count) * width, 0.0f);
    for (int row = 0; row < row_count; ++row) {
        for (int dimension = 5; dimension < width; ++dimension) {
            const int signed_value =
                ((row * 17 + dimension * 13) % 31) - 15;
            rows[static_cast<size_t>(row) * width + dimension] =
                static_cast<float>(signed_value) * 0x1p-18f;
        }
    }

    // Rows 0..11 are one actual K3 mean tail. Row 12 is the selected expert;
    // repeating rows 0 and 12 models stable duplicate-expert contributions.
    std::vector<int32_t> row_indices;
    std::vector<float> weights;
    for (int rank = 0; rank < 12; ++rank) {
        row_indices.push_back(rank);
        weights.push_back(1.0f);
    }
    row_indices.push_back(12);
    weights.push_back(1.0f);
    row_indices.push_back(0);
    weights.push_back(-0.0f);
    row_indices.push_back(12);
    weights.push_back(1.0f);
    const int calibrated_operations =
        static_cast<int>(row_indices.size());
    row_indices.push_back(13);
    weights.push_back(1.0f);
    row_indices.push_back(14);
    weights.push_back(1.0f);

    rows[0 * width + 0] = -0.0f;
    rows[1 * width + 0] = 0.0f;
    rows[0 * width + 1] = std::numeric_limits<float>::denorm_min();
    rows[0 * width + 2] = 1.0e20f;
    rows[13 * width + 2] = -1.0e20f;
    rows[14 * width + 2] = 1.0f;
    rows[1 * width + 3] = std::numeric_limits<float>::quiet_NaN();
    rows[12 * width + 4] = -0.0f;
    // If the second operation contracts, this is 2^-46; the frozen separate
    // multiply/add contract produces +0 exactly.
    for (int row = 0; row < row_count; ++row) {
        rows[static_cast<size_t>(row) * width + 5] = 0.0f;
    }
    rows[0 * width + 5] = -0x1.000004p0f;
    rows[1 * width + 5] = 0x1.000002p0f;
    weights[1] = 0x1.000002p0f;

    // Independent legacy-style teacher: operations are outermost, dimensions
    // innermost, with a separate exact-fallback subtotal and one final add.
    std::vector<float> expected(width, 0.0f);
    for (int operation = 0; operation < calibrated_operations; ++operation) {
        const float * row = rows.data() +
            static_cast<size_t>(row_indices[operation]) * width;
        for (int dimension = 0; dimension < width; ++dimension) {
            expected[dimension] = separate_mul_add(
                expected[dimension], weights[operation], row[dimension]);
        }
    }
    std::vector<float> fallback(width, 0.0f);
    for (int operation = calibrated_operations;
         operation < static_cast<int>(row_indices.size()); ++operation) {
        const float * row = rows.data() +
            static_cast<size_t>(row_indices[operation]) * width;
        for (int dimension = 0; dimension < width; ++dimension) {
            fallback[dimension] = separate_mul_add(
                fallback[dimension], weights[operation], row[dimension]);
        }
    }
    for (int dimension = 0; dimension < width; ++dimension) {
        volatile float joined = expected[dimension] + fallback[dimension];
        expected[dimension] = joined;
    }

    std::vector<float> reference(width);
    const char * failure = nullptr;
    CHECK(kimi_k3_ordered_join_reference(
        rows.data(), row_count, width, row_indices.data(), weights.data(),
        static_cast<int>(row_indices.size()), calibrated_operations,
        reference.data(), &failure));
    for (int dimension = 0; dimension < width; ++dimension) {
        if (std::isnan(expected[dimension])) {
            CHECK(std::isnan(reference[dimension]));
        } else {
            CHECK(bits(reference[dimension]) == bits(expected[dimension]));
        }
    }
    // The fallback subtotal is (-1e20 + 1) before its one final add. Directly
    // appending fallback rows would produce 1 here instead of the frozen 0.
    CHECK(bits(expected[0]) == bits(0.0f));
    CHECK(bits(expected[1]) == bits(std::numeric_limits<float>::denorm_min()));
    CHECK(expected[2] == 0.0f);
    CHECK(std::isnan(expected[3]));
    CHECK(bits(expected[5]) == bits(0.0f));
    CHECK(bits(std::fma(weights[1], rows[1 * width + 5],
                       rows[0 * width + 5])) != bits(expected[5]));

    std::vector<int32_t> invalid_indices = row_indices;
    invalid_indices[0] = row_count;
    CHECK(!kimi_k3_ordered_join_reference(
        rows.data(), row_count, width, invalid_indices.data(), weights.data(),
        static_cast<int>(invalid_indices.size()), calibrated_operations,
        reference.data(), &failure));

    int device_count = 0;
    if (cudaGetDeviceCount(&device_count) != cudaSuccess || device_count == 0) {
        std::fprintf(stderr, "SKIP: no GPU is visible\n");
        return 77;
    }
    int device = device_count > 1 ? 1 : 0;
    if (const char * value = std::getenv("DFLASH_TEST_GPU")) {
        device = std::atoi(value);
    }
    if (device < 0 || device >= device_count) {
        std::fprintf(stderr, "SKIP: requested GPU %d is unavailable\n", device);
        return 77;
    }
    const int peer_result = test_batched_peer_publication(device_count);
    if (peer_result != 0 && peer_result != 77) return peer_result;
    CHECK(cudaSetDevice(device) == cudaSuccess);
    float * device_rows = nullptr;
    int32_t * device_indices = nullptr;
    float * device_weights = nullptr;
    float * device_output = nullptr;
    CHECK(cudaMalloc(
        reinterpret_cast<void **>(&device_rows),
        rows.size() * sizeof(float)) == cudaSuccess);
    CHECK(cudaMalloc(
        reinterpret_cast<void **>(&device_indices),
        static_cast<size_t>(row_count) * sizeof(int32_t)) == cudaSuccess);
    CHECK(cudaMalloc(
        reinterpret_cast<void **>(&device_weights),
        static_cast<size_t>(row_count) * sizeof(float)) == cudaSuccess);
    CHECK(cudaMalloc(
        reinterpret_cast<void **>(&device_output),
        static_cast<size_t>(width) * sizeof(float)) == cudaSuccess);
    CHECK(cudaMemcpy(
        device_rows, rows.data(), rows.size() * sizeof(float),
        cudaMemcpyHostToDevice) == cudaSuccess);
    const auto launch = [&](const int32_t * indices, const float * factors,
                            int operations, int calibrated) {
        return cudaMemcpy(
                   device_indices, indices,
                   static_cast<size_t>(operations) * sizeof(int32_t),
                   cudaMemcpyHostToDevice) == cudaSuccess &&
            cudaMemcpy(
                   device_weights, factors,
                   static_cast<size_t>(operations) * sizeof(float),
                   cudaMemcpyHostToDevice) == cudaSuccess &&
            kimi_k3_ordered_join_launch(
                device_rows, row_count, width, nullptr, 0, device_indices,
                device_weights, operations, calibrated, device_output,
                &failure);
    };

    const int32_t mean_only_index = 0;
    const float mean_only_weight = -0.75f;
    CHECK(launch(&mean_only_index, &mean_only_weight, 1, 1));
    std::vector<float> actual(width);
    CHECK(cudaMemcpy(
        actual.data(), device_output,
        static_cast<size_t>(width) * sizeof(float),
        cudaMemcpyDeviceToHost) == cudaSuccess);
    for (int dimension = 0; dimension < width; ++dimension) {
        const float mean_expected = separate_mul_add(
            0.0f, mean_only_weight, rows[dimension]);
        if (std::isnan(mean_expected)) {
            CHECK(std::isnan(actual[dimension]));
        } else {
            CHECK(bits(actual[dimension]) == bits(mean_expected));
        }
    }
    const auto check_guarded_resident_index = [&](int32_t encoded) {
        if (cudaMemcpy(
                device_indices, &encoded, sizeof(encoded),
                cudaMemcpyHostToDevice) != cudaSuccess ||
            !kimi_k3_ordered_join_launch(
                device_rows, 0, width, device_rows, row_count,
                device_indices, device_weights, 1, 1, device_output,
                &failure) ||
            cudaMemcpy(
                actual.data(), device_output,
                static_cast<size_t>(width) * sizeof(float),
                cudaMemcpyDeviceToHost) != cudaSuccess) {
            return false;
        }
        return std::isnan(actual.front()) && std::isnan(actual.back());
    };
    CHECK(check_guarded_resident_index(-1 - row_count));
    CHECK(check_guarded_resident_index(
        std::numeric_limits<int32_t>::min()));

    // P42c mean-only schedules use negative descriptors while the transient
    // arena is empty. Exercise both boundaries and a middle resident row.
    const auto check_resident_mean = [&](int resident_row) {
        const int32_t encoded = -1 - resident_row;
        if (cudaMemcpy(
                device_indices, &encoded, sizeof(encoded),
                cudaMemcpyHostToDevice) != cudaSuccess ||
            cudaMemcpy(
                device_weights, &mean_only_weight, sizeof(mean_only_weight),
                cudaMemcpyHostToDevice) != cudaSuccess ||
            !kimi_k3_ordered_join_launch(
                device_rows, 0, width, device_rows, row_count,
                device_indices, device_weights, 1, 1, device_output,
                &failure) ||
            cudaMemcpy(
                actual.data(), device_output,
                static_cast<size_t>(width) * sizeof(float),
                cudaMemcpyDeviceToHost) != cudaSuccess) {
            return false;
        }
        for (int dimension = 0; dimension < width; ++dimension) {
            const float expected = separate_mul_add(
                0.0f, mean_only_weight,
                rows[static_cast<size_t>(resident_row) * width + dimension]);
            if (std::isnan(expected) ? !std::isnan(actual[dimension])
                                     : bits(actual[dimension]) != bits(expected)) {
                return false;
            }
        }
        return true;
    };
    CHECK(check_resident_mean(0));
    CHECK(check_resident_mean(row_count / 2));
    CHECK(check_resident_mean(row_count - 1));

    CHECK(launch(
        row_indices.data(), weights.data(),
        static_cast<int>(row_indices.size()), calibrated_operations));
    CHECK(cudaMemcpy(
        actual.data(), device_output,
        static_cast<size_t>(width) * sizeof(float),
        cudaMemcpyDeviceToHost) == cudaSuccess);
    for (int dimension = 0; dimension < width; ++dimension) {
        if (std::isnan(expected[dimension])) {
            CHECK(std::isnan(actual[dimension]));
        } else {
            if (bits(actual[dimension]) != bits(expected[dimension])) {
                std::fprintf(stderr,
                    "mixed mismatch dimension=%d expected=%a/%08x "
                    "actual=%a/%08x\n", dimension, expected[dimension],
                    bits(expected[dimension]), actual[dimension],
                    bits(actual[dimension]));
            }
            CHECK(bits(actual[dimension]) == bits(expected[dimension]));
        }
    }

    // Exercise the production descriptor ceiling: 16 transient expert rows,
    // 192 resident mean rows, and an eight-operation fallback subtotal.
    std::vector<int32_t> maximum_indices(row_count);
    std::vector<float> maximum_weights(row_count);
    constexpr int maximum_transient_rows = 16;
    constexpr int maximum_resident_operations = row_count - maximum_transient_rows;
    for (int operation = 0; operation < maximum_resident_operations;
         ++operation) {
        maximum_indices[operation] = -1 - operation;
        maximum_weights[operation] = (operation % 3 == 0) ? -0.25f : 0.5f;
    }
    for (int operation = maximum_resident_operations; operation < row_count;
         ++operation) {
        maximum_indices[operation] = operation - maximum_resident_operations;
        maximum_weights[operation] = (operation % 3 == 0) ? -0.25f : 0.5f;
    }
    constexpr int maximum_calibrated = row_count - 8;
    std::vector<float> maximum_expected(width, 0.0f);
    std::vector<float> maximum_fallback(width, 0.0f);
    for (int operation = 0; operation < maximum_calibrated; ++operation) {
        const int row_index = maximum_indices[operation] < 0
            ? -1 - maximum_indices[operation]
            : maximum_indices[operation];
        const float * row = rows.data() +
            static_cast<size_t>(row_index) * width;
        for (int dimension = 0; dimension < width; ++dimension) {
            maximum_expected[dimension] = separate_mul_add(
                maximum_expected[dimension], maximum_weights[operation],
                row[dimension]);
        }
    }
    for (int operation = maximum_calibrated; operation < row_count;
         ++operation) {
        const int row_index = maximum_indices[operation] < 0
            ? -1 - maximum_indices[operation]
            : maximum_indices[operation];
        const float * row = rows.data() +
            static_cast<size_t>(row_index) * width;
        for (int dimension = 0; dimension < width; ++dimension) {
            maximum_fallback[dimension] = separate_mul_add(
                maximum_fallback[dimension], maximum_weights[operation],
                row[dimension]);
        }
    }
    for (int dimension = 0; dimension < width; ++dimension) {
        volatile float joined =
            maximum_expected[dimension] + maximum_fallback[dimension];
        maximum_expected[dimension] = joined;
    }
    CHECK(cudaMemcpy(
        device_indices, maximum_indices.data(),
        static_cast<size_t>(row_count) * sizeof(int32_t),
        cudaMemcpyHostToDevice) == cudaSuccess);
    CHECK(cudaMemcpy(
        device_weights, maximum_weights.data(),
        static_cast<size_t>(row_count) * sizeof(float),
        cudaMemcpyHostToDevice) == cudaSuccess);
    CHECK(kimi_k3_ordered_join_launch(
        device_rows, maximum_transient_rows, width, device_rows, row_count,
        device_indices, device_weights, row_count, maximum_calibrated,
        device_output, &failure));
    CHECK(cudaMemcpy(
        actual.data(), device_output,
        static_cast<size_t>(width) * sizeof(float),
        cudaMemcpyDeviceToHost) == cudaSuccess);
    for (int dimension = 0; dimension < width; ++dimension) {
        if (std::isnan(maximum_expected[dimension])) {
            CHECK(std::isnan(actual[dimension]));
        } else {
            CHECK(bits(actual[dimension]) == bits(maximum_expected[dimension]));
        }
    }

    // Reuse the output/descriptor buffers after the maximum-size launch.
    CHECK(launch(
        row_indices.data(), weights.data(),
        static_cast<int>(row_indices.size()), calibrated_operations));
    CHECK(cudaMemcpy(
        actual.data(), device_output,
        static_cast<size_t>(width) * sizeof(float),
        cudaMemcpyDeviceToHost) == cudaSuccess);
    for (int dimension = 0; dimension < width; ++dimension) {
        if (std::isnan(expected[dimension])) {
            CHECK(std::isnan(actual[dimension]));
        } else {
            CHECK(bits(actual[dimension]) == bits(expected[dimension]));
        }
    }
    (void) cudaFree(device_output);
    (void) cudaFree(device_weights);
    (void) cudaFree(device_indices);
    (void) cudaFree(device_rows);
    return 0;
}
