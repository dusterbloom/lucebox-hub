#include "kimi_k3/kimi_k3_ordered_join.h"
#include "device_runtime.h"

#include <array>
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

    const char * failure = nullptr;
    // The fallback subtotal is (-1e20 + 1) before its one final add. Directly
    // appending fallback rows would produce 1 here instead of the frozen 0.
    CHECK(bits(expected[0]) == bits(0.0f));
    CHECK(bits(expected[1]) == bits(std::numeric_limits<float>::denorm_min()));
    CHECK(expected[2] == 0.0f);
    CHECK(std::isnan(expected[3]));
    CHECK(bits(expected[5]) == bits(0.0f));
    CHECK(bits(std::fma(weights[1], rows[1 * width + 5],
                       rows[0 * width + 5])) != bits(expected[5]));

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
    const auto check_guarded_index = [&](int32_t encoded, int transient_rows) {
        if (cudaMemcpy(
                device_indices, &encoded, sizeof(encoded),
                cudaMemcpyHostToDevice) != cudaSuccess ||
            !kimi_k3_ordered_join_launch(
                device_rows, transient_rows, width, device_rows, row_count,
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
    CHECK(check_guarded_index(row_count, row_count));
    CHECK(check_guarded_index(-1 - row_count, 0));
    CHECK(check_guarded_index(std::numeric_limits<int32_t>::min(), 0));

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

    // The production macro path batches only calibrated schedules. Compare
    // variable operation counts against both the scalar device contract and
    // an independent host teacher. The inherited rows cover signed zero,
    // denormals, cancellation, NaN propagation and non-FMA rounding.
    constexpr int batch_count = 4;
    constexpr int operation_stride = row_count;
    std::vector<int32_t> batch_indices(
        static_cast<size_t>(batch_count) * operation_stride, 0);
    std::vector<float> batch_weights(
        static_cast<size_t>(batch_count) * operation_stride, 0.0f);
    const std::array<int32_t, batch_count> batch_operations = {
        1, calibrated_operations, row_count, 4};
    const auto set_batch = [&](int batch, int operation, int32_t encoded,
                               float factor) {
        const size_t offset = static_cast<size_t>(batch) * operation_stride +
            static_cast<size_t>(operation);
        batch_indices[offset] = encoded;
        batch_weights[offset] = factor;
    };
    set_batch(0, 0, 0, 1.0f);
    for (int operation = 0; operation < calibrated_operations; ++operation) {
        set_batch(1, operation, row_indices[operation], weights[operation]);
    }
    for (int operation = 0; operation < row_count; ++operation) {
        set_batch(
            2, operation, maximum_indices[operation],
            maximum_weights[operation]);
    }
    set_batch(3, 0, -1, 1.0f);
    set_batch(3, 1, -2, 1.0f);
    set_batch(3, 2, 13, 1.0f);
    set_batch(3, 3, 14, 1.0f);

    std::vector<float> batch_teacher(
        static_cast<size_t>(batch_count) * width, 0.0f);
    for (int batch = 0; batch < batch_count; ++batch) {
        float * batch_expected = batch_teacher.data() +
            static_cast<size_t>(batch) * width;
        for (int operation = 0; operation < batch_operations[batch];
             ++operation) {
            const size_t offset = static_cast<size_t>(batch) *
                operation_stride + static_cast<size_t>(operation);
            const int32_t encoded = batch_indices[offset];
            const int source_row = encoded < 0 ? -1 - encoded : encoded;
            const float * source = rows.data() +
                static_cast<size_t>(source_row) * width;
            for (int dimension = 0; dimension < width; ++dimension) {
                batch_expected[dimension] = separate_mul_add(
                    batch_expected[dimension], batch_weights[offset],
                    source[dimension]);
            }
        }
    }

    int32_t * device_batch_indices = nullptr;
    float * device_batch_weights = nullptr;
    int32_t * device_batch_operations = nullptr;
    float * device_batch_outputs = nullptr;
    CHECK(cudaMalloc(
        reinterpret_cast<void **>(&device_batch_indices),
        batch_indices.size() * sizeof(int32_t)) == cudaSuccess);
    CHECK(cudaMalloc(
        reinterpret_cast<void **>(&device_batch_weights),
        batch_weights.size() * sizeof(float)) == cudaSuccess);
    CHECK(cudaMalloc(
        reinterpret_cast<void **>(&device_batch_operations),
        batch_operations.size() * sizeof(int32_t)) == cudaSuccess);
    CHECK(cudaMalloc(
        reinterpret_cast<void **>(&device_batch_outputs),
        static_cast<size_t>(batch_count) * width * sizeof(float)) ==
        cudaSuccess);
    CHECK(cudaMemcpy(
        device_batch_indices, batch_indices.data(),
        batch_indices.size() * sizeof(int32_t), cudaMemcpyHostToDevice) ==
        cudaSuccess);
    CHECK(cudaMemcpy(
        device_batch_weights, batch_weights.data(),
        batch_weights.size() * sizeof(float), cudaMemcpyHostToDevice) ==
        cudaSuccess);
    CHECK(cudaMemcpy(
        device_batch_operations, batch_operations.data(),
        batch_operations.size() * sizeof(int32_t),
        cudaMemcpyHostToDevice) == cudaSuccess);
    CHECK(kimi_k3_ordered_join_calibrated_batch_launch(
        device_rows, row_count, width, device_rows, row_count,
        device_batch_indices, device_batch_weights, operation_stride,
        device_batch_operations, batch_count, device_batch_outputs,
        &failure));
    std::vector<float> batch_actual(
        static_cast<size_t>(batch_count) * width);
    CHECK(cudaMemcpy(
        batch_actual.data(), device_batch_outputs,
        batch_actual.size() * sizeof(float), cudaMemcpyDeviceToHost) ==
        cudaSuccess);

    std::vector<float> scalar_actual(
        static_cast<size_t>(batch_count) * width);
    for (int batch = 0; batch < batch_count; ++batch) {
        const size_t offset = static_cast<size_t>(batch) * operation_stride;
        CHECK(cudaMemcpy(
            device_indices, batch_indices.data() + offset,
            static_cast<size_t>(batch_operations[batch]) * sizeof(int32_t),
            cudaMemcpyHostToDevice) == cudaSuccess);
        CHECK(cudaMemcpy(
            device_weights, batch_weights.data() + offset,
            static_cast<size_t>(batch_operations[batch]) * sizeof(float),
            cudaMemcpyHostToDevice) == cudaSuccess);
        CHECK(kimi_k3_ordered_join_launch(
            device_rows, row_count, width, device_rows, row_count,
            device_indices, device_weights, batch_operations[batch],
            batch_operations[batch], device_output, &failure));
        CHECK(cudaMemcpy(
            scalar_actual.data() + static_cast<size_t>(batch) * width,
            device_output, static_cast<size_t>(width) * sizeof(float),
            cudaMemcpyDeviceToHost) == cudaSuccess);
    }
    for (size_t index = 0; index < batch_actual.size(); ++index) {
        CHECK(bits(batch_actual[index]) == bits(scalar_actual[index]));
        CHECK(bits(batch_actual[index]) == bits(batch_teacher[index]));
    }
    (void) cudaFree(device_batch_outputs);
    (void) cudaFree(device_batch_operations);
    (void) cudaFree(device_batch_weights);
    (void) cudaFree(device_batch_indices);
    (void) cudaFree(device_output);
    (void) cudaFree(device_weights);
    (void) cudaFree(device_indices);
    (void) cudaFree(device_rows);
    return 0;
}
