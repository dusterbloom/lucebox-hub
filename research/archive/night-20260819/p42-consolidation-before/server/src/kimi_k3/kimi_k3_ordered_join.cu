#include "kimi_k3_ordered_join.h"

#include "../device_runtime.h"

#include <cstddef>

namespace dflash::common {
namespace {

bool validate_ordered_join_shape(
        const float * rows, int row_count, int width,
        const int32_t * row_indices, const float * weights,
        int operation_count, int calibrated_operations, float * output,
        const char ** failure_reason) {
    if (failure_reason) *failure_reason = nullptr;
    if (!rows || !row_indices || !weights || !output || row_count < 0 ||
        width <= 0 || operation_count <= 0 ||
        calibrated_operations < 0 ||
        calibrated_operations > operation_count) {
        if (failure_reason) *failure_reason = "invalid ordered-join shape";
        return false;
    }
    return true;
}

bool validate_ordered_join(
        const float * rows, int row_count, int width,
        const int32_t * row_indices, const float * weights,
        int operation_count, int calibrated_operations, float * output,
        const char ** failure_reason) {
    if (!validate_ordered_join_shape(
            rows, row_count, width, row_indices, weights, operation_count,
            calibrated_operations, output, failure_reason)) {
        return false;
    }
    for (int operation = 0; operation < operation_count; ++operation) {
        if (row_indices[operation] < 0 ||
            row_indices[operation] >= row_count) {
            if (failure_reason) {
                *failure_reason = "ordered-join row index is out of range";
            }
            return false;
        }
    }
    return true;
}

// Volatile materialization preserves the two-rounding host contract even when
// this translation unit is compiled with an aggressive host compiler.
inline float host_mul_add(float accumulator, float weight, float value) {
    volatile float product = weight * value;
    volatile float result = accumulator + product;
    return result;
}

__global__ void ordered_join_kernel(
        const float * rows, int row_count, const float * resident_means,
        int resident_mean_count, int width,
        const int32_t * row_indices,
        const float * weights, int operation_count,
        int calibrated_operations, float * output) {
    const int stride = blockDim.x * gridDim.x;
    for (int dimension = blockIdx.x * blockDim.x + threadIdx.x;
         dimension < width; dimension += stride) {
        float destination = 0.0f;
        for (int operation = 0; operation < calibrated_operations;
             ++operation) {
            const int32_t encoded = row_indices[operation];
            const bool transient = encoded >= 0;
            const bool valid = transient ? encoded < row_count :
                encoded != INT32_MIN && resident_means &&
                -1 - encoded < resident_mean_count;
            if (!valid) {
                output[dimension] = __int_as_float(0x7fffffff);
                return;
            }
            const float * source = transient ? rows : resident_means;
            const size_t row = transient ? static_cast<size_t>(encoded) :
                static_cast<size_t>(-1 - encoded);
            const float value = source[row * width + dimension];
            const float product = __fmul_rn(weights[operation], value);
            destination = __fadd_rn(destination, product);
        }
        if (calibrated_operations < operation_count) {
            float fallback = 0.0f;
            for (int operation = calibrated_operations;
                 operation < operation_count; ++operation) {
                const int32_t encoded = row_indices[operation];
                const bool transient = encoded >= 0;
                const bool valid = transient ? encoded < row_count :
                    encoded != INT32_MIN && resident_means &&
                    -1 - encoded < resident_mean_count;
                if (!valid) {
                    output[dimension] = __int_as_float(0x7fffffff);
                    return;
                }
                const float * source = transient ? rows : resident_means;
                const size_t row = transient ? static_cast<size_t>(encoded) :
                    static_cast<size_t>(-1 - encoded);
                const float value = source[row * width + dimension];
                const float product = __fmul_rn(weights[operation], value);
                fallback = __fadd_rn(fallback, product);
            }
            destination = __fadd_rn(destination, fallback);
        }
        output[dimension] = destination;
    }
}

} // namespace

bool kimi_k3_ordered_join_reference(
        const float * rows, int row_count, int width,
        const int32_t * row_indices, const float * weights,
        int operation_count, int calibrated_operations, float * output,
        const char ** failure_reason) {
    if (!validate_ordered_join(
            rows, row_count, width, row_indices, weights, operation_count,
            calibrated_operations, output, failure_reason)) {
        return false;
    }
    for (int dimension = 0; dimension < width; ++dimension) {
        float destination = 0.0f;
        for (int operation = 0; operation < calibrated_operations;
             ++operation) {
            destination = host_mul_add(
                destination, weights[operation],
                rows[static_cast<size_t>(row_indices[operation]) * width +
                     dimension]);
        }
        if (calibrated_operations < operation_count) {
            float fallback = 0.0f;
            for (int operation = calibrated_operations;
                 operation < operation_count; ++operation) {
                fallback = host_mul_add(
                    fallback, weights[operation],
                    rows[static_cast<size_t>(row_indices[operation]) * width +
                         dimension]);
            }
            volatile float joined = destination + fallback;
            destination = joined;
        }
        output[dimension] = destination;
    }
    return true;
}

bool kimi_k3_ordered_join_launch(
        const float * device_rows, int row_count, int width,
        const float * device_resident_means, int resident_mean_count,
        const int32_t * device_row_indices, const float * device_weights,
        int operation_count, int calibrated_operations,
        float * device_output, const char ** failure_reason) {
    if ((!device_resident_means && resident_mean_count != 0) ||
        resident_mean_count < 0 || !validate_ordered_join_shape(
            device_rows, row_count, width, device_row_indices, device_weights,
            operation_count, calibrated_operations, device_output,
            failure_reason)) {
        return false;
    }
    constexpr int threads = 256;
    const int blocks = (width + threads - 1) / threads;
    ordered_join_kernel<<<blocks, threads>>>(
        device_rows, row_count, device_resident_means, resident_mean_count,
        width, device_row_indices, device_weights,
        operation_count, calibrated_operations, device_output);
    if (cudaGetLastError() != cudaSuccess ||
        cudaStreamSynchronize(nullptr) != cudaSuccess) {
        if (failure_reason) *failure_reason = "ordered-join launch failed";
        return false;
    }
    return true;
}

} // namespace dflash::common
