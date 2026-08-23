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

__global__ void ordered_join_calibrated_batch_kernel(
        const float * rows, int row_count, const float * resident_means,
        int resident_mean_count, int width,
        const int32_t * row_indices, const float * weights,
        int operation_stride, const int32_t * operation_counts,
        float * outputs) {
    const int batch = blockIdx.y;
    const int operation_count = operation_counts[batch];
    const int32_t * batch_indices =
        row_indices + static_cast<size_t>(batch) * operation_stride;
    const float * batch_weights =
        weights + static_cast<size_t>(batch) * operation_stride;
    float * output = outputs + static_cast<size_t>(batch) * width;
    const int stride = blockDim.x * gridDim.x;
    for (int dimension = blockIdx.x * blockDim.x + threadIdx.x;
         dimension < width; dimension += stride) {
        float destination = 0.0f;
        for (int operation = 0; operation < operation_count; ++operation) {
            const int32_t encoded = batch_indices[operation];
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
            const float product = __fmul_rn(batch_weights[operation], value);
            destination = __fadd_rn(destination, product);
        }
        output[dimension] = destination;
    }
}

} // namespace

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

bool kimi_k3_ordered_join_calibrated_batch_launch(
        const float * device_rows, int row_count, int width,
        const float * device_resident_means, int resident_mean_count,
        const int32_t * device_row_indices, const float * device_weights,
        int operation_stride, const int32_t * device_operation_counts,
        int batch_count, float * device_outputs,
        const char ** failure_reason) {
    if (failure_reason) *failure_reason = nullptr;
    if (!device_rows || row_count < 0 || width <= 0 ||
        (!device_resident_means && resident_mean_count != 0) ||
        resident_mean_count < 0 || !device_row_indices || !device_weights ||
        operation_stride <= 0 || !device_operation_counts ||
        batch_count <= 0 || batch_count > 65535 || !device_outputs) {
        if (failure_reason) {
            *failure_reason = "invalid calibrated batch ordered-join shape";
        }
        return false;
    }
    constexpr int threads = 256;
    const int blocks = (width + threads - 1) / threads;
    const dim3 grid(blocks, batch_count);
    ordered_join_calibrated_batch_kernel<<<grid, threads>>>(
        device_rows, row_count, device_resident_means, resident_mean_count,
        width, device_row_indices, device_weights, operation_stride,
        device_operation_counts, device_outputs);
    if (cudaGetLastError() != cudaSuccess ||
        cudaStreamSynchronize(nullptr) != cudaSuccess) {
        if (failure_reason) {
            *failure_reason = "calibrated batch ordered-join launch failed";
        }
        return false;
    }
    return true;
}

} // namespace dflash::common
