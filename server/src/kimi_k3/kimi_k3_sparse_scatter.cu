#include "kimi_k3_sparse_scatter.h"

#include "../device_runtime.h"

#include <cstdint>

namespace {

__global__ void scatter_kimi_slabs(
        uint8_t * gate, uint8_t * up, uint8_t * down,
        const uint8_t * compact, int slab_count, size_t metadata_bytes,
        size_t gate_slab_bytes, size_t up_slab_bytes,
        size_t down_slab_bytes, size_t down_slab_row_bytes,
        size_t down_full_row_bytes, int output_dim) {
    const size_t record_bytes =
        gate_slab_bytes + up_slab_bytes + down_slab_bytes;
    const size_t logical_bytes = static_cast<size_t>(slab_count) * record_bytes;
    const size_t stride = static_cast<size_t>(blockDim.x) * gridDim.x;
    const auto * natural = reinterpret_cast<const uint16_t *>(compact);
    for (size_t index =
             static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         index < logical_bytes; index += stride) {
        const size_t slab = index / record_bytes;
        const size_t within = index - slab * record_bytes;
        const size_t natural_slab = natural[slab];
        const uint8_t value = compact[metadata_bytes + index];
        if (within < gate_slab_bytes) {
            gate[natural_slab * gate_slab_bytes + within] = value;
        } else if (within < gate_slab_bytes + up_slab_bytes) {
            const size_t offset = within - gate_slab_bytes;
            up[natural_slab * up_slab_bytes + offset] = value;
        } else {
            const size_t offset =
                within - gate_slab_bytes - up_slab_bytes;
            const size_t row = offset / down_slab_row_bytes;
            const size_t column = offset - row * down_slab_row_bytes;
            if (row < static_cast<size_t>(output_dim)) {
                down[row * down_full_row_bytes +
                     natural_slab * down_slab_row_bytes + column] = value;
            }
        }
    }
}
} // namespace

bool kimi_k3_sparse_scatter_upload(
        void * gate_device, size_t gate_full_bytes,
        void * up_device, size_t up_full_bytes,
        void * down_device, size_t down_full_bytes,
        void * compact_device, size_t compact_capacity,
        const void * compact_host, size_t compact_bytes,
        int slab_count, size_t metadata_bytes,
        size_t gate_slab_bytes, size_t up_slab_bytes,
        size_t down_slab_bytes, size_t down_slab_row_bytes,
        size_t down_full_row_bytes, int output_dim) {
    if (!gate_device || !up_device || !down_device || !compact_device ||
        !compact_host || slab_count <= 0 || slab_count > 12 ||
        metadata_bytes < static_cast<size_t>(slab_count) * sizeof(uint16_t) ||
        compact_bytes > compact_capacity || gate_slab_bytes == 0 ||
        up_slab_bytes == 0 || down_slab_bytes == 0 ||
        down_slab_row_bytes == 0 || down_full_row_bytes == 0 ||
        output_dim <= 0 ||
        down_slab_bytes !=
            down_slab_row_bytes * static_cast<size_t>(output_dim)) {
        return false;
    }
    cudaStream_t stream = nullptr;
    if (cudaMemsetAsync(gate_device, 0, gate_full_bytes, stream) != cudaSuccess ||
        cudaMemsetAsync(up_device, 0, up_full_bytes, stream) != cudaSuccess ||
        cudaMemsetAsync(down_device, 0, down_full_bytes, stream) != cudaSuccess ||
        cudaMemcpyAsync(compact_device, compact_host, compact_bytes,
                        cudaMemcpyHostToDevice, stream) != cudaSuccess) {
        return false;
    }
    const size_t record_bytes =
        gate_slab_bytes + up_slab_bytes + down_slab_bytes;
    const size_t logical_bytes = static_cast<size_t>(slab_count) * record_bytes;
    constexpr int threads = 256;
    const int blocks = static_cast<int>(
        (logical_bytes + threads - 1) / threads > 4096
            ? 4096 : (logical_bytes + threads - 1) / threads);
    scatter_kimi_slabs<<<blocks, threads, 0, stream>>>(
        static_cast<uint8_t *>(gate_device),
        static_cast<uint8_t *>(up_device),
        static_cast<uint8_t *>(down_device),
        static_cast<const uint8_t *>(compact_device), slab_count,
        metadata_bytes, gate_slab_bytes, up_slab_bytes, down_slab_bytes,
        down_slab_row_bytes, down_full_row_bytes, output_dim);
    if (cudaGetLastError() != cudaSuccess) return false;
    return cudaStreamSynchronize(stream) == cudaSuccess;
}
