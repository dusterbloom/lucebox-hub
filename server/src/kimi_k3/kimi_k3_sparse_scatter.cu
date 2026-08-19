#include "kimi_k3_sparse_scatter.h"

#include "../device_runtime.h"

#include <cstdint>
#include <cstring>

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
        if (natural_slab >= 12) continue;
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
        size_t down_full_row_bytes, int output_dim,
        const char ** failure_reason) {
    return kimi_k3_sparse_scatter_upload_incremental(
        gate_device, gate_full_bytes, up_device, up_full_bytes,
        down_device, down_full_bytes, compact_device, compact_capacity,
        compact_host, compact_bytes, slab_count, metadata_bytes,
        gate_slab_bytes, up_slab_bytes, down_slab_bytes,
        down_slab_row_bytes, down_full_row_bytes, output_dim, true,
        failure_reason);
}

bool kimi_k3_sparse_scatter_upload_incremental(
        void * gate_device, size_t gate_full_bytes,
        void * up_device, size_t up_full_bytes,
        void * down_device, size_t down_full_bytes,
        void * compact_device, size_t compact_capacity,
        const void * compact_host, size_t compact_bytes,
        int slab_count, size_t metadata_bytes,
        size_t gate_slab_bytes, size_t up_slab_bytes,
        size_t down_slab_bytes, size_t down_slab_row_bytes,
        size_t down_full_row_bytes, int output_dim,
        bool clear_destinations,
        const char ** failure_reason) {
    if (failure_reason) *failure_reason = nullptr;
    const auto invalid = [&](const char * reason) {
        if (failure_reason) *failure_reason = reason;
        return false;
    };
    if (!gate_device || !up_device || !down_device)
        return invalid("null expert destination");
    if (!compact_device || !compact_host)
        return invalid("null compact buffer");
    if (slab_count <= 0 || slab_count > 12)
        return invalid("invalid slab count");
    if (metadata_bytes <
            static_cast<size_t>(slab_count) * sizeof(uint16_t))
        return invalid("short metadata");
    if (compact_bytes > compact_capacity)
        return invalid("compact capacity");
    if (gate_slab_bytes == 0 || up_slab_bytes == 0 || down_slab_bytes == 0)
        return invalid("zero component bytes");
    if (down_slab_row_bytes == 0 || down_full_row_bytes == 0 ||
        output_dim <= 0)
        return invalid("invalid down geometry");
    if (down_slab_bytes !=
            down_slab_row_bytes * static_cast<size_t>(output_dim))
        return invalid("down slab extent mismatch");
    if (gate_slab_bytes > gate_full_bytes / 12 ||
        up_slab_bytes > up_full_bytes / 12 ||
        down_slab_row_bytes > down_full_row_bytes / 12 ||
        down_full_row_bytes > down_full_bytes /
            static_cast<size_t>(output_dim)) {
        return invalid("sparse destination extent mismatch");
    }
    uint16_t seen = 0;
    for (int slab = 0; slab < slab_count; ++slab) {
        uint16_t natural = 0;
        std::memcpy(
            &natural,
            static_cast<const uint8_t *>(compact_host) +
                static_cast<size_t>(slab) * sizeof(uint16_t),
            sizeof(uint16_t));
        if (natural >= 12) return invalid("natural slab out of range");
        const uint16_t bit = static_cast<uint16_t>(1u << natural);
        if ((seen & bit) != 0) return invalid("duplicate natural slab");
        seen = static_cast<uint16_t>(seen | bit);
    }
    cudaStream_t stream = nullptr;
    if (clear_destinations &&
        cudaMemsetAsync(gate_device, 0, gate_full_bytes, stream) != cudaSuccess) {
        if (failure_reason) *failure_reason = "gate memset";
        return false;
    }
    if (clear_destinations &&
        cudaMemsetAsync(up_device, 0, up_full_bytes, stream) != cudaSuccess) {
        if (failure_reason) *failure_reason = "up memset";
        return false;
    }
    if (clear_destinations &&
        cudaMemsetAsync(down_device, 0, down_full_bytes, stream) != cudaSuccess) {
        if (failure_reason) *failure_reason = "down memset";
        return false;
    }
    if (cudaMemcpyAsync(compact_device, compact_host, compact_bytes,
                        cudaMemcpyHostToDevice, stream) != cudaSuccess) {
        if (failure_reason) *failure_reason = "compact upload";
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
    if (cudaGetLastError() != cudaSuccess) {
        if (failure_reason) *failure_reason = "scatter launch";
        return false;
    }
    if (cudaStreamSynchronize(stream) != cudaSuccess) {
        if (failure_reason) *failure_reason = "scatter synchronize";
        return false;
    }
    return true;
}
