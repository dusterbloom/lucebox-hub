#include "kimi_k3/kimi_k3_sparse_scatter.h"
#include "device_runtime.h"

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <limits>
#include <vector>

#define CHECK(condition) do {                                                \
    if (!(condition)) {                                                      \
        std::fprintf(stderr, "CHECK failed at %s:%d: %s\n",               \
            __FILE__, __LINE__, #condition);                                 \
        return 1;                                                            \
    }                                                                        \
} while (false)

namespace {

constexpr size_t kMetadata = 8;
constexpr size_t kGateSlab = 4;
constexpr size_t kUpSlab = 3;
constexpr size_t kDownRowSlab = 2;
constexpr int kOutput = 2;
constexpr size_t kDownSlab = kDownRowSlab * kOutput;
constexpr size_t kRecord = kGateSlab + kUpSlab + kDownSlab;

std::vector<uint8_t> payload(
        const std::vector<uint16_t> & natural, uint8_t seed) {
    std::vector<uint8_t> result(kMetadata + natural.size() * kRecord, 0);
    std::memcpy(result.data(), natural.data(),
                natural.size() * sizeof(uint16_t));
    for (size_t i = kMetadata; i < result.size(); ++i) {
        result[i] = static_cast<uint8_t>(seed + i - kMetadata);
    }
    return result;
}

} // namespace

int main() {
    int device_count = 0;
    if (cudaGetDeviceCount(&device_count) != cudaSuccess || device_count == 0) {
        std::fprintf(stderr, "SKIP: no GPU is visible\n");
        return 77;
    }
    CHECK(cudaSetDevice(0) == cudaSuccess);

    constexpr size_t gate_bytes = 12 * kGateSlab;
    constexpr size_t up_bytes = 12 * kUpSlab;
    constexpr size_t down_row_bytes = 12 * kDownRowSlab;
    constexpr size_t down_bytes = kOutput * down_row_bytes;
    constexpr size_t compact_capacity = 128;
    void * gate_device = nullptr;
    void * up_device = nullptr;
    void * down_device = nullptr;
    void * compact_device = nullptr;
    CHECK(cudaMalloc(&gate_device, gate_bytes) == cudaSuccess);
    CHECK(cudaMalloc(&up_device, up_bytes) == cudaSuccess);
    CHECK(cudaMalloc(&down_device, down_bytes) == cudaSuccess);
    CHECK(cudaMalloc(&compact_device, compact_capacity) == cudaSuccess);

    const std::vector<uint8_t> first = payload({1, 3}, 11);
    const char * failure = nullptr;
    CHECK(!kimi_k3_sparse_scatter_upload_incremental(
        gate_device, gate_bytes, up_device, up_bytes,
        down_device, down_bytes, compact_device, compact_capacity,
        first.data(), first.size(), 2, sizeof(uint16_t),
        kGateSlab, kUpSlab, kDownSlab, kDownRowSlab,
        down_row_bytes, kOutput, false, &failure));
    CHECK(failure && std::strcmp(failure, "short metadata") == 0);

    std::vector<uint8_t> short_payload = first;
    short_payload.resize(short_payload.size() - 1);
    CHECK(!kimi_k3_sparse_scatter_upload_incremental(
        gate_device, gate_bytes, up_device, up_bytes,
        down_device, down_bytes, compact_device, compact_capacity,
        short_payload.data(), short_payload.size(), 2, kMetadata,
        kGateSlab, kUpSlab, kDownSlab, kDownRowSlab,
        down_row_bytes, kOutput, false, &failure));
    CHECK(failure && std::strcmp(failure, "short compact payload") == 0);

    CHECK(!kimi_k3_sparse_scatter_upload_incremental(
        gate_device, gate_bytes, up_device, up_bytes,
        down_device, down_bytes, compact_device, compact_capacity,
        first.data(), first.size(), 1, kMetadata,
        std::numeric_limits<size_t>::max(), 1, kDownSlab, kDownRowSlab,
        down_row_bytes, kOutput, false, &failure));
    CHECK(failure && std::strcmp(failure, "compact size overflow") == 0);

    CHECK(kimi_k3_sparse_scatter_upload(
        gate_device, gate_bytes, up_device, up_bytes,
        down_device, down_bytes, compact_device, compact_capacity,
        first.data(), first.size(), 2, kMetadata,
        kGateSlab, kUpSlab, kDownSlab, kDownRowSlab,
        down_row_bytes, kOutput, &failure));
    CHECK(failure == nullptr);

    std::vector<uint8_t> gate(gate_bytes);
    std::vector<uint8_t> up(up_bytes);
    std::vector<uint8_t> down(down_bytes);
    CHECK(cudaMemcpy(gate.data(), gate_device, gate.size(),
                     cudaMemcpyDeviceToHost) == cudaSuccess);
    CHECK(cudaMemcpy(up.data(), up_device, up.size(),
                     cudaMemcpyDeviceToHost) == cudaSuccess);
    CHECK(cudaMemcpy(down.data(), down_device, down.size(),
                     cudaMemcpyDeviceToHost) == cudaSuccess);
    for (size_t slab = 0; slab < 12; ++slab) {
        const bool selected = slab == 1 || slab == 3;
        for (size_t byte = 0; byte < kGateSlab; ++byte) {
            CHECK((gate[slab * kGateSlab + byte] != 0) == selected);
        }
        for (size_t byte = 0; byte < kUpSlab; ++byte) {
            CHECK((up[slab * kUpSlab + byte] != 0) == selected);
        }
        for (int row = 0; row < kOutput; ++row) {
            for (size_t byte = 0; byte < kDownRowSlab; ++byte) {
                CHECK((down[static_cast<size_t>(row) * down_row_bytes +
                            slab * kDownRowSlab + byte] != 0) == selected);
            }
        }
    }

    const std::vector<uint8_t> extension = payload({5}, 101);
    CHECK(kimi_k3_sparse_scatter_upload_incremental(
        gate_device, gate_bytes, up_device, up_bytes,
        down_device, down_bytes, compact_device, compact_capacity,
        extension.data(), extension.size(), 1, kMetadata,
        kGateSlab, kUpSlab, kDownSlab, kDownRowSlab,
        down_row_bytes, kOutput, false, &failure));
    CHECK(cudaMemcpy(gate.data(), gate_device, gate.size(),
                     cudaMemcpyDeviceToHost) == cudaSuccess);
    CHECK(gate[1 * kGateSlab] != 0);
    CHECK(gate[3 * kGateSlab] != 0);
    CHECK(gate[5 * kGateSlab] == 101);

    const std::vector<uint8_t> duplicate = payload({2, 2}, 1);
    CHECK(!kimi_k3_sparse_scatter_upload_incremental(
        gate_device, gate_bytes, up_device, up_bytes,
        down_device, down_bytes, compact_device, compact_capacity,
        duplicate.data(), duplicate.size(), 2, kMetadata,
        kGateSlab, kUpSlab, kDownSlab, kDownRowSlab,
        down_row_bytes, kOutput, false, &failure));
    CHECK(failure && std::strcmp(failure, "duplicate natural slab") == 0);

    CHECK(cudaFree(compact_device) == cudaSuccess);
    CHECK(cudaFree(down_device) == cudaSuccess);
    CHECK(cudaFree(up_device) == cudaSuccess);
    CHECK(cudaFree(gate_device) == cudaSuccess);
    return 0;
}
