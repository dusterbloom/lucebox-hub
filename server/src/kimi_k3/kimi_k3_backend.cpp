#if defined(_WIN32) && !defined(NOMINMAX)
#define NOMINMAX
#endif

#include "kimi_k3_backend.h"
#include "kimi_k3_dflash_target.h"
#include "kimi_k3_progressive_provider.h"

#include "common/dynamic_backend.h"
#include "common/dflash_spec_decode.h"
#include "common/moe_expert_package.h"
#include "common/moe_hybrid_placement.h"
#include "common/moe_stream_cache_policy.h"
#include "common/sampler.h"
#include "dflash27b.h"

#include "ggml-cpu.h"
#include "ggml-cuda.h"

#include <algorithm>
#include <cerrno>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <numeric>
#include <string>
#include <thread>
#include <utility>
#include <vector>

#if defined(_WIN32)
#include <fcntl.h>
#include <io.h>
#include <process.h>
#include <sys/stat.h>
#include <windows.h>
#else
#include <fcntl.h>
#include <sys/stat.h>
#include <unistd.h>
#endif

namespace dflash::common {
namespace {

constexpr int kMaxDsparkBlockSize = 16;

constexpr char kKimiLogitsTraceMagic[8] = {
    'K', '3', 'L', 'O', 'G', '0', '0', '1'};

bool parse_teacher_forced_tokens(const char * raw,
                                 std::vector<int32_t> & out) {
    out.clear();
    if (!raw || !*raw) return true;
    const char * cursor = raw;
    while (*cursor) {
        char * end = nullptr;
        errno = 0;
        const long value = std::strtol(cursor, &end, 10);
        if (errno != 0 || end == cursor || value < 0 ||
            value > std::numeric_limits<int32_t>::max()) {
            return false;
        }
        out.push_back(static_cast<int32_t>(value));
        if (*end == '\0') return true;
        if (*end != ',') return false;
        cursor = end + 1;
        if (*cursor == '\0') return false;
    }
    return true;
}

bool parse_optional_binary_environment(const char * name, bool & value,
                                       std::string & error) {
    value = false;
    const char * raw = std::getenv(name);
    if (!raw || !*raw || std::strcmp(raw, "0") == 0) return true;
    if (std::strcmp(raw, "1") == 0) {
        value = true;
        return true;
    }
    error = std::string(name) + " must be 0 or 1";
    return false;
}

uint64_t process_storage_read_bytes() {
#if defined(_WIN32)
    return 0;
#else
    std::FILE * input = std::fopen("/proc/self/io", "r");
    if (!input) return 0;
    char key[64]{};
    unsigned long long value = 0;
    uint64_t result = 0;
    while (std::fscanf(input, "%63s %llu", key, &value) == 2) {
        if (std::strcmp(key, "read_bytes:") == 0) {
            result = static_cast<uint64_t>(value);
            break;
        }
    }
    std::fclose(input);
    return result;
#endif
}

uint64_t monotonic_delta(uint64_t after, uint64_t before) {
    return after >= before ? after - before : 0;
}

KimiK3RoutedRuntimeStats routed_stats_delta(
        const KimiK3RoutedRuntimeStats & after,
        const KimiK3RoutedRuntimeStats & before) {
    KimiK3RoutedRuntimeStats result;
#define KIMI_K3_DELTA(field) \
    result.field = monotonic_delta(after.field, before.field)
    KIMI_K3_DELTA(logical_provider_bytes);
    KIMI_K3_DELTA(explicit_read_bytes);
    KIMI_K3_DELTA(physical_direct_read_bytes);
    KIMI_K3_DELTA(direct_io_ns);
    KIMI_K3_DELTA(payload_h2d_bytes);
    KIMI_K3_DELTA(metadata_h2d_bytes);
    KIMI_K3_DELTA(compact_pack_ns);
    KIMI_K3_DELTA(expert_graph_ns);
    KIMI_K3_DELTA(expert_readback_ns);
    KIMI_K3_DELTA(compact_attempted);
    KIMI_K3_DELTA(compact_completed);
    KIMI_K3_DELTA(compact_fallbacks);
    KIMI_K3_DELTA(compact_invalid);
    KIMI_K3_DELTA(async_begins);
    KIMI_K3_DELTA(async_jobs);
    KIMI_K3_DELTA(async_h2d_calls);
    KIMI_K3_DELTA(async_h2d_bytes);
    KIMI_K3_DELTA(async_input_d2d_copies);
    KIMI_K3_DELTA(async_input_d2d_bytes);
    KIMI_K3_DELTA(async_graph_enqueues);
    KIMI_K3_DELTA(async_layer_flushes);
    KIMI_K3_DELTA(async_abort_syncs);
    KIMI_K3_DELTA(ordered_expert_d2d_copies);
    KIMI_K3_DELTA(ordered_expert_d2d_bytes);
    KIMI_K3_DELTA(ordered_join_launches);
    KIMI_K3_DELTA(ordered_output_d2d_copies);
    KIMI_K3_DELTA(ordered_output_d2d_bytes);
    KIMI_K3_DELTA(layer_major_prefetches);
    KIMI_K3_DELTA(layer_major_requested_records);
    KIMI_K3_DELTA(layer_major_unique_records);
#undef KIMI_K3_DELTA
    return result;
}

void print_prefill_census(
        const char * phase, size_t positions, size_t forwards,
        double seconds, uint64_t process_read_bytes,
        const KimiK3RoutedRuntimeStats & stats) {
    const double rate = seconds > 0.0
        ? static_cast<double>(positions) / seconds : 0.0;
    std::fprintf(stderr,
        "[kimi-k3-p56] phase=%s positions=%zu forwards=%zu seconds=%.9f "
        "positions-per-second=%.9f process-read-bytes=%llu "
        "logical-provider-bytes=%llu explicit-read-bytes=%llu "
        "physical-direct-read-bytes=%llu direct-io-ns=%llu "
        "payload-h2d-bytes=%llu metadata-h2d-bytes=%llu "
        "compact-pack-ns=%llu expert-graph-ns=%llu "
        "expert-readback-ns=%llu compact-attempted=%llu "
        "compact-completed=%llu compact-fallbacks=%llu compact-invalid=%llu "
        "async-begins=%llu async-jobs=%llu async-h2d-calls=%llu "
        "async-h2d-bytes=%llu async-input-d2d-copies=%llu "
        "async-input-d2d-bytes=%llu async-graph-enqueues=%llu "
        "async-layer-flushes=%llu async-abort-syncs=%llu "
        "ordered-expert-d2d-copies=%llu ordered-expert-d2d-bytes=%llu "
        "ordered-join-launches=%llu ordered-output-d2d-copies=%llu "
        "ordered-output-d2d-bytes=%llu layer-major-prefetches=%llu "
        "layer-major-requested-records=%llu "
        "layer-major-unique-records=%llu\n",
        phase, positions, forwards, seconds, rate,
        static_cast<unsigned long long>(process_read_bytes),
        static_cast<unsigned long long>(stats.logical_provider_bytes),
        static_cast<unsigned long long>(stats.explicit_read_bytes),
        static_cast<unsigned long long>(stats.physical_direct_read_bytes),
        static_cast<unsigned long long>(stats.direct_io_ns),
        static_cast<unsigned long long>(stats.payload_h2d_bytes),
        static_cast<unsigned long long>(stats.metadata_h2d_bytes),
        static_cast<unsigned long long>(stats.compact_pack_ns),
        static_cast<unsigned long long>(stats.expert_graph_ns),
        static_cast<unsigned long long>(stats.expert_readback_ns),
        static_cast<unsigned long long>(stats.compact_attempted),
        static_cast<unsigned long long>(stats.compact_completed),
        static_cast<unsigned long long>(stats.compact_fallbacks),
        static_cast<unsigned long long>(stats.compact_invalid),
        static_cast<unsigned long long>(stats.async_begins),
        static_cast<unsigned long long>(stats.async_jobs),
        static_cast<unsigned long long>(stats.async_h2d_calls),
        static_cast<unsigned long long>(stats.async_h2d_bytes),
        static_cast<unsigned long long>(stats.async_input_d2d_copies),
        static_cast<unsigned long long>(stats.async_input_d2d_bytes),
        static_cast<unsigned long long>(stats.async_graph_enqueues),
        static_cast<unsigned long long>(stats.async_layer_flushes),
        static_cast<unsigned long long>(stats.async_abort_syncs),
        static_cast<unsigned long long>(stats.ordered_expert_d2d_copies),
        static_cast<unsigned long long>(stats.ordered_expert_d2d_bytes),
        static_cast<unsigned long long>(stats.ordered_join_launches),
        static_cast<unsigned long long>(stats.ordered_output_d2d_copies),
        static_cast<unsigned long long>(stats.ordered_output_d2d_bytes),
        static_cast<unsigned long long>(stats.layer_major_prefetches),
        static_cast<unsigned long long>(
            stats.layer_major_requested_records),
        static_cast<unsigned long long>(stats.layer_major_unique_records));
}

uint64_t fnv1a_update(uint64_t hash, const void * data, size_t bytes) {
    const auto * input = static_cast<const uint8_t *>(data);
    for (size_t index = 0; index < bytes; ++index) {
        hash ^= input[index];
        hash *= UINT64_C(1099511628211);
    }
    return hash;
}

uint64_t hash_tensor_bytes(ggml_tensor * tensor, size_t offset,
                           size_t bytes, uint64_t hash) {
    constexpr size_t kChunkBytes = 1024 * 1024;
    std::vector<uint8_t> chunk(std::min(kChunkBytes, bytes));
    size_t consumed = 0;
    while (consumed < bytes) {
        const size_t count = std::min(chunk.size(), bytes - consumed);
        ggml_backend_tensor_get(
            tensor, chunk.data(), offset + consumed, count);
        hash = fnv1a_update(hash, chunk.data(), count);
        consumed += count;
    }
    return hash;
}

uint64_t recurrent_cache_hash(const KimiK3Cache & cache) {
    uint64_t hash = UINT64_C(14695981039346656037);
    for (size_t layer_index = 0; layer_index < cache.layers.size();
         ++layer_index) {
        const KimiK3LayerCache & layer = cache.layers[layer_index];
        if (!layer.ssm_state) continue;
        hash = fnv1a_update(hash, &layer_index, sizeof(layer_index));
        hash = hash_tensor_bytes(
            layer.conv_state, 0, ggml_nbytes(layer.conv_state), hash);
        hash = hash_tensor_bytes(
            layer.ssm_state, 0, ggml_nbytes(layer.ssm_state), hash);
    }
    return hash;
}

std::vector<uint64_t> recurrent_layer_hashes(
        const KimiK3Cache & cache, bool convolution) {
    std::vector<uint64_t> hashes(cache.layers.size(), 0);
    for (size_t layer_index = 0; layer_index < cache.layers.size();
         ++layer_index) {
        const KimiK3LayerCache & layer = cache.layers[layer_index];
        ggml_tensor * tensor = convolution
            ? layer.conv_state : layer.ssm_state;
        if (!tensor) continue;
        uint64_t hash = UINT64_C(14695981039346656037);
        hash = fnv1a_update(hash, &layer_index, sizeof(layer_index));
        hashes[layer_index] = hash_tensor_bytes(
            tensor, 0, ggml_nbytes(tensor), hash);
    }
    return hashes;
}

uint64_t mla_rows_hash(const KimiK3Cache & cache, int base_pos,
                       int n_tokens) {
    uint64_t hash = UINT64_C(14695981039346656037);
    for (size_t layer_index = 0; layer_index < cache.layers.size();
         ++layer_index) {
        const KimiK3LayerCache & layer = cache.layers[layer_index];
        if (!layer.mla_k) continue;
        hash = fnv1a_update(hash, &layer_index, sizeof(layer_index));
        const size_t row_bytes =
            ggml_row_size(layer.mla_k->type, layer.mla_k->ne[0]);
        for (int token = 0; token < n_tokens; ++token) {
            const size_t offset = static_cast<size_t>(base_pos + token) *
                layer.mla_k->nb[2];
            hash = hash_tensor_bytes(
                layer.mla_k, offset, row_bytes, hash);
        }
    }
    return hash;
}

std::vector<uint64_t> mla_layer_row_hashes(
        const KimiK3Cache & cache, int base_pos, int n_tokens) {
    std::vector<uint64_t> hashes(cache.layers.size(), 0);
    for (size_t layer_index = 0; layer_index < cache.layers.size();
         ++layer_index) {
        const KimiK3LayerCache & layer = cache.layers[layer_index];
        if (!layer.mla_k) continue;
        uint64_t hash = UINT64_C(14695981039346656037);
        hash = fnv1a_update(hash, &layer_index, sizeof(layer_index));
        const size_t row_bytes =
            ggml_row_size(layer.mla_k->type, layer.mla_k->ne[0]);
        for (int token = 0; token < n_tokens; ++token) {
            const size_t offset = static_cast<size_t>(base_pos + token) *
                layer.mla_k->nb[2];
            hash = hash_tensor_bytes(
                layer.mla_k, offset, row_bytes, hash);
        }
        hashes[layer_index] = hash;
    }
    return hashes;
}

int first_hash_mismatch(const std::vector<uint64_t> & reference,
                        const std::vector<uint64_t> & candidate) {
    if (reference.size() != candidate.size()) return 0;
    for (size_t index = 0; index < reference.size(); ++index) {
        if (reference[index] != candidate[index]) {
            return static_cast<int>(index);
        }
    }
    return -1;
}

void maybe_release_kimi_mapped_pages(const KimiK3Weights & weights) {
    const char * raw = std::getenv("DFLASH_KIMI_MMAP_DROP_PAGES");
    if (!raw || !*raw || std::strcmp(raw, "0") == 0) return;
    for (const GgufMmap & mapping : weights.mapped_shards) {
        mapping.advise_dontneed();
    }
}

struct KimiLogitsTraceHeader {
    char magic[8];
    uint32_t version = 1;
    uint32_t vocabulary = 0;
    uint64_t rows = 0;
    uint64_t prompt_tokens = 0;
    uint64_t generated_tokens = 0;
    uint32_t storage = 0; // 0 = float32
    uint32_t reserved = 0;
};
static_assert(sizeof(KimiLogitsTraceHeader) == 48,
              "Kimi logits trace header must remain byte-stable");

void close_file_descriptor(int fd) {
#if defined(_WIN32)
    ::_close(fd);
#else
    ::close(fd);
#endif
}

class ScopedFileDescriptors {
public:
    ~ScopedFileDescriptors() {
        for (int fd : values_) close_file_descriptor(fd);
    }

    void add(int fd) { values_.push_back(fd); }

    void close(int fd) {
        const auto found = std::find(values_.begin(), values_.end(), fd);
        if (found == values_.end()) return;
        close_file_descriptor(fd);
        values_.erase(found);
    }

private:
    std::vector<int> values_;
};

bool open_nvme_source(const std::string & path,
                      ScopedFileDescriptors & descriptors,
                      MoeNvmeSource & out,
                      std::string & error) {
#if defined(_WIN32)
    const int fd = ::_open(path.c_str(), _O_RDONLY | _O_BINARY);
#else
    const int fd = ::open(path.c_str(), O_RDONLY | O_CLOEXEC);
#endif
    if (fd < 0) {
        error = "cannot open " + path + ": " + std::strerror(errno);
        return false;
    }
    uint64_t bytes = 0;
#if defined(_WIN32)
    struct _stat64 stat_buffer {};
    if (::_fstat64(fd, &stat_buffer) == 0 && stat_buffer.st_size > 0) {
        bytes = static_cast<uint64_t>(stat_buffer.st_size);
    }
#else
    struct stat stat_buffer {};
    if (::fstat(fd, &stat_buffer) == 0 && stat_buffer.st_size > 0) {
        bytes = static_cast<uint64_t>(stat_buffer.st_size);
    }
#endif
    if (bytes == 0 || bytes > std::numeric_limits<size_t>::max()) {
        close_file_descriptor(fd);
        error = "cannot determine file size for " + path;
        return false;
    }
    descriptors.add(fd);
    out = {nullptr, static_cast<size_t>(bytes), fd};
    return true;
}

bool file_path_exists(const char * path) {
    if (!path || !*path) return false;
#if defined(_WIN32)
    struct _stat64 st {};
    return ::_stat64(path, &st) == 0 && st.st_size > 0;
#else
    struct stat st {};
    return ::stat(path, &st) == 0 && st.st_size > 0;
#endif
}

bool create_package_output(const std::string & path,
                           ScopedFileDescriptors & descriptors,
                           int & fd, std::string & error) {
#if defined(_WIN32)
    fd = ::_open(path.c_str(),
                 _O_CREAT | _O_TRUNC | _O_RDWR | _O_BINARY,
                 _S_IREAD | _S_IWRITE);
#else
    fd = ::open(path.c_str(),
                O_CREAT | O_TRUNC | O_RDWR | O_CLOEXEC, 0644);
#endif
    if (fd < 0) {
        error = "cannot create " + path + ": " + std::strerror(errno);
        return false;
    }
    descriptors.add(fd);
    return true;
}

std::string package_temporary_path(const std::string & path) {
#if defined(_WIN32)
    const int process_id = ::_getpid();
#else
    const int process_id = static_cast<int>(::getpid());
#endif
    return path + ".partial." + std::to_string(process_id);
}

bool publish_package(const std::string & temporary,
                     const std::string & destination,
                     std::string & error) {
#if defined(_WIN32)
    if (::MoveFileExA(temporary.c_str(), destination.c_str(),
                      MOVEFILE_REPLACE_EXISTING |
                      MOVEFILE_WRITE_THROUGH)) {
        return true;
    }
    error = "cannot publish expert package " + destination +
            ": Windows error " + std::to_string(::GetLastError());
#else
    if (::rename(temporary.c_str(), destination.c_str()) == 0) return true;
    error = "cannot publish expert package " + destination + ": " +
            std::strerror(errno);
#endif
    return false;
}

bool sync_path(const std::string & path) {
#if defined(_WIN32)
    const int fd = ::_open(path.c_str(), _O_RDONLY | _O_BINARY);
    if (fd < 0) return false;
    const bool ok = ::_commit(fd) == 0;
#else
    const int fd = ::open(path.c_str(), O_RDONLY | O_CLOEXEC);
    if (fd < 0) return false;
    const bool ok = ::fsync(fd) == 0;
#endif
    close_file_descriptor(fd);
    return ok;
}

} // namespace

ggml_backend_t init_kimi_k3_core_backend(
        KimiK3CorePlacement placement, int gpu, std::string * error) {
    if (placement == KimiK3CorePlacement::Accelerator) {
        ggml_backend_t backend = ggml_backend_cuda_init(gpu);
        if (!backend && error) {
            *error = "accelerator backend init failed for device " +
                std::to_string(gpu);
        }
        return backend;
    }

    ggml_backend_t backend = ggml_backend_cpu_init();
    if (!backend) {
        if (error) *error = "CPU backend init failed";
        return nullptr;
    }
    int threads = std::max(1, static_cast<int>(
        std::thread::hardware_concurrency()));
    if (const char * raw = std::getenv("DFLASH_KIMI_CPU_THREADS")) {
        char * end = nullptr;
        const long parsed = std::strtol(raw, &end, 10);
        if (end != raw && *end == '\0' && parsed > 0 && parsed <= 1024) {
            threads = static_cast<int>(parsed);
        }
    }
    ggml_backend_cpu_set_n_threads(backend, threads);
    std::fprintf(stderr, "[kimi-k3] CPU core backend threads=%d\n", threads);
    return backend;
}

KimiK3Backend::KimiK3Backend(const KimiK3BackendConfig & cfg) : cfg_(cfg) {}

KimiK3Backend::~KimiK3Backend() {
    shutdown();
}

void KimiK3Backend::release_expert_backend() {
    free_kimi_k3_moe_core_offload(moe_core_offload_);
    if (expert_backend_) {
        ggml_backend_free(expert_backend_);
        expert_backend_ = nullptr;
    }
    expert_gpu_ = -1;
    expert_backend_kind_ = PlacementBackend::Auto;
}

bool KimiK3Backend::init_streaming() {
    routing_stats_.reset();
    routing_stats_out_path_.clear();
    if (!weights_.routed_experts_streamed ||
        weights_.streamed_layer_regions.empty() ||
        weights_.max_streamed_expert_bytes == 0) {
        std::fprintf(stderr,
                     "[kimi-k3] routed expert streaming metadata is incomplete\n");
        return false;
    }

    MoeExpertOwnerPlacement owner;
    std::string error;
    if (!resolve_moe_expert_owner_placement(
            cfg_.device.primary_gpu(), cfg_.expert_gpu,
            owner, &error)) {
        std::fprintf(stderr,
                     "[kimi-k3] invalid expert-owner placement: %s\n",
                     error.c_str());
        return false;
    }
    expert_gpu_ = owner.expert_gpu;
    const bool cpu_core =
        cfg_.core_placement == KimiK3CorePlacement::Cpu;
    const PlacementBackend primary_kind = cpu_core
        ? PlacementBackend::Auto
        : (cfg_.device.backend == PlacementBackend::Auto
            ? compiled_placement_backend() : cfg_.device.backend);
    PlacementBackend expert_kind = cpu_core
        ? compiled_placement_backend() : primary_kind;
    if (const char * raw = std::getenv("DFLASH_MOE_TP_BACKEND")) {
        if (*raw && (!parse_placement_backend(raw, expert_kind) ||
                     expert_kind == PlacementBackend::Auto)) {
            std::fprintf(stderr,
                         "[kimi-k3] invalid DFLASH_MOE_TP_BACKEND=%s; "
                         "expected cuda or hip\n", raw);
            return false;
        }
    }
    const bool accelerator_only_stream = cpu_core;
    const bool heterogeneous =
        expert_kind != primary_kind || owner.heterogeneous();
    if (heterogeneous) {
        expert_backend_ = init_placement_backend(
            expert_kind, expert_gpu_, &error);
        if (!expert_backend_) {
            std::fprintf(stderr,
                         "[kimi-k3] expert backend init failed for %s:%d: %s\n",
                         placement_backend_name(expert_kind), expert_gpu_,
                         error.c_str());
            expert_gpu_ = -1;
            return false;
        }
        expert_backend_kind_ = expert_kind;
        std::fprintf(stderr,
                     "[kimi-k3] in-process routed placement core=%s:%d "
                     "expert=%s:%d transfer=backend-staged\n",
                     cpu_core ? "cpu" : placement_backend_name(primary_kind),
                     cfg_.device.primary_gpu(),
                     placement_backend_name(expert_kind), expert_gpu_);
    }
    const bool dual_owner_streams =
        expert_backend_ && !accelerator_only_stream;
    auto fail_streaming = [&]() {
        routed_output_provider_.reset();
        dual_stream_executor_.destroy();
        stream_engine_.destroy();
        secondary_stream_engine_.destroy();
        stream_owner_policy_ = MoeStreamDualOwnerPolicy{};
        stream_placement_ = MoeHybridPlacement{};
        release_expert_backend();
        return false;
    };

    const char * moe_core_raw =
        std::getenv("DFLASH_KIMI_MOE_CORE_OFFLOAD");
    const bool moe_core_requested = cpu_core && expert_backend_ &&
        moe_core_raw && *moe_core_raw &&
        std::strcmp(moe_core_raw, "0") != 0;
    if (moe_core_requested) {
        if (!init_kimi_k3_moe_core_offload(
                expert_backend_, weights_, moe_core_offload_, &error)) {
            std::fprintf(stderr,
                "[kimi-k3] accelerator MoE-core offload failed: %s\n",
                error.c_str());
            return fail_streaming();
        }
    }

    size_t routed_pool_bytes = 0;
    for (const LayerExpertRegions & regions :
         weights_.streamed_layer_regions) {
        size_t bytes_per_expert = 0;
        bool component_overflow = false;
        for (size_t component : {
                 regions.expert_bytes_gate, regions.expert_bytes_up,
                 regions.expert_bytes_down, regions.expert_bytes_gate_up}) {
            if (component >
                std::numeric_limits<size_t>::max() - bytes_per_expert) {
                component_overflow = true;
                break;
            }
            bytes_per_expert += component;
        }
        if (component_overflow) {
            routed_pool_bytes = std::numeric_limits<size_t>::max();
            break;
        }
        if (bytes_per_expert >
            (std::numeric_limits<size_t>::max() - routed_pool_bytes) /
                static_cast<size_t>(weights_.n_expert)) {
            routed_pool_bytes = std::numeric_limits<size_t>::max();
            break;
        }
        routed_pool_bytes +=
            bytes_per_expert * static_cast<size_t>(weights_.n_expert);
    }
    auto stream_config_for = [&](ggml_backend_t owner_backend, int gpu,
                                 PlacementBackend backend_kind,
                                 const char * owner_name) {
        MoeStreamConfig stream_config = MoeStreamConfig::from_env();
        if (std::getenv("DFLASH_MOE_NVME_DEVICE_CACHE_MB")) {
            stream_config.device_cache_bytes =
                std::min(stream_config.device_cache_bytes, routed_pool_bytes);
            return stream_config;
        }
        size_t free_bytes = 0;
        size_t total_bytes = 0;
        if (ggml_backend_dev_t device =
                ggml_backend_get_device(owner_backend)) {
            ggml_backend_dev_memory(device, &free_bytes, &total_bytes);
        }
        // ggml's generic UMA query intentionally reports available system RAM
        // so model loaders can use managed memory. The streamed cache needs one
        // contiguous native allocation, so cap it using the runtime's native
        // pool even when the owner is provided by a dynamically loaded peer.
        size_t allocation_free = free_bytes;
        size_t allocation_total = total_bytes;
        backend_native_memory(
            owner_backend, &allocation_free, &allocation_total);
        const size_t gib = 1024ULL * 1024ULL * 1024ULL;
        const size_t system_reserve =
            std::max<size_t>(2 * gib, total_bytes / 20);
        const size_t allocation_reserve =
            std::max<size_t>(2 * gib, allocation_total / 20);
        const size_t system_budget = free_bytes > system_reserve
            ? free_bytes - system_reserve : 0;
        const size_t allocation_budget = allocation_free > allocation_reserve
            ? allocation_free - allocation_reserve : 0;
        stream_config.device_cache_bytes = std::min(
            {system_budget, allocation_budget, routed_pool_bytes});
        std::fprintf(stderr,
            "[kimi-k3] %s streamed cache: device=%s:%d free=%.2f GiB "
            "alloc-free=%.2f GiB reserve=%.2f/%.2f GiB pool=%.2f GiB "
            "cache=%.2f GiB\n",
            owner_name, placement_backend_name(backend_kind), gpu,
            static_cast<double>(free_bytes) / gib,
            static_cast<double>(allocation_free) / gib,
            static_cast<double>(system_reserve) / gib,
            static_cast<double>(allocation_reserve) / gib,
            static_cast<double>(routed_pool_bytes) / gib,
            static_cast<double>(stream_config.device_cache_bytes) / gib);
        return stream_config;
    };

    stream_owner_policy_ = MoeStreamDualOwnerPolicy::from_env();
    stream_placement_ = MoeHybridPlacement{};
    const char * placement_path = std::getenv("DFLASH_MOE_PLACEMENT");
    if (placement_path && *placement_path) {
        if (!MoeHybridPlacement::load_json(
                placement_path, stream_placement_, &error) ||
            !stream_placement_.matches(
                static_cast<int>(weights_.streamed_layer_regions.size()),
                weights_.n_expert, weights_.n_expert_used)) {
            std::fprintf(stderr,
                         "[kimi-k3] invalid streamed-expert placement %s: %s\n",
                         placement_path,
                         error.empty() ? "model shape mismatch" : error.c_str());
            return fail_streaming();
        }
        if (expert_backend_) {
            stream_owner_policy_.primary_placement = &stream_placement_;
        }
    }

    ScopedFileDescriptors descriptors;
    std::vector<MoeNvmeSource> model_sources;
    model_sources.reserve(weights_.shard_paths.size());
    for (const std::string & shard : weights_.shard_paths) {
        MoeNvmeSource source;
        if (!open_nvme_source(shard, descriptors, source, error)) {
            std::fprintf(stderr,
                         "[kimi-k3] %s\n", error.c_str());
            return fail_streaming();
        }
        model_sources.push_back(source);
    }

    std::vector<uint32_t> expert_counts(
        weights_.streamed_layer_regions.size(),
        static_cast<uint32_t>(weights_.n_expert));
    const uint64_t source_layout_hash = moe_expert_source_layout_hash(
        model_sources, weights_.streamed_layer_regions, expert_counts);
    std::vector<MoeNvmeSource> package_sources;
    MoeExpertPackageManifest package_manifest;
    const std::vector<MoeNvmeSource> * active_sources = &model_sources;
    const std::vector<LayerExpertRegions> * active_regions =
        &weights_.streamed_layer_regions;
    size_t max_streamed_expert_bytes = weights_.max_streamed_expert_bytes;
    const char * package_path = std::getenv("DFLASH_MOE_EXPERT_PACKAGE");
    const char * package_build =
        std::getenv("DFLASH_MOE_EXPERT_PACKAGE_BUILD");
    const bool build_requested =
        package_build && *package_build &&
        std::strcmp(package_build, "0") != 0;
    const bool force_build =
        build_requested && std::strcmp(package_build, "force") == 0;
    if (build_requested && (!package_path || !*package_path)) {
        std::fprintf(stderr,
            "[kimi-k3] DFLASH_MOE_EXPERT_PACKAGE_BUILD requires "
            "DFLASH_MOE_EXPERT_PACKAGE=<output>\n");
        return fail_streaming();
    }
    if (package_path && *package_path) {
        MoeNvmeSource package_source;
        auto load_matching_package = [&]() -> bool {
            MoeNvmeSource candidate;
            MoeExpertPackageManifest candidate_manifest;
            if (!open_nvme_source(
                    package_path, descriptors, candidate, error)) {
                return false;
            }
            if (!read_moe_expert_package(
                    candidate, candidate_manifest, &error)) {
                descriptors.close(candidate.fd);
                return false;
            }
            const bool shape_matches =
                candidate_manifest.layer_regions.size() ==
                    weights_.streamed_layer_regions.size() &&
                candidate_manifest.expert_counts == expert_counts;
            if (!shape_matches ||
                candidate_manifest.source_layout_hash != source_layout_hash) {
                descriptors.close(candidate.fd);
                error = "expert package does not match this model, "
                        "checkpoint, or shard layout";
                return false;
            }
            package_source = candidate;
            package_manifest = std::move(candidate_manifest);
            return true;
        };

        bool package_loaded = !force_build && load_matching_package();
        if (!package_loaded && !build_requested) {
            std::fprintf(stderr,
                         "[kimi-k3] invalid expert package %s: %s\n",
                         package_path, error.c_str());
            return fail_streaming();
        }
        if (!package_loaded) {
            if (!force_build && file_path_exists(package_path)) {
                std::fprintf(stderr,
                    "[kimi-k3] rebuilding invalid expert package %s: %s\n",
                    package_path, error.c_str());
            }
            error.clear();
            const std::string temporary_path =
                package_temporary_path(package_path);
            int output_fd = -1;
            if (!create_package_output(
                    temporary_path, descriptors, output_fd, error)) {
                std::fprintf(stderr,
                             "[kimi-k3] expert package build failed: %s\n",
                             error.c_str());
                return fail_streaming();
            }
            MoeExpertPackageOptions options;
            options.progress = [](size_t completed, size_t total, void *) {
                std::fprintf(stderr,
                    "[kimi-k3] expert package progress %zu/%zu layers\n",
                    completed, total);
                std::fflush(stderr);
            };
            std::fprintf(stderr,
                "[kimi-k3] compiling expert-major package=%s "
                "(exact weights, one aligned record/expert)\n",
                package_path);
            if (!write_moe_expert_package(
                    output_fd, model_sources,
                    weights_.streamed_layer_regions, expert_counts,
                    options, &package_manifest, &error)) {
                descriptors.close(output_fd);
                (void) std::remove(temporary_path.c_str());
                std::fprintf(stderr,
                             "[kimi-k3] expert package build failed: %s\n",
                             error.c_str());
                return fail_streaming();
            }
            descriptors.close(output_fd);
            if (!publish_package(
                    temporary_path, package_path, error)) {
                (void) std::remove(temporary_path.c_str());
                std::fprintf(stderr,
                             "[kimi-k3] expert package build failed: %s\n",
                             error.c_str());
                return fail_streaming();
            }
            package_loaded = load_matching_package();
            if (!package_loaded) {
                std::fprintf(stderr,
                    "[kimi-k3] published expert package is invalid %s: %s\n",
                    package_path, error.c_str());
                return fail_streaming();
            }
        }
        package_sources.push_back(package_source);
        active_sources = &package_sources;
        active_regions = &package_manifest.layer_regions;
        max_streamed_expert_bytes = std::max(
            max_streamed_expert_bytes, package_manifest.max_record_bytes);
        std::fprintf(stderr,
            "[kimi-k3] using expert-major package=%s layers=%zu "
            "record<=%.2f MiB (one read/expert)\n",
            package_path, package_manifest.layer_regions.size(),
            static_cast<double>(package_manifest.max_record_bytes) /
                (1024.0 * 1024.0));
    }

    ggml_backend_t stream_backend = accelerator_only_stream
        ? expert_backend_ : backend_;
    const PlacementBackend stream_kind = accelerator_only_stream
        ? expert_backend_kind_ : primary_kind;
    const int stream_gpu = accelerator_only_stream
        ? expert_gpu_ : cfg_.device.primary_gpu();
    const MoeStreamConfig primary_config = stream_config_for(
        stream_backend, stream_gpu, stream_kind,
        accelerator_only_stream ? "accelerator" : "primary");
    if (!stream_engine_.init(
            stream_backend, max_streamed_expert_bytes,
            primary_config, &error)) {
        std::fprintf(stderr,
                     "[kimi-k3] primary stream engine initialization failed: %s\n",
                     error.c_str());
        return fail_streaming();
    }
    if (dual_owner_streams) {
        const MoeStreamConfig secondary_config = stream_config_for(
            expert_backend_, expert_gpu_, expert_backend_kind_, "secondary");
        if (!secondary_stream_engine_.init(
                expert_backend_, max_streamed_expert_bytes,
                secondary_config, &error)) {
            std::fprintf(stderr,
                         "[kimi-k3] secondary stream engine initialization failed: %s\n",
                         error.c_str());
            return fail_streaming();
        }
    }

    const bool primary_bound = stream_engine_.bind_sources(
        *active_sources, *active_regions, &error);
    bool secondary_bound = true;
    std::string secondary_error;
    if (primary_bound && dual_owner_streams) {
        secondary_bound = secondary_stream_engine_.bind_sources(
            *active_sources, *active_regions, &secondary_error);
    }
    if (!primary_bound || !secondary_bound) {
        std::fprintf(stderr,
                     "[kimi-k3] stream source binding failed: %s\n",
                     primary_bound ? secondary_error.c_str() : error.c_str());
        return fail_streaming();
    }

    std::vector<MoeStreamExpertSpec> layer_specs;
    std::vector<uint64_t> layer_expert_bytes;
    layer_specs.reserve(weights_.streamed_layer_regions.size());
    layer_expert_bytes.reserve(weights_.streamed_layer_regions.size());
    for (size_t local_layer = 0;
         local_layer < weights_.streamed_layer_regions.size(); ++local_layer) {
        const size_t model_layer =
            (size_t) weights_.n_dense_lead + local_layer;
        if (model_layer >= weights_.layers.size()) {
            std::fprintf(stderr,
                         "[kimi-k3] streamed layer metadata is out of range\n");
            return fail_streaming();
        }
        const KimiK3Layer & layer = weights_.layers[model_layer];
        if (!layer.ffn_gate_exps || !layer.ffn_up_exps ||
            !layer.ffn_down_exps) {
            std::fprintf(stderr,
                         "[kimi-k3] streamed layer is missing expert types\n");
            return fail_streaming();
        }
        layer_specs.push_back(make_kimi_k3_stream_spec(weights_, layer));
        const LayerExpertRegions & regions =
            weights_.streamed_layer_regions[local_layer];
        layer_expert_bytes.push_back(
            (uint64_t) regions.expert_bytes_gate +
            (uint64_t) regions.expert_bytes_up +
            (uint64_t) regions.expert_bytes_down +
            (uint64_t) regions.expert_bytes_gate_up);
    }

    const char * hotness_path = std::getenv("DFLASH_MOE_HOTNESS_CSV");
    if (!hotness_path || !*hotness_path) {
        hotness_path = std::getenv("DFLASH_DS4_HOTNESS_CSV");
    }
    MoeHybridRoutingStats routing_profile;
    const bool have_routing_profile = hotness_path && *hotness_path;
    if (have_routing_profile &&
        (!MoeHybridRoutingStats::load_csv(
             hotness_path, routing_profile, &error) ||
         !routing_profile.matches(
             static_cast<int>(weights_.streamed_layer_regions.size()),
             weights_.n_expert, weights_.n_expert_used))) {
        std::fprintf(stderr,
                     "[kimi-k3] invalid expert hotness profile %s: %s\n",
                     hotness_path,
                     error.empty() ? "model shape mismatch" : error.c_str());
        return fail_streaming();
    }

    auto warm_owner = [&](MoeHybridStreamEngine & engine,
                          MoeStreamCacheOwner owner_kind,
                          const char * owner_name) -> bool {
        const int slots = engine.device_slot_count();
        // Preserve a meaningful adaptive region when the profile drifts. Tiny
        // caches still retain the two slots needed by the miss pipeline.
        const int reserve_slots = std::max(2, slots / 4);
        const size_t max_entries = slots > reserve_slots
            ? (size_t) (slots - reserve_slots) : 0;
        if (max_entries == 0) return true;

        std::vector<MoeStreamCacheWarmEntry> plan;
        if (have_routing_profile) {
            MoeStreamCachePlanConfig plan_config;
            plan_config.max_entries = max_entries;
            plan_config.owner = owner_kind;
            const MoeStreamDualOwnerPolicy * policy =
                owner_kind == MoeStreamCacheOwner::All
                    ? nullptr : &stream_owner_policy_;
            if (!build_moe_stream_cache_plan(
                    routing_profile, layer_expert_bytes,
                    plan_config, policy, plan, &error)) {
                return false;
            }
        } else if (!stream_placement_.empty() &&
                   owner_kind != MoeStreamCacheOwner::Secondary) {
            for (int layer = 0; layer < stream_placement_.n_layer; ++layer) {
                const auto & ids =
                    stream_placement_.hot_expert_ids[(size_t) layer];
                for (size_t rank = 0; rank < ids.size(); ++rank) {
                    plan.push_back({
                        (int32_t) layer, ids[rank],
                        (uint64_t) std::max<size_t>(1, ids.size() - rank),
                        layer_expert_bytes[(size_t) layer]});
                }
            }
        }
        if (plan.empty()) return true;

        MoeStreamCacheWarmStats warm_stats;
        if (!engine.warm_and_pin_device_cache(
                layer_specs, plan, reserve_slots, &warm_stats, &error)) {
            return false;
        }
        std::fprintf(stderr,
            "[kimi-k3] %s profile-warm cache: requested=%zu pinned=%zu "
            "resident=%zu capacity-drops=%zu source=%s\n",
            owner_name, warm_stats.requested, warm_stats.admitted,
            warm_stats.already_resident, warm_stats.capacity_drops,
            have_routing_profile ? hotness_path : placement_path);
        return true;
    };

    if (!warm_owner(
            stream_engine_,
            dual_owner_streams ? MoeStreamCacheOwner::Primary
                               : MoeStreamCacheOwner::All,
            "primary") ||
        (dual_owner_streams &&
         !warm_owner(secondary_stream_engine_,
                     MoeStreamCacheOwner::Secondary,
                     "secondary"))) {
        std::fprintf(stderr,
                     "[kimi-k3] profile-guided cache warmup failed: %s\n",
                     error.c_str());
        return fail_streaming();
    }

    if (dual_owner_streams && !dual_stream_executor_.init(
            stream_engine_, secondary_stream_engine_, &error)) {
        std::fprintf(stderr,
                     "[kimi-k3] dual-owner executor initialization failed: %s\n",
                     error.c_str());
        return fail_streaming();
    }
    if (const char * stats_path = std::getenv("DFLASH_MOE_ROUTE_STATS_OUT")) {
        if (*stats_path) {
            routing_stats_ = std::make_shared<MoeHybridRoutingStats>();
            if (!routing_stats_->init(
                    static_cast<int>(weights_.streamed_layer_regions.size()),
                    weights_.n_expert, weights_.n_expert_used)) {
                std::fprintf(stderr,
                    "[kimi-k3] failed to initialize routed-expert statistics\n");
                return fail_streaming();
            }
            routing_stats_out_path_ = stats_path;
            std::fprintf(stderr,
                "[kimi-k3] recording native route counts to %s\n",
                routing_stats_out_path_.c_str());
        }
    }
    if (!create_kimi_k3_progressive_provider_from_env(
            stream_engine_.compute_backend(), backend_,
            routed_output_provider_, &error)) {
        std::fprintf(stderr,
                     "[kimi-k3] H16 routed provider initialization failed: %s\n",
                     error.c_str());
        return fail_streaming();
    }
    if (dual_owner_streams) {
        std::fprintf(stderr,
            "[kimi-k3] routed experts dual-owner: shards=%zu layers=%zu "
            "primary=%s:%d/%s/%.2fGiB secondary=%s:%d/%s/%.2fGiB "
            "primary_share=%d/1000 placement=%s\n",
            weights_.shard_paths.size(),
            weights_.streamed_layer_regions.size(),
            placement_backend_name(primary_kind), cfg_.device.primary_gpu(),
            stream_engine_.io_backend_name(),
            static_cast<double>(stream_engine_.device_cache_bytes()) /
                (1024.0 * 1024.0 * 1024.0),
            placement_backend_name(expert_backend_kind_), expert_gpu_,
            secondary_stream_engine_.io_backend_name(),
            static_cast<double>(secondary_stream_engine_.device_cache_bytes()) /
                (1024.0 * 1024.0 * 1024.0),
            stream_owner_policy_.primary_share_per_mille,
            stream_owner_policy_.primary_placement ? "profile" : "hash");
    } else {
        std::fprintf(stderr,
            "[kimi-k3] routed experts file-backed: shards=%zu layers=%zu "
            "io=%s compute=%s:%d cache=%.2f GiB\n",
            weights_.shard_paths.size(),
            weights_.streamed_layer_regions.size(),
            stream_engine_.io_backend_name(),
            placement_backend_name(stream_kind), stream_gpu,
            static_cast<double>(stream_engine_.device_cache_bytes()) /
                (1024.0 * 1024.0 * 1024.0));
    }
    return true;
}

bool KimiK3Backend::init_draft() {
    if (!cfg_.draft_path || !*cfg_.draft_path) return true;
    if (draft_backend_ || draft_weights_.ctx) return true;

    draft_backend_ = ggml_backend_cuda_init(std::max(0, cfg_.draft_gpu));
    if (!draft_backend_) {
        std::fprintf(stderr,
                     "[kimi-k3-dspark] draft backend init failed for device %d\n",
                     cfg_.draft_gpu);
        return false;
    }
    if (!load_draft_gguf(cfg_.draft_path, draft_backend_, draft_weights_)) {
        std::fprintf(stderr,
                     "[kimi-k3-dspark] draft load failed: %s\n",
                     dflash27b_last_error());
        free_drafter();
        return false;
    }
    if (const char * raw = std::getenv("DFLASH_KIMI_DRAFT_MAX_BLOCK")) {
        const int requested = std::atoi(raw);
        if (requested >= 2 && requested < draft_weights_.block_size) {
            std::fprintf(stderr,
                "[kimi-k3-dspark] limiting draft block %d -> %d\n",
                draft_weights_.block_size, requested);
            draft_weights_.block_size = requested;
        }
    }

    bool compatible =
        draft_weights_.n_embd == weights_.n_embd &&
        draft_weights_.block_size > 1 &&
        draft_weights_.block_size <= kMaxDsparkBlockSize &&
        draft_weights_.n_target_layers > 0 &&
        draft_weights_.n_target_layers ==
            static_cast<int>(draft_weights_.capture_layer_ids.size()) &&
        draft_weights_.mask_token_id >= 0 &&
        draft_weights_.mask_token_id < weights_.n_vocab;
    for (int layer : draft_weights_.capture_layer_ids) {
        compatible = compatible && layer >= 0 && layer < weights_.n_layer;
    }
    if (draft_weights_.dspark.enabled) {
        compatible = compatible &&
            draft_weights_.dspark.vocab_size == weights_.n_vocab;
    }
    if (!compatible) {
        std::fprintf(stderr,
            "[kimi-k3-dspark] incompatible checkpoint: target "
            "hidden/vocab/layers=%d/%d/%d, draft hidden/block/captures/"
            "mask/vocab=%d/%d/%zu/%d/%d\n",
            weights_.n_embd, weights_.n_vocab, weights_.n_layer,
            draft_weights_.n_embd, draft_weights_.block_size,
            draft_weights_.capture_layer_ids.size(),
            draft_weights_.mask_token_id,
            draft_weights_.dspark.vocab_size);
        free_drafter();
        return false;
    }

    const int ring_cap = std::min(
        std::max(1, cfg_.device.max_ctx),
        std::max(2048, cfg_.draft_ctx_max));
    if (!draft_feature_mirror_init(
            feature_ring_, draft_backend_, std::max(0, cfg_.draft_gpu),
            cfg_.device.primary_gpu(), ring_cap,
            draft_weights_.n_target_layers, weights_.n_embd)) {
        std::fprintf(stderr,
                     "[kimi-k3-dspark] feature-ring allocation failed\n");
        free_drafter();
        return false;
    }
    std::fprintf(stderr,
        "[kimi-k3-dspark] shared DFlash runtime enabled: block=%d "
        "captures=%zu ring=%d draft_gpu=%d target_gpu=%d dspark=%d\n",
        draft_weights_.block_size,
        draft_weights_.capture_layer_ids.size(), ring_cap,
        cfg_.draft_gpu, cfg_.device.primary_gpu(),
        draft_weights_.dspark.enabled ? 1 : 0);
    return true;
}

bool KimiK3Backend::init() {
    if (!cfg_.model_path) {
        std::fprintf(stderr, "[kimi-k3] model path is null\n");
        return false;
    }
    std::string backend_error;
    if (!parse_optional_binary_environment(
            "DFLASH_KIMI_P58_EXACT_MULTIROW", p58_exact_multirow_,
            backend_error)) {
        std::fprintf(stderr, "[kimi-k3] %s\n", backend_error.c_str());
        return false;
    }
    if (!parse_kimi_k3_prefill_chunk(
            std::getenv("DFLASH_KIMI_PREFILL_CHUNK"), prefill_chunk_)) {
        std::fprintf(stderr,
            "[kimi-k3] DFLASH_KIMI_PREFILL_CHUNK must be 1, 2, 4, or 8\n");
        return false;
    }
    if (!kimi_k3_p58_configuration_valid(
            prefill_chunk_, p58_exact_multirow_)) {
        std::fprintf(stderr,
            "[kimi-k3] width-eight prefill requires both "
            "DFLASH_KIMI_PREFILL_CHUNK=8 and "
            "DFLASH_KIMI_P58_EXACT_MULTIROW=1\n");
        return false;
    }
    if (!parse_optional_binary_environment(
            "DFLASH_KIMI_P56_PREFILL_CENSUS", prefill_census_,
            backend_error)) {
        std::fprintf(stderr, "[kimi-k3] %s\n", backend_error.c_str());
        return false;
    }
    if (prefill_chunk_ > 1) {
        if ((cfg_.draft_path && *cfg_.draft_path) ||
            (std::getenv("DFLASH_KIMI_H16_CANDIDATE_LOGITS_OUT") &&
             *std::getenv("DFLASH_KIMI_H16_CANDIDATE_LOGITS_OUT"))) {
            std::fprintf(stderr,
                "[kimi-k3] chunked prefill is incompatible with drafting "
                "and paired H16 execution\n");
            return false;
        }
        for (const char * name : {
                 "DFLASH_KIMI_P42_ORDERED_DEVICE_JOIN",
                 "DFLASH_KIMI_P45_ASYNC_COMPACT_QUEUE",
                 "DFLASH_KIMI_P46_PERSISTENT_ROUTED_PREP",
                 "DFLASH_KIMI_P52_PERSISTENT_ROUTED_JOIN",
                 "DFLASH_KIMI_P53_DEVICE_HIDDEN_CHAIN"}) {
            bool enabled = false;
            if (!parse_optional_binary_environment(
                    name, enabled, backend_error)) {
                std::fprintf(stderr, "[kimi-k3] %s\n", backend_error.c_str());
                return false;
            }
            if (enabled) {
                std::fprintf(stderr,
                    "[kimi-k3] chunked prefill is incompatible with %s\n",
                    name);
                return false;
            }
        }
    }
    if (p58_exact_multirow_) {
        if (cfg_.core_placement != KimiK3CorePlacement::Accelerator ||
            cfg_.device.primary_gpu() != 1) {
            std::fprintf(stderr,
                "[kimi-k3] P58 exact multirow is qualified only with the "
                "accelerator core on GPU1\n");
            return false;
        }
        if (cfg_.logits_trace_path && *cfg_.logits_trace_path) {
            std::fprintf(stderr,
                "[kimi-k3] P58 exact multirow does not support logits tracing\n");
            return false;
        }
        for (const char * name : {
                 "DFLASH_KIMI_DIVERGENCE_TRACE_OUT",
                 "DFLASH_KIMI_LAYER1_TRACE_OUT",
                 "DFLASH_KIMI_P20_IO_TRACE",
                 "DFLASH_KIMI_P28_ORACLE_TRACE",
                 "DFLASH_KIMI_P40_CACHE_TRACE",
                 "DFLASH_MOE_ROUTE_STATS_OUT"}) {
            const char * value = std::getenv(name);
            if (value && *value) {
                std::fprintf(stderr,
                    "[kimi-k3] P58 exact multirow is incompatible with %s\n",
                    name);
                return false;
            }
        }
        for (const char * name : {
                 "DFLASH_MOE_DUAL_STREAM_TRACE",
                 "DFLASH_KIMI_S0_SERIAL_CORE_ROWS",
                 "DFLASH_KIMI_S0_SERIAL_EXPERT_ROWS"}) {
            const char * value = std::getenv(name);
            if (value && *value && std::strcmp(value, "0") != 0) {
                std::fprintf(stderr,
                    "[kimi-k3] P58 exact multirow is incompatible with %s\n",
                    name);
                return false;
            }
        }
    }
    backend_ = init_kimi_k3_core_backend(
        cfg_.core_placement, cfg_.device.primary_gpu(), &backend_error);
    if (!backend_) {
        std::fprintf(stderr, "[kimi-k3] %s\n", backend_error.c_str());
        return false;
    }
    const bool stream_routed_experts =
        cfg_.moe_storage != MoeStoragePolicy::Resident;
    KimiK3LoadOptions load_options;
    load_options.stream_routed_experts = stream_routed_experts;
    load_options.mmap_resident_tensors =
        cfg_.core_placement == KimiK3CorePlacement::Cpu;
    if (!load_kimi_k3_gguf(
            cfg_.model_path, backend_, weights_, load_options)) {
        std::fprintf(stderr, "[kimi-k3] model load failed: %s\n",
                     dflash27b_last_error());
        return false;
    }
    if (!init_draft()) return false;
    const int max_ctx = std::max(1, cfg_.device.max_ctx);
    int max_verify_tokens = draft_weights_.ctx
        ? draft_weights_.max_chain_verify_tokens() : 0;
    max_verify_tokens = std::max(
        max_verify_tokens, std::max(0, cfg_.oracle_verify_tokens));
    if (prefill_chunk_ > 1) {
        max_verify_tokens = std::max(max_verify_tokens, prefill_chunk_);
    }
    if (const char * paired =
            std::getenv("DFLASH_KIMI_H16_CANDIDATE_LOGITS_OUT")) {
        if (*paired) max_verify_tokens = std::max(max_verify_tokens, 1);
    }
    if (!create_kimi_k3_cache(
            backend_, weights_, max_ctx, cache_, max_verify_tokens)) {
        std::fprintf(stderr, "[kimi-k3] cache allocation failed (max_ctx=%d)\n",
                     max_ctx);
        return false;
    }
    if (weights_.routed_experts_streamed && !init_streaming()) return false;
    if (p58_exact_multirow_ &&
        (!weights_.routed_experts_streamed ||
         !routed_output_provider_ ||
         !routed_output_provider_->prefill_service() ||
         !routed_output_provider_->prefill_service()->supports_width(8) ||
         routed_output_provider_->requires_device_output() ||
         dual_stream_executor_.is_ready() || moe_core_offload_.enabled())) {
        std::fprintf(stderr,
            "[kimi-k3] P58 exact multirow requires the host-output, "
            "sidecar-authoritative all-layer calibrated96 provider on one "
            "expert owner\n");
        return false;
    }
    std::fprintf(stderr,
        "[kimi-k3] native backend ready core=%s:%d (max_ctx=%d, "
        "experts=%s, prefill-chunk=%d, p58-exact-multirow=%d)\n",
        kimi_k3_core_placement_name(cfg_.core_placement),
        cfg_.device.primary_gpu(), max_ctx,
        !weights_.routed_experts_streamed ? "resident" :
            (cfg_.core_placement == KimiK3CorePlacement::Cpu
                ? "nvme-accelerator" :
                (expert_backend_ ? "nvme-dual-owner" : "nvme-single-owner")),
        prefill_chunk_, p58_exact_multirow_ ? 1 : 0);
    std::fflush(stderr);
    return true;
}

bool KimiK3Backend::benchmark_oracle_verify(
        const std::vector<int32_t> & prompt,
        const std::vector<int32_t> & oracle_tokens,
        KimiK3OracleVerifyResult & result,
        std::string * error) {
    result = KimiK3OracleVerifyResult{};
    result.width = static_cast<int>(oracle_tokens.size());
    const int width = result.width;
    const int base_pos = static_cast<int>(prompt.size());
    const auto fail = [&](const std::string & message) {
        if (error) *error = message;
        return false;
    };
    const bool p58_candidate = kimi_k3_p58_oracle_candidate(
        p58_exact_multirow_, oracle_tokens.size(), true);
    if (p58_candidate && cfg_.oracle_layer_diagnostics) {
        return fail(
            "P58 exact multirow oracle qualification does not support "
            "per-layer hidden capture; disable DFLASH_KIMI_S0_LAYER_CAPTURE");
    }
    if (width > 1 && routed_output_provider_ &&
        routed_output_provider_->requires_device_output()) {
        return fail(
            "S0 oracle multi-token verification is incompatible with a "
            "device-output routed provider");
    }
    if (!backend_ || !weights_.ctx || !cache_.ctx || prompt.empty() ||
        width <= 0 || width > cache_.max_verify_tokens ||
        base_pos + width > cache_.max_ctx) {
        return fail("S0 oracle verify received an invalid prompt/span or "
                    "insufficient verify-cache capacity");
    }
    for (int32_t token : prompt) {
        if (token < 0 || token >= weights_.n_vocab) {
            return fail("S0 prompt token is outside the vocabulary");
        }
    }
    for (int32_t token : oracle_tokens) {
        if (token < 0 || token >= weights_.n_vocab) {
            return fail("S0 oracle token is outside the vocabulary");
        }
    }

    std::vector<int> all_layer_ids(static_cast<size_t>(weights_.n_layer));
    std::iota(all_layer_ids.begin(), all_layer_ids.end(), 0);

    const auto forward = [&](const std::vector<int32_t> & tokens,
                             int position, bool capture_replay,
                             bool capture_layers,
                             KimiK3ForwardResult & forward_result) {
        KimiK3ForwardOptions options;
        options.capture_layer_ids =
            capture_layers && cfg_.oracle_layer_diagnostics
                ? &all_layer_ids : nullptr;
        options.capture_replay = capture_replay;
        options.exact_multirow_core = kimi_k3_p58_oracle_candidate(
            p58_exact_multirow_, tokens.size(), capture_replay);
        options.read_logits = true;
        options.read_argmax = true;
        options.routed_output_provider = routed_output_provider_.get();
        options.moe_core_offload = moe_core_offload_.enabled()
            ? &moe_core_offload_ : nullptr;
        const bool ok = kimi_k3_forward(
            backend_, weights_, cache_, tokens, position, options,
            forward_result, &stream_engine_,
            dual_stream_executor_.is_ready()
                ? &dual_stream_executor_ : nullptr,
            &stream_owner_policy_, routing_stats_.get());
        if (ok) maybe_release_kimi_mapped_pages(weights_);
        return ok;
    };
    const auto rebuild_prompt = [&]() {
        reset_kimi_k3_cache(cache_);
        reset_kimi_k3_moe_core_offload_state(moe_core_offload_);
        for (size_t index = 0; index < prompt.size(); ++index) {
            KimiK3ForwardResult ignored;
            if (!forward({prompt[index]}, static_cast<int>(index), false, false,
                         ignored)) {
                return false;
            }
        }
        return true;
    };
    const auto provider_stats = [&]() {
        return routed_output_provider_
            ? routed_output_provider_->runtime_stats()
            : KimiK3RoutedRuntimeStats{};
    };

    if (!rebuild_prompt()) {
        return fail(std::string("S0 sequential prompt rebuild failed: ") +
                    dflash27b_last_error());
    }
    std::vector<float> sequential_logits;
    std::vector<int32_t> sequential_argmax;
    const size_t capture_row_values = static_cast<size_t>(weights_.n_embd);
    std::vector<float> sequential_hidden;
    if (cfg_.oracle_layer_diagnostics) {
        sequential_hidden.resize(
            static_cast<size_t>(weights_.n_layer) * width *
            capture_row_values);
    }
    sequential_logits.reserve(
        static_cast<size_t>(width) * weights_.n_vocab);
    sequential_argmax.reserve(static_cast<size_t>(width));
    const KimiK3RoutedRuntimeStats sequential_stats_begin = provider_stats();
    const uint64_t sequential_read_start = process_storage_read_bytes();
    const auto sequential_start = std::chrono::steady_clock::now();
    for (int token = 0; token < width; ++token) {
        KimiK3ForwardResult row;
        if (!forward({oracle_tokens[static_cast<size_t>(token)]},
                     base_pos + token, false, true, row)) {
            return fail(std::string("S0 sequential oracle row failed: ") +
                        dflash27b_last_error());
        }
        sequential_logits.insert(
            sequential_logits.end(), row.logits.begin(), row.logits.end());
        sequential_argmax.push_back(row.argmax.front());
        if (cfg_.oracle_layer_diagnostics) {
            const size_t expected_capture =
                static_cast<size_t>(weights_.n_layer) * capture_row_values;
            if (row.captured_hidden.size() != expected_capture) {
                return fail("S0 sequential layer capture has the wrong shape");
            }
            for (int layer = 0; layer < weights_.n_layer; ++layer) {
                std::memcpy(
                    sequential_hidden.data() +
                        (static_cast<size_t>(layer) * width + token) *
                            capture_row_values,
                    row.captured_hidden.data() +
                        static_cast<size_t>(layer) * capture_row_values,
                    capture_row_values * sizeof(float));
            }
        }
    }
    result.sequential_seconds = std::chrono::duration<double>(
        std::chrono::steady_clock::now() - sequential_start).count();
    const uint64_t sequential_read_end = process_storage_read_bytes();
    const KimiK3RoutedRuntimeStats sequential_stats = routed_stats_delta(
        provider_stats(), sequential_stats_begin);
    result.sequential_storage_bytes = sequential_read_end >= sequential_read_start
        ? sequential_read_end - sequential_read_start : 0;
    result.sequential_logical_provider_bytes =
        sequential_stats.logical_provider_bytes;
    result.sequential_compact_attempted = sequential_stats.compact_attempted;
    result.sequential_compact_completed = sequential_stats.compact_completed;
    result.sequential_compact_fallbacks = sequential_stats.compact_fallbacks;
    result.sequential_compact_invalid = sequential_stats.compact_invalid;
    result.sequential_recurrent_hash = recurrent_cache_hash(cache_);
    result.sequential_mla_hash = mla_rows_hash(cache_, base_pos, width);
    result.sequential_conv_layer_hashes =
        recurrent_layer_hashes(cache_, true);
    result.sequential_ssm_layer_hashes =
        recurrent_layer_hashes(cache_, false);
    result.sequential_mla_layer_hashes =
        mla_layer_row_hashes(cache_, base_pos, width);

    if (!rebuild_prompt()) {
        return fail(std::string("S0 verify prompt rebuild failed: ") +
                    dflash27b_last_error());
    }
    if (!kimi_k3_replay_snapshot(backend_, cache_)) {
        return fail("S0 ReplaySSM snapshot failed");
    }
    const KimiK3RoutedRuntimeStats verify_stats_begin = provider_stats();
    const uint64_t verify_read_start = process_storage_read_bytes();
    const auto verify_start = std::chrono::steady_clock::now();
    KimiK3ForwardResult verified;
    if (!forward(oracle_tokens, base_pos, true, true, verified)) {
        (void) kimi_k3_replay_restore(backend_, cache_);
        return fail(std::string("S0 causal verify batch failed: ") +
                    dflash27b_last_error());
    }
    result.verify_seconds = std::chrono::duration<double>(
        std::chrono::steady_clock::now() - verify_start).count();
    const auto commit_start = std::chrono::steady_clock::now();
    if (!kimi_k3_replay_commit(
            backend_, weights_, cache_, base_pos, width)) {
        return fail("S0 ReplaySSM commit failed");
    }
    result.commit_seconds = std::chrono::duration<double>(
        std::chrono::steady_clock::now() - commit_start).count();
    const uint64_t verify_read_end = process_storage_read_bytes();
    const KimiK3RoutedRuntimeStats verify_stats = routed_stats_delta(
        provider_stats(), verify_stats_begin);
    result.verify_storage_bytes = verify_read_end >= verify_read_start
        ? verify_read_end - verify_read_start : 0;
    result.verify_logical_provider_bytes =
        verify_stats.logical_provider_bytes;
    result.verify_compact_attempted = verify_stats.compact_attempted;
    result.verify_compact_completed = verify_stats.compact_completed;
    result.verify_compact_fallbacks = verify_stats.compact_fallbacks;
    result.verify_compact_invalid = verify_stats.compact_invalid;
    result.verify_recurrent_hash = recurrent_cache_hash(cache_);
    result.verify_mla_hash = mla_rows_hash(cache_, base_pos, width);
    result.verify_conv_layer_hashes = recurrent_layer_hashes(cache_, true);
    result.verify_ssm_layer_hashes = recurrent_layer_hashes(cache_, false);
    result.verify_mla_layer_hashes =
        mla_layer_row_hashes(cache_, base_pos, width);

    result.logits_bit_equal = sequential_logits.size() == verified.logits.size() &&
        std::memcmp(sequential_logits.data(), verified.logits.data(),
                    sequential_logits.size() * sizeof(float)) == 0;
    result.argmax_bit_equal = sequential_argmax == verified.argmax;
    double reference_norm2 = 0.0;
    double error_norm2 = 0.0;
    if (sequential_logits.size() != verified.logits.size()) {
        return fail("S0 sequential and verify logits have different shapes");
    }
    for (size_t index = 0; index < sequential_logits.size(); ++index) {
        const double reference = sequential_logits[index];
        const double difference =
            static_cast<double>(verified.logits[index]) - reference;
        reference_norm2 += reference * reference;
        error_norm2 += difference * difference;
        result.logits_max_abs = std::max(
            result.logits_max_abs, std::abs(difference));
    }
    result.logits_rel_l2 = std::sqrt(
        error_norm2 / std::max(reference_norm2, 1.0e-300));
    result.recurrent_state_hash_equal =
        result.sequential_recurrent_hash == result.verify_recurrent_hash;
    result.mla_rows_hash_equal =
        result.sequential_mla_hash == result.verify_mla_hash;
    result.first_conv_state_mismatch_layer = first_hash_mismatch(
        result.sequential_conv_layer_hashes,
        result.verify_conv_layer_hashes);
    result.first_ssm_state_mismatch_layer = first_hash_mismatch(
        result.sequential_ssm_layer_hashes,
        result.verify_ssm_layer_hashes);
    result.first_mla_row_mismatch_layer = first_hash_mismatch(
        result.sequential_mla_layer_hashes,
        result.verify_mla_layer_hashes);

    if (p58_candidate) {
        const size_t layer_count = static_cast<size_t>(weights_.n_layer);
        const bool complete_hash_vectors =
            result.sequential_conv_layer_hashes.size() == layer_count &&
            result.verify_conv_layer_hashes.size() == layer_count &&
            result.sequential_ssm_layer_hashes.size() == layer_count &&
            result.verify_ssm_layer_hashes.size() == layer_count &&
            result.sequential_mla_layer_hashes.size() == layer_count &&
            result.verify_mla_layer_hashes.size() == layer_count;
        const bool complete_rows =
            sequential_logits.size() ==
                static_cast<size_t>(width) * weights_.n_vocab &&
            verified.logits.size() == sequential_logits.size() &&
            sequential_argmax.size() == static_cast<size_t>(width) &&
            verified.argmax.size() == sequential_argmax.size();
        const bool exact_provider_traffic =
            result.sequential_logical_provider_bytes ==
                result.verify_logical_provider_bytes &&
            result.sequential_compact_attempted ==
                result.sequential_compact_completed &&
            result.verify_compact_attempted ==
                result.verify_compact_completed &&
            result.sequential_compact_fallbacks == 0 &&
            result.verify_compact_fallbacks == 0 &&
            result.sequential_compact_invalid == 0 &&
            result.verify_compact_invalid == 0;
        if (!complete_rows || !complete_hash_vectors ||
            !result.logits_bit_equal || !result.argmax_bit_equal ||
            !result.recurrent_state_hash_equal ||
            result.first_conv_state_mismatch_layer >= 0 ||
            result.first_ssm_state_mismatch_layer >= 0 ||
            !result.mla_rows_hash_equal ||
            result.first_mla_row_mismatch_layer >= 0 ||
            !exact_provider_traffic) {
            return fail(
                "P58 exact multirow candidate failed mandatory full-logit, "
                "argmax, recurrent conv/SSM, MLA, or provider-traffic parity");
        }
    }

    if (cfg_.oracle_layer_diagnostics) {
        const size_t expected_verified_capture =
            static_cast<size_t>(weights_.n_layer) * width * capture_row_values;
        if (verified.captured_hidden.size() != expected_verified_capture) {
            return fail("S0 verify layer capture has the wrong shape");
        }
        for (int layer = 0;
             layer < weights_.n_layer && result.first_hidden_mismatch_layer < 0;
             ++layer) {
            for (int token = 0; token < width; ++token) {
                const size_t offset =
                    (static_cast<size_t>(layer) * width + token) *
                    capture_row_values;
                const float * reference = sequential_hidden.data() + offset;
                const float * candidate =
                    verified.captured_hidden.data() + offset;
                if (std::memcmp(reference, candidate,
                                capture_row_values * sizeof(float)) == 0) {
                    continue;
                }
                result.first_hidden_mismatch_layer = layer;
                result.first_hidden_mismatch_token = token;
                double reference_norm2 = 0.0;
                double error_norm2 = 0.0;
                for (size_t value = 0; value < capture_row_values; ++value) {
                    const double left = reference[value];
                    const double difference =
                        static_cast<double>(candidate[value]) - left;
                    reference_norm2 += left * left;
                    error_norm2 += difference * difference;
                    result.first_hidden_max_abs = std::max(
                        result.first_hidden_max_abs, std::abs(difference));
                }
                result.first_hidden_rel_l2 = std::sqrt(
                    error_norm2 / std::max(reference_norm2, 1.0e-300));
                break;
            }
        }
    }
    return true;
}

void KimiK3Backend::print_ready_banner() const {
    std::printf("[kimi-k3-daemon] ready (layers=%d hidden=%d experts=%d "
                "vocab=%d max_ctx=%d)\n",
                weights_.n_layer, weights_.n_embd, weights_.n_expert,
                weights_.n_vocab, cache_.max_ctx);
    std::fflush(stdout);
}

bool KimiK3Backend::park(ParkTarget target) {
    bool handled = false;
    if (park_target_includes_draft_model(target) && draft_backend_) {
        free_drafter();
        handled = true;
    }
    if (park_target_includes_target_model(target) && !parked_) {
        dflash_target_.reset();
        maybe_save_routing_stats();
        routed_output_provider_.reset();
        dual_stream_executor_.destroy();
        stream_engine_.destroy();
        secondary_stream_engine_.destroy();
        release_expert_backend();
        free_kimi_k3_weights(weights_);
        parked_ = true;
        handled = true;
    }
    return handled;
}

bool KimiK3Backend::unpark(ParkTarget target) {
    bool handled = false;
    if (park_target_includes_target_model(target) && parked_) {
        if (!load_kimi_k3_gguf(
                cfg_.model_path, backend_, weights_,
                cfg_.moe_storage != MoeStoragePolicy::Resident) ||
            (weights_.routed_experts_streamed && !init_streaming())) {
            return false;
        }
        parked_ = false;
        handled = true;
    }
    if (park_target_includes_draft_model(target) &&
        cfg_.draft_path && *cfg_.draft_path && !draft_backend_) {
        if (!init_draft()) return false;
        handled = true;
    }
    return handled;
}

int32_t KimiK3Backend::choose_token(const std::vector<float> & logits,
                                    const SamplerCfg & sampler,
                                    const std::vector<int32_t> & history) {
    if (sampler.needs_logit_processing()) {
        return sample_logits(logits.data(), weights_.n_vocab,
                             sampler, history, rng_);
    }
    return static_cast<int32_t>(std::distance(logits.begin(),
        std::max_element(logits.begin(), logits.end())));
}

bool KimiK3Backend::write_logits_trace(
        const GenerateRequest & request,
        const GenerateResult & result,
        const std::vector<float> & rows,
        const char * destination_path) const {
    const char * selected_path = destination_path
        ? destination_path : cfg_.logits_trace_path;
    if (!selected_path || !*selected_path) return true;
    if (weights_.n_vocab <= 0 ||
        rows.size() % static_cast<size_t>(weights_.n_vocab) != 0) {
        std::fprintf(stderr, "[kimi-k3] invalid logits trace shape\n");
        return false;
    }
    const std::string destination = selected_path;
    const std::string temporary = package_temporary_path(destination);
#if defined(_WIN32)
    const int fd = ::_open(
        temporary.c_str(), _O_CREAT | _O_TRUNC | _O_WRONLY | _O_BINARY,
        _S_IREAD | _S_IWRITE);
#else
    const int fd = ::open(
        temporary.c_str(), O_CREAT | O_TRUNC | O_WRONLY | O_CLOEXEC, 0644);
#endif
    if (fd < 0) {
        std::fprintf(stderr, "[kimi-k3] cannot create logits trace %s: %s\n",
                     temporary.c_str(), std::strerror(errno));
        return false;
    }
    auto write_all = [&](const void * data, size_t bytes) {
        const auto * source = static_cast<const uint8_t *>(data);
        while (bytes > 0) {
#if defined(_WIN32)
            const int chunk = static_cast<int>(std::min<size_t>(
                bytes, static_cast<size_t>(std::numeric_limits<int>::max())));
            const int written = ::_write(fd, source, chunk);
#else
            const ssize_t written = ::write(fd, source, bytes);
#endif
            if (written <= 0) return false;
            source += written;
            bytes -= static_cast<size_t>(written);
        }
        return true;
    };
    KimiLogitsTraceHeader header;
    std::memcpy(header.magic, kKimiLogitsTraceMagic, sizeof(header.magic));
    header.vocabulary = static_cast<uint32_t>(weights_.n_vocab);
    header.rows = rows.size() / static_cast<size_t>(weights_.n_vocab);
    header.prompt_tokens = request.prompt.size();
    header.generated_tokens = result.tokens.size();
    const bool write_ok = write_all(&header, sizeof(header)) &&
        write_all(rows.data(), rows.size() * sizeof(float));
#if defined(_WIN32)
    const bool close_ok = ::_close(fd) == 0;
#else
    const bool close_ok = ::fsync(fd) == 0 && ::close(fd) == 0;
#endif
    if (!write_ok || !close_ok || !sync_path(temporary)) {
        (void) std::remove(temporary.c_str());
        std::fprintf(stderr, "[kimi-k3] cannot finish logits trace %s\n",
                     destination.c_str());
        return false;
    }
    std::string publish_error;
    if (!publish_package(temporary, destination, publish_error)) {
        (void) std::remove(temporary.c_str());
        std::fprintf(stderr, "[kimi-k3] %s\n", publish_error.c_str());
        return false;
    }
    std::fprintf(stderr,
        "[kimi-k3] wrote logits trace rows=%llu vocab=%u path=%s\n",
        static_cast<unsigned long long>(header.rows), header.vocabulary,
        destination.c_str());
    return true;
}

bool KimiK3Backend::supports_dflash_spec_decode() const {
    return draft_backend_ && draft_weights_.ctx && feature_ring_.target_feat &&
        !(routed_output_provider_ &&
          routed_output_provider_->requires_device_output());
}

DFlashTarget * KimiK3Backend::dflash_target() {
    if (!supports_dflash_spec_decode()) return nullptr;
    if (!dflash_target_) {
        dflash_target_ = std::make_unique<KimiK3DFlashTarget>(
            weights_, cache_, backend_, feature_ring_,
            draft_weights_.capture_layer_ids,
            draft_weights_.mask_token_id, cfg_.fast_rollback,
            weights_.routed_experts_streamed ? &stream_engine_ : nullptr,
            dual_stream_executor_.is_ready()
                ? &dual_stream_executor_ : nullptr,
            &stream_owner_policy_, routing_stats_.get(),
            moe_core_offload_.enabled() ? &moe_core_offload_ : nullptr);
        // DFlash is disabled while tracing logits, but keep the provider wired
        // for completeness when a research run explicitly supplies a draft.
        dflash_target_->set_routed_output_provider(
            routed_output_provider_.get());
    }
    return dflash_target_.get();
}

void KimiK3Backend::free_drafter() {
    dflash_target_.reset();
    draft_feature_mirror_free(feature_ring_);
    if (draft_weights_.ctx) free_draft_weights(draft_weights_);
    if (draft_backend_) {
        ggml_backend_free(draft_backend_);
        draft_backend_ = nullptr;
    }
}

GenerateResult KimiK3Backend::generate_impl(const GenerateRequest & req,
                                            const DaemonIO & io) {
    GenerateResult result;
    DaemonIO out_io = io.with_token_callback(req.on_token);
    std::vector<int32_t> teacher_forced_tokens;
    const char * teacher_forced_raw =
        std::getenv("DFLASH_KIMI_TEACHER_FORCED_TOKENS");
    if (!parse_teacher_forced_tokens(
            teacher_forced_raw, teacher_forced_tokens) ||
        (!teacher_forced_tokens.empty() &&
         teacher_forced_tokens.size() < static_cast<size_t>(req.n_gen))) {
        result.fail(GenerateErrorCode::BackendSpecific,
                    "invalid or too-short Kimi teacher-forced token list");
        out_io.emit(-1);
        return result;
    }
    if (parked_) {
        result.fail(GenerateErrorCode::ModelParked);
        out_io.emit(-1);
        return result;
    }
    if (req.prompt.empty()) {
        result.fail(GenerateErrorCode::PrefillFailed, "empty prompt");
        out_io.emit(-1);
        return result;
    }
    if (req.prompt.size() + static_cast<size_t>(std::max(0, req.n_gen)) >
        static_cast<size_t>(cache_.max_ctx)) {
        result.fail(GenerateErrorCode::ContextOverflow,
                    "prompt plus generation exceeds Kimi-K3 cache");
        out_io.emit(-1);
        return result;
    }
    if (req.do_sample && req.sampler.seed != 0) rng_.seed(req.sampler.seed);

    reset_kimi_k3_cache(cache_);
    reset_kimi_k3_moe_core_offload_state(moe_core_offload_);
    std::vector<float> logits;
    std::vector<float> logits_trace;
    std::vector<float> paired_candidate_trace;
    const char * paired_candidate_path =
        std::getenv("DFLASH_KIMI_H16_CANDIDATE_LOGITS_OUT");
    const bool paired_interventions =
        paired_candidate_path && *paired_candidate_path;
    if (paired_interventions &&
        (!routed_output_provider_ || cache_.max_verify_tokens < 1)) {
        result.fail(GenerateErrorCode::BackendSpecific,
                    "paired H16 mode requires a routed provider and cache snapshot");
        out_io.emit(-1);
        return result;
    }
    const bool trace_logits =
        cfg_.logits_trace_path && *cfg_.logits_trace_path;
    if (trace_logits) {
        const size_t expected_rows = req.prompt.size() +
            static_cast<size_t>(std::max(0, req.n_gen - 1));
        logits_trace.reserve(
            expected_rows * static_cast<size_t>(weights_.n_vocab));
        if (paired_interventions) {
            paired_candidate_trace.reserve(
                expected_rows * static_cast<size_t>(weights_.n_vocab));
        }
    }
    const auto append_logits_trace = [&](const std::vector<float> & rows) {
        if (trace_logits) {
            logits_trace.insert(
                logits_trace.end(), rows.begin(), rows.end());
        }
    };
    auto * spec_target = static_cast<KimiK3DFlashTarget *>(dflash_target());
    std::string paired_failure;
    const auto forward_token = [&](int32_t token, int position) -> bool {
        if (!paired_interventions) {
            const bool ok = spec_target
                ? spec_target->forward_token(token, position, logits)
                : kimi_k3_step(
                    backend_, weights_, cache_, token, position, logits,
                    &stream_engine_,
                    dual_stream_executor_.is_ready()
                        ? &dual_stream_executor_ : nullptr,
                    &stream_owner_policy_, routing_stats_.get(),
                    routed_output_provider_.get(),
                    moe_core_offload_.enabled()
                        ? &moe_core_offload_ : nullptr);
            if (ok) maybe_release_kimi_mapped_pages(weights_);
            return ok;
        }
        if (spec_target) {
            paired_failure = "paired H16 mode is incompatible with DFlash";
            return false;
        }
        if (!kimi_k3_replay_snapshot(backend_, cache_)) {
            paired_failure = "cannot snapshot exact Kimi state";
            return false;
        }
        std::vector<float> candidate;
        if (!kimi_k3_step(
                backend_, weights_, cache_, token, position, candidate,
                &stream_engine_, nullptr, &stream_owner_policy_,
                routing_stats_.get(), routed_output_provider_.get(),
                moe_core_offload_.enabled()
                    ? &moe_core_offload_ : nullptr)) {
            paired_failure = dflash27b_last_error();
            return false;
        }
        if (!kimi_k3_replay_restore(backend_, cache_)) {
            paired_failure = "cannot restore exact Kimi state";
            return false;
        }
        if (!kimi_k3_step(
                backend_, weights_, cache_, token, position, logits,
                &stream_engine_,
                dual_stream_executor_.is_ready()
                    ? &dual_stream_executor_ : nullptr,
                &stream_owner_policy_, routing_stats_.get(), nullptr,
                moe_core_offload_.enabled()
                    ? &moe_core_offload_ : nullptr)) {
            paired_failure = dflash27b_last_error();
            return false;
        }
        maybe_release_kimi_mapped_pages(weights_);
        paired_candidate_trace.insert(
            paired_candidate_trace.end(), candidate.begin(), candidate.end());
        return true;
    };
    const auto write_paired_trace = [&]() {
        return !paired_interventions || write_logits_trace(
            req, result, paired_candidate_trace, paired_candidate_path);
    };
    const auto provider_stats = [&]() {
        return routed_output_provider_
            ? routed_output_provider_->runtime_stats()
            : KimiK3RoutedRuntimeStats{};
    };
    const KimiK3RoutedRuntimeStats prefill_stats_begin =
        prefill_census_ ? provider_stats() : KimiK3RoutedRuntimeStats{};
    const uint64_t prefill_process_read_begin =
        prefill_census_ ? process_storage_read_bytes() : 0;
    const auto prefill_begin = std::chrono::steady_clock::now();
    const KimiK3PrefillPolicy prefill_policy{
        prefill_chunk_, p58_exact_multirow_};
    KimiK3PrefillContext prefill_context{
        backend_, weights_, cache_, stream_engine_,
        dual_stream_executor_.is_ready() ? &dual_stream_executor_ : nullptr,
        &stream_owner_policy_, routing_stats_.get(),
        routed_output_provider_.get(),
        moe_core_offload_.enabled() ? &moe_core_offload_ : nullptr};
    KimiK3PrefillExecutionResult prefill_execution;
    std::string executor_failure;
    const KimiK3PrefillExecutor prefill_executor(prefill_context);
    if (!prefill_executor.run(
            req.prompt, prefill_policy, forward_token,
            append_logits_trace,
            [&]() { maybe_release_kimi_mapped_pages(weights_); },
            logits, prefill_execution,
            &executor_failure)) {
        result.fail(GenerateErrorCode::PrefillFailed,
                    !executor_failure.empty() ? executor_failure :
                        (!paired_failure.empty() ? paired_failure :
                            dflash27b_last_error()));
        out_io.emit(-1);
        return result;
    }
    const auto prefill_end = std::chrono::steady_clock::now();
    result.prefill_s = std::chrono::duration<double>(prefill_end - prefill_begin).count();
    if (prefill_census_) {
        print_prefill_census(
            "prefill", req.prompt.size(), prefill_execution.forward_calls,
            result.prefill_s,
            monotonic_delta(
                process_storage_read_bytes(), prefill_process_read_begin),
            routed_stats_delta(provider_stats(), prefill_stats_begin));
    }

    if (req.n_gen <= 0 || out_io.cancelled) {
        maybe_save_routing_stats();
        out_io.emit(-1);
        result.succeed();
        if (!write_logits_trace(req, result, logits_trace) ||
            !write_paired_trace()) {
            result.fail(GenerateErrorCode::DecodeFailed,
                        "failed to write Kimi logits trace");
        }
        return result;
    }

    const KimiK3RoutedRuntimeStats decode_stats_begin =
        prefill_census_ ? provider_stats() : KimiK3RoutedRuntimeStats{};
    const uint64_t decode_process_read_begin =
        prefill_census_ ? process_storage_read_bytes() : 0;
    const auto decode_begin = std::chrono::steady_clock::now();
    int draft_delay_tokens = 0;
    if (const char * raw = std::getenv("DFLASH_KIMI_DRAFT_DELAY_TOKENS")) {
        char * end = nullptr;
        errno = 0;
        const long value = std::strtol(raw, &end, 10);
        if (errno != 0 || end == raw || *end != '\0' || value < 0 ||
            value > std::numeric_limits<int>::max()) {
            result.fail(GenerateErrorCode::BackendSpecific,
                        "DFLASH_KIMI_DRAFT_DELAY_TOKENS must be a nonnegative integer");
            out_io.emit(-1);
            return result;
        }
        draft_delay_tokens = static_cast<int>(value);
    }
    const bool can_spec = spec_target && !trace_logits &&
        !req.force_ar_decode &&
        teacher_forced_tokens.empty() &&
        req.budget_hook.close_token_ids.empty() &&
        !req.sampler.needs_logit_processing();
    if (can_spec && draft_delay_tokens > 0) {
        const int ar_tokens = std::min(draft_delay_tokens, req.n_gen);
        bool hit_eos = false;
        for (int index = 0; index < ar_tokens; ++index) {
            const int32_t next = choose_token(
                logits, req.sampler, result.tokens);
            result.tokens.push_back(next);
            out_io.emit(next);
            if (out_io.cancelled || next == weights_.eos_token_id) {
                hit_eos = next == weights_.eos_token_id;
                break;
            }
            if (static_cast<int>(result.tokens.size()) < req.n_gen) {
                if (!forward_token(next, cache_.cur_pos)) {
                    result.fail(GenerateErrorCode::DecodeFailed,
                                dflash27b_last_error());
                    out_io.emit(-1);
                    return result;
                }
            }
        }
        const int remaining =
            req.n_gen - static_cast<int>(result.tokens.size());
        if (out_io.cancelled || hit_eos || remaining <= 0) {
            result.decode_s = std::chrono::duration<double>(
                std::chrono::steady_clock::now() - decode_begin).count();
            maybe_save_routing_stats();
            out_io.emit(-1);
            result.succeed();
            return result;
        }

        std::vector<int32_t> draft_context = req.prompt;
        draft_context.insert(
            draft_context.end(), result.tokens.begin(), result.tokens.end());
        const int32_t seed = choose_token(
            logits, req.sampler, result.tokens);
        if (seed == weights_.eos_token_id) {
            result.tokens.push_back(seed);
            out_io.emit(seed);
            result.decode_s = std::chrono::duration<double>(
                std::chrono::steady_clock::now() - decode_begin).count();
            maybe_save_routing_stats();
            out_io.emit(-1);
            result.succeed();
            return result;
        }
        DaemonIO spec_io = out_io.with_token_callback(
            [&](int32_t token) -> bool {
                result.tokens.push_back(token);
                return true;
            });
        std::fprintf(stderr,
            "[kimi-k3-dspark] delayed activation ar-tokens=%zu remaining=%d\n",
            result.tokens.size(), remaining);
        double accept_rate = 0.0;
        const bool ok = run_dflash_spec_decode(
            *spec_target, draft_weights_, draft_backend_, feature_ring_,
            draft_context, remaining, seed, /*out_path=*/nullptr,
            cfg_.draft_ctx_max, spec_io, /*remote_draft=*/nullptr,
            req.hint_tokens, /*base_pos=*/0, &accept_rate);
        result.decode_s = std::chrono::duration<double>(
            std::chrono::steady_clock::now() - decode_begin).count();
        result.accept_rate = static_cast<float>(accept_rate);
        result.spec_decode_ran = true;
        maybe_save_routing_stats();
        spec_io.emit(-1);
        if (!ok) {
            result.fail(GenerateErrorCode::DecodeFailed,
                        dflash27b_last_error());
            return result;
        }
        result.succeed();
        return result;
    }
    if (can_spec) {
        const int32_t seed = choose_token(logits, req.sampler, result.tokens);
        DaemonIO spec_io = out_io.with_token_callback(
            [&](int32_t token) -> bool {
                result.tokens.push_back(token);
                return true;
            });
        double accept_rate = 0.0;
        const bool ok = run_dflash_spec_decode(
            *spec_target, draft_weights_, draft_backend_, feature_ring_,
            req.prompt, req.n_gen, seed, /*out_path=*/nullptr,
            cfg_.draft_ctx_max, spec_io, /*remote_draft=*/nullptr,
            req.hint_tokens, /*base_pos=*/0, &accept_rate);
        result.decode_s = std::chrono::duration<double>(
            std::chrono::steady_clock::now() - decode_begin).count();
        result.accept_rate = static_cast<float>(accept_rate);
        result.spec_decode_ran = true;
        maybe_save_routing_stats();
        spec_io.emit(-1);
        if (!ok) {
            result.fail(GenerateErrorCode::DecodeFailed,
                        dflash27b_last_error());
            return result;
        }
        result.succeed();
        return result;
    }

    bool budget_close_started = false;
    size_t close_inject_pos = 0;
    for (int i = 0; i < req.n_gen; ++i) {
        int32_t next = choose_token(logits, req.sampler, result.tokens);
        if (!teacher_forced_tokens.empty()) {
            next = teacher_forced_tokens[static_cast<size_t>(i)];
        }

        // Preserve the shared Level-2 budget contract even before speculative
        // decode support lands for Kimi-K3.
        const auto & close_ids = req.budget_hook.close_token_ids;
        if (!close_ids.empty()) {
            if (budget_close_started && close_inject_pos < close_ids.size()) {
                next = close_ids[close_inject_pos++];
                result.budget_forced_close = true;
            } else if (!budget_close_started &&
                       req.n_gen - i <=
                           req.budget_hook.hard_limit_remaining) {
                budget_close_started = true;
                if (next == close_ids.front()) {
                    close_inject_pos = 1;
                } else {
                    next = close_ids.front();
                    close_inject_pos = 1;
                    result.budget_forced_close = true;
                }
            }
        }

        result.tokens.push_back(next);
        out_io.emit(next);
        if (out_io.cancelled || next == weights_.eos_token_id) break;
        if (i + 1 < req.n_gen) {
            if (!forward_token(next, cache_.cur_pos)) {
                result.fail(GenerateErrorCode::DecodeFailed,
                            paired_failure.empty()
                                ? dflash27b_last_error() : paired_failure);
                out_io.emit(-1);
                return result;
            }
            append_logits_trace(logits);
        }
    }
    const auto decode_end = std::chrono::steady_clock::now();
    result.decode_s = std::chrono::duration<double>(decode_end - decode_begin).count();
    if (prefill_census_) {
        const size_t transitions =
            cache_.cur_pos > static_cast<int>(req.prompt.size())
                ? static_cast<size_t>(cache_.cur_pos) - req.prompt.size() : 0;
        print_prefill_census(
            "decode", transitions, transitions, result.decode_s,
            monotonic_delta(
                process_storage_read_bytes(), decode_process_read_begin),
            routed_stats_delta(provider_stats(), decode_stats_begin));
    }
    maybe_save_routing_stats();
    out_io.emit(-1);
    result.succeed();
    if (!write_logits_trace(req, result, logits_trace) ||
        !write_paired_trace()) {
        result.fail(GenerateErrorCode::DecodeFailed,
                    "failed to write Kimi logits trace");
    }
    return result;
}

bool KimiK3Backend::snapshot_save(int slot) {
    (void)slot;
    return false;
}

void KimiK3Backend::snapshot_free(int slot) {
    (void)slot;
}

bool KimiK3Backend::snapshot_used(int slot) const {
    (void)slot;
    return false;
}

int KimiK3Backend::snapshot_cur_pos(int slot) const {
    (void)slot;
    return 0;
}

GenerateResult KimiK3Backend::restore_and_generate_impl(
        int slot, const GenerateRequest & req, const DaemonIO & io) {
    (void)slot;
    (void)req;
    GenerateResult result;
    result.fail(GenerateErrorCode::InvalidSnapshotSlot,
                "Kimi-K3 prefix snapshots are not implemented yet");
    io.emit(-1);
    return result;
}

bool KimiK3Backend::handle_compress(const std::string & line,
                                    const DaemonIO & io) {
    (void)line;
    (void)io;
    return false;
}

void KimiK3Backend::shutdown() {
    free_drafter();
    maybe_save_routing_stats();
    routed_output_provider_.reset();
    dual_stream_executor_.destroy();
    stream_engine_.destroy();
    secondary_stream_engine_.destroy();
    release_expert_backend();
    free_kimi_k3_cache(cache_);
    free_kimi_k3_weights(weights_);
    if (backend_) {
        ggml_backend_free(backend_);
        backend_ = nullptr;
    }
    routing_stats_.reset();
    routing_stats_out_path_.clear();
    parked_ = false;
}

void KimiK3Backend::maybe_save_routing_stats() {
    if (!routing_stats_ || routing_stats_out_path_.empty()) return;
    std::string error;
    if (!routing_stats_->save_csv(routing_stats_out_path_, &error)) {
        std::fprintf(stderr,
            "[kimi-k3] failed to save route statistics %s: %s\n",
            routing_stats_out_path_.c_str(), error.c_str());
    }
}

} // namespace dflash::common
