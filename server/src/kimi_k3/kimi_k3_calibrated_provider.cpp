#include "kimi_k3_calibrated_provider.h"
#include "kimi_k3_ordered_join.h"
#include "kimi_k3_sparse_scatter.h"
#include "common/cuda_graph_overrides.h"
#include "device_runtime.h"

#include "ggml-alloc.h"
#include "ggml-cuda.h"

#include <algorithm>
#include <array>
#include <atomic>
#include <cerrno>
#include <chrono>
#include <cmath>
#include <condition_variable>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <deque>
#include <fstream>
#include <functional>
#include <future>
#include <list>
#include <limits>
#include <memory>
#include <mutex>
#include <sstream>
#include <string>
#include <thread>
#include <unordered_map>
#include <utility>
#include <vector>

#if defined(_WIN32)
#include <fcntl.h>
#include <io.h>
#include <sys/stat.h>
#else
#include <fcntl.h>
#include <sys/resource.h>
#include <sys/stat.h>
#include <unistd.h>
#endif

namespace dflash::common {

KimiK3SparseUpload kimi_k3_sparse_upload_for_call(
        KimiK3SparseDeliveryPolicy delivery, bool has_prepacked_payload) {
    if (has_prepacked_payload) return KimiK3SparseUpload::PrepackedCompact;
    return delivery == KimiK3SparseDeliveryPolicy::CompactPageable
        ? KimiK3SparseUpload::PageableCompact
        : delivery >= KimiK3SparseDeliveryPolicy::CompactPinned
            ? KimiK3SparseUpload::PinnedCompact
            : KimiK3SparseUpload::SlabCopies;
}

uint16_t kimi_k3_selected_natural_slab_mask(
        const uint16_t * natural_by_rank,
        const uint8_t * selected_by_rank,
        int slab_count) {
    uint16_t mask = 0;
    if (!natural_by_rank || !selected_by_rank || slab_count <= 0) return mask;
    for (int rank = 0; rank < slab_count; ++rank) {
        if (selected_by_rank[rank] && natural_by_rank[rank] < 12) {
            mask = static_cast<uint16_t>(
                mask | (1u << natural_by_rank[rank]));
        }
    }
    return mask;
}

void kimi_k3_suppress_resident_slab_ranks(
        const uint16_t * natural_by_rank,
        uint16_t missing_mask,
        uint8_t * selected_by_rank,
        int slab_count) {
    if (!natural_by_rank || !selected_by_rank || slab_count <= 0) return;
    for (int rank = 0; rank < slab_count; ++rank) {
        if (natural_by_rank[rank] >= 12 ||
            (missing_mask & (1u << natural_by_rank[rank])) == 0) {
            selected_by_rank[rank] = 0;
        }
    }
}

bool kimi_k3_sparse_natural_mask(
        const uint16_t * naturals, int slab_count, uint16_t * mask) {
    if (mask) *mask = 0;
    if (!naturals || !mask || slab_count <= 0 || slab_count > 12) return false;
    uint16_t seen = 0;
    for (int index = 0; index < slab_count; ++index) {
        if (naturals[index] >= 12) return false;
        const uint16_t bit = static_cast<uint16_t>(1u << naturals[index]);
        if ((seen & bit) != 0) return false;
        seen = static_cast<uint16_t>(seen | bit);
    }
    *mask = seen;
    return true;
}

bool kimi_k3_compact_wire_layout(
        int slab_count, size_t gate_slab_bytes, size_t up_slab_bytes,
        size_t down_slab_bytes, KimiK3CompactWireLayout * layout) {
    if (!layout || slab_count <= 0 || slab_count > 12 ||
        gate_slab_bytes == 0 || up_slab_bytes == 0 ||
        down_slab_bytes == 0) {
        return false;
    }
    constexpr size_t metadata_bytes = 32;
    const size_t count = static_cast<size_t>(slab_count);
    const auto add_component = [count](
            size_t cursor, size_t slab_bytes, size_t & next) {
        if (slab_bytes >
            (std::numeric_limits<size_t>::max() - cursor) / count) {
            return false;
        }
        next = cursor + count * slab_bytes;
        return true;
    };
    KimiK3CompactWireLayout value;
    value.gate_offset = metadata_bytes;
    if (!add_component(
            value.gate_offset, gate_slab_bytes, value.up_offset) ||
        !add_component(
            value.up_offset, up_slab_bytes, value.down_offset) ||
        !add_component(
            value.down_offset, down_slab_bytes, value.total_bytes)) {
        return false;
    }
    *layout = value;
    return true;
}

namespace {

constexpr int kExpertCount = 896;
constexpr int kDimension = 3584;
constexpr int kSlabSize = 256;
constexpr int kSlabCount = 12;
constexpr int kNativeTopK = 16;
constexpr size_t kAlignment = 4096;
constexpr size_t kSlabComponentBytes = 179200;
constexpr size_t kSlabBytes = 3 * kSlabComponentBytes;
constexpr size_t kExpertRecordBytes = kSlabCount * kSlabBytes;

#if defined(DFLASH27B_BACKEND_CUDA) || defined(DFLASH27B_BACKEND_HIP)
class BackendDeviceScope {
public:
    ~BackendDeviceScope() {
        if (changed_) (void) cudaSetDevice(previous_);
    }

    bool enter(
            ggml_backend_t backend,
            std::string * err,
            bool restore_previous = true) {
        ggml_backend_dev_t device = backend
            ? ggml_backend_get_device(backend) : nullptr;
        ggml_backend_reg_t registry = device
            ? ggml_backend_dev_backend_reg(device) : nullptr;
        int ordinal = -1;
        if (registry) {
            const size_t count = ggml_backend_reg_dev_count(registry);
            for (size_t index = 0; index < count; ++index) {
                if (ggml_backend_reg_dev_get(registry, index) == device) {
                    ordinal = static_cast<int>(index);
                    break;
                }
            }
        }
        if (ordinal < 0 || cudaGetDevice(&previous_) != cudaSuccess ||
            (previous_ != ordinal && cudaSetDevice(ordinal) != cudaSuccess)) {
            if (err) *err = "failed to select sparse expert backend device";
            return false;
        }
        device_ = ordinal;
        changed_ = restore_previous && previous_ != ordinal;
        return true;
    }

    int device() const { return device_; }

private:
    int previous_ = -1;
    int device_ = -1;
    bool changed_ = false;
};
#endif
constexpr uint64_t kNaturalSidecarCacheDomain = 0x4b334e4154555241ULL;

uint64_t artifact_generation(const uint8_t * digest, size_t bytes) {
    uint64_t value = 1469598103934665603ULL;
    for (size_t i = 0; i < bytes; ++i) {
        value = (value ^ digest[i]) * 1099511628211ULL;
    }
    return value ? value : 1;
}

uint64_t slab_mask_count(uint16_t mask) {
    uint64_t count = 0;
    for (; mask != 0; mask = static_cast<uint16_t>(mask & (mask - 1))) {
        ++count;
    }
    return count;
}

enum class SparseWorkspace : uint8_t { HostRecomposed, TransientDevice, PersistentDevice };

bool parse_positive_int(const char * raw, int & value);
bool parse_binary_flag(const char * raw, bool & enabled) {
    enabled = raw && std::strcmp(raw, "1") == 0;
    return !raw || !*raw || enabled || std::strcmp(raw, "0") == 0;
}

// P20's first layer-wide direct-I/O implementation created and destroyed 16
// operating-system threads at every routed layer.  The reads themselves were
// honest, but that lifecycle cost is part of the storage critical path.  Keep
// one deliberately small pool for the lifetime of the opt-in provider.  Jobs
// never outlive their layer call: callers wait on every returned future before
// any captured payload or file descriptor can be released.
class P20DirectReadPool {
public:
    explicit P20DirectReadPool(size_t workers) {
        threads_.reserve(workers);
        for (size_t worker = 0; worker < workers; ++worker) {
            threads_.emplace_back([this]() { run(); });
        }
    }

    ~P20DirectReadPool() {
        {
            std::lock_guard<std::mutex> lock(mutex_);
            stopping_ = true;
        }
        ready_.notify_all();
        for (std::thread & thread : threads_) thread.join();
    }

    P20DirectReadPool(const P20DirectReadPool &) = delete;
    P20DirectReadPool & operator=(const P20DirectReadPool &) = delete;

    std::future<void> submit(std::function<void()> function) {
        std::packaged_task<void()> task(std::move(function));
        std::future<void> future = task.get_future();
        {
            std::lock_guard<std::mutex> lock(mutex_);
            tasks_.push_back(std::move(task));
        }
        ready_.notify_one();
        return future;
    }

private:
    void run() {
        for (;;) {
            std::packaged_task<void()> task;
            {
                std::unique_lock<std::mutex> lock(mutex_);
                ready_.wait(lock, [this]() {
                    return stopping_ || !tasks_.empty();
                });
                if (stopping_ && tasks_.empty()) return;
                task = std::move(tasks_.front());
                tasks_.pop_front();
            }
            task();
        }
    }

    std::mutex mutex_;
    std::condition_variable ready_;
    std::deque<std::packaged_task<void()>> tasks_;
    std::vector<std::thread> threads_;
    bool stopping_ = false;
};

enum class P30ReadKind : uint8_t {
    SidecarSlab = 0,
    SlabMean = 1,
    NativeMean = 2,
};

struct P30ReadKey {
    int model_layer = 0;
    P30ReadKind kind = P30ReadKind::SidecarSlab;
    uint64_t offset = 0;
    size_t bytes = 0;

    bool operator==(const P30ReadKey & other) const {
        return model_layer == other.model_layer && kind == other.kind &&
            offset == other.offset && bytes == other.bytes;
    }
};

struct P30ReadKeyHash {
    size_t operator()(const P30ReadKey & key) const {
        size_t value = std::hash<uint64_t>{}(key.offset);
        value ^= std::hash<size_t>{}(key.bytes) + 0x9e3779b9U +
            (value << 6) + (value >> 2);
        value ^= std::hash<int>{}(key.model_layer) + 0x9e3779b9U +
            (value << 6) + (value >> 2);
        value ^= static_cast<size_t>(key.kind) + 0x9e3779b9U +
            (value << 6) + (value >> 2);
        return value;
    }
};

// P30's first cache is deliberately simple and semantics-free: immutable
// aligned sidecar records and immutable calibrated means are copied into a
// bounded host LRU.  It never caches exact-fallback experts and never changes
// selection, accumulation, or GPU arithmetic.  A new independent prompt
// clears residency so suite measurements cannot borrow bytes across users.
class P30BoundedReadCache {
public:
    struct Stats {
        uint64_t hits = 0;
        uint64_t misses = 0;
        uint64_t hit_bytes = 0;
        uint64_t inserted_bytes = 0;
        uint64_t evicted_bytes = 0;
        uint64_t sequence_resets = 0;
        size_t resident_bytes = 0;
        size_t entries = 0;
    };

    void set_capacity(size_t bytes) { capacity_ = bytes; }
    bool enabled() const { return capacity_ > 0; }

    bool get(const P30ReadKey & key, void * destination) {
        if (!enabled()) return false;
        std::lock_guard<std::mutex> lock(mutex_);
        const auto found = entries_.find(key);
        if (found == entries_.end()) {
            ++misses_;
            return false;
        }
        if (found->second.bytes.size() != key.bytes) {
            ++misses_;
            return false;
        }
        std::memcpy(destination, found->second.bytes.data(), key.bytes);
        lru_.splice(lru_.begin(), lru_, found->second.position);
        ++hits_;
        hit_bytes_ += key.bytes;
        return true;
    }

    void put(const P30ReadKey & key, const void * source) {
        if (!enabled() || key.bytes == 0 || key.bytes > capacity_) return;
        std::lock_guard<std::mutex> lock(mutex_);
        const auto existing = entries_.find(key);
        if (existing != entries_.end()) {
            lru_.splice(lru_.begin(), lru_, existing->second.position);
            return;
        }
        while (!lru_.empty() && resident_bytes_ + key.bytes > capacity_) {
            const P30ReadKey victim = lru_.back();
            const auto found = entries_.find(victim);
            if (found != entries_.end()) {
                resident_bytes_ -= found->second.bytes.size();
                evicted_bytes_ += found->second.bytes.size();
                entries_.erase(found);
            }
            lru_.pop_back();
        }
        lru_.push_front(key);
        Entry entry;
        entry.bytes.resize(key.bytes);
        std::memcpy(entry.bytes.data(), source, key.bytes);
        entry.position = lru_.begin();
        resident_bytes_ += key.bytes;
        inserted_bytes_ += key.bytes;
        entries_.emplace(key, std::move(entry));
    }

    void reset_sequence() {
        if (!enabled()) return;
        std::lock_guard<std::mutex> lock(mutex_);
        entries_.clear();
        lru_.clear();
        resident_bytes_ = 0;
        ++sequence_resets_;
    }

    Stats stats() const {
        std::lock_guard<std::mutex> lock(mutex_);
        return {hits_, misses_, hit_bytes_, inserted_bytes_, evicted_bytes_,
                sequence_resets_, resident_bytes_, entries_.size()};
    }

    size_t capacity() const { return capacity_; }

private:
    struct Entry {
        std::vector<uint8_t> bytes;
        std::list<P30ReadKey>::iterator position;
    };

    size_t capacity_ = 0;
    mutable std::mutex mutex_;
    std::list<P30ReadKey> lru_;
    std::unordered_map<P30ReadKey, Entry, P30ReadKeyHash> entries_;
    size_t resident_bytes_ = 0;
    uint64_t hits_ = 0;
    uint64_t misses_ = 0;
    uint64_t hit_bytes_ = 0;
    uint64_t inserted_bytes_ = 0;
    uint64_t evicted_bytes_ = 0;
    uint64_t sequence_resets_ = 0;
};

struct SlabAuxHeaderV2 {
    char magic[8];
    uint32_t version;
    uint32_t model_layer;
    uint32_t expert_count;
    uint32_t dimension;
    uint32_t slab_size;
    uint32_t slab_count;
    uint32_t storage;
    uint32_t alignment;
    uint64_t order_offset;
    uint64_t order_bytes;
    uint64_t slab_means_offset;
    uint64_t slab_means_bytes;
    uint64_t slab_importance_offset;
    uint64_t slab_importance_bytes;
    uint64_t native_means_offset;
    uint64_t native_means_bytes;
    uint64_t native_importance_offset;
    uint64_t native_importance_bytes;
    uint64_t calibrated_experts_offset;
    uint64_t calibrated_experts_bytes;
    uint64_t calibration_hit_counts_offset;
    uint64_t calibration_hit_counts_bytes;
    uint8_t fit_state_sha256[32];
    uint8_t capture_sha256[32];
    uint8_t sidecar_sha256[32];
    uint8_t model_registry_sha256[32];
};
static_assert(sizeof(SlabAuxHeaderV2) == 280,
              "slab runtime v2 header must remain byte-stable");

struct SlabSidecarHeaderV2 {
    char magic[8];
    uint32_t version;
    uint32_t model_layer;
    uint32_t expert_count;
    uint32_t dimension;
    uint32_t expert_width;
    uint32_t slab_size;
    uint32_t slab_count;
    uint32_t alignment;
    uint64_t order_offset;
    uint64_t order_bytes;
    uint64_t payload_offset;
    uint64_t slab_bytes;
    uint64_t record_bytes;
    uint64_t gate_slab_bytes;
    uint64_t up_slab_bytes;
    uint64_t down_slab_bytes;
};
static_assert(sizeof(SlabSidecarHeaderV2) == 104,
              "slab sidecar v2 header must remain byte-stable");

bool checked_span(uint64_t offset, uint64_t bytes, uint64_t file_bytes) {
    return offset <= file_bytes && bytes <= file_bytes - offset;
}

template <typename T>
bool read_array(std::ifstream & input, uint64_t offset, uint64_t bytes,
                std::vector<T> & out, std::string * err) {
    if (bytes % sizeof(T) != 0 ||
        bytes / sizeof(T) > std::numeric_limits<size_t>::max()) {
        if (err) *err = "invalid slab runtime array size";
        return false;
    }
    out.resize(static_cast<size_t>(bytes / sizeof(T)));
    input.seekg(static_cast<std::streamoff>(offset));
    input.read(reinterpret_cast<char *>(out.data()),
               static_cast<std::streamsize>(bytes));
    if (!input) {
        if (err) *err = "short read from slab runtime artifact";
        return false;
    }
    return true;
}

int open_read_only(const std::string & path) {
#if defined(_WIN32)
    return ::_open(path.c_str(), _O_RDONLY | _O_BINARY);
#else
    return ::open(path.c_str(), O_RDONLY | O_CLOEXEC);
#endif
}

int open_read_only_direct(const std::string & path) {
#if defined(_WIN32) || !defined(O_DIRECT)
    (void) path;
    return -1;
#else
    return ::open(path.c_str(), O_RDONLY | O_CLOEXEC | O_DIRECT);
#endif
}

void close_fd(int fd) {
#if defined(_WIN32)
    ::_close(fd);
#else
    ::close(fd);
#endif
}

bool file_size(int fd, uint64_t & bytes) {
#if defined(_WIN32)
    struct _stat64 value{};
    if (::_fstat64(fd, &value) != 0 || value.st_size < 0) return false;
#else
    struct stat value{};
    if (::fstat(fd, &value) != 0 || value.st_size < 0) return false;
#endif
    bytes = static_cast<uint64_t>(value.st_size);
    return true;
}

bool read_exact_at(int fd, void * destination, size_t bytes, uint64_t offset) {
#if defined(_WIN32)
    if (::_lseeki64(fd, static_cast<__int64>(offset), SEEK_SET) < 0) return false;
    return ::_read(fd, destination, static_cast<unsigned int>(bytes)) ==
           static_cast<int>(bytes);
#else
    return ::pread(fd, destination, bytes, static_cast<off_t>(offset)) ==
           static_cast<ssize_t>(bytes);
#endif
}

struct ProcessIoSnapshot {
    uint64_t read_bytes = 0;
    uint64_t rchar = 0;
    uint64_t syscr = 0;
    uint64_t minor_faults = 0;
    uint64_t major_faults = 0;
};

ProcessIoSnapshot process_io_snapshot() {
    ProcessIoSnapshot result;
#if !defined(_WIN32)
    std::ifstream input("/proc/self/io");
    std::string key;
    uint64_t value = 0;
    while (input >> key >> value) {
        if (key == "read_bytes:") result.read_bytes = value;
        else if (key == "rchar:") result.rchar = value;
        else if (key == "syscr:") result.syscr = value;
    }
    struct rusage usage{};
    if (::getrusage(RUSAGE_SELF, &usage) == 0) {
        result.minor_faults = static_cast<uint64_t>(usage.ru_minflt);
        result.major_faults = static_cast<uint64_t>(usage.ru_majflt);
    }
#endif
    return result;
}

uint64_t saturating_delta(uint64_t end, uint64_t begin) {
    return end >= begin ? end - begin : 0;
}

std::string natural_sidecar_path(const std::string & directory,
                                 int model_layer) {
    char name[96];
    std::snprintf(name, sizeof(name),
                  "kimi_layer%02d_natural_slabs.k3slab", model_layer);
    if (directory.empty() || directory.back() == '/' ||
        directory.back() == '\\') {
        return directory + name;
    }
    return directory + "/" + name;
}

std::string calibrated_aux_path(const std::string & directory,
                                int model_layer) {
    char name[96];
    std::snprintf(name, sizeof(name),
                  "kimi_layer%02d_calibrated96.k3aux", model_layer);
    if (directory.empty() || directory.back() == '/' ||
        directory.back() == '\\') {
        return directory + name;
    }
    return directory + "/" + name;
}

bool valid_slab_order(const std::vector<uint16_t> & order) {
    if (order.size() != static_cast<size_t>(kExpertCount * kSlabCount)) {
        return false;
    }
    for (int expert = 0; expert < kExpertCount; ++expert) {
        bool seen[kSlabCount]{};
        for (int rank = 0; rank < kSlabCount; ++rank) {
            const uint16_t slab = order[
                static_cast<size_t>(expert) * kSlabCount + rank];
            if (slab >= kSlabCount || seen[slab]) return false;
            seen[slab] = true;
        }
    }
    return true;
}

struct Candidate {
    float score = 0.0f;
    int route = 0;
    int expert = 0;
    int rank = 0;
};

bool better_candidate(const Candidate & left, const Candidate & right) {
    if (left.score != right.score) return left.score > right.score;
    if (left.expert != right.expert) return left.expert < right.expert;
    if (left.rank != right.rank) return left.rank < right.rank;
    return left.route < right.route;
}

// Exact host reference for selected-slab/full-down execution.
bool evaluate_host_sparse_expert(
        ggml_backend_t backend,
        const MoeStreamExpertSpec & spec,
        const float * input_data,
        const std::vector<uint8_t> & gate_bytes,
        const std::vector<uint8_t> & up_bytes,
        const std::vector<uint8_t> & down_bytes,
        const std::vector<float> * activation_mask_values,
        std::vector<float> & result,
        std::string * err);

ggml_tensor * probe_scale_tensor(ggml_context * context,
                                 ggml_tensor * value,
                                 float scale) {
    return scale == 1.0f ? value : ggml_scale(context, value, scale);
}

ggml_tensor * probe_gated_activation(
        ggml_context * context,
        const MoeStreamExpertSpec & spec,
        ggml_tensor * gate,
        ggml_tensor * up) {
    if (spec.gated_activation == MoeGatedActivation::Situ) {
        ggml_tensor * nonlinear = ggml_scale(
            context, gate, 1.0f / spec.situ_beta);
        nonlinear = ggml_tanh(context, nonlinear);
        nonlinear = ggml_scale(context, nonlinear, spec.situ_beta);
        nonlinear = ggml_mul(
            context, nonlinear, ggml_sigmoid(context, gate));
        ggml_tensor * linear = ggml_scale(
            context, up, 1.0f / spec.situ_linear_beta);
        linear = ggml_tanh(context, linear);
        linear = ggml_scale(context, linear, spec.situ_linear_beta);
        return ggml_mul(context, nonlinear, linear);
    }
    if (spec.swiglu_clamp > 0.0f) {
        return ggml_swiglu_ds4_split(
            context, gate, up, spec.swiglu_clamp);
    }
    return ggml_swiglu_split(context, gate, up);
}

bool evaluate_host_sparse_expert(
        ggml_backend_t backend,
        const MoeStreamExpertSpec & spec,
        const float * input_data,
        const std::vector<uint8_t> & gate_bytes,
        const std::vector<uint8_t> & up_bytes,
        const std::vector<uint8_t> & down_bytes,
        const std::vector<float> * activation_mask_values,
        std::vector<float> & result,
        std::string * err) {
    if (!backend || spec.fused_gate_up || !input_data) {
        if (err) *err = "host sparse expert requires separate gate/up tensors";
        return false;
    }
    ggml_init_params parameters{};
    parameters.mem_size = 32 * 1024 * 1024;
    parameters.no_alloc = true;
    ggml_context * context = ggml_init(parameters);
    if (!context) {
        if (err) *err = "host sparse expert ggml_init failed";
        return false;
    }
    ggml_tensor * input = ggml_new_tensor_2d(
        context, GGML_TYPE_F32, spec.input_dim, 1);
    ggml_tensor * gate = ggml_new_tensor_2d(
        context, spec.gate_type, spec.input_dim, spec.intermediate_dim);
    ggml_tensor * up = ggml_new_tensor_2d(
        context, spec.up_type, spec.input_dim, spec.intermediate_dim);
    ggml_tensor * down = ggml_new_tensor_2d(
        context, spec.down_type, spec.intermediate_dim, spec.output_dim);
    ggml_set_input(input);
    ggml_set_input(gate);
    ggml_set_input(up);
    ggml_set_input(down);
    ggml_tensor * gate_value = probe_scale_tensor(
        context, ggml_mul_mat(context, gate, input), spec.gate_scale);
    ggml_tensor * up_value = probe_scale_tensor(
        context, ggml_mul_mat(context, up, input), spec.up_scale);
    ggml_tensor * activated = probe_gated_activation(
        context, spec, gate_value, up_value);
    ggml_tensor * activation_mask = nullptr;
    if (activation_mask_values) {
        if (activation_mask_values->size() !=
                static_cast<size_t>(spec.intermediate_dim)) {
            if (err) *err = "host sparse activation mask has wrong size";
            ggml_free(context);
            return false;
        }
        activation_mask = ggml_new_tensor_1d(
            context, GGML_TYPE_F32, spec.intermediate_dim);
        ggml_set_input(activation_mask);
        activated = ggml_mul(context, activated, activation_mask);
    }
    ggml_tensor * output = probe_scale_tensor(
        context, ggml_mul_mat(context, down, activated), spec.down_scale);
    if (gate_bytes.size() != ggml_nbytes(gate) ||
        up_bytes.size() != ggml_nbytes(up) ||
        down_bytes.size() != ggml_nbytes(down)) {
        if (err) *err = "host sparse expert byte size mismatch";
        ggml_free(context);
        return false;
    }
    ggml_cgraph * graph = ggml_new_graph_custom(context, 512, false);
    ggml_set_output(output);
    ggml_build_forward_expand(graph, output);
    ggml_gallocr_t allocator = ggml_gallocr_new(
        ggml_backend_get_default_buffer_type(backend));
    if (!allocator || !ggml_gallocr_alloc_graph(allocator, graph)) {
        if (err) *err = "host sparse expert graph allocation failed";
        if (allocator) ggml_gallocr_free(allocator);
        ggml_free(context);
        return false;
    }
    ggml_backend_tensor_set(
        input, input_data, 0,
        static_cast<size_t>(spec.input_dim) * sizeof(float));
    ggml_backend_tensor_set(gate, gate_bytes.data(), 0, gate_bytes.size());
    ggml_backend_tensor_set(up, up_bytes.data(), 0, up_bytes.size());
    ggml_backend_tensor_set(down, down_bytes.data(), 0, down_bytes.size());
    if (activation_mask) {
        ggml_backend_tensor_set(
            activation_mask, activation_mask_values->data(), 0,
            activation_mask_values->size() * sizeof(float));
    }
    result.resize(static_cast<size_t>(spec.output_dim));
    const ggml_status status =
        ggml_backend_graph_compute(backend, graph);
    if (status == GGML_STATUS_SUCCESS) {
        ggml_backend_tensor_get(
            output, result.data(), 0, result.size() * sizeof(float));
    } else if (err) {
        *err = "host sparse expert graph compute failed";
    }
    ggml_gallocr_free(allocator);
    ggml_free(context);
    return status == GGML_STATUS_SUCCESS;
}

struct SparseSlabPayload {
    uint16_t natural = 0;
    std::vector<uint8_t> gate;
    std::vector<uint8_t> up;
    std::vector<uint8_t> down;
};

// P27 lets the direct-I/O workers build the exact P25 wire image once, in a
// reusable pinned route buffer.  The evaluator can upload it without the
// second slab-vector-to-compact-vector host copy.
struct SparseCompactPayload {
    SparseCompactPayload() = default;
    SparseCompactPayload(const SparseCompactPayload &) = delete;
    SparseCompactPayload & operator=(const SparseCompactPayload &) = delete;
    ~SparseCompactPayload() {
#if defined(DFLASH27B_BACKEND_CUDA) || defined(DFLASH27B_BACKEND_HIP)
        if (data && owns_data) cudaFreeHost(data);
#endif
    }

    bool ensure(size_t requested, std::string * err) {
#if !defined(DFLASH27B_BACKEND_CUDA) && !defined(DFLASH27B_BACKEND_HIP)
        (void) requested;
        if (err) *err = "P27 direct pinned payload requires CUDA or HIP";
        return false;
#else
        if (owns_data && capacity >= requested) return true;
        if (data && owns_data) {
            cudaFreeHost(data);
            data = nullptr;
            capacity = 0;
        }
        if (cudaHostAlloc(&data, requested, cudaHostAllocDefault) !=
                cudaSuccess) {
            if (err) *err = "P27 route pinned allocation failed";
            return false;
        }
        capacity = requested;
        owns_data = true;
        return true;
#endif
    }

    void set_external(void * pointer, size_t available) {
#if defined(DFLASH27B_BACKEND_CUDA) || defined(DFLASH27B_BACKEND_HIP)
        if (data && owns_data) cudaFreeHost(data);
#endif
        data = pointer;
        capacity = available;
        owns_data = false;
    }

    void * data = nullptr;
    size_t capacity = 0;
    size_t bytes = 0;
    int slab_count = 0;
    size_t metadata_bytes = 32;
    size_t gate_slab_bytes = 0;
    size_t up_slab_bytes = 0;
    size_t down_slab_bytes = 0;
    bool component_major = false;
    bool owns_data = true;
};

bool pack_sparse_component_major(
        const std::vector<SparseSlabPayload> & slabs,
        const SparseCompactPayload * prepacked,
        void * destination, size_t capacity,
        KimiK3CompactWireLayout & layout, uint16_t & uploaded_mask,
        std::string * err) {
    const size_t slab_count = prepacked
        ? static_cast<size_t>(prepacked->slab_count) : slabs.size();
    if (slab_count == 0 || slab_count > kSlabCount || !destination) {
        if (err) *err = "compact executor received an invalid slab count";
        return false;
    }
    const size_t gate_slab_bytes = prepacked
        ? prepacked->gate_slab_bytes : slabs.front().gate.size();
    const size_t up_slab_bytes = prepacked
        ? prepacked->up_slab_bytes : slabs.front().up.size();
    const size_t down_slab_bytes = prepacked
        ? prepacked->down_slab_bytes : slabs.front().down.size();
    if (!kimi_k3_compact_wire_layout(
            static_cast<int>(slab_count), gate_slab_bytes, up_slab_bytes,
            down_slab_bytes, &layout) || layout.total_bytes > capacity) {
        if (err) *err = "compact executor wire image exceeds staging";
        return false;
    }
    std::array<uint16_t, kSlabCount> naturals{};
    const size_t record_bytes =
        gate_slab_bytes + up_slab_bytes + down_slab_bytes;
    const size_t prepacked_bytes = prepacked && prepacked->component_major
        ? layout.total_bytes
        : layout.metadata_bytes + slab_count * record_bytes;
    if (prepacked && (!prepacked->data ||
            prepacked->metadata_bytes != layout.metadata_bytes ||
            prepacked->bytes != prepacked_bytes)) {
        if (err) *err = "compact executor received an invalid P27 payload";
        return false;
    }
    for (size_t index = 0; index < slab_count; ++index) {
        if (prepacked) {
            std::memcpy(
                &naturals[index],
                static_cast<const uint8_t *>(prepacked->data) +
                    index * sizeof(uint16_t),
                sizeof(uint16_t));
        } else {
            const SparseSlabPayload & slab = slabs[index];
            if (slab.gate.size() != gate_slab_bytes ||
                slab.up.size() != up_slab_bytes ||
                slab.down.size() != down_slab_bytes) {
                if (err) *err = "compact executor received uneven slabs";
                return false;
            }
            naturals[index] = slab.natural;
        }
    }
    if (!kimi_k3_sparse_natural_mask(
            naturals.data(), static_cast<int>(slab_count), &uploaded_mask)) {
        if (err) *err = "compact executor received invalid or duplicate IDs";
        return false;
    }

    if (prepacked && prepacked->component_major) {
        if (destination != prepacked->data) {
            std::memcpy(destination, prepacked->data, layout.total_bytes);
        }
        return true;
    }

    auto * output = static_cast<uint8_t *>(destination);
    std::memset(output, 0, layout.metadata_bytes);
    std::memcpy(output, naturals.data(), slab_count * sizeof(uint16_t));
    for (size_t index = 0; index < slab_count; ++index) {
        const uint8_t * gate = nullptr;
        const uint8_t * up = nullptr;
        const uint8_t * down = nullptr;
        if (prepacked) {
            gate = static_cast<const uint8_t *>(prepacked->data) +
                layout.metadata_bytes + index * record_bytes;
            up = gate + gate_slab_bytes;
            down = up + up_slab_bytes;
        } else {
            gate = slabs[index].gate.data();
            up = slabs[index].up.data();
            down = slabs[index].down.data();
        }
        std::memcpy(
            output + layout.gate_offset + index * gate_slab_bytes,
            gate, gate_slab_bytes);
        std::memcpy(
            output + layout.up_offset + index * up_slab_bytes,
            up, up_slab_bytes);
        std::memcpy(
            output + layout.down_offset + index * down_slab_bytes,
            down, down_slab_bytes);
    }
    return true;
}

#if defined(DFLASH27B_BACKEND_CUDA) || defined(DFLASH27B_BACKEND_HIP)
bool p42_qualified_device(int device) {
#if defined(DFLASH27B_BACKEND_HIP)
    cudaDeviceProp properties{};
    return cudaGetDeviceProperties(&properties, device) == cudaSuccess &&
        std::strncmp(properties.gcnArchName, "gfx1151", 7) == 0 &&
        (properties.gcnArchName[7] == '\0' ||
         properties.gcnArchName[7] == ':');
#else
    return device == 1;
#endif
}

struct P42MeanSource {
    std::string path;
    uint64_t offset = 0;
    uint64_t bytes = 0;
};

class P42OrderedJoinArena {
public:
    ~P42OrderedJoinArena() {
        release_backend_storage();
    }

    bool load_resident_means(
            ggml_backend_t backend, int width,
            const std::vector<P42MeanSource> & sources, std::string * err) {
        BackendDeviceScope scope;
        if (!scope.enter(backend, err)) return false;
        const uint64_t layer_bytes = static_cast<uint64_t>(
            kExpertCount) * kSlabCount * width * sizeof(float);
        if (!p42_qualified_device(scope.device())) {
            if (err) {
                *err = "P42c resident means require the qualified "
                    "gfx1151 device";
            }
            return false;
        }
        if (width != kDimension ||
            sources.size() != kRoutedLayerCount || buffer_) {
            if (err) *err = "P42c resident mean table geometry is invalid";
            return false;
        }
        // GGML treats presence of this variable (including "0") as enabling
        // managed allocation. P42c's qualification premise is fixed GPU1 VRAM.
        if (std::getenv("GGML_CUDA_ENABLE_UNIFIED_MEMORY")) {
            if (err) *err =
                "P42c resident means require unified memory to be unset";
            return false;
        }
        for (const P42MeanSource & source : sources) {
            if (source.path.empty() || source.bytes != layer_bytes) {
                if (err) *err = "P42c resident mean source is invalid";
                return false;
            }
        }
        const size_t total_bytes = static_cast<size_t>(layer_bytes) *
            sources.size();
        constexpr size_t reserve_bytes = static_cast<size_t>(8) << 30;
        ggml_init_params params{};
        params.mem_size =
            (kMaximumRows + 5) * ggml_tensor_overhead() + 1024;
        params.no_alloc = true;
        context_ = ggml_init(params);
        if (context_) {
            mean_tensor_ = ggml_new_tensor_2d(
                context_, GGML_TYPE_F32, width, kResidentMeanRows);
            rows_ = ggml_new_tensor_2d(
                context_, GGML_TYPE_F32, width, kMaximumRows);
            for (int row = 0; row < kMaximumRows; ++row) {
                row_tensors_[static_cast<size_t>(row)] = ggml_view_1d(
                    context_, rows_, width,
                    static_cast<size_t>(row) * width * sizeof(float));
            }
            row_indices_tensor_ = ggml_new_tensor_1d(
                context_, GGML_TYPE_I32, kMaximumOperations);
            weights_tensor_ = ggml_new_tensor_1d(
                context_, GGML_TYPE_F32, kMaximumOperations);
            output_tensor_ = ggml_new_tensor_1d(
                context_, GGML_TYPE_F32, width);
        }
        if (!context_ || !mean_tensor_ || !rows_ ||
            !row_indices_tensor_ || !weights_tensor_ || !output_tensor_) {
            release_backend_storage();
            if (err) *err = "P42c resident arena allocation failed";
            return false;
        }
        const ggml_backend_buffer_type_t buffer_type =
            ggml_backend_get_default_buffer_type(backend);
        const size_t required_bytes = buffer_type
            ? ggml_backend_alloc_ctx_tensors_from_buft_size(
                context_, buffer_type)
            : 0;
        size_t free_bytes = 0;
        size_t total_device_bytes = 0;
        const auto memory_status =
#if defined(DFLASH27B_BACKEND_HIP)
            hipMemGetInfo(&free_bytes, &total_device_bytes);
#else
            cudaMemGetInfo(&free_bytes, &total_device_bytes);
#endif
        if (!buffer_type || required_bytes == 0 ||
            required_bytes > std::numeric_limits<size_t>::max() -
                reserve_bytes ||
            memory_status != cudaSuccess ||
            free_bytes < required_bytes + reserve_bytes) {
            release_backend_storage();
            if (err) *err = "P42c resident arena violates the 8-GiB reserve";
            return false;
        }
        buffer_ = ggml_backend_alloc_ctx_tensors_from_buft(
            context_, buffer_type);
        if (!buffer_ || ggml_backend_buffer_get_size(buffer_) < required_bytes ||
            mean_tensor_->ne[0] != width ||
            mean_tensor_->ne[1] != kResidentMeanRows ||
            mean_tensor_->ne[2] != 1 || mean_tensor_->ne[3] != 1 ||
            ggml_nbytes(mean_tensor_) != total_bytes) {
            release_backend_storage();
            if (err) *err = "P42c resident arena allocation failed";
            return false;
        }
        ggml_backend_buffer_set_usage(
            buffer_, GGML_BACKEND_BUFFER_USAGE_COMPUTE);
        std::vector<uint8_t> layer(static_cast<size_t>(layer_bytes));
        for (size_t index = 0; index < sources.size(); ++index) {
            const P42MeanSource & source = sources[index];
            std::ifstream input(source.path, std::ios::binary);
            input.seekg(static_cast<std::streamoff>(source.offset));
            input.read(
                reinterpret_cast<char *>(layer.data()),
                static_cast<std::streamsize>(layer.size()));
            if (!input || input.gcount() !=
                    static_cast<std::streamsize>(layer.size())) {
                release_backend_storage();
                if (err) *err = "P42c resident mean preload read failed";
                return false;
            }
            ggml_backend_tensor_set(
                mean_tensor_, layer.data(), index * layer.size(), layer.size());
        }
        backend_ = backend;
        runtime_device_ = scope.device();
        resident_mean_bytes_ = total_bytes;
        return true;
    }

    bool begin(ggml_backend_t backend, int width, std::string * err) {
        BackendDeviceScope scope;
        if (!scope.enter(backend, err)) return false;
        if (!p42_qualified_device(scope.device()) || width != kDimension) {
            if (err) {
                *err = "P42 ordered join requires the qualified "
                    "gfx1151 backend";
            }
            return false;
        }
        if (backend_ && backend_ != backend) {
            if (err) *err = "P42 ordered join backend changed";
            return false;
        }
        if (pending_output_ || queued_backend_work_) {
            if (err) *err = "P42 ordered join work was not consumed";
            return false;
        }
        if (!buffer_ || !mean_tensor_ || !rows_ ||
            !row_indices_tensor_ || !weights_tensor_ || !output_tensor_) {
            if (err) *err = "P42c resident means were not initialized";
            return false;
        }
        width_ = width;
        row_count_ = 0;
        row_indices_.clear();
        weights_.clear();
        calibrated_operations_ = -1;
        return true;
    }

    bool stage_device_output(
            const ggml_tensor * source, int & row, std::string * err) {
        if (!source || source->type != GGML_TYPE_F32 || !source->data ||
            !ggml_is_contiguous(source) || source->ne[0] != width_ ||
            source->ne[1] != 1 || source->ne[2] != 1 ||
            source->ne[3] != 1 ||
            row_count_ >= kMaximumRows) {
            if (err) *err = "P42 ordered join saw an invalid expert output";
            return false;
        }
        row = row_count_++;
        ggml_backend_tensor_copy_async(
            backend_, backend_, source,
            row_tensors_[static_cast<size_t>(row)]);
        queued_backend_work_ = true;
        ++expert_d2d_copies_;
        expert_d2d_bytes_ += static_cast<uint64_t>(width_) * sizeof(float);
        return true;
    }

    bool append(int row, float weight, std::string * err) {
        if (row < 0 || row >= row_count_ ||
            row_indices_.size() >= kMaximumOperations) {
            if (err) *err = "P42 ordered join operation is invalid";
            return false;
        }
        row_indices_.push_back(row);
        weights_.push_back(weight);
        return true;
    }

    bool append_resident_mean(
            int model_layer, int expert, int rank, float weight,
            std::string * err) {
        const int64_t row = static_cast<int64_t>(
            model_layer - kFirstModelLayer) * kExpertCount * kSlabCount +
            static_cast<int64_t>(expert) * kSlabCount + rank;
        if (model_layer < kFirstModelLayer ||
            model_layer >= kFirstModelLayer + kRoutedLayerCount ||
            expert < 0 || expert >= kExpertCount || rank < 0 ||
            rank >= kSlabCount || row < 0 || row >= kResidentMeanRows ||
            row_indices_.size() >= kMaximumOperations) {
            if (err) *err = "P42c resident mean descriptor is invalid";
            return false;
        }
        row_indices_.push_back(-1 - static_cast<int32_t>(row));
        weights_.push_back(weight);
        return true;
    }

    void seal_calibrated() {
        if (calibrated_operations_ < 0) {
            calibrated_operations_ = static_cast<int>(row_indices_.size());
        }
    }

    bool finish(std::string * err) {
        seal_calibrated();
        if (row_indices_.empty()) {
            if (err) *err = "P42 ordered join has no contributions";
            return false;
        }
        const char * failure = nullptr;
        ggml_backend_tensor_set_async(
            backend_, row_indices_tensor_, row_indices_.data(), 0,
            row_indices_.size() * sizeof(int32_t));
        ggml_backend_tensor_set_async(
            backend_, weights_tensor_, weights_.data(), 0,
            weights_.size() * sizeof(float));
        // All route copies and descriptor uploads share the backend stream.
        // One barrier here protects the raw ordered-join launch without
        // serializing each staged expert.
        ggml_backend_synchronize(backend_);
        queued_backend_work_ = false;
        if (!kimi_k3_ordered_join_launch(
                static_cast<const float *>(rows_->data), row_count_, width_,
                static_cast<const float *>(mean_tensor_->data),
                kResidentMeanRows,
                static_cast<const int32_t *>(row_indices_tensor_->data),
                static_cast<const float *>(weights_tensor_->data),
                static_cast<int>(row_indices_.size()), calibrated_operations_,
                static_cast<float *>(output_tensor_->data), &failure)) {
            if (err) {
                *err = std::string("P42 ordered join failed: ") +
                    (failure ? failure : "unknown");
            }
            return false;
        }
        ++join_launches_;
        pending_output_ = true;
        return true;
    }

    bool copy_to(
            ggml_backend_t destination_backend, ggml_tensor * destination,
            std::string * err) {
        BackendDeviceScope scope;
        if (!scope.enter(destination_backend, err)) return false;
        if (!pending_output_ || destination_backend != backend_ ||
            scope.device() != runtime_device_ || !destination ||
            destination->type != GGML_TYPE_F32 || !destination->data ||
            !ggml_is_contiguous(destination) ||
            destination->ne[0] != width_ || destination->ne[1] != 1 ||
            destination->ne[2] != 1 || destination->ne[3] != 1) {
            if (err) *err = "P42 ordered join destination is incompatible";
            return false;
        }
        ggml_backend_tensor_copy_async(
            backend_, destination_backend, output_tensor_, destination);
        ++output_d2d_copies_;
        output_d2d_bytes_ += static_cast<uint64_t>(width_) * sizeof(float);
        pending_output_ = false;
        return true;
    }

    bool read_to_host(float * destination, size_t values, std::string * err) {
        BackendDeviceScope scope;
        if (!scope.enter(backend_, err)) return false;
        if (!pending_output_ || !destination || values !=
                static_cast<size_t>(width_)) {
            if (err) *err = "P42 ordered join host output is incompatible";
            return false;
        }
        ggml_backend_tensor_get(
            output_tensor_, destination, 0, values * sizeof(float));
        pending_output_ = false;
        return true;
    }

    void discard() {
        if (queued_backend_work_ && backend_) {
            ggml_backend_synchronize(backend_);
        }
        queued_backend_work_ = false;
        pending_output_ = false;
    }

    uint64_t resident_mean_bytes() const { return resident_mean_bytes_; }
    uint64_t expert_d2d_copies() const { return expert_d2d_copies_; }
    uint64_t expert_d2d_bytes() const { return expert_d2d_bytes_; }
    uint64_t join_launches() const { return join_launches_; }
    uint64_t output_d2d_copies() const { return output_d2d_copies_; }
    uint64_t output_d2d_bytes() const { return output_d2d_bytes_; }

private:
    void release_backend_storage() {
        if (buffer_ && backend_) ggml_backend_synchronize(backend_);
        if (buffer_) ggml_backend_buffer_free(buffer_);
        if (context_) ggml_free(context_);
        buffer_ = nullptr;
        context_ = nullptr;
        mean_tensor_ = nullptr;
        rows_ = nullptr;
        row_tensors_.fill(nullptr);
        row_indices_tensor_ = nullptr;
        weights_tensor_ = nullptr;
        output_tensor_ = nullptr;
        backend_ = nullptr;
        runtime_device_ = -1;
        queued_backend_work_ = false;
    }

    static constexpr int kFirstModelLayer = 1;
    static constexpr int kRoutedLayerCount = 92;
    static constexpr int kResidentMeanRows =
        kRoutedLayerCount * kExpertCount * kSlabCount;
    static constexpr int kMaximumRows = kNativeTopK;
    static constexpr size_t kMaximumOperations =
        static_cast<size_t>(kNativeTopK * (kSlabCount + 1));
    ggml_backend_t backend_ = nullptr;
    int runtime_device_ = -1;
    int width_ = 0;
    int row_count_ = 0;
    int calibrated_operations_ = -1;
    ggml_context * context_ = nullptr;
    ggml_backend_buffer_t buffer_ = nullptr;
    ggml_tensor * mean_tensor_ = nullptr;
    ggml_tensor * rows_ = nullptr;
    std::array<ggml_tensor *, kMaximumRows> row_tensors_{};
    ggml_tensor * row_indices_tensor_ = nullptr;
    ggml_tensor * weights_tensor_ = nullptr;
    ggml_tensor * output_tensor_ = nullptr;
    std::vector<int32_t> row_indices_;
    std::vector<float> weights_;
    bool queued_backend_work_ = false;
    bool pending_output_ = false;
    uint64_t resident_mean_bytes_ = 0;
    uint64_t expert_d2d_copies_ = 0;
    uint64_t expert_d2d_bytes_ = 0;
    uint64_t join_launches_ = 0;
    uint64_t output_d2d_copies_ = 0;
    uint64_t output_d2d_bytes_ = 0;
};
#endif

// P23 keeps the P20 full-width arithmetic intact while removing repeated
// ggml graph allocation and CUDA buffer allocation from the per-route hot
// path.  A small cache entry is retained for each qtype/mask geometry seen by
// the model.  Evaluation remains sequential, so expert-ID accumulation order
// and the frozen calibrated96 semantics do not change.
class SparseDeviceExpertEvaluator {
public:
    struct CompactAsyncStats {
        uint64_t begins = 0;
        uint64_t jobs = 0;
        uint64_t h2d_calls = 0;
        uint64_t h2d_bytes = 0;
        uint64_t input_d2d_copies = 0;
        uint64_t input_d2d_bytes = 0;
        uint64_t graph_enqueues = 0;
        uint64_t layer_flushes = 0;
        uint64_t abort_syncs = 0;
        uint64_t max_inflight = 0;
        uint64_t submit_ns = 0;
        uint64_t device_window_ns = 0;
    };

    SparseDeviceExpertEvaluator() = default;
    ~SparseDeviceExpertEvaluator() {
        abort_compact_async_batch();
    }
    SparseDeviceExpertEvaluator(const SparseDeviceExpertEvaluator &) = delete;
    SparseDeviceExpertEvaluator & operator=(
        const SparseDeviceExpertEvaluator &) = delete;

    uint64_t compact_pack_ns() const { return compact_pack_ns_; }
    uint64_t compact_scatter_ns() const { return compact_scatter_ns_; }
    uint64_t expert_graph_ns() const { return expert_graph_ns_; }
    uint64_t expert_readback_ns() const { return expert_readback_ns_; }
    uint64_t compact_layouts() const { return compact_layouts_; }
    uint64_t compact_uploads() const { return compact_uploads_; }
    uint64_t compact_gate_stages() const { return compact_gate_stages_; }
    uint64_t compact_up_stages() const { return compact_up_stages_; }
    uint64_t compact_situ_stages() const { return compact_situ_stages_; }
    uint64_t compact_down_stages() const { return compact_down_stages_; }
    const CompactAsyncStats & compact_async_stats() const {
        return compact_async_stats_;
    }

    bool begin_compact_async_batch(
            ggml_backend_t backend, int max_jobs, std::string * err,
            const ggml_tensor * device_input = nullptr) {
#if !defined(DFLASH27B_BACKEND_CUDA) && !defined(DFLASH27B_BACKEND_HIP)
        (void) backend; (void) max_jobs; (void) device_input;
        if (err) *err = "P45 async compact queue requires CUDA or HIP";
        return false;
#else
        BackendDeviceScope device_scope;
        if (!device_scope.enter(backend, err)) return false;
        if (compact_async_active_ || max_jobs <= 0 ||
            max_jobs > kNativeTopK) {
            if (err) *err = "P45 async compact queue state is invalid";
            return false;
        }
        compact_async_backend_ = backend;
        compact_async_limit_ = max_jobs;
        compact_async_jobs_ = 0;
        compact_async_active_ = true;
        compact_async_device_input_ = device_input;
        compact_async_input_ready_ = false;
        compact_async_started_ = {};
        ++compact_async_stats_.begins;
        return true;
#endif
    }

    bool complete_compact_async_batch_after_sync(std::string * err) {
#if !defined(DFLASH27B_BACKEND_CUDA) && !defined(DFLASH27B_BACKEND_HIP)
        if (err) *err = "P45 async compact queue requires CUDA or HIP";
        return false;
#else
        if (!compact_async_active_) {
            if (err) *err = "P45 async compact queue is not active";
            return false;
        }
        if (compact_async_jobs_ > 0) {
            compact_async_stats_.device_window_ns += static_cast<uint64_t>(
                std::chrono::duration_cast<std::chrono::nanoseconds>(
                    std::chrono::steady_clock::now() -
                    compact_async_started_).count());
        }
        ++compact_async_stats_.layer_flushes;
        reset_compact_async_batch();
        return true;
#endif
    }

    void abort_compact_async_batch() {
#if defined(DFLASH27B_BACKEND_CUDA) || defined(DFLASH27B_BACKEND_HIP)
        if (!compact_async_active_) return;
        if (compact_async_backend_) {
            ggml_backend_synchronize(compact_async_backend_);
            ++compact_async_stats_.abort_syncs;
        }
        reset_compact_async_batch();
#endif
    }

    bool acquire_cache_lease(
            ggml_backend_t backend, const MoeStreamExpertSpec & spec,
            bool needs_mask, MoeHybridStreamEngine & engine,
            const MoeStreamExternalKey & key, uint16_t requested_mask,
            MoeStreamExternalLease & lease, std::string * err) {
#if !defined(DFLASH27B_BACKEND_CUDA) && !defined(DFLASH27B_BACKEND_HIP)
        (void) backend; (void) spec; (void) needs_mask; (void) engine;
        (void) key; (void) requested_mask; (void) lease; (void) err;
        return true;
#else
        if (backend != engine.compute_backend()) {
            if (err) *err =
                "P40 device cache backend does not match its stream engine";
            return false;
        }
        BackendDeviceScope device_scope;
        if (!device_scope.enter(backend, err)) return false;
        Entry * entry = find(backend, spec, needs_mask);
        if (!entry) entry = create(backend, spec, needs_mask, err);
        return entry && engine.acquire_external_device_lease(
            key, entry->weight_bytes, requested_mask, lease, err);
#endif
    }

    bool evaluate(
            ggml_backend_t backend,
            const MoeStreamExpertSpec & spec,
            const float * input_data,
            const std::vector<SparseSlabPayload> & slabs,
            const SparseCompactPayload * prepacked_compact,
            const std::vector<float> & activation_mask_values,
            size_t down_slab_row_bytes,
            std::vector<float> & result,
            uint64_t & authoritative_h2d_bytes,
            uint64_t & metadata_h2d_bytes,
            uint64_t & device_zero_bytes,
            KimiK3SparseUpload upload,
            MoeStreamExternalLease * cache_lease,
            uint16_t requested_mask,
            bool compact_executor,
            bool & compact_invalid,
            std::string * err) {
        compact_invalid = false;
        ggml_tensor * device_output = nullptr;
        const bool evaluated = compact_executor
            ? eval_compact_into(
                backend, spec, input_data, slabs, prepacked_compact,
                requested_mask, down_slab_row_bytes, device_output,
                authoritative_h2d_bytes, metadata_h2d_bytes,
                compact_invalid, err)
            : eval_into(
                backend, spec, input_data, slabs, prepacked_compact,
                activation_mask_values, down_slab_row_bytes, device_output,
                authoritative_h2d_bytes, metadata_h2d_bytes,
                device_zero_bytes, upload, cache_lease, err);
        if (!evaluated) {
            return false;
        }
#if !defined(DFLASH27B_BACKEND_CUDA) && !defined(DFLASH27B_BACKEND_HIP)
        (void) result;
        return false;
#else
        BackendDeviceScope device_scope;
        if (!device_scope.enter(backend, err)) return false;
        result.resize(static_cast<size_t>(spec.output_dim));
        const auto readback_started = std::chrono::steady_clock::now();
        ggml_backend_tensor_get(
            device_output, result.data(), 0, result.size() * sizeof(float));
        expert_readback_ns_ += static_cast<uint64_t>(
            std::chrono::duration_cast<std::chrono::nanoseconds>(
                std::chrono::steady_clock::now() - readback_started).count());
        return true;
#endif
    }

    bool evaluate_compact_device(
            ggml_backend_t backend,
            const MoeStreamExpertSpec & spec,
            const float * input_data,
            const std::vector<SparseSlabPayload> & slabs,
            const SparseCompactPayload * prepacked_compact,
            uint16_t requested_mask,
            size_t down_slab_row_bytes,
            ggml_tensor *& device_output,
            uint64_t & authoritative_h2d_bytes,
            uint64_t & metadata_h2d_bytes,
            bool & invalid,
            std::string * err) {
        return eval_compact_into(
            backend, spec, input_data, slabs, prepacked_compact,
            requested_mask, down_slab_row_bytes, device_output,
            authoritative_h2d_bytes, metadata_h2d_bytes, invalid, err);
    }

    using CompactUnionConsumer = std::function<bool(
        int base_row, int valid_rows,
        const std::array<ggml_tensor *, 8> & outputs,
        std::string * err)>;

    // Executes one expert's component-major union payload over several rows.
    // The union weights have their own backend buffer: unlike graph scratch,
    // their lifetime spans every width-sized graph replay in this call.
    bool evaluate_compact_union_device(
            ggml_backend_t backend,
            const MoeStreamExpertSpec & spec,
            const float * input_data,
            int rows,
            int graph_width,
            const SparseCompactPayload & compact,
            const uint16_t * requested_masks,
            size_t down_slab_row_bytes,
            const CompactUnionConsumer & consume,
            uint64_t & authoritative_h2d_bytes,
            uint64_t & metadata_h2d_bytes,
            uint64_t & graph_ns,
            bool & invalid,
            std::string * err) {
        invalid = false;
#if !defined(DFLASH27B_BACKEND_CUDA) && !defined(DFLASH27B_BACKEND_HIP)
        (void) backend; (void) spec; (void) input_data; (void) rows;
        (void) graph_width; (void) compact; (void) requested_masks;
        (void) down_slab_row_bytes; (void) consume;
        (void) authoritative_h2d_bytes; (void) metadata_h2d_bytes;
        (void) graph_ns;
        invalid = true;
        if (err) *err = "compact union executor requires CUDA or HIP";
        return false;
#else
        if (!backend || !ggml_backend_is_cuda(backend) || !input_data ||
            !requested_masks || !consume || rows <= 0 ||
            (graph_width != 1 && graph_width != 2 && graph_width != 8) ||
            spec.input_dim != 3584 || spec.intermediate_dim != 3072 ||
            spec.output_dim != 3584 || spec.fused_gate_up ||
            spec.gate_type != GGML_TYPE_IQ1_S ||
            spec.up_type != GGML_TYPE_IQ1_S ||
            spec.down_type != GGML_TYPE_IQ2_XXS ||
            spec.gated_activation != MoeGatedActivation::Situ ||
            compact.slab_count <= 0 || compact.slab_count > kSlabCount ||
            !compact.component_major || !compact.data) {
            invalid = true;
            if (err) *err = "compact union executor geometry is unsupported";
            return false;
        }
        BackendDeviceScope device_scope;
        if (!device_scope.enter(backend, err)) return false;

        KimiK3CompactWireLayout layout;
        if (!kimi_k3_compact_wire_layout(
                compact.slab_count, compact.gate_slab_bytes,
                compact.up_slab_bytes, compact.down_slab_bytes, &layout) ||
            compact.bytes != layout.total_bytes ||
            compact.capacity < compact.bytes ||
            compact.down_slab_bytes != down_slab_row_bytes *
                static_cast<size_t>(spec.output_dim)) {
            invalid = true;
            if (err) *err = "compact union executor wire is invalid";
            return false;
        }
        const auto * wire = static_cast<const uint8_t *>(compact.data);
        const auto * naturals = reinterpret_cast<const uint16_t *>(wire);
        uint16_t union_mask = 0;
        if (!kimi_k3_sparse_natural_mask(
                naturals, compact.slab_count, &union_mask)) {
            invalid = true;
            if (err) *err = "compact union executor natural IDs are invalid";
            return false;
        }
        for (int row = 0; row < rows; ++row) {
            if (requested_masks[row] == 0 ||
                (requested_masks[row] & ~union_mask) != 0) {
                invalid = true;
                if (err) *err = "compact union row mask is not resident";
                return false;
            }
        }

        CompactUnionEntry * entry = find_compact_union(
            backend, spec, compact.slab_count, graph_width);
        if (!entry) {
            entry = create_compact_union(
                backend, spec, compact.slab_count, graph_width, err);
            if (!entry) return false;
        }
        ggml_backend_tensor_set(
            entry->gate, wire + layout.gate_offset, 0,
            compact.gate_slab_bytes * compact.slab_count);
        ggml_backend_tensor_set(
            entry->up, wire + layout.up_offset, 0,
            compact.up_slab_bytes * compact.slab_count);
        ggml_backend_tensor_set(
            entry->down, wire + layout.down_offset, 0,
            compact.down_slab_bytes * compact.slab_count);
        authoritative_h2d_bytes += static_cast<uint64_t>(compact.slab_count) *
            (compact.gate_slab_bytes + compact.up_slab_bytes +
             compact.down_slab_bytes);

        const size_t input_row_bytes =
            static_cast<size_t>(spec.input_dim) * sizeof(float);
        std::vector<float> chunk_inputs(
            static_cast<size_t>(graph_width) * spec.input_dim);
        std::array<std::array<int32_t, kSlabCount>, 8> maps{};
        ScopedCudaGraphOverrides exact_scope(
            /* disable_graphs = */ true,
            /* mmvq_max_ncols = */ 8,
            /* skip_property_check = */ false);
        for (int base = 0; base < rows; base += graph_width) {
            const int valid = std::min(graph_width, rows - base);
            for (int lane = 0; lane < graph_width; ++lane) {
                const int source_row = base + std::min(lane, valid - 1);
                std::memcpy(
                    chunk_inputs.data() +
                        static_cast<size_t>(lane) * spec.input_dim,
                    input_data +
                        static_cast<size_t>(source_row) * spec.input_dim,
                    input_row_bytes);
                maps[static_cast<size_t>(lane)].fill(-1);
                const uint16_t mask = requested_masks[source_row];
                for (int slot = 0; slot < compact.slab_count; ++slot) {
                    const uint16_t natural = naturals[slot];
                    if (mask & (1u << natural)) {
                        maps[static_cast<size_t>(lane)][natural] = slot;
                    }
                }
            }
            ggml_backend_tensor_set(
                entry->input, chunk_inputs.data(), 0,
                chunk_inputs.size() * sizeof(float));
            for (int lane = 0; lane < graph_width; ++lane) {
                ggml_backend_tensor_set(
                    entry->maps[static_cast<size_t>(lane)],
                    maps[static_cast<size_t>(lane)].data(), 0,
                    sizeof(maps[static_cast<size_t>(lane)]));
            }
            metadata_h2d_bytes += chunk_inputs.size() * sizeof(float) +
                static_cast<uint64_t>(graph_width) *
                    sizeof(std::array<int32_t, kSlabCount>);
            const auto started = std::chrono::steady_clock::now();
            const ggml_status status =
                ggml_backend_graph_compute(backend, entry->graph);
            graph_ns += static_cast<uint64_t>(
                std::chrono::duration_cast<std::chrono::nanoseconds>(
                    std::chrono::steady_clock::now() - started).count());
            if (status != GGML_STATUS_SUCCESS) {
                if (err) *err = "compact union executor graph failed";
                return false;
            }
            if (!consume(base, valid, entry->outputs, err)) return false;
            // A consumer may stage these rows with an asynchronous D2D copy.
            // Retire that copy before the next replay reuses output scratch.
            // A later double-buffered path may replace this conservative
            // boundary, but must preserve the same ownership contract.
            ggml_backend_synchronize(backend);
        }
        return true;
#endif
    }

    bool evaluate_cached_device(
            ggml_backend_t backend,
            const MoeStreamExpertSpec & spec,
            const float * input_data,
            const std::vector<SparseSlabPayload> & slabs,
            const SparseCompactPayload * prepacked_compact,
            const std::vector<float> & activation_mask_values,
            size_t down_slab_row_bytes,
            ggml_tensor *& device_output,
            uint64_t & authoritative_h2d_bytes,
            uint64_t & metadata_h2d_bytes,
            uint64_t & device_zero_bytes,
            KimiK3SparseUpload upload,
            MoeStreamExternalLease * cache_lease,
            std::string * err) {
        return eval_into(
            backend, spec, input_data, slabs, prepacked_compact,
            activation_mask_values, down_slab_row_bytes, device_output,
            authoritative_h2d_bytes, metadata_h2d_bytes, device_zero_bytes,
            upload, cache_lease, err);
    }

    bool eval_into(
            ggml_backend_t backend,
            const MoeStreamExpertSpec & spec,
            const float * input_data,
            const std::vector<SparseSlabPayload> & slabs,
            const SparseCompactPayload * prepacked_compact,
            const std::vector<float> & activation_mask_values,
            size_t down_slab_row_bytes,
            ggml_tensor *& device_output,
            uint64_t & authoritative_h2d_bytes,
            uint64_t & metadata_h2d_bytes,
            uint64_t & device_zero_bytes,
            KimiK3SparseUpload upload,
            MoeStreamExternalLease * cache_lease,
            std::string * err) {
        device_output = nullptr;
#if !defined(DFLASH27B_BACKEND_CUDA) && !defined(DFLASH27B_BACKEND_HIP)
        (void) backend; (void) spec; (void) input_data; (void) slabs;
        (void) prepacked_compact;
        (void) activation_mask_values; (void) down_slab_row_bytes;
        (void) authoritative_h2d_bytes;
        (void) metadata_h2d_bytes; (void) device_zero_bytes;
        (void) upload; (void) cache_lease;
        if (err) *err = "P23 sparse scratch currently requires CUDA or HIP";
        return false;
#else
        if (!backend || !ggml_backend_is_cuda(backend) ||
            spec.fused_gate_up || !input_data ||
            activation_mask_values.size() !=
                static_cast<size_t>(spec.intermediate_dim)) {
            if (err) *err =
                "P23 sparse scratch received an incompatible expert";
            return false;
        }
        if (prepacked_compact && prepacked_compact->component_major) {
            if (err) {
                *err = "P23/P40 requires record-major compact payloads";
            }
            return false;
        }
        BackendDeviceScope device_scope;
        if (!device_scope.enter(backend, err)) return false;
        const size_t slab_count = prepacked_compact
            ? static_cast<size_t>(prepacked_compact->slab_count)
            : slabs.size();
        const bool needs_mask = std::any_of(
            activation_mask_values.begin(), activation_mask_values.end(),
            [](float value) { return value == 0.0f; });
        Entry * entry = find(backend, spec, needs_mask);
        if (!entry) {
            entry = create(backend, spec, needs_mask, err);
            if (!entry) return false;
        }
        const bool cached = cache_lease && *cache_lease;
        if (cached) {
            if (!cache_lease->bind_tensor(
                    entry->gate, entry->gate_offset, err) ||
                !cache_lease->bind_tensor(
                    entry->up, entry->up_offset, err) ||
                !cache_lease->bind_tensor(
                    entry->down, entry->down_offset, err)) {
                return false;
            }
        } else {
            entry->gate->buffer = entry->gate_owned_buffer;
            entry->gate->data = entry->gate_owned_data;
            entry->up->buffer = entry->up_owned_buffer;
            entry->up->data = entry->up_owned_data;
            entry->down->buffer = entry->down_owned_buffer;
            entry->down->data = entry->down_owned_data;
        }

        auto cuda_ok = [&](cudaError_t status, const char * operation) {
            if (status == cudaSuccess) return true;
            if (err) {
                *err = std::string("P23 ") + operation + " failed: " +
                    cudaGetErrorString(status);
            }
            return false;
        };
        const bool cache_hit = cached && cache_lease->missing_mask() == 0;
        if (!cache_hit && (slab_count == 0 || slab_count > kSlabCount)) {
            if (err) *err = "sparse payload slab count is outside 1..12";
            return false;
        }
        bool ok = true;
        bool clear_destinations = !cached;
        if (cached && cache_lease->clear_required()) {
            ok = cache_lease->clear_prefix(entry->weight_bytes, err);
            device_zero_bytes += entry->weight_bytes;
        } else if (clear_destinations) {
            device_zero_bytes += ggml_nbytes(entry->gate) +
                ggml_nbytes(entry->up) + ggml_nbytes(entry->down);
        }
        uint16_t uploaded_mask = 0;
        if (ok && !cache_hit) {
            std::array<uint16_t, kSlabCount> uploaded_naturals{};
            if (prepacked_compact) {
                if (!prepacked_compact->data ||
                    prepacked_compact->metadata_bytes <
                        slab_count * sizeof(uint16_t) ||
                    prepacked_compact->bytes <
                        prepacked_compact->metadata_bytes) {
                    if (err) *err = "P27 invalid prepacked compact payload";
                    return false;
                }
                for (size_t index = 0; index < slab_count; ++index) {
                    std::memcpy(
                        &uploaded_naturals[index],
                        static_cast<const uint8_t *>(
                            prepacked_compact->data) +
                            index * sizeof(uint16_t),
                        sizeof(uint16_t));
                }
            } else {
                for (size_t index = 0; index < slab_count; ++index) {
                    uploaded_naturals[index] = slabs[index].natural;
                }
            }
            if (!kimi_k3_sparse_natural_mask(
                    uploaded_naturals.data(), static_cast<int>(slab_count),
                    &uploaded_mask)) {
                if (err) *err = "sparse payload has invalid natural slab metadata";
                return false;
            }
            if (cached && uploaded_mask != cache_lease->missing_mask()) {
                if (err) *err =
                    "sparse payload does not match the device-cache fill mask";
                return false;
            }
        }
        if (ok && !cache_hit &&
            upload != KimiK3SparseUpload::SlabCopies) {
            const auto pack_started = std::chrono::steady_clock::now();
            constexpr size_t metadata_bytes = 32;
            if (slab_count == 0) {
                if (err) *err = "P25 compact upload received no slabs";
                return false;
            }
            const size_t gate_slab_bytes = prepacked_compact
                ? prepacked_compact->gate_slab_bytes
                : slabs.front().gate.size();
            const size_t up_slab_bytes = prepacked_compact
                ? prepacked_compact->up_slab_bytes
                : slabs.front().up.size();
            const size_t down_slab_bytes = prepacked_compact
                ? prepacked_compact->down_slab_bytes
                : slabs.front().down.size();
            const size_t record_bytes =
                gate_slab_bytes + up_slab_bytes + down_slab_bytes;
            const size_t compact_bytes =
                metadata_bytes + slab_count * record_bytes;
            if (!ensure_compact_staging(
                    *entry, compact_bytes,
                    upload == KimiK3SparseUpload::PinnedCompact, err)) {
                return false;
            }
            std::vector<uint8_t> pageable_compact;
            uint8_t * compact_host = nullptr;
            if (prepacked_compact) {
                if (!prepacked_compact->data ||
                    prepacked_compact->bytes != compact_bytes ||
                    prepacked_compact->metadata_bytes != metadata_bytes) {
                    if (err) *err = "P27 invalid prepacked compact payload";
                    return false;
                }
                compact_host = static_cast<uint8_t *>(
                    prepacked_compact->data);
            } else if (upload == KimiK3SparseUpload::PinnedCompact) {
                compact_host = static_cast<uint8_t *>(
                    entry->compact_host_staging);
                std::memset(compact_host, 0, metadata_bytes);
            } else {
                pageable_compact.assign(compact_bytes, 0);
                compact_host = pageable_compact.data();
            }
            if (!prepacked_compact) {
                size_t payload_offset = metadata_bytes;
                for (size_t index = 0; index < slabs.size(); ++index) {
                    const SparseSlabPayload & slab = slabs[index];
                    if (slab.natural >= kSlabCount ||
                        slab.gate.size() != gate_slab_bytes ||
                        slab.up.size() != up_slab_bytes ||
                        slab.down.size() != down_slab_bytes) {
                        if (err) *err =
                            "P25 compact upload has uneven slabs";
                        return false;
                    }
                    std::memcpy(
                        compact_host + index * sizeof(uint16_t),
                        &slab.natural, sizeof(uint16_t));
                    std::memcpy(
                        compact_host + payload_offset,
                        slab.gate.data(), slab.gate.size());
                    payload_offset += slab.gate.size();
                    std::memcpy(
                        compact_host + payload_offset,
                        slab.up.data(), slab.up.size());
                    payload_offset += slab.up.size();
                    std::memcpy(
                        compact_host + payload_offset,
                        slab.down.data(), slab.down.size());
                    payload_offset += slab.down.size();
                }
            }
            compact_pack_ns_ += static_cast<uint64_t>(
                std::chrono::duration_cast<std::chrono::nanoseconds>(
                    std::chrono::steady_clock::now() - pack_started).count());
            const auto scatter_started = std::chrono::steady_clock::now();
            const char * scatter_failure = nullptr;
            ok = kimi_k3_sparse_scatter_upload_incremental(
                entry->gate->data, ggml_nbytes(entry->gate),
                entry->up->data, ggml_nbytes(entry->up),
                entry->down->data, ggml_nbytes(entry->down),
                entry->compact_staging, entry->compact_capacity,
                compact_host, compact_bytes,
                static_cast<int>(slab_count), metadata_bytes,
                gate_slab_bytes, up_slab_bytes, down_slab_bytes,
                down_slab_row_bytes, entry->down->nb[1], spec.output_dim,
                clear_destinations,
                &scatter_failure);
            compact_scatter_ns_ += static_cast<uint64_t>(
                std::chrono::duration_cast<std::chrono::nanoseconds>(
                    std::chrono::steady_clock::now() - scatter_started).count());
            if (!ok && err) {
                *err = std::string("P25 compact sparse scatter failed: ") +
                    (scatter_failure ? scatter_failure : "unknown");
            }
            authoritative_h2d_bytes += slab_count * record_bytes;
            metadata_h2d_bytes += metadata_bytes;
        } else if (ok && !cache_hit) {
            ok =
                (!clear_destinations || cuda_ok(cudaMemset(
                    entry->gate->data, 0, ggml_nbytes(entry->gate)),
                    "gate zero")) &&
                (!clear_destinations || cuda_ok(cudaMemset(
                    entry->up->data, 0, ggml_nbytes(entry->up)),
                    "up zero")) &&
                (!clear_destinations || cuda_ok(cudaMemset(
                    entry->down->data, 0, ggml_nbytes(entry->down)),
                    "down zero"));
            for (const SparseSlabPayload & slab : slabs) {
                if (!ok || slab.natural >= kSlabCount || slab.gate.empty() ||
                    slab.up.empty() || slab.down.empty()) {
                    ok = false;
                    if (err && err->empty()) {
                        *err = "P23 invalid sparse slab payload";
                    }
                    break;
                }
                const size_t gate_offset =
                    static_cast<size_t>(slab.natural) * slab.gate.size();
                const size_t up_offset =
                    static_cast<size_t>(slab.natural) * slab.up.size();
                const size_t down_offset =
                    static_cast<size_t>(slab.natural) * down_slab_row_bytes;
                ok = cuda_ok(cudaMemcpy(
                        static_cast<uint8_t *>(entry->gate->data) + gate_offset,
                        slab.gate.data(), slab.gate.size(),
                        cudaMemcpyHostToDevice), "gate slab upload") &&
                    cuda_ok(cudaMemcpy(
                        static_cast<uint8_t *>(entry->up->data) + up_offset,
                        slab.up.data(), slab.up.size(),
                        cudaMemcpyHostToDevice), "up slab upload") &&
                    cuda_ok(cudaMemcpy2D(
                        static_cast<uint8_t *>(entry->down->data) + down_offset,
                        entry->down->nb[1], slab.down.data(),
                        down_slab_row_bytes, down_slab_row_bytes,
                        static_cast<size_t>(spec.output_dim),
                        cudaMemcpyHostToDevice), "down slab upload");
                authoritative_h2d_bytes +=
                    slab.gate.size() + slab.up.size() + slab.down.size();
            }
        }
        if (ok && cached && cache_lease->missing_mask() != 0) {
            ok = cache_lease->commit(uploaded_mask, err);
        }
        if (ok) {
            const size_t input_bytes =
                static_cast<size_t>(spec.input_dim) * sizeof(float);
            const size_t mask_bytes = entry->activation_mask
                ? activation_mask_values.size() * sizeof(float) : 0;
            const auto graph_started = std::chrono::steady_clock::now();
            ggml_status status = GGML_STATUS_FAILED;
            if (compact_async_active_) {
                if (backend != compact_async_backend_ ||
                    compact_async_jobs_ < 0 ||
                    compact_async_jobs_ >= compact_async_limit_) {
                    if (err) *err = "P45 cached queue capacity is invalid";
                    return false;
                }
                CompactAsyncSlot & slot = compact_async_slots_[
                    static_cast<size_t>(compact_async_jobs_)];
                if (!ensure_compact_async_slot(
                        slot, input_bytes + mask_bytes, err)) {
                    return false;
                }
                auto * staging = static_cast<uint8_t *>(slot.host_staging);
                std::memcpy(staging, input_data, input_bytes);
                if (mask_bytes != 0) {
                    std::memcpy(
                        staging + input_bytes,
                        activation_mask_values.data(), mask_bytes);
                }
                if (compact_async_jobs_ == 0) {
                    compact_async_started_ = graph_started;
                }
                ggml_backend_tensor_set_async(
                    backend, entry->input, staging, 0, input_bytes);
                if (mask_bytes != 0) {
                    ggml_backend_tensor_set_async(
                        backend, entry->activation_mask,
                        staging + input_bytes, 0, mask_bytes);
                }
                status = ggml_backend_graph_compute_async(
                    backend, entry->graph);
                compact_async_stats_.submit_ns += static_cast<uint64_t>(
                    std::chrono::duration_cast<std::chrono::nanoseconds>(
                        std::chrono::steady_clock::now() -
                            graph_started).count());
                if (status == GGML_STATUS_SUCCESS) {
                    ++compact_async_jobs_;
                    ++compact_async_stats_.jobs;
                    compact_async_stats_.h2d_calls +=
                        1 + (mask_bytes != 0 ? 1 : 0);
                    compact_async_stats_.h2d_bytes +=
                        input_bytes + mask_bytes;
                    ++compact_async_stats_.graph_enqueues;
                    compact_async_stats_.max_inflight = std::max<uint64_t>(
                        compact_async_stats_.max_inflight,
                        static_cast<uint64_t>(compact_async_jobs_));
                }
            } else {
                ggml_backend_tensor_set(
                    entry->input, input_data, 0, input_bytes);
                if (mask_bytes != 0) {
                    ggml_backend_tensor_set(
                        entry->activation_mask,
                        activation_mask_values.data(), 0, mask_bytes);
                }
                status = ggml_backend_graph_compute(backend, entry->graph);
                expert_graph_ns_ += static_cast<uint64_t>(
                    std::chrono::duration_cast<std::chrono::nanoseconds>(
                        std::chrono::steady_clock::now() -
                            graph_started).count());
            }
            metadata_h2d_bytes += input_bytes + mask_bytes;
            ok = status == GGML_STATUS_SUCCESS;
            if (!ok && err) {
                *err = "P23 persistent sparse graph compute failed";
            }
        }
        if (ok) device_output = entry->output;
        return ok;
#endif
    }

    bool eval_compact_into(
            ggml_backend_t backend,
            const MoeStreamExpertSpec & spec,
            const float * input_data,
            const std::vector<SparseSlabPayload> & slabs,
            const SparseCompactPayload * prepacked_compact,
            uint16_t requested_mask,
            size_t down_slab_row_bytes,
            ggml_tensor *& device_output,
            uint64_t & authoritative_h2d_bytes,
            uint64_t & metadata_h2d_bytes,
            bool & invalid,
            std::string * err) {
        device_output = nullptr;
        invalid = false;
#if !defined(DFLASH27B_BACKEND_CUDA) && !defined(DFLASH27B_BACKEND_HIP)
        (void) backend; (void) spec; (void) input_data; (void) slabs;
        (void) prepacked_compact; (void) requested_mask;
        (void) down_slab_row_bytes; (void) authoritative_h2d_bytes;
        (void) metadata_h2d_bytes;
        if (err) *err = "compact sparse executor requires CUDA or HIP";
        return false;
#else
        const bool device_input_available = compact_async_active_ &&
            compact_async_device_input_ != nullptr;
        if (!backend || !ggml_backend_is_cuda(backend) ||
            (!input_data && !device_input_available) ||
            spec.fused_gate_up || spec.intermediate_dim != 3072 ||
            (spec.down_type != GGML_TYPE_IQ1_S &&
             spec.down_type != GGML_TYPE_IQ2_XXS)) {
            if (err) *err = "compact sparse executor geometry is unsupported";
            return false;
        }
        const size_t slab_count = prepacked_compact
            ? static_cast<size_t>(prepacked_compact->slab_count)
            : slabs.size();
        if (slab_count == 0 || slab_count > kSlabCount) {
            invalid = true;
            if (err) *err = "compact sparse executor slab count is invalid";
            return false;
        }
        BackendDeviceScope device_scope;
        if (!device_scope.enter(
                backend, err, /* restore_previous = */ !compact_async_active_)) {
            return false;
        }
        CompactEntry * entry = find_compact(
            backend, spec, static_cast<int>(slab_count));
        if (!entry) {
            entry = create_compact(
                backend, spec, static_cast<int>(slab_count), err);
            if (!entry) return false;
        }
        const size_t gate_slab_bytes = prepacked_compact
            ? prepacked_compact->gate_slab_bytes : slabs.front().gate.size();
        const size_t up_slab_bytes = prepacked_compact
            ? prepacked_compact->up_slab_bytes : slabs.front().up.size();
        const size_t down_slab_bytes = prepacked_compact
            ? prepacked_compact->down_slab_bytes : slabs.front().down.size();
        KimiK3CompactWireLayout layout;
        if (!kimi_k3_compact_wire_layout(
                static_cast<int>(slab_count), gate_slab_bytes,
                up_slab_bytes, down_slab_bytes, &layout)) {
            invalid = true;
            return false;
        }
        const size_t input_bytes =
            static_cast<size_t>(spec.input_dim) * sizeof(float);
        constexpr size_t map_bytes =
            sizeof(std::array<int32_t, kSlabCount>);
        const bool direct_component_wire = prepacked_compact &&
            prepacked_compact->component_major;
        const bool upload_device_input = device_input_available &&
            !compact_async_input_ready_;
        const bool upload_host_input = !compact_async_active_ ||
            (!device_input_available && compact_async_input_source_ == nullptr);
        void * host_staging = nullptr;
        size_t host_capacity = 0;
        CompactAsyncSlot * async_slot = nullptr;
        if (compact_async_active_) {
            if (backend != compact_async_backend_ || compact_async_jobs_ < 0 ||
                compact_async_jobs_ >= compact_async_limit_ ||
                (!device_input_available && compact_async_input_source_ &&
                 compact_async_input_source_ != input_data) ||
                (upload_host_input && input_bytes >
                    std::numeric_limits<size_t>::max() - map_bytes) ||
                (!direct_component_wire &&
                 layout.total_bytes > std::numeric_limits<size_t>::max() -
                    (upload_host_input ? input_bytes : 0) - map_bytes)) {
                if (err) *err = "P45 async compact queue capacity is invalid";
                return false;
            }
            async_slot = &compact_async_slots_[
                static_cast<size_t>(compact_async_jobs_)];
            const size_t required = (upload_host_input ? input_bytes : 0) +
                map_bytes +
                (direct_component_wire ? 0 : layout.total_bytes);
            if (!ensure_compact_async_slot(*async_slot, required, err)) {
                return false;
            }
            host_staging = direct_component_wire
                ? prepacked_compact->data : async_slot->host_staging;
            host_capacity = direct_component_wire
                ? prepacked_compact->capacity : async_slot->host_capacity;
        } else if (direct_component_wire) {
            host_staging = prepacked_compact->data;
            host_capacity = prepacked_compact->capacity;
        } else {
            if (!ensure_compact_host_staging(
                    *entry, layout.total_bytes, err)) return false;
            host_staging = entry->host_staging;
            host_capacity = entry->host_capacity;
        }
        uint16_t uploaded_mask = 0;
        const auto pack_started = std::chrono::steady_clock::now();
        if (!pack_sparse_component_major(
                slabs, prepacked_compact, host_staging,
                host_capacity, layout, uploaded_mask, err)) {
            invalid = true;
            return false;
        }
        compact_pack_ns_ += static_cast<uint64_t>(
            std::chrono::duration_cast<std::chrono::nanoseconds>(
                std::chrono::steady_clock::now() - pack_started).count());
        if (requested_mask == 0 ||
            (requested_mask & ~uploaded_mask) != 0) {
            invalid = true;
            if (err) {
                *err = "compact sparse requested mask is not resident";
            }
            return false;
        }
        if (down_slab_bytes != down_slab_row_bytes *
                static_cast<size_t>(spec.output_dim) ||
            gate_slab_bytes * slab_count != ggml_nbytes(entry->gate) ||
            up_slab_bytes * slab_count != ggml_nbytes(entry->up) ||
            down_slab_bytes * slab_count != ggml_nbytes(entry->down)) {
            invalid = true;
            if (err) *err = "compact sparse tensor extents disagree with wire";
            return false;
        }
        ++compact_layouts_;
        auto * wire = static_cast<uint8_t *>(host_staging);
        std::array<int32_t, kSlabCount> natural_to_compact;
        natural_to_compact.fill(-1);
        const auto * naturals = reinterpret_cast<const uint16_t *>(wire);
        for (size_t slot = 0; slot < slab_count; ++slot) {
            if ((requested_mask & (1u << naturals[slot])) != 0) {
                natural_to_compact[naturals[slot]] =
                    static_cast<int32_t>(slot);
            }
        }
        std::array<bool, kSlabCount> compact_slot_seen{};
        bool any_requested = false;
        for (const int32_t slot : natural_to_compact) {
            if (slot == -1) continue;
            if (slot < 0 || static_cast<size_t>(slot) >= slab_count ||
                compact_slot_seen[static_cast<size_t>(slot)]) {
                invalid = true;
                if (err) *err = "compact sparse natural map is invalid";
                return false;
            }
            compact_slot_seen[static_cast<size_t>(slot)] = true;
            any_requested = true;
        }
        if (!any_requested) {
            invalid = true;
            if (err) *err = "compact sparse natural map is empty";
            return false;
        }
        ggml_status status = GGML_STATUS_FAILED;
        const auto graph_started = std::chrono::steady_clock::now();
        if (compact_async_active_) {
            auto * input_staging = direct_component_wire
                ? static_cast<uint8_t *>(async_slot->host_staging)
                : wire + layout.total_bytes;
            auto * map_staging = input_staging +
                (upload_host_input ? input_bytes : 0);
            if (upload_host_input) {
                std::memcpy(input_staging, input_data, input_bytes);
            }
            std::memcpy(
                map_staging, natural_to_compact.data(),
                sizeof(natural_to_compact));
            if (compact_async_jobs_ == 0) {
                compact_async_started_ = graph_started;
            }
            if (upload_device_input) {
                if (compact_async_device_input_->type != GGML_TYPE_F32 ||
                    compact_async_device_input_->ne[0] != spec.input_dim ||
                    compact_async_device_input_->ne[1] != 1 ||
                    ggml_nbytes(compact_async_device_input_) != input_bytes) {
                    if (err) *err =
                        "P45 compact device input shape is incompatible";
                    return false;
                }
                ggml_backend_tensor_copy_async(
                    backend, backend, compact_async_device_input_,
                    entry->input);
                compact_async_input_ready_ = true;
                ++compact_async_stats_.input_d2d_copies;
                compact_async_stats_.input_d2d_bytes += input_bytes;
            } else if (upload_host_input) {
                ggml_backend_tensor_set_async(
                    backend, entry->input, input_staging, 0, input_bytes);
                compact_async_input_source_ = input_data;
                compact_async_input_ready_ = true;
            }
            ggml_backend_tensor_set_async(
                backend, entry->gate, wire + layout.gate_offset, 0,
                gate_slab_bytes * slab_count);
            ggml_backend_tensor_set_async(
                backend, entry->up, wire + layout.up_offset, 0,
                up_slab_bytes * slab_count);
            ggml_backend_tensor_set_async(
                backend, entry->down, wire + layout.down_offset, 0,
                down_slab_bytes * slab_count);
            ggml_backend_tensor_set_async(
                backend, entry->natural_to_compact, map_staging, 0,
                sizeof(natural_to_compact));
            status = ggml_backend_graph_compute_async(backend, entry->graph);
            compact_async_stats_.submit_ns += static_cast<uint64_t>(
                std::chrono::duration_cast<std::chrono::nanoseconds>(
                    std::chrono::steady_clock::now() - graph_started).count());
            if (status == GGML_STATUS_SUCCESS) {
                ++compact_async_jobs_;
                ++compact_async_stats_.jobs;
                compact_async_stats_.h2d_calls +=
                    upload_host_input ? 5 : 4;
                compact_async_stats_.h2d_bytes +=
                    (upload_host_input ? input_bytes : 0) +
                    gate_slab_bytes * slab_count +
                    up_slab_bytes * slab_count +
                    down_slab_bytes * slab_count +
                    sizeof(natural_to_compact);
                ++compact_async_stats_.graph_enqueues;
                compact_async_stats_.max_inflight = std::max<uint64_t>(
                    compact_async_stats_.max_inflight,
                    static_cast<uint64_t>(compact_async_jobs_));
            }
        } else {
            ggml_backend_tensor_set(
                entry->input, input_data, 0, input_bytes);
            ggml_backend_tensor_set(
                entry->gate, wire + layout.gate_offset, 0,
                gate_slab_bytes * slab_count);
            ggml_backend_tensor_set(
                entry->up, wire + layout.up_offset, 0,
                up_slab_bytes * slab_count);
            ggml_backend_tensor_set(
                entry->down, wire + layout.down_offset, 0,
                down_slab_bytes * slab_count);
            ggml_backend_tensor_set(
                entry->natural_to_compact, natural_to_compact.data(), 0,
                sizeof(natural_to_compact));
            status = ggml_backend_graph_compute(backend, entry->graph);
            expert_graph_ns_ += static_cast<uint64_t>(
                std::chrono::duration_cast<std::chrono::nanoseconds>(
                    std::chrono::steady_clock::now() - graph_started).count());
        }
        authoritative_h2d_bytes +=
            slab_count * (gate_slab_bytes + up_slab_bytes + down_slab_bytes);
        metadata_h2d_bytes +=
            (upload_host_input ? input_bytes : 0) +
            sizeof(natural_to_compact);
        ++compact_uploads_;
        if (status != GGML_STATUS_SUCCESS) {
            if (err) *err = "compact sparse executor graph failed";
            return false;
        }
        ++compact_gate_stages_;
        ++compact_up_stages_;
        ++compact_situ_stages_;
        ++compact_down_stages_;
        device_output = entry->output;
        return true;
#endif
    }

private:
#if defined(DFLASH27B_BACKEND_CUDA) || defined(DFLASH27B_BACKEND_HIP)
    struct Entry {
        ~Entry() {
            int previous_device = -1;
            const bool restore_device = runtime_device >= 0 &&
                cudaGetDevice(&previous_device) == cudaSuccess &&
                previous_device != runtime_device &&
                cudaSetDevice(runtime_device) == cudaSuccess;
            if (compact_host_staging) cudaFreeHost(compact_host_staging);
            if (compact_staging) cudaFree(compact_staging);
            if (allocator) ggml_gallocr_free(allocator);
            if (context) ggml_free(context);
            if (restore_device) (void) cudaSetDevice(previous_device);
        }

        ggml_backend_t backend = nullptr;
        int runtime_device = -1;
        MoeStreamExpertSpec spec{};
        bool needs_mask = false;
        ggml_context * context = nullptr;
        ggml_gallocr_t allocator = nullptr;
        ggml_cgraph * graph = nullptr;
        ggml_tensor * input = nullptr;
        ggml_tensor * gate = nullptr;
        ggml_tensor * up = nullptr;
        ggml_tensor * down = nullptr;
        ggml_tensor * activation_mask = nullptr;
        ggml_tensor * output = nullptr;
        ggml_backend_buffer_t gate_owned_buffer = nullptr;
        ggml_backend_buffer_t up_owned_buffer = nullptr;
        ggml_backend_buffer_t down_owned_buffer = nullptr;
        void * gate_owned_data = nullptr;
        void * up_owned_data = nullptr;
        void * down_owned_data = nullptr;
        size_t gate_offset = 0;
        size_t up_offset = 0;
        size_t down_offset = 0;
        size_t weight_bytes = 0;
        void * compact_staging = nullptr;
        void * compact_host_staging = nullptr;
        size_t compact_capacity = 0;
    };

    struct CompactEntry {
        ~CompactEntry() {
            int previous_device = -1;
            const bool restore_device = runtime_device >= 0 &&
                cudaGetDevice(&previous_device) == cudaSuccess &&
                previous_device != runtime_device &&
                cudaSetDevice(runtime_device) == cudaSuccess;
            if (host_staging) cudaFreeHost(host_staging);
            if (allocator) ggml_gallocr_free(allocator);
            if (context) ggml_free(context);
            if (restore_device) (void) cudaSetDevice(previous_device);
        }

        ggml_backend_t backend = nullptr;
        int runtime_device = -1;
        MoeStreamExpertSpec spec{};
        int slab_count = 0;
        ggml_context * context = nullptr;
        ggml_gallocr_t allocator = nullptr;
        ggml_cgraph * graph = nullptr;
        ggml_tensor * input = nullptr;
        ggml_tensor * gate = nullptr;
        ggml_tensor * up = nullptr;
        ggml_tensor * down = nullptr;
        ggml_tensor * natural_to_compact = nullptr;
        ggml_tensor * output = nullptr;
        void * host_staging = nullptr;
        size_t host_capacity = 0;
    };

    struct CompactUnionEntry {
        ~CompactUnionEntry() {
            int previous_device = -1;
            const bool restore_device = runtime_device >= 0 &&
                cudaGetDevice(&previous_device) == cudaSuccess &&
                previous_device != runtime_device &&
                cudaSetDevice(runtime_device) == cudaSuccess;
            if (backend) ggml_backend_synchronize(backend);
            if (allocator) ggml_gallocr_free(allocator);
            if (weight_buffer) ggml_backend_buffer_free(weight_buffer);
            if (context) ggml_free(context);
            if (restore_device) (void) cudaSetDevice(previous_device);
        }

        ggml_backend_t backend = nullptr;
        int runtime_device = -1;
        MoeStreamExpertSpec spec{};
        int slab_count = 0;
        int graph_width = 0;
        ggml_context * context = nullptr;
        ggml_gallocr_t allocator = nullptr;
        ggml_cgraph * graph = nullptr;
        ggml_backend_buffer_t weight_buffer = nullptr;
        ggml_tensor * input = nullptr;
        ggml_tensor * gate = nullptr;
        ggml_tensor * up = nullptr;
        ggml_tensor * down = nullptr;
        std::array<ggml_tensor *, 8> maps{};
        std::array<ggml_tensor *, 8> outputs{};
    };

    struct CompactAsyncSlot {
        ~CompactAsyncSlot() {
            if (host_staging) cudaFreeHost(host_staging);
        }

        void * host_staging = nullptr;
        size_t host_capacity = 0;
    };

    static bool ensure_compact_async_slot(
            CompactAsyncSlot & slot, size_t bytes, std::string * err) {
        if (slot.host_capacity >= bytes) return true;
        if (slot.host_staging) cudaFreeHost(slot.host_staging);
        slot.host_staging = nullptr;
        slot.host_capacity = 0;
        if (cudaHostAlloc(
                &slot.host_staging, bytes, cudaHostAllocDefault) !=
                cudaSuccess) {
            if (err) *err = "P45 async compact host slot allocation failed";
            return false;
        }
        slot.host_capacity = bytes;
        return true;
    }

    void reset_compact_async_batch() {
        compact_async_backend_ = nullptr;
        compact_async_limit_ = 0;
        compact_async_jobs_ = 0;
        compact_async_active_ = false;
        compact_async_input_source_ = nullptr;
        compact_async_device_input_ = nullptr;
        compact_async_input_ready_ = false;
        compact_async_started_ = {};
    }

    CompactEntry * find_compact(
            ggml_backend_t backend, const MoeStreamExpertSpec & spec,
            int slab_count) {
        for (const std::unique_ptr<CompactEntry> & entry : compact_entries_) {
            if (entry->backend == backend && entry->slab_count == slab_count &&
                same_moe_stream_expert_spec(entry->spec, spec)) {
                return entry.get();
            }
        }
        return nullptr;
    }

    CompactEntry * create_compact(
            ggml_backend_t backend, const MoeStreamExpertSpec & spec,
            int slab_count, std::string * err) {
        if (slab_count <= 0 || slab_count > kSlabCount ||
            spec.intermediate_dim != kSlabCount * kSlabSize) {
            if (err) *err = "compact sparse graph geometry is invalid";
            return nullptr;
        }
        auto entry = std::make_unique<CompactEntry>();
        entry->backend = backend;
        (void) cudaGetDevice(&entry->runtime_device);
        entry->spec = spec;
        entry->slab_count = slab_count;
        ggml_init_params parameters{};
        parameters.mem_size = 32 * 1024 * 1024;
        parameters.no_alloc = true;
        entry->context = ggml_init(parameters);
        if (!entry->context) {
            if (err) *err = "compact sparse ggml_init failed";
            return nullptr;
        }
        if (compact_shared_input_) {
            if (compact_shared_input_backend_ != backend ||
                compact_shared_input_->type != GGML_TYPE_F32 ||
                compact_shared_input_->ne[0] != spec.input_dim ||
                compact_shared_input_->ne[1] != 1) {
                if (err) *err = "compact sparse shared input is incompatible";
                return nullptr;
            }
            entry->input = compact_shared_input_;
        } else {
            entry->input = ggml_new_tensor_2d(
                entry->context, GGML_TYPE_F32, spec.input_dim, 1);
        }
        entry->gate = ggml_new_tensor_2d(
            entry->context, spec.gate_type, spec.input_dim,
            static_cast<int64_t>(slab_count) * kSlabSize);
        entry->up = ggml_new_tensor_2d(
            entry->context, spec.up_type, spec.input_dim,
            static_cast<int64_t>(slab_count) * kSlabSize);
        entry->down = ggml_new_tensor_4d(
            entry->context, spec.down_type, kSlabSize, spec.output_dim,
            slab_count, 1);
        entry->natural_to_compact = ggml_new_tensor_1d(
            entry->context, GGML_TYPE_I32, kSlabCount);
        ggml_set_input(entry->input);
        ggml_set_input(entry->gate);
        ggml_set_input(entry->up);
        ggml_set_input(entry->down);
        ggml_set_input(entry->natural_to_compact);
        ggml_tensor * gate_value = probe_scale_tensor(
            entry->context,
            ggml_mul_mat(entry->context, entry->gate, entry->input),
            spec.gate_scale);
        ggml_tensor * up_value = probe_scale_tensor(
            entry->context,
            ggml_mul_mat(entry->context, entry->up, entry->input),
            spec.up_scale);
        ggml_tensor * activated = probe_gated_activation(
            entry->context, spec, gate_value, up_value);
        ggml_tensor * x_blocks = ggml_reshape_2d(
            entry->context, activated, kSlabSize, slab_count);
        entry->output = probe_scale_tensor(
            entry->context,
            ggml_mul_mat_sparse_k_blocks(
                entry->context, entry->down, x_blocks,
                entry->natural_to_compact, spec.intermediate_dim),
            spec.down_scale);
        entry->graph = ggml_new_graph_custom(entry->context, 512, false);
        ggml_set_output(entry->output);
        ggml_build_forward_expand(entry->graph, entry->output);
        entry->allocator = ggml_gallocr_new(
            ggml_backend_get_default_buffer_type(backend));
        if (!entry->allocator ||
            !ggml_gallocr_alloc_graph(entry->allocator, entry->graph)) {
            if (err) *err = "compact sparse graph allocation failed";
            return nullptr;
        }
        if (!compact_shared_input_) {
            compact_shared_input_ = entry->input;
            compact_shared_input_backend_ = backend;
        }
        compact_entries_.push_back(std::move(entry));
        return compact_entries_.back().get();
    }

    CompactUnionEntry * find_compact_union(
            ggml_backend_t backend, const MoeStreamExpertSpec & spec,
            int slab_count, int graph_width) {
        for (const std::unique_ptr<CompactUnionEntry> & entry :
                compact_union_entries_) {
            if (entry->backend == backend &&
                entry->slab_count == slab_count &&
                entry->graph_width == graph_width &&
                same_moe_stream_expert_spec(entry->spec, spec)) {
                return entry.get();
            }
        }
        return nullptr;
    }

    CompactUnionEntry * create_compact_union(
            ggml_backend_t backend, const MoeStreamExpertSpec & spec,
            int slab_count, int graph_width, std::string * err) {
        auto entry = std::make_unique<CompactUnionEntry>();
        entry->backend = backend;
        (void) cudaGetDevice(&entry->runtime_device);
        entry->spec = spec;
        entry->slab_count = slab_count;
        entry->graph_width = graph_width;
        ggml_init_params parameters{};
        parameters.mem_size = 32 * 1024 * 1024;
        parameters.no_alloc = true;
        entry->context = ggml_init(parameters);
        if (!entry->context) {
            if (err) *err = "compact union ggml_init failed";
            return nullptr;
        }
        entry->input = ggml_new_tensor_2d(
            entry->context, GGML_TYPE_F32, spec.input_dim, graph_width);
        entry->gate = ggml_new_tensor_2d(
            entry->context, spec.gate_type, spec.input_dim,
            static_cast<int64_t>(slab_count) * kSlabSize);
        entry->up = ggml_new_tensor_2d(
            entry->context, spec.up_type, spec.input_dim,
            static_cast<int64_t>(slab_count) * kSlabSize);
        entry->down = ggml_new_tensor_4d(
            entry->context, spec.down_type, kSlabSize, spec.output_dim,
            slab_count, 1);
        ggml_set_input(entry->input);
        ggml_set_input(entry->gate);
        ggml_set_input(entry->up);
        ggml_set_input(entry->down);

        const ggml_backend_buffer_type_t buft =
            ggml_backend_get_default_buffer_type(backend);
        const size_t alignment = std::max<size_t>(
            256, ggml_backend_buft_get_alignment(buft));
        size_t cursor = 0;
        const auto reserve = [&](ggml_tensor * tensor) {
            cursor = ((cursor + alignment - 1) / alignment) * alignment;
            const size_t offset = cursor;
            cursor += ggml_backend_buft_get_alloc_size(buft, tensor);
            return offset;
        };
        const size_t gate_offset = reserve(entry->gate);
        const size_t up_offset = reserve(entry->up);
        const size_t down_offset = reserve(entry->down);
        cursor = ((cursor + alignment - 1) / alignment) * alignment;
        entry->weight_buffer =
            ggml_backend_buft_alloc_buffer(buft, cursor);
        if (!entry->weight_buffer) {
            if (err) *err = "compact union weight allocation failed";
            return nullptr;
        }
        ggml_backend_buffer_set_usage(
            entry->weight_buffer, GGML_BACKEND_BUFFER_USAGE_WEIGHTS);
        auto * base = static_cast<uint8_t *>(
            ggml_backend_buffer_get_base(entry->weight_buffer));
        if (ggml_backend_tensor_alloc(
                entry->weight_buffer, entry->gate, base + gate_offset) !=
                GGML_STATUS_SUCCESS ||
            ggml_backend_tensor_alloc(
                entry->weight_buffer, entry->up, base + up_offset) !=
                GGML_STATUS_SUCCESS ||
            ggml_backend_tensor_alloc(
                entry->weight_buffer, entry->down, base + down_offset) !=
                GGML_STATUS_SUCCESS) {
            if (err) *err = "compact union weight binding failed";
            return nullptr;
        }

        ggml_tensor * gate_value = probe_scale_tensor(
            entry->context,
            ggml_mul_mat(entry->context, entry->gate, entry->input),
            spec.gate_scale);
        ggml_tensor * up_value = probe_scale_tensor(
            entry->context,
            ggml_mul_mat(entry->context, entry->up, entry->input),
            spec.up_scale);
        ggml_tensor * activated = probe_gated_activation(
            entry->context, spec, gate_value, up_value);
        for (int lane = 0; lane < graph_width; ++lane) {
            entry->maps[static_cast<size_t>(lane)] = ggml_new_tensor_1d(
                entry->context, GGML_TYPE_I32, kSlabCount);
            ggml_set_input(entry->maps[static_cast<size_t>(lane)]);
            ggml_tensor * activation_row = ggml_view_1d(
                entry->context, activated, activated->ne[0],
                static_cast<size_t>(lane) * activated->nb[1]);
            ggml_tensor * blocks = ggml_reshape_2d(
                entry->context, activation_row, kSlabSize, slab_count);
            entry->outputs[static_cast<size_t>(lane)] = probe_scale_tensor(
                entry->context,
                ggml_mul_mat_sparse_k_blocks(
                    entry->context, entry->down, blocks,
                    entry->maps[static_cast<size_t>(lane)],
                    spec.intermediate_dim),
                spec.down_scale);
        }
        entry->graph = ggml_new_graph_custom(entry->context, 1024, false);
        for (int lane = 0; lane < graph_width; ++lane) {
            ggml_set_output(entry->outputs[static_cast<size_t>(lane)]);
            ggml_build_forward_expand(
                entry->graph, entry->outputs[static_cast<size_t>(lane)]);
        }
        entry->allocator = ggml_gallocr_new(buft);
        if (!entry->allocator ||
            !ggml_gallocr_alloc_graph(entry->allocator, entry->graph)) {
            if (err) *err = "compact union graph allocation failed";
            return nullptr;
        }
        compact_union_entries_.push_back(std::move(entry));
        return compact_union_entries_.back().get();
    }

    static bool ensure_compact_host_staging(
            CompactEntry & entry, size_t bytes, std::string * err) {
        if (entry.host_capacity >= bytes) return true;
        if (entry.host_staging) cudaFreeHost(entry.host_staging);
        entry.host_staging = nullptr;
        entry.host_capacity = 0;
        if (cudaHostAlloc(
                &entry.host_staging, bytes, cudaHostAllocDefault) !=
                cudaSuccess) {
            if (err) *err = "compact sparse pinned staging allocation failed";
            return false;
        }
        entry.host_capacity = bytes;
        return true;
    }

    Entry * find(
            ggml_backend_t backend, const MoeStreamExpertSpec & spec,
            bool needs_mask) {
        for (const std::unique_ptr<Entry> & entry : entries_) {
            if (entry->backend == backend &&
                entry->needs_mask == needs_mask &&
                same_moe_stream_expert_spec(entry->spec, spec)) {
                return entry.get();
            }
        }
        return nullptr;
    }

    Entry * create(
            ggml_backend_t backend, const MoeStreamExpertSpec & spec,
            bool needs_mask, std::string * err) {
        auto entry = std::make_unique<Entry>();
        entry->backend = backend;
        (void) cudaGetDevice(&entry->runtime_device);
        entry->spec = spec;
        entry->needs_mask = needs_mask;
        ggml_init_params parameters{};
        parameters.mem_size = 32 * 1024 * 1024;
        parameters.no_alloc = true;
        entry->context = ggml_init(parameters);
        if (!entry->context) {
            if (err) *err = "P23 persistent sparse ggml_init failed";
            return nullptr;
        }
        entry->input = ggml_new_tensor_2d(
            entry->context, GGML_TYPE_F32, spec.input_dim, 1);
        entry->gate = ggml_new_tensor_2d(
            entry->context, spec.gate_type,
            spec.input_dim, spec.intermediate_dim);
        entry->up = ggml_new_tensor_2d(
            entry->context, spec.up_type,
            spec.input_dim, spec.intermediate_dim);
        entry->down = ggml_new_tensor_2d(
            entry->context, spec.down_type,
            spec.intermediate_dim, spec.output_dim);
        entry->activation_mask = needs_mask
            ? ggml_new_tensor_1d(
                entry->context, GGML_TYPE_F32, spec.intermediate_dim)
            : nullptr;
        ggml_set_input(entry->input);
        ggml_set_input(entry->gate);
        ggml_set_input(entry->up);
        ggml_set_input(entry->down);
        if (entry->activation_mask) {
            ggml_set_input(entry->activation_mask);
        }
        ggml_tensor * gate_value = probe_scale_tensor(
            entry->context,
            ggml_mul_mat(entry->context, entry->gate, entry->input),
            spec.gate_scale);
        ggml_tensor * up_value = probe_scale_tensor(
            entry->context,
            ggml_mul_mat(entry->context, entry->up, entry->input),
            spec.up_scale);
        ggml_tensor * activated = probe_gated_activation(
            entry->context, spec, gate_value, up_value);
        if (entry->activation_mask) {
            activated = ggml_mul(
                entry->context, activated, entry->activation_mask);
        }
        entry->output = probe_scale_tensor(
            entry->context,
            ggml_mul_mat(entry->context, entry->down, activated),
            spec.down_scale);
        entry->graph = ggml_new_graph_custom(entry->context, 512, false);
        ggml_set_output(entry->output);
        ggml_build_forward_expand(entry->graph, entry->output);
        entry->allocator = ggml_gallocr_new(
            ggml_backend_get_default_buffer_type(backend));
        if (!entry->allocator ||
            !ggml_gallocr_alloc_graph(entry->allocator, entry->graph)) {
            if (err) *err =
                "P23 persistent sparse graph allocation failed";
            return nullptr;
        }
        const ggml_backend_buffer_type_t buft =
            ggml_backend_get_default_buffer_type(backend);
        const size_t alignment = std::max<size_t>(
            256, ggml_backend_buft_get_alignment(buft));
        const auto place = [&](ggml_tensor * tensor, size_t & offset,
                               size_t & cursor) {
            offset = (cursor + alignment - 1) & ~(alignment - 1);
            cursor = offset + ggml_backend_buft_get_alloc_size(buft, tensor);
        };
        size_t cursor = 0;
        place(entry->gate, entry->gate_offset, cursor);
        place(entry->up, entry->up_offset, cursor);
        place(entry->down, entry->down_offset, cursor);
        entry->weight_bytes =
            (cursor + alignment - 1) & ~(alignment - 1);
        entry->gate_owned_buffer = entry->gate->buffer;
        entry->up_owned_buffer = entry->up->buffer;
        entry->down_owned_buffer = entry->down->buffer;
        entry->gate_owned_data = entry->gate->data;
        entry->up_owned_data = entry->up->data;
        entry->down_owned_data = entry->down->data;
        entries_.push_back(std::move(entry));
        return entries_.back().get();
    }

    static bool ensure_compact_staging(
            Entry & entry, size_t bytes, bool pinned_host,
            std::string * err) {
        if (entry.compact_capacity >= bytes &&
            (!pinned_host || entry.compact_host_staging)) return true;
        if (entry.compact_staging) {
            cudaFree(entry.compact_staging);
            entry.compact_staging = nullptr;
        }
        if (entry.compact_host_staging) {
            cudaFreeHost(entry.compact_host_staging);
            entry.compact_host_staging = nullptr;
        }
        entry.compact_capacity = 0;
        if (cudaMalloc(&entry.compact_staging, bytes) != cudaSuccess) {
            if (err) *err = "P25 compact staging allocation failed";
            return false;
        }
        if (pinned_host && cudaHostAlloc(
                &entry.compact_host_staging, bytes,
                cudaHostAllocDefault) != cudaSuccess) {
            cudaFree(entry.compact_staging);
            entry.compact_staging = nullptr;
            if (err) *err = "P26 pinned compact staging allocation failed";
            return false;
        }
        entry.compact_capacity = bytes;
        return true;
    }

    std::vector<std::unique_ptr<Entry>> entries_;
    std::vector<std::unique_ptr<CompactEntry>> compact_entries_;
    std::vector<std::unique_ptr<CompactUnionEntry>> compact_union_entries_;
    ggml_tensor * compact_shared_input_ = nullptr;
    ggml_backend_t compact_shared_input_backend_ = nullptr;
    std::array<CompactAsyncSlot, kNativeTopK> compact_async_slots_{};
    ggml_backend_t compact_async_backend_ = nullptr;
    int compact_async_limit_ = 0;
    int compact_async_jobs_ = 0;
    bool compact_async_active_ = false;
    const float * compact_async_input_source_ = nullptr;
    const ggml_tensor * compact_async_device_input_ = nullptr;
    bool compact_async_input_ready_ = false;
    std::chrono::steady_clock::time_point compact_async_started_{};
#endif
    uint64_t compact_pack_ns_ = 0;
    uint64_t compact_scatter_ns_ = 0;
    uint64_t expert_graph_ns_ = 0;
    uint64_t expert_readback_ns_ = 0;
    uint64_t compact_layouts_ = 0;
    uint64_t compact_uploads_ = 0;
    uint64_t compact_gate_stages_ = 0;
    uint64_t compact_up_stages_ = 0;
    uint64_t compact_situ_stages_ = 0;
    uint64_t compact_down_stages_ = 0;
    CompactAsyncStats compact_async_stats_{};
};

// Every routed layer owns independent, provenance-checked calibration cards.
// Any absent, malformed, or provenance-free layer stays exact.  Within a valid
// layer, experts below the exporter's minimum-hit threshold also stay exact.
// The requested budget is 96 slab records, while measured traffic separately
// reports selected sidecar bytes and exact-fallback expert bytes.
class CalibratedAllLayerProvider final :
        public KimiK3RoutedOutputProvider,
        public KimiK3RoutedPrefillService {
public:
    ~CalibratedAllLayerProvider() override {
        finish_metrics();
    }

    bool init(ggml_backend_t backend, const std::string & aux_directory,
              const std::string & sidecar_directory,
              bool ordered_device_join,
              bool async_compact_queue,
              bool p40_wide_async_join,
              const char * metrics_path,
              std::string * err) {
        if (!backend || aux_directory.empty() || sidecar_directory.empty()) {
            if (err) *err =
                "calibrated96 needs a compute backend, aux directory, and sidecars";
            return false;
        }
        backend_ = backend;
        budget_ = 96;
        if (const char * authoritative =
                std::getenv("DFLASH_KIMI_SIDECAR_AUTHORITATIVE")) {
            if (std::strcmp(authoritative, "1") == 0) {
                sidecar_authoritative_ = true;
            } else if (std::strcmp(authoritative, "0") != 0 &&
                       *authoritative) {
                if (err) *err =
                    "DFLASH_KIMI_SIDECAR_AUTHORITATIVE must be 0 or 1";
                return false;
            }
        }
        const char * raw_budget =
            std::getenv("DFLASH_KIMI_P20_SLAB_BUDGET");
        const char * raw_budget_table =
            std::getenv("DFLASH_KIMI_H22_LAYER_BUDGETS");
        if (raw_budget_table && *raw_budget_table) {
            if (raw_budget && *raw_budget) {
                if (err) *err =
                    "DFLASH_KIMI_H22_LAYER_BUDGETS is incompatible with "
                    "DFLASH_KIMI_P20_SLAB_BUDGET";
                return false;
            }
            if (!parse_kimi_k3_layer_budget_table(
                    raw_budget_table, layer_budgets_, err)) {
                return false;
            }
            layer_budget_path_ = raw_budget_table;
        } else if (raw_budget && *raw_budget) {
            char * end = nullptr;
            const long parsed = std::strtol(raw_budget, &end, 10);
            if (end == raw_budget || *end != '\0' ||
                (parsed != 96 && parsed != 192)) {
                if (err) *err =
                    "DFLASH_KIMI_P20_SLAB_BUDGET must be 96 or 192";
                return false;
            }
            budget_ = static_cast<int>(parsed);
        }
        if (const char * layout =
                std::getenv("DFLASH_KIMI_P20_PHYSICAL_LAYOUT")) {
            if (std::strcmp(layout, "reference") == 0 || *layout == '\0') {
                sparse_workspace_ = SparseWorkspace::HostRecomposed;
            } else if (std::strcmp(layout, "scratch") == 0) {
                sparse_workspace_ = SparseWorkspace::TransientDevice;
            } else {
                if (err) *err =
                    "DFLASH_KIMI_P20_PHYSICAL_LAYOUT must be reference or scratch";
                return false;
            }
        }
        if (const char * io_backend =
                std::getenv("DFLASH_KIMI_P20_IO_BACKEND")) {
            if (std::strcmp(io_backend, "current") == 0 ||
                *io_backend == '\0') {
                sparse_delivery_ = KimiK3SparseDeliveryPolicy::BufferedSlabs;
            } else if (std::strcmp(io_backend, "direct-pread") == 0) {
                sparse_delivery_ = KimiK3SparseDeliveryPolicy::DirectSlabs;
            } else {
                if (err) *err =
                    "DFLASH_KIMI_P20_IO_BACKEND must be current or direct-pread";
                return false;
            }
        }
        if (direct_reads() && !sparse_device()) {
            if (err) *err =
                "P20 direct-pread currently requires the scratch layout";
            return false;
        }
        if (const char * cache =
                std::getenv("DFLASH_KIMI_P30_HOST_CACHE_MB")) {
            int cache_mib = 0;
            if (!parse_positive_int(cache, cache_mib) || cache_mib > 16384) {
                if (err) *err =
                    "DFLASH_KIMI_P30_HOST_CACHE_MB must be in 1..16384";
                return false;
            }
            if (!direct_reads()) {
                if (err) *err =
                    "P30 host cache requires P20 direct-pread";
                return false;
            }
            read_cache_.set_capacity(
                static_cast<size_t>(cache_mib) * 1024 * 1024);
        }
        bool enabled = false;
        if (!parse_binary_flag(std::getenv(
                "DFLASH_KIMI_P23_PERSISTENT_SCRATCH"), enabled)) {
            if (err) *err =
                "DFLASH_KIMI_P23_PERSISTENT_SCRATCH must be 0 or 1";
            return false;
        }
        if (enabled && !sparse_device()) {
            if (err) *err =
                "P23 persistent scratch requires the scratch layout";
            return false;
        }
        if (enabled) sparse_workspace_ = SparseWorkspace::PersistentDevice;
        if (!parse_binary_flag(std::getenv(
                "DFLASH_KIMI_P25_COMPACT_UPLOAD"), enabled)) {
            if (err) *err =
                "DFLASH_KIMI_P25_COMPACT_UPLOAD must be 0 or 1";
            return false;
        }
        if (enabled && (sparse_workspace_ != SparseWorkspace::PersistentDevice ||
                sparse_delivery_ != KimiK3SparseDeliveryPolicy::DirectSlabs)) {
            if (err) *err =
                "P25 compact upload requires persistent scratch and direct-pread";
            return false;
        }
        if (enabled) sparse_delivery_ = KimiK3SparseDeliveryPolicy::CompactPageable;
        if (!parse_binary_flag(std::getenv(
                "DFLASH_KIMI_P26_PINNED_COMPACT"), enabled)) {
            if (err) *err =
                "DFLASH_KIMI_P26_PINNED_COMPACT must be 0 or 1";
            return false;
        }
        if (enabled && sparse_delivery_ != KimiK3SparseDeliveryPolicy::CompactPageable) {
            if (err) *err = "P26 pinned compact staging requires compact upload";
            return false;
        }
        if (enabled) sparse_delivery_ = KimiK3SparseDeliveryPolicy::CompactPinned;
        if (!parse_binary_flag(std::getenv(
                "DFLASH_KIMI_P27_DIRECT_PINNED_COMPACT"), enabled)) {
            if (err) *err =
                "DFLASH_KIMI_P27_DIRECT_PINNED_COMPACT must be 0 or 1";
            return false;
        }
        if (enabled && sparse_delivery_ != KimiK3SparseDeliveryPolicy::CompactPinned) {
            if (err) *err = "P27 direct pinned compact requires P26 pinned compact";
            return false;
        }
        if (enabled) sparse_delivery_ = KimiK3SparseDeliveryPolicy::DirectPinnedCompact;
        if (!parse_binary_flag(std::getenv(
                "DFLASH_KIMI_P40_DEVICE_VARIANT_CACHE"), enabled)) {
            if (err) *err =
                "DFLASH_KIMI_P40_DEVICE_VARIANT_CACHE must be 0 or 1";
            return false;
        }
        if (enabled && (sparse_workspace_ != SparseWorkspace::PersistentDevice ||
                !direct_reads())) {
            if (err) *err =
                "P40 device variant cache requires persistent direct sparse execution";
            return false;
        }
        device_variant_cache_ = enabled;
        if (device_variant_cache_) {
            int cache_mib = 0;
            if (!parse_positive_int(std::getenv(
                    "DFLASH_MOE_NVME_DEVICE_CACHE_MB"), cache_mib) ||
                (cache_mib != 8192 && cache_mib != 16384)) {
                if (err) {
                    *err = "P40 requires DFLASH_MOE_NVME_DEVICE_CACHE_MB=8192 or 16384";
                }
                return false;
            }
            p40_requested_device_bytes_ =
                static_cast<size_t>(cache_mib) * 1024 * 1024;
        }
        if (!parse_binary_flag(std::getenv(
                "DFLASH_KIMI_P40_LAYER_EPOCH"), enabled)) {
            if (err) *err = "DFLASH_KIMI_P40_LAYER_EPOCH must be 0 or 1";
            return false;
        }
        if (enabled && !device_variant_cache_) {
            if (err) *err = "P40 layer epoch requires the P40 device cache";
            return false;
        }
        p40_layer_epoch_ = enabled;
        if (!parse_binary_flag(std::getenv(
                "DFLASH_KIMI_P41_COMPACT_EXECUTOR"), enabled)) {
            if (err) *err =
                "DFLASH_KIMI_P41_COMPACT_EXECUTOR must be 0 or 1";
            return false;
        }
        if (enabled && (sparse_workspace_ != SparseWorkspace::PersistentDevice ||
                sparse_delivery_ !=
                    KimiK3SparseDeliveryPolicy::DirectPinnedCompact)) {
            if (err) {
                *err = "P41 compact executor requires the persistent P27 path";
            }
            return false;
        }
        compact_executor_ = enabled;
        if (ordered_device_join &&
                (!compact_executor_ || !sidecar_authoritative_)) {
            if (err) {
                *err = "P42 ordered join requires P41, authoritative sidecars, "
                    "all routed layers, and calibrated96 slab mode";
            }
            return false;
        }
        if (ordered_device_join) {
#if defined(DFLASH27B_BACKEND_CUDA) || defined(DFLASH27B_BACKEND_HIP)
            BackendDeviceScope scope;
            if (!scope.enter(backend_, err)) return false;
#if defined(DFLASH27B_BACKEND_HIP)
            if (!p42_qualified_device(scope.device())) {
                if (err) {
                    *err = "P42 ordered join is qualified only on gfx1151";
                }
                return false;
            }
#else
            if (scope.device() != 1) {
                if (err) *err = "P42 ordered join is qualified only on GPU1";
                return false;
            }
#endif
#else
            if (err) *err = "P42 ordered join requires a GPU backend";
            return false;
#endif
        }
        ordered_device_join_ = ordered_device_join;
        if (async_compact_queue && !ordered_device_join_) {
            if (err) *err = "P45 async compact queue requires P42 ordered join";
            return false;
        }
        async_compact_queue_ = async_compact_queue;
        if (p40_wide_async_join &&
            (!device_variant_cache_ || !p40_layer_epoch_ ||
             sparse_workspace_ != SparseWorkspace::PersistentDevice ||
             sparse_delivery_ !=
                KimiK3SparseDeliveryPolicy::DirectPinnedCompact ||
             !compact_executor_ || !ordered_device_join_ ||
             !async_compact_queue_ || !sidecar_authoritative_)) {
            if (err) {
                *err = "P40 wide async join requires P40, layer epoch, "
                    "persistent P27, P41, P42, P45, and authoritative sidecars";
            }
            return false;
        }
        p40_wide_async_join_ = p40_wide_async_join;
        if (const char * trace =
                std::getenv("DFLASH_KIMI_P40_CACHE_TRACE")) {
            if (*trace && !device_variant_cache_) {
                if (err) *err = "P40 cache trace requires the P40 device cache";
                return false;
            }
            if (*trace) {
                p40_trace_.open(trace, std::ios::out | std::ios::trunc);
                if (!p40_trace_) {
                    if (err) *err = "cannot create P40 cache trace";
                    return false;
                }
                p40_trace_ <<
                    "requested\tresident_before\taction\tmissing\tdevice_bytes\n";
            }
        }
        if (direct_reads()) {
            direct_read_pool_ = std::make_unique<P20DirectReadPool>(16);
        }
        if (const char * trace_path = std::getenv("DFLASH_KIMI_P20_IO_TRACE")) {
            if (*trace_path) {
                io_trace_.open(trace_path, std::ios::out | std::ios::trunc);
                if (!io_trace_) {
                    if (err) *err = std::string("cannot create P20 I/O trace ") +
                        trace_path;
                    return false;
                }
                io_trace_ <<
                    "request_id\tprompt_id\tbase_pos\ttoken_index\tmodel_layer"
                    "\texpert_id\tregion\tqtype\tprefix_depth\texact_fallback"
                    "\tfile_path\tfile_offset\tlogical_length\taligned_offset"
                    "\taligned_length\tdestination_kind\tdestination_offset"
                    "\texplicit_read_bytes\n";
                process_io_start_ = process_io_snapshot();
                const char * prompt_id =
                    std::getenv("DFLASH_KIMI_P20_PROMPT_ID");
                prompt_id_ = prompt_id && *prompt_id ? prompt_id : "0";
            }
        }
        layers_.resize(kLastRoutedLayer + 1);
        int valid_layers = 0;
        for (int layer = kFirstRoutedLayer; layer <= kLastRoutedLayer; ++layer) {
            LayerState & state = layers_[static_cast<size_t>(layer)];
            state.aux_path = calibrated_aux_path(aux_directory, layer);
            state.sidecar_path = natural_sidecar_path(sidecar_directory, layer);
            std::string layer_error;
            if (load_layer(layer, state, &layer_error)) {
                state.valid = true;
                ++valid_layers;
            } else {
                // Exact is the safe state, including startup with a partially
                // generated all-layer export.  Do not turn one bad layer into
                // a process-wide failure.
                std::fprintf(stderr,
                    "[kimi-k3-calibrated96] layer=%d action=exact reason=%s\n",
                    layer, layer_error.c_str());
            }
        }
        if (sidecar_authoritative_ && valid_layers != kLastRoutedLayer) {
            if (err) *err =
                "sidecar-authoritative mode requires 92 valid calibrated layers";
            return false;
        }
        if (ordered_device_join_) {
#if defined(DFLASH27B_BACKEND_CUDA) || defined(DFLASH27B_BACKEND_HIP)
            std::vector<P42MeanSource> mean_sources;
            mean_sources.reserve(kLastRoutedLayer);
            for (int layer = kFirstRoutedLayer;
                 layer <= kLastRoutedLayer; ++layer) {
                const LayerState & state = layers_[static_cast<size_t>(layer)];
                mean_sources.push_back({
                    state.aux_path, state.means_offset, state.means_bytes});
            }
            if (!ordered_join_arena_.load_resident_means(
                    backend_, kDimension, mean_sources, err)) {
                return false;
            }
            std::fprintf(stderr,
                "[kimi-k3-p42c] resident-mean-bytes=%llu "
                "hot-mean-reads=0 hot-mean-h2d-bytes=0\n",
                static_cast<unsigned long long>(
                    ordered_join_arena_.resident_mean_bytes()));
#endif
        }
        metrics_path_ = metrics_path && *metrics_path ? metrics_path : "";
        const std::string budget_description = layer_budgets_.empty()
            ? std::to_string(budget_) : "table:" + layer_budget_path_;
        std::fprintf(stderr,
            "[kimi-k3-calibrated96] status=exact policy=calibrated-slabs "
            "requested-budget=%s physical-layout=%s "
            "io-backend=%s persistent-scratch=%s compact-upload=%s "
            "pinned-compact=%s direct-pinned-compact=%s "
            "p40-device-cache=%s p40-layer-epoch=%s "
            "p41-compact-executor=%s p42-ordered-join=%s "
            "p45-async-queue=%s p40-wide-async-join=%s "
            "exact-source=%s "
            "p30-host-cache-mib=%.1f "
            "valid-layers=%d/92 "
            "invalid-layer-action=exact insufficient-expert-action=exact\n",
            budget_description.c_str(),
            sparse_device() ? "scratch" : "reference",
            direct_reads() ? "direct-pread" : "current",
            sparse_workspace_ == SparseWorkspace::PersistentDevice
                ? "enabled" : "disabled",
            sparse_delivery_ >= KimiK3SparseDeliveryPolicy::CompactPageable
                ? "enabled" : "disabled",
            sparse_delivery_ >= KimiK3SparseDeliveryPolicy::CompactPinned
                ? "enabled" : "disabled",
            sparse_delivery_ == KimiK3SparseDeliveryPolicy::DirectPinnedCompact
                ? "enabled" : "disabled",
            device_variant_cache_ ? "enabled" : "disabled",
            p40_layer_epoch_ ? "enabled" : "disabled",
            compact_executor_ ? "enabled" : "disabled",
            ordered_device_join_ ? "enabled" : "disabled",
            async_compact_queue_ ? "enabled" : "disabled",
            p40_wide_async_join_ ? "enabled" : "disabled",
            sidecar_authoritative_ ? "sidecar" : "native-model",
            static_cast<double>(read_cache_.capacity()) / (1024.0 * 1024.0),
            valid_layers);
        return true;
    }

    bool handles_layer(int model_layer) const override {
        return model_layer >= kFirstRoutedLayer &&
            model_layer <= kLastRoutedLayer &&
            (sidecar_authoritative_ ||
             budget_for_layer(model_layer) < kNativeTopK * kSlabCount);
    }

    KimiK3RoutedPrefillService * prefill_service() override {
        return supports_width(8) ? this : nullptr;
    }

    bool supports_width(size_t width) const override {
        if (width != 8 && width != 64 && width != 1024) {
            return false;
        }
        if (!sidecar_authoritative_ || io_trace_.is_open() ||
            p40_trace_.is_open()) {
            return false;
        }
        for (int layer = kFirstRoutedLayer;
             layer <= kLastRoutedLayer; ++layer) {
            if (budget_for_layer(layer) != 96) return false;
        }
        return true;
    }

    bool evaluate_layer(
            int model_layer, int base_pos,
            const MoeStreamExpertSpec & spec,
            const MoeStreamRouteBatch & routes,
            MoeHybridStreamEngine & exact_engine,
            std::vector<float> & output,
            std::string * err) override {
        if (!supports_width(static_cast<size_t>(routes.n_tokens))) {
            if (err) *err = "calibrated96 prefill width is not qualified";
            return false;
        }
        // The P58 A/B starts one cold external-cache epoch per wide layer.
        // Width-one evaluation/decode enters evaluate() directly and never
        // resets P40 residency.
        if (p40_layer_epoch_ &&
                !exact_engine.reset_external_device_cache(err)) {
            return false;
        }
        return evaluate(
            model_layer, base_pos, spec, routes, exact_engine, output, err);
    }

    bool evaluate(int model_layer, int base_pos,
                  const MoeStreamExpertSpec & spec,
                  const MoeStreamRouteBatch & routes,
                  MoeHybridStreamEngine & exact_engine,
                  std::vector<float> & output,
                  std::string * err) override {
        if (ordered_device_join_ && routes.n_tokens == 1) {
            if (err) *err = "P42 ordered join requires the device-output API";
            return false;
        }
        if (!handles_layer(model_layer) || routes.n_expert != kExpertCount ||
            routes.top_k != kNativeTopK || spec.input_dim != kDimension ||
            spec.output_dim != kDimension) {
            if (err) *err = "calibrated96 received an incompatible routed batch";
            return false;
        }
        constexpr size_t kP40AllocationTolerance = 64ULL * 1024 * 1024;
        if (device_variant_cache_ && routes.n_tokens > 1 &&
                exact_engine.external_device_cache_bytes() +
                kP40AllocationTolerance < p40_requested_device_bytes_) {
            if (err) {
                *err = "P40 shared device cache allocation fell below its requested budget";
            }
            return false;
        }
        LayerState & state = layers_[static_cast<size_t>(model_layer)];
        if (!state.valid || spec.fused_gate_up ||
            !geometry_matches(state, spec)) {
            if (sidecar_authoritative_) {
                if (err) *err =
                    "sidecar-authoritative layer is invalid or has incompatible geometry";
                return false;
            }
            const bool ok = eval_moe_streamed_experts(
                exact_engine, spec, routes, output, err);
            if (ok) observe_exact_layer(
                state, routes.n_tokens, spec, budget_for_layer(model_layer));
            return ok;
        }
        const bool p40_wide_async_join =
            p40_wide_async_join_ && routes.n_tokens > 1;
        return evaluate_calibrated(
            model_layer, base_pos, state, spec, routes, exact_engine, output,
            p40_wide_async_join, p40_wide_async_join, err);
    }

    KimiK3RoutedRuntimeStats runtime_stats() const override {
        KimiK3RoutedRuntimeStats result;
        for (const LayerState & state : layers_) {
            result.logical_provider_bytes +=
                state.traffic.selected_sidecar_bytes +
                state.traffic.exact_fallback_bytes;
        }
        result.explicit_read_bytes = explicit_read_bytes_;
        result.physical_direct_read_bytes = direct_physical_bytes_;
        result.direct_io_ns = direct_io_ns_;
        result.payload_h2d_bytes = authoritative_h2d_bytes_;
        result.metadata_h2d_bytes = metadata_h2d_bytes_;
        result.compact_pack_ns = sparse_device_evaluator_.compact_pack_ns();
        result.expert_graph_ns = sparse_device_evaluator_.expert_graph_ns();
        result.expert_readback_ns =
            sparse_device_evaluator_.expert_readback_ns();
        result.compact_attempted = p41_attempted_;
        result.compact_completed = p41_completed_;
        result.compact_fallbacks = p41_fallbacks_;
        result.compact_invalid = p41_invalid_;
        result.p40_requested_slabs = p40_requested_slabs_;
        result.p40_resident_before_slabs = p40_resident_before_slabs_;
        result.p40_hits = p40_hits_;
        result.p40_extensions = p40_extensions_;
        result.p40_cold = p40_cold_;
        result.p40_unavailable = p40_unavailable_;
        result.p40_completed = p40_completed_;
        result.p40_aborted = p40_aborted_;
        result.p40_fallbacks = p40_fallbacks_;
        result.p40_evictions = p40_evictions_;
        result.p40_h2d_bytes = p40_h2d_bytes_;
        result.p40_scatter_calls = p40_scatter_calls_;
        result.p40_scatter_avoided = p40_scatter_avoided_;
        const SparseDeviceExpertEvaluator::CompactAsyncStats & async =
            sparse_device_evaluator_.compact_async_stats();
        result.async_begins = async.begins;
        result.async_jobs = async.jobs;
        result.async_h2d_calls = async.h2d_calls;
        result.async_h2d_bytes = async.h2d_bytes;
        result.async_input_d2d_copies = async.input_d2d_copies;
        result.async_input_d2d_bytes = async.input_d2d_bytes;
        result.async_graph_enqueues = async.graph_enqueues;
        result.async_layer_flushes = async.layer_flushes;
        result.async_abort_syncs = async.abort_syncs;
#if defined(DFLASH27B_BACKEND_CUDA) || defined(DFLASH27B_BACKEND_HIP)
        result.ordered_expert_d2d_copies =
            ordered_join_arena_.expert_d2d_copies();
        result.ordered_expert_d2d_bytes =
            ordered_join_arena_.expert_d2d_bytes();
        result.ordered_join_launches = ordered_join_arena_.join_launches();
        result.ordered_output_d2d_copies =
            ordered_join_arena_.output_d2d_copies();
        result.ordered_output_d2d_bytes =
            ordered_join_arena_.output_d2d_bytes();
#endif
        return result;
    }

    bool requires_device_output() const override {
        return ordered_device_join_;
    }

    bool evaluate_device(
            int model_layer, int base_pos,
            const MoeStreamExpertSpec & spec,
            const MoeStreamRouteBatch & routes,
            MoeHybridStreamEngine & exact_engine,
            ggml_backend_t destination_backend,
            std::string * err) override {
#if !defined(DFLASH27B_BACKEND_CUDA) && !defined(DFLASH27B_BACKEND_HIP)
        (void) model_layer; (void) base_pos; (void) spec; (void) routes;
        (void) exact_engine; (void) destination_backend;
        if (err) *err = "P42 ordered join requires a GPU backend";
        return false;
#else
        if (!ordered_device_join_ || destination_backend != backend_ ||
            routes.n_tokens != 1 || !handles_layer(model_layer) ||
            routes.n_expert != kExpertCount || routes.top_k != kNativeTopK ||
            spec.input_dim != kDimension || spec.output_dim != kDimension ||
            (routes.inputs == nullptr) == (routes.device_inputs == nullptr)) {
            if (err) *err = "P42 ordered join received an incompatible request";
            return false;
        }
        LayerState & state = layers_[static_cast<size_t>(model_layer)];
        if (!state.valid || spec.fused_gate_up ||
            !geometry_matches(state, spec)) {
            if (err) *err = "P42 ordered join layer is incompatible";
            return false;
        }
        std::vector<float> unused;
        return evaluate_calibrated(
            model_layer, base_pos, state, spec, routes, exact_engine, unused,
            true, false, err);
#endif
    }

    bool copy_device_output(
            ggml_backend_t destination_backend,
            ggml_tensor * destination,
            std::string * err) override {
#if defined(DFLASH27B_BACKEND_CUDA) || defined(DFLASH27B_BACKEND_HIP)
        return ordered_join_arena_.copy_to(
            destination_backend, destination, err);
#else
        (void) destination_backend; (void) destination;
        if (err) *err = "P42 ordered join requires a GPU backend";
        return false;
#endif
    }

    void discard_device_output() override {
#if defined(DFLASH27B_BACKEND_CUDA) || defined(DFLASH27B_BACKEND_HIP)
        sparse_device_evaluator_.abort_compact_async_batch();
        ordered_join_arena_.discard();
#endif
    }

private:
    enum class CompactAttemptOutcome { Success, Invalid, FallbackMiss };

    CompactAttemptOutcome account_compact_attempt(
            bool success, bool invalid) {
        ++p41_attempted_;
        if (success) ++p41_completed_;
        else if (invalid) ++p41_invalid_;
        else ++p41_fallbacks_;
        return success ? CompactAttemptOutcome::Success :
            invalid ? CompactAttemptOutcome::Invalid :
                      CompactAttemptOutcome::FallbackMiss;
    }

    bool sparse_device() const {
        return sparse_workspace_ != SparseWorkspace::HostRecomposed;
    }
    bool direct_reads() const {
        return sparse_delivery_ != KimiK3SparseDeliveryPolicy::BufferedSlabs;
    }

    int budget_for_layer(int model_layer) const {
        if (layer_budgets_.empty()) return budget_;
        if (model_layer < kFirstRoutedLayer ||
            model_layer > kLastRoutedLayer) {
            return kNativeTopK * kSlabCount;
        }
        return layer_budgets_[static_cast<size_t>(model_layer - 1)];
    }


    struct Traffic {
        uint64_t tokens = 0;
        uint64_t requested_nominal_slabs = 0;
        uint64_t selected_slab_records = 0;
        uint64_t calibrated_routes = 0;
        uint64_t exact_fallback_routes = 0;
        uint64_t selected_sidecar_bytes = 0;
        uint64_t exact_fallback_bytes = 0;
    };

    struct LayerState {
        bool valid = false;
        std::string aux_path;
        std::string sidecar_path;
        uint64_t means_offset = 0;
        uint64_t means_bytes = 0;
        uint64_t payload_offset = 0;
        uint64_t slab_bytes = 0;
        uint64_t record_bytes = 0;
        uint64_t gate_slab_bytes = 0;
        uint64_t up_slab_bytes = 0;
        uint64_t down_slab_bytes = 0;
        uint64_t source_generation = 0;
        std::vector<uint16_t> order;
        std::vector<float> importance;
        std::vector<uint8_t> calibrated;
        std::vector<uint32_t> hit_counts;
        Traffic traffic;
    };

    static bool nonzero_digest(const uint8_t * digest) {
        for (int i = 0; i < 32; ++i) if (digest[i] != 0) return true;
        return false;
    }

    bool load_layer(int model_layer, LayerState & state, std::string * err) {
        std::ifstream input(state.aux_path, std::ios::binary | std::ios::ate);
        if (!input) {
            if (err) *err = "missing runtime aux " + state.aux_path;
            return false;
        }
        const uint64_t aux_bytes = static_cast<uint64_t>(input.tellg());
        input.seekg(0);
        SlabAuxHeaderV2 aux{};
        input.read(reinterpret_cast<char *>(&aux), sizeof(aux));
        const uint64_t expected_order =
            static_cast<uint64_t>(kExpertCount * kSlabCount) * sizeof(uint16_t);
        const uint64_t expected_means =
            static_cast<uint64_t>(kExpertCount) * kSlabCount * kDimension *
            sizeof(float);
        const uint64_t expected_importance =
            static_cast<uint64_t>(kExpertCount * kSlabCount) * sizeof(float);
        const bool aux_valid = input &&
            std::memcmp(aux.magic, "K3AUX001", 8) == 0 && aux.version == 2 &&
            aux.model_layer == static_cast<uint32_t>(model_layer) &&
            aux.expert_count == kExpertCount && aux.dimension == kDimension &&
            aux.slab_size == kSlabSize && aux.slab_count == kSlabCount &&
            aux.storage == 0 && aux.alignment == kAlignment &&
            aux.order_bytes == expected_order &&
            aux.slab_means_bytes == expected_means &&
            aux.slab_importance_bytes == expected_importance &&
            aux.calibrated_experts_bytes == kExpertCount &&
            aux.calibration_hit_counts_bytes ==
                static_cast<uint64_t>(kExpertCount) * sizeof(uint32_t) &&
            checked_span(aux.order_offset, aux.order_bytes, aux_bytes) &&
            checked_span(aux.slab_means_offset, aux.slab_means_bytes, aux_bytes) &&
            checked_span(aux.slab_importance_offset,
                         aux.slab_importance_bytes, aux_bytes) &&
            checked_span(aux.calibrated_experts_offset,
                         aux.calibrated_experts_bytes, aux_bytes) &&
            checked_span(aux.calibration_hit_counts_offset,
                         aux.calibration_hit_counts_bytes, aux_bytes) &&
            nonzero_digest(aux.fit_state_sha256) &&
            nonzero_digest(aux.capture_sha256) &&
            nonzero_digest(aux.sidecar_sha256) &&
            nonzero_digest(aux.model_registry_sha256);
        if (!aux_valid) {
            if (err) *err = "invalid or provenance-free runtime aux";
            return false;
        }
        if (!read_array(input, aux.order_offset, aux.order_bytes,
                        state.order, err) ||
            !read_array(input, aux.slab_importance_offset,
                        aux.slab_importance_bytes, state.importance, err) ||
            !read_array(input, aux.calibrated_experts_offset,
                        aux.calibrated_experts_bytes, state.calibrated, err) ||
            !read_array(input, aux.calibration_hit_counts_offset,
                        aux.calibration_hit_counts_bytes,
                        state.hit_counts, err) ||
            !valid_slab_order(state.order)) {
            if (err && err->empty()) *err = "invalid calibrated layer arrays";
            return false;
        }
        for (int expert = 0; expert < kExpertCount; ++expert) {
            const uint8_t flag = state.calibrated[static_cast<size_t>(expert)];
            if (flag > 1 || (flag != 0 &&
                    state.hit_counts[static_cast<size_t>(expert)] == 0)) {
                if (err) *err = "calibrated mask disagrees with hit counts";
                return false;
            }
            for (int rank = 0; rank < kSlabCount; ++rank) {
                const float score = state.importance[
                    static_cast<size_t>(expert) * kSlabCount + rank];
                if (!std::isfinite(score) || score < 0.0f) {
                    if (err) *err = "invalid calibrated slab importance";
                    return false;
                }
            }
        }
        state.means_offset = aux.slab_means_offset;
        state.means_bytes = aux.slab_means_bytes;

        const int fd = open_read_only(state.sidecar_path);
        if (fd < 0) {
            if (err) *err = "missing natural sidecar " + state.sidecar_path;
            return false;
        }
        uint64_t file_bytes = 0;
        SlabSidecarHeaderV2 sidecar{};
        bool header_ok = file_size(fd, file_bytes) &&
            read_exact_at(
                fd, &sidecar,
                offsetof(SlabSidecarHeaderV2, gate_slab_bytes), 0) &&
            std::memcmp(sidecar.magic, "K3SLB001", 8) == 0 &&
            (sidecar.version == 1 || sidecar.version == 2);
        if (header_ok && sidecar.version == 2) {
            header_ok = read_exact_at(fd, &sidecar, sizeof(sidecar), 0);
        }
        const uint64_t gate_bytes = sidecar.version == 1
            ? kSlabComponentBytes : sidecar.gate_slab_bytes;
        const uint64_t up_bytes = sidecar.version == 1
            ? kSlabComponentBytes : sidecar.up_slab_bytes;
        const uint64_t down_bytes = sidecar.version == 1
            ? kSlabComponentBytes : sidecar.down_slab_bytes;
        std::vector<uint16_t> natural(
            static_cast<size_t>(kExpertCount * kSlabCount));
        const bool sidecar_valid = header_ok &&
            sidecar.model_layer == static_cast<uint32_t>(model_layer) &&
            sidecar.expert_count == kExpertCount &&
            sidecar.dimension == kDimension &&
            sidecar.expert_width == kSlabSize * kSlabCount &&
            sidecar.slab_size == kSlabSize &&
            sidecar.slab_count == kSlabCount &&
            sidecar.alignment == kAlignment &&
            gate_bytes > 0 && up_bytes > 0 && down_bytes > 0 &&
            sidecar.slab_bytes == gate_bytes + up_bytes + down_bytes &&
            sidecar.record_bytes == sidecar.slab_bytes * kSlabCount &&
            sidecar.order_bytes == natural.size() * sizeof(uint16_t) &&
            checked_span(sidecar.payload_offset,
                static_cast<uint64_t>(kExpertCount) * sidecar.record_bytes,
                file_bytes) &&
            read_exact_at(fd, natural.data(),
                natural.size() * sizeof(uint16_t), sidecar.order_offset);
        close_fd(fd);
        if (!sidecar_valid) {
            if (err) *err = "invalid mixed-layout natural sidecar";
            return false;
        }
        for (int expert = 0; expert < kExpertCount; ++expert) {
            for (int rank = 0; rank < kSlabCount; ++rank) {
                if (natural[static_cast<size_t>(expert) * kSlabCount + rank] !=
                        rank) {
                    if (err) *err = "sidecar is not in natural slab order";
                    return false;
                }
            }
        }
        state.payload_offset = sidecar.payload_offset;
        state.slab_bytes = sidecar.slab_bytes;
        state.record_bytes = sidecar.record_bytes;
        state.gate_slab_bytes = gate_bytes;
        state.up_slab_bytes = up_bytes;
        state.down_slab_bytes = down_bytes;
        state.source_generation = artifact_generation(
            aux.sidecar_sha256, sizeof(aux.sidecar_sha256));
        return true;
    }

    static uint64_t exact_record_bytes(const MoeStreamExpertSpec & spec) {
        if (spec.fused_gate_up) {
            return ggml_row_size(spec.gate_up_type, spec.input_dim) *
                    static_cast<uint64_t>(2 * spec.intermediate_dim) +
                ggml_row_size(spec.down_type, spec.intermediate_dim) *
                    static_cast<uint64_t>(spec.output_dim);
        }
        return ggml_row_size(spec.gate_type, spec.input_dim) *
                static_cast<uint64_t>(spec.intermediate_dim) +
            ggml_row_size(spec.up_type, spec.input_dim) *
                static_cast<uint64_t>(spec.intermediate_dim) +
            ggml_row_size(spec.down_type, spec.intermediate_dim) *
                static_cast<uint64_t>(spec.output_dim);
    }

    static bool geometry_matches(const LayerState & state,
                                 const MoeStreamExpertSpec & spec) {
        if (state.down_slab_bytes % spec.output_dim != 0) return false;
        return state.gate_slab_bytes * kSlabCount ==
                ggml_row_size(spec.gate_type, spec.input_dim) *
                    static_cast<uint64_t>(spec.intermediate_dim) &&
            state.up_slab_bytes * kSlabCount ==
                ggml_row_size(spec.up_type, spec.input_dim) *
                    static_cast<uint64_t>(spec.intermediate_dim) &&
            (state.down_slab_bytes / spec.output_dim) * kSlabCount ==
                ggml_row_size(spec.down_type, spec.intermediate_dim);
    }

    void observe_exact_layer(LayerState & state, int n_tokens,
                             const MoeStreamExpertSpec & spec,
                             int requested_budget) {
        const uint64_t routes = static_cast<uint64_t>(n_tokens) * kNativeTopK;
        state.traffic.tokens += n_tokens;
        state.traffic.requested_nominal_slabs +=
            static_cast<uint64_t>(n_tokens) * requested_budget;
        state.traffic.exact_fallback_routes += routes;
        state.traffic.exact_fallback_bytes +=
            routes * exact_record_bytes(spec);
    }

    bool traced_read_exact_at(
            int fd, void * destination, size_t bytes, uint64_t offset,
            int model_layer, int base_pos, int token_index, int expert,
            const char * region, const char * qtype, int prefix_depth,
            bool exact_fallback, const std::string & path,
            const char * destination_kind, uint64_t destination_offset) {
        const bool slab_mean = std::strcmp(region, "slab-mean") == 0;
        const bool native_mean = std::strcmp(region, "native-mean") == 0;
        const bool cacheable = read_cache_.enabled() &&
            (slab_mean || native_mean);
        const P30ReadKey cache_key{
            model_layer,
            native_mean ? P30ReadKind::NativeMean : P30ReadKind::SlabMean,
            offset, bytes};
        const bool cache_hit = cacheable &&
            read_cache_.get(cache_key, destination);
        const bool ok = cache_hit || read_exact_at(
            fd, destination, bytes, offset);
        if (ok && cacheable && !cache_hit) {
            read_cache_.put(cache_key, destination);
        }
        const uint64_t physical_bytes = ok && !cache_hit ? bytes : 0;
        explicit_read_bytes_ += physical_bytes;
        if (!io_trace_) return ok;
        const uint64_t aligned_offset = offset & ~(kAlignment - 1);
        const uint64_t end = offset + bytes;
        const uint64_t aligned_end =
            (end + kAlignment - 1) & ~(static_cast<uint64_t>(kAlignment) - 1);
        io_trace_ << next_request_id_++ << '\t' << prompt_id_ << '\t'
                  << base_pos << '\t' << token_index << '\t' << model_layer
                  << '\t' << expert << '\t' << region << '\t' << qtype
                  << '\t' << prefix_depth << '\t'
                  << (exact_fallback ? 1 : 0) << '\t' << path << '\t'
                  << offset << '\t' << bytes << '\t' << aligned_offset
                  << '\t' << aligned_end - aligned_offset << '\t'
                  << destination_kind << '\t' << destination_offset << '\t'
                  << physical_bytes << '\n';
        return ok;
    }

    void trace_fallback(int model_layer, int base_pos, int token_index,
                        int expert, const MoeStreamExpertSpec & spec) {
        if (!io_trace_) return;
        io_trace_ << next_request_id_++ << '\t' << prompt_id_ << '\t'
                  << base_pos << '\t' << token_index << '\t' << model_layer
                  << '\t' << expert
                  << "\tnative-exact-expert\tmixed\t12\t1"
                  << "\t<native-model-shards>\t-1\t"
                  << exact_record_bytes(spec)
                  << "\t-1\t-1\tmoe-stream-engine\t0\t0\n";
    }

    bool read_direct_sidecar_record(
            int fd, int model_layer, uint64_t aligned_offset,
            size_t aligned_bytes, void * destination,
            bool & cache_hit) {
#if defined(_WIN32) || !defined(O_DIRECT)
        (void) fd; (void) model_layer; (void) aligned_offset;
        (void) aligned_bytes; (void) destination;
        cache_hit = false;
        return false;
#else
        const P30ReadKey cache_key{
            model_layer, P30ReadKind::SidecarSlab,
            aligned_offset, aligned_bytes};
        cache_hit = read_cache_.get(cache_key, destination);
        const ssize_t got = cache_hit
            ? static_cast<ssize_t>(aligned_bytes)
            : ::pread(fd, destination, aligned_bytes,
                      static_cast<off_t>(aligned_offset));
        if (got != static_cast<ssize_t>(aligned_bytes)) return false;
        if (!cache_hit) read_cache_.put(cache_key, destination);
        return true;
#endif
    }

    bool read_sparse_payloads_direct(
            int fd, const LayerState & state,
            const MoeStreamExpertSpec & spec, int model_layer, int base_pos,
            int token_index, int expert, int prefix_depth,
            bool exact_fallback, std::vector<SparseSlabPayload> & slabs,
            std::string * err) {
#if defined(_WIN32) || !defined(O_DIRECT)
        (void) fd; (void) state; (void) spec; (void) model_layer;
        (void) base_pos; (void) token_index; (void) expert;
        (void) prefix_depth; (void) exact_fallback; (void) slabs;
        if (err) *err = "P20 direct-pread is unavailable on this platform";
        return false;
#else
        struct Completion {
            bool ok = false;
            bool cache_hit = false;
            uint64_t aligned_offset = 0;
            size_t aligned_bytes = 0;
        };
        std::vector<Completion> completions(slabs.size());
        const auto io_started = std::chrono::steady_clock::now();
        std::atomic<size_t> next{0};
        std::atomic<bool> failed{false};
        const size_t workers = std::min<size_t>(16, slabs.size());
        std::vector<std::future<void>> workers_done;
        workers_done.reserve(workers);
        for (size_t worker = 0; worker < workers; ++worker) {
            workers_done.push_back(direct_read_pool_->submit([&]() {
                for (;;) {
                    const size_t index = next.fetch_add(1);
                    if (index >= slabs.size()) break;
                    SparseSlabPayload & slab = slabs[index];
                    const uint64_t record = state.payload_offset +
                        static_cast<uint64_t>(
                            expert * kSlabCount + slab.natural) *
                            state.slab_bytes;
                    const uint64_t aligned_offset =
                        record & ~(static_cast<uint64_t>(kAlignment) - 1);
                    const size_t prefix = static_cast<size_t>(
                        record - aligned_offset);
                    const size_t aligned_bytes = static_cast<size_t>(
                        (prefix + state.slab_bytes + kAlignment - 1) &
                        ~(static_cast<uint64_t>(kAlignment) - 1));
                    void * raw = nullptr;
                    if (::posix_memalign(&raw, kAlignment, aligned_bytes) != 0) {
                        failed = true;
                        break;
                    }
                    bool cache_hit = false;
                    if (read_direct_sidecar_record(
                            fd, model_layer, aligned_offset, aligned_bytes,
                            raw, cache_hit)) {
                        const auto * payload =
                            static_cast<const uint8_t *>(raw) + prefix;
                        std::memcpy(
                            slab.gate.data(), payload, slab.gate.size());
                        std::memcpy(
                            slab.up.data(), payload + slab.gate.size(),
                            slab.up.size());
                        std::memcpy(
                            slab.down.data(),
                            payload + slab.gate.size() + slab.up.size(),
                            slab.down.size());
                        completions[index] = {
                            true, cache_hit, aligned_offset, aligned_bytes};
                    } else {
                        failed = true;
                    }
                    std::free(raw);
                    if (failed.load()) break;
                }
            }));
        }
        for (std::future<void> & done : workers_done) done.get();
        direct_io_ns_ += static_cast<uint64_t>(
            std::chrono::duration_cast<std::chrono::nanoseconds>(
                std::chrono::steady_clock::now() - io_started).count());
        if (failed.load() || std::any_of(
                completions.begin(), completions.end(),
                [](const Completion & value) { return !value.ok; })) {
            if (err) *err = "P20 aligned direct sidecar read failed";
            return false;
        }
        for (size_t index = 0; index < slabs.size(); ++index) {
            const SparseSlabPayload & slab = slabs[index];
            const uint64_t record = state.payload_offset +
                static_cast<uint64_t>(expert * kSlabCount + slab.natural) *
                    state.slab_bytes;
            const uint64_t physical_bytes = completions[index].cache_hit
                ? 0 : completions[index].aligned_bytes;
            explicit_read_bytes_ += physical_bytes;
            direct_physical_bytes_ += physical_bytes;
            if (!io_trace_) continue;
            const auto emit = [&](const char * region, const char * qtype,
                                  uint64_t offset, size_t logical,
                                  uint64_t destination_offset,
                                  uint64_t explicit_bytes) {
                io_trace_ << next_request_id_++ << '\t' << prompt_id_ << '\t'
                          << base_pos << '\t' << token_index << '\t'
                          << model_layer << '\t' << expert << '\t' << region
                          << '\t' << qtype << '\t' << prefix_depth
                          << '\t' << (exact_fallback ? 1 : 0) << '\t'
                          << state.sidecar_path << '\t' << offset
                          << '\t' << logical << '\t'
                          << completions[index].aligned_offset << '\t'
                          << completions[index].aligned_bytes
                          << "\thost-compact-slab\t" << destination_offset
                          << '\t' << explicit_bytes << '\n';
            };
            emit("gate", ggml_type_name(spec.gate_type), record,
                 slab.gate.size(), 0, physical_bytes);
            emit("up", ggml_type_name(spec.up_type),
                 record + slab.gate.size(), slab.up.size(),
                 slab.gate.size(), 0);
            emit("down", ggml_type_name(spec.down_type),
                 record + slab.gate.size() + slab.up.size(), slab.down.size(),
                 slab.gate.size() + slab.up.size(), 0);
        }
        return true;
#endif
    }

    bool read_sparse_payloads_direct_batch(
            int fd, const LayerState & state,
            const MoeStreamExpertSpec & spec, int model_layer, int base_pos,
            int token_index, size_t route_offset,
            const MoeStreamRouteBatch & routes,
            const std::vector<int> & calibrated_routes,
            const std::vector<uint8_t> & selected_by_route,
            std::vector<std::vector<SparseSlabPayload>> & payloads,
            std::array<SparseCompactPayload, kNativeTopK> * compact_payloads,
            bool component_major_payloads,
            std::string * err) {
#if defined(_WIN32) || !defined(O_DIRECT)
        (void) fd; (void) state; (void) spec; (void) model_layer;
        (void) base_pos; (void) token_index; (void) route_offset;
        (void) routes; (void) calibrated_routes; (void) selected_by_route;
        (void) payloads; (void) compact_payloads;
        (void) component_major_payloads;
        if (err) *err = "P20 direct-pread is unavailable on this platform";
        return false;
#else
        struct Task {
            int route = 0;
            int expert = 0;
            int prefix_depth = 0;
            size_t slab_index = 0;
            uint16_t natural = 0;
            uint64_t aligned_offset = 0;
            size_t aligned_bytes = 0;
            bool cache_hit = false;
            bool ok = false;
        };
        payloads.clear();
        payloads.resize(kNativeTopK);
        std::vector<Task> tasks;
        for (const int route : calibrated_routes) {
            const int expert = routes.selected_ids[route_offset + route];
            int prefix_depth = 0;
            for (int rank = 0; rank < kSlabCount; ++rank) {
                if (selected_by_route[
                        static_cast<size_t>(route * kSlabCount + rank)]) {
                    ++prefix_depth;
                }
            }
            std::vector<SparseSlabPayload> & route_payloads =
                payloads[static_cast<size_t>(route)];
            route_payloads.reserve(static_cast<size_t>(prefix_depth));
            SparseCompactPayload * compact = compact_payloads
                ? &(*compact_payloads)[static_cast<size_t>(route)] : nullptr;
            if (compact) {
                compact->slab_count = prefix_depth;
                compact->gate_slab_bytes =
                    static_cast<size_t>(state.gate_slab_bytes);
                compact->up_slab_bytes =
                    static_cast<size_t>(state.up_slab_bytes);
                compact->down_slab_bytes =
                    static_cast<size_t>(state.down_slab_bytes);
                compact->component_major = component_major_payloads &&
                    prefix_depth > 0;
                KimiK3CompactWireLayout compact_layout;
                if (compact->component_major &&
                    !kimi_k3_compact_wire_layout(
                        prefix_depth, compact->gate_slab_bytes,
                        compact->up_slab_bytes, compact->down_slab_bytes,
                        &compact_layout)) {
                    if (err) *err = "P41 direct compact layout is invalid";
                    return false;
                }
                compact->bytes = compact->component_major
                    ? compact_layout.total_bytes
                    : compact->metadata_bytes +
                        static_cast<size_t>(prefix_depth) * state.slab_bytes;
                if (prefix_depth > 0 &&
                    !compact->ensure(compact->bytes, err)) return false;
                if (prefix_depth > 0) {
                    std::memset(
                        compact->data, 0, compact->metadata_bytes);
                }
            }
            size_t compact_index = 0;
            for (int rank = 0; rank < kSlabCount; ++rank) {
                if (!selected_by_route[
                        static_cast<size_t>(route * kSlabCount + rank)]) {
                    continue;
                }
                const uint16_t natural = state.order[
                    static_cast<size_t>(expert) * kSlabCount + rank];
                const size_t slab_index = route_payloads.size();
                if (compact) {
                    std::memcpy(
                        static_cast<uint8_t *>(compact->data) +
                            compact_index * sizeof(uint16_t),
                        &natural, sizeof(natural));
                    tasks.push_back({route, expert, prefix_depth,
                                     compact_index, natural});
                    ++compact_index;
                } else {
                    SparseSlabPayload slab;
                    slab.natural = natural;
                    slab.gate.resize(
                        static_cast<size_t>(state.gate_slab_bytes));
                    slab.up.resize(
                        static_cast<size_t>(state.up_slab_bytes));
                    slab.down.resize(
                        static_cast<size_t>(state.down_slab_bytes));
                    route_payloads.push_back(std::move(slab));
                    tasks.push_back({route, expert, prefix_depth,
                                     slab_index, natural});
                }
            }
        }
        const auto io_started = std::chrono::steady_clock::now();
        std::atomic<size_t> next{0};
        std::atomic<bool> failed{false};
        const size_t workers = std::min<size_t>(16, tasks.size());
        std::vector<std::future<void>> workers_done;
        workers_done.reserve(workers);
        for (size_t worker = 0; worker < workers; ++worker) {
            workers_done.push_back(direct_read_pool_->submit([&]() {
                for (;;) {
                    const size_t task_index = next.fetch_add(1);
                    if (task_index >= tasks.size()) break;
                    Task & task = tasks[task_index];
                    const uint64_t record = state.payload_offset +
                        static_cast<uint64_t>(
                            task.expert * kSlabCount + task.natural) *
                            state.slab_bytes;
                    task.aligned_offset =
                        record & ~(static_cast<uint64_t>(kAlignment) - 1);
                    const size_t prefix = static_cast<size_t>(
                        record - task.aligned_offset);
                    task.aligned_bytes = static_cast<size_t>(
                        (prefix + state.slab_bytes + kAlignment - 1) &
                        ~(static_cast<uint64_t>(kAlignment) - 1));
                    void * raw = nullptr;
                    if (::posix_memalign(
                            &raw, kAlignment, task.aligned_bytes) != 0) {
                        failed = true;
                        break;
                    }
                    if (read_direct_sidecar_record(
                            fd, model_layer, task.aligned_offset,
                            task.aligned_bytes, raw, task.cache_hit)) {
                        const auto * source =
                            static_cast<const uint8_t *>(raw) + prefix;
                        if (compact_payloads) {
                            SparseCompactPayload & compact =
                                (*compact_payloads)[
                                    static_cast<size_t>(task.route)];
                            if (compact.component_major) {
                                KimiK3CompactWireLayout layout;
                                if (!kimi_k3_compact_wire_layout(
                                        compact.slab_count,
                                        compact.gate_slab_bytes,
                                        compact.up_slab_bytes,
                                        compact.down_slab_bytes, &layout)) {
                                    failed = true;
                                    std::free(raw);
                                    break;
                                }
                                auto * destination =
                                    static_cast<uint8_t *>(compact.data);
                                std::memcpy(
                                    destination + layout.gate_offset +
                                        task.slab_index *
                                            compact.gate_slab_bytes,
                                    source, compact.gate_slab_bytes);
                                std::memcpy(
                                    destination + layout.up_offset +
                                        task.slab_index * compact.up_slab_bytes,
                                    source + compact.gate_slab_bytes,
                                    compact.up_slab_bytes);
                                std::memcpy(
                                    destination + layout.down_offset +
                                        task.slab_index *
                                            compact.down_slab_bytes,
                                    source + compact.gate_slab_bytes +
                                        compact.up_slab_bytes,
                                    compact.down_slab_bytes);
                            } else {
                                std::memcpy(
                                    static_cast<uint8_t *>(compact.data) +
                                        compact.metadata_bytes +
                                        task.slab_index * state.slab_bytes,
                                    source,
                                    static_cast<size_t>(state.slab_bytes));
                            }
                        } else {
                            SparseSlabPayload & slab = payloads[
                                static_cast<size_t>(task.route)][
                                    task.slab_index];
                            std::memcpy(
                                slab.gate.data(), source, slab.gate.size());
                            std::memcpy(
                                slab.up.data(), source + slab.gate.size(),
                                slab.up.size());
                            std::memcpy(
                                slab.down.data(),
                                source + slab.gate.size() + slab.up.size(),
                                slab.down.size());
                        }
                        task.ok = true;
                    } else {
                        failed = true;
                    }
                    std::free(raw);
                    if (failed.load()) break;
                }
            }));
        }
        for (std::future<void> & done : workers_done) done.get();
        direct_io_ns_ += static_cast<uint64_t>(
            std::chrono::duration_cast<std::chrono::nanoseconds>(
                std::chrono::steady_clock::now() - io_started).count());
        if (failed.load() || std::any_of(
                tasks.begin(), tasks.end(),
                [](const Task & task) { return !task.ok; })) {
            if (err) *err = "P20 aligned layer-batch sidecar read failed";
            return false;
        }
        for (const Task & task : tasks) {
            const uint64_t record = state.payload_offset +
                static_cast<uint64_t>(
                    task.expert * kSlabCount + task.natural) *
                    state.slab_bytes;
            const uint64_t physical_bytes = task.cache_hit
                ? 0 : task.aligned_bytes;
            explicit_read_bytes_ += physical_bytes;
            direct_physical_bytes_ += physical_bytes;
            if (!io_trace_) continue;
            uint64_t gate_destination_offset = 0;
            uint64_t up_destination_offset = state.gate_slab_bytes;
            uint64_t down_destination_offset =
                state.gate_slab_bytes + state.up_slab_bytes;
            if (compact_payloads) {
                const SparseCompactPayload & compact =
                    (*compact_payloads)[static_cast<size_t>(task.route)];
                if (compact.component_major) {
                    KimiK3CompactWireLayout layout;
                    if (!kimi_k3_compact_wire_layout(
                            compact.slab_count, compact.gate_slab_bytes,
                            compact.up_slab_bytes, compact.down_slab_bytes,
                            &layout)) {
                        if (err) *err = "P41 trace compact layout is invalid";
                        return false;
                    }
                    gate_destination_offset = layout.gate_offset +
                        task.slab_index * compact.gate_slab_bytes;
                    up_destination_offset = layout.up_offset +
                        task.slab_index * compact.up_slab_bytes;
                    down_destination_offset = layout.down_offset +
                        task.slab_index * compact.down_slab_bytes;
                }
            }
            const auto emit = [&](const char * region, const char * qtype,
                                  uint64_t offset, size_t logical,
                                  uint64_t destination_offset,
                                  uint64_t explicit_bytes) {
                io_trace_ << next_request_id_++ << '\t' << prompt_id_ << '\t'
                          << base_pos << '\t' << token_index << '\t'
                          << model_layer << '\t' << task.expert << '\t'
                          << region << '\t' << qtype << '\t'
                          << task.prefix_depth << "\t0\t"
                          << state.sidecar_path << '\t' << offset << '\t'
                          << logical << '\t' << task.aligned_offset << '\t'
                          << task.aligned_bytes
                          << "\thost-compact-slab\t" << destination_offset
                          << '\t' << (task.cache_hit ? 0 : explicit_bytes)
                          << '\n';
            };
            emit("gate", ggml_type_name(spec.gate_type), record,
                 static_cast<size_t>(state.gate_slab_bytes),
                 gate_destination_offset,
                 task.aligned_bytes);
            emit("up", ggml_type_name(spec.up_type),
                 record + state.gate_slab_bytes,
                 static_cast<size_t>(state.up_slab_bytes),
                 up_destination_offset, 0);
            emit("down", ggml_type_name(spec.down_type),
                 record + state.gate_slab_bytes + state.up_slab_bytes,
                 static_cast<size_t>(state.down_slab_bytes),
                 down_destination_offset, 0);
        }
        return true;
#endif
    }

    void account_device_variant_lease(
            const MoeStreamExternalLease & lease, uint16_t requested_mask,
            size_t device_bytes) {
        p40_requested_slabs_ += slab_mask_count(requested_mask);
        p40_device_bytes_ = device_bytes;
        const uint16_t resident = lease ? lease.resident_mask() : 0;
        const uint16_t missing = lease
            ? lease.missing_mask() : requested_mask;
        const char * action = !lease ? "unavailable" :
            lease.cache_hit() ? "hit" :
            resident != 0 ? "extension" : "cold";
        if (p40_trace_) {
            p40_trace_ << requested_mask << '\t' << resident << '\t'
                       << action << '\t' << missing << '\t'
                       << device_bytes << '\n';
        }
        if (!lease) {
            ++p40_unavailable_;
            return;
        }
        p40_resident_before_slabs_ += slab_mask_count(
            static_cast<uint16_t>(requested_mask & lease.resident_mask()));
        if (lease.cache_hit()) {
            ++p40_hits_;
        } else if (lease.resident_mask() != 0) {
            ++p40_extensions_;
        } else {
            ++p40_cold_;
        }
        if (lease.evicted()) ++p40_evictions_;
    }

    bool evaluate_sparse_payload(
            const MoeStreamExpertSpec & spec, const float * input,
            const std::vector<SparseSlabPayload> & slabs,
            const SparseCompactPayload * compact,
            const std::vector<float> & mask, size_t down_slab_row_bytes,
            uint16_t requested_mask, std::vector<float> & result,
            MoeStreamExternalLease * cache_lease,
            std::string * err) {
        SparseDeviceExpertEvaluator transient;
        SparseDeviceExpertEvaluator & evaluator =
            sparse_workspace_ == SparseWorkspace::PersistentDevice
            ? sparse_device_evaluator_ : transient;
        const uint16_t missing_before = cache_lease
            ? cache_lease->missing_mask() : 0;
        const uint64_t h2d_before = authoritative_h2d_bytes_;
        // A live P40 lease owns the expanded weight layout. Keep that
        // qualified path authoritative until compact parity is established;
        // P41 runs when the request is not being served by that cache.
        if (compact_executor_ && !cache_lease) {
            bool invalid = false;
            const bool success = evaluator.evaluate(
                backend_, spec, input, slabs, compact, mask,
                down_slab_row_bytes, result, authoritative_h2d_bytes_,
                metadata_h2d_bytes_, device_zero_bytes_,
                kimi_k3_sparse_upload_for_call(
                    sparse_delivery_, compact != nullptr), nullptr,
                requested_mask, true, invalid, err);
            const CompactAttemptOutcome outcome = account_compact_attempt(
                success, invalid);
            if (outcome == CompactAttemptOutcome::Success) return true;
            if (outcome == CompactAttemptOutcome::Invalid) return false;
            if (err) err->clear();
        }
        bool compact_invalid = false;
        const bool ok = evaluator.evaluate(
            backend_, spec, input, slabs, compact, mask, down_slab_row_bytes,
            result, authoritative_h2d_bytes_, metadata_h2d_bytes_,
            device_zero_bytes_, kimi_k3_sparse_upload_for_call(
                sparse_delivery_, compact != nullptr), cache_lease,
            requested_mask, false, compact_invalid, err);
        if (cache_lease) {
            if (ok) {
                ++p40_completed_;
                if (!*cache_lease) ++p40_fallbacks_;
            } else {
                ++p40_aborted_;
            }
        }
        if (ok && cache_lease && *cache_lease) {
            p40_h2d_bytes_ += authoritative_h2d_bytes_ - h2d_before;
            if (missing_before != 0) ++p40_scatter_calls_;
            else ++p40_scatter_avoided_;
        }
        return ok;
    }

    bool evaluate_sparse_payload_device(
            const MoeStreamExpertSpec & spec, const float * input,
            const std::vector<SparseSlabPayload> & slabs,
            const SparseCompactPayload * compact,
            size_t down_slab_row_bytes, uint16_t requested_mask,
            ggml_tensor *& device_output, std::string * err) {
        device_output = nullptr;
        if (!compact_executor_) {
            if (err) *err = "P42 device output requires P41";
            return false;
        }
        bool invalid = false;
        const bool success = sparse_device_evaluator_.evaluate_compact_device(
            backend_, spec, input, slabs, compact, requested_mask,
            down_slab_row_bytes, device_output,
            authoritative_h2d_bytes_, metadata_h2d_bytes_, invalid, err);
        return account_compact_attempt(success, invalid) ==
            CompactAttemptOutcome::Success;
    }

    bool evaluate_sparse_payload_cached_device(
            const MoeStreamExpertSpec & spec, const float * input,
            const std::vector<SparseSlabPayload> & slabs,
            const SparseCompactPayload * compact,
            const std::vector<float> & mask, size_t down_slab_row_bytes,
            MoeStreamExternalLease & cache_lease,
            ggml_tensor *& device_output, std::string * err) {
        device_output = nullptr;
        if (!cache_lease) {
            if (err) *err = "P40 cached device output requires a live lease";
            return false;
        }
        const uint16_t missing_before = cache_lease.missing_mask();
        const uint64_t h2d_before = authoritative_h2d_bytes_;
        const bool success = sparse_device_evaluator_.evaluate_cached_device(
            backend_, spec, input, slabs, compact, mask,
            down_slab_row_bytes, device_output, authoritative_h2d_bytes_,
            metadata_h2d_bytes_, device_zero_bytes_,
            kimi_k3_sparse_upload_for_call(
                sparse_delivery_, compact != nullptr),
            &cache_lease, err);
        if (success) {
            ++p40_completed_;
            p40_h2d_bytes_ += authoritative_h2d_bytes_ - h2d_before;
            if (missing_before != 0) ++p40_scatter_calls_;
            else ++p40_scatter_avoided_;
        } else {
            ++p40_aborted_;
        }
        return success;
    }

    bool evaluate_sidecar_exact_expert(
            int sidecar_fd, int model_layer, int base_pos, int token_index,
            const LayerState & state, const MoeStreamExpertSpec & spec,
            int local_layer, int expert, const float * input,
            MoeHybridStreamEngine & exact_engine,
            bool use_device_variant_cache,
            MoeStreamExternalLease * retained_cache_lease,
            std::vector<float> & result, ggml_tensor ** device_result,
            std::string * err) {
        const size_t gate_full_bytes = static_cast<size_t>(
            state.gate_slab_bytes * kSlabCount);
        const size_t up_full_bytes = static_cast<size_t>(
            state.up_slab_bytes * kSlabCount);
        const size_t down_slab_row_bytes = static_cast<size_t>(
            state.down_slab_bytes / spec.output_dim);
        const size_t down_full_row_bytes = down_slab_row_bytes * kSlabCount;
        std::vector<float> full_mask(
            static_cast<size_t>(spec.intermediate_dim), 1.0f);

        if (sparse_device()) {
            MoeStreamExternalLease local_cache_lease;
            MoeStreamExternalLease & cache_lease = retained_cache_lease
                ? *retained_cache_lease : local_cache_lease;
            if (retained_cache_lease && cache_lease) {
                if (err) *err = "P40 fallback lease slot is already active";
                return false;
            }
            const MoeStreamExternalKey cache_key{
                kNaturalSidecarCacheDomain, state.source_generation,
                local_layer, expert, spec};
            if (use_device_variant_cache &&
                !sparse_device_evaluator_.acquire_cache_lease(
                    backend_, spec, false, exact_engine, cache_key, 0x0fff,
                    cache_lease, err)) {
                return false;
            }
            if (use_device_variant_cache) {
                account_device_variant_lease(
                    cache_lease, 0x0fff,
                    exact_engine.external_device_cache_bytes());
            }
            const uint16_t missing = cache_lease
                ? cache_lease.missing_mask() : 0x0fff;
            std::vector<SparseSlabPayload> slabs;
            slabs.reserve(kSlabCount);
            for (int natural = 0; natural < kSlabCount; ++natural) {
                if ((missing & (1u << natural)) == 0) continue;
                SparseSlabPayload slab;
                slab.natural = static_cast<uint16_t>(natural);
                slab.gate.resize(static_cast<size_t>(state.gate_slab_bytes));
                slab.up.resize(static_cast<size_t>(state.up_slab_bytes));
                slab.down.resize(static_cast<size_t>(state.down_slab_bytes));
                slabs.push_back(std::move(slab));
            }
            if (!slabs.empty() && direct_reads()) {
                if (!read_sparse_payloads_direct(
                        sidecar_fd, state, spec, model_layer, base_pos,
                        token_index, expert, static_cast<int>(slabs.size()),
                        true, slabs, err)) {
                    return false;
                }
            } else if (!slabs.empty()) {
                for (SparseSlabPayload & slab : slabs) {
                    const uint64_t record = state.payload_offset +
                        static_cast<uint64_t>(expert * kSlabCount +
                                              slab.natural) *
                            state.slab_bytes;
                    if (!traced_read_exact_at(
                            sidecar_fd, slab.gate.data(), slab.gate.size(),
                            record, model_layer, base_pos, token_index, expert,
                            "gate", ggml_type_name(spec.gate_type),
                            kSlabCount, true, state.sidecar_path,
                            "host-compact-slab", 0) ||
                        !traced_read_exact_at(
                            sidecar_fd, slab.up.data(), slab.up.size(),
                            record + state.gate_slab_bytes, model_layer,
                            base_pos, token_index, expert, "up",
                            ggml_type_name(spec.up_type), kSlabCount, true,
                            state.sidecar_path, "host-compact-slab", 0) ||
                        !traced_read_exact_at(
                            sidecar_fd, slab.down.data(), slab.down.size(),
                            record + state.gate_slab_bytes +
                                state.up_slab_bytes,
                            model_layer, base_pos, token_index, expert, "down",
                            ggml_type_name(spec.down_type), kSlabCount, true,
                            state.sidecar_path, "host-compact-slab", 0)) {
                        if (err && err->empty()) {
                            *err = "short sidecar exact-fallback read";
                        }
                        return false;
                    }
                }
            }
            if (device_result) {
                if (cache_lease) {
                    return evaluate_sparse_payload_cached_device(
                        spec, input, slabs, nullptr, full_mask,
                        down_slab_row_bytes, cache_lease,
                        *device_result, err);
                }
                return evaluate_sparse_payload_device(
                    spec, input, slabs, nullptr, down_slab_row_bytes,
                    0x0fff, *device_result, err);
            }
            return evaluate_sparse_payload(
                spec, input, slabs, nullptr, full_mask,
                down_slab_row_bytes, 0x0fff, result,
                use_device_variant_cache ? &cache_lease : nullptr, err);
        }

        std::vector<uint8_t> gate(gate_full_bytes);
        std::vector<uint8_t> up(up_full_bytes);
        std::vector<uint8_t> down(
            down_full_row_bytes * static_cast<size_t>(spec.output_dim));
        std::vector<uint8_t> slab_down(
            static_cast<size_t>(state.down_slab_bytes));
        for (int natural = 0; natural < kSlabCount; ++natural) {
            const uint64_t record = state.payload_offset +
                static_cast<uint64_t>(expert * kSlabCount + natural) *
                    state.slab_bytes;
            if (!traced_read_exact_at(
                    sidecar_fd,
                    gate.data() + static_cast<size_t>(natural) *
                        state.gate_slab_bytes,
                    static_cast<size_t>(state.gate_slab_bytes), record,
                    model_layer, base_pos, token_index, expert, "gate",
                    ggml_type_name(spec.gate_type), kSlabCount, true,
                    state.sidecar_path, "host-full-width",
                    static_cast<uint64_t>(natural) * state.gate_slab_bytes) ||
                !traced_read_exact_at(
                    sidecar_fd,
                    up.data() + static_cast<size_t>(natural) *
                        state.up_slab_bytes,
                    static_cast<size_t>(state.up_slab_bytes),
                    record + state.gate_slab_bytes, model_layer, base_pos,
                    token_index, expert, "up", ggml_type_name(spec.up_type),
                    kSlabCount, true, state.sidecar_path, "host-full-width",
                    static_cast<uint64_t>(natural) * state.up_slab_bytes) ||
                !traced_read_exact_at(
                    sidecar_fd, slab_down.data(), slab_down.size(),
                    record + state.gate_slab_bytes + state.up_slab_bytes,
                    model_layer, base_pos, token_index, expert, "down",
                    ggml_type_name(spec.down_type), kSlabCount, true,
                    state.sidecar_path, "host-compact-down", 0)) {
                if (err && err->empty()) {
                    *err = "short sidecar exact-fallback read";
                }
                return false;
            }
            for (int dimension = 0; dimension < spec.output_dim;
                 ++dimension) {
                std::memcpy(
                    down.data() + static_cast<size_t>(dimension) *
                        down_full_row_bytes +
                        static_cast<size_t>(natural) * down_slab_row_bytes,
                    slab_down.data() + static_cast<size_t>(dimension) *
                        down_slab_row_bytes,
                    down_slab_row_bytes);
            }
        }
        reference_full_weight_h2d_bytes_ +=
            gate.size() + up.size() + down.size();
        if (device_result) {
            if (err) *err = "P42 exact output requires sparse device layout";
            return false;
        }
        return evaluate_host_sparse_expert(
            backend_, spec, input, gate, up, down, nullptr, result, err);
    }

    bool evaluate_calibrated(
            int model_layer, int base_pos, LayerState & state,
            const MoeStreamExpertSpec & spec,
            const MoeStreamRouteBatch & routes,
            MoeHybridStreamEngine & exact_engine,
            std::vector<float> & output, bool device_ordered,
            bool host_ordered_output,
            std::string * err) {
        const bool use_device_variant_cache =
            device_variant_cache_ && routes.n_tokens > 1;
        if (host_ordered_output &&
            (!device_ordered || routes.n_tokens <= 1 || !routes.inputs ||
             routes.device_inputs || !p40_layer_epoch_ ||
             !use_device_variant_cache || !compact_executor_ ||
             !ordered_device_join_ || !async_compact_queue_ ||
             !p40_wide_async_join_ || !sidecar_authoritative_ ||
             sparse_workspace_ !=
                SparseWorkspace::PersistentDevice ||
             sparse_delivery_ !=
                KimiK3SparseDeliveryPolicy::DirectPinnedCompact)) {
            if (err) {
                *err = "P40 wide async join profile is not qualified";
            }
            return false;
        }
        if (read_cache_.enabled() && model_layer == kFirstRoutedLayer &&
            base_pos == 0) {
            if (cache_sequence_started_) read_cache_.reset_sequence();
            cache_sequence_started_ = true;
        }
        const int layer_budget = budget_for_layer(model_layer);
        const int aux_fd = open_read_only(state.aux_path);
        const int sidecar_fd = direct_reads()
            ? open_read_only_direct(state.sidecar_path)
            : open_read_only(state.sidecar_path);
        if (aux_fd < 0 || sidecar_fd < 0) {
            if (aux_fd >= 0) close_fd(aux_fd);
            if (sidecar_fd >= 0) close_fd(sidecar_fd);
            state.valid = false;
            if (sidecar_authoritative_) {
                if (err) *err =
                    "sidecar-authoritative provider cannot open a required artifact";
                return false;
            }
            const bool ok = eval_moe_streamed_experts(
                exact_engine, spec, routes, output, err);
            if (ok) observe_exact_layer(
                state, routes.n_tokens, spec, layer_budget);
            return ok;
        }
        const auto exact_layer_fallback = [&](const char * reason) {
#if defined(DFLASH27B_BACKEND_CUDA) || defined(DFLASH27B_BACKEND_HIP)
            if (device_ordered) {
                sparse_device_evaluator_.abort_compact_async_batch();
                ordered_join_arena_.discard();
            }
#endif
            close_fd(aux_fd);
            close_fd(sidecar_fd);
            state.valid = false;
            if (sidecar_authoritative_) {
                if (err) {
                    *err = std::string(
                        "sidecar-authoritative provider failed closed: ") +
                        reason;
                }
                return false;
            }
            std::string exact_error;
            const bool ok = eval_moe_streamed_experts(
                exact_engine, spec, routes, output, &exact_error);
            if (ok) {
                observe_exact_layer(
                    state, routes.n_tokens, spec, layer_budget);
                if (err) err->clear();
                std::fprintf(stderr,
                    "[kimi-k3-calibrated96] layer=%d action=exact "
                    "runtime-reason=%s\n", model_layer, reason);
            } else if (err) {
                *err = std::string(reason) + "; exact fallback failed: " +
                    exact_error;
            }
            return ok;
        };
        const auto fail_device_join = [&]() {
#if defined(DFLASH27B_BACKEND_CUDA) || defined(DFLASH27B_BACKEND_HIP)
            sparse_device_evaluator_.abort_compact_async_batch();
            ordered_join_arena_.discard();
#endif
            close_fd(aux_fd);
            close_fd(sidecar_fd);
            return false;
        };
        const size_t gate_full_bytes = static_cast<size_t>(
            state.gate_slab_bytes * kSlabCount);
        const size_t up_full_bytes = static_cast<size_t>(
            state.up_slab_bytes * kSlabCount);
        const size_t down_slab_row_bytes = static_cast<size_t>(
            state.down_slab_bytes / spec.output_dim);
        const size_t down_full_row_bytes = down_slab_row_bytes * kSlabCount;
        std::vector<uint8_t> gate(
            sparse_device() ? 0 : gate_full_bytes, 0);
        std::vector<uint8_t> up(
            sparse_device() ? 0 : up_full_bytes, 0);
        std::vector<uint8_t> down(sparse_device() ? 0 :
            down_full_row_bytes * static_cast<size_t>(spec.output_dim), 0);
        std::vector<uint8_t> slab_down(
            static_cast<size_t>(state.down_slab_bytes));
        std::vector<float> mask(
            static_cast<size_t>(spec.intermediate_dim), 0.0f);
        std::vector<float> means(
            device_ordered ? 0 : static_cast<size_t>(kSlabCount * kDimension));
        std::vector<float> expert_output;
        if (device_ordered) {
#if defined(DFLASH27B_BACKEND_CUDA) || defined(DFLASH27B_BACKEND_HIP)
            if (host_ordered_output) {
                output.assign(
                    static_cast<size_t>(routes.n_tokens) * spec.output_dim,
                    0.0f);
            } else {
                output.clear();
            }
#else
            if (err) *err = "P42 ordered join requires a GPU backend";
            return fail_device_join();
#endif
        } else {
            output.assign(
                static_cast<size_t>(routes.n_tokens) * spec.output_dim, 0.0f);
        }

        for (int token = 0; token < routes.n_tokens; ++token) {
            const size_t route_offset =
                static_cast<size_t>(token) * kNativeTopK;
            std::vector<uint8_t> effective_calibrated = state.calibrated;
            if (layer_budget == kNativeTopK * kSlabCount) {
                std::fill(
                    effective_calibrated.begin(),
                    effective_calibrated.end(), static_cast<uint8_t>(1));
            }
            const KimiK3CalibratedSlabPlan plan =
                plan_kimi_k3_calibrated_slabs(
                    routes.selected_ids + route_offset,
                    routes.selected_weights + route_offset, kNativeTopK,
                    state.importance.data(), effective_calibrated.data(),
                    kExpertCount, kSlabCount, layer_budget);
            std::vector<int> calibrated_routes;
            std::vector<int> fallback_routes;
            for (int route = 0; route < kNativeTopK; ++route) {
                const int expert = routes.selected_ids[route_offset + route];
                if (expert < 0 || expert >= kExpertCount) {
                    close_fd(aux_fd); close_fd(sidecar_fd);
                    if (err) *err = "calibrated96 saw an invalid expert id";
                    return false;
                }
                (effective_calibrated[static_cast<size_t>(expert)]
                    ? calibrated_routes : fallback_routes).push_back(route);
            }
            const int selected_count = static_cast<int>(
                plan.selected_slab_ids.size());
            if (plan.exact_route_indices.size() != fallback_routes.size() ||
                !std::equal(plan.exact_route_indices.begin(),
                            plan.exact_route_indices.end(),
                            fallback_routes.begin())) {
                return exact_layer_fallback("calibrated96 planner failed");
            }
            std::vector<uint8_t> selected_by_route(
                static_cast<size_t>(kNativeTopK * kSlabCount), 0);
            for (const int pseudo : plan.selected_slab_ids) {
                const int expert = pseudo / kSlabCount;
                const int rank = pseudo % kSlabCount;
                const auto found = std::find_if(
                    calibrated_routes.begin(), calibrated_routes.end(),
                    [&](int route) {
                        return routes.selected_ids[route_offset + route] == expert;
                    });
                if (found == calibrated_routes.end()) {
                    return exact_layer_fallback(
                        "selected slab has no calibrated route");
                }
                selected_by_route[
                    static_cast<size_t>(*found * kSlabCount + rank)] = 1;
            }
            std::vector<std::vector<SparseSlabPayload>> direct_payloads;
            std::vector<uint8_t> read_selected_by_route = selected_by_route;
            std::array<MoeStreamExternalLease, kNativeTopK> cache_leases;
            if (direct_reads() && use_device_variant_cache) {
                for (const int route : calibrated_routes) {
                    const int expert =
                        routes.selected_ids[route_offset + route];
                    const uint16_t * natural_by_rank = state.order.data() +
                        static_cast<size_t>(expert) * kSlabCount;
                    const uint8_t * selected_by_rank =
                        selected_by_route.data() +
                        static_cast<size_t>(route) * kSlabCount;
                    const uint16_t requested_mask =
                        kimi_k3_selected_natural_slab_mask(
                            natural_by_rank, selected_by_rank, kSlabCount);
                    if (requested_mask == 0) continue;
                    const MoeStreamExternalKey cache_key{
                        kNaturalSidecarCacheDomain, state.source_generation,
                        routes.layer, expert, spec};
                    if (!sparse_device_evaluator_.acquire_cache_lease(
                            backend_, spec, requested_mask != 0x0fff,
                            exact_engine, cache_key, requested_mask,
                            cache_leases[static_cast<size_t>(route)], err)) {
                        return exact_layer_fallback(
                            "device variant cache acquisition failed");
                    }
                    const MoeStreamExternalLease & lease =
                        cache_leases[static_cast<size_t>(route)];
                    account_device_variant_lease(
                        lease, requested_mask,
                        exact_engine.external_device_cache_bytes());
                    if (!lease) continue;
                    const uint16_t missing = lease.missing_mask();
                    kimi_k3_suppress_resident_slab_ranks(
                        natural_by_rank, missing,
                        read_selected_by_route.data() +
                            static_cast<size_t>(route) * kSlabCount,
                        kSlabCount);
                }
            }
            if (direct_reads() &&
                !read_sparse_payloads_direct_batch(
                    sidecar_fd, state, spec, model_layer, base_pos, token,
                    route_offset, routes, calibrated_routes,
                    read_selected_by_route, direct_payloads,
                    sparse_delivery_ ==
                        KimiK3SparseDeliveryPolicy::DirectPinnedCompact
                        ? &direct_compact_payloads_ : nullptr,
                    compact_executor_ && !use_device_variant_cache,
                    err)) {
                return exact_layer_fallback(
                    "P20 direct layer-batch sidecar read failed");
            }
            std::vector<int> stable_routes = calibrated_routes;
            std::stable_sort(stable_routes.begin(), stable_routes.end(),
                [&](int left, int right) {
                    return routes.selected_ids[route_offset + left] <
                        routes.selected_ids[route_offset + right];
                });
#if defined(DFLASH27B_BACKEND_CUDA) || defined(DFLASH27B_BACKEND_HIP)
            if (device_ordered &&
                !ordered_join_arena_.begin(
                    backend_, spec.output_dim, err)) {
                return fail_device_join();
            }
            if (device_ordered && async_compact_queue_ &&
                !sparse_device_evaluator_.begin_compact_async_batch(
                    backend_, kNativeTopK, err,
                    host_ordered_output ? nullptr : routes.device_inputs)) {
                return fail_device_join();
            }
#endif
            float * destination = device_ordered ? nullptr : output.data() +
                static_cast<size_t>(token) * spec.output_dim;
            const float * input = routes.inputs
                ? routes.inputs + static_cast<size_t>(token) * spec.input_dim
                : nullptr;
            for (const int route : stable_routes) {
                const int expert = routes.selected_ids[route_offset + route];
                const uint64_t mean_offset = state.means_offset +
                    static_cast<uint64_t>(expert) * kSlabCount * kDimension *
                        sizeof(float);
                const int prefix_depth = static_cast<int>(std::count(
                    selected_by_route.begin() +
                        static_cast<ptrdiff_t>(route * kSlabCount),
                    selected_by_route.begin() +
                        static_cast<ptrdiff_t>((route + 1) * kSlabCount),
                    static_cast<uint8_t>(1)));
                const float weight =
                    routes.selected_weights[route_offset + route];
                if (!device_ordered && !traced_read_exact_at(
                        aux_fd, means.data(), means.size() * sizeof(float),
                        mean_offset, model_layer, base_pos, token, expert,
                        "slab-mean", "f32", prefix_depth, false,
                        state.aux_path, "host-mean", 0)) {
                    return exact_layer_fallback("short calibrated mean read");
                }
                // Add omitted cards directly.  Avoid add/subtract cancellation
                // so the full-width path has stable arithmetic.
                for (int rank = 0; rank < kSlabCount; ++rank) {
                    if (selected_by_route[
                            static_cast<size_t>(route * kSlabCount + rank)]) {
                        continue;
                    }
                    if (device_ordered) {
#if defined(DFLASH27B_BACKEND_CUDA) || defined(DFLASH27B_BACKEND_HIP)
                        if (!ordered_join_arena_.append_resident_mean(
                                model_layer, expert, rank, weight, err)) {
                            return fail_device_join();
                        }
#endif
                    } else {
                        const float * mean = means.data() +
                            static_cast<size_t>(rank) * kDimension;
                        for (int d = 0; d < kDimension; ++d) {
                            destination[d] += weight * mean[d];
                        }
                    }
                }

                std::fill(gate.begin(), gate.end(), 0);
                std::fill(up.begin(), up.end(), 0);
                std::fill(down.begin(), down.end(), 0);
                std::fill(mask.begin(), mask.end(), 0.0f);
                std::vector<SparseSlabPayload> sparse_slabs =
                    direct_reads()
                    ? std::move(direct_payloads[static_cast<size_t>(route)])
                    : std::vector<SparseSlabPayload>{};
                const SparseCompactPayload * prepacked_compact = nullptr;
                if (sparse_delivery_ ==
                        KimiK3SparseDeliveryPolicy::DirectPinnedCompact) {
                    prepacked_compact = &direct_compact_payloads_[
                        static_cast<size_t>(route)];
                }
                sparse_slabs.reserve(kSlabCount);
                const int retained = prefix_depth;
                for (int rank = 0; rank < kSlabCount; ++rank) {
                    if (!selected_by_route[static_cast<size_t>(
                            route * kSlabCount + rank)]) continue;
                    const uint16_t natural = state.order[
                        static_cast<size_t>(expert) * kSlabCount + rank];
                    std::fill_n(mask.begin() +
                        static_cast<size_t>(natural) * kSlabSize,
                        kSlabSize, 1.0f);
                }
                for (int rank = 0; !direct_reads() && rank < kSlabCount;
                     ++rank) {
                    if (!selected_by_route[
                            static_cast<size_t>(route * kSlabCount + rank)]) {
                        continue;
                    }
                    const uint16_t natural = state.order[
                        static_cast<size_t>(expert) * kSlabCount + rank];
                    const uint64_t record = state.payload_offset +
                        static_cast<uint64_t>(expert * kSlabCount + natural) *
                            state.slab_bytes;
                    SparseSlabPayload sparse;
                    if (sparse_device()) {
                        sparse.natural = natural;
                        sparse.gate.resize(
                            static_cast<size_t>(state.gate_slab_bytes));
                        sparse.up.resize(
                            static_cast<size_t>(state.up_slab_bytes));
                        sparse.down.resize(
                            static_cast<size_t>(state.down_slab_bytes));
                    }
                    uint8_t * gate_destination = sparse_device()
                        ? sparse.gate.data()
                        : gate.data() + static_cast<size_t>(natural) *
                            state.gate_slab_bytes;
                    uint8_t * up_destination = sparse_device()
                        ? sparse.up.data()
                        : up.data() + static_cast<size_t>(natural) *
                            state.up_slab_bytes;
                    uint8_t * down_destination = sparse_device()
                        ? sparse.down.data() : slab_down.data();
                    const char * host_destination = sparse_device()
                        ? "host-compact-slab" : "host-full-width";
                    if (!traced_read_exact_at(
                            sidecar_fd, gate_destination,
                            static_cast<size_t>(state.gate_slab_bytes), record,
                            model_layer, base_pos, token, expert, "gate",
                            ggml_type_name(spec.gate_type), prefix_depth, false,
                            state.sidecar_path, host_destination,
                            sparse_device() ? 0 :
                                static_cast<uint64_t>(natural) *
                                    state.gate_slab_bytes) ||
                        !traced_read_exact_at(
                            sidecar_fd, up_destination,
                            static_cast<size_t>(state.up_slab_bytes),
                            record + state.gate_slab_bytes, model_layer,
                            base_pos, token, expert, "up",
                            ggml_type_name(spec.up_type), prefix_depth, false,
                            state.sidecar_path, host_destination,
                            sparse_device() ? 0 :
                                static_cast<uint64_t>(natural) *
                                    state.up_slab_bytes) ||
                        !traced_read_exact_at(
                            sidecar_fd, down_destination,
                            static_cast<size_t>(state.down_slab_bytes),
                            record + state.gate_slab_bytes +
                                state.up_slab_bytes, model_layer, base_pos,
                            token, expert, "down",
                            ggml_type_name(spec.down_type), prefix_depth, false,
                            state.sidecar_path,
                            sparse_device() ? "host-compact-slab" :
                                "host-compact-down", 0)) {
                        return exact_layer_fallback(
                            "short mixed-layout slab read");
                    }
                    if (sparse_device()) {
                        sparse_slabs.push_back(std::move(sparse));
                    } else {
                        for (int d = 0; d < spec.output_dim; ++d) {
                            std::memcpy(
                                down.data() + static_cast<size_t>(d) *
                                    down_full_row_bytes +
                                    static_cast<size_t>(natural) *
                                        down_slab_row_bytes,
                                slab_down.data() + static_cast<size_t>(d) *
                                    down_slab_row_bytes,
                                down_slab_row_bytes);
                        }
                    }
                }
                const uint16_t requested_mask =
                    kimi_k3_selected_natural_slab_mask(
                        state.order.data() +
                            static_cast<size_t>(expert) * kSlabCount,
                        selected_by_route.data() +
                            static_cast<size_t>(route) * kSlabCount,
                        kSlabCount);
                ggml_tensor * device_expert_output = nullptr;
                bool evaluated = retained == 0;
                if (retained > 0 && device_ordered) {
                    evaluated = use_device_variant_cache
                        ? evaluate_sparse_payload_cached_device(
                            spec, input, sparse_slabs, prepacked_compact, mask,
                            down_slab_row_bytes,
                            cache_leases[static_cast<size_t>(route)],
                            device_expert_output, err)
                        : evaluate_sparse_payload_device(
                            spec, input, sparse_slabs, prepacked_compact,
                            down_slab_row_bytes, requested_mask,
                            device_expert_output, err);
                } else if (retained > 0 && sparse_device()) {
                    evaluated = evaluate_sparse_payload(
                        spec, input, sparse_slabs, prepacked_compact, mask,
                        down_slab_row_bytes, requested_mask, expert_output,
                        use_device_variant_cache
                            ? &cache_leases[static_cast<size_t>(route)]
                            : nullptr,
                        err);
                } else if (retained > 0) {
                    evaluated = evaluate_host_sparse_expert(
                        backend_, spec, input, gate, up, down,
                        retained == kSlabCount ? nullptr : &mask,
                        expert_output, err);
                }
                if (retained > 0 && !evaluated) {
                    const std::string detail = err && !err->empty()
                        ? "full-width recomposition failed: " + *err
                        : "full-width recomposition failed";
                    return exact_layer_fallback(detail.c_str());
                }
                if (!sparse_device() && retained > 0) {
                    reference_full_weight_h2d_bytes_ +=
                        gate.size() + up.size() + down.size();
                }
                if (retained > 0) {
                    if (device_ordered) {
#if defined(DFLASH27B_BACKEND_CUDA) || defined(DFLASH27B_BACKEND_HIP)
                        int expert_row = -1;
                        if (!ordered_join_arena_.stage_device_output(
                                device_expert_output, expert_row, err) ||
                            !ordered_join_arena_.append(
                                expert_row, weight, err)) {
                            return fail_device_join();
                        }
#endif
                    } else {
                        for (int d = 0; d < spec.output_dim; ++d) {
                            destination[d] += weight *
                                expert_output[static_cast<size_t>(d)];
                        }
                    }
                }
            }

            if (device_ordered) {
#if defined(DFLASH27B_BACKEND_CUDA) || defined(DFLASH27B_BACKEND_HIP)
                ordered_join_arena_.seal_calibrated();
#endif
            }
            if (!fallback_routes.empty()) {
                std::vector<float> exact;
                if (sidecar_authoritative_) {
                    if (!device_ordered) {
                        exact.assign(
                            static_cast<size_t>(spec.output_dim), 0.0f);
                    }
                    std::vector<int> stable_fallback = fallback_routes;
                    std::stable_sort(
                        stable_fallback.begin(), stable_fallback.end(),
                        [&](int left, int right) {
                            return routes.selected_ids[route_offset + left] <
                                routes.selected_ids[route_offset + right];
                        });
                    std::vector<float> exact_expert;
                    for (const int route : stable_fallback) {
                        const int expert =
                            routes.selected_ids[route_offset + route];
                        ggml_tensor * device_exact_output = nullptr;
                        if (!evaluate_sidecar_exact_expert(
                                sidecar_fd, model_layer, base_pos, token,
                                state, spec, routes.layer, expert, input,
                                exact_engine, use_device_variant_cache,
                                device_ordered && use_device_variant_cache
                                    ? &cache_leases[
                                        static_cast<size_t>(route)]
                                    : nullptr,
                                exact_expert,
                                device_ordered ? &device_exact_output : nullptr,
                                err)) {
                            return fail_device_join();
                        }
                        const float weight =
                            routes.selected_weights[route_offset + route];
                        if (device_ordered) {
#if defined(DFLASH27B_BACKEND_CUDA) || defined(DFLASH27B_BACKEND_HIP)
                            int exact_row = -1;
                            if (!ordered_join_arena_.stage_device_output(
                                    device_exact_output, exact_row, err) ||
                                !ordered_join_arena_.append(
                                    exact_row, weight, err)) {
                                return fail_device_join();
                            }
#endif
                        } else {
                            for (int d = 0; d < spec.output_dim; ++d) {
                                exact[static_cast<size_t>(d)] += weight *
                                    exact_expert[static_cast<size_t>(d)];
                            }
                        }
                    }
                } else {
                    if (device_ordered) {
                        if (err) {
                            *err = "P42 ordered join requires authoritative fallback";
                        }
                        return fail_device_join();
                    }
                    std::vector<int32_t> fallback_ids;
                    std::vector<float> fallback_weights;
                    for (const int route : fallback_routes) {
                        fallback_ids.push_back(
                            routes.selected_ids[route_offset + route]);
                        fallback_weights.push_back(
                            routes.selected_weights[route_offset + route]);
                    }
                    MoeStreamRouteBatch fallback = routes;
                    fallback.n_tokens = 1;
                    fallback.top_k =
                        static_cast<int>(fallback_routes.size());
                    fallback.inputs = input;
                    fallback.selected_ids = fallback_ids.data();
                    fallback.selected_weights = fallback_weights.data();
                    fallback.expert_observer = nullptr;
                    for (const int route : fallback_routes) {
                        trace_fallback(model_layer, base_pos, token,
                            routes.selected_ids[route_offset + route], spec);
                    }
                    if (!eval_moe_streamed_experts(
                            exact_engine, spec, fallback, exact, err)) {
                        close_fd(aux_fd); close_fd(sidecar_fd);
                        return false;
                    }
                }
                if (!device_ordered) {
                    for (int d = 0; d < spec.output_dim; ++d) {
                        destination[d] += exact[static_cast<size_t>(d)];
                    }
                }
            }

            if (device_ordered) {
#if defined(DFLASH27B_BACKEND_CUDA) || defined(DFLASH27B_BACKEND_HIP)
                const bool joined = ordered_join_arena_.finish(err);
                const bool queue_finished = !async_compact_queue_ ||
                    sparse_device_evaluator_.
                        complete_compact_async_batch_after_sync(err);
                if (!joined || !queue_finished) {
                    return fail_device_join();
                }
                if (host_ordered_output &&
                    !ordered_join_arena_.read_to_host(
                        output.data() + static_cast<size_t>(token) *
                            spec.output_dim,
                        static_cast<size_t>(spec.output_dim), err)) {
                    return fail_device_join();
                }
#endif
            }

            ++state.traffic.tokens;
            state.traffic.requested_nominal_slabs += layer_budget;
            state.traffic.selected_slab_records += selected_count;
            state.traffic.calibrated_routes += calibrated_routes.size();
            state.traffic.exact_fallback_routes += fallback_routes.size();
            state.traffic.selected_sidecar_bytes +=
                static_cast<uint64_t>(selected_count) * state.slab_bytes;
            state.traffic.exact_fallback_bytes +=
                static_cast<uint64_t>(fallback_routes.size()) *
                    state.record_bytes;
        }
        close_fd(aux_fd);
        close_fd(sidecar_fd);
        (void) model_layer;
        return true;
    }

    void finish_metrics() {
        if (layers_.empty()) return;
        std::ostringstream report;
        report << "model_layer\ttokens\trequested_nominal_slabs"
                  "\tselected_slab_records\tcalibrated_routes"
                  "\texact_fallback_routes\tselected_sidecar_bytes"
                  "\texact_fallback_bytes\ttotal_provider_bytes\n";
        for (int layer = kFirstRoutedLayer; layer <= kLastRoutedLayer; ++layer) {
            const Traffic & value =
                layers_[static_cast<size_t>(layer)].traffic;
            if (value.tokens == 0) continue;
            report << layer << '\t' << value.tokens << '\t'
                   << value.requested_nominal_slabs << '\t'
                   << value.selected_slab_records << '\t'
                   << value.calibrated_routes << '\t'
                   << value.exact_fallback_routes << '\t'
                   << value.selected_sidecar_bytes << '\t'
                   << value.exact_fallback_bytes << '\t'
                   << value.selected_sidecar_bytes + value.exact_fallback_bytes
                   << '\n';
        }
        if (metrics_path_.empty()) {
            std::fprintf(stderr, "%s", report.str().c_str());
        } else {
            std::ofstream output(metrics_path_);
            output << report.str();
            if (!output) {
                std::fprintf(stderr,
                    "[kimi-k3-calibrated96] cannot write traffic metrics %s\n",
                    metrics_path_.c_str());
            }
        }
        if (io_trace_) {
            io_trace_.flush();
            const ProcessIoSnapshot end = process_io_snapshot();
            std::ofstream summary(metrics_path_.empty()
                ? "k3_p20_io_process.tsv"
                : metrics_path_ + ".process.tsv");
            summary << "explicit_provider_read_bytes\tprocess_read_bytes"
                       "\tprocess_rchar\tprocess_read_syscalls"
                       "\tminor_faults\tmajor_faults\n"
                    << explicit_read_bytes_ << '\t'
                    << saturating_delta(
                           end.read_bytes, process_io_start_.read_bytes) << '\t'
                    << saturating_delta(end.rchar, process_io_start_.rchar)
                    << '\t'
                    << saturating_delta(end.syscr, process_io_start_.syscr)
                    << '\t'
                    << saturating_delta(
                           end.minor_faults, process_io_start_.minor_faults)
                    << '\t'
                    << saturating_delta(
                           end.major_faults, process_io_start_.major_faults)
                    << '\n';
        }
        std::fprintf(stderr,
            "[kimi-k3-p20] physical-layout=%s "
            "reference-full-weight-h2d=%llu "
            "sparse-authoritative-h2d=%llu metadata-h2d=%llu "
            "device-zero-bytes=%llu explicit-provider-reads=%llu "
            "direct-physical-bytes=%llu direct-io-ns=%llu "
            "compact-pack-ns=%llu compact-scatter-ns=%llu "
            "expert-graph-ns=%llu expert-readback-ns=%llu\n",
            sparse_device() ? "scratch" : "reference",
            static_cast<unsigned long long>(reference_full_weight_h2d_bytes_),
            static_cast<unsigned long long>(authoritative_h2d_bytes_),
            static_cast<unsigned long long>(metadata_h2d_bytes_),
            static_cast<unsigned long long>(device_zero_bytes_),
            static_cast<unsigned long long>(explicit_read_bytes_),
            static_cast<unsigned long long>(direct_physical_bytes_),
            static_cast<unsigned long long>(direct_io_ns_),
            static_cast<unsigned long long>(
                sparse_device_evaluator_.compact_pack_ns()),
            static_cast<unsigned long long>(
                sparse_device_evaluator_.compact_scatter_ns()),
            static_cast<unsigned long long>(
                sparse_device_evaluator_.expert_graph_ns()),
            static_cast<unsigned long long>(
                sparse_device_evaluator_.expert_readback_ns()));
        if (read_cache_.enabled()) {
            const P30BoundedReadCache::Stats cache = read_cache_.stats();
            std::fprintf(stderr,
                "[kimi-k3-p30] capacity-bytes=%zu resident-bytes=%zu "
                "entries=%zu hits=%llu misses=%llu hit-bytes=%llu "
                "inserted-bytes=%llu evicted-bytes=%llu sequence-resets=%llu\n",
                read_cache_.capacity(), cache.resident_bytes, cache.entries,
                static_cast<unsigned long long>(cache.hits),
                static_cast<unsigned long long>(cache.misses),
                static_cast<unsigned long long>(cache.hit_bytes),
                static_cast<unsigned long long>(cache.inserted_bytes),
                static_cast<unsigned long long>(cache.evicted_bytes),
                static_cast<unsigned long long>(cache.sequence_resets));
        }
        if (device_variant_cache_) {
            std::fprintf(stderr,
                "[kimi-k3-p40] requested-slabs=%llu resident-before-slabs=%llu "
                "hits=%llu extensions=%llu cold=%llu unavailable=%llu "
                "completed=%llu aborted=%llu fallbacks=%llu "
                "evictions=%llu device-bytes=%zu h2d-bytes=%llu "
                "scatter-calls=%llu scatter-avoided=%llu\n",
                static_cast<unsigned long long>(p40_requested_slabs_),
                static_cast<unsigned long long>(p40_resident_before_slabs_),
                static_cast<unsigned long long>(p40_hits_),
                static_cast<unsigned long long>(p40_extensions_),
                static_cast<unsigned long long>(p40_cold_),
                static_cast<unsigned long long>(p40_unavailable_),
                static_cast<unsigned long long>(p40_completed_),
                static_cast<unsigned long long>(p40_aborted_),
                static_cast<unsigned long long>(p40_fallbacks_),
                static_cast<unsigned long long>(p40_evictions_),
                p40_device_bytes_,
                static_cast<unsigned long long>(p40_h2d_bytes_),
                static_cast<unsigned long long>(p40_scatter_calls_),
                static_cast<unsigned long long>(p40_scatter_avoided_));
        }
        if (compact_executor_) {
            std::fprintf(stderr,
                "[kimi-k3-p41] attempted=%llu layouts=%llu uploads=%llu "
                "gate=%llu up=%llu situ=%llu sparse-down=%llu "
                "completed=%llu fallbacks=%llu invalid=%llu\n",
                static_cast<unsigned long long>(p41_attempted_),
                static_cast<unsigned long long>(
                    sparse_device_evaluator_.compact_layouts()),
                static_cast<unsigned long long>(
                    sparse_device_evaluator_.compact_uploads()),
                static_cast<unsigned long long>(
                    sparse_device_evaluator_.compact_gate_stages()),
                static_cast<unsigned long long>(
                    sparse_device_evaluator_.compact_up_stages()),
                static_cast<unsigned long long>(
                    sparse_device_evaluator_.compact_situ_stages()),
                static_cast<unsigned long long>(
                    sparse_device_evaluator_.compact_down_stages()),
                static_cast<unsigned long long>(p41_completed_),
                static_cast<unsigned long long>(p41_fallbacks_),
                static_cast<unsigned long long>(p41_invalid_));
        }
        if (async_compact_queue_) {
            const SparseDeviceExpertEvaluator::CompactAsyncStats & stats =
                sparse_device_evaluator_.compact_async_stats();
            std::fprintf(stderr,
                "[kimi-k3-p45] begins=%llu jobs=%llu h2d-calls=%llu "
                "h2d-bytes=%llu input-d2d-copies=%llu "
                "input-d2d-bytes=%llu graph-enqueues=%llu layer-flushes=%llu "
                "abort-syncs=%llu max-inflight=%llu submit-ns=%llu "
                "device-window-ns=%llu\n",
                static_cast<unsigned long long>(stats.begins),
                static_cast<unsigned long long>(stats.jobs),
                static_cast<unsigned long long>(stats.h2d_calls),
                static_cast<unsigned long long>(stats.h2d_bytes),
                static_cast<unsigned long long>(stats.input_d2d_copies),
                static_cast<unsigned long long>(stats.input_d2d_bytes),
                static_cast<unsigned long long>(stats.graph_enqueues),
                static_cast<unsigned long long>(stats.layer_flushes),
                static_cast<unsigned long long>(stats.abort_syncs),
                static_cast<unsigned long long>(stats.max_inflight),
                static_cast<unsigned long long>(stats.submit_ns),
                static_cast<unsigned long long>(stats.device_window_ns));
        }
#if defined(DFLASH27B_BACKEND_CUDA) || defined(DFLASH27B_BACKEND_HIP)
        if (ordered_device_join_) {
            std::fprintf(stderr,
                "[kimi-k3-p42c] resident-mean-bytes=%llu "
                "hot-mean-reads=0 hot-mean-h2d-bytes=0 "
                "expert-d2d-copies=%llu expert-d2d-bytes=%llu "
                "join-launches=%llu output-d2d-copies=%llu "
                "output-d2d-bytes=%llu\n",
                static_cast<unsigned long long>(
                    ordered_join_arena_.resident_mean_bytes()),
                static_cast<unsigned long long>(
                    ordered_join_arena_.expert_d2d_copies()),
                static_cast<unsigned long long>(
                    ordered_join_arena_.expert_d2d_bytes()),
                static_cast<unsigned long long>(
                    ordered_join_arena_.join_launches()),
                static_cast<unsigned long long>(
                    ordered_join_arena_.output_d2d_copies()),
                static_cast<unsigned long long>(
                    ordered_join_arena_.output_d2d_bytes()));
        }
#endif
        layers_.clear();
    }

    static constexpr int kFirstRoutedLayer = 1;
    static constexpr int kLastRoutedLayer = 92;
    ggml_backend_t backend_ = nullptr;
    std::vector<LayerState> layers_;
    std::string metrics_path_;
    std::ofstream io_trace_;
    std::ofstream p40_trace_;
    std::string prompt_id_ = "0";
    ProcessIoSnapshot process_io_start_{};
    uint64_t next_request_id_ = 0;
    uint64_t explicit_read_bytes_ = 0;
    SparseWorkspace sparse_workspace_ = SparseWorkspace::HostRecomposed;
    SparseDeviceExpertEvaluator sparse_device_evaluator_;
    KimiK3SparseDeliveryPolicy sparse_delivery_ =
        KimiK3SparseDeliveryPolicy::BufferedSlabs;
    bool device_variant_cache_ = false;
    bool p40_layer_epoch_ = false;
    bool compact_executor_ = false;
    bool ordered_device_join_ = false;
    bool async_compact_queue_ = false;
    bool p40_wide_async_join_ = false;
#if defined(DFLASH27B_BACKEND_CUDA) || defined(DFLASH27B_BACKEND_HIP)
    P42OrderedJoinArena ordered_join_arena_;
#endif
    size_t p40_device_bytes_ = 0;
    size_t p40_requested_device_bytes_ = 0;
    uint64_t p40_requested_slabs_ = 0;
    uint64_t p40_resident_before_slabs_ = 0;
    uint64_t p40_hits_ = 0;
    uint64_t p40_extensions_ = 0;
    uint64_t p40_cold_ = 0;
    uint64_t p40_unavailable_ = 0;
    uint64_t p40_completed_ = 0;
    uint64_t p40_aborted_ = 0;
    uint64_t p40_fallbacks_ = 0;
    uint64_t p40_evictions_ = 0;
    uint64_t p40_h2d_bytes_ = 0;
    uint64_t p40_scatter_calls_ = 0;
    uint64_t p40_scatter_avoided_ = 0;
    uint64_t p41_attempted_ = 0;
    uint64_t p41_completed_ = 0;
    uint64_t p41_fallbacks_ = 0;
    uint64_t p41_invalid_ = 0;
    bool sidecar_authoritative_ = false;
    std::unique_ptr<P20DirectReadPool> direct_read_pool_;
    std::array<SparseCompactPayload, kNativeTopK>
        direct_compact_payloads_;
    int budget_ = 96;
    std::vector<int32_t> layer_budgets_;
    std::string layer_budget_path_;
    uint64_t reference_full_weight_h2d_bytes_ = 0;
    uint64_t authoritative_h2d_bytes_ = 0;
    uint64_t metadata_h2d_bytes_ = 0;
    uint64_t device_zero_bytes_ = 0;
    uint64_t direct_physical_bytes_ = 0;
    uint64_t direct_io_ns_ = 0;
    P30BoundedReadCache read_cache_;
    bool cache_sequence_started_ = false;
};

bool parse_positive_int(const char * raw, int & value) {
    if (!raw || !*raw) return false;
    char * end = nullptr;
    const long parsed = std::strtol(raw, &end, 10);
    if (end == raw || *end != '\0' || parsed <= 0 ||
        parsed > std::numeric_limits<int>::max()) {
        return false;
    }
    value = static_cast<int>(parsed);
    return true;
}

#if defined(DFLASH_KIMI_P45_ASYNC_TEST_HOOK)
#include "../../test/kimi_k3_p45_async_compact_sentinel.inc"
#endif

} // namespace

#if defined(DFLASH_KIMI_P45_ASYNC_TEST_HOOK)
bool kimi_k3_run_p45_async_compact_sentinel(
        ggml_backend_t backend, std::string * err) {
    return p45_async_compact_sentinel_impl(backend, err) &&
        compact_union_owned_buffer_sentinel_impl(backend, err);
}
#endif


bool parse_kimi_k3_layer_budget_table(
        const std::string & path, std::vector<int32_t> & budgets,
        std::string * err) {
    constexpr int kLayers = 92;
    const auto allowed = [](int budget) {
        return budget >= 24 && budget <= 192 && budget % 24 == 0;
    };
    std::ifstream input(path);
    if (!input) {
        if (err) *err = "cannot open H22 layer budget table " + path;
        return false;
    }
    std::vector<int32_t> parsed(kLayers, 0);
    std::vector<uint8_t> seen(kLayers, 0);
    std::string line;
    int line_number = 0;
    while (std::getline(input, line)) {
        ++line_number;
        const size_t comment = line.find('#');
        if (comment != std::string::npos) line.resize(comment);
        std::istringstream row(line);
        int layer = 0;
        int budget = 0;
        if (!(row >> layer)) continue;
        std::string trailing;
        if (!(row >> budget) || (row >> trailing) || layer < 1 ||
            layer > kLayers || !allowed(budget) ||
            seen[static_cast<size_t>(layer - 1)] != 0) {
            if (err) *err = "invalid H22 layer budget row " +
                std::to_string(line_number);
            return false;
        }
        parsed[static_cast<size_t>(layer - 1)] = budget;
        seen[static_cast<size_t>(layer - 1)] = 1;
    }
    if (!input.eof() || std::any_of(
            seen.begin(), seen.end(), [](uint8_t value) { return value == 0; })) {
        if (err) *err = "H22 layer budget table must name layers 1..92 exactly once";
        return false;
    }
    budgets = std::move(parsed);
    return true;
}

std::vector<int32_t> select_kimi_k3_slab_prefix_ids(
        const int32_t * expert_ids, const float * router_weights, int top_k,
        const float * ordered_importance, int expert_count,
        int slabs_per_expert, int budget) {
    if (!expert_ids || !router_weights || !ordered_importance || top_k <= 0 ||
        expert_count <= 0 || slabs_per_expert <= 0 || budget <= 0 ||
        budget > top_k * slabs_per_expert) {
        return {};
    }
    std::vector<Candidate> candidates;
    candidates.reserve(static_cast<size_t>(top_k * slabs_per_expert));
    for (int route = 0; route < top_k; ++route) {
        const int expert = expert_ids[route];
        if (expert < 0 || expert >= expert_count) return {};
        for (int rank = 0; rank < slabs_per_expert; ++rank) {
            candidates.push_back({
                std::abs(router_weights[route]) *
                    ordered_importance[
                        static_cast<size_t>(expert) * slabs_per_expert + rank],
                route, expert, rank});
        }
    }
    std::stable_sort(candidates.begin(), candidates.end(), better_candidate);
    std::vector<int32_t> selected;
    selected.reserve(static_cast<size_t>(budget));
    for (int i = 0; i < budget; ++i) {
        selected.push_back(
            candidates[static_cast<size_t>(i)].expert * slabs_per_expert +
            candidates[static_cast<size_t>(i)].rank);
    }
    return selected;
}

KimiK3CalibratedSlabPlan plan_kimi_k3_calibrated_slabs(
        const int32_t * expert_ids, const float * router_weights, int top_k,
        const float * ordered_importance,
        const uint8_t * calibrated_experts, int expert_count,
        int slabs_per_expert, int requested_budget) {
    KimiK3CalibratedSlabPlan result;
    result.requested_budget = requested_budget;
    if (!expert_ids || !router_weights || !ordered_importance ||
        !calibrated_experts || top_k <= 0 || expert_count <= 0 ||
        slabs_per_expert <= 0 || requested_budget <= 0) {
        return result;
    }
    std::vector<int32_t> selected_experts;
    std::vector<float> selected_weights;
    selected_experts.reserve(static_cast<size_t>(top_k));
    selected_weights.reserve(static_cast<size_t>(top_k));
    for (int route = 0; route < top_k; ++route) {
        const int expert = expert_ids[route];
        if (expert < 0 || expert >= expert_count) {
            result.selected_slab_ids.clear();
            result.exact_route_indices.clear();
            return result;
        }
        if (calibrated_experts[expert] != 0) {
            selected_experts.push_back(expert);
            selected_weights.push_back(router_weights[route]);
        } else {
            result.exact_route_indices.push_back(route);
        }
    }
    const int actual_budget = std::min(
        requested_budget,
        static_cast<int>(selected_experts.size()) * slabs_per_expert);
    if (actual_budget > 0) {
        result.selected_slab_ids = select_kimi_k3_slab_prefix_ids(
            selected_experts.data(), selected_weights.data(),
            static_cast<int>(selected_experts.size()), ordered_importance,
            expert_count, slabs_per_expert, actual_budget);
    }
    return result;
}

bool create_kimi_k3_calibrated_provider_from_env(
        ggml_backend_t expert_backend,
        ggml_backend_t destination_backend,
        std::unique_ptr<KimiK3RoutedOutputProvider> & out,
        std::string * err) {
    out.reset();
    const char * kind = std::getenv("DFLASH_KIMI_LAYER1_PROVIDER");
    bool ordered_join = false;
    bool async_queue = false;
    bool p40_wide_async_join = false;
    if (!parse_binary_flag(
            std::getenv("DFLASH_KIMI_P42_ORDERED_DEVICE_JOIN"),
            ordered_join) ||
        !parse_binary_flag(
            std::getenv("DFLASH_KIMI_P45_ASYNC_COMPACT_QUEUE"),
            async_queue)) {
        if (err) *err = "P42/P45 controls must be 0 or 1";
        return false;
    }
    if (!parse_binary_flag(
            std::getenv("DFLASH_KIMI_P40_WIDE_ASYNC_JOIN"),
            p40_wide_async_join)) {
        if (err) *err = "DFLASH_KIMI_P40_WIDE_ASYNC_JOIN must be 0 or 1";
        return false;
    }
    if (async_queue && !ordered_join) {
        if (err) *err = "P45 async compact queue requires P42 ordered join";
        return false;
    }
    if (!kind || !*kind || std::strcmp(kind, "exact") == 0) {
        if (ordered_join || async_queue || p40_wide_async_join) {
            if (err) {
                *err = "P42/P45/P40 wide async join require "
                    "all-layers-calibrated96";
            }
            return false;
        }
        return true;
    }
    if (std::strcmp(kind, "all-layers-calibrated96") != 0) {
        if (err) {
            *err = "DFLASH_KIMI_LAYER1_PROVIDER must be exact or "
                "all-layers-calibrated96";
        }
        return false;
    }
    if (expert_backend != destination_backend) {
        if (err) *err = "calibrated96 requires one expert/core backend";
        return false;
    }

    const char * aux = std::getenv("DFLASH_KIMI_CALIBRATED96_AUX_DIR");
    const char * sidecars =
        std::getenv("DFLASH_KIMI_ALL_SLAB_SIDECAR_DIR");
    if (!aux || !*aux || !sidecars || !*sidecars) {
        if (err) {
            *err = "calibrated96 requires auxiliary and sidecar directories";
        }
        return false;
    }

    auto provider = std::make_unique<CalibratedAllLayerProvider>();
    if (!provider->init(
            expert_backend, aux, sidecars, ordered_join, async_queue,
            p40_wide_async_join,
            std::getenv("DFLASH_KIMI_CALIBRATED96_METRICS_OUT"), err)) {
        return false;
    }
    out = std::move(provider);
    return true;
}

} // namespace dflash::common
