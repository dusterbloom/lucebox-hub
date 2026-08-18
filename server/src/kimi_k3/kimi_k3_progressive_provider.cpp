#include "kimi_k3_progressive_provider.h"
#include "kimi_k3_sparse_scatter.h"
#include "device_runtime.h"

#include "ggml-cuda.h"

#include <algorithm>
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
#include <iomanip>
#include <list>
#include <limits>
#include <map>
#include <memory>
#include <mutex>
#include <numeric>
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

bool parse_positive_int(const char * raw, int & value);

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

struct SlabAuxHeader {
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
};
static_assert(sizeof(SlabAuxHeader) == 120,
              "slab runtime header must remain byte-stable");

// v2 keeps the v1 prefix byte-for-byte and appends the honesty/provenance
// fields required by the all-layer calibrated provider.  Means stay on disk;
// loading all 92 copies would consume roughly 14 GiB of host RAM.
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

// The calibrated96 artifacts deliberately omitted native expert means to keep
// their all-layer footprint small.  The four-route experiment needs those
// means for routes that are not read at all, plus one scalar importance per
// expert.  Keep that additional information in a compact, provenance-bound
// companion instead of rewriting the 14 GiB calibrated96 substrate.
struct RouteStatsHeader {
    char magic[8];
    uint32_t version;
    uint32_t model_layer;
    uint32_t expert_count;
    uint32_t dimension;
    uint32_t storage;
    uint32_t alignment;
    uint64_t native_means_offset;
    uint64_t native_means_bytes;
    uint64_t native_importance_offset;
    uint64_t native_importance_bytes;
    uint8_t fit_state_sha256[32];
};
static_assert(sizeof(RouteStatsHeader) == 96,
              "route stats header must remain byte-stable");

struct SlabSidecarHeader {
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
};
static_assert(sizeof(SlabSidecarHeader) == 80,
              "slab sidecar header must remain byte-stable");

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

struct InterventionTraceHeader {
    char magic[8];
    uint32_t version;
    uint32_t provider;
    uint32_t budget;
    uint32_t dimension;
    uint32_t top_k;
    uint32_t model_layer;
    uint64_t records;
    uint64_t record_bytes;
    uint64_t reserved[2];
};
static_assert(sizeof(InterventionTraceHeader) == 64,
              "intervention trace header must remain byte-stable");

struct Calibration {
    std::vector<uint16_t> order;
    std::vector<float> slab_means;
    std::vector<float> slab_importance;
    std::vector<float> native_means;
    std::vector<float> native_importance;
};

enum class ProviderKind : uint32_t {
    Slabs = 1,
    Whole = 2,
    // Reads only selected slab records but restores their natural positions
    // and performs one full-width down projection.  This is the partial-byte
    // counterpart to H17's all-192 recomposition control.
    RecomposedSlabs = 3,
    // H21 one-shot gate: select four complete expert routes by calibrated
    // native-response importance, then keep a six-slab prefix of each route.
    // Omitted selected-route slabs use slab means; all other routes use their
    // native expert means. Native-width arithmetic is preserved.
    FourRouteHalfSlabsRecomposed = 4,
};

struct LayerNumerics {
    uint64_t tokens = 0;
    double cosine_sum = 0.0;
    double relative_l2_sum = 0.0;
    double maximum_relative_l2 = 0.0;
};

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

bool load_calibration(const std::string & path, int model_layer,
                      Calibration & out,
                      std::string * err) {
    std::ifstream input(path, std::ios::binary | std::ios::ate);
    if (!input) {
        if (err) *err = "cannot open slab runtime artifact " + path;
        return false;
    }
    const uint64_t file_bytes = static_cast<uint64_t>(input.tellg());
    input.seekg(0);
    SlabAuxHeader header{};
    input.read(reinterpret_cast<char *>(&header), sizeof(header));
    if (!input || std::memcmp(header.magic, "K3AUX001", 8) != 0 ||
        header.version != 1 ||
        header.model_layer != static_cast<uint32_t>(model_layer) ||
        header.expert_count != kExpertCount ||
        header.dimension != kDimension || header.slab_size != kSlabSize ||
        header.slab_count != kSlabCount || header.storage != 0 ||
        header.alignment != kAlignment) {
        if (err) *err = "slab runtime header is incompatible with Kimi H16";
        return false;
    }
    const uint64_t offsets[] = {
        header.order_offset, header.slab_means_offset,
        header.slab_importance_offset, header.native_means_offset,
        header.native_importance_offset};
    const uint64_t sizes[] = {
        header.order_bytes, header.slab_means_bytes,
        header.slab_importance_bytes, header.native_means_bytes,
        header.native_importance_bytes};
    for (int i = 0; i < 5; ++i) {
        if (!checked_span(offsets[i], sizes[i], file_bytes)) {
            if (err) *err = "slab runtime array lies outside its file";
            return false;
        }
    }
    if (!read_array(input, header.order_offset, header.order_bytes,
                    out.order, err) ||
        !read_array(input, header.slab_means_offset, header.slab_means_bytes,
                    out.slab_means, err) ||
        !read_array(input, header.slab_importance_offset,
                    header.slab_importance_bytes, out.slab_importance, err) ||
        !read_array(input, header.native_means_offset,
                    header.native_means_bytes, out.native_means, err) ||
        !read_array(input, header.native_importance_offset,
                    header.native_importance_bytes,
                    out.native_importance, err)) {
        return false;
    }
    if (out.order.size() != static_cast<size_t>(kExpertCount * kSlabCount) ||
        out.slab_means.size() !=
            static_cast<size_t>(kExpertCount) * kSlabCount * kDimension ||
        out.slab_importance.size() !=
            static_cast<size_t>(kExpertCount * kSlabCount) ||
        out.native_means.size() !=
            static_cast<size_t>(kExpertCount) * kDimension ||
        out.native_importance.size() != static_cast<size_t>(kExpertCount)) {
        if (err) *err = "slab runtime array shape mismatch";
        return false;
    }
    for (int expert = 0; expert < kExpertCount; ++expert) {
        bool seen[kSlabCount]{};
        for (int rank = 0; rank < kSlabCount; ++rank) {
            const uint16_t slab = out.order[
                static_cast<size_t>(expert) * kSlabCount + rank];
            if (slab >= kSlabCount || seen[slab]) {
                if (err) *err = "slab runtime order is not a permutation";
                return false;
            }
            seen[slab] = true;
        }
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

std::string route_stats_path(const std::string & directory,
                             int model_layer) {
    char name[96];
    std::snprintf(name, sizeof(name),
                  "kimi_layer%02d_route_stats.k3route", model_layer);
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

// Defined below with the H17 arithmetic probes.  The progressive provider also
// uses it for H19's selected-slab/full-down execution contract.
bool evaluate_host_recomposed_expert(
        ggml_backend_t backend,
        const MoeStreamExpertSpec & spec,
        const float * input_data,
        const std::vector<uint8_t> & gate_bytes,
        const std::vector<uint8_t> & up_bytes,
        const std::vector<uint8_t> & down_bytes,
        const std::vector<float> * activation_mask_values,
        std::vector<float> & result,
        std::string * err);

class ProgressiveProvider final : public KimiK3RoutedOutputProvider {
public:
    ~ProgressiveProvider() override {
        slab_engine_.destroy();
        finish_trace();
    }

    bool init(ggml_backend_t expert_backend, ProviderKind kind, int budget,
              const std::string & aux_path, const std::string & sidecar_path,
              const char * trace_path, int model_layer, int active_position,
              std::string * err) {
        kind_ = kind;
        budget_ = budget;
        model_layer_ = model_layer;
        active_position_ = active_position;
        if (!load_calibration(
                aux_path, model_layer_, calibration_, err)) return false;
        if ((kind_ == ProviderKind::Slabs ||
             kind_ == ProviderKind::RecomposedSlabs ||
             kind_ == ProviderKind::FourRouteHalfSlabsRecomposed) && !init_sidecar(
                expert_backend, sidecar_path, err)) {
            return false;
        }
        if (trace_path && *trace_path && !init_trace(trace_path, err)) {
            return false;
        }
        std::fprintf(stderr,
            "[kimi-k3-h16] provider=%s budget=%d model-layer=1 "
            "teacher=exact model-layer=%d active-position=%d trace=%s\n",
            kind_ == ProviderKind::Slabs ? "slabs" :
            kind_ == ProviderKind::RecomposedSlabs ? "slabs-recomposed" :
            kind_ == ProviderKind::FourRouteHalfSlabsRecomposed
                ? "four-route-half-slabs-recomposed" :
            "whole",
            budget_, model_layer_, active_position_,
            trace_path && *trace_path ? trace_path : "disabled");
        return true;
    }

    bool handles_layer(int model_layer) const override {
        return model_layer == model_layer_;
    }

    bool evaluate(int model_layer, int base_pos,
                  const MoeStreamExpertSpec & exact_spec,
                  const MoeStreamRouteBatch & routes,
                  MoeHybridStreamEngine & exact_engine,
                  std::vector<float> & output,
                  std::string * err) override {
        if (!handles_layer(model_layer) || routes.n_expert != kExpertCount ||
            routes.top_k != kNativeTopK ||
            exact_spec.input_dim != kDimension ||
            exact_spec.output_dim != kDimension) {
            if (err) *err = "H16 provider received an incompatible routed batch";
            return false;
        }
        std::vector<float> exact;
        if (!eval_moe_streamed_experts(
                exact_engine, exact_spec, routes, exact, err)) {
            return false;
        }
        if (active_position_ >= 0) {
            if (routes.n_tokens != 1) {
                if (err) *err =
                    "position-gated H16 provider requires sequential forwards";
                return false;
            }
            if (base_pos != active_position_) {
                output = exact;
                return append_trace(base_pos, routes, exact, output, err);
            }
        }
        const bool ok = kind_ == ProviderKind::Slabs
            ? evaluate_slabs(exact_spec, routes, output, err)
            : (kind_ == ProviderKind::RecomposedSlabs ||
               kind_ == ProviderKind::FourRouteHalfSlabsRecomposed)
                ? evaluate_recomposed_slabs(
                    exact_spec, routes, exact_engine.compute_backend(),
                    output, err)
                : evaluate_whole(exact_spec, routes, exact_engine, output, err);
        if (!ok) return false;
        if (!append_trace(base_pos, routes, exact, output, err)) return false;
        return true;
    }

private:
    bool init_sidecar(ggml_backend_t backend, const std::string & path,
                      std::string * err) {
        const int fd = open_read_only(path);
        if (fd < 0) {
            if (err) *err = "cannot open slab sidecar " + path + ": " +
                std::strerror(errno);
            return false;
        }
        uint64_t bytes = 0;
        SlabSidecarHeader header{};
        const bool header_ok = file_size(fd, bytes) &&
            read_exact_at(fd, &header, sizeof(header), 0);
        if (!header_ok || std::memcmp(header.magic, "K3SLB001", 8) != 0 ||
            header.version != 1 ||
            header.model_layer != static_cast<uint32_t>(model_layer_) ||
            header.expert_count != kExpertCount ||
            header.dimension != kDimension ||
            header.expert_width != kSlabSize * kSlabCount ||
            header.slab_size != kSlabSize ||
            header.slab_count != kSlabCount ||
            header.alignment != kAlignment ||
            header.slab_bytes != kSlabBytes ||
            header.record_bytes != kExpertRecordBytes ||
            !checked_span(header.payload_offset,
                          static_cast<uint64_t>(kExpertCount) *
                              kExpertRecordBytes, bytes)) {
            close_fd(fd);
            if (err) *err = "progressive slab sidecar header is incompatible";
            return false;
        }
        std::vector<uint16_t> sidecar_order(
            static_cast<size_t>(kExpertCount * kSlabCount));
        if (header.order_bytes != sidecar_order.size() * sizeof(uint16_t) ||
            !read_exact_at(fd, sidecar_order.data(),
                           sidecar_order.size() * sizeof(uint16_t),
                           header.order_offset) ||
            sidecar_order != calibration_.order) {
            close_fd(fd);
            if (err) *err = "slab sidecar order disagrees with calibration";
            return false;
        }

        MoeStreamConfig config = MoeStreamConfig::from_env();
        config.device_cache_bytes = 0;
        config.device_slots = std::max(2, config.device_slots);
        config.fused_decode = true;
        if (!slab_engine_.init(backend, kSlabBytes, config, err)) {
            close_fd(fd);
            return false;
        }
        LayerExpertRegions regions;
        regions.expert_bytes_gate = kSlabComponentBytes;
        regions.expert_bytes_up = kSlabComponentBytes;
        regions.expert_bytes_down = kSlabComponentBytes;
        regions.expert_major.enabled = true;
        regions.expert_major.experts = {
            static_cast<size_t>(header.payload_offset),
            static_cast<size_t>(kExpertCount) * kExpertRecordBytes, 0};
        regions.expert_major.expert_stride = kSlabBytes;
        regions.expert_major.gate_offset = 0;
        regions.expert_major.up_offset = kSlabComponentBytes;
        regions.expert_major.down_offset = 2 * kSlabComponentBytes;
        const bool bound = slab_engine_.bind_sources(
            {{nullptr, static_cast<size_t>(bytes), fd}}, {regions}, err);
        close_fd(fd);
        if (!bound) {
            slab_engine_.destroy();
            return false;
        }
        sidecar_path_ = path;
        return true;
    }

    bool init_trace(const std::string & path, std::string * err) {
        trace_ = std::fopen(path.c_str(), "wb+");
        if (!trace_) {
            if (err) *err = "cannot create intervention trace " + path;
            return false;
        }
        trace_header_ = {};
        std::memcpy(trace_header_.magic, "K3INT001", 8);
        trace_header_.version = 1;
        trace_header_.provider = static_cast<uint32_t>(kind_);
        trace_header_.budget = static_cast<uint32_t>(budget_);
        trace_header_.dimension = kDimension;
        trace_header_.top_k = kNativeTopK;
        trace_header_.model_layer = static_cast<uint32_t>(model_layer_);
        trace_header_.record_bytes =
            2 * sizeof(int32_t) +
            kNativeTopK * (sizeof(int32_t) + sizeof(float)) +
            3 * kDimension * sizeof(float);
        if (std::fwrite(&trace_header_, sizeof(trace_header_), 1, trace_) != 1) {
            if (err) *err = "cannot write intervention trace header";
            std::fclose(trace_);
            trace_ = nullptr;
            return false;
        }
        return true;
    }

    void finish_trace() {
        if (!trace_) return;
        trace_header_.records = trace_records_;
        std::fflush(trace_);
        std::fseek(trace_, 0, SEEK_SET);
        (void) std::fwrite(&trace_header_, sizeof(trace_header_), 1, trace_);
        std::fflush(trace_);
        std::fclose(trace_);
        trace_ = nullptr;
    }

    bool append_trace(int base_pos, const MoeStreamRouteBatch & routes,
                      const std::vector<float> & exact,
                      const std::vector<float> & approximate,
                      std::string * err) {
        if (!trace_) return true;
        std::vector<float> delta(kDimension);
        for (int token = 0; token < routes.n_tokens; ++token) {
            const size_t route_offset =
                static_cast<size_t>(token) * kNativeTopK;
            const size_t vector_offset =
                static_cast<size_t>(token) * kDimension;
            for (int d = 0; d < kDimension; ++d) {
                delta[static_cast<size_t>(d)] =
                    approximate[vector_offset + d] - exact[vector_offset + d];
            }
            const int32_t position = base_pos;
            const int32_t token_offset = token;
            const bool ok =
                std::fwrite(&position, sizeof(position), 1, trace_) == 1 &&
                std::fwrite(&token_offset, sizeof(token_offset), 1, trace_) == 1 &&
                std::fwrite(routes.selected_ids + route_offset,
                            sizeof(int32_t), kNativeTopK, trace_) == kNativeTopK &&
                std::fwrite(routes.selected_weights + route_offset,
                            sizeof(float), kNativeTopK, trace_) == kNativeTopK &&
                std::fwrite(exact.data() + vector_offset,
                            sizeof(float), kDimension, trace_) == kDimension &&
                std::fwrite(approximate.data() + vector_offset,
                            sizeof(float), kDimension, trace_) == kDimension &&
                std::fwrite(delta.data(), sizeof(float), kDimension, trace_) ==
                    kDimension;
            if (!ok) {
                if (err) *err = "cannot append intervention trace";
                return false;
            }
            ++trace_records_;
        }
        return true;
    }

    bool evaluate_whole(const MoeStreamExpertSpec & spec,
                        const MoeStreamRouteBatch & routes,
                        MoeHybridStreamEngine & exact_engine,
                        std::vector<float> & output,
                        std::string * err) {
        const size_t vectors =
            static_cast<size_t>(routes.n_tokens) * kDimension;
        output.assign(vectors, 0.0f);
        std::vector<int32_t> ids(
            static_cast<size_t>(routes.n_tokens) * budget_, -1);
        std::vector<float> weights(ids.size(), 0.0f);
        for (int token = 0; token < routes.n_tokens; ++token) {
            const size_t native_offset =
                static_cast<size_t>(token) * kNativeTopK;
            const std::vector<int32_t> selected =
                select_kimi_k3_whole_expert_routes(
                    routes.selected_ids + native_offset,
                    routes.selected_weights + native_offset,
                    kNativeTopK, calibration_.native_importance.data(),
                    kExpertCount, budget_);
            for (int rank = 0; rank < kNativeTopK; ++rank) {
                const int expert = routes.selected_ids[native_offset + rank];
                const float weight = routes.selected_weights[native_offset + rank];
                const float * mean = calibration_.native_means.data() +
                    static_cast<size_t>(expert) * kDimension;
                float * target = output.data() +
                    static_cast<size_t>(token) * kDimension;
                for (int d = 0; d < kDimension; ++d) {
                    target[d] += weight * mean[d];
                }
            }
            for (int i = 0; i < budget_; ++i) {
                const int route = selected[static_cast<size_t>(i)];
                ids[static_cast<size_t>(token) * budget_ + i] =
                    routes.selected_ids[native_offset + route];
                weights[static_cast<size_t>(token) * budget_ + i] =
                    routes.selected_weights[native_offset + route];
                const int expert = ids[static_cast<size_t>(token) * budget_ + i];
                const float * mean = calibration_.native_means.data() +
                    static_cast<size_t>(expert) * kDimension;
                float * target = output.data() +
                    static_cast<size_t>(token) * kDimension;
                const float weight = weights[
                    static_cast<size_t>(token) * budget_ + i];
                for (int d = 0; d < kDimension; ++d) {
                    target[d] -= weight * mean[d];
                }
            }
        }
        MoeStreamRouteBatch selected_routes = routes;
        selected_routes.top_k = budget_;
        selected_routes.selected_ids = ids.data();
        selected_routes.selected_weights = weights.data();
        selected_routes.expert_observer = nullptr;
        std::vector<float> exact_selected;
        if (!eval_moe_streamed_experts(
                exact_engine, spec, selected_routes, exact_selected, err)) {
            return false;
        }
        for (size_t i = 0; i < vectors; ++i) output[i] += exact_selected[i];
        return true;
    }

    bool evaluate_recomposed_slabs(
            const MoeStreamExpertSpec & spec,
            const MoeStreamRouteBatch & routes,
            ggml_backend_t backend,
            std::vector<float> & output,
            std::string * err) {
        // H19 arithmetic contract:
        //
        //   selected sidecar bytes -> natural tensor positions -> one native
        //   full-width down reduction -> aggregate mean tail.
        //
        // The predecessor `evaluate_slabs` uses twelve independent 256-wide
        // down reductions.  That is mathematically additive but not execution
        // identical on K3's quantized kernels; the all-192 control amplified
        // the resulting ~1e-6 routed error into terminal KL.  Here only the
        // selected slab records are read.  Missing records become zero-weight
        // rows/blocks locally, and the one full-width down graph sees an
        // explicit activation mask.  At budget 192 every byte is restored in
        // natural order and the mask is omitted, providing the identity gate.
        if (!backend || spec.fused_gate_up || sidecar_path_.empty()) {
            if (err) *err = "H19 recomposed slabs need separate K3 gate/up sidecar";
            return false;
        }
        const int descriptor = open_read_only(sidecar_path_);
        if (descriptor < 0) {
            if (err) *err = "cannot open H19 slab sidecar " + sidecar_path_;
            return false;
        }
        SlabSidecarHeader header{};
        uint64_t file_bytes = 0;
        const bool header_ok = file_size(descriptor, file_bytes) &&
            read_exact_at(descriptor, &header, sizeof(header), 0) &&
            std::memcmp(header.magic, "K3SLB001", 8) == 0 &&
            header.version == 1 &&
            header.model_layer == static_cast<uint32_t>(model_layer_) &&
            header.expert_count == kExpertCount &&
            header.dimension == kDimension &&
            header.expert_width == kSlabSize * kSlabCount &&
            header.slab_size == kSlabSize &&
            header.slab_count == kSlabCount &&
            header.slab_bytes == kSlabBytes &&
            header.record_bytes == kExpertRecordBytes &&
            checked_span(
                header.payload_offset,
                static_cast<uint64_t>(kExpertCount) * kExpertRecordBytes,
                file_bytes);
        if (!header_ok) {
            close_fd(descriptor);
            if (err) *err = "H19 progressive sidecar header is incompatible";
            return false;
        }
        const size_t gate_full_bytes = ggml_row_size(
            spec.gate_type, spec.input_dim) *
            static_cast<size_t>(spec.intermediate_dim);
        const size_t up_full_bytes = ggml_row_size(
            spec.up_type, spec.input_dim) *
            static_cast<size_t>(spec.intermediate_dim);
        const size_t down_full_row_bytes = ggml_row_size(
            spec.down_type, spec.intermediate_dim);
        const size_t down_full_bytes = down_full_row_bytes *
            static_cast<size_t>(spec.output_dim);
        const size_t down_slab_row_bytes = kSlabComponentBytes /
            static_cast<size_t>(spec.output_dim);
        if (gate_full_bytes != kSlabComponentBytes * kSlabCount ||
            up_full_bytes != kSlabComponentBytes * kSlabCount ||
            down_slab_row_bytes * kSlabCount != down_full_row_bytes) {
            close_fd(descriptor);
            if (err) *err = "H19 recomposed tensor geometry mismatch";
            return false;
        }
        output.assign(
            static_cast<size_t>(routes.n_tokens) * spec.output_dim, 0.0f);
        std::vector<uint8_t> gate(gate_full_bytes, 0);
        std::vector<uint8_t> up(up_full_bytes, 0);
        std::vector<uint8_t> down(down_full_bytes, 0);
        std::vector<uint8_t> slab_down(kSlabComponentBytes);
        std::vector<float> activation_mask(
            static_cast<size_t>(spec.intermediate_dim), 0.0f);
        std::vector<float> expert_output;

        for (int token = 0; token < routes.n_tokens; ++token) {
            const size_t route_offset =
                static_cast<size_t>(token) * kNativeTopK;
            const bool route_prefix_policy =
                kind_ == ProviderKind::FourRouteHalfSlabsRecomposed;
            const std::vector<int32_t> selected = route_prefix_policy
                ? select_kimi_k3_route_slab_prefix_ids(
                    routes.selected_ids + route_offset,
                    routes.selected_weights + route_offset, kNativeTopK,
                    calibration_.native_importance.data(), kExpertCount,
                    kSlabCount, 4, 6)
                : select_kimi_k3_slab_prefix_ids(
                    routes.selected_ids + route_offset,
                    routes.selected_weights + route_offset,
                    kNativeTopK, calibration_.slab_importance.data(),
                    kExpertCount, kSlabCount, budget_);
            if (selected.size() != static_cast<size_t>(budget_)) {
                close_fd(descriptor);
                if (err) *err = "H19 slab selector returned the wrong budget";
                return false;
            }
            std::vector<uint8_t> selected_by_route(
                static_cast<size_t>(kNativeTopK * kSlabCount), 0);
            std::vector<uint8_t> selected_route(
                static_cast<size_t>(kNativeTopK), 0);
            float * destination = output.data() +
                static_cast<size_t>(token) * spec.output_dim;
            // Start from the calibrated aggregate tail, then replace exactly
            // the selected per-expert/rank cards with live computations.
            // At full budget there is no tail.  Do not add and subtract its
            // cards: that harmless-looking cancellation would make the H19
            // full-width identity control depend on F32 rounding.
            const bool use_mean_tail = budget_ < kNativeTopK * kSlabCount;
            if (use_mean_tail) {
                for (const int32_t pseudo : selected) {
                    const int expert = pseudo / kSlabCount;
                    for (int route = 0; route < kNativeTopK; ++route) {
                        if (routes.selected_ids[route_offset + route] == expert) {
                            selected_route[static_cast<size_t>(route)] = 1;
                            break;
                        }
                    }
                }
                for (int route = 0; route < kNativeTopK; ++route) {
                    const int expert = routes.selected_ids[route_offset + route];
                    const float weight = routes.selected_weights[route_offset + route];
                    if (route_prefix_policy &&
                        !selected_route[static_cast<size_t>(route)]) {
                        const float * mean = calibration_.native_means.data() +
                            static_cast<size_t>(expert) * kDimension;
                        for (int dimension = 0; dimension < kDimension; ++dimension) {
                            destination[dimension] += weight * mean[dimension];
                        }
                        continue;
                    }
                    for (int rank = 0; rank < kSlabCount; ++rank) {
                        const float * mean = calibration_.slab_means.data() +
                            (static_cast<size_t>(expert) * kSlabCount + rank) *
                                kDimension;
                        for (int dimension = 0; dimension < kDimension; ++dimension) {
                            destination[dimension] += weight * mean[dimension];
                        }
                    }
                }
            }
            for (const int32_t pseudo : selected) {
                const int expert = pseudo / kSlabCount;
                const int rank = pseudo % kSlabCount;
                int route = 0;
                while (route < kNativeTopK &&
                       routes.selected_ids[route_offset + route] != expert) {
                    ++route;
                }
                if (route == kNativeTopK) {
                    close_fd(descriptor);
                    if (err) *err = "H19 selected slab has no active expert route";
                    return false;
                }
                uint8_t & present = selected_by_route[
                    static_cast<size_t>(route * kSlabCount + rank)];
                if (present) {
                    close_fd(descriptor);
                    if (err) *err = "H19 selector returned a duplicate slab";
                    return false;
                }
                present = 1;
                if (use_mean_tail) {
                    const float weight = routes.selected_weights[route_offset + route];
                    const float * mean = calibration_.slab_means.data() +
                        (static_cast<size_t>(expert) * kSlabCount + rank) *
                            kDimension;
                    for (int dimension = 0; dimension < kDimension; ++dimension) {
                        destination[dimension] -= weight * mean[dimension];
                    }
                }
            }
            std::vector<int> route_order(kNativeTopK);
            std::iota(route_order.begin(), route_order.end(), 0);
            std::stable_sort(route_order.begin(), route_order.end(),
                [&](int left, int right) {
                    return routes.selected_ids[route_offset + left] <
                        routes.selected_ids[route_offset + right];
                });
            const float * input = routes.inputs +
                static_cast<size_t>(token) * spec.input_dim;
            for (const int route : route_order) {
                const int expert = routes.selected_ids[route_offset + route];
                std::fill(gate.begin(), gate.end(), 0);
                std::fill(up.begin(), up.end(), 0);
                std::fill(down.begin(), down.end(), 0);
                std::fill(activation_mask.begin(), activation_mask.end(), 0.0f);
                int retained = 0;
                for (int rank = 0; rank < kSlabCount; ++rank) {
                    if (!selected_by_route[
                            static_cast<size_t>(route * kSlabCount + rank)]) {
                        continue;
                    }
                    const uint16_t natural = calibration_.order[
                        static_cast<size_t>(expert) * kSlabCount + rank];
                    const uint64_t record = header.payload_offset +
                        static_cast<uint64_t>(expert * kSlabCount + rank) *
                            header.slab_bytes;
                    if (!read_exact_at(
                            descriptor,
                            gate.data() + static_cast<size_t>(natural) *
                                kSlabComponentBytes,
                            kSlabComponentBytes, record) ||
                        !read_exact_at(
                            descriptor,
                            up.data() + static_cast<size_t>(natural) *
                                kSlabComponentBytes,
                            kSlabComponentBytes,
                            record + kSlabComponentBytes) ||
                        !read_exact_at(
                            descriptor, slab_down.data(), slab_down.size(),
                            record + 2 * kSlabComponentBytes)) {
                        close_fd(descriptor);
                        if (err) *err = "short read while restoring H19 slab";
                        return false;
                    }
                    for (int dimension = 0; dimension < spec.output_dim;
                         ++dimension) {
                        std::memcpy(
                            down.data() +
                                static_cast<size_t>(dimension) * down_full_row_bytes +
                                static_cast<size_t>(natural) * down_slab_row_bytes,
                            slab_down.data() +
                                static_cast<size_t>(dimension) * down_slab_row_bytes,
                            down_slab_row_bytes);
                    }
                    std::fill_n(
                        activation_mask.begin() +
                            static_cast<size_t>(natural) * kSlabSize,
                        kSlabSize, 1.0f);
                    ++retained;
                }
                const std::vector<float> * mask = retained == kSlabCount
                    ? nullptr : &activation_mask;
                if (retained == 0) continue;
                if (!evaluate_host_recomposed_expert(
                        backend, spec, input, gate, up, down, mask,
                        expert_output, err)) {
                    close_fd(descriptor);
                    return false;
                }
                const float weight = routes.selected_weights[route_offset + route];
                for (int dimension = 0; dimension < spec.output_dim;
                     ++dimension) {
                    destination[dimension] +=
                        weight * expert_output[static_cast<size_t>(dimension)];
                }
            }
        }
        close_fd(descriptor);
        return true;
    }

    bool evaluate_slabs(const MoeStreamExpertSpec & exact_spec,
                        const MoeStreamRouteBatch & routes,
                        std::vector<float> & output,
                        std::string * err) {
        const size_t vectors =
            static_cast<size_t>(routes.n_tokens) * kDimension;
        output.assign(vectors, 0.0f);
        std::vector<int32_t> ids(
            static_cast<size_t>(routes.n_tokens) * budget_, -1);
        std::vector<float> weights(ids.size(), 0.0f);
        for (int token = 0; token < routes.n_tokens; ++token) {
            const size_t native_offset =
                static_cast<size_t>(token) * kNativeTopK;
            const std::vector<int32_t> selected =
                select_kimi_k3_slab_prefix_ids(
                    routes.selected_ids + native_offset,
                    routes.selected_weights + native_offset,
                    kNativeTopK, calibration_.slab_importance.data(),
                    kExpertCount, kSlabCount, budget_);
            float * target = output.data() +
                static_cast<size_t>(token) * kDimension;
            for (int route = 0; route < kNativeTopK; ++route) {
                const int expert = routes.selected_ids[native_offset + route];
                const float weight = routes.selected_weights[native_offset + route];
                for (int slab = 0; slab < kSlabCount; ++slab) {
                    const float * mean = calibration_.slab_means.data() +
                        (static_cast<size_t>(expert) * kSlabCount + slab) *
                            kDimension;
                    for (int d = 0; d < kDimension; ++d) {
                        target[d] += weight * mean[d];
                    }
                }
            }
            for (int i = 0; i < budget_; ++i) {
                const int pseudo = selected[static_cast<size_t>(i)];
                ids[static_cast<size_t>(token) * budget_ + i] = pseudo;
                const int expert = pseudo / kSlabCount;
                const int slab = pseudo % kSlabCount;
                int route = 0;
                while (route < kNativeTopK &&
                       routes.selected_ids[native_offset + route] != expert) {
                    ++route;
                }
                if (route == kNativeTopK) {
                    if (err) *err = "selected slab has no native expert route";
                    return false;
                }
                const float weight = routes.selected_weights[native_offset + route];
                weights[static_cast<size_t>(token) * budget_ + i] = weight;
                const float * mean = calibration_.slab_means.data() +
                    (static_cast<size_t>(expert) * kSlabCount + slab) *
                        kDimension;
                for (int d = 0; d < kDimension; ++d) {
                    target[d] -= weight * mean[d];
                }
            }
        }
        MoeStreamExpertSpec slab_spec = exact_spec;
        slab_spec.intermediate_dim = kSlabSize;
        MoeStreamRouteBatch slab_routes = routes;
        slab_routes.layer = 0;
        slab_routes.n_expert = kExpertCount * kSlabCount;
        slab_routes.top_k = budget_;
        slab_routes.selected_ids = ids.data();
        slab_routes.selected_weights = weights.data();
        slab_routes.expert_observer = nullptr;
        std::vector<float> exact_selected;
        if (!eval_moe_streamed_experts(
                slab_engine_, slab_spec, slab_routes, exact_selected, err)) {
            return false;
        }
        for (size_t i = 0; i < vectors; ++i) output[i] += exact_selected[i];
        return true;
    }

    ProviderKind kind_ = ProviderKind::Slabs;
    int budget_ = 0;
    int model_layer_ = 1;
    int active_position_ = -1;
    Calibration calibration_;
    MoeHybridStreamEngine slab_engine_;
    std::string sidecar_path_;
    std::FILE * trace_ = nullptr;
    InterventionTraceHeader trace_header_{};
    uint64_t trace_records_ = 0;
};

// H17's first gate deliberately selects every slab. It needs no calibration
// means or learned ordering: it only asks whether the byte-aligned numerical
// decomposition can be composed through every routed layer. The ordinary
// exact evaluator remains immutable and is evaluated alongside the slab sum
// so every layer receives a direct numerical control.
class GroupedSlabObserver final : public MoeStreamExpertObserver {
public:
    bool init(int n_tokens, const std::vector<int32_t> & selected_ids,
              std::string * err) {
        constexpr int kActiveSlabs = kNativeTopK * kSlabCount;
        if (n_tokens <= 0 || selected_ids.size() !=
                static_cast<size_t>(n_tokens * kActiveSlabs)) {
            if (err) *err = "grouped slab observer received invalid routes";
            return false;
        }
        n_tokens_ = n_tokens;
        slot_by_token_slab_.assign(
            static_cast<size_t>(n_tokens) * kExpertCount * kSlabCount, -1);
        outputs_.assign(
            static_cast<size_t>(n_tokens) * kActiveSlabs * kDimension, 0.0f);
        seen_.assign(static_cast<size_t>(n_tokens) * kActiveSlabs, false);
        for (int token = 0; token < n_tokens; ++token) {
            for (int slot = 0; slot < kActiveSlabs; ++slot) {
                const int slab = selected_ids[
                    static_cast<size_t>(token) * kActiveSlabs + slot];
                if (slab < 0 || slab >= kExpertCount * kSlabCount) {
                    if (err) *err = "grouped slab observer saw invalid slab ID";
                    return false;
                }
                int & mapped = slot_by_token_slab_[
                    static_cast<size_t>(token) * kExpertCount * kSlabCount +
                    slab];
                if (mapped >= 0) {
                    if (err) *err = "grouped slab observer saw duplicate slab ID";
                    return false;
                }
                mapped = slot;
            }
        }
        return true;
    }

    bool observe(int, int token, int slab, float,
                 const float *, int input_dimension,
                 const float * expert_output, int output_dimension,
                 std::string * err) override {
        constexpr int kActiveSlabs = kNativeTopK * kSlabCount;
        if (token < 0 || token >= n_tokens_ || slab < 0 ||
            slab >= kExpertCount * kSlabCount || !expert_output ||
            input_dimension != kDimension || output_dimension != kDimension) {
            if (err) *err = "grouped slab observer received invalid output";
            return false;
        }
        const int slot = slot_by_token_slab_[
            static_cast<size_t>(token) * kExpertCount * kSlabCount + slab];
        if (slot < 0 || slot >= kActiveSlabs) {
            if (err) *err = "grouped slab observer saw an unrequested slab";
            return false;
        }
        const size_t observation =
            static_cast<size_t>(token) * kActiveSlabs + slot;
        if (seen_[observation]) {
            if (err) *err = "grouped slab observer saw a duplicate output";
            return false;
        }
        std::memcpy(
            outputs_.data() + observation * kDimension,
            expert_output, sizeof(float) * kDimension);
        seen_[observation] = true;
        return true;
    }

    bool complete() const {
        return std::all_of(seen_.begin(), seen_.end(),
                           [](bool value) { return value; });
    }

    const float * output(int token, int active_slot) const {
        constexpr int kActiveSlabs = kNativeTopK * kSlabCount;
        return outputs_.data() +
            (static_cast<size_t>(token) * kActiveSlabs + active_slot) *
                kDimension;
    }

private:
    int n_tokens_ = 0;
    std::vector<int> slot_by_token_slab_;
    std::vector<float> outputs_;
    std::vector<bool> seen_;
};

enum class AllSlabsMode {
    Direct,
    Grouped,
    Recomposed,
    RecomposedNatural96,
    RecomposedNatural144,
    StaticNatural96,
    OracleNatural96,
    OracleNatural144,
};

const char * all_slabs_mode_name(AllSlabsMode mode) {
    switch (mode) {
        case AllSlabsMode::Direct: return "all-slabs";
        case AllSlabsMode::Grouped: return "all-slabs-grouped";
        case AllSlabsMode::Recomposed: return "all-slabs-recomposed";
        case AllSlabsMode::RecomposedNatural96:
            return "all-slabs-recomposed-natural96-zero-tail";
        case AllSlabsMode::RecomposedNatural144:
            return "all-slabs-recomposed-natural144-zero-tail";
        case AllSlabsMode::StaticNatural96:
            return "all-slabs-static-natural96-zero-tail";
        case AllSlabsMode::OracleNatural96:
            return "all-slabs-oracle-natural96-zero-tail";
        case AllSlabsMode::OracleNatural144:
            return "all-slabs-oracle-natural144-zero-tail";
    }
    return "all-slabs-unknown";
}

int all_slabs_mode_budget(AllSlabsMode mode) {
    switch (mode) {
        case AllSlabsMode::StaticNatural96:
        case AllSlabsMode::OracleNatural96: return 96;
        case AllSlabsMode::RecomposedNatural96: return 96;
        case AllSlabsMode::RecomposedNatural144: return 144;
        case AllSlabsMode::OracleNatural144: return 144;
        case AllSlabsMode::Direct:
        case AllSlabsMode::Grouped:
        case AllSlabsMode::Recomposed:
            return kNativeTopK * kSlabCount;
    }
    return 0;
}

struct ProbeDifference {
    bool bit_equal = false;
    double maximum_absolute = 0.0;
    double relative_l2 = 0.0;
};

ProbeDifference compare_probe_vectors(const float * reference,
                                      const float * candidate,
                                      size_t count) {
    ProbeDifference result;
    result.bit_equal = std::memcmp(
        reference, candidate, count * sizeof(float)) == 0;
    double reference_norm2 = 0.0;
    double error_norm2 = 0.0;
    for (size_t index = 0; index < count; ++index) {
        const double left = reference[index];
        const double difference =
            static_cast<double>(candidate[index]) - left;
        reference_norm2 += left * left;
        error_norm2 += difference * difference;
        result.maximum_absolute = std::max(
            result.maximum_absolute, std::abs(difference));
    }
    result.relative_l2 = std::sqrt(
        error_norm2 / std::max(1.0e-300, reference_norm2));
    return result;
}

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

bool capture_probe_activation(MoeHybridStreamEngine & engine,
                              int layer,
                              int expert,
                              const MoeStreamExpertSpec & spec,
                              const float * input_data,
                              std::vector<float> & activation,
                              std::string * err) {
    if (!engine.stream_expert_sync(layer, expert, err)) return false;
    ggml_backend_t backend = engine.compute_backend();
    ggml_init_params parameters{};
    parameters.mem_size = 32 * 1024 * 1024;
    parameters.no_alloc = true;
    ggml_context * context = ggml_init(parameters);
    if (!context) {
        if (err) *err = "H17 pre-down probe ggml_init failed";
        return false;
    }

    ggml_tensor * input = ggml_new_tensor_2d(
        context, GGML_TYPE_F32, spec.input_dim, 1);
    ggml_set_input(input);
    ggml_tensor * gate = nullptr;
    ggml_tensor * up = nullptr;
    ggml_tensor * gate_up = nullptr;
    ggml_tensor * activated = nullptr;
    if (spec.fused_gate_up) {
        gate_up = ggml_new_tensor_2d(
            context, spec.gate_up_type,
            spec.input_dim, 2 * spec.intermediate_dim);
        ggml_set_input(gate_up);
        ggml_tensor * combined = probe_scale_tensor(
            context, ggml_mul_mat(context, gate_up, input),
            spec.gate_up_scale);
        ggml_tensor * gate_part = ggml_view_2d(
            context, combined, spec.intermediate_dim, 1,
            combined->nb[1], 0);
        ggml_tensor * up_part = ggml_view_2d(
            context, combined, spec.intermediate_dim, 1,
            combined->nb[1],
            static_cast<size_t>(spec.intermediate_dim) * sizeof(float));
        activated = probe_gated_activation(
            context, spec, ggml_cont(context, gate_part),
            ggml_cont(context, up_part));
    } else {
        gate = ggml_new_tensor_2d(
            context, spec.gate_type,
            spec.input_dim, spec.intermediate_dim);
        up = ggml_new_tensor_2d(
            context, spec.up_type,
            spec.input_dim, spec.intermediate_dim);
        ggml_set_input(gate);
        ggml_set_input(up);
        ggml_tensor * gate_value = probe_scale_tensor(
            context, ggml_mul_mat(context, gate, input), spec.gate_scale);
        ggml_tensor * up_value = probe_scale_tensor(
            context, ggml_mul_mat(context, up, input), spec.up_scale);
        activated = probe_gated_activation(
            context, spec, gate_value, up_value);
    }
    ggml_cgraph * graph = ggml_new_graph_custom(context, 256, false);
    activation.resize(static_cast<size_t>(spec.intermediate_dim));
    ggml_set_output(activated);
    ggml_build_forward_expand(graph, activated);
    ggml_gallocr_t allocator = ggml_gallocr_new(
        ggml_backend_get_default_buffer_type(backend));
    if (!allocator || !ggml_gallocr_alloc_graph(allocator, graph)) {
        if (err) *err = "H17 activation probe allocation failed";
        if (allocator) ggml_gallocr_free(allocator);
        ggml_free(context);
        return false;
    }
    ggml_backend_tensor_set(
        input, input_data, 0,
        static_cast<size_t>(spec.input_dim) * sizeof(float));
    if (gate_up) {
        gate_up->data = const_cast<void *>(engine.scratch_gate_data());
    } else {
        gate->data = const_cast<void *>(engine.scratch_gate_data());
        up->data = const_cast<void *>(engine.scratch_up_data());
    }
    const ggml_status status =
        ggml_backend_graph_compute(backend, graph);
    if (status == GGML_STATUS_SUCCESS) {
        ggml_backend_tensor_get(
            activated, activation.data(), 0,
            activation.size() * sizeof(float));
    } else if (err) {
        *err = "H17 activation probe compute failed";
    }
    ggml_gallocr_free(allocator);
    ggml_free(context);
    return status == GGML_STATUS_SUCCESS;
}

bool project_probe_down(MoeHybridStreamEngine & engine,
                        int layer,
                        int expert,
                        const MoeStreamExpertSpec & spec,
                        const float * activation,
                        std::vector<float> & projected,
                        std::string * err) {
    if (!engine.stream_expert_sync(layer, expert, err)) return false;
    ggml_backend_t backend = engine.compute_backend();
    ggml_init_params parameters{};
    parameters.mem_size = 8 * 1024 * 1024;
    parameters.no_alloc = true;
    ggml_context * context = ggml_init(parameters);
    if (!context) {
        if (err) *err = "H17 down probe ggml_init failed";
        return false;
    }
    ggml_tensor * input = ggml_new_tensor_2d(
        context, GGML_TYPE_F32, spec.intermediate_dim, 1);
    ggml_set_input(input);
    ggml_tensor * down = ggml_new_tensor_2d(
        context, spec.down_type,
        spec.intermediate_dim, spec.output_dim);
    ggml_set_input(down);
    ggml_tensor * output = probe_scale_tensor(
        context, ggml_mul_mat(context, down, input), spec.down_scale);
    ggml_cgraph * graph = ggml_new_graph_custom(context, 128, false);
    ggml_set_output(output);
    ggml_build_forward_expand(graph, output);
    ggml_gallocr_t allocator = ggml_gallocr_new(
        ggml_backend_get_default_buffer_type(backend));
    if (!allocator || !ggml_gallocr_alloc_graph(allocator, graph)) {
        if (err) *err = "H17 down probe allocation failed";
        if (allocator) ggml_gallocr_free(allocator);
        ggml_free(context);
        return false;
    }
    ggml_backend_tensor_set(
        input, activation, 0,
        static_cast<size_t>(spec.intermediate_dim) * sizeof(float));
    down->data = const_cast<void *>(engine.scratch_down_data());
    projected.resize(static_cast<size_t>(spec.output_dim));
    const ggml_status status =
        ggml_backend_graph_compute(backend, graph);
    if (status == GGML_STATUS_SUCCESS) {
        ggml_backend_tensor_get(
            output, projected.data(), 0,
            projected.size() * sizeof(float));
    } else if (err) {
        *err = "H17 down probe compute failed";
    }
    ggml_gallocr_free(allocator);
    ggml_free(context);
    return status == GGML_STATUS_SUCCESS;
}

bool evaluate_host_recomposed_expert(
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
        if (err) *err = "H17 recomposed expert requires separate gate/up tensors";
        return false;
    }
    ggml_init_params parameters{};
    parameters.mem_size = 32 * 1024 * 1024;
    parameters.no_alloc = true;
    ggml_context * context = ggml_init(parameters);
    if (!context) {
        if (err) *err = "H17 recomposed expert ggml_init failed";
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
            if (err) *err = "H19 recomposed activation mask has wrong size";
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
        if (err) *err = "H17 recomposed expert byte size mismatch";
        ggml_free(context);
        return false;
    }
    ggml_cgraph * graph = ggml_new_graph_custom(context, 512, false);
    ggml_set_output(output);
    ggml_build_forward_expand(graph, output);
    ggml_gallocr_t allocator = ggml_gallocr_new(
        ggml_backend_get_default_buffer_type(backend));
    if (!allocator || !ggml_gallocr_alloc_graph(allocator, graph)) {
        if (err) *err = "H17 recomposed expert graph allocation failed";
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
        *err = "H17 recomposed expert graph compute failed";
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
#if defined(DFLASH27B_BACKEND_CUDA)
        if (data && owns_data) cudaFreeHost(data);
#endif
    }

    bool ensure(size_t requested, std::string * err) {
#if !defined(DFLASH27B_BACKEND_CUDA)
        (void) requested;
        if (err) *err = "P27 direct pinned payload requires CUDA";
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
#if defined(DFLASH27B_BACKEND_CUDA)
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
    bool owns_data = true;
};

struct P28PinnedArena {
    ~P28PinnedArena() {
#if defined(DFLASH27B_BACKEND_CUDA)
        if (data) cudaFreeHost(data);
#endif
    }
    bool ensure(size_t requested, std::string * err) {
#if !defined(DFLASH27B_BACKEND_CUDA)
        (void) requested;
        if (err) *err = "P28 pinned arena requires CUDA";
        return false;
#else
        if (capacity >= requested) return true;
        if (data) cudaFreeHost(data);
        data = nullptr;
        capacity = 0;
        if (cudaHostAlloc(&data, requested, cudaHostAllocDefault) !=
                cudaSuccess) {
            if (err) *err = "P28 layer pinned arena allocation failed";
            return false;
        }
        capacity = requested;
        return true;
#endif
    }
    void * data = nullptr;
    size_t capacity = 0;
};

// P28 is deliberately an oracle replay, not a predictor.  A frozen P27 trace
// names the selected experts and natural slab records for a future layer.  The
// live route is validated before any prefetched payload can affect execution.
struct P28OracleRoute {
    int expert = -1;
    std::vector<uint16_t> naturals;
};

struct P28OracleLayer {
    int base_pos = -1;
    int model_layer = -1;
    std::vector<P28OracleRoute> routes;
};

struct P28OracleReadResult {
    bool ok = false;
    uint64_t physical_bytes = 0;
    uint64_t elapsed_ns = 0;
    std::string error;
};

// P23 keeps the P20 full-width arithmetic intact while removing repeated
// ggml graph allocation and CUDA buffer allocation from the per-route hot
// path.  A small cache entry is retained for each qtype/mask geometry seen by
// the model.  Evaluation remains sequential, so expert-ID accumulation order
// and the frozen calibrated96 semantics do not change.
class SparseDeviceExpertEvaluator {
public:
    SparseDeviceExpertEvaluator() = default;
    SparseDeviceExpertEvaluator(const SparseDeviceExpertEvaluator &) = delete;
    SparseDeviceExpertEvaluator & operator=(
        const SparseDeviceExpertEvaluator &) = delete;

    uint64_t compact_pack_ns() const { return compact_pack_ns_; }
    uint64_t compact_scatter_ns() const { return compact_scatter_ns_; }
    uint64_t expert_graph_ns() const { return expert_graph_ns_; }
    uint64_t expert_readback_ns() const { return expert_readback_ns_; }

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
            bool compact_upload,
            bool pinned_compact,
            std::string * err) {
#if !defined(DFLASH27B_BACKEND_CUDA)
        (void) backend; (void) spec; (void) input_data; (void) slabs;
        (void) prepacked_compact;
        (void) activation_mask_values; (void) down_slab_row_bytes;
        (void) result; (void) authoritative_h2d_bytes;
        (void) metadata_h2d_bytes; (void) device_zero_bytes;
        (void) compact_upload; (void) pinned_compact;
        if (err) *err = "P23 sparse scratch currently requires CUDA";
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
        const size_t slab_count = prepacked_compact
            ? static_cast<size_t>(prepacked_compact->slab_count)
            : slabs.size();
        const bool needs_mask = slab_count != kSlabCount;
        Entry * entry = find(backend, spec, needs_mask);
        if (!entry) {
            entry = create(backend, spec, needs_mask, err);
            if (!entry) return false;
        }

        auto cuda_ok = [&](cudaError_t status, const char * operation) {
            if (status == cudaSuccess) return true;
            if (err) {
                *err = std::string("P23 ") + operation + " failed: " +
                    cudaGetErrorString(status);
            }
            return false;
        };
        bool ok = true;
        device_zero_bytes += ggml_nbytes(entry->gate) +
            ggml_nbytes(entry->up) + ggml_nbytes(entry->down);
        if (compact_upload) {
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
                    pinned_compact && !prepacked_compact, err)) {
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
            } else if (pinned_compact) {
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
            ok = kimi_k3_sparse_scatter_upload(
                entry->gate->data, ggml_nbytes(entry->gate),
                entry->up->data, ggml_nbytes(entry->up),
                entry->down->data, ggml_nbytes(entry->down),
                entry->compact_staging, entry->compact_capacity,
                compact_host, compact_bytes,
                static_cast<int>(slab_count), metadata_bytes,
                gate_slab_bytes, up_slab_bytes, down_slab_bytes,
                down_slab_row_bytes, entry->down->nb[1], spec.output_dim,
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
        } else {
            ok =
                cuda_ok(cudaMemset(
                    entry->gate->data, 0, ggml_nbytes(entry->gate)),
                    "gate zero") &&
                cuda_ok(cudaMemset(
                    entry->up->data, 0, ggml_nbytes(entry->up)),
                    "up zero") &&
                cuda_ok(cudaMemset(
                    entry->down->data, 0, ggml_nbytes(entry->down)),
                    "down zero");
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
        if (ok) {
            ggml_backend_tensor_set(
                entry->input, input_data, 0,
                static_cast<size_t>(spec.input_dim) * sizeof(float));
            if (entry->activation_mask) {
                ggml_backend_tensor_set(
                    entry->activation_mask, activation_mask_values.data(), 0,
                    activation_mask_values.size() * sizeof(float));
            }
            metadata_h2d_bytes +=
                static_cast<uint64_t>(spec.input_dim) * sizeof(float) +
                (entry->activation_mask
                    ? activation_mask_values.size() * sizeof(float) : 0);
            const auto graph_started = std::chrono::steady_clock::now();
            const ggml_status status =
                ggml_backend_graph_compute(backend, entry->graph);
            expert_graph_ns_ += static_cast<uint64_t>(
                std::chrono::duration_cast<std::chrono::nanoseconds>(
                    std::chrono::steady_clock::now() - graph_started).count());
            ok = status == GGML_STATUS_SUCCESS;
            if (!ok && err) {
                *err = "P23 persistent sparse graph compute failed";
            }
        }
        if (ok) {
            result.resize(static_cast<size_t>(spec.output_dim));
            const auto readback_started = std::chrono::steady_clock::now();
            ggml_backend_tensor_get(
                entry->output, result.data(), 0,
                result.size() * sizeof(float));
            expert_readback_ns_ += static_cast<uint64_t>(
                std::chrono::duration_cast<std::chrono::nanoseconds>(
                    std::chrono::steady_clock::now() - readback_started).count());
        }
        return ok;
#endif
    }

private:
#if defined(DFLASH27B_BACKEND_CUDA)
    struct Entry {
        ~Entry() {
            if (compact_host_staging) cudaFreeHost(compact_host_staging);
            if (compact_staging) cudaFree(compact_staging);
            if (allocator) ggml_gallocr_free(allocator);
            if (context) ggml_free(context);
        }

        ggml_backend_t backend = nullptr;
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
        void * compact_staging = nullptr;
        void * compact_host_staging = nullptr;
        size_t compact_capacity = 0;
    };

    static bool same_spec(
            const MoeStreamExpertSpec & left,
            const MoeStreamExpertSpec & right) {
        return left.input_dim == right.input_dim &&
            left.intermediate_dim == right.intermediate_dim &&
            left.output_dim == right.output_dim &&
            left.gate_type == right.gate_type &&
            left.up_type == right.up_type &&
            left.down_type == right.down_type &&
            left.fused_gate_up == right.fused_gate_up &&
            left.gated_activation == right.gated_activation &&
            left.situ_beta == right.situ_beta &&
            left.situ_linear_beta == right.situ_linear_beta &&
            left.gate_scale == right.gate_scale &&
            left.up_scale == right.up_scale &&
            left.down_scale == right.down_scale;
    }

    Entry * find(
            ggml_backend_t backend, const MoeStreamExpertSpec & spec,
            bool needs_mask) {
        for (const std::unique_ptr<Entry> & entry : entries_) {
            if (entry->backend == backend &&
                entry->needs_mask == needs_mask &&
                same_spec(entry->spec, spec)) {
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
#endif
    uint64_t compact_pack_ns_ = 0;
    uint64_t compact_scatter_ns_ = 0;
    uint64_t expert_graph_ns_ = 0;
    uint64_t expert_readback_ns_ = 0;
};

// P20's first production-shaped baseline keeps the native full-width graph but
// initializes its device tensors in place and patches only selected slab bytes.
// The activation mask makes the omitted gate/up/down bytes semantically inert.
// In particular, no reconstructed full expert crosses PCIe.
bool evaluate_sparse_device_expert(
        SparseDeviceExpertEvaluator * persistent_evaluator,
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
        bool compact_upload,
        bool pinned_compact,
        std::string * err) {
    SparseDeviceExpertEvaluator one_shot_evaluator;
    SparseDeviceExpertEvaluator & evaluator = persistent_evaluator
        ? *persistent_evaluator : one_shot_evaluator;
    return evaluator.evaluate(
        backend, spec, input_data, slabs, prepacked_compact,
        activation_mask_values,
        down_slab_row_bytes, result, authoritative_h2d_bytes,
        metadata_h2d_bytes, device_zero_bytes, compact_upload,
        pinned_compact, err);
}

class AllSlabsProvider final : public KimiK3RoutedOutputProvider {
public:
    ~AllSlabsProvider() override {
        finish_metrics();
        slab_engine_.destroy();
    }

    bool init(ggml_backend_t expert_backend, const std::string & directory,
              const char * metrics_path, AllSlabsMode mode,
              std::string * err) {
        if (!expert_backend || directory.empty()) {
            if (err) *err = "all-slab provider needs a backend and sidecar directory";
            return false;
        }
        std::vector<MoeNvmeSource> sources;
        std::vector<LayerExpertRegions> regions(kRoutedLayerCount);
        std::vector<int> descriptors;
        sources.reserve(kRoutedLayerCount);
        descriptors.reserve(kRoutedLayerCount);
        const auto close_descriptors = [&]() {
            for (int fd : descriptors) close_fd(fd);
            descriptors.clear();
        };

        for (int model_layer = kFirstRoutedLayer;
             model_layer <= kLastRoutedLayer; ++model_layer) {
            const std::string path = natural_sidecar_path(
                directory, model_layer);
            const int fd = open_read_only(path);
            if (fd < 0) {
                close_descriptors();
                if (err) *err = "cannot open all-slab sidecar " + path +
                    ": " + std::strerror(errno);
                return false;
            }
            uint64_t bytes = 0;
            SlabSidecarHeaderV2 header{};
            std::vector<uint16_t> order(
                static_cast<size_t>(kExpertCount * kSlabCount));
            const bool prefix_valid = file_size(fd, bytes) &&
                read_exact_at(fd, &header, sizeof(SlabSidecarHeader), 0) &&
                std::memcmp(header.magic, "K3SLB001", 8) == 0 &&
                (header.version == 1 || header.version == 2);
            if (prefix_valid && header.version == 2 &&
                !read_exact_at(fd, &header, sizeof(header), 0)) {
                close_fd(fd);
                close_descriptors();
                if (err) *err = "short all-slab v2 header " + path;
                return false;
            }
            const uint64_t gate_slab_bytes = header.version == 1
                ? kSlabComponentBytes : header.gate_slab_bytes;
            const uint64_t up_slab_bytes = header.version == 1
                ? kSlabComponentBytes : header.up_slab_bytes;
            const uint64_t down_slab_bytes = header.version == 1
                ? kSlabComponentBytes : header.down_slab_bytes;
            const bool valid = prefix_valid &&
                header.model_layer == static_cast<uint32_t>(model_layer) &&
                header.expert_count == kExpertCount &&
                header.dimension == kDimension &&
                header.expert_width == kSlabSize * kSlabCount &&
                header.slab_size == kSlabSize &&
                header.slab_count == kSlabCount &&
                header.alignment == kAlignment &&
                gate_slab_bytes > 0 && up_slab_bytes > 0 &&
                down_slab_bytes > 0 &&
                header.slab_bytes ==
                    gate_slab_bytes + up_slab_bytes + down_slab_bytes &&
                header.record_bytes == header.slab_bytes * kSlabCount &&
                header.order_bytes == order.size() * sizeof(uint16_t) &&
                checked_span(
                    header.payload_offset,
                    static_cast<uint64_t>(kExpertCount) * kExpertRecordBytes,
                    bytes) &&
                read_exact_at(fd, order.data(),
                              order.size() * sizeof(uint16_t),
                              header.order_offset) &&
                valid_slab_order(order);
            if (!valid) {
                close_fd(fd);
                close_descriptors();
                if (err) *err = "incompatible all-slab sidecar " + path;
                return false;
            }
            const uint32_t source_index =
                static_cast<uint32_t>(sources.size());
            sources.push_back({nullptr, static_cast<size_t>(bytes), fd});
            descriptors.push_back(fd);

            max_slab_bytes_ = std::max(
                max_slab_bytes_, static_cast<size_t>(header.slab_bytes));

            LayerExpertRegions & layer =
                regions[static_cast<size_t>(model_layer - 1)];
            layer.expert_bytes_gate = static_cast<size_t>(gate_slab_bytes);
            layer.expert_bytes_up = static_cast<size_t>(up_slab_bytes);
            layer.expert_bytes_down = static_cast<size_t>(down_slab_bytes);
            layer.expert_major.enabled = true;
            layer.expert_major.experts = {
                static_cast<size_t>(header.payload_offset),
                static_cast<size_t>(kExpertCount) *
                    static_cast<size_t>(header.record_bytes),
                source_index};
            layer.expert_major.expert_stride =
                static_cast<size_t>(header.slab_bytes);
            layer.expert_major.gate_offset = 0;
            layer.expert_major.up_offset =
                static_cast<size_t>(gate_slab_bytes);
            layer.expert_major.down_offset = static_cast<size_t>(
                gate_slab_bytes + up_slab_bytes);
        }

        MoeStreamConfig config = MoeStreamConfig::from_env();
        config.device_cache_bytes = 0;
        config.device_slots = std::max(2, config.device_slots);
        config.fused_decode = true;
        const bool initialized = slab_engine_.init(
            expert_backend, max_slab_bytes_, config, err) &&
            slab_engine_.bind_sources(sources, regions, err);
        close_descriptors();
        if (!initialized) {
            slab_engine_.destroy();
            return false;
        }
        metrics_path_ = metrics_path && *metrics_path ? metrics_path : "";
        sidecar_directory_ = directory;
        if (const char * probe =
                std::getenv("DFLASH_KIMI_H17_DOWN_PROBE")) {
            probe_prefix_ = probe;
        }
        mode_ = mode;
        numerics_.resize(kLastRoutedLayer + 1);
        std::fprintf(stderr,
            "[kimi-k3-h17] provider=%s budget=%d layers=1..92 "
            "sidecars=%s metrics=%s down-probe=%s\n",
            all_slabs_mode_name(mode_), all_slabs_mode_budget(mode_),
            directory.c_str(),
            metrics_path_.empty() ? "stderr" : metrics_path_.c_str(),
            probe_prefix_.empty() ? "disabled" : probe_prefix_.c_str());
        return true;
    }

    bool handles_layer(int model_layer) const override {
        return model_layer >= kFirstRoutedLayer &&
            model_layer <= kLastRoutedLayer;
    }

    bool evaluate(int model_layer, int base_pos,
                  const MoeStreamExpertSpec & exact_spec,
                  const MoeStreamRouteBatch & routes,
                  MoeHybridStreamEngine & exact_engine,
                  std::vector<float> & output,
                  std::string * err) override {
        (void) base_pos;
        if (!handles_layer(model_layer) || routes.n_expert != kExpertCount ||
            routes.top_k != kNativeTopK ||
            exact_spec.input_dim != kDimension ||
            exact_spec.output_dim != kDimension) {
            if (err) *err = "H17 all-slab provider received an incompatible batch";
            return false;
        }

        const bool partial_recomposition =
            mode_ == AllSlabsMode::RecomposedNatural96 ||
            mode_ == AllSlabsMode::RecomposedNatural144;
        std::vector<float> native_exact;
        if (!partial_recomposition &&
            !eval_moe_streamed_experts(
                exact_engine, exact_spec, routes, native_exact, err)) {
            return false;
        }
        if (mode_ == AllSlabsMode::Recomposed ||
            mode_ == AllSlabsMode::RecomposedNatural96 ||
            mode_ == AllSlabsMode::RecomposedNatural144) {
            if (!evaluate_recomposed(
                    model_layer, exact_spec, routes,
                    exact_engine.compute_backend(), output, err)) {
                return false;
            }
            if (!partial_recomposition) {
                observe_numerics(
                    model_layer, routes.n_tokens, native_exact, output);
            }
            return true;
        }

        constexpr int kActiveSlabs = kNativeTopK * kSlabCount;
        std::vector<int32_t> ids(
            static_cast<size_t>(routes.n_tokens) * kActiveSlabs);
        std::vector<float> weights(ids.size());
        for (int token = 0; token < routes.n_tokens; ++token) {
            const size_t route_offset =
                static_cast<size_t>(token) * kNativeTopK;
            const size_t slab_offset =
                static_cast<size_t>(token) * kActiveSlabs;
            int cursor = 0;
            for (int route = 0; route < kNativeTopK; ++route) {
                const int expert = routes.selected_ids[route_offset + route];
                const float weight =
                    routes.selected_weights[route_offset + route];
                for (int rank = 0; rank < kSlabCount; ++rank) {
                    ids[slab_offset + cursor] = expert * kSlabCount + rank;
                    weights[slab_offset + cursor] = weight;
                    ++cursor;
                }
            }
        }

        MoeStreamExpertSpec slab_spec = exact_spec;
        slab_spec.intermediate_dim = kSlabSize;
        MoeStreamRouteBatch slab_routes = routes;
        slab_routes.layer = model_layer - kFirstRoutedLayer;
        slab_routes.n_expert = kExpertCount * kSlabCount;
        slab_routes.top_k = kActiveSlabs;
        slab_routes.selected_ids = ids.data();
        slab_routes.selected_weights = weights.data();
        GroupedSlabObserver grouped_observer;
        const bool needs_individual_slabs =
            mode_ != AllSlabsMode::Direct || !probe_prefix_.empty();
        if (needs_individual_slabs &&
            !grouped_observer.init(routes.n_tokens, ids, err)) {
            return false;
        }
        slab_routes.expert_observer = needs_individual_slabs
            ? &grouped_observer : nullptr;
        if (!eval_moe_streamed_experts(
                slab_engine_, slab_spec, slab_routes, output, err)) {
            return false;
        }
        if (needs_individual_slabs) {
            if (!grouped_observer.complete()) {
                if (err) *err = "grouped slab reduction missed an output";
                return false;
            }
            output.assign(
                static_cast<size_t>(routes.n_tokens) * kDimension, 0.0f);
            std::vector<int> route_order(kNativeTopK);
            std::vector<float> expert_sum(kDimension);
            std::vector<float> full_grouped(kDimension);
            std::vector<float> remaining(kDimension);
            std::vector<int> prefix_lengths(kNativeTopK);
            for (int token = 0; token < routes.n_tokens; ++token) {
                std::iota(route_order.begin(), route_order.end(), 0);
                const size_t route_offset =
                    static_cast<size_t>(token) * kNativeTopK;
                std::stable_sort(
                    route_order.begin(), route_order.end(),
                    [&](int left, int right) {
                        return routes.selected_ids[route_offset + left] <
                            routes.selected_ids[route_offset + right];
                    });
                const auto recompose = [&](const std::vector<int> & prefixes,
                                           float * destination) {
                    std::fill(destination, destination + kDimension, 0.0f);
                    for (const int route : route_order) {
                        std::fill(
                            expert_sum.begin(), expert_sum.end(), 0.0f);
                        for (int rank = 0;
                             rank < prefixes[static_cast<size_t>(route)];
                             ++rank) {
                            const float * contribution =
                                grouped_observer.output(
                                    token, route * kSlabCount + rank);
                            for (int dimension = 0;
                                 dimension < kDimension; ++dimension) {
                                expert_sum[static_cast<size_t>(dimension)] +=
                                    contribution[dimension];
                            }
                        }
                        const float weight =
                            routes.selected_weights[route_offset + route];
                        for (int dimension = 0;
                             dimension < kDimension; ++dimension) {
                            destination[dimension] += weight *
                                expert_sum[static_cast<size_t>(dimension)];
                        }
                    }
                };

                if (mode_ == AllSlabsMode::Direct ||
                    mode_ == AllSlabsMode::Grouped) {
                    std::fill(
                        prefix_lengths.begin(), prefix_lengths.end(),
                        kSlabCount);
                } else if (mode_ == AllSlabsMode::StaticNatural96) {
                    std::fill(prefix_lengths.begin(), prefix_lengths.end(), 6);
                } else {
                    std::fill(prefix_lengths.begin(), prefix_lengths.end(), 0);
                    std::vector<int> full_prefixes(kNativeTopK, kSlabCount);
                    recompose(full_prefixes, full_grouped.data());
                    remaining = full_grouped;
                    const int budget = all_slabs_mode_budget(mode_);
                    for (int selected_count = 0;
                         selected_count < budget; ++selected_count) {
                        int best_route = -1;
                        double best_reduction =
                            -std::numeric_limits<double>::infinity();
                        for (int route = 0;
                             route < kNativeTopK; ++route) {
                            const int rank = prefix_lengths[
                                static_cast<size_t>(route)];
                            if (rank >= kSlabCount) continue;
                            const float * contribution =
                                grouped_observer.output(
                                    token, route * kSlabCount + rank);
                            const double weight = routes.selected_weights[
                                route_offset + route];
                            double dot = 0.0;
                            double norm2 = 0.0;
                            for (int dimension = 0;
                                 dimension < kDimension; ++dimension) {
                                const double value =
                                    weight * contribution[dimension];
                                dot += remaining[
                                    static_cast<size_t>(dimension)] * value;
                                norm2 += value * value;
                            }
                            const double reduction = 2.0 * dot - norm2;
                            if (reduction > best_reduction) {
                                best_reduction = reduction;
                                best_route = route;
                            }
                        }
                        if (best_route < 0) {
                            if (err) *err =
                                "oracle prefix exhausted before its budget";
                            return false;
                        }
                        const int rank = prefix_lengths[
                            static_cast<size_t>(best_route)]++;
                        const float * contribution = grouped_observer.output(
                            token, best_route * kSlabCount + rank);
                        const float weight = routes.selected_weights[
                            route_offset + best_route];
                        for (int dimension = 0;
                             dimension < kDimension; ++dimension) {
                            remaining[static_cast<size_t>(dimension)] -=
                                weight * contribution[dimension];
                        }
                    }
                }
                recompose(
                    prefix_lengths,
                    output.data() + static_cast<size_t>(token) * kDimension);
            }
        }
        if (!probe_completed_ && !probe_prefix_.empty() &&
            model_layer == kFirstRoutedLayer && base_pos == 0) {
            if (!run_predown_probe(
                    model_layer, base_pos, exact_spec, slab_spec, routes,
                    exact_engine, grouped_observer, native_exact, err)) {
                return false;
            }
            probe_completed_ = true;
        }
        observe_numerics(model_layer, routes.n_tokens, native_exact, output);
        return true;
    }

private:
    bool evaluate_recomposed(
            int model_layer,
            const MoeStreamExpertSpec & spec,
            const MoeStreamRouteBatch & routes,
            ggml_backend_t backend,
            std::vector<float> & output,
            std::string * err) {
        const std::string path = natural_sidecar_path(
            sidecar_directory_, model_layer);
        const int descriptor = open_read_only(path);
        if (descriptor < 0) {
            if (err) *err = "cannot open H17 recomposition sidecar " + path;
            return false;
        }
        SlabSidecarHeaderV2 header{};
        bool header_ok = read_exact_at(
            descriptor, &header, sizeof(SlabSidecarHeader), 0) &&
            std::memcmp(header.magic, "K3SLB001", 8) == 0 &&
            (header.version == 1 || header.version == 2);
        if (header_ok && header.version == 2) {
            header_ok = read_exact_at(
                descriptor, &header, sizeof(header), 0);
        }
        header_ok = header_ok &&
            header.model_layer == static_cast<uint32_t>(model_layer) &&
            header.slab_count == kSlabCount &&
            header.slab_size == kSlabSize &&
            header.expert_count == kExpertCount;
        if (!header_ok) {
            close_fd(descriptor);
            if (err) *err = "invalid H17 recomposition sidecar " + path;
            return false;
        }
        const uint64_t gate_slab_bytes = header.version == 1
            ? kSlabComponentBytes : header.gate_slab_bytes;
        const uint64_t up_slab_bytes = header.version == 1
            ? kSlabComponentBytes : header.up_slab_bytes;
        const uint64_t down_slab_bytes = header.version == 1
            ? kSlabComponentBytes : header.down_slab_bytes;
        const size_t gate_full_bytes = static_cast<size_t>(
            gate_slab_bytes * kSlabCount);
        const size_t up_full_bytes = static_cast<size_t>(
            up_slab_bytes * kSlabCount);
        const size_t down_full_row_bytes = ggml_row_size(
            spec.down_type, spec.intermediate_dim);
        if (down_slab_bytes % spec.output_dim != 0 ||
            gate_full_bytes != ggml_row_size(
                spec.gate_type, spec.input_dim) *
                    static_cast<size_t>(spec.intermediate_dim) ||
            up_full_bytes != ggml_row_size(
                spec.up_type, spec.input_dim) *
                    static_cast<size_t>(spec.intermediate_dim)) {
            close_fd(descriptor);
            if (err) *err = "H17 recomposition tensor geometry mismatch";
            return false;
        }
        const size_t down_slab_row_bytes = static_cast<size_t>(
            down_slab_bytes / spec.output_dim);
        if (down_slab_row_bytes * kSlabCount != down_full_row_bytes) {
            close_fd(descriptor);
            if (err) *err = "H17 recomposition down-row geometry mismatch";
            return false;
        }
        const size_t down_full_bytes =
            down_full_row_bytes * static_cast<size_t>(spec.output_dim);
        std::vector<uint8_t> gate(gate_full_bytes);
        std::vector<uint8_t> up(up_full_bytes);
        std::vector<uint8_t> down(down_full_bytes);
        std::vector<uint8_t> slab_down(
            static_cast<size_t>(down_slab_bytes));
        std::vector<float> expert_output;
        const int retained_slabs = mode_ == AllSlabsMode::RecomposedNatural96
            ? 6
            : mode_ == AllSlabsMode::RecomposedNatural144 ? 9 : kSlabCount;
        const int retained_neurons = retained_slabs * kSlabSize;
        std::vector<float> retained_prefix_mask;
        if (retained_neurons < spec.intermediate_dim) {
            retained_prefix_mask.assign(
                static_cast<size_t>(spec.intermediate_dim), 0.0f);
            std::fill_n(retained_prefix_mask.begin(), retained_neurons, 1.0f);
        }
        output.assign(
            static_cast<size_t>(routes.n_tokens) * spec.output_dim, 0.0f);

        for (int token = 0; token < routes.n_tokens; ++token) {
            std::vector<int> route_order(routes.top_k);
            std::iota(route_order.begin(), route_order.end(), 0);
            const size_t route_offset =
                static_cast<size_t>(token) * routes.top_k;
            std::stable_sort(
                route_order.begin(), route_order.end(),
                [&](int left, int right) {
                    return routes.selected_ids[route_offset + left] <
                        routes.selected_ids[route_offset + right];
                });
            float * destination = output.data() +
                static_cast<size_t>(token) * spec.output_dim;
            const float * input = routes.inputs +
                static_cast<size_t>(token) * spec.input_dim;
            for (const int route : route_order) {
                const int expert = routes.selected_ids[route_offset + route];
                for (int rank = 0; rank < kSlabCount; ++rank) {
                    const uint64_t record = header.payload_offset +
                        static_cast<uint64_t>(
                            expert * kSlabCount + rank) *
                            header.slab_bytes;
                    if (!read_exact_at(
                            descriptor,
                            gate.data() +
                                static_cast<size_t>(rank) *
                                    gate_slab_bytes,
                            static_cast<size_t>(gate_slab_bytes),
                            record) ||
                        !read_exact_at(
                            descriptor,
                            up.data() +
                                static_cast<size_t>(rank) *
                                    up_slab_bytes,
                            static_cast<size_t>(up_slab_bytes),
                            record + gate_slab_bytes) ||
                        !read_exact_at(
                            descriptor, slab_down.data(),
                            slab_down.size(),
                            record + gate_slab_bytes +
                                up_slab_bytes)) {
                        close_fd(descriptor);
                        if (err) *err =
                            "short read while recomposing H17 expert";
                        return false;
                    }
                    for (int dimension = 0;
                         dimension < spec.output_dim; ++dimension) {
                        std::memcpy(
                            down.data() +
                                static_cast<size_t>(dimension) *
                                    down_full_row_bytes +
                                static_cast<size_t>(rank) *
                                    down_slab_row_bytes,
                            slab_down.data() +
                                static_cast<size_t>(dimension) *
                                    down_slab_row_bytes,
                            down_slab_row_bytes);
                    }
                }
                if (!evaluate_host_recomposed_expert(
                        backend, spec, input, gate, up, down,
                        retained_prefix_mask.empty()
                            ? nullptr : &retained_prefix_mask,
                        expert_output, err)) {
                    close_fd(descriptor);
                    return false;
                }
                const float weight =
                    routes.selected_weights[route_offset + route];
                for (int dimension = 0;
                     dimension < spec.output_dim; ++dimension) {
                    destination[dimension] +=
                        weight * expert_output[
                            static_cast<size_t>(dimension)];
                }
            }
        }
        close_fd(descriptor);
        return true;
    }

    bool run_predown_probe(
            int model_layer,
            int base_pos,
            const MoeStreamExpertSpec & exact_spec,
            const MoeStreamExpertSpec & slab_spec,
            const MoeStreamRouteBatch & routes,
            MoeHybridStreamEngine & exact_engine,
            const GroupedSlabObserver & grouped_observer,
            const std::vector<float> & native_exact,
            std::string * err) {
        if (routes.n_tokens <= 0 || !grouped_observer.complete()) {
            if (err) *err = "H17 pre-down probe requires complete slab outputs";
            return false;
        }
        struct RouteProbe {
            int expert = -1;
            float weight = 0.0f;
            std::vector<float> native_activation;
            std::vector<float> slab_activation;
            std::vector<float> native_full_down;
            std::vector<float> slab_full_down;
            std::vector<float> slab_split_down;
        };
        std::vector<RouteProbe> probes(kNativeTopK);
        const float * token_input = routes.inputs;
        for (int route = 0; route < kNativeTopK; ++route) {
            RouteProbe & probe = probes[static_cast<size_t>(route)];
            probe.expert = routes.selected_ids[route];
            probe.weight = routes.selected_weights[route];
            if (!capture_probe_activation(
                    exact_engine, routes.layer, probe.expert, exact_spec,
                    token_input, probe.native_activation, err)) {
                return false;
            }
            probe.slab_activation.resize(
                static_cast<size_t>(exact_spec.intermediate_dim));
            for (int rank = 0; rank < kSlabCount; ++rank) {
                std::vector<float> slab_part;
                if (!capture_probe_activation(
                        slab_engine_, routes.layer,
                        probe.expert * kSlabCount + rank,
                        slab_spec, token_input, slab_part, err)) {
                    return false;
                }
                if (slab_part.size() != kSlabSize) {
                    if (err) *err = "H17 slab activation has wrong size";
                    return false;
                }
                std::copy(
                    slab_part.begin(), slab_part.end(),
                    probe.slab_activation.begin() +
                        static_cast<size_t>(rank * kSlabSize));
            }
            if (!project_probe_down(
                    exact_engine, routes.layer, probe.expert, exact_spec,
                    probe.native_activation.data(),
                    probe.native_full_down, err) ||
                !project_probe_down(
                    exact_engine, routes.layer, probe.expert, exact_spec,
                    probe.slab_activation.data(),
                    probe.slab_full_down, err)) {
                return false;
            }
            probe.slab_split_down.assign(kDimension, 0.0f);
            for (int rank = 0; rank < kSlabCount; ++rank) {
                const float * contribution = grouped_observer.output(
                    0, route * kSlabCount + rank);
                for (int dimension = 0;
                     dimension < kDimension; ++dimension) {
                    probe.slab_split_down[static_cast<size_t>(dimension)] +=
                        contribution[dimension];
                }
            }
        }

        std::vector<int> route_order(kNativeTopK);
        std::iota(route_order.begin(), route_order.end(), 0);
        std::stable_sort(
            route_order.begin(), route_order.end(),
            [&](int left, int right) {
                return probes[static_cast<size_t>(left)].expert <
                    probes[static_cast<size_t>(right)].expert;
            });
        std::vector<float> aggregate_native(kDimension, 0.0f);
        std::vector<float> aggregate_slab_full(kDimension, 0.0f);
        std::vector<float> aggregate_slab_split(kDimension, 0.0f);
        for (const int route : route_order) {
            const RouteProbe & probe = probes[static_cast<size_t>(route)];
            for (int dimension = 0; dimension < kDimension; ++dimension) {
                const size_t index = static_cast<size_t>(dimension);
                aggregate_native[index] +=
                    probe.weight * probe.native_full_down[index];
                aggregate_slab_full[index] +=
                    probe.weight * probe.slab_full_down[index];
                aggregate_slab_split[index] +=
                    probe.weight * probe.slab_split_down[index];
            }
        }

        const std::string raw_path = probe_prefix_ + ".bin";
        const std::string table_path = probe_prefix_ + ".tsv";
        std::ofstream raw(raw_path, std::ios::binary);
        if (!raw) {
            if (err) *err = "cannot create H17 pre-down probe " + raw_path;
            return false;
        }
        const char magic[8] = {'K','3','P','D','N','0','0','1'};
        const uint32_t version = 1;
        const uint32_t layer = static_cast<uint32_t>(model_layer);
        const uint32_t position = static_cast<uint32_t>(base_pos);
        const uint32_t route_count = kNativeTopK;
        const uint32_t activation_dimension =
            static_cast<uint32_t>(exact_spec.intermediate_dim);
        const uint32_t output_dimension = kDimension;
        raw.write(magic, sizeof(magic));
        for (const uint32_t value : {
                 version, layer, position, route_count,
                 activation_dimension, output_dimension}) {
            raw.write(
                reinterpret_cast<const char *>(&value), sizeof(value));
        }
        for (int route = 0; route < kNativeTopK; ++route) {
            const RouteProbe & probe = probes[static_cast<size_t>(route)];
            const int32_t route_index = route;
            const int32_t expert = probe.expert;
            raw.write(reinterpret_cast<const char *>(&route_index),
                      sizeof(route_index));
            raw.write(reinterpret_cast<const char *>(&expert), sizeof(expert));
            raw.write(reinterpret_cast<const char *>(&probe.weight),
                      sizeof(probe.weight));
            const uint32_t reserved = 0;
            raw.write(reinterpret_cast<const char *>(&reserved),
                      sizeof(reserved));
            for (const std::vector<float> * values : {
                     &probe.native_activation, &probe.slab_activation,
                     &probe.native_full_down, &probe.slab_full_down,
                     &probe.slab_split_down}) {
                raw.write(
                    reinterpret_cast<const char *>(values->data()),
                    static_cast<std::streamsize>(
                        values->size() * sizeof(float)));
            }
        }
        if (!raw) {
            if (err) *err = "failed to write H17 pre-down probe " + raw_path;
            return false;
        }

        std::ofstream table(table_path);
        if (!table) {
            if (err) *err = "cannot create H17 pre-down table " + table_path;
            return false;
        }
        table << std::setprecision(17)
              << "route\texpert\tweight\tactivation_bit_equal"
                 "\tactivation_maxabs\tactivation_rel_l2"
                 "\tfull_down_bit_equal\tfull_down_maxabs"
                 "\tfull_down_rel_l2\tsplit_down_bit_equal"
                 "\tsplit_down_maxabs\tsplit_down_rel_l2\n";
        for (int route = 0; route < kNativeTopK; ++route) {
            const RouteProbe & probe = probes[static_cast<size_t>(route)];
            const ProbeDifference activation = compare_probe_vectors(
                probe.native_activation.data(),
                probe.slab_activation.data(),
                probe.native_activation.size());
            const ProbeDifference full_down = compare_probe_vectors(
                probe.native_full_down.data(),
                probe.slab_full_down.data(),
                probe.native_full_down.size());
            const ProbeDifference split_down = compare_probe_vectors(
                probe.native_full_down.data(),
                probe.slab_split_down.data(),
                probe.native_full_down.size());
            table << route << '\t' << probe.expert << '\t' << probe.weight
                  << '\t' << activation.bit_equal
                  << '\t' << activation.maximum_absolute
                  << '\t' << activation.relative_l2
                  << '\t' << full_down.bit_equal
                  << '\t' << full_down.maximum_absolute
                  << '\t' << full_down.relative_l2
                  << '\t' << split_down.bit_equal
                  << '\t' << split_down.maximum_absolute
                  << '\t' << split_down.relative_l2 << '\n';
        }
        const ProbeDifference production_self_check = compare_probe_vectors(
            native_exact.data(), aggregate_native.data(), kDimension);
        const ProbeDifference aggregate_full_down = compare_probe_vectors(
            aggregate_native.data(), aggregate_slab_full.data(), kDimension);
        const ProbeDifference aggregate_split_down = compare_probe_vectors(
            aggregate_native.data(), aggregate_slab_split.data(), kDimension);
        table << "aggregate\t-1\t1\t"
              << production_self_check.bit_equal << '\t'
              << production_self_check.maximum_absolute << '\t'
              << production_self_check.relative_l2 << '\t'
              << aggregate_full_down.bit_equal << '\t'
              << aggregate_full_down.maximum_absolute << '\t'
              << aggregate_full_down.relative_l2 << '\t'
              << aggregate_split_down.bit_equal << '\t'
              << aggregate_split_down.maximum_absolute << '\t'
              << aggregate_split_down.relative_l2 << '\n';
        if (!table) {
            if (err) *err = "failed to write H17 pre-down table " + table_path;
            return false;
        }
        std::fprintf(stderr,
            "[kimi-k3-h17] pre-down probe wrote %s and %s "
            "activation-full-down-rel=%.9g split-down-rel=%.9g\n",
            raw_path.c_str(), table_path.c_str(),
            aggregate_full_down.relative_l2,
            aggregate_split_down.relative_l2);
        return true;
    }

    void observe_numerics(int model_layer, int n_tokens,
                          const std::vector<float> & exact,
                          const std::vector<float> & candidate) {
        LayerNumerics & total = numerics_[static_cast<size_t>(model_layer)];
        for (int token = 0; token < n_tokens; ++token) {
            const size_t offset = static_cast<size_t>(token) * kDimension;
            double dot = 0.0;
            double exact_norm2 = 0.0;
            double candidate_norm2 = 0.0;
            double error_norm2 = 0.0;
            for (int d = 0; d < kDimension; ++d) {
                const double left = exact[offset + d];
                const double right = candidate[offset + d];
                const double difference = right - left;
                dot += left * right;
                exact_norm2 += left * left;
                candidate_norm2 += right * right;
                error_norm2 += difference * difference;
            }
            const double cosine = dot /
                std::sqrt(std::max(1.0e-300, exact_norm2 * candidate_norm2));
            const double relative_l2 = std::sqrt(
                error_norm2 / std::max(1.0e-300, exact_norm2));
            ++total.tokens;
            total.cosine_sum += cosine;
            total.relative_l2_sum += relative_l2;
            total.maximum_relative_l2 = std::max(
                total.maximum_relative_l2, relative_l2);
        }
    }

    void finish_metrics() {
        if (numerics_.empty()) return;
        std::ostringstream report;
        report << "model_layer\ttokens\tmean_cosine\tmean_relative_l2"
                  "\tmaximum_relative_l2\n";
        for (int layer = kFirstRoutedLayer;
             layer <= kLastRoutedLayer; ++layer) {
            const LayerNumerics & value =
                numerics_[static_cast<size_t>(layer)];
            if (value.tokens == 0) continue;
            report << layer << '\t' << value.tokens << '\t'
                   << value.cosine_sum / value.tokens << '\t'
                   << value.relative_l2_sum / value.tokens << '\t'
                   << value.maximum_relative_l2 << '\n';
        }
        if (!metrics_path_.empty()) {
            std::ofstream output(metrics_path_);
            output << report.str();
            if (!output) {
                std::fprintf(stderr,
                    "[kimi-k3-h17] cannot write numerical metrics %s\n",
                    metrics_path_.c_str());
            }
        } else {
            std::fprintf(stderr, "%s", report.str().c_str());
        }
        numerics_.clear();
    }

    static constexpr int kFirstRoutedLayer = 1;
    static constexpr int kLastRoutedLayer = 92;
    static constexpr int kRoutedLayerCount = 92;
    MoeHybridStreamEngine slab_engine_;
    size_t max_slab_bytes_ = 0;
    std::vector<LayerNumerics> numerics_;
    std::string metrics_path_;
    std::string sidecar_directory_;
    std::string probe_prefix_;
    bool probe_completed_ = false;
    AllSlabsMode mode_ = AllSlabsMode::Direct;
};

// Honest H20 substrate: each routed layer owns independent calibration cards.
// Any absent, malformed, or provenance-free layer stays exact.  Within a valid
// layer, experts below the exporter's minimum-hit threshold also stay exact.
// The requested budget is 96 slab records, while measured traffic separately
// reports selected sidecar bytes and exact-fallback expert bytes.
class CalibratedAllLayerProvider final : public KimiK3RoutedOutputProvider {
public:
    ~CalibratedAllLayerProvider() override {
        finish_oracle_prefetch();
        finish_metrics();
    }

    bool init(ggml_backend_t backend, const std::string & aux_directory,
              const std::string & sidecar_directory,
              const std::string & route_stats_directory,
              int route_prefix_depth, const char * metrics_path,
              std::string * err) {
        if (!backend || aux_directory.empty() || sidecar_directory.empty()) {
            if (err) *err =
                "calibrated96 needs a compute backend, aux directory, and sidecars";
            return false;
        }
        if (route_prefix_depth != 0 && route_prefix_depth != 6 &&
            route_prefix_depth != 12) {
            if (err) *err = "route prefix depth must be 0, 6, or 12";
            return false;
        }
        if (route_prefix_depth != 0 && route_stats_directory.empty()) {
            if (err) *err =
                "four-route prefix policy needs a route-stats directory";
            return false;
        }
        backend_ = backend;
        route_prefix_depth_ = route_prefix_depth;
        budget_ = route_prefix_depth_ > 0 ? 4 * route_prefix_depth_ : 96;
        if (route_prefix_depth_ > 0) {
            if (const char * raw_phase =
                    std::getenv("DFLASH_KIMI_H21_LAYER_PHASE")) {
                if (*raw_phase == '\0' || std::strcmp(raw_phase, "all") == 0) {
                    layer_phase_ = LayerPhase::All;
                } else if (std::strcmp(raw_phase, "block-ends") == 0) {
                    layer_phase_ = LayerPhase::BlockEnds;
                } else if (std::strcmp(raw_phase, "block-starts") == 0) {
                    layer_phase_ = LayerPhase::BlockStarts;
                } else if (std::strcmp(raw_phase, "block-middles") == 0) {
                    layer_phase_ = LayerPhase::BlockMiddles;
                } else {
                    if (err) *err =
                        "DFLASH_KIMI_H21_LAYER_PHASE must be all, block-ends, block-starts, or block-middles";
                    return false;
                }
            }
        }
        const char * raw_budget =
            std::getenv("DFLASH_KIMI_P20_SLAB_BUDGET");
        const char * raw_budget_table =
            std::getenv("DFLASH_KIMI_H22_LAYER_BUDGETS");
        if (raw_budget_table && *raw_budget_table) {
            if (route_prefix_depth_ != 0 || (raw_budget && *raw_budget)) {
                if (err) *err =
                    "DFLASH_KIMI_H22_LAYER_BUDGETS is incompatible with "
                    "route-prefix policies and DFLASH_KIMI_P20_SLAB_BUDGET";
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
                (route_prefix_depth_ > 0 ? parsed != budget_ :
                    (parsed != 96 && parsed != 192))) {
                if (err) *err =
                    route_prefix_depth_ > 0
                        ? "DFLASH_KIMI_P20_SLAB_BUDGET disagrees with the four-route prefix policy"
                        : "DFLASH_KIMI_P20_SLAB_BUDGET must be 96 or 192";
                return false;
            }
            budget_ = static_cast<int>(parsed);
        }
        if (const char * dynamic =
                std::getenv("DFLASH_KIMI_H22_DYNAMIC_ACTIVE_LAYER")) {
            if (std::strcmp(dynamic, "1") != 0 || route_prefix_depth_ != 0 ||
                !layer_budgets_.empty()) {
                if (err) *err =
                    "DFLASH_KIMI_H22_DYNAMIC_ACTIVE_LAYER must be 1 and "
                    "cannot be combined with route-prefix or budget-table modes";
                return false;
            }
            int initial_layer = 0;
            if (!parse_positive_int(
                    std::getenv("DFLASH_KIMI_H22_ACTIVE_LAYER"),
                    initial_layer) ||
                initial_layer < kFirstRoutedLayer ||
                initial_layer > kLastRoutedLayer) {
                if (err) *err =
                    "dynamic H22 sweep requires DFLASH_KIMI_H22_ACTIVE_LAYER in 1..92";
                return false;
            }
            dynamic_active_layer_ = true;
        }
        if (const char * layout =
                std::getenv("DFLASH_KIMI_P20_PHYSICAL_LAYOUT")) {
            if (std::strcmp(layout, "reference") == 0 || *layout == '\0') {
                sparse_scratch_ = false;
            } else if (std::strcmp(layout, "scratch") == 0) {
                sparse_scratch_ = true;
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
                direct_pread_ = false;
            } else if (std::strcmp(io_backend, "direct-pread") == 0) {
                direct_pread_ = true;
            } else {
                if (err) *err =
                    "DFLASH_KIMI_P20_IO_BACKEND must be current or direct-pread";
                return false;
            }
        }
        if (direct_pread_ && !sparse_scratch_) {
            if (err) *err =
                "P20 direct-pread currently requires the scratch layout";
            return false;
        }
        if (const char * cache =
                std::getenv("DFLASH_KIMI_P30_HOST_CACHE_MB")) {
            int cache_mib = 0;
            if (!parse_positive_int(cache, cache_mib) || cache_mib > 8192) {
                if (err) *err =
                    "DFLASH_KIMI_P30_HOST_CACHE_MB must be in 1..8192";
                return false;
            }
            if (!direct_pread_) {
                if (err) *err =
                    "P30 host cache requires P20 direct-pread";
                return false;
            }
            read_cache_.set_capacity(
                static_cast<size_t>(cache_mib) * 1024 * 1024);
        }
        if (const char * persistent =
                std::getenv("DFLASH_KIMI_P23_PERSISTENT_SCRATCH")) {
            if (std::strcmp(persistent, "1") == 0) {
                persistent_sparse_ = true;
            } else if (std::strcmp(persistent, "0") != 0 && *persistent) {
                if (err) *err =
                    "DFLASH_KIMI_P23_PERSISTENT_SCRATCH must be 0 or 1";
                return false;
            }
        }
        if (persistent_sparse_ && !sparse_scratch_) {
            if (err) *err =
                "P23 persistent scratch requires the scratch layout";
            return false;
        }
        if (const char * compact =
                std::getenv("DFLASH_KIMI_P25_COMPACT_UPLOAD")) {
            if (std::strcmp(compact, "1") == 0) {
                compact_upload_ = true;
            } else if (std::strcmp(compact, "0") != 0 && *compact) {
                if (err) *err =
                    "DFLASH_KIMI_P25_COMPACT_UPLOAD must be 0 or 1";
                return false;
            }
        }
        if (compact_upload_ && (!persistent_sparse_ || !sparse_scratch_ ||
                !direct_pread_)) {
            if (err) *err =
                "P25 compact upload requires persistent scratch and direct-pread";
            return false;
        }
        if (const char * pinned =
                std::getenv("DFLASH_KIMI_P26_PINNED_COMPACT")) {
            if (std::strcmp(pinned, "1") == 0) {
                pinned_compact_ = true;
            } else if (std::strcmp(pinned, "0") != 0 && *pinned) {
                if (err) *err =
                    "DFLASH_KIMI_P26_PINNED_COMPACT must be 0 or 1";
                return false;
            }
        }
        if (pinned_compact_ && !compact_upload_) {
            if (err) *err =
                "P26 pinned compact staging requires compact upload";
            return false;
        }
        if (const char * direct_pinned =
                std::getenv("DFLASH_KIMI_P27_DIRECT_PINNED_COMPACT")) {
            if (std::strcmp(direct_pinned, "1") == 0) {
                direct_pinned_compact_ = true;
            } else if (std::strcmp(direct_pinned, "0") != 0 &&
                       *direct_pinned) {
                if (err) *err =
                    "DFLASH_KIMI_P27_DIRECT_PINNED_COMPACT must be 0 or 1";
                return false;
            }
        }
        if (direct_pinned_compact_ && !pinned_compact_) {
            if (err) *err =
                "P27 direct pinned compact requires P26 pinned compact";
            return false;
        }
        if (const char * oracle_trace =
                std::getenv("DFLASH_KIMI_P28_ORACLE_TRACE")) {
            if (*oracle_trace) oracle_trace_path_ = oracle_trace;
        }
        if (!oracle_trace_path_.empty() && !direct_pinned_compact_) {
            if (err) *err =
                "P28 oracle replay requires P27 direct pinned compact";
            return false;
        }
        if (direct_pread_) {
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
            if (route_prefix_depth_ > 0) {
                state.route_stats_path = route_stats_path(
                    route_stats_directory, layer);
            }
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
        if (!oracle_trace_path_.empty() && !load_oracle_trace(err)) {
            return false;
        }
        metrics_path_ = metrics_path && *metrics_path ? metrics_path : "";
        const std::string budget_description = layer_budgets_.empty()
            ? std::to_string(budget_) : "table:" + layer_budget_path_;
        std::fprintf(stderr,
            "[kimi-k3-calibrated96] status=PILOT policy=%s quality-certified=false "
            "speed-claim=false requested-budget=%s layer-phase=%s dynamic-layer=%s physical-layout=%s "
            "io-backend=%s persistent-scratch=%s compact-upload=%s "
            "pinned-compact=%s direct-pinned-compact=%s p28-oracle=%s "
            "p30-host-cache-mib=%.1f "
            "valid-layers=%d/92 "
            "invalid-layer-action=exact insufficient-expert-action=exact\n",
            route_prefix_depth_ == 6 ? "four-route-half" :
                route_prefix_depth_ == 12 ? "four-route-full" :
                "calibrated-slabs",
            budget_description.c_str(), layer_phase_name(),
            dynamic_active_layer_ ? "enabled" : "disabled",
            sparse_scratch_ ? "scratch" : "reference",
            direct_pread_ ? "direct-pread" : "current",
            persistent_sparse_ ? "enabled" : "disabled",
            compact_upload_ ? "enabled" : "disabled",
            pinned_compact_ ? "enabled" : "disabled",
            direct_pinned_compact_ ? "enabled" : "disabled",
            oracle_trace_path_.empty() ? "disabled" : "one-layer",
            static_cast<double>(read_cache_.capacity()) / (1024.0 * 1024.0),
            valid_layers);
        return true;
    }

    bool handles_layer(int model_layer) const override {
        return model_layer >= kFirstRoutedLayer &&
            model_layer <= kLastRoutedLayer && selected_layer(model_layer) &&
            budget_for_layer(model_layer) < kNativeTopK * kSlabCount;
    }

    bool evaluate(int model_layer, int base_pos,
                  const MoeStreamExpertSpec & spec,
                  const MoeStreamRouteBatch & routes,
                  MoeHybridStreamEngine & exact_engine,
                  std::vector<float> & output,
                  std::string * err) override {
        if (!handles_layer(model_layer) || routes.n_expert != kExpertCount ||
            routes.top_k != kNativeTopK || spec.input_dim != kDimension ||
            spec.output_dim != kDimension) {
            if (err) *err = "calibrated96 received an incompatible routed batch";
            return false;
        }
        LayerState & state = layers_[static_cast<size_t>(model_layer)];
        if (!state.valid || spec.fused_gate_up ||
            !geometry_matches(state, spec)) {
            const bool ok = eval_moe_streamed_experts(
                exact_engine, spec, routes, output, err);
            if (ok) observe_exact_layer(
                state, routes.n_tokens, spec, budget_for_layer(model_layer));
            return ok;
        }
        return evaluate_calibrated(
            model_layer, base_pos, state, spec, routes, exact_engine, output,
            err);
    }

private:
    enum class LayerPhase {
        All,
        BlockEnds,
        BlockStarts,
        BlockMiddles,
    };

    bool selected_layer(int model_layer) const {
        if (dynamic_active_layer_) {
            int active_layer = 0;
            return parse_positive_int(
                       std::getenv("DFLASH_KIMI_H22_ACTIVE_LAYER"),
                       active_layer) &&
                active_layer == model_layer;
        }
        if (route_prefix_depth_ == 0 || layer_phase_ == LayerPhase::All) {
            return true;
        }
        const int phase = model_layer % 12;
        if (layer_phase_ == LayerPhase::BlockEnds) {
            return phase == 11 && model_layer <= 83;
        }
        if (layer_phase_ == LayerPhase::BlockStarts) {
            return phase == 0 && model_layer >= 12 && model_layer <= 84;
        }
        return phase == 6 && model_layer >= 6 && model_layer <= 78;
    }

    int budget_for_layer(int model_layer) const {
        if (layer_budgets_.empty()) return budget_;
        if (model_layer < kFirstRoutedLayer ||
            model_layer > kLastRoutedLayer) {
            return kNativeTopK * kSlabCount;
        }
        return layer_budgets_[static_cast<size_t>(model_layer - 1)];
    }

    const char * layer_phase_name() const {
        if (layer_phase_ == LayerPhase::BlockEnds) return "block-ends";
        if (layer_phase_ == LayerPhase::BlockStarts) return "block-starts";
        if (layer_phase_ == LayerPhase::BlockMiddles) return "block-middles";
        return "all";
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
        std::string route_stats_path;
        uint64_t means_offset = 0;
        uint64_t means_bytes = 0;
        uint64_t native_means_offset = 0;
        uint64_t native_means_bytes = 0;
        uint64_t payload_offset = 0;
        uint64_t slab_bytes = 0;
        uint64_t record_bytes = 0;
        uint64_t gate_slab_bytes = 0;
        uint64_t up_slab_bytes = 0;
        uint64_t down_slab_bytes = 0;
        std::vector<uint16_t> order;
        std::vector<float> importance;
        std::vector<float> native_importance;
        std::vector<uint8_t> calibrated;
        std::vector<uint32_t> hit_counts;
        Traffic traffic;
    };

    static uint64_t oracle_key(int base_pos, int model_layer) {
        return (static_cast<uint64_t>(static_cast<uint32_t>(base_pos)) << 8) |
            static_cast<uint64_t>(model_layer & 0xff);
    }

    static std::vector<std::string> split_tsv(const std::string & line) {
        std::vector<std::string> fields;
        size_t begin = 0;
        for (;;) {
            const size_t tab = line.find('\t', begin);
            fields.push_back(line.substr(
                begin, tab == std::string::npos
                    ? std::string::npos : tab - begin));
            if (tab == std::string::npos) break;
            begin = tab + 1;
        }
        return fields;
    }

    bool load_oracle_trace(std::string * err) {
        std::ifstream input(oracle_trace_path_);
        if (!input) {
            if (err) *err = "cannot open P28 oracle trace " +
                oracle_trace_path_;
            return false;
        }
        std::string line;
        if (!std::getline(input, line) ||
            line.find("request_id\tprompt_id\tbase_pos") != 0) {
            if (err) *err = "P28 oracle trace has an invalid header";
            return false;
        }
        size_t rows = 0;
        try {
            while (std::getline(input, line)) {
                const std::vector<std::string> fields = split_tsv(line);
                if (fields.size() < 18) continue;
                const int base_pos = std::stoi(fields[2]);
                const int model_layer = std::stoi(fields[4]);
                const int expert = std::stoi(fields[5]);
                if (model_layer < kFirstRoutedLayer ||
                    model_layer > kLastRoutedLayer || expert < 0 ||
                    expert >= kExpertCount) {
                    continue;
                }
                const std::string & region = fields[6];
                if (region != "gate" &&
                    region != "native-exact-expert") continue;
                const uint64_t key = oracle_key(base_pos, model_layer);
                P28OracleLayer & layer = oracle_layers_[key];
                layer.base_pos = base_pos;
                layer.model_layer = model_layer;
                auto route = std::find_if(
                    layer.routes.begin(), layer.routes.end(),
                    [&](const P28OracleRoute & value) {
                        return value.expert == expert;
                    });
                if (route == layer.routes.end()) {
                    layer.routes.push_back({expert, {}});
                    route = std::prev(layer.routes.end());
                }
                if (region == "gate") {
                    const LayerState & state =
                        layers_[static_cast<size_t>(model_layer)];
                    const uint64_t offset = std::stoull(fields[11]);
                    const uint64_t expert_base = state.payload_offset +
                        static_cast<uint64_t>(expert) * state.record_bytes;
                    if (offset < expert_base || state.slab_bytes == 0 ||
                        (offset - expert_base) % state.slab_bytes != 0) {
                        if (err) *err =
                            "P28 oracle gate offset is not a slab boundary";
                        return false;
                    }
                    const uint64_t natural =
                        (offset - expert_base) / state.slab_bytes;
                    if (natural >= kSlabCount) {
                        if (err) *err =
                            "P28 oracle natural slab is out of range";
                        return false;
                    }
                    route->naturals.push_back(
                        static_cast<uint16_t>(natural));
                }
                ++rows;
            }
        } catch (const std::exception & exception) {
            if (err) *err = std::string("P28 oracle trace parse failed: ") +
                exception.what();
            return false;
        }
        if (!input.eof() || oracle_layers_.empty()) {
            if (err) *err = "P28 oracle trace is empty or unreadable";
            return false;
        }
        size_t max_arena_bytes = 0;
        for (auto & item : oracle_layers_) {
            P28OracleLayer & layer = item.second;
            std::sort(layer.routes.begin(), layer.routes.end(),
                [](const P28OracleRoute & left,
                   const P28OracleRoute & right) {
                    return left.expert < right.expert;
                });
            if (layer.routes.empty() || layer.routes.size() > kNativeTopK) {
                if (err) *err =
                    "P28 oracle layer has an invalid unique-expert count";
                return false;
            }
            for (P28OracleRoute & route : layer.routes) {
                std::sort(route.naturals.begin(), route.naturals.end());
                if (route.naturals.size() > kSlabCount ||
                    std::adjacent_find(
                        route.naturals.begin(), route.naturals.end()) !=
                            route.naturals.end()) {
                    if (err) *err =
                        "P28 oracle route has invalid selected slabs";
                    return false;
                }
            }
            const LayerState & state =
                layers_[static_cast<size_t>(layer.model_layer)];
            const size_t layer_arena_bytes = std::accumulate(
                layer.routes.begin(), layer.routes.end(), size_t{0},
                [&](size_t total, const P28OracleRoute & route) {
                    return route.naturals.empty() ? total :
                        total + 32 + route.naturals.size() * state.slab_bytes;
                });
            max_arena_bytes = std::max(max_arena_bytes, layer_arena_bytes);
            oracle_order_.push_back(item.first);
        }
        for (size_t index = 1; index < oracle_order_.size(); ++index) {
            oracle_next_[oracle_order_[index - 1]] = oracle_order_[index];
        }
        // CUDA host allocation from the asynchronous read worker proved unsafe
        // under WSL's dxg bridge.  P28 has a frozen trace, so reserve its exact
        // high-water mark once on the initializing thread instead.
        if (!oracle_compact_arena_.ensure(max_arena_bytes, err)) return false;
        std::fprintf(stderr,
            "[kimi-k3-p28] oracle-trace=%s layer-rows=%zu source-rows=%zu "
            "lookahead=1 predictor=none pinned-high-water=%zu\n",
            oracle_trace_path_.c_str(), oracle_layers_.size(), rows,
            max_arena_bytes);
        return true;
    }

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

        if (route_prefix_depth_ > 0) {
            std::ifstream route_stats(
                state.route_stats_path, std::ios::binary | std::ios::ate);
            if (!route_stats) {
                if (err) *err = "missing route stats " +
                    state.route_stats_path;
                return false;
            }
            const uint64_t route_stats_bytes =
                static_cast<uint64_t>(route_stats.tellg());
            route_stats.seekg(0);
            RouteStatsHeader header{};
            route_stats.read(reinterpret_cast<char *>(&header), sizeof(header));
            const uint64_t expected_native_means =
                static_cast<uint64_t>(kExpertCount) * kDimension *
                    sizeof(float);
            const uint64_t expected_native_importance =
                static_cast<uint64_t>(kExpertCount) * sizeof(float);
            const bool header_valid = route_stats &&
                std::memcmp(header.magic, "K3ROUTE1", 8) == 0 &&
                header.version == 1 &&
                header.model_layer == static_cast<uint32_t>(model_layer) &&
                header.expert_count == kExpertCount &&
                header.dimension == kDimension && header.storage == 0 &&
                header.alignment == kAlignment &&
                header.native_means_bytes == expected_native_means &&
                header.native_importance_bytes == expected_native_importance &&
                checked_span(header.native_means_offset,
                             header.native_means_bytes, route_stats_bytes) &&
                checked_span(header.native_importance_offset,
                             header.native_importance_bytes,
                             route_stats_bytes) &&
                header.native_importance_offset +
                    header.native_importance_bytes == route_stats_bytes &&
                std::memcmp(header.fit_state_sha256,
                            aux.fit_state_sha256, 32) == 0;
            if (!header_valid || !read_array(
                    route_stats, header.native_importance_offset,
                    header.native_importance_bytes,
                    state.native_importance, err)) {
                if (err && err->empty()) {
                    *err = "invalid or stale route stats";
                }
                return false;
            }
            if (state.native_importance.size() != kExpertCount ||
                std::any_of(state.native_importance.begin(),
                            state.native_importance.end(),
                            [](float value) {
                                return !std::isfinite(value) || value < 0.0f;
                            })) {
                if (err) *err = "invalid native route importance";
                return false;
            }
            state.native_means_offset = header.native_means_offset;
            state.native_means_bytes = header.native_means_bytes;
        }

        const int fd = open_read_only(state.sidecar_path);
        if (fd < 0) {
            if (err) *err = "missing natural sidecar " + state.sidecar_path;
            return false;
        }
        uint64_t file_bytes = 0;
        SlabSidecarHeaderV2 sidecar{};
        bool header_ok = file_size(fd, file_bytes) &&
            read_exact_at(fd, &sidecar, sizeof(SlabSidecarHeader), 0) &&
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

    bool read_sparse_payloads_direct(
            int fd, const LayerState & state,
            const MoeStreamExpertSpec & spec, int model_layer, int base_pos,
            int token_index, int expert, int prefix_depth,
            std::vector<SparseSlabPayload> & slabs, std::string * err) {
#if defined(_WIN32) || !defined(O_DIRECT)
        (void) fd; (void) state; (void) spec; (void) model_layer;
        (void) base_pos; (void) token_index; (void) expert;
        (void) prefix_depth; (void) slabs;
        if (err) *err = "P20 direct-pread is unavailable on this platform";
        return false;
#else
        struct Completion {
            bool ok = false;
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
                    const ssize_t got = ::pread(
                        fd, raw, aligned_bytes,
                        static_cast<off_t>(aligned_offset));
                    if (got == static_cast<ssize_t>(aligned_bytes)) {
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
                            true, aligned_offset, aligned_bytes};
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
            explicit_read_bytes_ += completions[index].aligned_bytes;
            direct_physical_bytes_ += completions[index].aligned_bytes;
            if (!io_trace_) continue;
            const auto emit = [&](const char * region, const char * qtype,
                                  uint64_t offset, size_t logical,
                                  uint64_t destination_offset,
                                  uint64_t explicit_bytes) {
                io_trace_ << next_request_id_++ << '\t' << prompt_id_ << '\t'
                          << base_pos << '\t' << token_index << '\t'
                          << model_layer << '\t' << expert << '\t' << region
                          << '\t' << qtype << '\t' << prefix_depth
                          << "\t0\t" << state.sidecar_path << '\t' << offset
                          << '\t' << logical << '\t'
                          << completions[index].aligned_offset << '\t'
                          << completions[index].aligned_bytes
                          << "\thost-compact-slab\t" << destination_offset
                          << '\t' << explicit_bytes << '\n';
            };
            emit("gate", ggml_type_name(spec.gate_type), record,
                 slab.gate.size(), 0, completions[index].aligned_bytes);
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
            std::string * err) {
#if defined(_WIN32) || !defined(O_DIRECT)
        (void) fd; (void) state; (void) spec; (void) model_layer;
        (void) base_pos; (void) token_index; (void) route_offset;
        (void) routes; (void) calibrated_routes; (void) selected_by_route;
        (void) payloads; (void) compact_payloads;
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
                compact->bytes = compact->metadata_bytes +
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
                    const P30ReadKey cache_key{
                        model_layer, P30ReadKind::SidecarSlab,
                        task.aligned_offset, task.aligned_bytes};
                    task.cache_hit = read_cache_.get(cache_key, raw);
                    const ssize_t got = task.cache_hit
                        ? static_cast<ssize_t>(task.aligned_bytes)
                        : ::pread(fd, raw, task.aligned_bytes,
                                  static_cast<off_t>(task.aligned_offset));
                    if (got == static_cast<ssize_t>(task.aligned_bytes)) {
                        if (!task.cache_hit) read_cache_.put(cache_key, raw);
                        const auto * source =
                            static_cast<const uint8_t *>(raw) + prefix;
                        if (compact_payloads) {
                            SparseCompactPayload & compact =
                                (*compact_payloads)[
                                    static_cast<size_t>(task.route)];
                            std::memcpy(
                                static_cast<uint8_t *>(compact.data) +
                                    compact.metadata_bytes +
                                    task.slab_index * state.slab_bytes,
                                source, static_cast<size_t>(state.slab_bytes));
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
                 static_cast<size_t>(state.gate_slab_bytes), 0,
                 task.aligned_bytes);
            emit("up", ggml_type_name(spec.up_type),
                 record + state.gate_slab_bytes,
                 static_cast<size_t>(state.up_slab_bytes),
                 state.gate_slab_bytes, 0);
            emit("down", ggml_type_name(spec.down_type),
                 record + state.gate_slab_bytes + state.up_slab_bytes,
                 static_cast<size_t>(state.down_slab_bytes),
                 state.gate_slab_bytes + state.up_slab_bytes, 0);
        }
        return true;
#endif
    }

    std::array<SparseCompactPayload, kNativeTopK> & oracle_slot(int slot) {
        return slot == 0 ? direct_compact_payloads_ :
            oracle_compact_payloads_;
    }

    P28OracleReadResult read_oracle_layer(
            uint64_t key, int slot) {
        P28OracleReadResult result;
#if defined(_WIN32) || !defined(O_DIRECT)
        (void) key; (void) slot;
        result.error = "P28 oracle direct read is unavailable";
        return result;
#else
        const auto found = oracle_layers_.find(key);
        if (found == oracle_layers_.end()) {
            result.error = "P28 oracle layer is absent";
            return result;
        }
        const P28OracleLayer & plan = found->second;
        const LayerState & state =
            layers_[static_cast<size_t>(plan.model_layer)];
        const int fd = open_read_only_direct(state.sidecar_path);
        if (fd < 0) {
            result.error = "P28 cannot open oracle sidecar";
            return result;
        }
        struct Task {
            size_t route_index = 0;
            size_t slab_index = 0;
            int expert = -1;
            uint16_t natural = 0;
            uint64_t aligned_offset = 0;
            size_t aligned_bytes = 0;
            bool ok = false;
        };
        std::array<SparseCompactPayload, kNativeTopK> & payloads =
            oracle_slot(slot);
        std::vector<Task> tasks;
        std::string setup_error;
        size_t arena_offset = 0;
        for (size_t route_index = 0;
             route_index < plan.routes.size(); ++route_index) {
            const P28OracleRoute & route = plan.routes[route_index];
            SparseCompactPayload & compact = payloads[route_index];
            compact.slab_count = static_cast<int>(route.naturals.size());
            compact.gate_slab_bytes =
                static_cast<size_t>(state.gate_slab_bytes);
            compact.up_slab_bytes =
                static_cast<size_t>(state.up_slab_bytes);
            compact.down_slab_bytes =
                static_cast<size_t>(state.down_slab_bytes);
            compact.bytes = compact.metadata_bytes +
                route.naturals.size() * state.slab_bytes;
            if (!route.naturals.empty()) {
                if (slot == 1) {
                    compact.set_external(
                        static_cast<uint8_t *>(oracle_compact_arena_.data) +
                            arena_offset,
                        compact.bytes);
                    arena_offset += compact.bytes;
                } else if (!compact.ensure(compact.bytes, &setup_error)) {
                    close_fd(fd);
                    result.error = setup_error;
                    return result;
                }
            }
            if (!route.naturals.empty()) {
                std::memset(compact.data, 0, compact.metadata_bytes);
            }
            for (size_t slab_index = 0;
                 slab_index < route.naturals.size(); ++slab_index) {
                const uint16_t natural = route.naturals[slab_index];
                std::memcpy(
                    static_cast<uint8_t *>(compact.data) +
                        slab_index * sizeof(uint16_t),
                    &natural, sizeof(natural));
                tasks.push_back({route_index, slab_index, route.expert,
                                 natural});
            }
        }
        const auto started = std::chrono::steady_clock::now();
        std::atomic<size_t> next{0};
        std::atomic<bool> failed{false};
        const size_t workers = std::min<size_t>(16, tasks.size());
        std::vector<std::future<void>> done;
        done.reserve(workers);
        for (size_t worker = 0; worker < workers; ++worker) {
            done.push_back(direct_read_pool_->submit([&]() {
                for (;;) {
                    const size_t index = next.fetch_add(1);
                    if (index >= tasks.size()) break;
                    Task & task = tasks[index];
                    const uint64_t record = state.payload_offset +
                        static_cast<uint64_t>(
                            task.expert * kSlabCount + task.natural) *
                            state.slab_bytes;
                    task.aligned_offset = record &
                        ~(static_cast<uint64_t>(kAlignment) - 1);
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
                    const ssize_t got = ::pread(
                        fd, raw, task.aligned_bytes,
                        static_cast<off_t>(task.aligned_offset));
                    if (got == static_cast<ssize_t>(task.aligned_bytes)) {
                        const uint64_t record_prefix =
                            record - task.aligned_offset;
                        SparseCompactPayload & compact =
                            payloads[task.route_index];
                        std::memcpy(
                            static_cast<uint8_t *>(compact.data) +
                                compact.metadata_bytes +
                                task.slab_index * state.slab_bytes,
                            static_cast<const uint8_t *>(raw) + record_prefix,
                            static_cast<size_t>(state.slab_bytes));
                        task.ok = true;
                    } else {
                        failed = true;
                    }
                    std::free(raw);
                    if (failed.load()) break;
                }
            }));
        }
        for (std::future<void> & value : done) value.get();
        result.elapsed_ns = static_cast<uint64_t>(
            std::chrono::duration_cast<std::chrono::nanoseconds>(
                std::chrono::steady_clock::now() - started).count());
        close_fd(fd);
        result.physical_bytes = std::accumulate(
            tasks.begin(), tasks.end(), uint64_t{0},
            [](uint64_t total, const Task & task) {
                return total + task.aligned_bytes;
            });
        result.ok = !failed.load() && std::all_of(
            tasks.begin(), tasks.end(),
            [](const Task & task) { return task.ok; });
        if (!result.ok) result.error = "P28 aligned oracle read failed";
        return result;
#endif
    }

    bool oracle_matches_live_plan(
            const P28OracleLayer & oracle, const LayerState & state,
            size_t route_offset, const MoeStreamRouteBatch & routes,
            const std::vector<uint8_t> & selected_by_route) const {
        std::vector<P28OracleRoute> live;
        live.reserve(kNativeTopK);
        for (int route = 0; route < kNativeTopK; ++route) {
            const int expert = routes.selected_ids[route_offset + route];
            auto found = std::find_if(
                live.begin(), live.end(),
                [&](const P28OracleRoute & value) {
                    return value.expert == expert;
                });
            if (found == live.end()) {
                live.push_back({expert, {}});
                found = std::prev(live.end());
            }
            for (int rank = 0; rank < kSlabCount; ++rank) {
                if (selected_by_route[
                        static_cast<size_t>(route * kSlabCount + rank)]) {
                    found->naturals.push_back(state.order[
                        static_cast<size_t>(expert) * kSlabCount +
                            rank]);
                }
            }
        }
        for (P28OracleRoute & value : live) {
            std::sort(value.naturals.begin(), value.naturals.end());
            value.naturals.erase(
                std::unique(value.naturals.begin(), value.naturals.end()),
                value.naturals.end());
        }
        std::sort(live.begin(), live.end(),
            [](const P28OracleRoute & left, const P28OracleRoute & right) {
                return left.expert < right.expert;
            });
        if (live.size() != oracle.routes.size()) return false;
        for (size_t index = 0; index < live.size(); ++index) {
            if (live[index].expert != oracle.routes[index].expert ||
                live[index].naturals != oracle.routes[index].naturals) {
                return false;
            }
        }
        return true;
    }

    void account_oracle_result(
            const P28OracleReadResult & result, bool useful) {
        oracle_read_ns_ += result.elapsed_ns;
        explicit_read_bytes_ += result.physical_bytes;
        direct_physical_bytes_ += result.physical_bytes;
        oracle_physical_bytes_ += result.physical_bytes;
        if (!useful) oracle_wasted_bytes_ += result.physical_bytes;
    }

    void trace_oracle_layer(
            const P28OracleLayer & plan, const LayerState & state,
            const MoeStreamExpertSpec & spec) {
        if (!io_trace_) return;
        for (const P28OracleRoute & route : plan.routes) {
            const int prefix_depth =
                static_cast<int>(route.naturals.size());
            for (const uint16_t natural : route.naturals) {
                const uint64_t record = state.payload_offset +
                    static_cast<uint64_t>(
                        route.expert * kSlabCount + natural) *
                        state.slab_bytes;
                const uint64_t aligned_offset = record &
                    ~(static_cast<uint64_t>(kAlignment) - 1);
                const uint64_t record_prefix = record - aligned_offset;
                const uint64_t aligned_bytes =
                    (record_prefix + state.slab_bytes + kAlignment - 1) &
                    ~(static_cast<uint64_t>(kAlignment) - 1);
                const auto emit = [&](const char * region,
                                      const char * qtype, uint64_t offset,
                                      uint64_t logical,
                                      uint64_t destination_offset,
                                      uint64_t explicit_bytes) {
                    io_trace_ << next_request_id_++ << '\t' << prompt_id_
                              << '\t' << plan.base_pos << "\t0\t"
                              << plan.model_layer << '\t' << route.expert
                              << '\t' << region << '\t' << qtype << '\t'
                              << prefix_depth << "\t0\t"
                              << state.sidecar_path << '\t' << offset << '\t'
                              << logical << '\t' << aligned_offset << '\t'
                              << aligned_bytes
                              << "\thost-compact-slab\t"
                              << destination_offset << '\t'
                              << explicit_bytes << '\n';
                };
                emit("gate", ggml_type_name(spec.gate_type), record,
                     state.gate_slab_bytes, 0, aligned_bytes);
                emit("up", ggml_type_name(spec.up_type),
                     record + state.gate_slab_bytes,
                     state.up_slab_bytes, state.gate_slab_bytes, 0);
                emit("down", ggml_type_name(spec.down_type),
                     record + state.gate_slab_bytes + state.up_slab_bytes,
                     state.down_slab_bytes,
                     state.gate_slab_bytes + state.up_slab_bytes, 0);
            }
        }
    }

    bool consume_oracle_prefetch(
            uint64_t key, const LayerState & state, size_t route_offset,
            const MoeStreamRouteBatch & routes,
            const std::vector<uint8_t> & selected_by_route,
            const MoeStreamExpertSpec & spec,
            int & slot) {
        if (!oracle_future_.valid()) return false;
        const auto wait_started = std::chrono::steady_clock::now();
        const P28OracleReadResult result = oracle_future_.get();
        oracle_wait_ns_ += static_cast<uint64_t>(
            std::chrono::duration_cast<std::chrono::nanoseconds>(
                std::chrono::steady_clock::now() - wait_started).count());
        slot = oracle_future_slot_;
        if (oracle_future_key_ != key) {
            account_oracle_result(result, false);
            ++oracle_misses_;
            oracle_future_key_ = 0;
            return false;
        }
        const auto found = oracle_layers_.find(key);
        const bool useful = result.ok && found != oracle_layers_.end() &&
            oracle_matches_live_plan(
                found->second, state, route_offset, routes,
                selected_by_route);
        account_oracle_result(result, useful);
        if (useful) {
            ++oracle_hits_;
            trace_oracle_layer(found->second, state, spec);
        } else {
            ++oracle_misses_;
            if (!result.ok) {
                std::fprintf(stderr,
                    "[kimi-k3-p28] action=sync-fallback reason=%s\n",
                    result.error.c_str());
            }
        }
        oracle_future_key_ = 0;
        return useful;
    }

    void launch_oracle_next(uint64_t current_key, int current_slot) {
        if (oracle_trace_path_.empty() || oracle_future_.valid()) return;
        const auto next = oracle_next_.find(current_key);
        if (next == oracle_next_.end()) return;
        oracle_future_key_ = next->second;
        oracle_future_slot_ = 1 - current_slot;
        const uint64_t key = oracle_future_key_;
        const int slot = oracle_future_slot_;
        oracle_future_ = std::async(
            std::launch::async,
            [this, key, slot]() { return read_oracle_layer(key, slot); });
        ++oracle_launches_;
    }

    void finish_oracle_prefetch() {
        if (!oracle_future_.valid()) return;
        const P28OracleReadResult result = oracle_future_.get();
        account_oracle_result(result, false);
        oracle_future_key_ = 0;
    }

    bool evaluate_calibrated(
            int model_layer, int base_pos, LayerState & state,
            const MoeStreamExpertSpec & spec,
            const MoeStreamRouteBatch & routes,
            MoeHybridStreamEngine & exact_engine,
            std::vector<float> & output, std::string * err) {
        if (read_cache_.enabled() && model_layer == kFirstRoutedLayer &&
            base_pos == 0) {
            if (cache_sequence_started_) read_cache_.reset_sequence();
            cache_sequence_started_ = true;
        }
        const int layer_budget = budget_for_layer(model_layer);
        const int aux_fd = open_read_only(state.aux_path);
        const int sidecar_fd = direct_pread_
            ? open_read_only_direct(state.sidecar_path)
            : open_read_only(state.sidecar_path);
        const int route_stats_fd = route_prefix_depth_ > 0
            ? open_read_only(state.route_stats_path) : -1;
        if (aux_fd < 0 || sidecar_fd < 0 ||
            (route_prefix_depth_ > 0 && route_stats_fd < 0)) {
            if (aux_fd >= 0) close_fd(aux_fd);
            if (sidecar_fd >= 0) close_fd(sidecar_fd);
            if (route_stats_fd >= 0) close_fd(route_stats_fd);
            state.valid = false;
            const bool ok = eval_moe_streamed_experts(
                exact_engine, spec, routes, output, err);
            if (ok) observe_exact_layer(
                state, routes.n_tokens, spec, layer_budget);
            return ok;
        }
        const auto exact_layer_fallback = [&](const char * reason) {
            close_fd(aux_fd);
            close_fd(sidecar_fd);
            if (route_stats_fd >= 0) close_fd(route_stats_fd);
            state.valid = false;
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
        const size_t gate_full_bytes = static_cast<size_t>(
            state.gate_slab_bytes * kSlabCount);
        const size_t up_full_bytes = static_cast<size_t>(
            state.up_slab_bytes * kSlabCount);
        const size_t down_slab_row_bytes = static_cast<size_t>(
            state.down_slab_bytes / spec.output_dim);
        const size_t down_full_row_bytes = down_slab_row_bytes * kSlabCount;
        std::vector<uint8_t> gate(
            sparse_scratch_ ? 0 : gate_full_bytes, 0);
        std::vector<uint8_t> up(
            sparse_scratch_ ? 0 : up_full_bytes, 0);
        std::vector<uint8_t> down(sparse_scratch_ ? 0 :
            down_full_row_bytes * static_cast<size_t>(spec.output_dim), 0);
        std::vector<uint8_t> slab_down(
            static_cast<size_t>(state.down_slab_bytes));
        std::vector<float> mask(
            static_cast<size_t>(spec.intermediate_dim), 0.0f);
        std::vector<float> means(
            static_cast<size_t>(kSlabCount * kDimension));
        std::vector<float> expert_output;
        output.assign(
            static_cast<size_t>(routes.n_tokens) * spec.output_dim, 0.0f);

        for (int token = 0; token < routes.n_tokens; ++token) {
            const size_t route_offset =
                static_cast<size_t>(token) * kNativeTopK;
            std::vector<uint8_t> effective_calibrated = state.calibrated;
            if (layer_budget == kNativeTopK * kSlabCount) {
                std::fill(
                    effective_calibrated.begin(),
                    effective_calibrated.end(), static_cast<uint8_t>(1));
            }
            const KimiK3CalibratedSlabPlan plan = route_prefix_depth_ > 0
                ? plan_kimi_k3_calibrated_route_prefixes(
                    routes.selected_ids + route_offset,
                    routes.selected_weights + route_offset, kNativeTopK,
                    state.native_importance.data(),
                    effective_calibrated.data(), kExpertCount, kSlabCount,
                    4, route_prefix_depth_)
                : plan_kimi_k3_calibrated_slabs(
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
                    if (route_stats_fd >= 0) close_fd(route_stats_fd);
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
            int compact_slot_index = 0;
            const uint64_t current_oracle_key =
                oracle_key(base_pos, model_layer);
            const bool oracle_hit = !oracle_trace_path_.empty() &&
                routes.n_tokens == 1 && consume_oracle_prefetch(
                    current_oracle_key, state, route_offset, routes,
                    selected_by_route, spec, compact_slot_index);
            std::array<SparseCompactPayload, kNativeTopK> &
                current_compact_payloads = oracle_slot(compact_slot_index);
            if (direct_pread_ && !oracle_hit &&
                !read_sparse_payloads_direct_batch(
                    sidecar_fd, state, spec, model_layer, base_pos, token,
                    route_offset, routes, calibrated_routes,
                    selected_by_route, direct_payloads,
                    direct_pinned_compact_
                        ? &current_compact_payloads : nullptr,
                    err)) {
                return exact_layer_fallback(
                    "P20 direct layer-batch sidecar read failed");
            }
            if (!oracle_trace_path_.empty() && routes.n_tokens == 1) {
                launch_oracle_next(
                    current_oracle_key, compact_slot_index);
            }

            std::vector<int> stable_routes = calibrated_routes;
            std::stable_sort(stable_routes.begin(), stable_routes.end(),
                [&](int left, int right) {
                    return routes.selected_ids[route_offset + left] <
                        routes.selected_ids[route_offset + right];
                });
            float * destination = output.data() +
                static_cast<size_t>(token) * spec.output_dim;
            const float * input = routes.inputs +
                static_cast<size_t>(token) * spec.input_dim;
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
                if (route_prefix_depth_ > 0 && prefix_depth == 0) {
                    const uint64_t native_mean_offset =
                        state.native_means_offset +
                        static_cast<uint64_t>(expert) * kDimension *
                            sizeof(float);
                    if (!traced_read_exact_at(
                            route_stats_fd, means.data(),
                            static_cast<size_t>(kDimension) * sizeof(float),
                            native_mean_offset, model_layer, base_pos, token,
                            expert, "native-mean", "f32", 0, false,
                            state.route_stats_path, "host-native-mean", 0)) {
                        return exact_layer_fallback(
                            "short native route mean read");
                    }
                    for (int d = 0; d < kDimension; ++d) {
                        destination[d] += weight * means[static_cast<size_t>(d)];
                    }
                    continue;
                }
                if (!traced_read_exact_at(
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
                    const float * mean = means.data() +
                        static_cast<size_t>(rank) * kDimension;
                    for (int d = 0; d < kDimension; ++d) {
                        destination[d] += weight * mean[d];
                    }
                }

                std::fill(gate.begin(), gate.end(), 0);
                std::fill(up.begin(), up.end(), 0);
                std::fill(down.begin(), down.end(), 0);
                std::fill(mask.begin(), mask.end(), 0.0f);
                std::vector<SparseSlabPayload> sparse_slabs =
                    direct_pread_ && !oracle_hit
                    ? std::move(direct_payloads[static_cast<size_t>(route)])
                    : std::vector<SparseSlabPayload>{};
                const SparseCompactPayload * prepacked_compact = nullptr;
                if (direct_pinned_compact_ && !oracle_hit) {
                    prepacked_compact = &current_compact_payloads[
                        static_cast<size_t>(route)];
                } else if (direct_pinned_compact_ && oracle_hit) {
                    const auto found = oracle_layers_.find(
                        current_oracle_key);
                    if (found != oracle_layers_.end()) {
                        const auto planned = std::find_if(
                            found->second.routes.begin(),
                            found->second.routes.end(),
                            [&](const P28OracleRoute & value) {
                                return value.expert == expert;
                            });
                        if (planned != found->second.routes.end()) {
                            const size_t index = static_cast<size_t>(
                                std::distance(
                                    found->second.routes.begin(), planned));
                            prepacked_compact =
                                &current_compact_payloads[index];
                        }
                    }
                    if (!prepacked_compact) {
                        return exact_layer_fallback(
                            "P28 matched oracle route has no payload");
                    }
                }
                sparse_slabs.reserve(kSlabCount);
                int retained = prepacked_compact
                    ? prepacked_compact->slab_count
                    : static_cast<int>(sparse_slabs.size());
                if (prepacked_compact && retained > 0) {
                    const auto * natural = static_cast<const uint16_t *>(
                        prepacked_compact->data);
                    for (int index = 0; index < retained; ++index) {
                        if (natural[index] >= kSlabCount) {
                            return exact_layer_fallback(
                                "P27 invalid natural slab index");
                        }
                        std::fill_n(mask.begin() +
                            static_cast<size_t>(natural[index]) * kSlabSize,
                            kSlabSize, 1.0f);
                    }
                } else if (direct_pread_) {
                    for (const SparseSlabPayload & slab : sparse_slabs) {
                        std::fill_n(mask.begin() +
                            static_cast<size_t>(slab.natural) * kSlabSize,
                            kSlabSize, 1.0f);
                    }
                }
                for (int rank = 0; !direct_pread_ && rank < kSlabCount;
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
                    if (sparse_scratch_) {
                        sparse.natural = natural;
                        sparse.gate.resize(
                            static_cast<size_t>(state.gate_slab_bytes));
                        sparse.up.resize(
                            static_cast<size_t>(state.up_slab_bytes));
                        sparse.down.resize(
                            static_cast<size_t>(state.down_slab_bytes));
                    }
                    uint8_t * gate_destination = sparse_scratch_
                        ? sparse.gate.data()
                        : gate.data() + static_cast<size_t>(natural) *
                            state.gate_slab_bytes;
                    uint8_t * up_destination = sparse_scratch_
                        ? sparse.up.data()
                        : up.data() + static_cast<size_t>(natural) *
                            state.up_slab_bytes;
                    uint8_t * down_destination = sparse_scratch_
                        ? sparse.down.data() : slab_down.data();
                    const char * host_destination = sparse_scratch_
                        ? "host-compact-slab" : "host-full-width";
                    if (!traced_read_exact_at(
                            sidecar_fd, gate_destination,
                            static_cast<size_t>(state.gate_slab_bytes), record,
                            model_layer, base_pos, token, expert, "gate",
                            ggml_type_name(spec.gate_type), prefix_depth, false,
                            state.sidecar_path, host_destination,
                            sparse_scratch_ ? 0 :
                                static_cast<uint64_t>(natural) *
                                    state.gate_slab_bytes) ||
                        !traced_read_exact_at(
                            sidecar_fd, up_destination,
                            static_cast<size_t>(state.up_slab_bytes),
                            record + state.gate_slab_bytes, model_layer,
                            base_pos, token, expert, "up",
                            ggml_type_name(spec.up_type), prefix_depth, false,
                            state.sidecar_path, host_destination,
                            sparse_scratch_ ? 0 :
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
                            sparse_scratch_ ? "host-compact-slab" :
                                "host-compact-down", 0)) {
                        return exact_layer_fallback(
                            "short mixed-layout slab read");
                    }
                    if (sparse_scratch_) {
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
                    std::fill_n(mask.begin() +
                        static_cast<size_t>(natural) * kSlabSize,
                        kSlabSize, 1.0f);
                    ++retained;
                }
                const bool evaluated = retained == 0 || (sparse_scratch_
                    ? evaluate_sparse_device_expert(
                        persistent_sparse_ ? &sparse_device_evaluator_ : nullptr,
                        backend_, spec, input,
                        sparse_slabs, prepacked_compact, mask,
                        down_slab_row_bytes, expert_output,
                        authoritative_h2d_bytes_, metadata_h2d_bytes_,
                        device_zero_bytes_, compact_upload_,
                        pinned_compact_, err)
                    : evaluate_host_recomposed_expert(
                        backend_, spec, input, gate, up, down,
                        retained == kSlabCount ? nullptr : &mask,
                        expert_output, err));
                if (retained > 0 && !evaluated) {
                    const std::string detail = err && !err->empty()
                        ? "full-width recomposition failed: " + *err
                        : "full-width recomposition failed";
                    return exact_layer_fallback(detail.c_str());
                }
                if (!sparse_scratch_ && retained > 0) {
                    reference_full_weight_h2d_bytes_ +=
                        gate.size() + up.size() + down.size();
                }
                if (retained > 0) {
                    for (int d = 0; d < spec.output_dim; ++d) {
                        destination[d] += weight *
                            expert_output[static_cast<size_t>(d)];
                    }
                }
            }

            if (!fallback_routes.empty()) {
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
                fallback.top_k = static_cast<int>(fallback_routes.size());
                fallback.inputs = input;
                fallback.selected_ids = fallback_ids.data();
                fallback.selected_weights = fallback_weights.data();
                fallback.expert_observer = nullptr;
                for (const int route : fallback_routes) {
                    trace_fallback(model_layer, base_pos, token,
                        routes.selected_ids[route_offset + route], spec);
                }
                std::vector<float> exact;
                if (!eval_moe_streamed_experts(
                        exact_engine, spec, fallback, exact, err)) {
                    close_fd(aux_fd); close_fd(sidecar_fd);
                    if (route_stats_fd >= 0) close_fd(route_stats_fd);
                    return false;
                }
                for (int d = 0; d < spec.output_dim; ++d) {
                    destination[d] += exact[static_cast<size_t>(d)];
                }
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
        if (route_stats_fd >= 0) close_fd(route_stats_fd);
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
            sparse_scratch_ ? "scratch" : "reference",
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
        if (!oracle_trace_path_.empty()) {
            std::fprintf(stderr,
                "[kimi-k3-p28] launches=%llu hits=%llu misses=%llu "
                "oracle-read-ns=%llu demand-wait-ns=%llu "
                "physical-bytes=%llu wasted-bytes=%llu extra-pinned-bytes=%zu\n",
                static_cast<unsigned long long>(oracle_launches_),
                static_cast<unsigned long long>(oracle_hits_),
                static_cast<unsigned long long>(oracle_misses_),
                static_cast<unsigned long long>(oracle_read_ns_),
                static_cast<unsigned long long>(oracle_wait_ns_),
                static_cast<unsigned long long>(oracle_physical_bytes_),
                static_cast<unsigned long long>(oracle_wasted_bytes_),
                oracle_compact_arena_.capacity);
        }
        layers_.clear();
    }

    static constexpr int kFirstRoutedLayer = 1;
    static constexpr int kLastRoutedLayer = 92;
    ggml_backend_t backend_ = nullptr;
    std::vector<LayerState> layers_;
    std::string metrics_path_;
    std::ofstream io_trace_;
    std::string prompt_id_ = "0";
    ProcessIoSnapshot process_io_start_{};
    uint64_t next_request_id_ = 0;
    uint64_t explicit_read_bytes_ = 0;
    int route_prefix_depth_ = 0;
    LayerPhase layer_phase_ = LayerPhase::All;
    bool sparse_scratch_ = false;
    SparseDeviceExpertEvaluator sparse_device_evaluator_;
    bool persistent_sparse_ = false;
    bool compact_upload_ = false;
    bool pinned_compact_ = false;
    bool direct_pinned_compact_ = false;
    bool direct_pread_ = false;
    std::unique_ptr<P20DirectReadPool> direct_read_pool_;
    std::array<SparseCompactPayload, kNativeTopK>
        direct_compact_payloads_;
    std::array<SparseCompactPayload, kNativeTopK>
        oracle_compact_payloads_;
    P28PinnedArena oracle_compact_arena_;
    std::string oracle_trace_path_;
    std::map<uint64_t, P28OracleLayer> oracle_layers_;
    std::vector<uint64_t> oracle_order_;
    std::map<uint64_t, uint64_t> oracle_next_;
    std::future<P28OracleReadResult> oracle_future_;
    uint64_t oracle_future_key_ = 0;
    int oracle_future_slot_ = 0;
    uint64_t oracle_launches_ = 0;
    uint64_t oracle_hits_ = 0;
    uint64_t oracle_misses_ = 0;
    uint64_t oracle_read_ns_ = 0;
    uint64_t oracle_wait_ns_ = 0;
    uint64_t oracle_physical_bytes_ = 0;
    uint64_t oracle_wasted_bytes_ = 0;
    int budget_ = 96;
    std::vector<int32_t> layer_budgets_;
    std::string layer_budget_path_;
    bool dynamic_active_layer_ = false;
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

} // namespace

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

std::vector<int32_t> select_kimi_k3_whole_expert_routes(
        const int32_t * expert_ids, const float * router_weights, int top_k,
        const float * expert_importance, int expert_count, int budget) {
    if (!expert_ids || !router_weights || !expert_importance || top_k <= 0 ||
        expert_count <= 0 || budget <= 0 || budget > top_k) {
        return {};
    }
    std::vector<Candidate> candidates;
    candidates.reserve(static_cast<size_t>(top_k));
    for (int route = 0; route < top_k; ++route) {
        const int expert = expert_ids[route];
        if (expert < 0 || expert >= expert_count) return {};
        candidates.push_back({
            std::abs(router_weights[route]) * expert_importance[expert],
            route, expert, 0});
    }
    std::stable_sort(candidates.begin(), candidates.end(), better_candidate);
    std::vector<int32_t> selected;
    selected.reserve(static_cast<size_t>(budget));
    for (int i = 0; i < budget; ++i) {
        selected.push_back(candidates[static_cast<size_t>(i)].route);
    }
    return selected;
}

std::vector<int32_t> select_kimi_k3_route_slab_prefix_ids(
        const int32_t * expert_ids, const float * router_weights, int top_k,
        const float * expert_importance, int expert_count,
        int slabs_per_expert, int route_budget, int slabs_per_route) {
    if (slabs_per_expert <= 0 || slabs_per_route <= 0 ||
        slabs_per_route > slabs_per_expert) {
        return {};
    }
    const std::vector<int32_t> routes = select_kimi_k3_whole_expert_routes(
        expert_ids, router_weights, top_k, expert_importance, expert_count,
        route_budget);
    if (routes.size() != static_cast<size_t>(route_budget)) return {};
    std::vector<int32_t> selected;
    selected.reserve(static_cast<size_t>(route_budget * slabs_per_route));
    for (const int32_t route : routes) {
        const int expert = expert_ids[route];
        for (int rank = 0; rank < slabs_per_route; ++rank) {
            selected.push_back(expert * slabs_per_expert + rank);
        }
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

KimiK3CalibratedSlabPlan plan_kimi_k3_calibrated_route_prefixes(
        const int32_t * expert_ids, const float * router_weights, int top_k,
        const float * expert_importance,
        const uint8_t * calibrated_experts, int expert_count,
        int slabs_per_expert, int route_budget, int slabs_per_route) {
    KimiK3CalibratedSlabPlan result;
    result.requested_budget = route_budget * slabs_per_route;
    if (!expert_ids || !router_weights || !expert_importance ||
        !calibrated_experts || top_k <= 0 || expert_count <= 0 ||
        slabs_per_expert <= 0 || route_budget <= 0 ||
        slabs_per_route <= 0 || slabs_per_route > slabs_per_expert) {
        return result;
    }
    std::vector<int32_t> calibrated_ids;
    std::vector<float> calibrated_weights;
    calibrated_ids.reserve(static_cast<size_t>(top_k));
    calibrated_weights.reserve(static_cast<size_t>(top_k));
    for (int route = 0; route < top_k; ++route) {
        const int expert = expert_ids[route];
        if (expert < 0 || expert >= expert_count) {
            result.selected_slab_ids.clear();
            result.exact_route_indices.clear();
            return result;
        }
        if (calibrated_experts[expert] != 0) {
            calibrated_ids.push_back(expert);
            calibrated_weights.push_back(router_weights[route]);
        } else {
            result.exact_route_indices.push_back(route);
        }
    }
    const int selected_routes = std::min(
        route_budget, static_cast<int>(calibrated_ids.size()));
    if (selected_routes > 0) {
        result.selected_slab_ids = select_kimi_k3_route_slab_prefix_ids(
            calibrated_ids.data(), calibrated_weights.data(),
            static_cast<int>(calibrated_ids.size()), expert_importance,
            expert_count, slabs_per_expert, selected_routes,
            slabs_per_route);
    }
    return result;
}

bool create_kimi_k3_progressive_provider_from_env(
        ggml_backend_t expert_backend,
        std::unique_ptr<KimiK3RoutedOutputProvider> & out,
        std::string * err) {
    out.reset();
    const char * raw_kind = std::getenv("DFLASH_KIMI_LAYER1_PROVIDER");
    if (!raw_kind || !*raw_kind || std::strcmp(raw_kind, "exact") == 0) {
        return true;
    }
    const bool all_layers_calibrated96 =
        std::strcmp(raw_kind, "all-layers-calibrated96") == 0;
    const bool all_layers_four_route_half =
        std::strcmp(raw_kind,
                    "all-layers-four-route-half-slabs") == 0;
    const bool all_layers_four_route_full =
        std::strcmp(raw_kind,
                    "all-layers-four-route-full-slabs") == 0;
    if (all_layers_calibrated96 || all_layers_four_route_half ||
        all_layers_four_route_full) {
        const char * aux_directory =
            std::getenv("DFLASH_KIMI_CALIBRATED96_AUX_DIR");
        const char * sidecar_directory =
            std::getenv("DFLASH_KIMI_ALL_SLAB_SIDECAR_DIR");
        const char * route_stats_directory =
            std::getenv("DFLASH_KIMI_ROUTE_STATS_DIR");
        if (!aux_directory || !*aux_directory ||
            !sidecar_directory || !*sidecar_directory ||
            ((all_layers_four_route_half || all_layers_four_route_full) &&
                (!route_stats_directory || !*route_stats_directory))) {
            if (err) *err =
                "all-layer calibrated providers require "
                "DFLASH_KIMI_CALIBRATED96_AUX_DIR and "
                "DFLASH_KIMI_ALL_SLAB_SIDECAR_DIR; four-route policies also "
                "requires DFLASH_KIMI_ROUTE_STATS_DIR";
            return false;
        }
        auto provider = std::make_unique<CalibratedAllLayerProvider>();
        if (!provider->init(
                expert_backend, aux_directory, sidecar_directory,
                route_stats_directory ? route_stats_directory : "",
                all_layers_four_route_half ? 6 :
                    all_layers_four_route_full ? 12 : 0,
                std::getenv("DFLASH_KIMI_CALIBRATED96_METRICS_OUT"), err)) {
            return false;
        }
        out = std::move(provider);
        return true;
    }
    AllSlabsMode all_slabs_mode = AllSlabsMode::Direct;
    bool is_all_slabs = true;
    if (std::strcmp(raw_kind, "all-slabs") == 0) {
        all_slabs_mode = AllSlabsMode::Direct;
    } else if (std::strcmp(raw_kind, "all-slabs-grouped") == 0) {
        all_slabs_mode = AllSlabsMode::Grouped;
    } else if (std::strcmp(raw_kind, "all-slabs-recomposed") == 0) {
        all_slabs_mode = AllSlabsMode::Recomposed;
    } else if (std::strcmp(
                   raw_kind, "all-slabs-recomposed-natural96") == 0) {
        all_slabs_mode = AllSlabsMode::RecomposedNatural96;
    } else if (std::strcmp(
                   raw_kind, "all-slabs-recomposed-natural144") == 0) {
        all_slabs_mode = AllSlabsMode::RecomposedNatural144;
    } else if (std::strcmp(raw_kind, "all-slabs-static96") == 0) {
        all_slabs_mode = AllSlabsMode::StaticNatural96;
    } else if (std::strcmp(raw_kind, "all-slabs-oracle96") == 0) {
        all_slabs_mode = AllSlabsMode::OracleNatural96;
    } else if (std::strcmp(raw_kind, "all-slabs-oracle144") == 0) {
        all_slabs_mode = AllSlabsMode::OracleNatural144;
    } else {
        is_all_slabs = false;
    }
    if (is_all_slabs) {
        const char * directory =
            std::getenv("DFLASH_KIMI_ALL_SLAB_SIDECAR_DIR");
        if (!directory || !*directory) {
            if (err) *err =
                "DFLASH_KIMI_ALL_SLAB_SIDECAR_DIR is required for all-slabs";
            return false;
        }
        auto provider = std::make_unique<AllSlabsProvider>();
        if (!provider->init(
                expert_backend, directory,
                std::getenv("DFLASH_KIMI_ALL_SLAB_METRICS_OUT"),
                all_slabs_mode, err)) {
            return false;
        }
        out = std::move(provider);
        return true;
    }
    ProviderKind kind;
    if (std::strcmp(raw_kind, "slabs") == 0) kind = ProviderKind::Slabs;
    else if (std::strcmp(raw_kind, "slabs-recomposed") == 0) {
        kind = ProviderKind::RecomposedSlabs;
    }
    else if (std::strcmp(raw_kind, "four-route-half-slabs-recomposed") == 0) {
        kind = ProviderKind::FourRouteHalfSlabsRecomposed;
    }
    else if (std::strcmp(raw_kind, "whole") == 0) kind = ProviderKind::Whole;
    else {
        if (err) *err =
            "DFLASH_KIMI_LAYER1_PROVIDER must be exact, slabs, "
            "slabs-recomposed, four-route-half-slabs-recomposed, whole, "
            "all-layers-calibrated96, "
            "all-slabs, all-slabs-grouped, all-slabs-recomposed, "
            "all-slabs-recomposed-natural96, "
            "all-slabs-recomposed-natural144, "
            "all-slabs-static96, "
            "all-slabs-oracle96, or all-slabs-oracle144";
        return false;
    }
    int budget = kind == ProviderKind::FourRouteHalfSlabsRecomposed ? 24 : 0;
    if ((kind != ProviderKind::FourRouteHalfSlabsRecomposed &&
         !parse_positive_int(
            std::getenv("DFLASH_KIMI_LAYER1_BUDGET"), budget)) ||
        ((kind == ProviderKind::Slabs ||
          kind == ProviderKind::RecomposedSlabs) &&
         budget > kNativeTopK * kSlabCount) ||
        (kind == ProviderKind::Whole && budget > kNativeTopK)) {
        if (err) *err = "invalid DFLASH_KIMI_LAYER1_BUDGET for provider";
        return false;
    }
    const char * aux = std::getenv("DFLASH_KIMI_SLAB_AUX");
    if (!aux || !*aux) {
        if (err) *err = "DFLASH_KIMI_SLAB_AUX is required";
        return false;
    }
    const char * sidecar = std::getenv("DFLASH_KIMI_SLAB_SIDECAR");
    if ((kind == ProviderKind::Slabs ||
         kind == ProviderKind::RecomposedSlabs ||
         kind == ProviderKind::FourRouteHalfSlabsRecomposed) &&
        (!sidecar || !*sidecar)) {
        if (err) *err = "DFLASH_KIMI_SLAB_SIDECAR is required for slabs";
        return false;
    }
    auto provider = std::make_unique<ProgressiveProvider>();
    int model_layer = 1;
    if (const char * raw_layer =
            std::getenv("DFLASH_KIMI_PROVIDER_LAYER")) {
        if (!parse_positive_int(raw_layer, model_layer)) {
            if (err) *err = "invalid DFLASH_KIMI_PROVIDER_LAYER";
            return false;
        }
    }
    int active_position = -1;
    if (const char * raw_position =
            std::getenv("DFLASH_KIMI_LAYER1_ACTIVE_POSITION")) {
        char * end = nullptr;
        const long parsed = std::strtol(raw_position, &end, 10);
        if (end == raw_position || *end != '\0' || parsed < 0 ||
            parsed > std::numeric_limits<int>::max()) {
            if (err) *err = "invalid DFLASH_KIMI_LAYER1_ACTIVE_POSITION";
            return false;
        }
        active_position = static_cast<int>(parsed);
    }
    if (!provider->init(
            expert_backend, kind, budget, aux, sidecar ? sidecar : "",
            std::getenv("DFLASH_KIMI_LAYER1_TRACE_OUT"), model_layer,
            active_position, err)) {
        return false;
    }
    out = std::move(provider);
    return true;
}

} // namespace dflash::common
