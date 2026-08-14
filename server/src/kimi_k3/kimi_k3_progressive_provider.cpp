#include "kimi_k3_progressive_provider.h"

#include <algorithm>
#include <cerrno>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <limits>
#include <numeric>
#include <string>
#include <utility>
#include <vector>

#if defined(_WIN32)
#include <fcntl.h>
#include <io.h>
#include <sys/stat.h>
#else
#include <fcntl.h>
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

enum class ProviderKind : uint32_t { Slabs = 1, Whole = 2 };

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
        if (kind_ == ProviderKind::Slabs && !init_sidecar(
                expert_backend, sidecar_path, err)) {
            return false;
        }
        if (trace_path && *trace_path && !init_trace(trace_path, err)) {
            return false;
        }
        std::fprintf(stderr,
            "[kimi-k3-h16] provider=%s budget=%d model-layer=1 "
            "teacher=exact model-layer=%d active-position=%d trace=%s\n",
            kind_ == ProviderKind::Slabs ? "slabs" : "whole",
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

bool create_kimi_k3_progressive_provider_from_env(
        ggml_backend_t expert_backend,
        std::unique_ptr<KimiK3RoutedOutputProvider> & out,
        std::string * err) {
    out.reset();
    const char * raw_kind = std::getenv("DFLASH_KIMI_LAYER1_PROVIDER");
    if (!raw_kind || !*raw_kind || std::strcmp(raw_kind, "exact") == 0) {
        return true;
    }
    ProviderKind kind;
    if (std::strcmp(raw_kind, "slabs") == 0) kind = ProviderKind::Slabs;
    else if (std::strcmp(raw_kind, "whole") == 0) kind = ProviderKind::Whole;
    else {
        if (err) *err = "DFLASH_KIMI_LAYER1_PROVIDER must be exact, slabs, or whole";
        return false;
    }
    int budget = 0;
    if (!parse_positive_int(
            std::getenv("DFLASH_KIMI_LAYER1_BUDGET"), budget) ||
        (kind == ProviderKind::Slabs && budget > kNativeTopK * kSlabCount) ||
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
    if (kind == ProviderKind::Slabs && (!sidecar || !*sidecar)) {
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
