#include "kimi_k3_progressive_provider.h"

#include <algorithm>
#include <cerrno>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <limits>
#include <numeric>
#include <sstream>
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
             kind_ == ProviderKind::RecomposedSlabs) && !init_sidecar(
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
            : kind_ == ProviderKind::RecomposedSlabs
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
            const std::vector<int32_t> selected =
                select_kimi_k3_slab_prefix_ids(
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
            float * destination = output.data() +
                static_cast<size_t>(token) * spec.output_dim;
            // Start from the calibrated aggregate tail, then replace exactly
            // the selected per-expert/rank cards with live computations.
            // At full budget there is no tail.  Do not add and subtract its
            // cards: that harmless-looking cancellation would make the H19
            // full-width identity control depend on F32 rounding.
            const bool use_mean_tail = budget_ < kNativeTopK * kSlabCount;
            if (use_mean_tail) {
                for (int route = 0; route < kNativeTopK; ++route) {
                    const int expert = routes.selected_ids[route_offset + route];
                    const float weight = routes.selected_weights[route_offset + route];
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
    else if (std::strcmp(raw_kind, "whole") == 0) kind = ProviderKind::Whole;
    else {
        if (err) *err =
            "DFLASH_KIMI_LAYER1_PROVIDER must be exact, slabs, "
            "slabs-recomposed, whole, "
            "all-slabs, all-slabs-grouped, all-slabs-recomposed, "
            "all-slabs-recomposed-natural96, "
            "all-slabs-recomposed-natural144, "
            "all-slabs-static96, "
            "all-slabs-oracle96, or all-slabs-oracle144";
        return false;
    }
    int budget = 0;
    if (!parse_positive_int(
            std::getenv("DFLASH_KIMI_LAYER1_BUDGET"), budget) ||
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
         kind == ProviderKind::RecomposedSlabs) &&
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
