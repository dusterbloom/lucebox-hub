#include "kimi_k3_backend.h"

#include "common/sampler.h"
#include "common/snapshot_backend.h"
#include "common/platform_env.h"
#include "device_runtime.h"
#include "dflash27b.h"
#include "internal.h"

// ggml retains the cuda-named accelerator API for both CUDA and HIP builds.
#include "ggml-cuda.h"

#include <algorithm>
#include <cerrno>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iterator>
#include <limits>
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

bool parse_binary_env(const char * name, bool & value, std::string & error) {
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

bool parse_prefill_width(const char * raw, int & width) {
    width = 1;
    if (!raw || !*raw || std::strcmp(raw, "1") == 0) return true;
    for (int candidate : {8, 64, 1024}) {
        if (std::strcmp(raw, std::to_string(candidate).c_str()) == 0) {
            width = candidate;
            return true;
        }
    }
    return false;
}

bool apply_kimi_k3_production_defaults(std::string & error) {
    const char * profile = std::getenv("DFLASH_KIMI_PRODUCTION_DEFAULTS");
    if (profile && *profile && std::strcmp(profile, "0") != 0 &&
        std::strcmp(profile, "1") != 0) {
        error = "DFLASH_KIMI_PRODUCTION_DEFAULTS must be 0 or 1";
        return false;
    }
    if (profile && std::strcmp(profile, "0") == 0) return true;

    const char * provider = std::getenv("DFLASH_KIMI_LAYER1_PROVIDER");
    const char * aux = std::getenv("DFLASH_KIMI_CALIBRATED96_AUX_DIR");
    const char * sidecars =
        std::getenv("DFLASH_KIMI_ALL_SLAB_SIDECAR_DIR");
    const bool has_aux = aux && *aux;
    const bool has_sidecars = sidecars && *sidecars;
    if ((!provider || !*provider) && has_aux != has_sidecars) {
        error = "Kimi-K3 production defaults require both calibrated96 "
                "auxiliary and sidecar directories";
        return false;
    }
    if ((!provider || !*provider) && has_aux && has_sidecars) {
        if (set_environment_variable(
                "DFLASH_KIMI_LAYER1_PROVIDER",
                "all-layers-calibrated96", false) != 0) {
            error = "cannot select the Kimi-K3 production provider";
            return false;
        }
        provider = std::getenv("DFLASH_KIMI_LAYER1_PROVIDER");
    }
    if (!provider || std::strcmp(provider, "all-layers-calibrated96") != 0) {
        return true;
    }

    struct DefaultValue { const char * name; const char * value; };
    static constexpr DefaultValue defaults[] = {
        {"ROCBLAS_USE_HIPBLASLT", "0"},
        {"DFLASH_MOE_NVME_DIRECT", "on"},
        {"DFLASH_MOE_NVME_DEVICE_CACHE_MB", "8192"},
        {"DFLASH_CUDA_MMVQ_TOKENWISE", "1"},
        {"DFLASH_CUDA_MMVQ_QK_EXACT_WIDTH8", "1"},
        {"DFLASH_KIMI_PREFILL_CHUNK", "1024"},
        {"DFLASH_KIMI_P58_EXACT_MULTIROW", "1"},
        {"DFLASH_KIMI_SIDECAR_AUTHORITATIVE", "1"},
        {"DFLASH_KIMI_P20_PHYSICAL_LAYOUT", "scratch"},
        {"DFLASH_KIMI_P20_IO_BACKEND", "direct-pread"},
        {"DFLASH_KIMI_P20_SLAB_BUDGET", "96"},
        {"DFLASH_KIMI_P23_PERSISTENT_SCRATCH", "1"},
        {"DFLASH_KIMI_P25_COMPACT_UPLOAD", "1"},
        {"DFLASH_KIMI_P26_PINNED_COMPACT", "1"},
        {"DFLASH_KIMI_P27_DIRECT_PINNED_COMPACT", "1"},
        {"DFLASH_KIMI_P30_HOST_CACHE_MB", "16384"},
        {"DFLASH_KIMI_P30_BORROWED_RECORDS", "1"},
        {"DFLASH_KIMI_P40_DEVICE_VARIANT_CACHE", "1"},
        {"DFLASH_KIMI_P40_LAYER_EPOCH", "1"},
        {"DFLASH_KIMI_P41_COMPACT_EXECUTOR", "1"},
        {"DFLASH_KIMI_P42_ORDERED_DEVICE_JOIN", "1"},
        {"DFLASH_KIMI_P45_ASYNC_COMPACT_QUEUE", "1"},
        {"DFLASH_KIMI_P46_PERSISTENT_ROUTED_PREP", "1"},
        {"DFLASH_KIMI_ROUTER_WIDTH8", "1"},
        {"DFLASH_KIMI_P58_EXACT_CORE_GROUP_WIDTH", "8"},
        {"DFLASH_KIMI_P58_EXACT_MLA_GROUP_WIDTH", "8"},
        {"DFLASH_KIMI_P58_EXACT_TAIL_GROUP_WIDTH", "8"},
        {"DFLASH_KIMI_EXACT_MACRO_UNION", "1"},
        {"DFLASH_KIMI_EXACT_MACRO_UNION_PREFETCH", "1"},
    };
    for (const DefaultValue & value : defaults) {
        if (set_environment_variable(
                value.name, value.value, false) != 0) {
            error = std::string("cannot set Kimi-K3 production default ") +
                value.name;
            return false;
        }
    }
    std::fprintf(stderr,
        "[kimi-k3] production-defaults=enabled profile=exact-m1024 "
        "operator-overrides=preserved\n");
    return true;
}

void close_descriptor(int fd) {
#if defined(_WIN32)
    if (fd >= 0) ::_close(fd);
#else
    if (fd >= 0) ::close(fd);
#endif
}

struct ScopedDescriptors {
    std::vector<int> values;
    ~ScopedDescriptors() {
        for (int fd : values) close_descriptor(fd);
    }
};

bool open_stream_source(const std::string & path,
                        ScopedDescriptors & descriptors,
                        MoeNvmeSource & source,
                        std::string & error) {
#if defined(_WIN32)
    const int fd = ::_open(path.c_str(), _O_RDONLY | _O_BINARY);
#else
    const int fd = ::open(path.c_str(), O_RDONLY | O_CLOEXEC);
#endif
    if (fd < 0) {
        error = "cannot open expert shard " + path + ": " +
            std::strerror(errno);
        return false;
    }
    descriptors.values.push_back(fd);

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
        error = "cannot determine expert shard size: " + path;
        return false;
    }
    source = {nullptr, static_cast<size_t>(bytes), fd};
    return true;
}

bool is_exact_gfx1151(const cudaDeviceProp & properties) {
    return std::strncmp(properties.gcnArchName, "gfx1151", 7) == 0 &&
        (properties.gcnArchName[7] == '\0' ||
         properties.gcnArchName[7] == ':');
}

int largest_committed_snapshot_boundary(
        int base_pos,
        size_t token_count,
        int requested_pos,
        const KimiK3PrefillPolicy & policy) {
    if (base_pos < 0 || requested_pos < base_pos) return -1;
    int boundary = requested_pos == base_pos ? base_pos : -1;
    size_t offset = 0;
    int position = base_pos;
    while (offset < token_count) {
        const size_t width = policy.next_width(token_count - offset);
        if (width == 0 || width > token_count - offset ||
            width > static_cast<size_t>(std::numeric_limits<int>::max()) ||
            position > std::numeric_limits<int>::max() -
                static_cast<int>(width)) {
            return -1;
        }
        const int next = position + static_cast<int>(width);
        if (next > requested_pos) break;
        boundary = next;
        position = next;
        offset += width;
    }
    return boundary > 0 ? boundary : -1;
}

constexpr size_t kKimiK3SnapshotBudget = size_t{8} << 30;
constexpr size_t kKimiK3SnapshotAllocationMargin = size_t{64} << 20;

size_t snapshot_storage_bytes(const KimiK3PrefixSnapshot & snapshot) {
    const size_t buffer_bytes = snapshot.buf
        ? ggml_backend_buffer_get_size(snapshot.buf) : 0;
    const size_t logits_bytes = snapshot.final_logits.size() * sizeof(float);
    return buffer_bytes + logits_bytes;
}

size_t estimated_snapshot_admission_bytes(
        const KimiK3Cache & cache,
        int vocabulary) {
    size_t bytes = vocabulary > 0
        ? static_cast<size_t>(vocabulary) * sizeof(float) : 0;
    for (const KimiK3LayerCache & layer : cache.layers) {
        size_t layer_bytes = 0;
        if (layer.conv_state && layer.ssm_state) {
            layer_bytes = ggml_nbytes(layer.conv_state) +
                ggml_nbytes(layer.ssm_state);
        } else if (layer.mla_k && cache.cur_pos > 0) {
            const size_t rows = static_cast<size_t>(cache.cur_pos);
            if (layer.mla_k->nb[2] >
                std::numeric_limits<size_t>::max() / rows) {
                return std::numeric_limits<size_t>::max();
            }
            layer_bytes = layer.mla_k->nb[2] * rows;
        }
        if (bytes > std::numeric_limits<size_t>::max() - layer_bytes) {
            return std::numeric_limits<size_t>::max();
        }
        bytes += layer_bytes;
    }
    if (bytes > std::numeric_limits<size_t>::max() -
        kKimiK3SnapshotAllocationMargin) {
        return std::numeric_limits<size_t>::max();
    }
    return bytes + kKimiK3SnapshotAllocationMargin;
}

} // namespace

KimiK3Backend::KimiK3Backend(const KimiK3BackendConfig & cfg) : cfg_(cfg) {}

KimiK3Backend::~KimiK3Backend() {
    shutdown();
}

bool KimiK3Backend::resolve_prefill_policy(std::string & error) {
    int width = 1;
    if (!parse_prefill_width(
            std::getenv("DFLASH_KIMI_PREFILL_CHUNK"), width)) {
        error = "DFLASH_KIMI_PREFILL_CHUNK must be 1, 8, 64, or 1024";
        return false;
    }

    bool exact_multirow = false;
    if (!parse_binary_env(
            "DFLASH_KIMI_P58_EXACT_MULTIROW", exact_multirow, error)) {
        return false;
    }
    prefill_policy_ = {width, exact_multirow};
    if (!prefill_policy_.valid()) {
        error = width == 1
            ? "DFLASH_KIMI_P58_EXACT_MULTIROW requires macro width 8, 64, or 1024"
            : "Kimi-K3 macro prefill requires DFLASH_KIMI_P58_EXACT_MULTIROW=1";
        return false;
    }
    return true;
}

bool KimiK3Backend::init_streaming(std::string & error) {
    if (!weights_.routed_experts_streamed ||
        weights_.streamed_layer_regions.empty() ||
        weights_.max_streamed_expert_bytes == 0 ||
        weights_.shard_paths.empty()) {
        error = "routed expert streaming metadata is incomplete";
        return false;
    }

    MoeStreamConfig stream_config = MoeStreamConfig::from_env();
    if (!stream_engine_.init(
            backend_, weights_.max_streamed_expert_bytes,
            stream_config, &error)) {
        return false;
    }

    ScopedDescriptors descriptors;
    std::vector<MoeNvmeSource> sources;
    sources.reserve(weights_.shard_paths.size());
    for (const std::string & path : weights_.shard_paths) {
        MoeNvmeSource source;
        if (!open_stream_source(path, descriptors, source, error)) {
            stream_engine_.destroy();
            return false;
        }
        sources.push_back(source);
    }
    if (!stream_engine_.bind_sources(
            sources, weights_.streamed_layer_regions, &error)) {
        stream_engine_.destroy();
        return false;
    }

    if (!create_kimi_k3_calibrated_provider_from_env(
            backend_, backend_, routed_output_provider_, &error)) {
        stream_engine_.destroy();
        return false;
    }
    return true;
}

bool KimiK3Backend::init() {
    if (initialized_) return true;
    if (!cfg_.model_path || !*cfg_.model_path) {
        std::fprintf(stderr, "[kimi-k3] model path is empty\n");
        return false;
    }
    if (cfg_.device.is_multi_device()) {
        std::fprintf(stderr,
            "[kimi-k3] production backend requires one HIP device\n");
        return false;
    }
    const PlacementBackend requested = cfg_.device.backend;
    if (compiled_placement_backend() != PlacementBackend::Hip ||
        (requested != PlacementBackend::Auto &&
         requested != PlacementBackend::Hip)) {
        std::fprintf(stderr,
            "[kimi-k3] production backend requires a HIP/ROCm build and placement\n");
        return false;
    }

    std::string error;
    if (!apply_kimi_k3_production_defaults(error)) {
        std::fprintf(stderr, "[kimi-k3] %s\n", error.c_str());
        return false;
    }
    if (!resolve_prefill_policy(error)) {
        std::fprintf(stderr, "[kimi-k3] %s\n", error.c_str());
        return false;
    }

    backend_ = ggml_backend_cuda_init(cfg_.device.primary_gpu());
    if (!backend_) {
        std::fprintf(stderr,
            "[kimi-k3] HIP backend initialization failed for device %d\n",
            cfg_.device.primary_gpu());
        return false;
    }
    cudaDeviceProp properties{};
    if (cudaGetDeviceProperties(
            &properties, cfg_.device.primary_gpu()) != cudaSuccess ||
        !is_exact_gfx1151(properties)) {
        std::fprintf(stderr,
            "[kimi-k3] production backend requires gfx1151; logical "
            "device %d is %s\n",
            cfg_.device.primary_gpu(),
            properties.gcnArchName[0] ? properties.gcnArchName : "unknown");
        shutdown();
        return false;
    }
    snapshot_backend_ = create_snapshot_backend(backend_);
    if (!snapshot_backend_) {
        std::fprintf(stderr,
            "[kimi-k3] prefix snapshot backend initialization failed\n");
        shutdown();
        return false;
    }

    KimiK3LoadOptions load_options;
    load_options.stream_routed_experts = true;
    if (!load_kimi_k3_gguf(
            cfg_.model_path, backend_, weights_, load_options)) {
        std::fprintf(stderr, "[kimi-k3] model load failed: %s\n",
                     dflash27b_last_error());
        shutdown();
        return false;
    }

    const int max_ctx = std::max(1, cfg_.device.max_ctx);
    const int replay_width = prefill_policy_.exact_multirow
        ? prefill_policy_.macro_width : 0;
    if (!create_kimi_k3_cache(
            backend_, weights_, max_ctx, cache_, replay_width)) {
        std::fprintf(stderr,
            "[kimi-k3] cache allocation failed (max_ctx=%d)\n", max_ctx);
        shutdown();
        return false;
    }
    if (!init_streaming(error)) {
        std::fprintf(stderr,
            "[kimi-k3] routed stream initialization failed: %s\n",
            error.c_str());
        shutdown();
        return false;
    }

    if (prefill_policy_.exact_multirow) {
        KimiK3RoutedPrefillService * service = routed_output_provider_
            ? routed_output_provider_->prefill_service() : nullptr;
        if (!service || !service->supports_width(
                static_cast<size_t>(prefill_policy_.macro_width))) {
            std::fprintf(stderr,
                "[kimi-k3] macro prefill requires the authoritative "
                "all-layer calibrated96 service at width %d\n",
                prefill_policy_.macro_width);
            shutdown();
            return false;
        }
    }

    initialized_ = true;
    std::fprintf(stderr,
        "[kimi-k3] production backend ready topology=single-owner "
        "core=hip:%d experts=hip:%d join=hip:%d io=%s "
        "cache=%.2fGiB prefill-width=%d exact-macro=%d provider=%s\n",
        cfg_.device.primary_gpu(), cfg_.device.primary_gpu(),
        cfg_.device.primary_gpu(), stream_engine_.io_backend_name(),
        static_cast<double>(stream_engine_.device_cache_bytes()) /
            (1024.0 * 1024.0 * 1024.0),
        prefill_policy_.macro_width,
        prefill_policy_.exact_multirow ? 1 : 0,
        routed_output_provider_ ? "calibrated96" : "native-exact");
    std::fflush(stderr);
    return true;
}

void KimiK3Backend::print_ready_banner() const {
    std::printf(
        "[kimi-k3-daemon] ready (layers=%d hidden=%d experts=%d "
        "vocab=%d max_ctx=%d device=hip:%d single-owner=1)\n",
        weights_.n_layer, weights_.n_embd, weights_.n_expert,
        weights_.n_vocab, cache_.max_ctx, cfg_.device.primary_gpu());
    std::fflush(stdout);
}

bool KimiK3Backend::park(ParkTarget target) {
    (void) target;
    // The persistent graph state retains weight references. Partial parking
    // would make those references stale, so this backend fails closed.
    return false;
}

bool KimiK3Backend::unpark(ParkTarget target) {
    (void) target;
    return false;
}

int32_t KimiK3Backend::choose_token(
        const std::vector<float> & logits,
        const SamplerCfg & sampler,
        bool do_sample,
        const std::vector<int32_t> & history) {
    if (do_sample) {
        return sample_logits(
            logits.data(), weights_.n_vocab, sampler, history, rng_);
    }
    return static_cast<int32_t>(std::distance(
        logits.begin(), std::max_element(logits.begin(), logits.end())));
}

GenerateResult KimiK3Backend::generate_impl(
        const GenerateRequest & req, const DaemonIO & io) {
    return generate_from_state(req, io, nullptr);
}

bool KimiK3Backend::capture_last_logits(std::string & error) const {
    const char * path = std::getenv("DFLASH_KIMI_LOGITS_OUT");
    if (!path || !*path) return true;
    if (last_logits_.empty()) {
        error = "DFLASH_KIMI_LOGITS_OUT requested without a logits row";
        return false;
    }
    std::ofstream output(path, std::ios::binary | std::ios::trunc);
    if (!output) {
        error = std::string("cannot open Kimi-K3 logits output: ") + path;
        return false;
    }
    const size_t bytes = last_logits_.size() * sizeof(float);
    output.write(reinterpret_cast<const char *>(last_logits_.data()),
                 static_cast<std::streamsize>(bytes));
    output.close();
    if (!output) {
        error = std::string("cannot write Kimi-K3 logits output: ") + path;
        return false;
    }
    std::fprintf(stderr,
        "[kimi-k3] logits-capture path=%s position=%d values=%zu bytes=%zu\n",
        path, last_logits_pos_, last_logits_.size(), bytes);
    return true;
}

GenerateResult KimiK3Backend::generate_from_state(
        const GenerateRequest & req,
        const DaemonIO & io,
        const KimiK3PrefixSnapshot * snapshot) {
    GenerateResult result;
    DaemonIO out_io = io.with_token_callback(req.on_token);
    const auto fail = [&](GenerateErrorCode code, std::string detail) {
        result.fail(code, std::move(detail));
        out_io.emit(-1);
        return result;
    };

    if (!initialized_) {
        return fail(GenerateErrorCode::ModelParked,
                    "Kimi-K3 backend is not initialized");
    }
    if (req.prompt.empty()) {
        return fail(GenerateErrorCode::PrefillFailed, "empty prompt");
    }
    const size_t context_capacity = static_cast<size_t>(cache_.max_ctx);
    if (req.n_gen < 0 || req.prompt.size() > context_capacity ||
        static_cast<size_t>(std::max(0, req.n_gen)) >
            context_capacity - req.prompt.size()) {
        return fail(GenerateErrorCode::ContextOverflow,
                    "prompt plus generation exceeds Kimi-K3 cache");
    }
    if (req.do_sample && req.sampler.seed != 0) rng_.seed(req.sampler.seed);

    const auto prefill_begin = std::chrono::steady_clock::now();
    int restored_prefix = 0;
    std::vector<float> logits;
    if (snapshot && static_cast<int>(req.prompt.size()) >= snapshot->cur_pos) {
        if (!restore_kimi_k3_prefix_snapshot(*snapshot, cache_)) {
            return fail(GenerateErrorCode::InvalidSnapshotSlot,
                        dflash27b_last_error());
        }
        restored_prefix = snapshot->cur_pos;
        logits = snapshot->final_logits;
        last_logits_ = logits;
        last_logits_pos_ = cache_.cur_pos;
    } else {
        if (snapshot) {
            std::fprintf(stderr,
                "[kimi-k3-snap] snapshot longer than prompt "
                "(snap=%d prompt=%zu); using cold prefill\n",
                snapshot->cur_pos, req.prompt.size());
        }
        reset_kimi_k3_cache(cache_);
        last_logits_.clear();
        last_logits_pos_ = -1;
    }
    result.restored_prefix_tokens = restored_prefix;

    const std::vector<int32_t> delta(
        req.prompt.begin() + restored_prefix, req.prompt.end());
    const int capture_boundary = req.snap_slot >= 0 &&
        req.snap_slot < ModelBackend::kMaxSlots
        ? largest_committed_snapshot_boundary(
            restored_prefix, delta.size(), req.snap_pos, prefill_policy_)
        : -1;
    bool snapshot_attempted = false;
    const auto maybe_capture = [&](const std::vector<float> & rows) {
        if (snapshot_attempted || cache_.cur_pos != capture_boundary) return;
        const size_t vocabulary = static_cast<size_t>(weights_.n_vocab);
        if (vocabulary == 0 || rows.size() < vocabulary ||
            rows.size() % vocabulary != 0) {
            return;
        }
        last_logits_.assign(rows.end() -
            static_cast<std::vector<float>::difference_type>(vocabulary),
            rows.end());
        last_logits_pos_ = cache_.cur_pos;
        snapshot_attempted = true;
        if (!snapshot_save(req.snap_slot)) {
            std::fprintf(stderr,
                "[kimi-k3-snap] capture failed slot=%d pos=%d: %s\n",
                req.snap_slot, cache_.cur_pos, dflash27b_last_error());
        }
    };

    // A restored checkpoint can itself be the largest safe boundary. Copy it
    // before executing a later macro rather than changing macro width to reach
    // an arbitrary requested cut.
    if (capture_boundary == restored_prefix && restored_prefix > 0) {
        maybe_capture(logits);
    }

    const auto forward_token = [&](int32_t token, int position) {
        if (out_io.is_cancelled()) return false;
        return kimi_k3_step(
            backend_, weights_, cache_, token, position, logits,
            &stream_engine_, routed_output_provider_.get());
    };
    KimiK3PrefillContext prefill_context{
        backend_, weights_, cache_, stream_engine_,
        routed_output_provider_.get()};
    KimiK3PrefillExecutionResult prefill_execution;
    std::string prefill_error;
    const KimiK3PrefillExecutor prefill_executor(prefill_context);
    if (!delta.empty()) {
        if (!prefill_executor.run(
                delta, prefill_policy_, forward_token, maybe_capture, []() {},
                [&out_io]() { return out_io.is_cancelled(); }, logits,
                prefill_execution, &prefill_error)) {
            return fail(
                GenerateErrorCode::PrefillFailed,
                !prefill_error.empty()
                    ? prefill_error : dflash27b_last_error());
        }
        if (logits.size() == static_cast<size_t>(weights_.n_vocab)) {
            last_logits_ = logits;
            last_logits_pos_ = cache_.cur_pos;
        }
    }
    const auto prefill_end = std::chrono::steady_clock::now();
    result.prefill_s =
        std::chrono::duration<double>(prefill_end - prefill_begin).count();

    if (req.n_gen == 0 || prefill_execution.cancelled ||
        out_io.is_cancelled()) {
        std::string capture_error;
        if (!capture_last_logits(capture_error)) {
            return fail(GenerateErrorCode::PrefillFailed,
                        std::move(capture_error));
        }
        out_io.emit(-1);
        result.succeed();
        return result;
    }
    if (logits.size() != static_cast<size_t>(weights_.n_vocab)) {
        return fail(GenerateErrorCode::DecodeSeedMissing,
                    "prefill did not produce one final logits row");
    }

    const auto decode_begin = std::chrono::steady_clock::now();
    std::vector<int32_t> history;
    history.reserve(req.prompt.size() + static_cast<size_t>(req.n_gen));
    history.insert(history.end(), req.prompt.begin(), req.prompt.end());
    bool budget_close_started = false;
    size_t close_inject_pos = 0;
    for (int i = 0; i < req.n_gen; ++i) {
        int32_t next = choose_token(
            logits, req.sampler, req.do_sample, history);

        const auto & close_ids = req.budget_hook.close_token_ids;
        if (!close_ids.empty()) {
            if (budget_close_started && close_inject_pos < close_ids.size()) {
                next = close_ids[close_inject_pos++];
                result.budget_forced_close = true;
            } else if (!budget_close_started &&
                       req.n_gen - i <=
                           req.budget_hook.hard_limit_remaining) {
                budget_close_started = true;
                if (next != close_ids.front()) {
                    next = close_ids.front();
                    result.budget_forced_close = true;
                }
                close_inject_pos = 1;
            }
        }

        result.tokens.push_back(next);
        history.push_back(next);
        out_io.emit(next);
        if (out_io.is_cancelled() || next == weights_.eos_token_id) break;
        if (i + 1 < req.n_gen && !kimi_k3_step(
                backend_, weights_, cache_, next, cache_.cur_pos, logits,
                &stream_engine_, routed_output_provider_.get())) {
            return fail(GenerateErrorCode::DecodeFailed,
                        dflash27b_last_error());
        }
        if (i + 1 < req.n_gen &&
            logits.size() != static_cast<size_t>(weights_.n_vocab)) {
            return fail(GenerateErrorCode::DecodeFailed,
                        "decode did not produce one logits row");
        }
        if (i + 1 < req.n_gen) {
            last_logits_ = logits;
            last_logits_pos_ = cache_.cur_pos;
        }
    }
    const auto decode_end = std::chrono::steady_clock::now();
    result.decode_s =
        std::chrono::duration<double>(decode_end - decode_begin).count();
    std::string capture_error;
    if (!capture_last_logits(capture_error)) {
        return fail(GenerateErrorCode::DecodeFailed,
                    std::move(capture_error));
    }
    out_io.emit(-1);
    result.succeed();
    return result;
}

bool KimiK3Backend::snapshot_save(int slot) {
    if (slot < 0 || slot >= ModelBackend::kMaxSlots ||
        !snapshot_backend_ || cache_.cur_pos <= 0 ||
        last_logits_pos_ != cache_.cur_pos ||
        last_logits_.size() != static_cast<size_t>(weights_.n_vocab)) {
        set_last_error("Kimi-K3 prefix snapshot has no committed logits row");
        return false;
    }
    KimiK3PrefixSnapshot & snapshot =
        prefix_snapshots_[static_cast<size_t>(slot)];
    const size_t old_bytes = snapshot_storage_bytes(snapshot);
    const size_t admission_bytes = estimated_snapshot_admission_bytes(
        cache_, weights_.n_vocab);
    const size_t retained_bytes = snapshot_bytes_ >= old_bytes
        ? snapshot_bytes_ - old_bytes : 0;
    if (admission_bytes > kKimiK3SnapshotBudget ||
        retained_bytes > kKimiK3SnapshotBudget - admission_bytes) {
        set_last_error("Kimi-K3 in-memory prefix snapshot budget exhausted");
        std::fprintf(stderr,
            "[kimi-k3-snap] admission rejected slot=%d retained=%zu "
            "admission=%zu budget=%zu\n",
            slot, retained_bytes, admission_bytes, kKimiK3SnapshotBudget);
        return false;
    }
    const bool saved = save_kimi_k3_prefix_snapshot(
        weights_, cache_, snapshot_backend_, last_logits_,
        snapshot);
    if (!saved) {
        free_kimi_k3_prefix_snapshot(snapshot);
        snapshot_bytes_ = retained_bytes;
        return false;
    }
    const size_t slot_bytes = snapshot_storage_bytes(snapshot);
    if (slot_bytes > kKimiK3SnapshotBudget ||
        retained_bytes > kKimiK3SnapshotBudget - slot_bytes) {
        free_kimi_k3_prefix_snapshot(snapshot);
        snapshot_bytes_ = retained_bytes;
        set_last_error(
            "Kimi-K3 prefix snapshot exceeded its in-memory budget");
        return false;
    }
    snapshot_bytes_ = retained_bytes + slot_bytes;
    std::fprintf(stderr,
        "[kimi-k3-snap] saved slot=%d pos=%d slot_bytes=%zu "
        "aggregate_bytes=%zu budget=%zu\n",
        slot, cache_.cur_pos, slot_bytes, snapshot_bytes_,
        kKimiK3SnapshotBudget);
    return true;
}

void KimiK3Backend::snapshot_free(int slot) {
    if (slot < 0 || slot >= ModelBackend::kMaxSlots) return;
    KimiK3PrefixSnapshot & snapshot =
        prefix_snapshots_[static_cast<size_t>(slot)];
    const size_t bytes = snapshot_storage_bytes(snapshot);
    free_kimi_k3_prefix_snapshot(snapshot);
    snapshot_bytes_ = snapshot_bytes_ >= bytes
        ? snapshot_bytes_ - bytes : 0;
}

bool KimiK3Backend::snapshot_used(int slot) const {
    if (slot < 0 || slot >= ModelBackend::kMaxSlots) return false;
    const KimiK3PrefixSnapshot & snapshot =
        prefix_snapshots_[static_cast<size_t>(slot)];
    return snapshot.ctx && snapshot.buf && snapshot.cur_pos > 0 &&
        snapshot.final_logits.size() ==
            static_cast<size_t>(weights_.n_vocab);
}

int KimiK3Backend::snapshot_cur_pos(int slot) const {
    return snapshot_used(slot)
        ? prefix_snapshots_[static_cast<size_t>(slot)].cur_pos : 0;
}

GenerateResult KimiK3Backend::restore_and_generate_impl(
        int slot, const GenerateRequest & req, const DaemonIO & io) {
    if (!snapshot_used(slot)) {
        GenerateResult result;
        result.fail(GenerateErrorCode::InvalidSnapshotSlot,
                    "Kimi-K3 prefix snapshot slot is empty");
        io.emit(-1);
        return result;
    }
    return generate_from_state(
        req, io, &prefix_snapshots_[static_cast<size_t>(slot)]);
}

bool KimiK3Backend::handle_compress(
        const std::string & line, const DaemonIO & io) {
    (void) line;
    (void) io;
    return false;
}

void KimiK3Backend::shutdown() {
    routed_output_provider_.reset();
    stream_engine_.destroy();
    for (KimiK3PrefixSnapshot & snapshot : prefix_snapshots_) {
        free_kimi_k3_prefix_snapshot(snapshot);
    }
    snapshot_bytes_ = 0;
    last_logits_.clear();
    last_logits_pos_ = -1;
    free_kimi_k3_cache(cache_);
    free_kimi_k3_weights(weights_);
    free_snapshot_backend(snapshot_backend_, backend_);
    snapshot_backend_ = nullptr;
    if (backend_) {
        ggml_backend_free(backend_);
        backend_ = nullptr;
    }
    initialized_ = false;
}

} // namespace dflash::common
