#include "kimi_k3_backend.h"

#include "common/sampler.h"
#include "device_runtime.h"
#include "dflash27b.h"

// ggml retains the cuda-named accelerator API for both CUDA and HIP builds.
#include "ggml-cuda.h"

#include <algorithm>
#include <cerrno>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
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

    reset_kimi_k3_cache(cache_);
    std::vector<float> logits;
    const auto prefill_begin = std::chrono::steady_clock::now();
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
    if (!prefill_executor.run(
            req.prompt, prefill_policy_, forward_token,
            [](const std::vector<float> &) {}, []() {},
            [&out_io]() { return out_io.is_cancelled(); }, logits,
            prefill_execution, &prefill_error)) {
        return fail(
            GenerateErrorCode::PrefillFailed,
            !prefill_error.empty() ? prefill_error : dflash27b_last_error());
    }
    const auto prefill_end = std::chrono::steady_clock::now();
    result.prefill_s =
        std::chrono::duration<double>(prefill_end - prefill_begin).count();

    if (req.n_gen == 0 || prefill_execution.cancelled ||
        out_io.is_cancelled()) {
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
    }
    const auto decode_end = std::chrono::steady_clock::now();
    result.decode_s =
        std::chrono::duration<double>(decode_end - decode_begin).count();
    out_io.emit(-1);
    result.succeed();
    return result;
}

bool KimiK3Backend::snapshot_save(int slot) {
    (void) slot;
    return false;
}

void KimiK3Backend::snapshot_free(int slot) {
    (void) slot;
}

bool KimiK3Backend::snapshot_used(int slot) const {
    (void) slot;
    return false;
}

int KimiK3Backend::snapshot_cur_pos(int slot) const {
    (void) slot;
    return 0;
}

GenerateResult KimiK3Backend::restore_and_generate_impl(
        int slot, const GenerateRequest & req, const DaemonIO & io) {
    (void) slot;
    (void) req;
    GenerateResult result;
    result.fail(GenerateErrorCode::InvalidSnapshotSlot,
                "Kimi-K3 prefix snapshots are not supported");
    io.emit(-1);
    return result;
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
    free_kimi_k3_cache(cache_);
    free_kimi_k3_weights(weights_);
    if (backend_) {
        ggml_backend_free(backend_);
        backend_ = nullptr;
    }
    initialized_ = false;
}

} // namespace dflash::common
