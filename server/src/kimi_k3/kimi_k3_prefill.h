#pragma once

#include "kimi_k3_internal.h"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <string>
#include <vector>

namespace dflash::common {

// Prompt-phase policy is resolved once during backend initialization.  The
// executor consumes this value rather than consulting process environment in
// the hot path.
struct KimiK3PrefillPolicy {
    int macro_width = 1;
    bool exact_multirow = false;

    bool valid() const {
        return (macro_width == 8) == exact_multirow;
    }

    size_t next_width(size_t remaining) const {
        if (remaining == 0 || macro_width <= 1) {
            return std::min<size_t>(remaining, 1);
        }
        // P58 never exposes a partial macro to the calibrated provider.  A
        // short tail remains on the established one-row path.
        if (exact_multirow && macro_width == 8 && remaining < 8) return 1;
        return std::min(remaining, static_cast<size_t>(macro_width));
    }
};

inline bool parse_kimi_k3_prefill_chunk(const char * value, int & out) {
    out = 1;
    if (!value || !*value || std::string(value) == "1") return true;
    if (std::string(value) == "2") {
        out = 2;
        return true;
    }
    if (std::string(value) == "4") {
        out = 4;
        return true;
    }
    if (std::string(value) == "8") {
        out = 8;
        return true;
    }
    return false;
}

inline size_t kimi_k3_prefill_chunk_size(
        size_t remaining, int configured, bool exact_multirow) {
    return KimiK3PrefillPolicy{configured, exact_multirow}.next_width(remaining);
}

inline bool kimi_k3_p58_configuration_valid(
        int configured, bool exact_multirow) {
    return KimiK3PrefillPolicy{configured, exact_multirow}.valid();
}

inline bool kimi_k3_p58_oracle_candidate(
        bool exact_multirow, size_t width, bool capture_replay) {
    return exact_multirow && width == 8 && capture_replay;
}

struct KimiK3PrefillExecutionResult {
    size_t forward_calls = 0;
};

// All mutable execution dependencies are borrowed from KimiK3Backend.  This
// keeps the executor responsible for the prompt phase without creating a
// second owner for model, cache, stream, or provider lifetime.
struct KimiK3PrefillContext {
    ggml_backend_t backend;
    const KimiK3Weights & weights;
    KimiK3Cache & cache;
    MoeHybridStreamEngine & stream_engine;
    MoeStreamDualOwnerExecutor * dual_stream_executor;
    const MoeStreamDualOwnerPolicy * stream_owner_policy;
    MoeHybridRoutingStats * routing_stats;
    KimiK3RoutedOutputProvider * routed_output_provider;
    KimiK3MoeCoreOffload * moe_core_offload;
};

class KimiK3PrefillExecutor {
public:
    using ForwardToken = std::function<bool(int32_t token, int position)>;
    using LogitsSink = std::function<void(const std::vector<float> &)>;
    using MacroComplete = std::function<void()>;

    explicit KimiK3PrefillExecutor(KimiK3PrefillContext context)
        : context_(context) {}

    bool run(const std::vector<int32_t> & prompt,
             const KimiK3PrefillPolicy & policy,
             const ForwardToken & forward_token,
             const LogitsSink & logits_sink,
             const MacroComplete & macro_complete,
             std::vector<float> & last_logits,
             KimiK3PrefillExecutionResult & result,
             std::string * error = nullptr) const;

private:
    KimiK3PrefillContext context_;
};

} // namespace dflash::common
