#pragma once

#include "kimi_k3_internal.h"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <string>
#include <vector>

namespace dflash::common {

// Resolved once by the backend. Prompt execution does not consult process
// environment or infer a width from the remaining prompt.
struct KimiK3PrefillPolicy {
    int macro_width = 1;
    bool exact_multirow = false;

    bool valid() const {
        if (macro_width == 1) return !exact_multirow;
        return exact_multirow && macro_width > 0 &&
            kimi_k3_exact_multirow_width(static_cast<size_t>(macro_width));
    }

    size_t next_width(size_t remaining) const {
        if (remaining == 0) return 0;
        if (macro_width == 1 ||
            remaining < static_cast<size_t>(macro_width)) {
            return 1;
        }
        return static_cast<size_t>(macro_width);
    }
};

struct KimiK3PrefillExecutionResult {
    size_t forward_calls = 0;
    bool cancelled = false;
};

// Model, semantic state, stream engine, and provider remain owned by the
// backend. The prompt executor only sequences their exact causal seam.
struct KimiK3PrefillContext {
    ggml_backend_t backend = nullptr;
    const KimiK3Weights & weights;
    KimiK3Cache & cache;
    MoeHybridStreamEngine & stream_engine;
    KimiK3RoutedOutputProvider * routed_output_provider = nullptr;
};

class KimiK3PrefillExecutor {
public:
    using ForwardToken = std::function<bool(int32_t token, int position)>;
    using LogitsSink = std::function<void(const std::vector<float> &)>;
    using MacroComplete = std::function<void()>;
    using CancellationProbe = std::function<bool()>;

    explicit KimiK3PrefillExecutor(KimiK3PrefillContext context)
        : context_(context) {}

    bool run(const std::vector<int32_t> & prompt,
             const KimiK3PrefillPolicy & policy,
             const ForwardToken & forward_token,
             const LogitsSink & logits_sink,
             const MacroComplete & macro_complete,
             const CancellationProbe & is_cancelled,
             std::vector<float> & last_logits,
             KimiK3PrefillExecutionResult & result,
             std::string * error = nullptr) const;

private:
    KimiK3PrefillContext context_;
};

} // namespace dflash::common
