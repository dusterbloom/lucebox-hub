#include "kimi_k3_prefill.h"

#include <limits>

namespace dflash::common {
namespace {

class ReplayRollbackGuard {
public:
    ReplayRollbackGuard(ggml_backend_t backend, KimiK3Cache & cache)
        : backend_(backend), cache_(cache) {}

    ReplayRollbackGuard(const ReplayRollbackGuard &) = delete;
    ReplayRollbackGuard & operator=(const ReplayRollbackGuard &) = delete;

    ~ReplayRollbackGuard() {
        if (armed_) (void) kimi_k3_replay_restore(backend_, cache_);
    }

    void disarm() { armed_ = false; }

private:
    ggml_backend_t backend_;
    KimiK3Cache & cache_;
    bool armed_ = true;
};

} // namespace

bool KimiK3PrefillExecutor::run(
        const std::vector<int32_t> & prompt,
        const KimiK3PrefillPolicy & policy,
        const ForwardToken & forward_token,
        const LogitsSink & logits_sink,
        const MacroComplete & macro_complete,
        const CancellationProbe & is_cancelled,
        std::vector<float> & last_logits,
        KimiK3PrefillExecutionResult & result,
        std::string * error) const {
    result = {};
    const auto fail = [&](const char * message) {
        if (error) *error = message;
        return false;
    };

    if (!policy.valid()) {
        return fail("Kimi-K3 prefill policy is outside the exact envelope");
    }
    if (!context_.backend || !forward_token || !logits_sink ||
        !macro_complete) {
        return fail("Kimi-K3 prefill executor has incomplete dependencies");
    }
    if (context_.weights.n_vocab <= 0) {
        return fail("Kimi-K3 prefill executor has an invalid vocabulary");
    }

    const size_t vocabulary =
        static_cast<size_t>(context_.weights.n_vocab);
    for (size_t offset = 0; offset < prompt.size();) {
        if (is_cancelled && is_cancelled()) {
            result.cancelled = true;
            return true;
        }
        const size_t width = policy.next_width(prompt.size() - offset);
        if (width == 0 || offset > prompt.size() - width) {
            return fail("Kimi-K3 prefill policy produced an invalid span");
        }
        if (context_.cache.cur_pos < 0) {
            return fail("Kimi-K3 prefill cache position is invalid");
        }
        const int base_pos = context_.cache.cur_pos;
        if (width > static_cast<size_t>(std::numeric_limits<int>::max()) ||
            base_pos > std::numeric_limits<int>::max() -
                static_cast<int>(width)) {
            return fail("Kimi-K3 prefill position overflows");
        }

        if (width == 1) {
            if (!forward_token(prompt[offset], base_pos)) {
                if (is_cancelled && is_cancelled()) {
                    result.cancelled = true;
                    return true;
                }
                return false;
            }
            if (context_.cache.cur_pos != base_pos + 1) {
                return fail("Kimi-K3 one-row prefill did not advance state");
            }
            logits_sink(last_logits);
        } else {
            if (!context_.routed_output_provider ||
                !kimi_k3_exact_multirow_width(width)) {
                return fail("Kimi-K3 macro prefill has no exact provider");
            }
            if (vocabulary > std::numeric_limits<size_t>::max() / width) {
                return fail("Kimi-K3 macro prefill shape overflows");
            }
            if (!kimi_k3_replay_snapshot(context_.backend, context_.cache)) {
                return fail("Kimi-K3 macro prefill cannot snapshot state");
            }
            ReplayRollbackGuard rollback(context_.backend, context_.cache);

            const auto begin = prompt.begin() +
                static_cast<std::vector<int32_t>::difference_type>(offset);
            const std::vector<int32_t> tokens(
                begin,
                begin + static_cast<std::vector<int32_t>::difference_type>(
                    width));

            KimiK3ForwardOptions options;
            options.capture_replay = true;
            options.read_logits = true;
            options.read_argmax = false;
            options.routed_output_provider = context_.routed_output_provider;
            options.exact_multirow_core = true;

            KimiK3ForwardResult forward_result;
            if (!kimi_k3_forward(
                    context_.backend, context_.weights, context_.cache,
                    tokens, base_pos, options, forward_result,
                    &context_.stream_engine)) {
                if (is_cancelled && is_cancelled()) {
                    result.cancelled = true;
                    return true;
                }
                return false;
            }
            if (is_cancelled && is_cancelled()) {
                result.cancelled = true;
                return true;
            }
            if (forward_result.logits.size() != width * vocabulary) {
                return fail("Kimi-K3 macro prefill returned invalid logits");
            }
            if (!kimi_k3_replay_commit(
                    context_.backend, context_.weights, context_.cache,
                    base_pos, static_cast<int>(width))) {
                return fail("Kimi-K3 macro prefill cannot commit state");
            }
            rollback.disarm();

            macro_complete();
            logits_sink(forward_result.logits);
            last_logits.assign(
                forward_result.logits.end() -
                    static_cast<std::vector<float>::difference_type>(
                        vocabulary),
                forward_result.logits.end());
        }

        ++result.forward_calls;
        offset += width;
    }
    return true;
}

} // namespace dflash::common
