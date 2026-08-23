#include "kimi_k3_prefill.h"

#include "kimi_k3_progressive_provider.h"

#include <cstddef>
#include <utility>

namespace dflash::common {

bool KimiK3PrefillExecutor::run(
        const std::vector<int32_t> & prompt,
        const KimiK3PrefillPolicy & policy,
        const ForwardToken & forward_token,
        const LogitsSink & logits_sink,
        const MacroComplete & macro_complete,
        std::vector<float> & last_logits,
        KimiK3PrefillExecutionResult & result,
        std::string * error) const {
    result = KimiK3PrefillExecutionResult{};
    const auto fail = [&](const char * message) {
        if (error) *error = message;
        return false;
    };

    for (size_t i = 0; i < prompt.size();) {
        const size_t width = policy.next_width(prompt.size() - i);
        bool ok = false;
        if (width == 1) {
            ok = forward_token(prompt[i], static_cast<int>(i));
            if (ok) logits_sink(last_logits);
        } else {
            const bool snapshot_ok =
                kimi_k3_replay_snapshot(context_.backend, context_.cache);
            if (!snapshot_ok) {
                return fail("chunked prefill cannot snapshot recurrent state");
            }

            KimiK3ForwardOptions options;
            options.read_logits = true;
            options.read_argmax = false;
            options.capture_replay = true;
            options.exact_multirow_core =
                policy.exact_multirow && kimi_k3_exact_multirow_width(width);
            options.routed_output_provider = context_.routed_output_provider;
            options.moe_core_offload = context_.moe_core_offload;

            KimiK3ForwardResult forward_result;
            const std::vector<int32_t> tokens(
                prompt.begin() + static_cast<std::ptrdiff_t>(i),
                prompt.begin() + static_cast<std::ptrdiff_t>(i + width));
            ok = kimi_k3_forward(
                context_.backend, context_.weights, context_.cache,
                tokens, static_cast<int>(i), options, forward_result,
                &context_.stream_engine, context_.dual_stream_executor,
                context_.stream_owner_policy, context_.routing_stats);
            if (ok) {
                const size_t vocabulary =
                    static_cast<size_t>(context_.weights.n_vocab);
                if (forward_result.logits.size() != width * vocabulary) {
                    ok = false;
                    if (error) {
                        *error = "chunked prefill returned an invalid logits shape";
                    }
                } else if (!kimi_k3_replay_commit(
                               context_.backend, context_.weights,
                               context_.cache, static_cast<int>(i),
                               static_cast<int>(width))) {
                    ok = false;
                    if (error) {
                        *error = "chunked prefill cannot commit recurrent state";
                    }
                } else {
                    macro_complete();
                    logits_sink(forward_result.logits);
                    last_logits.assign(
                        forward_result.logits.end() - vocabulary,
                        forward_result.logits.end());
                }
            }
            if (!ok && context_.cache.snapshot_valid) {
                (void) kimi_k3_replay_restore(
                    context_.backend, context_.cache);
            }
        }

        if (!ok) return false;
        ++result.forward_calls;
        i += width;
    }
    return true;
}

} // namespace dflash::common
