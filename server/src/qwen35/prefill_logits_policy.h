#pragma once

#include <cstddef>

namespace dflash::common {

struct Qwen35PrefillLogitsPolicy {
    bool last_token_logits_only = true;
    std::size_t argmax_offset_bytes = 0;
    std::size_t logits_offset_bytes = 0;
};

inline Qwen35PrefillLogitsPolicy qwen35_prefill_logits_policy(
        int n_tokens,
        int vocab) {
    (void)n_tokens;
    (void)vocab;
    // Prefill only consumes the final token logits/argmax for continuation.
    // Keeping every chunk in last-token-only projection avoids a full
    // [vocab, n_tokens] LM-head output on the final chunk.
    return Qwen35PrefillLogitsPolicy{};
}

}  // namespace dflash::common
