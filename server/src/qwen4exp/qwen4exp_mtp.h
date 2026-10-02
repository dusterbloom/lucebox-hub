#pragma once

#include "common/adaptive_spec_width.h"

#include <algorithm>
#include <array>
#include <cassert>
#include <cstdint>
#include <cstdlib>

namespace luce::common {

constexpr int QWEN4EXP_MTP_MAX_DRAFT = 4;
constexpr int QWEN4EXP_MTP_MAX_VERIFY = QWEN4EXP_MTP_MAX_DRAFT + 1;

// Malformed values use the default; numeric values (including overflow) clamp.
inline int qwen4exp_mtp_draft_length(const char * value) {
    if (!value || !*value) return 1;
    char * end = nullptr;
    const long n = std::strtol(value, &end, 10);
    if (end == value || *end) return 1;
    return (int) std::clamp(n, 1L, (long) QWEN4EXP_MTP_MAX_DRAFT);
}

// Server --verify-width: 0 = adaptive k=1..3, 1 = off, 2..5 = fixed k=1..4.
// Keep the old environment override for existing fixed-width A/B runs.
inline int qwen4exp_mtp_verify_width(int configured, const char * legacy_draft) {
    return configured == 0 && legacy_draft && *legacy_draft
        ? qwen4exp_mtp_draft_length(legacy_draft) + 1 : configured;
}

inline AdaptiveSpecWidth qwen4exp_mtp_width_policy(int max_draft, bool adaptive) {
    AdaptiveSpecWidth policy(max_draft + 1, 2, adaptive);
    // Total draft + verify + rollback ms, indexed by seed-inclusive width.
    // gfx1151 UD-Q4_K_XL: clean k=1/2/3 runs commit 2/3/4 tokens at
    // 31.3/36.6/41.4 tok/s. The shared controller refines costs and prefix
    // survival online, ignores cold cost samples, and can re-widen after rejects.
    policy.set_relative_costs({0.0f, 0.0f, 64.0f, 82.0f, 97.0f});
    return policy;
}

struct Qwen4ExpMtpAcceptance {
    int n_accepted = 0;
    int n_emitted = 0;
    std::array<int32_t, QWEN4EXP_MTP_MAX_VERIFY> emitted{};
};

// Samples are the trunk's own draws, in order, after any budget substitution.
// A partial sample prefix allows callers to stop at EOS without drawing future
// tokens or advancing RNG/history. Retain n_emitted verify input positions:
// the last emitted token is the next (as yet unprocessed) input, as in AR decode.
inline Qwen4ExpMtpAcceptance qwen4exp_mtp_accept(
        const int32_t * drafts, int k, const int32_t * samples, int n_samples) {
    assert(k >= 0 && k <= QWEN4EXP_MTP_MAX_DRAFT);
    assert(n_samples >= 0 && n_samples <= k + 1);
    Qwen4ExpMtpAcceptance result;
    for (int i = 0; i < n_samples; ++i) {
        result.emitted[result.n_emitted++] = samples[i];
        if (i == k || samples[i] != drafts[i]) break;
        ++result.n_accepted;
    }
    return result;
}

// A speculative forward can complete a pooled block whose suffix is rejected.
// Only wholly retained blocks remain authoritative; the next completion must
// recompute the invalidated row using replacement raw keys.
inline int qwen4exp_mtp_retained_blocks(int pooled, int retained_pos, int ratio) {
    return ratio > 1 ? std::min(pooled, retained_pos / ratio) : pooled;
}

}  // namespace luce::common
