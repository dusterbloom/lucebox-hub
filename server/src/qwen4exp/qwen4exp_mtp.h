#pragma once

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
