#pragma once

#include "common/adaptive_spec_width.h"

#include <algorithm>
#include <array>
#include <cassert>
#include <cstdint>

namespace luce::common {

constexpr int QWEN4EXP_MTP_MAX_DRAFT = 4;
constexpr int QWEN4EXP_MTP_MAX_VERIFY = QWEN4EXP_MTP_MAX_DRAFT + 1;

// Server --verify-width: 0 = adaptive k=1..3, 1 = off, 2..5 = fixed k=1..4.
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

// ne[] dimensions of the six MTP sidecar tensors that mtp_forward_batch() (qwen4exp_graph.cpp) feeds
// straight into matmuls/reshapes with no further validation: a mismatch here is a crash or silent garbage
// at inference time, not a load-time error, unless caught first.
struct Qwen4ExpMtpShapeDims {
    int64_t eh_proj_ne0 = 0, eh_proj_ne1 = 0;      // [2H, H]: mm(eh_proj, [2H, hc*T]) reshaped to [H, hc, T]
    int64_t enorm_ne0 = 0;                          // [H]: elementwise with rms_norm(inp_emb) which is [H, T]
    int64_t hnorm_ne0 = 0;                          // [H*hc]: elementwise with rms_norm(inp_h) which is [H*hc, T]
    int64_t head_norm_ne0 = 0;                      // [H*hc]: hc_mix's rms_norm gamma over the draft head
    int64_t head_down_ne0 = 0, head_down_ne1 = 0;   // [H*hc, hc_lr]: low-rank down-projection
    int64_t head_up_ne0 = 0, head_up_ne1 = 0;       // [hc_lr, H*hc]: low-rank up-projection
};

// Pure: true iff every MTP tensor shape the forward graph relies on is internally consistent with the
// trunk's embedding/hyper-connection config. Takes plain ne[] values (not ggml_tensor*) so it is
// unit-testable with synthetic good/bad shapes, without a GGUF file or a GPU.
inline bool qwen4exp_mtp_shapes_valid(const Qwen4ExpMtpShapeDims & t,
                                      int64_t n_embd, int64_t n_hc, int64_t hc_lowrank) {
    const int64_t hc_dim = n_hc * n_embd;
    return t.eh_proj_ne0 == 2 * n_embd && t.eh_proj_ne1 == n_embd &&
           t.enorm_ne0 == n_embd &&
           t.hnorm_ne0 == hc_dim &&
           t.head_norm_ne0 == hc_dim &&
           t.head_down_ne0 == hc_dim && t.head_down_ne1 == hc_lowrank &&
           t.head_up_ne0 == hc_lowrank && t.head_up_ne1 == hc_dim;
}

}  // namespace luce::common
