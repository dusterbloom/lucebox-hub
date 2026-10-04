#pragma once

#include "common/adaptive_spec_width.h"

#include <algorithm>
#include <array>
#include <cassert>
#include <cstdint>

namespace luce::common {

constexpr int QWEN4EXP_MTP_MAX_DRAFT = 7;
constexpr int QWEN4EXP_MTP_MAX_VERIFY = QWEN4EXP_MTP_MAX_DRAFT + 1;

// Server --verify-width: 0 = adaptive k=1..7, 1 = off, 2..8 = fixed k=1..7.
// Eight verify rows is the RDNA3 batch-invariant MMVQ/MMID ceiling.
inline AdaptiveSpecWidth qwen4exp_mtp_width_policy(int max_draft, bool adaptive, int prompt_tokens = 0) {
    AdaptiveSpecWidth policy(max_draft + 1, 2, adaptive);
    // Session86: 45.9/47.6 ms AR at 16K/64K, 115.96 ms per counting
    // cycle (3.931 / 33.9). Subtract ~9 ms for subset head + last-row
    // catch-up + host argmax: R4 prior ~107 ms. Other widths extrapolate
    // ~20 ms/draft+verify row. Priors, not new measurements; observe refits.
    const float seed = prompt_tokens >= 32768 ? 47.0f : 45.0f;
    std::vector<float> costs(QWEN4EXP_MTP_MAX_VERIFY + 1);
    for (int width = 2; width <= QWEN4EXP_MTP_MAX_VERIFY; ++width) {
        costs[width] = seed + 20.0f * (width - 1);
    }
    policy.set_relative_costs(costs);
    return policy;
}

// Probe beyond the initial two drafts only as clean acceptance supports it.
// Fixed widths bypass both feedback and cost adaptation.
inline int qwen4exp_mtp_next_width(const AdaptiveSpecWidth & policy) {
    return policy.next_width_cost_aware({}, policy.next_width());
}

// Fixed production choice; smoke/session controls can measure other budgets.
constexpr int QWEN4EXP_MTP_VOCAB = 64000;

// Low BPE IDs are a rank heuristic, not a corpus-frequency guarantee. Reserve
// every tokenizer control first, then fill with the lowest ordinary IDs. Sort
// so local subset IDs and their embedding rows have one deterministic mapping.
inline std::vector<int32_t> qwen4exp_mtp_vocab_ids(int n_vocab, int budget,
                                                  const std::vector<int32_t> & required) {
    if (n_vocab <= 0 || budget <= 0) return {};
    std::vector<bool> keep((size_t) n_vocab, false);
    int count = 0;
    for (int32_t id : required) if (id >= 0 && id < n_vocab && !keep[id]) {
        keep[id] = true;
        ++count;
    }
    for (int id = 0; id < n_vocab && count < budget; ++id) if (!keep[id]) {
        keep[id] = true;
        ++count;
    }
    std::vector<int32_t> ids;
    ids.reserve(count);
    for (int id = 0; id < n_vocab; ++id) if (keep[id]) ids.push_back(id);
    return ids;
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
