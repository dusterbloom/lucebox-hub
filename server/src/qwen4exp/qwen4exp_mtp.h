#pragma once

#include <algorithm>
#include <array>
#include <cassert>
#include <cstdint>
#include <cmath>
#include <vector>

namespace luce::common {

constexpr int QWEN4EXP_MTP_MAX_DRAFT = 7;
constexpr int QWEN4EXP_MTP_MAX_VERIFY = QWEN4EXP_MTP_MAX_DRAFT + 1;

// Cycle cost model: bounded per-width sample means where a width has samples,
// the line base + slope * k (k drafted rows) elsewhere. Samples at one width
// move the line's intercept only; its slope comes from differences between
// widths with samples, ridge-blended with the prior. A request that sits at
// k=1 therefore learns its own k=1 cost without tilting what the wide windows
// are believed to cost, and a probed width reports its own measured price.
struct Qwen4ExpMtpCostFit {
    double base, slope;

    Qwen4ExpMtpCostFit(double prior_base, double prior_slope)
        : base(prior_base), slope(prior_slope), prior_base_(prior_base), prior_slope_(prior_slope) {}

    double line(int k) const { return base + slope * k; }

    double predict(int k) const {
        return (sum_[k] + kLineWeight * line(k)) / (n_[k] + kLineWeight);
    }

    void observe(int k, double cycle_ms) {
        const double predicted = predict(k);
        const double measured = std::clamp(cycle_ms, predicted * 0.75, predicted * 1.25);
        if (n_[k] >= 128.0) {
            n_[k] *= 127.0 / 128.0;
            sum_[k] *= 127.0 / 128.0;
        }
        n_[k] += 1.0;
        sum_[k] += measured;
        double total = 0.0, sx = 0.0, sy = 0.0;
        for (int w = 1; w < QWEN4EXP_MTP_MAX_VERIFY; ++w) {
            total += n_[w];
            sx += n_[w] * w;
            sy += sum_[w];
        }
        const double mean_x = sx / total, mean_y = sy / total;
        double sxx = 0.0, sxy = 0.0;
        for (int w = 1; w < QWEN4EXP_MTP_MAX_VERIFY; ++w) if (n_[w] > 0.0) {
            sxx += n_[w] * (w - mean_x) * (w - mean_x);
            sxy += n_[w] * (w - mean_x) * (sum_[w] / n_[w] - mean_y);
        }
        slope = std::max(0.1, (sxy + kSlopePrior * prior_slope_) / (sxx + kSlopePrior));
        base = std::max(1.0, (sy - slope * sx + kBasePrior * prior_base_) / (total + kBasePrior));
    }

private:
    // Ridge strength of the prior slope (in units of sample x-spread), prior
    // tail of the intercept, and the line's weight against a width's own
    // samples (eight samples' worth: pooled across widths against per-width
    // noise, a probed width still speaks for itself within a request).
    static constexpr double kSlopePrior = 16.0, kBasePrior = 4.0, kLineWeight = 8.0;
    double prior_base_, prior_slope_;
    std::array<double, QWEN4EXP_MTP_MAX_VERIFY> n_{}, sum_{};
};

// The policy of 8d08d436 and before: one decayed least-squares line over every
// sample plus 16 synthetic cycles at k=1 and k=7 of the prior curve. Kept as
// the kill switch (LUCE_MTP_WIDTH_POLICY=legacy).
struct Qwen4ExpMtpLegacyCostFit {
    double base, slope = 20.0;

    explicit Qwen4ExpMtpLegacyCostFit(double prior_base)
        : base(prior_base), sy_(32.0 * base + 128.0 * slope), sxy_(128.0 * base + 800.0 * slope) {}

    double predict(int k) const { return base + slope * k; }

    void observe(int k, double cycle_ms) {
        const double predicted = predict(k);
        const double measured = std::clamp(cycle_ms, predicted * 0.75, predicted * 1.25);
        constexpr double decay = 127.0 / 128.0;
        sw_ = decay * sw_ + 1.0;
        sx_ = decay * sx_ + k;
        sxx_ = decay * sxx_ + k * k;
        sy_ = decay * sy_ + measured;
        sxy_ = decay * sxy_ + k * measured;
        const double determinant = sw_ * sxx_ - sx_ * sx_;
        if (determinant > 1e-6) {
            slope = std::max(0.1, (sw_ * sxy_ - sx_ * sy_) / determinant);
            base = std::max(1.0, (sy_ - slope * sx_) / sw_);
        }
    }

private:
    double sw_ = 32.0, sx_ = 128.0, sxx_ = 800.0, sy_, sxy_;
};

// Pure: the width k in 1..cap maximising expected committed tokens per ms,
// expected tokens = 1 + sum of the prefix survivals (products of the per-depth
// conditional acceptance estimates successes/trials).
template <typename Fit>
inline int qwen4exp_mtp_pick_width(const std::array<double, QWEN4EXP_MTP_MAX_VERIFY> & trials,
                                   const std::array<double, QWEN4EXP_MTP_MAX_VERIFY> & successes,
                                   int cap, const Fit & fit) {
    double survival = 1.0, commits = 1.0, best = 0.0;
    int chosen = 1;
    for (int k = 1; k <= cap; ++k) {
        survival *= successes[k] / trials[k];
        commits += survival;
        const double utility = commits / fit.predict(k);
        if (utility > best) { best = utility; chosen = k; }
    }
    return chosen;
}

// MTP-only, per-request policy. Widths include the seed (k = width - 1).
// Learn conditional acceptance only at reached depths: a clean short draft
// says nothing about the next depth, and a rejection is not another failure
// of every deeper conditional. Products give monotone prefix survivals.
class Qwen4ExpMtpWidth {
public:
    Qwen4ExpMtpWidth(int max_draft, bool adaptive, int prompt_tokens, bool legacy = false)
        : cap_(std::clamp(max_draft, 1, QWEN4EXP_MTP_MAX_DRAFT)), adaptive_(adaptive), legacy_(legacy),
          // Prior curve: lucebox4 in-server step ms on 16-20K prompts after the
          // retained verify graphs and the T>1 Q8_0 launches (MTP-E2E.md):
          // k=1 59, k=4 94, i.e. 47.5 + 11.8 k; the fit adapts per request.
          fit_(prompt_tokens >= 32768 ? 50.0 : 47.5, 12.0),
          legacy_fit_(prompt_tokens >= 32768 ? 47.0 : 45.0) {
        // Per-depth acceptance prior. The legacy policy's 16 trials at 90%
        // outlived a 256-token request at the depths k=3-4 steps never reach,
        // and with a cost line that no longer hides the wide windows it sent
        // the policy to k=4-6 on a 64K code prompt whose depth-4+ acceptance
        // is 0.2 (ratio 0.94 of the best fixed width): 4 trials at 80% is
        // overridden by the first real reaches, and counting still climbs to
        // k=7 through the warm-up rule.
        trials_.fill(legacy ? 16.0 : 4.0);
        successes_.fill(legacy ? 16.0 * 0.90 : 4.0 * 0.80);
    }

    bool enabled() const { return adaptive_; }
    bool legacy() const { return legacy_; }
    double base() const { return legacy_ ? legacy_fit_.base : fit_.base; }
    double slope() const { return legacy_ ? legacy_fit_.slope : fit_.slope; }

    int next_width() const {
        if (!adaptive_) return cap_ + 1;
        int chosen = legacy_ ? qwen4exp_mtp_pick_width(trials_, successes_, cap_, legacy_fit_)
                             : qwen4exp_mtp_pick_width(trials_, successes_, cap_, fit_);
        // A wider probe every 17 cycles is unconditional on timing or a
        // clean streak. It refreshes the first censored depth even at k=1.
        if (steps_ >= 16 && (steps_ - 16) % 17 == 0) {
            chosen = std::min(cap_, chosen + 1);
            // Every eighth probe refreshes ALL depths, including after a
            // phase change where two adjacent widths both look unprofitable.
            if (((steps_ - 16) / 17) % 8 == 7) chosen = cap_;
        }
        // Require 16 cycles before narrowing below k=3; clean warmup
        // cycles probe upward immediately so counting reaches k=7 quickly.
        return (steps_ < 16 ? std::max(chosen, std::min(warmup_k_, cap_)) : chosen) + 1;
    }

    // A missing timing (e.g. the oracle smoke) updates acceptance only.
    void observe(int accepted_width, int offered_width, float cycle_ms = -1.0f) {
        if (!adaptive_ || offered_width <= 1) return;
        const int k = std::clamp(offered_width - 1, 1, cap_);
        const int accepted = std::clamp(accepted_width - 1, 0, k);
        ++steps_;
        warmup_k_ = accepted == k ? std::min(cap_, k + 1) : 3;
        for (int depth = 1; depth <= std::min(k, accepted + 1); ++depth) {
            // Bounded evidence adapts to changing text without the old 0.2
            // EMA's effective five-sample window. Unreached depths stay put.
            if (trials_[depth] >= 128.0) {
                trials_[depth] *= 127.0 / 128.0;
                successes_[depth] *= 127.0 / 128.0;
            }
            trials_[depth] += 1.0;
            successes_[depth] += depth <= accepted;
        }
        if (!std::isfinite(cycle_ms) || cycle_ms <= 0.0f) return;
        // Cold builds of a width's verify graph: one eager call and one
        // capture (the legacy policy skipped four, so its k=7 probes, every
        // 136 steps, never reached its fit).
        if (++cost_samples_[k] <= (legacy_ ? 4 : 2)) return;
        if (legacy_) legacy_fit_.observe(k, cycle_ms); else fit_.observe(k, cycle_ms);
    }

private:
    int cap_, steps_ = 0, warmup_k_ = 3;
    bool adaptive_, legacy_;
    std::array<double, QWEN4EXP_MTP_MAX_VERIFY> trials_{}, successes_{};
    std::array<int, QWEN4EXP_MTP_MAX_VERIFY> cost_samples_{};
    Qwen4ExpMtpCostFit fit_;
    Qwen4ExpMtpLegacyCostFit legacy_fit_;
};

// Server --verify-width: 0 = adaptive k=1..7, 1 = off, 2..8 = fixed k=1..7.
// Eight verify rows is the RDNA3 batch-invariant MMVQ/MMID ceiling.
inline Qwen4ExpMtpWidth qwen4exp_mtp_width_policy(int max_draft, bool adaptive, int prompt_tokens = 0,
                                                 bool legacy = false) {
    return Qwen4ExpMtpWidth(max_draft, adaptive, prompt_tokens, legacy);
}

inline int qwen4exp_mtp_next_width(const Qwen4ExpMtpWidth & policy) {
    return policy.next_width();
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
