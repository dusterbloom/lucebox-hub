// Pure scoring math for the DS4.1 bridge probe: NLL of the true token, KL
// from the native arm's full-vocab distribution, top-1 agreement, median.
// No GPU, no IO; tested by test_ds41_bridge_metrics.
#pragma once

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <vector>

namespace ds41 {

// log-softmax of one logit row, in double.
inline std::vector<double> log_softmax(const std::vector<float> & logits) {
    double mx = -INFINITY;
    for (float v : logits) mx = std::max(mx, (double) v);
    double sum = 0.0;
    for (float v : logits) sum += std::exp((double) v - mx);
    const double lse = mx + std::log(sum);
    std::vector<double> out(logits.size());
    for (size_t i = 0; i < logits.size(); ++i) out[i] = (double) logits[i] - lse;
    return out;
}

inline int32_t argmax_row(const std::vector<float> & logits) {
    return (int32_t) (std::max_element(logits.begin(), logits.end()) - logits.begin());
}

struct ArmScore {
    double nll = 0.0;   // mean -log p_arm(true token)
    double kl = 0.0;    // mean KL(A || arm)
    double top1 = 0.0;  // fraction of steps where argmax(arm) == argmax(A)
};

// Scores `arm` against the native arm `ref` over the same teacher-forced
// steps; `truth[i]` is the token that step i predicts.
inline ArmScore score_arm(const std::vector<std::vector<float>> & ref,
                          const std::vector<std::vector<float>> & arm,
                          const std::vector<int32_t> & truth) {
    ArmScore s;
    const size_t n = std::min({ref.size(), arm.size(), truth.size()});
    for (size_t i = 0; i < n; ++i) {
        const std::vector<double> lp_ref = log_softmax(ref[i]);
        const std::vector<double> lp_arm = log_softmax(arm[i]);
        s.nll -= lp_arm[(size_t) truth[i]];
        double kl = 0.0;
        for (size_t v = 0; v < lp_ref.size(); ++v) {
            const double p = std::exp(lp_ref[v]);
            if (p > 0.0) kl += p * (lp_ref[v] - lp_arm[v]);
        }
        s.kl += kl;
        s.top1 += argmax_row(ref[i]) == argmax_row(arm[i]) ? 1.0 : 0.0;
    }
    if (n) { s.nll /= n; s.kl /= n; s.top1 /= n; }
    return s;
}

inline double median(std::vector<double> v) {
    if (v.empty()) return NAN;
    std::sort(v.begin(), v.end());
    const size_t m = v.size() / 2;
    return v.size() % 2 ? v[m] : 0.5 * (v[m - 1] + v[m]);
}

inline double mean(const std::vector<double> & v) {
    if (v.empty()) return NAN;
    double s = 0.0;
    for (double x : v) s += x;
    return s / (double) v.size();
}

// Replay start for a doc of L tokens after BOS and a replay of R tokens:
// L - R, rounded down to even (no ratio-2 compressor state pending at s).
inline int replay_start(int L, int R) {
    const int s = L - R;
    return s - (s & 1);
}

}  // namespace ds41
