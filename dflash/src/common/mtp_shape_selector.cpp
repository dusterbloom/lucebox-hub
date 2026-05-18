// mtp_shape_selector.cpp — see header for contract.

#include "common/mtp_shape_selector.h"

#include <algorithm>
#include <cmath>
#include <cstdlib>

namespace dflash27b {
namespace mtp {

namespace {

// Stable softmax + Shannon entropy (base 2, bits) over k logits.
// Bits, not nats — makes H ∈ [0, log2(k)] match the env thresholds whose
// scale is calibrated for "bits of uncertainty across the top-K".
// Returns 0 when k ≤ 0 or any non-finite intermediate would arise.
float softmax_entropy_bits(const float * logits, int k) {
    if (!logits || k <= 0) return 0.0f;
    float m = logits[0];
    for (int i = 1; i < k; i++) if (logits[i] > m) m = logits[i];
    float Z = 0.0f;
    for (int i = 0; i < k; i++) Z += std::exp(logits[i] - m);
    if (!(Z > 0.0f) || !std::isfinite(Z)) return 0.0f;
    float H = 0.0f;
    for (int i = 0; i < k; i++) {
        const float p = std::exp(logits[i] - m) / Z;
        if (p > 0.0f) H -= p * std::log2(p);
    }
    return std::isfinite(H) ? H : 0.0f;
}

float env_float(const char * name, float defv) {
    const char * s = std::getenv(name);
    if (!s || !*s) return defv;
    char * end = nullptr;
    const float v = std::strtof(s, &end);
    if (end == s || !std::isfinite(v) || v < 0.0f) return defv;
    return v;
}

}  // namespace

SpecShapeConfig SpecShapeConfig::from_env() {
    SpecShapeConfig c;
    c.h_low  = env_float("DFLASH27B_MTP_H_LOW",  c.h_low);
    c.h_high = env_float("DFLASH27B_MTP_H_HIGH", c.h_high);
    if (c.h_high < c.h_low) c.h_high = c.h_low;
    return c;
}

SpecShape select_spec_shape(const float * topk_logprobs, int k,
                            const SpecShapeConfig & cfg) {
    if (!topk_logprobs || k <= 0) {
        return SpecShape{ SpecShapeKind::ArOnly, 1, 1, 1, 0.0f };
    }
    const float H = softmax_entropy_bits(topk_logprobs, k);
    if (H < cfg.h_low) {
        return SpecShape{ SpecShapeKind::Chain, cfg.chain_gamma, 1, 1, H };
    }
    if (H < cfg.h_high) {
        return SpecShape{ SpecShapeKind::Tree, cfg.tree_gamma, cfg.tree_B, cfg.tree_K, H };
    }
    return SpecShape{ SpecShapeKind::ArOnly, 1, 1, 1, H };
}

}  // namespace mtp
}  // namespace dflash27b
