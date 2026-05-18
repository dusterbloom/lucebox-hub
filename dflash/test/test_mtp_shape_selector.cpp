// Tests for the pure entropy-adaptive spec-shape selector.
// Logits-in / shape-out: no module pointer, no global state.

#include "common/mtp_shape_selector.h"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <vector>

namespace {

using dflash27b::mtp::select_spec_shape;
using dflash27b::mtp::SpecShape;
using dflash27b::mtp::SpecShapeConfig;
using dflash27b::mtp::SpecShapeKind;

void require(bool ok, const char * msg) {
    if (!ok) {
        std::fprintf(stderr, "FAIL: %s\n", msg);
        std::abort();
    }
}

// T1: very peaked top-K → Chain{γ=3}.
void t1_low_entropy_returns_chain() {
    const float logits[4] = { 10.0f, 0.0f, 0.0f, 0.0f };
    SpecShape s = select_spec_shape(logits, 4, SpecShapeConfig{});
    require(s.kind == SpecShapeKind::Chain, "T1 kind=Chain");
    require(s.gamma == 3, "T1 gamma=3");
    require(s.branches == 1 && s.topk == 1, "T1 chain has B=1,K=1");
}

// T2: mid-confidence top-K → Tree{B=2,K=2,γ=2}.
void t2_mid_entropy_returns_tree() {
    const float logits[4] = { 2.0f, 1.0f, 0.0f, -1.0f };
    SpecShape s = select_spec_shape(logits, 4, SpecShapeConfig{});
    require(s.kind == SpecShapeKind::Tree, "T2 kind=Tree");
    require(s.gamma == 2 && s.branches == 2 && s.topk == 2, "T2 tree B=2,K=2,γ=2");
}

// T3: uniform top-K → ArOnly.
void t3_high_entropy_returns_ar_only() {
    const float logits[4] = { 0.0f, 0.0f, 0.0f, 0.0f };
    SpecShape s = select_spec_shape(logits, 4, SpecShapeConfig{});
    require(s.kind == SpecShapeKind::ArOnly, "T3 kind=ArOnly");
    require(s.gamma == 1, "T3 ArOnly gamma=1");
}

// T4: env override raises H_LOW above T1's entropy → T1 logits now Tree.
void t4_env_thresholds_override() {
    setenv("DFLASH27B_MTP_H_LOW", "2.0", 1);
    setenv("DFLASH27B_MTP_H_HIGH", "3.0", 1);
    SpecShapeConfig cfg = SpecShapeConfig::from_env();
    require(cfg.h_low > 1.5f, "T4 h_low picked up from env");
    require(cfg.h_high > 2.5f, "T4 h_high picked up from env");

    const float logits[4] = { 10.0f, 0.0f, 0.0f, 0.0f };
    SpecShape s = select_spec_shape(logits, 4, cfg);
    // T1's peaked logits have H ≈ 0 < 2.0, so still Chain — but raise more.
    require(s.kind == SpecShapeKind::Chain, "T4 peaked still Chain under h_low=2.0");

    // Now make the mid logits land in Chain region (H ≈ 0.95 < 2.0).
    const float mid_logits[4] = { 2.0f, 1.0f, 0.0f, -1.0f };
    SpecShape s2 = select_spec_shape(mid_logits, 4, cfg);
    require(s2.kind == SpecShapeKind::Chain, "T4 mid logits Chain under raised h_low");

    unsetenv("DFLASH27B_MTP_H_LOW");
    unsetenv("DFLASH27B_MTP_H_HIGH");
}

// T5: monotonicity — sweeping from peaked to uniform never raises γ.
void t5_monotonicity() {
    int prev_gamma = 1000;
    for (int step = 0; step <= 20; step++) {
        const float t = (float)step / 20.0f;  // 0 = peaked, 1 = uniform
        float logits[4];
        logits[0] = 10.0f * (1.0f - t);
        logits[1] = 0.0f;
        logits[2] = 0.0f;
        logits[3] = 0.0f;
        SpecShape s = select_spec_shape(logits, 4, SpecShapeConfig{});
        require(s.gamma <= prev_gamma, "T5 gamma non-increasing");
        prev_gamma = s.gamma;
    }
}

// T6: defensive empty/null input → ArOnly.
void t6_defensive_empty_input() {
    SpecShape s0 = select_spec_shape(nullptr, 4, SpecShapeConfig{});
    require(s0.kind == SpecShapeKind::ArOnly, "T6 null pointer → ArOnly");
    require(s0.gamma == 1, "T6 null gamma=1");

    const float logits[1] = { 0.0f };
    SpecShape s1 = select_spec_shape(logits, 0, SpecShapeConfig{});
    require(s1.kind == SpecShapeKind::ArOnly, "T6 k=0 → ArOnly");

    SpecShape s2 = select_spec_shape(logits, -1, SpecShapeConfig{});
    require(s2.kind == SpecShapeKind::ArOnly, "T6 k<0 → ArOnly");
}

// T7: extreme logit magnitudes do not produce NaN/Inf.
void t7_numerical_stability() {
    const float big[4]   = { 1.0e6f, -1.0e6f, -1.0e6f, -1.0e6f };
    SpecShape sb = select_spec_shape(big, 4, SpecShapeConfig{});
    require(sb.gamma >= 1 && sb.gamma <= 8, "T7 big-positive gamma sane");
    require(sb.kind == SpecShapeKind::Chain, "T7 huge spike → Chain");

    const float small[4] = { -1.0e6f, -1.0e6f, -1.0e6f, -1.0e6f };
    SpecShape ss = select_spec_shape(small, 4, SpecShapeConfig{});
    require(ss.gamma >= 1 && ss.gamma <= 8, "T7 all-tiny gamma sane");
    // All equal logits, regardless of magnitude, must be uniform → ArOnly.
    require(ss.kind == SpecShapeKind::ArOnly, "T7 all-equal → ArOnly");
}

}  // namespace

int main() {
    t1_low_entropy_returns_chain();
    t2_mid_entropy_returns_tree();
    t3_high_entropy_returns_ar_only();
    t4_env_thresholds_override();
    t5_monotonicity();
    t6_defensive_empty_input();
    t7_numerical_stability();
    std::printf("test_mtp_shape_selector: 7/7 passed\n");
    return 0;
}
