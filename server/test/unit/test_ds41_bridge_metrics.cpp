// CPU-only checks for the probe's scoring math (test/ds41_bridge_metrics.h).
#include "ds41_bridge_metrics.h"

#include <cstdio>

static int failures = 0;
static void check(bool ok, const char * what) {
    if (!ok) { std::fprintf(stderr, "FAIL: %s\n", what); ++failures; }
}
static bool near(double a, double b, double tol = 1e-9) { return std::fabs(a - b) <= tol; }

int main() {
    using namespace ds41;
    // Identical rows: KL 0, top-1 1, NLL of token 1 under softmax([0, ln 3]) = ln(4/3).
    const std::vector<std::vector<float>> a = {{0.0f, (float) std::log(3.0)}};
    const ArmScore same = score_arm(a, a, {1});
    check(near(same.kl, 0.0), "identical KL == 0");
    check(near(same.top1, 1.0), "identical top1 == 1");
    check(near(same.nll, std::log(4.0 / 3.0), 1e-6), "NLL of the likelier token");

    // KL(A||B), A = (3/4, 1/4), B = (1/2, 1/2): 3/4 ln(3/2) + 1/4 ln(1/2).
    const std::vector<std::vector<float>> b = {{0.0f, 0.0f}};
    const ArmScore s = score_arm(a, b, {0});
    check(near(s.kl, 0.75 * std::log(1.5) + 0.25 * std::log(0.5), 1e-6), "KL value");
    check(near(s.nll, std::log(2.0), 1e-9), "NLL under uniform");

    // Top-1 disagreement is counted per step.
    const std::vector<std::vector<float>> r2 = {{1.0f, 0.0f}, {1.0f, 0.0f}};
    const std::vector<std::vector<float>> x2 = {{1.0f, 0.0f}, {0.0f, 1.0f}};
    check(near(score_arm(r2, x2, {0, 0}).top1, 0.5), "top1 half agree");

    // Large logits must not overflow.
    const std::vector<std::vector<float>> big = {{1000.0f, 1001.0f}};
    check(std::isfinite(score_arm(big, big, {0}).nll), "stable log-softmax");

    check(near(median({3, 1, 2}), 2), "median odd");
    check(near(median({4, 1, 2, 3}), 2.5), "median even");
    check(replay_start(2048, 128) == 1920 && replay_start(8191, 128) == 8062 &&
          replay_start(8192, 255) == 7936, "replay_start rounds down to even");
    // Top-N: order, ties (lower id first), N == vocab, N > vocab, extreme values.
    const TopK t = topk_logprobs({1.0f, 3.0f, 2.0f, 3.0f}, 3);
    check(t.ids == std::vector<int32_t>({1, 3, 2}), "topk order with tie -> lower id");
    check(t.logprobs[0] == t.logprobs[1] && t.logprobs[0] > t.logprobs[2], "topk tie logprobs equal, descending");
    const TopK all = topk_logprobs({0.5f, -1.0f, 2.0f}, 3);
    double tot = 0.0;
    for (float l : all.logprobs) tot += std::exp((double) l);
    check(all.ids == std::vector<int32_t>({2, 0, 1}) && near(tot, 1.0, 1e-6), "topk N=vocab sums to 1");
    check(topk_logprobs({0.0f, 1.0f}, 9).ids.size() == 2, "topk N clamped to vocab");
    check(topk_logprobs({0.0f, 1.0f}, 0).ids.empty(), "topk N=0");
    const TopK ext = topk_logprobs({1e30f, -1e30f, 0.0f}, 3);
    check(ext.ids[0] == 0 && std::isfinite(ext.logprobs[0]) && near(ext.logprobs[0], 0.0, 1e-6) &&
          ext.logprobs[2] < -1e29f, "topk extreme values");
    const std::vector<float> row = {0.3f, 2.5f, -0.7f, 1.1f};
    check(near(-topk_logprobs(row, 4).logprobs[0], score_arm({row}, {row}, {1}).nll, 1e-6),
          "topk logprob agrees with NLL of the argmax token");
    if (failures) return 1;
    std::fprintf(stderr, "PASS: ds41 bridge metrics\n");
    return 0;
}
