// CPU-only: c++ -std=c++17 -Iserver/src server/test/unit/test_qwen4exp_mtp.cpp -o /tmp/test_qwen4exp_mtp
#include "qwen4exp/qwen4exp_mtp.h"
#include "qwen4exp/qwen4exp_internal.h"

#include <cstdio>
#include <stdexcept>
#include <vector>

using namespace luce::common;

#define CHECK(condition) do { if (!(condition)) throw std::runtime_error(#condition); } while (0)

// P1: free_qwen4exp_weights()/load failure must clear every MTP pointer, or a later load that skips the
// sidecar (missing file, override "0") leaves qwen4exp_cache.cpp's `mtp_eh_proj != nullptr` check seeing a
// stale descriptor from a freed ggml_context. Heap-allocated and intentionally never deleted: Qwen4ExpWeights
// embeds CpuEmbedder/Qwen4ExpPleReader, whose destructors live in qwen4exp_loader.cpp, which this CPU-only
// target does not link; never destroying the object avoids requiring that link for a pure field check.
static void test_reset_qwen4exp_mtp_fields() {
    auto * w = new Qwen4ExpWeights();
    ggml_tensor sentinel{};   // address only; never dereferenced
    w->mtp.wq = &sentinel;
    w->mtp.ffn_down_exps = &sentinel;
    w->mtp_enorm = &sentinel;
    w->mtp_hnorm = &sentinel;
    w->mtp_eh_proj = &sentinel;
    w->mtp_head_norm = &sentinel;
    w->mtp_head_down = &sentinel;
    w->mtp_head_up = &sentinel;
    w->mtp_output = w->mtp_embd = &sentinel;
    w->mtp_vocab_ids = {1, 2};
    w->tok_embd = &sentinel;   // unrelated field: must survive the reset untouched

    reset_qwen4exp_mtp_fields(*w);

    CHECK(w->mtp.wq == nullptr);
    CHECK(w->mtp.ffn_down_exps == nullptr);
    CHECK(w->mtp_enorm == nullptr);
    CHECK(w->mtp_hnorm == nullptr);
    CHECK(w->mtp_eh_proj == nullptr);
    CHECK(w->mtp_head_norm == nullptr);
    CHECK(w->mtp_head_down == nullptr);
    CHECK(w->mtp_head_up == nullptr);
    CHECK(w->tok_embd == &sentinel);
    CHECK(!w->mtp_output && !w->mtp_embd && w->mtp_vocab_ids.empty());
}

// P2: the loader checked eh_proj.ne[0] but not its output dimension, nor any of the other five MTP tensor
// shapes mtp_forward_batch() (qwen4exp_graph.cpp) relies on. An otherwise-valid sidecar with a wrong eh_proj
// output dim (or any other mismatched shape) would reach the graph and crash or silently misbehave instead
// of failing the load with a clear error.
static void test_qwen4exp_mtp_shapes_valid() {
    const int64_t n_embd = 2560, n_hc = 4, hc_lr = 320, hc_dim = n_hc * n_embd;
    const Qwen4ExpMtpShapeDims good{2 * n_embd, n_embd, n_embd, hc_dim, hc_dim, hc_dim, hc_lr, hc_lr, hc_dim};
    CHECK(qwen4exp_mtp_shapes_valid(good, n_embd, n_hc, hc_lr));

    Qwen4ExpMtpShapeDims bad;
    bad = good; bad.eh_proj_ne0 = 2 * n_embd - 1;
    CHECK(!qwen4exp_mtp_shapes_valid(bad, n_embd, n_hc, hc_lr));
    bad = good; bad.eh_proj_ne1 = n_embd + 1;   // the bug this guards: output dim was never checked
    CHECK(!qwen4exp_mtp_shapes_valid(bad, n_embd, n_hc, hc_lr));
    bad = good; bad.enorm_ne0 = n_embd - 1;
    CHECK(!qwen4exp_mtp_shapes_valid(bad, n_embd, n_hc, hc_lr));
    bad = good; bad.hnorm_ne0 = hc_dim - 1;
    CHECK(!qwen4exp_mtp_shapes_valid(bad, n_embd, n_hc, hc_lr));
    bad = good; bad.head_norm_ne0 = hc_dim + 1;
    CHECK(!qwen4exp_mtp_shapes_valid(bad, n_embd, n_hc, hc_lr));
    bad = good; bad.head_down_ne0 = hc_dim - 1;
    CHECK(!qwen4exp_mtp_shapes_valid(bad, n_embd, n_hc, hc_lr));
    bad = good; bad.head_down_ne1 = hc_lr - 1;
    CHECK(!qwen4exp_mtp_shapes_valid(bad, n_embd, n_hc, hc_lr));
    bad = good; bad.head_up_ne0 = hc_lr + 1;
    CHECK(!qwen4exp_mtp_shapes_valid(bad, n_embd, n_hc, hc_lr));
    bad = good; bad.head_up_ne1 = hc_dim - 1;
    CHECK(!qwen4exp_mtp_shapes_valid(bad, n_embd, n_hc, hc_lr));
}

int main() {
    test_reset_qwen4exp_mtp_fields();
    test_qwen4exp_mtp_shapes_valid();
    CHECK(qwen4exp_mtp_vocab_ids(10, 4, {8, 9, 9, -1, 10}) == std::vector<int32_t>({0, 1, 8, 9}));
    CHECK(qwen4exp_mtp_vocab_ids(4, 10, {}) == std::vector<int32_t>({0, 1, 2, 3}));
    CHECK(qwen4exp_mtp_vocab_ids(10, 1, {8, 9}) == std::vector<int32_t>({8, 9}));
    CHECK(qwen4exp_mtp_vocab_ids(10, 0, {8}).empty());

    // Fixed overrides ignore all feedback, including invalid/cold timings.
    for (int k = 1; k <= QWEN4EXP_MTP_MAX_DRAFT; ++k) {
        auto fixed = qwen4exp_mtp_width_policy(k, false);
        for (int i = 0; i < 32; ++i) {
            fixed.observe(1, k + 1, 1000.0f);
            CHECK(qwen4exp_mtp_next_width(fixed) == k + 1);
        }
    }
    auto recover = qwen4exp_mtp_width_policy(7, true, 66060);
    for (int i = 0; i < 128; ++i) recover.observe(1, qwen4exp_mtp_next_width(recover));
    int probes = 0;
    for (int i = 0; i < 64; ++i) {
        const int w = qwen4exp_mtp_next_width(recover);
        probes += w > 2;
        recover.observe(1, w);
    }
    CHECK(probes >= 3); // k=1 cannot suppress the exploration floor
    for (int i = 0; i < 1500; ++i) {
        const int w = qwen4exp_mtp_next_width(recover);
        recover.observe(w, w);
    }
    CHECK(qwen4exp_mtp_next_width(recover) == 8);

    // Stationary prefix distributions, including a sharp depth-2 cliff. The
    // same latent prefix stream is used regardless of the selected width.
    // Test actual useful tokens/time against every fixed k, not final width.
    const std::vector<std::array<double, 7>> distributions{
        {1, 1, 1, 1, 1, 1, 1}, {.94, .86, .75, .62, .50, .40, .30},
        {.85, .69, .47, .30, .20, .10, .05}, {.79, .56, .36, .20, .10, .05, .02},
        {.68, .34, .18, .09, .045, .02, .01}, {.40, .16, .064, .0256, .01, .004, .001},
        {.95, .10, .08, .06, .04, .02, .01}};
    double worst_ratio = 1.0;
    for (int context : {16000, 66060}) for (const auto & survival : distributions) {
        const double base = context < 32768 ? 45 : 47;
        // Include a substantially different per-context slope to exercise
        // measured cost adaptation, not merely agreement with the priors.
        for (double slope : {16.0, 20.0, 28.0}) for (bool noisy : {false, true}) {
            auto policy = qwen4exp_mtp_width_policy(7, true, context);
            uint32_t random = 1;
            double tokens = 0, elapsed = 0;
            for (int i = 0; i < 12000; ++i) {
                const int width = qwen4exp_mtp_next_width(policy);
                CHECK(width >= 2 && width <= 8);
                random = random * 1664525u + 1013904223u;
                const double draw = random / 4294967296.0;
                int accepted = 1;
                while (accepted < width && draw < survival[accepted - 1]) ++accepted;
                const double cost = base + slope * (width - 1);
                // Early shape costs and a delayed 900ms spike, then zero-mean
                // +/-20% jitter. Accounting uses the true cost, not its noise.
                const double measured = noisy ? (i < 8 || i == 80 ? cost + 900 :
                                                  cost * (i % 2 ? 1.2 : .8)) : cost;
                policy.observe(accepted, width, measured);
                if (i >= 1000) { tokens += accepted; elapsed += cost; }
            }
            double best = 0, expected = 1;
            for (int k = 1; k <= 7; ++k) {
                expected += survival[k - 1];
                best = std::max(best, expected / (base + slope * k));
            }
            worst_ratio = std::min(worst_ratio, tokens / elapsed / best);
            if (tokens / elapsed < 0.95 * best)
                std::printf("FAIL ctx=%d s1=%.2f slope=%.0f noisy=%d ratio=%.3f fit %.1f+%.1fk\n", context, survival[0], slope,
                            (int) noisy, tokens / elapsed / best, policy.base(), policy.slope());
            CHECK(tokens / elapsed >= 0.95 * best);
        }
    }
    std::printf("MTP policy stationary/noisy worst ratio to best fixed: %.4f\n", worst_ratio);

    // Pure cost fit. Samples at one width can move the intercept only: the slope is the prior's until a second
    // width supplies a difference (this is what the legacy ridge fit got wrong: k=1-only samples cheaper than its
    // prior dragged the base down and left the slope at 20, so every wide window looked 2x its real price).
    {
        Qwen4ExpMtpCostFit fit(47.0, 12.0);
        for (int i = 0; i < 200; ++i) fit.observe(1, 70.0 + (i % 2 ? 2.0 : -2.0));
        CHECK(std::fabs(fit.slope - 12.0) < 1e-9);
        CHECK(std::fabs(fit.base - 58.0) < 0.5);              // 70 = base + 12, prior base 47 decayed out
        CHECK(std::fabs(fit.predict(4) - (58.0 + 48.0)) < 0.5);
        for (int i = 0; i < 200; ++i) fit.observe(4, 130.0);  // true curve 50 + 20k
        CHECK(fit.slope > 18.0 && fit.slope < 20.5);
        CHECK(std::fabs(fit.predict(1) - 70.0) < 2.0 && std::fabs(fit.predict(4) - 130.0) < 2.0);
        Qwen4ExpMtpCostFit spike(47.0, 12.0);
        spike.observe(2, 1000.0);                              // clamped to 1.25x the prediction
        CHECK(spike.predict(2) < 1.25 * (47.0 + 24.0) + 1e-9);
    }

    // Pure pick: survivals and a cost curve in, the best width out.
    {
        std::array<double, QWEN4EXP_MTP_MAX_VERIFY> trials, successes;
        trials.fill(100.0);
        for (int d = 0; d < QWEN4EXP_MTP_MAX_VERIFY; ++d) successes[d] = 100.0 * (d == 1 ? 0.69 : 0.81);
        CHECK(qwen4exp_mtp_pick_width(trials, successes, 7, Qwen4ExpMtpCostFit(47.5, 11.8)) == 3);  // prose16k
        CHECK(qwen4exp_mtp_pick_width(trials, successes, 7, Qwen4ExpMtpCostFit(45.0, 20.0)) == 2);  // legacy prior
        CHECK(qwen4exp_mtp_pick_width(trials, successes, 2, Qwen4ExpMtpCostFit(47.5, 11.8)) == 2);  // cap
        successes.fill(99.5);
        CHECK(qwen4exp_mtp_pick_width(trials, successes, 7, Qwen4ExpMtpCostFit(49.7, 9.8)) == 7);   // counting
        successes.fill(20.0);
        CHECK(qwen4exp_mtp_pick_width(trials, successes, 7, Qwen4ExpMtpCostFit(47.5, 11.8)) == 1);
        // The policy's own 66K-code beliefs (step 40 of a traced request: depth 3-4 at 0.84-0.95, learned under
        // its k=3-5 steps) put k=2/3/4 within 3% of each other; fixed k=2 beat fixed k=4 by 10% e2e there.
        for (int d = 0; d < QWEN4EXP_MTP_MAX_VERIFY; ++d) successes[d] = 100.0 * (d == 1 ? 0.73 : d == 2 ? 0.70 : d == 3 ? 0.84 : 0.95);
        CHECK(qwen4exp_mtp_pick_width(trials, successes, 7, Qwen4ExpMtpCostFit(48.4, 14.7)) == 3);
    }

    // The measured lucebox4 curves (MTP-E2E.md: step ms = 47.5 + 11.8 k on the 16-20K prompts, 49.7 + 9.8 k on
    // the 72-token one) with the per-depth acceptance of the e2e runs, in 256-token requests (~133 steps, the
    // policy restarts per request). The legacy policy sat at k=1 on 60-80% of these steps (ratio ~0.93); the
    // retuned one has to be within 3% of the best fixed width on every workload and above the legacy one.
    {
        struct Workload { const char * name; double base, slope; int steps; std::array<double, 7> survival; double min_gain; };
        const Workload workloads[] = {
            {"prose16k", 47.5, 11.8, 133, {.69, .559, .453, .367, .297, .24, .195}, 0.03},
            {"code16k", 47.3, 11.5, 120, {.657, .493, .37, .277, .208, .156, .117}, 0.03},
            {"counting", 49.7, 9.8, 40, {.995, .99, .985, .98, .975, .97, .965}, 0.0},
            // code16k's per-depth acceptance from its fixed k=7 trace (depth 4+ at 0.2) on the measured 64K-context
            // curve (48.1 + 14.6 k): with the old 16-at-90% acceptance prior the retuned fit picked k=3-5 here and
            // lost 10% e2e to the legacy policy's k=1-2.
            {"code64k", 48.1, 14.6, 85, {.767, .507, .356, .205, .151, .096, .055}, 0.0},
            // A heavier (unmeasured) curve: the prior is off by a third, the fit has to learn it within a request.
            {"ctx64k", 56.0, 16.0, 133, {.69, .559, .453, .367, .297, .24, .195}, -0.02},
        };
        for (const auto & wl : workloads) {
            double best = 0, expected = 1;
            for (int k = 1; k <= 7; ++k) {
                expected += wl.survival[k - 1];
                best = std::max(best, expected / (wl.base + wl.slope * k));
            }
            double ratio[2]{};
            for (bool legacy : {true, false}) {
                uint32_t random = 7;
                double tokens = 0, elapsed = 0;
                std::array<int, 8> histogram{};
                for (int rep = 0; rep < 20; ++rep) {
                    auto policy = qwen4exp_mtp_width_policy(7, true, 20037, legacy);
                    for (int i = 0; i < wl.steps; ++i) {
                        const int width = qwen4exp_mtp_next_width(policy);
                        ++histogram[width - 1];
                        random = random * 1664525u + 1013904223u;
                        const double draw = random / 4294967296.0;
                        int accepted = 1;
                        while (accepted < width && draw < wl.survival[accepted - 1]) ++accepted;
                        const double cost = wl.base + wl.slope * (width - 1);
                        policy.observe(accepted, width, cost * (i % 2 ? 1.05 : .95));
                        tokens += accepted;
                        elapsed += cost;
                    }
                }
                ratio[legacy] = tokens / elapsed / best;
                std::printf("MTP measured %s %s ratio to best fixed %.3f k1..7=", wl.name, legacy ? "legacy" : "retuned", ratio[legacy]);
                for (int k = 1; k <= 7; ++k) std::printf("%s%d", k == 1 ? "" : "/", histogram[k]);
                std::printf("\n");
            }
            CHECK(ratio[0] >= 0.97);
            CHECK(ratio[0] >= ratio[1] + wl.min_gain);
        }
    }

    // Short requests resemble the session's code length and counting cell.
    // Feed identical latent acceptance to clean/noisy controllers; a spike
    // must not create a throughput collapse or an absorbing narrow width.
    for (int seed = 1; seed <= 3; ++seed) for (bool counting : {false, true}) {
        double rates[2]{};
        for (int noise = 0; noise < 2; ++noise) {
            auto policy = qwen4exp_mtp_width_policy(7, true, 66060);
            uint32_t random = seed;
            double elapsed = 0;
            int tokens = 0, steps = 0;
            std::array<int, 8> histogram{};
            while (tokens < (counting ? 511 : 1023)) {
                const int width = qwen4exp_mtp_next_width(policy);
                ++histogram[width - 1];
                random = random * 1664525u + 1013904223u;
                const double draw = random / 4294967296.0;
                int accepted = 1;
                while (accepted < width && draw < distributions[counting ? 0 : 1][accepted - 1]) ++accepted;
                const double cost = 47 + 20 * (width - 1);
                policy.observe(accepted, width, noise ? (steps == 80 ? cost + 900 :
                                                           cost * (steps % 2 ? 1.2 : .8)) : cost);
                tokens += accepted;
                elapsed += cost;
                ++steps;
            }
            rates[noise] = tokens / elapsed;
            CHECK(histogram[1] < steps / 10);
            std::printf("MTP synthetic %s seed=%d noise=%d tps=%.3f tokens/step=%.3f k1..7=",
                        counting ? "counting" : "code", seed, noise, rates[noise] * 1000, double(tokens) / steps);
            for (int k = 1; k <= 7; ++k) std::printf("%s%d", k == 1 ? "" : "/", histogram[k]);
            std::printf("\n");
        }
        CHECK(rates[1] >= .98 * rates[0]);
    }

    int cases = 0;
    for (int k = 0; k <= QWEN4EXP_MTP_MAX_DRAFT; ++k) {
        std::array<int32_t, QWEN4EXP_MTP_MAX_VERIFY> drafts{11, 22, 33, 44, 55, 66, 77, 88}, samples{};
        // Every equality pattern, including later matches following a reject.
        for (int mask = 0; mask < (1 << (k + 1)); ++mask) {
            for (int i = 0; i <= k; ++i) samples[i] = drafts[i] + ((mask >> i) & 1);
            for (int n = 0; n <= k + 1; ++n) {
                const auto result = qwen4exp_mtp_accept(drafts.data(), k, samples.data(), n);
                int accepted = 0;
                while (accepted < k && accepted < n && samples[accepted] == drafts[accepted]) ++accepted;
                const int emitted = std::min(n, accepted + 1);
                CHECK(result.n_accepted == accepted);
                CHECK(result.n_emitted == emitted);
                for (int i = 0; i < emitted; ++i) CHECK(result.emitted[i] == samples[i]);
                ++cases;
            }
        }
    }

    // Rollback at all block alignments and accepted prefixes, including a
    // five-token verify that completes TWO blocks. Stale pooled rows must be
    // excluded, then recomputed from replacement raw keys on block completion.
    for (int k = 1; k <= QWEN4EXP_MTP_MAX_DRAFT; ++k) {
        for (int alignment = 0; alignment < 4; ++alignment) {
            for (int retained = 1; retained <= k + 1; ++retained) {
                const int pos = 2200 + alignment, end = pos + retained;
                const int verified_blocks = (pos + k + 1) / 4;
                const int blocks = qwen4exp_mtp_retained_blocks(verified_blocks, end, 4);
                CHECK(blocks == end / 4);
                CHECK(qwen4exp_mtp_retained_blocks(0, end, 4) == 0); // dense prefix has no pooled keys
                std::vector<int> raw(pos + k + 8), pooled(raw.size() / 4, -1);
                for (size_t i = 0; i < raw.size(); ++i) raw[i] = (int) i;
                auto pool = [&](int block) {
                    return raw[4 * block] + raw[4 * block + 1] + raw[4 * block + 2] + raw[4 * block + 3];
                };
                for (int b = 0; b < verified_blocks; ++b) pooled[b] = pool(b);
                const int next_end = (end / 4 + 1) * 4;
                for (int p = end; p < next_end; ++p) raw[p] = -p;
                for (int b = blocks; b < next_end / 4; ++b) pooled[b] = pool(b);
                for (int b = 0; b < next_end / 4; ++b) CHECK(pooled[b] == pool(b));
                ++cases;
            }
        }
    }
    std::printf("qwen4exp MTP: %d acceptance/rollback cases passed; config and adaptive policy passed\n", cases);
}
