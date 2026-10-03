// CPU-only: c++ -std=c++17 -Iserver/src server/test/test_qwen4exp_mtp.cpp -o /tmp/test_qwen4exp_mtp
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

    // Same controller and cost seeds as the server. A rejection must narrow;
    // clean drafts at that narrower width must recover without unseen-depth
    // evidence being frozen forever. Fixed widths must never adapt.
    for (int k = 1; k <= 4; ++k) {
        auto fixed = qwen4exp_mtp_width_policy(k, false);
        for (int i = 0; i < 32; ++i) {
            fixed.observe(1, k + 1, 1000.0f);
            CHECK(fixed.next_width_cost_aware({}) == k + 1);
        }
    }
    auto adaptive = qwen4exp_mtp_width_policy(3, true);
    CHECK(adaptive.next_width_cost_aware({}, 1) == 1); // remaining-token cap
    for (int i = 0; i < 32; ++i) adaptive.observe(1, adaptive.next_width_cost_aware({}));
    CHECK(adaptive.next_width_cost_aware({}) == 2);
    for (int i = 0; i < 32; ++i) {
        const int width = adaptive.next_width_cost_aware({});
        CHECK(width >= 2 && width <= 4);
        adaptive.observe(width, width);
    }
    CHECK(adaptive.next_width_cost_aware({}) == 4);

    // Deterministic stationary prefix distributions: structured, code-like,
    // intermediate and prose-like acceptance. Check realized throughput, not
    // just the last chosen width, against the best fixed width for each case.
    // These are synthetic policy tests, not predictions of GPU performance.
    const std::array<float, 5> costs{0, 0, 64, 82, 97};
    for (const auto & survival : std::vector<std::array<float, 3>>{
            {1.0f, 1.0f, 1.0f}, {0.94f, 0.86f, 0.75f},
            {0.85f, 0.69f, 0.47f}, {0.79f, 0.56f, 0.36f},
            {0.68f, 0.34f, 0.18f}, {0.40f, 0.16f, 0.064f}}) {
        auto policy = qwen4exp_mtp_width_policy(3, true);
        uint32_t random = 1;
        double tokens = 0, elapsed = 0;
        for (int i = 0; i < 10000; ++i) {
            const int width = policy.next_width_cost_aware({});
            CHECK(width >= 2 && width <= 4);
            random = random * 1664525u + 1013904223u;
            const double draw = random / 4294967296.0;
            int accepted = 1;
            while (accepted < width && draw < survival[accepted - 1]) ++accepted;
            policy.observe(accepted, width, costs[width]);
            if (i >= 100) {
                tokens += accepted;
                elapsed += costs[width];
            }
        }
        double best = 0, expected = 1;
        for (int width = 2; width <= 4; ++width) {
            expected += survival[width - 2];
            best = std::max(best, expected / costs[width]);
        }
        CHECK(tokens / elapsed >= 0.95 * best);
    }

    int cases = 0;
    for (int k = 0; k <= 4; ++k) {
        std::array<int32_t, 5> drafts{11, 22, 33, 44, 55}, samples{};
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
    for (int k = 1; k <= 4; ++k) {
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
