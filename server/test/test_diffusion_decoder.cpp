// Numerics/behaviour test for the model-agnostic diffusion decode loop.
//
// Drives run_diffusion_generate() against a synthetic DiffusionModelGraph whose
// forward returns a sharply-peaked logit at a known target token per absolute
// position. This makes the loop deterministic and lets us assert: masked ->
// unmasked convergence, semi-AR block accounting, the ParallelThreshold fast
// path, EOS early-stop, and stream cancellation — all on CPU with no GPU/ggml.
//
// Self-contained: link against diffusion_decoder.cpp + sampler.cpp only.

#include "diffusion/diffusion_decoder.h"

#include <cstdio>
#include <numeric>
#include <vector>

using namespace dflash::common;

namespace {

int g_failed = 0;
int g_passed = 0;

void check(bool ok, const char * name, const std::string & detail = "") {
    if (ok) { ++g_passed; std::printf("  ✅ %s\n", name); }
    else    { ++g_failed; std::printf("  ❌ %s%s%s\n", name,
                                      detail.empty() ? "" : ": ", detail.c_str()); }
}

// Synthetic model: vocab tokens, EOS=30, MASK=31. forward_block returns a peaked
// logit at target_for(absolute_position). Ignores the (noised) canvas content —
// we are testing the loop, not a model.
struct SyntheticModel : DiffusionModelGraph {
    int                  vocab_  = 32;
    int32_t              eos_    = 30;
    int32_t              mask_   = 31;
    int                  prefix_ = 0;
    int                  n_ctx_  = 0;       // 0 = unlimited
    std::vector<int32_t> gen_targets;       // target for absolute position prefix_+i
    int                  forwards = 0;

    int     vocab() const override     { return vocab_; }
    int32_t eos_token() const override { return eos_; }
    int32_t mask_token() const override { return mask_; }
    int     n_ctx_max() const override { return n_ctx_; }

    bool prepare(const std::vector<int32_t> & prompt, int & out_prefix) override {
        out_prefix = (int)prompt.size();
        return true;
    }

    bool forward_block(const std::vector<int32_t> & canvas, int block_begin,
                       int block_len, bool /*bidir*/,
                       std::vector<float> & out) override {
        (void)canvas;
        ++forwards;
        out.assign((size_t)block_len * vocab_, 0.0f);
        for (int j = 0; j < block_len; ++j) {
            const int abs = block_begin + j;
            const int idx = abs - prefix_;
            int tgt = 0;
            if (idx >= 0 && idx < (int)gen_targets.size()) tgt = gen_targets[idx];
            out[(size_t)j * vocab_ + tgt] = 20.0f;  // softmax prob ~= 1.0
        }
        return true;
    }
};

std::vector<int32_t> iota_targets(int n) {
    std::vector<int32_t> t(n);
    for (int i = 0; i < n; ++i) t[i] = (int32_t)(i % 29);  // stay clear of EOS(30)/MASK(31)
    return t;
}

DiffusionStream collect(std::vector<int32_t> & out) {
    DiffusionStream s;
    s.on_token = [&out](int32_t tok) { out.push_back(tok); return true; };
    return s;
}

}  // namespace

int main() {
    std::printf("Diffusion decoder unit tests\n");

    SamplerCfg greedy{};  // temp=0 => argmax path

    // ── 1. LowConfidence masked, greedy, multi-block convergence ──────────
    {
        SyntheticModel m;
        m.gen_targets = iota_targets(20);
        DiffusionConfig cfg;
        cfg.block_size = 8;
        cfg.n_steps    = 8;
        cfg.remasking  = DiffusionRemask::LowConfidence;
        cfg.noise_scheme = DiffusionNoise::Masked;  // mask id resolved from model

        std::vector<int32_t> streamed;
        auto r = run_diffusion_generate(m, /*prompt=*/{}, /*n_gen=*/20, cfg,
                                        greedy, /*do_sample=*/false, collect(streamed));
        check(r.ok, "lowconf: ok");
        check(r.tokens == m.gen_targets, "lowconf: tokens converge to targets");
        check(streamed == m.gen_targets, "lowconf: streamed order matches");
        check(r.stats.blocks == 3, "lowconf: 3 blocks (8+8+4)",
              "got " + std::to_string(r.stats.blocks));
        check(r.stats.forward_passes >= 3 && r.stats.forward_passes <= 24,
              "lowconf: forward_passes in range",
              "got " + std::to_string(r.stats.forward_passes));
    }

    // ── 2. ParallelThreshold fast path: all positions finalize in one step ─
    {
        SyntheticModel m;
        m.gen_targets = iota_targets(16);
        DiffusionConfig cfg;
        cfg.block_size           = 16;
        cfg.n_steps              = 8;
        cfg.remasking            = DiffusionRemask::ParallelThreshold;
        cfg.confidence_threshold = 0.9f;

        std::vector<int32_t> streamed;
        auto r = run_diffusion_generate(m, {}, 16, cfg, greedy, false, collect(streamed));
        check(r.ok && r.tokens == m.gen_targets, "threshold: tokens correct");
        check(r.stats.blocks == 1, "threshold: single block");
        check(r.stats.forward_passes == 1, "threshold: one forward pass",
              "got " + std::to_string(r.stats.forward_passes));
    }

    // ── 3. EOS early-stop mid-block ───────────────────────────────────────
    {
        SyntheticModel m;
        m.gen_targets = iota_targets(20);
        m.gen_targets[5] = m.eos_;  // EOS at the 6th generated position
        DiffusionConfig cfg;
        cfg.block_size = 8;
        cfg.n_steps    = 4;

        std::vector<int32_t> streamed;
        auto r = run_diffusion_generate(m, {}, 20, cfg, greedy, false, collect(streamed));
        check(r.ok, "eos: ok");
        check(r.tokens.size() == 6, "eos: stops after 6 tokens",
              "got " + std::to_string(r.tokens.size()));
        check(!r.tokens.empty() && r.tokens.back() == m.eos_, "eos: last token is EOS");
        check(r.stats.blocks == 1, "eos: only one block committed");
    }

    // ── 4. Stream cancellation ────────────────────────────────────────────
    {
        SyntheticModel m;
        m.gen_targets = iota_targets(20);
        DiffusionConfig cfg;
        cfg.block_size = 16;
        cfg.n_steps    = 4;

        std::vector<int32_t> streamed;
        DiffusionStream s;
        int calls = 0;
        s.on_token = [&](int32_t tok) { streamed.push_back(tok); return ++calls < 3; };
        auto r = run_diffusion_generate(m, {}, 20, cfg, greedy, false, s);
        check(r.ok, "cancel: ok");
        check(r.tokens.size() == 3, "cancel: stops at 3 tokens",
              "got " + std::to_string(r.tokens.size()));
    }

    // ── 5. Uniform-state noise (no MASK token) + non-empty prompt ──────────
    {
        SyntheticModel m;
        m.prefix_ = 4;                       // pretend a 4-token prompt
        m.gen_targets = iota_targets(12);
        DiffusionConfig cfg;
        cfg.block_size   = 6;
        cfg.n_steps      = 6;
        cfg.noise_scheme = DiffusionNoise::UniformState;  // mask id irrelevant

        std::vector<int32_t> prompt = {100, 101, 102, 103};
        std::vector<int32_t> streamed;
        auto r = run_diffusion_generate(m, prompt, 12, cfg, greedy, false, collect(streamed));
        check(r.ok && r.tokens == m.gen_targets, "uniform: tokens correct with prompt prefix");
        check(r.stats.blocks == 2, "uniform: 2 blocks (6+6)",
              "got " + std::to_string(r.stats.blocks));
    }

    // ── 6. Config guard: masked scheme with no mask id available ───────────
    {
        struct NoMaskModel : SyntheticModel {
            int32_t mask_token() const override { return -1; }
        } m;
        m.gen_targets = iota_targets(4);
        DiffusionConfig cfg;
        cfg.block_size   = 4;
        cfg.noise_scheme = DiffusionNoise::Masked;
        cfg.mask_token_id = -1;
        std::vector<int32_t> streamed;
        auto r = run_diffusion_generate(m, {}, 4, cfg, greedy, false, collect(streamed));
        check(!r.ok && !r.error.empty(), "guard: rejects masked scheme without mask id");
    }

    // ── EntropyBound tests ────────────────────────────────────────────────────
    // Synthetic EB model: forward_block returns peaked logits (prob ~1 at target
    // token per position), so entropy is near-zero everywhere. Also tracks set_sc.
    struct EBModel : SyntheticModel {
        int             sc_calls   = 0;
        float           last_sc_use    = -1.0f;
        float           last_sc_temp   = -1.0f;
        const float *   last_sc_logits = nullptr;

        void set_sc(const float * logits, float use, float temp_inv) override {
            ++sc_calls;
            last_sc_logits = logits;
            last_sc_use    = use;
            last_sc_temp   = temp_inv;
        }
    };

    // ── 7. EntropyBound: basic convergence with near-zero entropy ─────────────
    // With sharply-peaked logits, entropy per position ≈ 0, so all positions are
    // accepted on the first step and the argmax matches targets.
    {
        EBModel m;
        m.prefix_ = 0;
        m.gen_targets = iota_targets(8);
        DiffusionConfig cfg;
        cfg.remasking             = DiffusionRemask::EntropyBound;
        cfg.noise_scheme          = DiffusionNoise::UniformState;
        cfg.eb_max_steps          = 4;
        cfg.eb_t_min              = 0.4f;
        cfg.eb_t_max              = 0.8f;
        cfg.eb_entropy_bound      = 0.1f;
        cfg.eb_stability_threshold = 1;
        cfg.eb_confidence_threshold = 0.005f;
        cfg.seed                  = 42;

        std::vector<int32_t> streamed;
        auto r = run_diffusion_generate(m, {}, 8, cfg, greedy, false, collect(streamed));
        check(r.ok, "eb-basic: ok");
        check(r.tokens == m.gen_targets, "eb-basic: argmax tokens match targets");
        check(r.stats.blocks == 1, "eb-basic: single block");
        // With near-zero entropy and stability=1, should stop early (1-2 steps)
        check(r.stats.forward_passes >= 1 && r.stats.forward_passes <= 4,
              "eb-basic: forward_passes in [1,4]",
              "got " + std::to_string(r.stats.forward_passes));
        check((int)streamed.size() == 8, "eb-basic: 8 tokens streamed",
              "got " + std::to_string(streamed.size()));
    }

    // ── 8. EntropyBound: SC threading — set_sc called each step ──────────────
    // Verify step 0 passes sc_use=0 (nullptr), step 1+ passes sc_use=1.
    {
        EBModel m;
        m.prefix_ = 0;
        m.gen_targets = iota_targets(4);
        DiffusionConfig cfg;
        cfg.remasking              = DiffusionRemask::EntropyBound;
        cfg.noise_scheme           = DiffusionNoise::UniformState;
        cfg.eb_max_steps           = 6;
        cfg.eb_t_min               = 0.4f;
        cfg.eb_t_max               = 0.8f;
        cfg.eb_entropy_bound       = 0.1f;
        cfg.eb_stability_threshold = 100; // never stop early — run all 6 steps
        cfg.eb_confidence_threshold = 0.0f;
        cfg.seed = 1;

        std::vector<int32_t> streamed;
        auto r = run_diffusion_generate(m, {}, 4, cfg, greedy, false, collect(streamed));
        check(r.ok, "eb-sc: ok");
        check(m.sc_calls == 6, "eb-sc: set_sc called once per step (6 steps)",
              "got " + std::to_string(m.sc_calls));
        // Step 0 call: sc_use=0, nullptr logits
        // We can't directly check that without per-call capture, but last call is step 5
        // which has sc_use=1.0 and non-null logits.
        check(m.last_sc_use == 1.0f, "eb-sc: final step has sc_use=1.0",
              "got " + std::to_string(m.last_sc_use));
        check(m.last_sc_logits != nullptr, "eb-sc: final step has non-null sc logits");
    }

    // ── 9. EntropyBound: temperature schedule ────────────────────────────────
    // Run 3 steps and verify the loop doesn't crash; with t_min=t_max the
    // temperature is constant — this exercises the schedule edge case.
    {
        EBModel m;
        m.prefix_ = 0;
        m.gen_targets = iota_targets(6);
        DiffusionConfig cfg;
        cfg.remasking              = DiffusionRemask::EntropyBound;
        cfg.noise_scheme           = DiffusionNoise::UniformState;
        cfg.eb_max_steps           = 3;
        cfg.eb_t_min               = 0.5f;
        cfg.eb_t_max               = 0.5f;  // constant temperature
        cfg.eb_entropy_bound       = 10.0f; // wide bound => accept all
        cfg.eb_stability_threshold = 1;
        cfg.eb_confidence_threshold = 0.005f;
        cfg.seed = 7;

        std::vector<int32_t> streamed;
        auto r = run_diffusion_generate(m, {}, 6, cfg, greedy, false, collect(streamed));
        check(r.ok, "eb-temp: ok with constant temperature");
        check(r.tokens == m.gen_targets, "eb-temp: tokens correct");
    }

    // ── 10. EntropyBound: entropy-sorted acceptance ───────────────────────────
    // Use a model that returns uniform logits (high entropy) for position 0 and
    // peaked logits (low entropy) for positions 1+. With a tight entropy_bound,
    // only the low-entropy positions should be accepted first.
    {
        struct HighEntropyPos0Model : EBModel {
            bool forward_block(const std::vector<int32_t> & canvas, int block_begin,
                               int block_len, bool bidir,
                               std::vector<float> & out) override {
                ++forwards;
                out.assign((size_t)block_len * vocab_, 0.0f);
                for (int j = 0; j < block_len; ++j) {
                    const int idx = block_begin + j - prefix_;
                    int tgt = 0;
                    if (idx >= 0 && idx < (int)gen_targets.size()) tgt = gen_targets[idx];
                    if (j == 0) {
                        // Position 0: uniform logits => high entropy
                        float val = 1.0f;
                        for (int v = 0; v < vocab_; ++v) out[(size_t)j * vocab_ + v] = val;
                    } else {
                        // Positions 1+: peaked logit => low entropy
                        out[(size_t)j * vocab_ + tgt] = 20.0f;
                    }
                }
                return true;
            }
        } m;
        m.prefix_ = 0;
        m.gen_targets = iota_targets(4);

        DiffusionConfig cfg;
        cfg.remasking              = DiffusionRemask::EntropyBound;
        cfg.noise_scheme           = DiffusionNoise::UniformState;
        cfg.eb_max_steps           = 8;
        cfg.eb_t_min               = 0.4f;
        cfg.eb_t_max               = 0.8f;
        // Tight bound: only positions whose prior cumulative entropy is very small pass
        cfg.eb_entropy_bound       = 0.01f;
        cfg.eb_stability_threshold = 1;
        cfg.eb_confidence_threshold = 100.0f; // don't stop early on mean entropy
        cfg.seed = 99;

        std::vector<int32_t> streamed;
        auto r = run_diffusion_generate(m, {}, 4, cfg, greedy, false, collect(streamed));
        check(r.ok, "eb-accept: ok");
        // The argmax is correct for all positions (since argmax of uniform = 0
        // and argmax of peaked = target), but we specifically check that tokens
        // 1..3 match targets (low entropy → accepted → denoiser = sampled near target).
        // Position 0 has high entropy so argmax(uniform) = 0 (index 0 of vocab).
        check(r.tokens[1] == m.gen_targets[1] &&
              r.tokens[2] == m.gen_targets[2] &&
              r.tokens[3] == m.gen_targets[3],
              "eb-accept: low-entropy positions 1-3 converge to targets");
        check(r.stats.forward_passes >= 1, "eb-accept: at least one forward");
    }

    // ── 11. EntropyBound: stop condition — stability_threshold ───────────────
    // Force argmax to be stable from step 0 (peaked model). With stability=2,
    // the loop must run at least 2 steps before stopping even if confident.
    {
        EBModel m;
        m.prefix_ = 0;
        m.gen_targets = iota_targets(4);
        DiffusionConfig cfg;
        cfg.remasking              = DiffusionRemask::EntropyBound;
        cfg.noise_scheme           = DiffusionNoise::UniformState;
        cfg.eb_max_steps           = 10;
        cfg.eb_t_min               = 0.4f;
        cfg.eb_t_max               = 0.8f;
        cfg.eb_entropy_bound       = 0.1f;
        cfg.eb_stability_threshold = 2;          // require 2 consecutive identical argmax
        cfg.eb_confidence_threshold = 0.005f;
        cfg.seed = 5;

        std::vector<int32_t> streamed;
        auto r = run_diffusion_generate(m, {}, 4, cfg, greedy, false, collect(streamed));
        check(r.ok, "eb-stability: ok");
        // With peaked logits, argmax is stable from step 1 (step 0 sets prev_argmax,
        // step 1 sees same argmax → held=1; step 2 → held=2 → stop). So ≥3 steps.
        check(r.stats.forward_passes >= 2, "eb-stability: at least 2 forward passes",
              "got " + std::to_string(r.stats.forward_passes));
        check(r.tokens == m.gen_targets, "eb-stability: tokens correct");
    }

    // ── 12. EntropyBound: prev-logits carry for SC ────────────────────────────
    // Verify that sc_buffer (prev logits) from step N is non-null and passed into
    // step N+1. We do this by checking set_sc receives a non-null pointer starting
    // from step 1. EBModel.last_sc_logits tracks the most-recent call.
    {
        struct SCTrackModel : EBModel {
            bool sc_null_on_step0    = false;
            bool sc_nonnull_on_step1 = false;
            int  call_idx            = 0;

            void set_sc(const float * logits, float use, float temp_inv) override {
                EBModel::set_sc(logits, use, temp_inv);
                if (call_idx == 0) sc_null_on_step0    = (logits == nullptr && use == 0.0f);
                if (call_idx == 1) sc_nonnull_on_step1 = (logits != nullptr && use == 1.0f);
                ++call_idx;
            }
        } m;
        m.prefix_ = 0;
        m.gen_targets = iota_targets(4);
        DiffusionConfig cfg;
        cfg.remasking              = DiffusionRemask::EntropyBound;
        cfg.noise_scheme           = DiffusionNoise::UniformState;
        cfg.eb_max_steps           = 4;
        cfg.eb_t_min               = 0.4f;
        cfg.eb_t_max               = 0.8f;
        cfg.eb_entropy_bound       = 0.1f;
        cfg.eb_stability_threshold = 100; // run all steps
        cfg.eb_confidence_threshold = 0.0f;
        cfg.seed = 17;

        std::vector<int32_t> streamed;
        auto r = run_diffusion_generate(m, {}, 4, cfg, greedy, false, collect(streamed));
        check(r.ok, "eb-prev-logits: ok");
        check(m.sc_null_on_step0,    "eb-prev-logits: step 0 passes null SC + use=0");
        check(m.sc_nonnull_on_step1, "eb-prev-logits: step 1 passes non-null SC + use=1");
    }

    std::printf("\nResults: %d/%d passed, %d failed\n",
                g_passed, g_passed + g_failed, g_failed);
    return g_failed == 0 ? 0 : 1;
}
