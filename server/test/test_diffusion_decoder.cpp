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

    std::printf("\nResults: %d/%d passed, %d failed\n",
                g_passed, g_passed + g_failed, g_failed);
    return g_failed == 0 ? 0 : 1;
}
