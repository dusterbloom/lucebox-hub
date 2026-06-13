// Model-agnostic diffusion decode loop. See diffusion_decoder.h.

#include "diffusion_decoder.h"

#include <algorithm>
#include <cmath>
#include <random>
#include <utility>

namespace dflash::common {

namespace {

// Numerically-stable softmax: returns the argmax token and its probability.
std::pair<int32_t, float> softmax_argmax_prob(const float * row, int vocab) {
    int   best     = 0;
    float maxlogit = row[0];
    for (int i = 1; i < vocab; ++i) {
        if (row[i] > maxlogit) { maxlogit = row[i]; best = i; }
    }
    double denom = 0.0;
    for (int i = 0; i < vocab; ++i) denom += std::exp((double)row[i] - (double)maxlogit);
    // The argmax term contributes exp(0)=1, so prob == 1/denom.
    const float prob = denom > 0.0 ? (float)(1.0 / denom) : 1.0f;
    return { (int32_t)best, prob };
}

// Positions that should be finalized after completing 0-indexed denoising step
// `s` of `steps`, under a linear schedule.
int target_finalized_after(int s, int steps, int block_len) {
    if (steps <= 1) return block_len;
    int t = (int)std::llround((double)block_len * (double)(s + 1) / (double)steps);
    if (t < 0)         t = 0;
    if (t > block_len) t = block_len;
    return t;
}

}  // namespace

DiffusionDecodeResult run_diffusion_generate(
    DiffusionModelGraph &        model,
    const std::vector<int32_t> & prompt,
    int                          n_gen,
    const DiffusionConfig &      cfg_in,
    const SamplerCfg &           sampler,
    bool                         do_sample,
    const DiffusionStream &      stream) {

    DiffusionDecodeResult res;

    // ── Resolve / validate config ────────────────────────────────────
    DiffusionConfig cfg = cfg_in;
    if (cfg.block_size <= 0) { res.error = "config: block_size must be > 0"; return res; }
    if (cfg.n_steps <= 0)    cfg.n_steps = cfg.block_size;
    if (cfg.noise_scheme == DiffusionNoise::Masked && cfg.mask_token_id < 0) {
        cfg.mask_token_id = model.mask_token();
        if (cfg.mask_token_id < 0) {
            res.error = "config: Masked scheme requires a mask token id";
            return res;
        }
    }

    const int     vocab = model.vocab();
    const int32_t eos   = model.eos_token();
    if (vocab <= 0) { res.error = "config: model vocab() must be > 0"; return res; }
    if (n_gen <= 0) { res.ok = true; return res; }  // nothing to generate

    int prefix_len = 0;
    if (!model.prepare(prompt, prefix_len)) { res.error = "prepare"; return res; }
    if (prefix_len < 0 || prefix_len > (int)prompt.size()) prefix_len = (int)prompt.size();

    // Canvas = prompt prefix, grown one committed block at a time.
    std::vector<int32_t> canvas(prompt.begin(), prompt.begin() + prefix_len);
    int committed = prefix_len;

    // History fed to the sampler's penalty chain (committed tokens only).
    std::vector<int32_t> history = canvas;

    const uint64_t seed = cfg.seed ? cfg.seed
                        : (sampler.seed ? sampler.seed : 0x9E3779B97F4A7C15ULL);
    std::mt19937_64 rng(seed);
    auto noise_token = [&]() -> int32_t {
        if (cfg.noise_scheme == DiffusionNoise::Masked) return cfg.mask_token_id;
        std::uniform_int_distribution<int> d(0, vocab - 1);
        return (int32_t)d(rng);
    };

    const int n_ctx = model.n_ctx_max();
    int  emitted = 0;
    bool stop    = false;

    // Scratch reused across steps/blocks.
    std::vector<float>   logits;
    std::vector<int32_t> chosen(cfg.block_size);
    std::vector<float>   conf(cfg.block_size);
    std::vector<char>    finalized(cfg.block_size);

    while (emitted < n_gen && !stop) {
        const int block_len   = std::min(cfg.block_size, n_gen - emitted);
        const int block_begin = committed;
        if (n_ctx > 0 && block_begin + block_len > n_ctx) break;  // context exhausted

        // Seed the block with noise.
        canvas.resize((size_t)block_begin + block_len);
        for (int j = 0; j < block_len; ++j) canvas[block_begin + j] = noise_token();

        std::fill(finalized.begin(), finalized.begin() + block_len, (char)0);
        int       n_final = 0;
        const int steps   = cfg.n_steps;

        for (int s = 0; s < steps && n_final < block_len; ++s) {
            if (!model.forward_block(canvas, block_begin, block_len,
                                     /*bidirectional=*/true, logits)) {
                res.error = "forward";
                return res;
            }
            res.stats.forward_passes++;
            if ((int)logits.size() < block_len * vocab) {
                res.error = "forward: short logits";
                return res;
            }

            // Per-position greedy confidence + provisional token.
            for (int j = 0; j < block_len; ++j) {
                if (finalized[j]) continue;
                const float * row = logits.data() + (size_t)j * vocab;
                std::pair<int32_t, float> ap = softmax_argmax_prob(row, vocab);
                conf[j]   = ap.second;
                chosen[j] = do_sample
                          ? (int32_t)sample_logits(row, vocab, sampler, history, rng)
                          : ap.first;
            }

            // Choose which unfinalized positions to finalize this step.
            const bool       last_step = (s == steps - 1);
            std::vector<int> to_finalize;
            if (cfg.remasking == DiffusionRemask::ParallelThreshold && !last_step) {
                for (int j = 0; j < block_len; ++j)
                    if (!finalized[j] && conf[j] >= cfg.confidence_threshold)
                        to_finalize.push_back(j);
                if (to_finalize.empty()) {  // guarantee progress: most-confident position
                    int best = -1; float bp = -1.0f;
                    for (int j = 0; j < block_len; ++j)
                        if (!finalized[j] && conf[j] > bp) { bp = conf[j]; best = j; }
                    if (best >= 0) to_finalize.push_back(best);
                }
            } else if (last_step) {
                for (int j = 0; j < block_len; ++j)
                    if (!finalized[j]) to_finalize.push_back(j);
            } else {
                const int target = target_finalized_after(s, steps, block_len);
                int       k      = std::max(1, target - n_final);
                k = std::min(k, block_len - n_final);
                std::vector<int> cand;
                for (int j = 0; j < block_len; ++j) if (!finalized[j]) cand.push_back(j);
                if (cfg.remasking == DiffusionRemask::Random) {
                    std::shuffle(cand.begin(), cand.end(), rng);
                } else {  // LowConfidence: finalize the most confident first
                    std::sort(cand.begin(), cand.end(),
                              [&](int a, int b) { return conf[a] > conf[b]; });
                }
                for (int i = 0; i < k && i < (int)cand.size(); ++i)
                    to_finalize.push_back(cand[i]);
            }

            for (int j : to_finalize) {
                canvas[block_begin + j] = chosen[j];
                finalized[j] = 1;
                ++n_final;
            }
            // Unfinalized positions are re-noised for the next step.
            for (int j = 0; j < block_len; ++j)
                if (!finalized[j]) canvas[block_begin + j] = noise_token();
        }

        // Commit + stream the block left-to-right.
        for (int j = 0; j < block_len; ++j) {
            const int32_t tok = canvas[block_begin + j];
            res.tokens.push_back(tok);
            history.push_back(tok);
            ++emitted;
            if (stream.on_token && !stream.on_token(tok)) { stop = true; break; }
            if (tok == eos) { stop = true; break; }
        }
        committed += block_len;
        res.stats.blocks++;
    }

    res.stats.tokens = (int)res.tokens.size();
    res.ok = true;
    return res;
}

}  // namespace dflash::common
