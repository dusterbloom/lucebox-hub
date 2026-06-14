// Model-agnostic diffusion decode loop. See diffusion_decoder.h.

#include "diffusion_decoder.h"

#include <algorithm>
#include <cmath>
#include <numeric>
#include <random>
#include <utility>
#include <vector>

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

// ── Entropy-bound denoiser (DiffusionGemma-style) ──────────────────────────
// Implements diffusion_generate_entropy_bound from the reference
// (diffusion.cpp:442-672). Single canvas, uniform-random init, linear temp
// schedule, accept by sorted Shannon entropy within MI bound. SC threaded via
// model.set_sc() before each forward_block call.
static DiffusionDecodeResult run_eb_generate(
    DiffusionModelGraph &        model,
    const std::vector<int32_t> & prompt,
    int                          n_gen,
    const DiffusionConfig &      cfg,
    uint64_t                     seed,
    const DiffusionStream &      stream) {

    DiffusionDecodeResult res;

    const int vocab = model.vocab();
    if (vocab <= 0) { res.error = "eb: model vocab() must be > 0"; return res; }
    if (n_gen <= 0) { res.ok = true; return res; }

    // Prepare: record prefix length (no KV prefill in Phase 2 — full recompute).
    int prefix_len = 0;
    if (!model.prepare(prompt, prefix_len)) { res.error = "eb: prepare"; return res; }
    if (prefix_len < 0 || prefix_len > (int)prompt.size())
        prefix_len = (int)prompt.size();

    const int C = n_gen;               // canvas length
    const int P = prefix_len;          // prompt prefix length
    const int S = std::max(1, cfg.eb_max_steps);

    // Full canvas: [prompt | canvas_tokens]; canvas positions [P, P+C)
    std::vector<int32_t> canvas(prompt.begin(), prompt.begin() + P);
    canvas.resize((size_t)P + C);

    // ref diffusion.cpp:469-473: init canvas to uniform random vocab tokens
    std::mt19937                            rng(seed);
    std::uniform_int_distribution<int32_t> vocab_dist(0, vocab - 1);
    std::uniform_real_distribution<float>  uni01(0.0f, 1.0f);
    for (int i = 0; i < C; ++i) canvas[(size_t)P + i] = vocab_dist(rng);

    // Buffers for per-step processing.
    // sc_buffer holds prev-step raw logits for the CPU-fallback SC path.
    // The CUDA override keeps SC device-resident and ignores sc_buffer.
    std::vector<float>   sc_buffer;  // populated by CPU fallback; empty for CUDA path
    std::vector<int32_t> argmax_canvas(C, 0);
    std::vector<int32_t> prev_argmax(C, -1);   // -1 => step 0 always differs
    std::vector<float>   entropy_vec(C, 0.0f);
    std::vector<int32_t> denoiser(C, 0);       // multinomial-sampled token per pos
    std::vector<int32_t> order(C);             // entropy sort order

    float prev_temp_inv = 1.0f;
    int   held          = 0;
    bool  finished      = false;

    // ref diffusion.cpp:522: loop cur_step from S down to 1
    for (int cur_step = S; cur_step >= 1 && !finished; --cur_step) {
        const int   step_idx = S - cur_step;  // 0-based index (ref:523)
        // ref diffusion.cpp:524: t = t_min + (t_max - t_min) * (cur_step / S)
        const float t        = cfg.eb_t_min +
                               (cfg.eb_t_max - cfg.eb_t_min) *
                               ((float)cur_step / (float)S);
        const float temp_inv = 1.0f / t;

        // ref diffusion.cpp:550-551: SC gate and prev-step temp_inv
        // sc_use=0 on step 0 (no prior step), sc_use=1 thereafter.
        // For the GPU path (DiffusionGemmaGraph CUDA override), set_sc() records
        // the sc_use/temp_inv scalars; the SC device buffer is handled internally.
        // For the CPU fallback path, sc_buffer.data() supplies the host logits.
        const float * sc_ptr = (step_idx == 0 || sc_buffer.empty())
                             ? nullptr : sc_buffer.data();
        model.set_sc(sc_ptr, step_idx == 0 ? 0.0f : 1.0f, prev_temp_inv);

        // ref diffusion.cpp:672-708: pre-draw step randomness for reproducibility
        std::vector<float>   u(C);
        std::vector<int32_t> renoise(C);
        for (int pos = 0; pos < C; ++pos) {
            u[pos]       = uni01(rng);
            renoise[pos] = vocab_dist(rng);
        }

        // forward_block_dev: runs the forward pass AND sampling in one call.
        // The CUDA override (DiffusionGemmaGraph) keeps logits device-resident,
        // runs the multinomial+entropy kernel on GPU, and copies only ~3 KB to host.
        // The default fallback calls forward_block then CPU-samples from host logits.
        DiffusionModelGraph::DevSampleResult sample_res;
        if (!model.forward_block_dev(canvas, P, C, /*bidirectional=*/true,
                                     u, temp_inv, sample_res)) {
            res.error = "eb: forward_block_dev";
            return res;
        }
        res.stats.forward_passes++;

        for (int pos = 0; pos < C; ++pos) {
            entropy_vec[pos]   = sample_res.entropy[pos];
            argmax_canvas[pos] = sample_res.argmax[pos];
            denoiser[pos]      = sample_res.sampled[pos];
        }

        // SC host buffer: the CPU fallback populates sample_res.logits with the
        // full [C*vocab] F32 logits; copy to sc_buffer for next step's set_sc().
        // The CUDA override leaves sample_res.logits empty (SC is device-resident).
        if (!sample_res.logits.empty()) {
            sc_buffer = std::move(sample_res.logits);
        }

        // ref diffusion.cpp:739-747: acceptance — sort positions ascending by entropy,
        // accept while cumulative entropy of STRICTLY-PRIOR accepted <= entropy_bound
        std::iota(order.begin(), order.end(), 0);
        std::sort(order.begin(), order.end(),
                  [&](int a, int b) { return entropy_vec[a] < entropy_vec[b]; });

        std::vector<char> accepted(C, 0);
        double cumE = 0.0;
        for (int k = 0; k < C; ++k) {
            const int pos = order[k];
            // Accept if cumulative entropy BEFORE this position <= bound
            // (ref diffusion.cpp:745-746: cumE - entropy[pos] <= bound)
            if (cumE <= (double)cfg.eb_entropy_bound) { accepted[pos] = 1; }
            cumE += (double)entropy_vec[pos];
        }

        // ref diffusion.cpp:750-755: renoise rejected, output is argmax canvas
        float entropy_sum = 0.0f;
        for (int pos = 0; pos < C; ++pos) {
            canvas[(size_t)P + pos] = accepted[pos] ? denoiser[pos] : renoise[pos];
            entropy_sum += entropy_vec[pos];
        }

        // ref diffusion.cpp:757-760: adaptive stop
        // argmax stable for stability_threshold consecutive steps AND mean entropy low
        const bool same = (prev_argmax == argmax_canvas);
        held = same ? held + 1 : 0;
        const float mean_entropy = entropy_sum / (float)C;
        if (held >= cfg.eb_stability_threshold &&
            mean_entropy < cfg.eb_confidence_threshold) {
            finished = true;
        }
        prev_argmax   = argmax_canvas;
        prev_temp_inv = temp_inv;

        // Step callback (via stream) — not applicable for plain stream sink;
        // stream is invoked only when committing the final canvas.
    }

    // Commit the argmax canvas as output
    res.tokens = std::vector<int32_t>(argmax_canvas.begin(), argmax_canvas.end());
    res.stats.blocks  = 1;
    res.stats.tokens  = C;

    for (int pos = 0; pos < C; ++pos) {
        if (stream.on_token && !stream.on_token(argmax_canvas[pos])) break;
    }

    res.ok = true;
    return res;
}

DiffusionDecodeResult run_diffusion_generate(
    DiffusionModelGraph &        model,
    const std::vector<int32_t> & prompt,
    int                          n_gen,
    const DiffusionConfig &      cfg_in,
    const SamplerCfg &           sampler,
    bool                         do_sample,
    const DiffusionStream &      stream) {

    DiffusionDecodeResult res;

    // ── EntropyBound path ─────────────────────────────────────────────
    if (cfg_in.remasking == DiffusionRemask::EntropyBound) {
        const uint64_t seed = cfg_in.seed ? cfg_in.seed
                            : (sampler.seed ? sampler.seed : 0x9E3779B97F4A7C15ULL);
        return run_eb_generate(model, prompt, n_gen, cfg_in, seed, stream);
    }

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
