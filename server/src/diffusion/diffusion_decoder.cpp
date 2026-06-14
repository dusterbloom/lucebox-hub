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

bool resolve_prefix(DiffusionModelGraph &        model,
                    const std::vector<int32_t> & prompt,
                    int                          prepared_prefix_len,
                    const char *                 error_prefix,
                    int &                        out_prefix_len,
                    DiffusionDecodeResult &      res) {
    if (prepared_prefix_len >= 0) {
        if (prepared_prefix_len > (int)prompt.size()) {
            res.error = (error_prefix && error_prefix[0] != '\0')
                      ? (std::string(error_prefix) + ": prepared prefix longer than prompt")
                      : "prepared prefix longer than prompt";
            return false;
        }
        out_prefix_len = prepared_prefix_len;
        return true;
    }

    if (!model.prepare(prompt, out_prefix_len)) {
        res.error = (error_prefix && error_prefix[0] != '\0')
                  ? (std::string(error_prefix) + ": prepare")
                  : "prepare";
        return false;
    }
    if (out_prefix_len < 0 || out_prefix_len > (int)prompt.size()) {
        out_prefix_len = (int)prompt.size();
    }
    return true;
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
    const DiffusionStream &      stream,
    int                          prepared_prefix_len) {

    DiffusionDecodeResult res;

    const int vocab = model.vocab();
    if (vocab <= 0) { res.error = "eb: model vocab() must be > 0"; return res; }
    if (n_gen <= 0) { res.ok = true; return res; }

    int prefix_len = 0;
    if (!resolve_prefix(model, prompt, prepared_prefix_len, "eb",
                        prefix_len, res)) {
        return res;
    }

    const int P0 = prefix_len;          // prompt prefix length (fixed)
    const int S  = std::max(1, cfg.eb_max_steps);
    // Schedule horizon: temperature decays t_max→t_min over this many steps,
    // then clamps at t_min. Independent of the hard cap S so a large S does
    // not thin out the schedule and delay convergence.
    const int SH = std::max(1, cfg.eb_schedule_steps);

    // Canvas-block cap. The full-canvas EB path (C = n_gen) falls off a step-time
    // cliff above C≈1024 (swept 2026-06-15: 187ms/step @512 → 6457ms @2048),
    // emits empty content at C≥1024, and aborts under load. Denoising in ≤cap blocks
    // with KV-prefix reuse keeps every forward in the linear regime (~168-180 tok/s)
    // and fixes all three. ponytail: env-tunable; 512 is the swept sweet spot.
    const char * cap_env   = std::getenv("DG_EB_BLOCK_CAP");
    const int    BLOCK_CAP = (cap_env && std::atoi(cap_env) > 0) ? std::atoi(cap_env) : 512;
    const bool   use_l2    = cfg.enable_l2_interblock && (std::getenv("DG_NO_L2") == nullptr);

    std::mt19937                            rng(seed);
    std::uniform_int_distribution<int32_t> vocab_dist(0, vocab - 1);
    std::uniform_real_distribution<float>  uni01(0.0f, 1.0f);

    // Canvas grows one committed block at a time; starts as the prompt prefix.
    std::vector<int32_t> canvas(prompt.begin(), prompt.begin() + P0);

    int  committed = P0;
    int  emitted   = 0;
    bool stop      = false;

    while (emitted < n_gen && !stop) {
        const int block_len   = std::min(BLOCK_CAP, n_gen - emitted);
        const int block_begin = committed;

        // L2′: restore the KV snapshot so the model sees only the committed prefix as
        // cached KV — each forward then processes just block_len positions (off the cliff).
        if (use_l2 && block_begin > P0) model.on_block_starting(block_begin);

        // Seed this block's canvas region with uniform-random noise.
        canvas.resize((size_t)block_begin + block_len);
        for (int i = 0; i < block_len; ++i) canvas[block_begin + i] = vocab_dist(rng);

        // Per-block EB state (sized to block_len).
        std::vector<float>   sc_buffer;
        std::vector<int32_t> argmax_canvas(block_len, 0);
        std::vector<int32_t> prev_argmax(block_len, -1);   // -1 => step 0 always differs
        std::vector<float>   entropy_vec(block_len, 0.0f);
        std::vector<int32_t> denoiser(block_len, 0);
        std::vector<int32_t> order(block_len);
        float prev_temp_inv = 1.0f;
        int   held          = 0;
        bool  finished      = false;

        // ref diffusion.cpp:522: loop cur_step from S down to 1
        for (int cur_step = S; cur_step >= 1 && !finished; --cur_step) {
            const int   step_idx = S - cur_step;  // 0-based index (ref:523)
            const float sched_frac = (SH <= 1)
                ? 1.0f
                : std::min(1.0f, (float)step_idx / (float)(SH - 1));
            const float t        = cfg.eb_t_max - (cfg.eb_t_max - cfg.eb_t_min) * sched_frac;
            const float temp_inv = 1.0f / t;

            const float * sc_ptr = (step_idx == 0 || sc_buffer.empty())
                                 ? nullptr : sc_buffer.data();
            model.set_sc(sc_ptr, step_idx == 0 ? 0.0f : 1.0f, prev_temp_inv);

            // ref diffusion.cpp:672-708: pre-draw step randomness for reproducibility
            std::vector<float>   u(block_len);
            std::vector<int32_t> renoise(block_len);
            for (int pos = 0; pos < block_len; ++pos) {
                u[pos]       = uni01(rng);
                renoise[pos] = vocab_dist(rng);
            }

            // forward_block_dev runs forward + on-GPU sampling; with the committed
            // prefix served from cached KV it processes only block_len positions.
            DiffusionModelGraph::DevSampleResult sample_res;
            if (!model.forward_block_dev(canvas, block_begin, block_len, /*bidirectional=*/true,
                                         u, temp_inv, sample_res)) {
                res.error = "eb: forward_block_dev";
                return res;
            }
            res.stats.forward_passes++;

            for (int pos = 0; pos < block_len; ++pos) {
                entropy_vec[pos]   = sample_res.entropy[pos];
                argmax_canvas[pos] = sample_res.argmax[pos];
                denoiser[pos]      = sample_res.sampled[pos];
            }
            if (!sample_res.logits.empty()) sc_buffer = std::move(sample_res.logits);

            // ref diffusion.cpp:739-747: accept ascending-entropy while cumulative
            // entropy of strictly-prior accepted <= entropy_bound.
            std::iota(order.begin(), order.end(), 0);
            std::sort(order.begin(), order.end(),
                      [&](int a, int b) { return entropy_vec[a] < entropy_vec[b]; });

            std::vector<char> accepted(block_len, 0);
            double cumE = 0.0;
            for (int k = 0; k < block_len; ++k) {
                const int pos = order[k];
                if (cumE <= (double)cfg.eb_entropy_bound) { accepted[pos] = 1; }
                cumE += (double)entropy_vec[pos];
            }

            // ref diffusion.cpp:750-755: renoise rejected, output is argmax canvas.
            float entropy_sum = 0.0f;
            for (int pos = 0; pos < block_len; ++pos) {
                canvas[(size_t)block_begin + pos] = accepted[pos] ? denoiser[pos] : renoise[pos];
                entropy_sum += entropy_vec[pos];
            }

            // ref diffusion.cpp:757-760: adaptive stop within this block.
            const bool same = (prev_argmax == argmax_canvas);
            held = same ? held + 1 : 0;
            const float mean_entropy = entropy_sum / (float)block_len;
            if (held >= cfg.eb_stability_threshold &&
                mean_entropy < cfg.eb_confidence_threshold) {
                finished = true;
            }
            prev_argmax   = argmax_canvas;
            prev_temp_inv = temp_inv;
        }

        // Commit the block's argmax left-to-right; stream tokens (callback handles EOS/length).
        for (int pos = 0; pos < block_len; ++pos) {
            const int32_t tok = argmax_canvas[pos];
            canvas[(size_t)block_begin + pos] = tok;
            res.tokens.push_back(tok);
            ++emitted;
            if (stream.on_token && !stream.on_token(tok)) { stop = true; break; }
        }
        committed += block_len;
        res.stats.blocks++;

        // L2′: prefill + snapshot KV at `committed` so the next block forwards only its new tokens.
        if (use_l2 && !stop) model.on_block_committed(canvas, committed);
    }

    res.stats.tokens = (int)res.tokens.size();
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
    const DiffusionStream &      stream,
    int                          prepared_prefix_len) {

    DiffusionDecodeResult res;

    // ── EntropyBound path ─────────────────────────────────────────────
    if (cfg_in.remasking == DiffusionRemask::EntropyBound) {
        const uint64_t seed = cfg_in.seed ? cfg_in.seed
                            : (sampler.seed ? sampler.seed : 0x9E3779B97F4A7C15ULL);
        return run_eb_generate(model, prompt, n_gen, cfg_in, seed, stream,
                               prepared_prefix_len);
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
    if (!resolve_prefix(model, prompt, prepared_prefix_len, "",
                        prefix_len, res)) {
        return res;
    }

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
    // L2′ inter-block snapshot: enabled by cfg.enable_l2_interblock (default true)
    // unless the environment variable DG_NO_L2=1 overrides it.
    // When enabled, after each committed block the model saves a KV snapshot;
    // the next block restores it so it only forwards C new tokens (not P+b*C).
    const bool use_l2 = cfg.enable_l2_interblock &&
                        (std::getenv("DG_NO_L2") == nullptr);
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

        // L2′: before block b+1, restore the snapshot saved after block b so the
        // model sees only the committed prefix as its cached KV.  The first block
        // (block_begin == prefix_len) never needs a restore — there is no prior block.
        if (use_l2 && block_begin > prefix_len) {
            model.on_block_starting(block_begin);
        }

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

        // L2′: after committing, prefill+snapshot the KV at position `committed`
        // so the next block only forwards its C new tokens.
        if (use_l2 && !stop) {
            model.on_block_committed(canvas, committed);
        }
    }

    res.stats.tokens = (int)res.tokens.size();
    res.ok = true;
    return res;
}

}  // namespace dflash::common
