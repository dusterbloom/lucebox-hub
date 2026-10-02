#include "qwen4exp_backend.h"
#include "qwen4exp_graph.h"

#include "common/platform_env.h"
#include "common/sampler.h"

#include "ggml-cuda.h"

#if defined(LUCE_BACKEND_HIP) || defined(GGML_USE_HIP)
#include "common/gpu_runtime_compat.h"
#endif

#include <algorithm>
#include <chrono>
#include <cstddef>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <utility>
#include <vector>

namespace luce::common {

namespace {
bool enabled_env(const char * name) {
    const char * value = std::getenv(name);
    return value && std::atoi(value) != 0;
}

// gfx1151 prefill profile, qualified together on UD-Q4_K_XL, IQ4_NL and GSQ (long-prompt quality gate): MMB WMMA
// prefill, hipBLASLt on the bf16 weight shadow for the dense projections, bf16 HC activations, QSA sparse attention,
// and no automatic managed memory (it duplicates the weights). Must run before the backend reads any of them.
// An explicit value wins; QWEN4EXP_UPSTREAM keeps the reference configuration.
void apply_gfx1151_defaults(int gpu) {
#if defined(LUCE_BACKEND_HIP) || defined(GGML_USE_HIP)
    cudaDeviceProp prop{};
    if (enabled_env("QWEN4EXP_UPSTREAM") || cudaGetDeviceProperties(&prop, gpu) != cudaSuccess ||
        std::strncmp(prop.gcnArchName, "gfx1151", 7) != 0) return;
    static const char * const defaults[][2] = {
        {"LUCE_HIP_NO_AUTO_UMA", "1"}, {"GGML_CUDA_MMB", "1"}, {"QWEN4EXP_MMB_CUBLAS", "5"},
        {"LUCE_MMB_SHADOW", "1"}, {"LLAMA_MMB_HC16", "2"}, {"QWEN4EXP_QSA", "1"},
    };
    for (const auto & kv : defaults) set_environment_variable(kv[0], kv[1], false);
    std::fprintf(stderr, "[qwen4exp] gfx1151: prefill profile on (MMB, hipBLASLt shadow, HC16, QSA)\n");
#else
    (void) gpu;
#endif
}
}

Qwen4ExpBackend::Qwen4ExpBackend(Qwen4ExpBackendConfig cfg)
    : cfg_(std::move(cfg)) {}

Qwen4ExpBackend::~Qwen4ExpBackend() {
    shutdown();
}

bool Qwen4ExpBackend::init() {
    if (cfg_.device.is_layer_split()) {
        std::fprintf(stderr, "[qwen4exp] layer split is not supported yet\n");
        return false;
    }
    apply_gfx1151_defaults(cfg_.device.gpu);
    backend_ = ggml_backend_cuda_init(cfg_.device.gpu);
    if (!backend_) {
        std::fprintf(stderr, "[qwen4exp] backend init failed for GPU %d\n",
                     cfg_.device.gpu);
        return false;
    }
    if (!load_qwen4exp_gguf(cfg_.model_path, backend_, weights_)) {
        std::fprintf(stderr, "[qwen4exp] model load failed: %s\n",
                     luce_last_error());
        return false;
    }
    if (!create_qwen4exp_cache(backend_, weights_, cfg_.device.max_ctx,
                               GGML_TYPE_F16, cache_, /*mtp=*/true)) {
        std::fprintf(stderr, "[qwen4exp] cache creation failed\n");
        return false;
    }
    if (cfg_.max_concurrency > 1 && enabled_env("LUCE_QWEN4EXP_SEQ_ENGINE") &&
        enabled_env("QWEN4EXP_BATCHED_DECODE") &&
        !enabled_env("QWEN4EXP_UPSTREAM")) {
        if (cfg_.max_concurrency > 4 || cfg_.device.max_ctx != 32768) {
            std::fprintf(stderr, "[qwen4exp] sequence engine v1 requires <=4 slots at ctx=32768\n");
            return false;
        }
        seq_caches_.resize((size_t)cfg_.max_concurrency - 1);
        std::vector<Qwen4ExpCache *> caches;
        caches.reserve((size_t)cfg_.max_concurrency);
        caches.push_back(&cache_);
        for (Qwen4ExpCache & cache : seq_caches_) {
            if (!create_qwen4exp_cache(backend_, weights_, cfg_.device.max_ctx,
                                       GGML_TYPE_F16, cache)) {
                std::fprintf(stderr, "[qwen4exp] full-cache slot allocation failed\n");
                return false;
            }
            caches.push_back(&cache);
        }
        seq_engine_ = std::make_unique<Qwen4ExpSeqEngine>(
            backend_, weights_, std::move(caches), cfg_.device.max_ctx,
            std::min(cfg_.chunk, 512));
        std::fprintf(stderr,
            "[qwen4exp-seq] experimental independent-slot engine enabled: %d full F16 caches, ctx=%d\n",
            cfg_.max_concurrency, cfg_.device.max_ctx);
    }
    return true;
}

void Qwen4ExpBackend::print_ready_banner() const {
    const Qwen4ExpWeights & w = weights_;
    std::printf(
        "[qwen4exp-daemon] ready layers=%d linear=%d full=%d experts=%d/%d "
        "hc=%d/%d ple=%zu ngram=%d ctx=%d\n",
        w.n_layer,
        w.n_layer - w.n_layer / w.full_attention_interval,
        w.n_layer / w.full_attention_interval,
        w.n_expert_used, w.n_expert, w.n_hc, w.hc_lowrank,
        w.ple_layer_ids.size(), w.ple_ngram_size, cfg_.device.max_ctx);
    std::fflush(stdout);
}

bool Qwen4ExpBackend::park(ParkTarget target) {
    if (target != ParkTarget::TargetModel && target != ParkTarget::All) {
        return false;
    }
    if (parked_) return true;
    seq_engine_.reset();
    for (Qwen4ExpCache & cache : seq_caches_) free_qwen4exp_cache(cache);
    seq_caches_.clear();
    free_qwen4exp_cache(cache_);
    free_qwen4exp_weights(weights_);
    parked_ = true;
    std::printf("[qwen4exp] target parked\n");
    std::fflush(stdout);
    return true;
}

bool Qwen4ExpBackend::unpark(ParkTarget target) {
    if (target != ParkTarget::TargetModel && target != ParkTarget::All) {
        return false;
    }
    if (!parked_) return true;
    if (!load_qwen4exp_gguf(cfg_.model_path, backend_, weights_)) {
        std::fprintf(stderr, "[qwen4exp] unpark reload failed: %s\n",
                     luce_last_error());
        return false;
    }
    if (!create_qwen4exp_cache(backend_, weights_, cfg_.device.max_ctx,
                               GGML_TYPE_F16, cache_, /*mtp=*/true)) {
        std::fprintf(stderr, "[qwen4exp] unpark cache creation failed\n");
        free_qwen4exp_weights(weights_);
        return false;
    }
    if (cfg_.max_concurrency > 1 && enabled_env("LUCE_QWEN4EXP_SEQ_ENGINE") &&
        enabled_env("QWEN4EXP_BATCHED_DECODE") &&
        !enabled_env("QWEN4EXP_UPSTREAM")) {
        if (cfg_.max_concurrency > 4 || cfg_.device.max_ctx != 32768) {
            std::fprintf(stderr, "[qwen4exp] sequence engine v1 requires <=4 slots at ctx=32768\n");
            free_qwen4exp_cache(cache_);
            free_qwen4exp_weights(weights_);
            return false;
        }
        seq_caches_.resize((size_t)cfg_.max_concurrency - 1);
        std::vector<Qwen4ExpCache *> caches{&cache_};
        for (Qwen4ExpCache & cache : seq_caches_) {
            if (!create_qwen4exp_cache(backend_, weights_, cfg_.device.max_ctx,
                                       GGML_TYPE_F16, cache)) return false;
            caches.push_back(&cache);
        }
        seq_engine_ = std::make_unique<Qwen4ExpSeqEngine>(
            backend_, weights_, std::move(caches), cfg_.device.max_ctx,
            std::min(cfg_.chunk, 512));
    }
    parked_ = false;
    std::printf("[qwen4exp] target unparked\n");
    std::fflush(stdout);
    return true;
}

GenerateResult Qwen4ExpBackend::generate_impl(const GenerateRequest & req,
                                              const DaemonIO & io) {
    GenerateResult result;
    if (parked_) {
        result.fail(GenerateErrorCode::ModelParked, "qwen4exp target is parked");
        return result;
    }
    if (req.prompt.empty()) {
        result.fail(GenerateErrorCode::PrefillFailed, "empty prompt");
        return result;
    }

    std::vector<float> logits;
    const int chunk = std::max(1, cfg_.chunk);
    int pos = 0;

    // A generate() call starts a fresh sequence.
    reset_qwen4exp_state(backend_, cache_);

    // MTP speculation (sidecar loaded, default graph, a budget that leaves room for a draft): the draft head
    // predicts x_{p+2} from the pair (h_p, x_{p+1}), h_p being the trunk's final HC residual at p. Pairs at positions
    // [mtp_pos, ...) not yet run through the draft layer: mtp_h holds their hidden rows, mtp_tok the tokens known so
    // far (a pair's token arrives with the next forward).
    const bool spec = !req.force_ar_decode && req.n_gen > 2 && qwen4exp_verify_supported(cache_);
    const size_t hd = (size_t) weights_.n_embd * weights_.n_hc;
    std::vector<float> hidden, mtp_h, mtp_logits;
    std::vector<int32_t> mtp_tok;
    int mtp_pos = 0;

    const auto t_pre0 = std::chrono::steady_clock::now();
    for (size_t i = 0; i < req.prompt.size(); i += (size_t) chunk) {
        const int n = (int) std::min((size_t) chunk, req.prompt.size() - i);
        const Qwen4ExpForwardResult r = qwen4exp_forward(
            backend_, weights_, cache_, req.prompt.data() + i, n, pos, logits, spec ? &hidden : nullptr);
        if (!r.ok) {
            result.fail(GenerateErrorCode::PrefillFailed,
                        "qwen4exp prefill forward failed");
            return result;
        }
        if (spec) {   // this chunk's tokens complete every pending pair but the one of its own last row
            mtp_tok.assign(req.prompt.begin() + pos + (pos == 0 ? 1 : 0), req.prompt.begin() + pos + n);
            mtp_h.insert(mtp_h.end(), hidden.begin(), hidden.end());
            const int n_pairs = (int) mtp_tok.size();
            if (n_pairs > 0 && !qwen4exp_mtp_forward(backend_, weights_, cache_, mtp_tok.data(), mtp_h.data(),
                                                     n_pairs, mtp_pos, mtp_logits)) {
                result.fail(GenerateErrorCode::PrefillFailed, "qwen4exp MTP catch-up failed");
                return result;
            }
            mtp_h.erase(mtp_h.begin(), mtp_h.begin() + (std::ptrdiff_t) ((size_t) n_pairs * hd));
            mtp_pos += n_pairs;
        }
        pos += n;
        if (io.is_cancelled()) {
            result.fail(GenerateErrorCode::Cancelled, "cancelled during prefill");
            return result;
        }
    }
    const auto t_pre1 = std::chrono::steady_clock::now();
    result.prefill_s = std::chrono::duration<double>(t_pre1 - t_pre0).count();

    std::mt19937_64 rng(req.sampler.seed != 0 ? req.sampler.seed
                                              : std::random_device{}());
    std::vector<int32_t> history = req.prompt;
    auto sample = [&](const float * row) {
        return (int32_t) sample_logits(row, weights_.n_vocab, req.sampler, history, rng);
    };
    BudgetHookState budget;   // thinking force-close: keeps the reply reserve of the budget for the answer
    bool cancelled = false;
    // Commits a sampled token, after the budget hook's substitution; false once generation ends.
    auto commit = [&](int32_t & tok) {
        if (budget.apply(req.budget_hook, (int) result.tokens.size(), req.n_gen, tok)) result.budget_forced_close = true;
        result.tokens.push_back(tok);
        io.emit(tok);
        if (io.is_cancelled()) {
            cancelled = true;
            return false;
        }
        return tok != weights_.eos_id && tok != weights_.eos_chat_id && (int) result.tokens.size() < req.n_gen;
    };

    // Sample trunk rows lazily, updating history and applying the budget hook
    // exactly once per emitted token. No RNG draws for unvisited verify rows.
    long long drafts = 0, accepted = 0, steps = 0;
    double draft_s = 0.0;
    const auto t_dec0 = std::chrono::steady_clock::now();
    int32_t next = sample(logits.data());
    bool more = req.n_gen > 0 && commit(next);
    if (spec) mtp_tok.assign(1, next);
    std::vector<int32_t> draft_tokens;
    while (more) {
        const int k = spec ? std::max(0, std::min({cache_.mtp_draft,
            req.n_gen - (int) result.tokens.size() - 1, cache_.max_ctx - pos - 1})) : 0;
        const bool verify = k > 0;
        if (verify) {
            const auto td0 = std::chrono::steady_clock::now();
            if (!qwen4exp_mtp_draft(backend_, weights_, cache_, mtp_tok.data(), mtp_h.data(), (int) mtp_tok.size(),
                                    mtp_pos, k, draft_tokens)) {
                result.fail(GenerateErrorCode::DecodeFailed, "qwen4exp MTP draft failed");
                return result;
            }
            draft_s += std::chrono::duration<double>(std::chrono::steady_clock::now() - td0).count();
        }
        std::array<int32_t, QWEN4EXP_MTP_MAX_VERIFY> in{}, samples{};
        in[0] = next;
        if (verify) std::copy(draft_tokens.begin(), draft_tokens.end(), in.begin() + 1);
        const Qwen4ExpForwardResult r = qwen4exp_forward(
            backend_, weights_, cache_, in.data(), k + 1, pos, logits, verify ? &hidden : nullptr, verify);
        if (!r.ok) {
            result.fail(GenerateErrorCode::DecodeFailed, "qwen4exp decode forward failed");
            return result;
        }
        ++steps;
        drafts += k;
        Qwen4ExpMtpAcceptance decision;
        for (int i = 0; i <= k; ++i) {
            history.push_back(in[i]);
            int32_t tok = sample(logits.data() + (size_t) i * weights_.n_vocab);
            more = commit(tok);
            samples[i] = tok;
            decision = qwen4exp_mtp_accept(draft_tokens.data(), k, samples.data(), i + 1);
            if (!more || decision.n_accepted != i + 1) break;
        }
        const int retained = decision.n_emitted;
        next = decision.emitted[retained - 1];
        accepted += decision.n_accepted;
        if (verify) {
            // Replace every predicted MTP hidden/KV row with the corresponding
            // trunk pair on next catch-up, including after a partial accept.
            mtp_h.assign(hidden.begin(), hidden.begin() + (std::ptrdiff_t) ((size_t) retained * hd));
            mtp_tok.assign(decision.emitted.begin(), decision.emitted.begin() + retained);
            mtp_pos = pos;
            if (!qwen4exp_verify_rollback(backend_, weights_, cache_, pos, retained)) {
                result.fail(GenerateErrorCode::DecodeFailed, "qwen4exp verify rollback failed");
                return result;
            }
        }
        pos += retained;
    }
    if (cancelled) {
        result.fail(GenerateErrorCode::Cancelled, "cancelled during decode");
        return result;
    }
    const auto t_dec1 = std::chrono::steady_clock::now();
    result.decode_s = std::chrono::duration<double>(t_dec1 - t_dec0).count();
    result.spec_decode_ran = drafts > 0;
    result.accept_rate = drafts > 0 ? (float) accepted / (float) drafts : 0.0f;
    if (drafts > 0) {
        const double decoded = (double) result.tokens.size() - 1.0;   // the first token came from the prefill
        std::fprintf(stderr,
            "[qwen4exp-mtp] k=%d drafts=%lld accepted=%lld rate=%.3f tokens_per_step=%.3f draft_ms=%.2f decode=%.2f tok/s\n",
            cache_.mtp_draft, drafts, accepted, (double) accepted / (double) drafts, steps > 0 ? decoded / (double) steps : 0.0,
            1e3 * draft_s / (double) drafts, result.decode_s > 0.0 ? decoded / result.decode_s : 0.0);
    }

    result.succeed();
    return result;
}

bool Qwen4ExpBackend::snapshot_save(int slot) {
    (void) slot;
    return false;
}

void Qwen4ExpBackend::snapshot_free(int slot) {
    (void) slot;
}

bool Qwen4ExpBackend::snapshot_used(int slot) const {
    (void) slot;
    return false;
}

int Qwen4ExpBackend::snapshot_cur_pos(int slot) const {
    (void) slot;
    return 0;
}

GenerateResult Qwen4ExpBackend::restore_and_generate_impl(
        int slot, const GenerateRequest & req, const DaemonIO & io) {
    (void) slot;
    (void) req;
    (void) io;
    GenerateResult result;
    result.fail(GenerateErrorCode::InvalidSnapshotSlot,
                "qwen4exp snapshots are not implemented yet");
    return result;
}

bool Qwen4ExpBackend::handle_compress(const std::string & line,
                                      const DaemonIO & io) {
    (void) line;
    (void) io;
    return false;
}

void Qwen4ExpBackend::free_drafter() {}

void Qwen4ExpBackend::shutdown() {
    seq_engine_.reset();
    for (Qwen4ExpCache & cache : seq_caches_) free_qwen4exp_cache(cache);
    seq_caches_.clear();
    free_qwen4exp_cache(cache_);
    free_qwen4exp_weights(weights_);
    if (backend_) {
        ggml_backend_free(backend_);
        backend_ = nullptr;
    }
}

}  // namespace luce::common
