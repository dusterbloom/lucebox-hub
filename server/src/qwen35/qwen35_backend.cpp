#include "qwen35_backend.h"
#include "qwen35_image_request.h"
#include "concurrency/qwen35_seq_engine.h"
#include "common/chain_rollback_policy.h"
#include "common/adaptive_spec_width.h"
#include "common/spec_acceptance.h"
#include "common/draft_block_size.h"
#include "common/draft_swa.h"
#include "placement/skip_park_guard.h"
#include "qwen35_dflash_target.h"
#include "graph_builders.h"
#include "dflash_feature_ring.h"
#include "dflash_capture.h"
#include "common/dflash_draft_graph.h"
#include "peer_access.h"
#include "attn_masks.h"
#include "prefill_helpers.h"
#include "common/sampler.h"
#ifdef LUCE_HAVE_GPU_SAMPLER
#include "common/geometric_sampler_cuda.h"
#include <random>
#endif
#include "common/dspark_head.h"
#include "common/dflash2_head.h"
#include "common/io_utils.h"
#include "common/restore_delta.h"
#include "common/specla_mode.h"
#include "qwen35_tensor_parallel.h"
#include "pflash/pflash_drafter.h"
#include "pflash/pflash_compress.h"
#include "pflash/kvflash_drafter_scorer.h"
#include "common/logits_dump.h"
#include "common/kvflash_prefill_reselect.h"
#include "common/kvflash_exact_mass.h"
#include "common/kvflash_exact_fast.h"
#include "common/kvflash_gpu_mass.h"
#include "common/kvflash_slot_pager.h"

#include "ggml-cuda.h"
#include "ggml-backend-impl.h"
#include "common/snapshot_backend.h"
#include "pflash_ggml_adapter.h"
#include "flashprefill.h"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <climits>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <utility>
#if !defined(_WIN32)
#include <fcntl.h>
#include <sys/stat.h>
#include <unistd.h>
#endif

#include "kv_quant.h"

namespace luce::common {

namespace {
static float bf16_bits_to_f32(uint16_t bits) {
    union {
        uint32_t u;
        float f;
    } v;
    v.u = (uint32_t)bits << 16;
    return v.f;
}

static bool tokens_contain_recent_sequence(const std::vector<int32_t> & tokens,
                                           const std::vector<int32_t> & needle,
                                           size_t max_trailing) {
    if (needle.empty() || tokens.size() < needle.size()) return false;
    const size_t last_end = tokens.size();
    const size_t first_end = std::max(
        needle.size(),
        last_end > max_trailing ? last_end - max_trailing : needle.size());
    for (size_t end = first_end; end <= last_end; ++end) {
        const size_t start = end - needle.size();
        if (std::equal(needle.begin(), needle.end(), tokens.begin() + start)) {
            return true;
        }
    }
    return false;
}

static bool tokens_have_recent_any(const std::vector<int32_t> & tokens,
                                   const std::vector<int32_t> & candidates,
                                   size_t max_trailing) {
    if (tokens.empty() || candidates.empty()) return false;
    for (size_t trailing = 0; trailing <= max_trailing; ++trailing) {
        if (tokens.size() <= trailing) break;
        const int32_t tok = tokens[tokens.size() - 1 - trailing];
        if (std::find(candidates.begin(), candidates.end(), tok) != candidates.end()) {
            return true;
        }
    }
    return false;
}

static int env_int_or_default(const char * name, int fallback) {
    if (const char * raw = std::getenv(name)) {
        if (*raw) return std::atoi(raw);
    }
    return fallback;
}

static void configure_concurrent_hipblaslt_default(
        const Qwen35Config & cfg) {
#if (defined(LUCE_BACKEND_HIP) || defined(GGML_USE_HIP)) && !defined(_WIN32)
    if (!cfg.paged_attention || cfg.max_concurrency <= 1 ||
        std::getenv("ROCBLAS_USE_HIPBLASLT") != nullptr) {
        return;
    }

    cudaDeviceProp prop{};
    if (cudaGetDeviceProperties(&prop, cfg.device.gpu) != cudaSuccess ||
        std::strncmp(prop.gcnArchName, "gfx1151", 7) != 0) {
        return;
    }

    // The concurrent Qwen workload spends most of its time in wide Q4_K
    // dequantize + GEMM steps. ROCm 7.2's hipBLASLt route is consistently
    // faster for those gfx1151 shapes. Preserve an explicit environment value
    // so ROCBLAS_USE_HIPBLASLT=0 remains an opt-out.
    if (::setenv("ROCBLAS_USE_HIPBLASLT", "1", 0) == 0) {
        std::fprintf(
            stderr,
            "[qwen35] gfx1151 concurrent mode: enabled rocBLAS hipBLASLt "
            "(set ROCBLAS_USE_HIPBLASLT=0 to disable)\n");
    }
#else
    (void)cfg;
#endif
}

static int dflash_min_tokens_floor() {
    static const int value = env_int_or_default("LUCE_MIN_TOKENS", 0);
    return value;
}

static FILE * open_dflash_floor_log() {
#if defined(_WIN32)
    // Simple append-mode log on Windows (no file size check).
    return std::fopen("dflash_floor.log", "a");
#else
    static constexpr const char * kPath = "/tmp/luce_floor.log";
    static constexpr off_t kMaxBytes = 1024 * 1024;

    int flags = O_WRONLY | O_CREAT | O_APPEND;
#ifdef O_CLOEXEC
    flags |= O_CLOEXEC;
#endif
#ifdef O_NOFOLLOW
    flags |= O_NOFOLLOW;
#endif
    int fd = ::open(kPath, flags, 0600);
    if (fd < 0) return nullptr;

    struct stat st;
    if (::fstat(fd, &st) != 0 || !S_ISREG(st.st_mode)) {
        ::close(fd);
        return nullptr;
    }
    if (st.st_size > kMaxBytes) {
#ifdef O_NOFOLLOW
        if (::ftruncate(fd, 0) != 0) {
            ::close(fd);
            return nullptr;
        }
#else
        ::close(fd);
        return nullptr;
#endif
    }

    FILE * out = fdopen(fd, "a");
    if (!out) ::close(fd);
    return out;
#endif
}

// Persistent concurrent-cache bytes that do not scale with the physical
// paged pool. The estimate mirrors create_target_cache_partial(): recurrent
// state for N slots, target features, paged metadata, Q capture, and the
// dead-row scratch block. Prefill reads the pool directly, so there is no
// staging K/V or staging recurrent slab to reserve.
static int64_t concurrent_fixed_cache_bytes(
        const TargetWeights & w, int max_ctx, int n_slots,
        int64_t kv_bytes_per_token, int64_t scratch_tokens,
        bool fixed_chain) {
    const int64_t n_full_attn =
        w.n_layer / w.full_attention_interval;
    const int64_t n_delta = w.n_layer - n_full_attn;
    const int64_t head_v_dim = w.ssm_d_inner / w.ssm_dt_rank;
    const int64_t conv_ch =
        w.ssm_d_inner + 2LL * w.ssm_n_group * w.ssm_d_state;
    const int64_t state_per_layer =
        (head_v_dim * head_v_dim * w.ssm_dt_rank +
         (int64_t)(w.ssm_d_conv - 1) * conv_ch) *
        (int64_t)sizeof(float);
    const int64_t recurrent =
        state_per_layer * n_delta * (int64_t)n_slots;
    const int64_t target_feat =
        (int64_t)w.n_capture_layers * w.n_embd *
        (fixed_chain
             ? (int64_t)std::min(max_ctx, 4096) * n_slots + 1
             : (int64_t)std::min(max_ctx, 4096)) *
        (int64_t)sizeof(uint16_t);
    const int64_t q_capture =
        (int64_t)w.n_embd_head_k * w.n_head * n_full_attn *
        (int64_t)sizeof(float) *
        (1 + luce::common::kvflash_q_capture_all_tokens() +
         luce::common::kvflash_q_stage_rows(LUCE_DRAFT_BLOCK_SIZE));
    const int64_t paged_metadata =
        ((int64_t)paged_block_count(max_ctx) * n_slots + n_slots) *
        (int64_t)sizeof(int32_t);
    const int64_t scratch =
        kv_bytes_per_token * scratch_tokens;
    return recurrent + target_feat + q_capture +
           paged_metadata + scratch;
}
}  // namespace

#define IS_EOS_TOK(tok, w)                                         \
    ( ((w).eos_chat_id >= 0 && (tok) == (w).eos_chat_id)                  \
   || ((w).eos_id      >= 0 && (tok) == (w).eos_id     ) )

static bool qwen35_empty_visible_output(const std::vector<int32_t> & tokens,
                                        const TargetWeights & w) {
    if (tokens.empty()) return false;
    for (int32_t tok : tokens) {
        if (!IS_EOS_TOK(tok, w)) return false;
    }
    return true;
}

// Drafters trained on explicit target layers (GGUF dflash.target_layer_ids)
// override the evenly-spaced derivation: capturing different layers than the
// drafter was trained on silently destroys acceptance.
static void apply_drafter_capture_layer_ids(const DraftWeights & dw, TargetWeights & w) {
    if (dw.capture_layer_ids.empty()) return;
    const int n = (int)dw.capture_layer_ids.size();
    bool ok = (n == w.n_capture_layers);
    for (int k = 0; ok && k < n; k++)
        ok = dw.capture_layer_ids[k] >= 0 && dw.capture_layer_ids[k] < w.n_layer;
    if (!ok) {
        std::fprintf(stderr,
            "[draft]  drafter target_layer_ids invalid (n=%d, slots=%d); "
            "keeping derived capture layers\n", n, w.n_capture_layers);
        return;
    }
    bool changed = false;
    for (int k = 0; k < n; k++) {
        changed |= w.capture_layer_ids[k] != dw.capture_layer_ids[k];
        w.capture_layer_ids[k] = dw.capture_layer_ids[k];
    }
    if (changed) {
        std::printf("[draft]  target capture layers from drafter GGUF:");
        for (int k = 0; k < n; k++) std::printf(" %d", w.capture_layer_ids[k]);
        std::printf("\n");
    }
}

// ── Construction / destruction ──────────────────────────────────────────

Qwen35Backend::Qwen35Backend(Qwen35Config cfg) : cfg_(std::move(cfg)) {}

Qwen35Backend::~Qwen35Backend() { shutdown(); }

// "auto" pool budget: device-free minus a reserve (compute buffers + drafter
// when expected), converted at this model's pooled-KV density. Shared with MoE
// placement so reservation and runtime allocation size the pool identically.
KvFlashAutoBudget Qwen35Backend::make_kvflash_budget(const TargetWeights & w,
                                                     int64_t gpu_free) const {
    ggml_type kv_k = GGML_TYPE_Q8_0, kv_v = GGML_TYPE_Q8_0;
    luce::resolve_kv_types(kv_k, kv_v, cfg_.cache_type_k, cfg_.cache_type_v);
    KvFlashAutoBudget b;
    b.free_bytes      = gpu_free;
    // Single source of truth with the qwen35moe placement path — see kv_quant.h.
    b.bytes_per_token = (int64_t)luce::kv_reservation_bytes_per_token(
        w.n_layer, w.full_attention_interval, w.n_head_kv,
        kv_k, w.n_embd_head_k, kv_v, w.n_embd_head_v);
    b.reserve_bytes   = (int64_t)(1.5 * 1073741824.0) +
        (kvflash_drafter_path_.empty() ? 0 : (int64_t)(1.7 * 1073741824.0));
    return b;
}

// ── init() ──────────────────────────────────────────────────────────────

bool Qwen35Backend::init() {
    configure_concurrent_hipblaslt_default(cfg_);

    const bool use_remote_draft = cfg_.remote_draft.enabled();
    const bool tensor_parallel = cfg_.device.is_tensor_parallel();
    split_gpus_ = !use_remote_draft && cfg_.draft_path &&
                  (tensor_parallel || cfg_.device.gpu != cfg_.draft_gpu);

    if (tensor_parallel) {
        tensor_parallel_ = Qwen35TensorParallelContext::create(cfg_.device, w_);
        target_backend_ = tensor_parallel_ ? tensor_parallel_->init_backend() : nullptr;
    } else {
        target_backend_ = ggml_backend_cuda_init(cfg_.device.gpu);
    }
    if (!target_backend_) {
        std::fprintf(stderr, "target backend init failed\n");
        return false;
    }
    draft_backend_ = use_remote_draft || !cfg_.draft_path ? nullptr : target_backend_;
    if (split_gpus_) {
        draft_backend_ = ggml_backend_cuda_init(cfg_.draft_gpu);
        if (!draft_backend_) {
            std::fprintf(stderr, "draft cuda init failed\n");
            return false;
        }
    }
    if (split_gpus_ && g_peer_access_opt_in) {
        enable_peer_access_pair(cfg_.device.gpu, cfg_.draft_gpu);
    }

    // Snapshot backend: on discrete GPU uses system RAM; on unified memory
    // (Metal, iGPU) stays on compute backend.
    snap_backend_ = create_snapshot_backend(target_backend_);
    if (!snap_backend_) {
        std::fprintf(stderr, "snapshot backend init failed\n");
        return false;
    }

    // Load target
    if (!load_target_model(target_backend_, w_)) {
        std::fprintf(stderr, "target load: %s\n", luce_last_error());
        return false;
    }
    std::printf("[target] %s\n", luce_last_error());
    if (!load_vision()) return false;
    if (cfg_.paged_attention &&
        (w_.n_embd_head_k != 256 || w_.n_embd_head_v != 256)) {
        std::fprintf(stderr,
            "[paged-attention] unsupported attention head dimensions "
            "K=%d V=%d; this kernel requires K=V=256\n",
            w_.n_embd_head_k, w_.n_embd_head_v);
        set_last_error("paged attention requires 256-wide K/V heads");
        return false;
    }

    // Load draft
    if (cfg_.draft_path && use_remote_draft) {
        const int cap = cfg_.remote_draft.ring_cap > 0
            ? std::min(cfg_.remote_draft.ring_cap, cfg_.device.max_ctx)
            : std::min(cfg_.device.max_ctx, cfg_.draft_ctx_max);
        if (!remote_draft_.start(cfg_.remote_draft.ipc_bin, *cfg_.draft_path,
                                 cfg_.draft_gpu, cap,
                                 cfg_.remote_draft.work_dir)) {
            std::fprintf(stderr, "remote draft start failed\n");
            return false;
        }
        dw_.n_embd = LUCE_TARGET_HIDDEN;
        dw_.block_size = LUCE_DRAFT_BLOCK_SIZE;
        dw_.n_target_layers = LUCE_DRAFT_N_TARGET_LAYERS;
        std::printf("[draft]  remote ipc ready gpu=%d cap=%d\n",
                    cfg_.draft_gpu, cap);
    } else if (cfg_.draft_path) {
        const std::string & dp = *cfg_.draft_path;
        bool draft_ok = (dp.size() >= 5 && dp.substr(dp.size() - 5) == ".gguf")
            ? load_draft_gguf(*cfg_.draft_path, draft_backend_, dw_, &w_)
            : load_draft_safetensors(*cfg_.draft_path, draft_backend_, dw_, &w_);
        if (!draft_ok) {
            std::fprintf(stderr, "draft load: %s\n", luce_last_error());
            return false;
        }
        std::printf("[draft]  loaded\n");
        apply_drafter_capture_layer_ids(dw_, w_);

        if (cfg_.draft_swa_window > 0) {
            const DraftSwaOverrideResult swa =
                apply_draft_swa_window_override(dw_, cfg_.draft_swa_window);
            std::printf("[draft]  SWA layers: %d/%d (window=%d)\n",
                        swa.swa_layers, swa.total_layers, swa.effective_window);
        }

        // Legacy 8-layer drafter YaRN from config flags. Applied here AND in
        // unpark() so the rotary encoding cannot flip mid-process (it used
        // to switch from plain RoPE to YaRN on the first park/unpark).
        if (dw_.rope_ext_factor == 0.0f && dw_.n_layer == 8 && dw_.n_embd == 2048) {
            float yf = cfg_.draft_yarn_factor > 1.0f ? cfg_.draft_yarn_factor : 64.0f;
            dw_.rope_freq_scale = 1.0f / yf;
            dw_.rope_ext_factor = 1.0f; dw_.rope_attn_factor = 1.0f;
            dw_.rope_beta_fast = cfg_.draft_yarn_beta_fast;
            dw_.rope_beta_slow = cfg_.draft_yarn_beta_slow;
            dw_.rope_n_ctx_orig = cfg_.draft_yarn_orig_ctx;
        }

        // The checkpoint metadata is the drafter's published proposal
        // horizon. Greedy chain verification keeps output byte-identical to
        // plain decode at any width, so widening only risks acceptance
        // depth; it is allowed up to 2x the horizon (measured on Qwen3.8
        // DFlash2: byte-identical completions across widths 8-24, commits
        // grow through 16, step time cliffs past 16 with no commit gain).
        if (cfg_.draft_block_size != 0) {
            const int checkpoint_block_size = dw_.block_size;
            if (!draft_block_size_override_supported(
                    cfg_.draft_block_size, checkpoint_block_size)) {
                std::fprintf(stderr,
                    "[draft] --draft-block-size must be in [2, %d] for this "
                    "drafter (up to 2x the checkpoint horizon); got %d\n",
                    2 * checkpoint_block_size, cfg_.draft_block_size);
                return false;
            }
            std::printf("[draft]  block size override: %d -> %d%s\n",
                        checkpoint_block_size, cfg_.draft_block_size,
                        cfg_.draft_block_size > checkpoint_block_size
                            ? " (beyond the published horizon; exact greedy verify)"
                            : "");
            dw_.block_size = cfg_.draft_block_size;
        }
    }

    // Create KV cache
    const int max_verify_tokens = cfg_.ddtree_mode
        ? std::max<int>(dw_.block_size, cfg_.ddtree_budget + 1)
        : dw_.block_size;
    // kvflash (bounded residency): pool size from the env, rounded/floored/
    // clamped by the shared reader (256-stride keeps FA vec-kernel
    // eligibility; the floor keeps eviction from deadlocking).
    // Drafter-scored residency is the DEFAULT policy: explicit
    // --prefill-drafter first, then the well-known locations next to the
    // model (Spark's pattern). LRU is the fallback when nothing is found
    // (or the explicit choice via --kvflash-policy lru).
    kvflash_qk_policy_ = kvflash_policy_is_qk();
    {
        const int m = env_int_or_default("LUCE_KVFLASH_PREFILL_RESELECT", 0);
        kvflash_prefill_reselect_ = kvflash_qk_policy_ && (m >= 1 && m <= 5) ? m : 0;
        kvflash_score_pool_recent_ = env_int_or_default("LUCE_KVFLASH_SCORE_POOL_RECENT", 0) == 1;
        kvflash_log_resident_ = env_int_or_default("LUCE_KVFLASH_LOG_RESIDENT", 0) == 1;
        const char * sc = std::getenv("LUCE_KVFLASH_RESELECT_SCORER");
        kvflash_exact_scorer_ = sc && std::string(sc) == "exact";
        kvflash_pflash2_scorer_ = sc && std::string(sc) == "pflash2";
        if (kvflash_prefill_reselect_ == 5 && !kvflash_pflash2_scorer_) {
            std::fprintf(stderr, "[kvflash] LUCE_KVFLASH_PREFILL_RESELECT=5 needs LUCE_KVFLASH_RESELECT_SCORER=pflash2; reselect off\n");
            kvflash_prefill_reselect_ = 0;
        }
        const char * sdev = std::getenv("LUCE_KVFLASH_SCORER_DEVICE");   // selector v2 scorer: cpu (default) or gpu
        kvflash_scorer_gpu_ = sdev && std::string(sdev) == "gpu";
        kvflash_recall_blocks_ = std::max(0, env_int_or_default("LUCE_KVFLASH_RECALL_BLOCKS", 0));
        kvflash_select_queries_ = kvflash_prefill_reselect_ >= 3 && kvflash_exact_scorer_
                                      ? luce::common::kvflash_select_queries() : 0;
        kvflash_select_unit_ = std::max(1, env_int_or_default("LUCE_KVFLASH_SELECT_UNIT", 10));
    }
    if (std::getenv("LUCE_KVFLASH") && (!kvflash_qk_policy_ || kvflash_prefill_reselect_ == 5)) {
        kvflash_drafter_path_ = kvflash_find_drafter(
            cfg_.target_path.c_str());
    }
    // "auto" sizes the pool from the GPU: weights are resident at this
    // point and the cache is not yet allocated, so device-free minus a
    // reserve (compute buffers + the drafter when expected) is what the
    // pool can really use, converted at this model's pooled-KV density.
    size_t gpu_free = 0, gpu_total = 0;
    if (ggml_backend_dev_t dev = ggml_backend_get_device(target_backend_)) {
        ggml_backend_dev_memory(dev, &gpu_free, &gpu_total);
    }
    KvFlashAutoBudget kvf_budget = make_kvflash_budget(w_, (int64_t)gpu_free);
    kvflash_tokens_ = kvflash_pool_from_env(cfg_.device.max_ctx, KvFlashConfig{},
                                            kvflash_scorer_expected(),
                                            cfg_.paged_attention
                                                ? kvflash_budget_per_slot(kvf_budget, concurrent_slots())
                                                : kvf_budget);
    if (kvflash_tokens_ > 0) {
        kvflash_tau_ = std::max(1, env_int_or_default("LUCE_KVFLASH_TAU", 64));
    }
    // LUCE_KVFLASH_PREAMBLE_ROWS=R: the pool grows by R rounded up to a chunk so the pinned
    // preamble does not eat the history budget. Decided here, before the KV cache is sized.
    kvflash_preamble_reserve_ = kvflash_preamble_rows_ = 0;
    if (kvflash_tokens_ > 0) {
        int rows = luce::common::kvflash_preamble_rows_from_env();
        if (rows > 0) {
            const std::string why = luce::common::kvflash_preamble_gate(w_.im_start_id, w_.system_id);
            if (!why.empty()) {
                std::fprintf(stderr, "[kvflash] preamble pin disabled: %s\n", why.c_str());
                rows = 0;
            }
        }
        const int ct0 = KvFlashConfig{}.chunk_tokens;
        const int reserve = luce::common::kvflash_preamble_reserve_tokens(rows, ct0);
        if (reserve > 0) {
            const int pool = luce::common::kvflash_pool_with_preamble(kvflash_tokens_, reserve,
                                                                      cfg_.device.max_ctx, ct0);
            std::fprintf(stderr, "[kvflash] preamble reserve: rows=%d reserve=%d pool %d -> %d tokens%s\n",
                         rows, reserve, kvflash_tokens_, pool,
                         pool - kvflash_tokens_ < reserve ? " (clamped by max_ctx)" : "");
            kvflash_preamble_reserve_ = pool - kvflash_tokens_;
            kvflash_preamble_rows_ = std::min(rows, kvflash_preamble_reserve_);
            kvflash_tokens_ = pool;
        }
    }
    kvflash_uq_queries_ = luce::common::kvflash_uq_queries();
    if (kvflash_uq_queries_ > 0) {
        const std::string why = luce::common::kvflash_uq_gate(w_.im_start_id, w_.user_id, w_.im_end_id, w_.tool_response_id);
        if (!why.empty()) {
            std::fprintf(stderr, "[kvflash] uq disabled: %s\n", why.c_str());
            kvflash_uq_queries_ = 0;
        }
    }
    if (kvflash_prefill_reselect_ == 5) {   // mode 5 runs K arm a's release profile; a missing setting is a refusal, not a silent no-recall run
        const std::string why = luce::common::kvflash_pflash2_config_error(
            [](const char * k) { const char * v = std::getenv(k); return std::string(v ? v : ""); }, concurrent_slots());
        if (!why.empty()) { set_last_error("LUCE_KVFLASH_PREFILL_RESELECT=5: " + why); return false; }
    }
    // Concurrent KVFlash (paged + KVFlash): decided on the REQUESTED combination
    // and the exact arch, before any subclass policy below can erase it.
    if (cfg_.paged_attention && kvflash_active()) {
        luce::common::KvFlashConcurrentRequest rq;
        rq.arch = kvflash_arch();
        rq.kvflash_requested = true;
        const char * wmma = std::getenv("LUCE_PAGED_WMMA");
        rq.wmma_env = wmma && std::string(wmma) == "1";
        // The WMMA kernel skips chunks with no named block (paged-attn.cu,
        // paged_attn_wmma_chunk_all_null) and the C1 HIP gate's Mode H ran
        // it over KVFlash holes on the R9700 (2026-10-26, top-1 agreement
        // 100% at p16-p122): one binary serves the plain and the KVFlash
        // WMMA arms of the Track C hardware screen.
        rq.wmma_null_exclusion_ready = true;
        rq.draft = cfg_.draft_path.has_value();
        rq.tree = cfg_.ddtree_mode;
        if (const auto why = luce::common::kvflash_concurrent_refusal(rq)) {
            std::fprintf(stderr, "[kvflash] %s\n", why->c_str());
            set_last_error(*why);
            return false;
        }
    }
    // Subclass gate (e.g. MoE all-hot): may zero kvflash_tokens_ before the KV
    // cache is sized, so create_target_cache allocates full max_ctx KV.
    if (!post_kvflash_init_gate()) return false;
    // KVFlash is resolved from env at init; this is the authoritative
    // paged×KVFlash compatibility check.
    if (cfg_.paged_attention && kvflash_active() && luce::common::kvflash_concurrent_test_switch()) {
        std::fprintf(stderr, "[kvflash] TEST SWITCH LUCE_KVFLASH_CONCURRENT_TEST=1: concurrent KVFlash slots "
                             "opened for the C1.3 GPU gates (not a supported configuration)\n");
    } else if (cfg_.paged_attention && kvflash_active()) {
        std::fprintf(stderr,
            "[paged-attention] cannot be combined with KVFlash "
            "(resident pool %d tokens)\n", kvflash_tokens_);
        set_last_error("paged attention cannot be combined with KVFlash");
        return false;
    }
    // Paged mode sizes the KV cache to whole blocks; otherwise KVFlash
    // decides the allocation (0 = full max_ctx).
    const int n_slots = concurrent_slots();
    const int max_concurrent_prefills = n_slots > 1
        ? std::clamp(
              env_int_or_default("LUCE_MAX_CONCURRENT_PREFILLS", 8),
              1, std::min(n_slots, 8))
        : 1;
    const int mixed_prefill_tokens = std::max(
        1, env_int_or_default("LUCE_MIXED_PREFILL_TOKENS", 2048));
    const int long_mixed_prefill_tokens = std::max(
        1, env_int_or_default("LUCE_LONG_MIXED_PREFILL_TOKENS", 4096));
    const int long_prefill_threshold = std::max(
        1, env_int_or_default("LUCE_LONG_PREFILL_THRESHOLD", 768));
    const int idle_prefill_tokens = std::max(
        1, env_int_or_default("LUCE_IDLE_PREFILL_TOKENS", 4096));
    const int prefill_quantum = std::max(
        1, env_int_or_default("LUCE_PREFILL_ALLOCATION_QUANTUM", 512));
    if (n_slots > 1 && !cfg_.paged_attention) {
        set_last_error("--max-concurrency requires --paged-attention");
        return false;
    }
    FixedChainConfig fixed_chain;
    fixed_chain.enabled =
        n_slots > 1 && cfg_.paged_attention && cfg_.fa_window == 0 &&
        cfg_.draft_path && !cfg_.ddtree_mode && !use_remote_draft &&
        !tensor_parallel && !split_gpus_ &&
        target_backend_ == draft_backend_ &&
        cfg_.device.gpu == cfg_.draft_gpu &&
        dw_.selector.enabled && dw_.block_size > 1 &&
        dw_.block_size <= 16 && w_.output;
    if (fixed_chain.enabled) {
        fixed_chain.width = dw_.block_size;
        fixed_chain.scratch_stride = paged_token_capacity(fixed_chain.width);
    }
    if (n_slots > 1 && cfg_.draft_path && !fixed_chain.enabled) {
        set_last_error(
            "concurrent paged DFlash2 requires a selector-enabled local "
            "same-device draft with block size in [2, 16] and a target "
            "lm_head");
        return false;
    }
    const int64_t scratch_tokens = PAGED_BLOCK_SIZE +
        (int64_t)n_slots * fixed_chain.scratch_stride;
    // Concurrent slots share one physical pool. An explicit
    // --kv-pool-tokens is rounded up to a whole block; otherwise capacity is
    // derived from device-free memory after subtracting fixed concurrent cache
    // state and a runtime graph reserve. This decouples max-concurrency from
    // startup K/V allocation while retaining at least one full logical context.
    // One extra scratch block is appended to the K/V tensors (outside the
    // pool's index space) as the write target of dead decode-batch rows.
    int64_t pool_tokens = 0;
    const bool slot_pool = seq_engine_wanted();
    // KVFlash slots (C1.3-05): one arena of N x Q x 64 rows plus a scratch block,
    // no PagedKvPool; --kv-pool-tokens is derived, not a size. Unreachable until
    // the paged x KVFlash refusal above is lifted (C1.3-06).
    const bool kvflash_slots = slot_pool && kvflash_active();
    KvFlashArenaSizing arena;
    if (kvflash_slots) {
        arena = kvflash_arena_sizing(n_slots, kvflash_quota_chunks(kvflash_tokens_), cfg_.kv_pool_tokens);
        if (!arena.error.empty()) { std::fprintf(stderr, "[kvflash] %s\n", arena.error.c_str()); set_last_error(arena.error); return false; }
    }
    if (slot_pool && !kvflash_slots) {
        PagedKvAutoBudget budget;
        // TODO: Size tensor-parallel pools from each device's free memory
        // and account for fixed cache tensors mirrored by the meta backend.
        budget.free_bytes = (int64_t)gpu_free;
        budget.bytes_per_token = kvf_budget.bytes_per_token;
        budget.reserve_bytes =
            kvf_budget.reserve_bytes + cfg_.kv_pool_reserve_bytes;
        budget.fixed_cache_bytes = concurrent_fixed_cache_bytes(
            w_, cfg_.device.max_ctx, n_slots, budget.bytes_per_token,
            scratch_tokens, fixed_chain.enabled);
        if (cfg_.kv_pool_tokens > 0) {
            const int64_t max_pool_tokens =
                ((int64_t)INT32_MAX - scratch_tokens) /
                PAGED_BLOCK_SIZE * PAGED_BLOCK_SIZE;
            pool_tokens = (int64_t)paged_token_capacity(
                (int)std::min<int64_t>(
                    cfg_.kv_pool_tokens, max_pool_tokens));
            // Memory set aside for a model beside the slots (the PFlash
            // drafter) bounds an explicit pool too.
            if (cfg_.kv_pool_reserve_bytes > 0 && budget.bytes_per_token > 0) {
                const int64_t fit = (std::max<int64_t>)(0,
                    budget.free_bytes - budget.fixed_cache_bytes -
                    budget.reserve_bytes) / budget.bytes_per_token;
                if (pool_tokens > fit) {
                    set_last_error(
                        "--kv-pool-tokens " + std::to_string(pool_tokens) +
                        " leaves less than the " +
                        std::to_string(cfg_.kv_pool_reserve_bytes >> 20) +
                        " MiB reserved for the PFlash drafter; at most " +
                        std::to_string(fit / PAGED_BLOCK_SIZE * PAGED_BLOCK_SIZE) +
                        " tokens fit");
                    std::fprintf(stderr, "[parallel] %s\n", luce_last_error());
                    return false;
                }
            }
        } else {
            pool_tokens = paged_kv_auto_pool_tokens(
                cfg_.device.max_ctx, n_slots, budget);
            const int64_t one_context =
                paged_token_capacity(cfg_.device.max_ctx);
            std::fprintf(stderr,
                "[parallel] auto KV pool: %lld tokens "
                "(free %.2f GiB - fixed %.2f GiB - reserve %.2f GiB, "
                "%.2f KiB/token; logical cap %lld)\n",
                (long long)pool_tokens,
                budget.free_bytes / 1073741824.0,
                budget.fixed_cache_bytes / 1073741824.0,
                budget.reserve_bytes / 1073741824.0,
                budget.bytes_per_token / 1024.0,
                (long long)n_slots * one_context);
            if (pool_tokens < one_context) {
                set_last_error(
                    "not enough device memory for one max_ctx paged sequence; "
                    "lower --max-ctx or set --kv-pool-tokens explicitly");
                return false;
            }
        }
        if (pool_tokens + scratch_tokens > INT32_MAX) {
            set_last_error("paged KV pool exceeds INT32_MAX tokens");
            return false;
        }
    }
    const int ctx_alloc = kvflash_slots
        ? (int)arena.rows
        : slot_pool
        ? (int)(pool_tokens + scratch_tokens)
        : (cfg_.paged_attention
               ? paged_token_capacity(cfg_.device.max_ctx)
               : kvflash_tokens_);
    if (!create_target_cache(w_, cfg_.device.max_ctx, max_verify_tokens, target_backend_, cache_,
                             /*prefill_only=*/true, ctx_alloc,
                             cfg_.paged_attention, n_slots,
                             fixed_chain.enabled, cfg_.cache_type_k, cfg_.cache_type_v,
                             slot_pool)) {
        std::fprintf(stderr, "cache: %s\n", luce_last_error());
        return false;
    }
    if (cfg_.paged_attention) {
        const auto supported_type = [](ggml_type type) {
            return type == GGML_TYPE_F16 ||
                   type == GGML_TYPE_Q4_0 ||
                   type == GGML_TYPE_Q8_0;
        };
        if (!supported_type(cache_.kv_k_type) ||
            !supported_type(cache_.kv_v_type)) {
            std::fprintf(stderr,
                "[paged-attention] unsupported KV types K=%s V=%s; use "
                "f16, q4_0, or q8_0\n",
                ggml_type_name(cache_.kv_k_type),
                ggml_type_name(cache_.kv_v_type));
            return false;
        }
        if (kvflash_slots) {
            // No PagedKvPool: the slot pagers own the arena's rows.
            if (!init_kvflash_slots(n_slots, kvflash_quota_chunks(kvflash_tokens_), cfg_.device.max_ctx,
                                    arena.pool_tokens, max_concurrent_prefills, mixed_prefill_tokens,
                                    long_mixed_prefill_tokens, long_prefill_threshold, idle_prefill_tokens)) {
                return false;
            }
        } else try {
            const uint32_t pool_blocks = slot_pool
                ? (uint32_t)(pool_tokens / PAGED_BLOCK_SIZE)
                : (uint32_t)paged_block_count(cfg_.device.max_ctx);
            paged_kv_pool_ = std::make_unique<PagedKvPool>(
                pool_blocks, /*max_sequences=*/(uint32_t)n_slots,
                PAGED_BLOCK_SIZE);
        } catch (const std::exception & e) {
            std::fprintf(stderr, "[paged-attention] pool init failed: %s\n",
                         e.what());
            return false;
        }
        if (kvflash_slots) {
            std::printf("[kvflash] %d slots x %d chunks, arena %lld rows, headroom 8, LRU only (no scoring)\n",
                        n_slots, kvflash_quota_chunks(kvflash_tokens_), (long long)arena.rows);
            std::fflush(stdout);
        } else if (slot_pool) {
            fixed_chain.scratch_base = (int)pool_tokens;
            const int64_t dead_scratch_row = pool_tokens +
                (int64_t)n_slots * fixed_chain.scratch_stride;
            seq_engine_ = std::make_unique<Qwen35SeqEngine>(
                *this, *paged_kv_pool_, cfg_.device.max_ctx,
                dead_scratch_row, fixed_chain,
                max_concurrent_prefills, mixed_prefill_tokens,
                long_mixed_prefill_tokens, long_prefill_threshold,
                idle_prefill_tokens, prefill_quantum);
            if (fixed_chain.enabled) {
                std::fprintf(stderr,
                    "[parallel-chain] fixed DFlash2 width=%d, "
                    "same-device paged full-attention greedy lanes\n",
                    fixed_chain.width);
            }
            std::printf("[parallel] %d decode slots, up to %d packed prefills "
                        "(mixed short/long %d/%d at >=%d tokens, "
                        "idle %d, quantum %d), "
                        "pool %u blocks x %u tokens"
                        " (%lld shared tokens, per-seq max_ctx %d)\n",
                        n_slots, max_concurrent_prefills,
                        mixed_prefill_tokens, long_mixed_prefill_tokens,
                        long_prefill_threshold, idle_prefill_tokens, prefill_quantum,
                        paged_kv_pool_->physical_block_count(),
                        paged_kv_pool_->block_size(),
                        (long long)pool_tokens, cfg_.device.max_ctx);
        }
        if (!kvflash_slots) std::printf("[paged-attention] %u physical blocks x %u tokens "
                    "(%llu pool tokens, per-sequence max_ctx %d)\n",
                    paged_kv_pool_->physical_block_count(),
                    paged_kv_pool_->block_size(),
                    (unsigned long long)paged_kv_pool_->physical_block_count() *
                        paged_kv_pool_->block_size(),
                    cfg_.device.max_ctx);
        std::fflush(stdout);
    }
    if (kvflash_active() && !kvflash_slots) {
        KvFlashConfig pc;
        pc.pool_tokens = kvflash_tokens_;
        if (!kvflash_pager_.attach(pc, cache_.attn_k, cache_.attn_v)) {
            std::fprintf(stderr, "kvflash: pager attach failed (pool=%d)\n", kvflash_tokens_);
            return false;
        }
        // Page-out and recall run on the pager's own stream; make the compute
        // stream wait for them on the device before it touches a reused slot
        // (without this, slot_for() waits on the host instead).
        if (!tensor_parallel_ && target_backend_ && ggml_backend_is_cuda(target_backend_))
            kvflash_pager_.set_compute_stream(ggml_backend_cuda_get_stream(target_backend_));
        if (kvflash_qk_policy_) {
            KvFlashQkDims qd;
            qd.n_layers   = (int)cache_.attn_k.size();
            qd.n_q_heads  = w_.n_head;
            qd.n_kv_heads = w_.n_head_kv;
            qd.head_dim   = w_.n_embd_head_k;
            kvflash_qk_pool_.reset(qd);
            auto qs = std::make_unique<KvFlashTargetQkScorer>(&kvflash_qk_pool_);
            kvflash_qk_scorer_ = qs.get();
            kvflash_scorer_ = std::move(qs);
        }
        std::printf("[kvflash] resident pool %d tokens (logical max_ctx %d), "
                    "tau=%d, policy=%s\n",
                    kvflash_tokens_, cfg_.device.max_ctx, kvflash_tau_,
                    kvflash_qk_policy_ ? "qk (target pooled-K vs decode query)"
                    : !kvflash_drafter_path_.empty()
                        ? "drafter (attaches on first reselect)"
                        : "lru (recency-only: no Qwen3.5-0.8B drafter found "
                          "next to the model or in --prefill-drafter)");
        std::fflush(stdout);
    }

    // Init feature mirror when draft model is available (needed for spec decode).
    // On single-GPU, this is an F32 conversion buffer; on split-GPU, a cross-device mirror.
    if (cfg_.draft_path && !use_remote_draft &&
        !fixed_chain.enabled) {
        const int mirror_cap = std::min({cfg_.draft_ctx_max, cfg_.device.max_ctx,
                                         cache_.target_feat_cap > 0 ? cache_.target_feat_cap : cfg_.device.max_ctx});
        if (!draft_feature_mirror_init(feature_mirror_, draft_backend_,
                                       cfg_.draft_gpu, cfg_.device.gpu, mirror_cap,
                                       w_.n_capture_layers,
                                       w_.n_embd)) {
            std::fprintf(stderr, "warning: feature mirror init failed, spec decode will use AR fallback\n");
        }
    }

    return true;
}

bool Qwen35Backend::load_target_model(ggml_backend_t backend, TargetWeights & out) {
    return load_target_gguf(cfg_.target_path, backend, out);
}

bool Qwen35Backend::run_ar_decode_path(int committed, ThinkingBudget & budget,
                                       std::vector<int32_t> & out_tokens,
                                       const DaemonIO & io,
                                       bool * forced_close_out,
                                       bool * degenerate_close_out) {
    return do_ar_decode(committed, budget, out_tokens, io,
                        forced_close_out, degenerate_close_out);
}

bool Qwen35Backend::begin_paged_sequence(uint32_t prompt_tokens) {
    if (!paged_kv_pool_) return false;

    paged_kv_pool_->reset();
    paged_sequence_.reset();
    ++paged_request_id_;

    PagedKvSequenceHandle handle;
    PagedKvStatus status = paged_kv_pool_->acquire(paged_request_id_, handle);
    if (status != PagedKvStatus::Ok) {
        std::fprintf(stderr, "[paged-attention] acquire failed: %s\n",
                     paged_kv_status_string(status));
        return false;
    }
    paged_sequence_ = handle;

    PagedKvAppendResult append = paged_kv_pool_->append(
        handle, prompt_tokens, /*only_first_last_slots=*/true);
    if (!append) {
        std::fprintf(stderr,
            "[paged-attention] prompt allocation (%u tokens) failed: %s\n",
            prompt_tokens, paged_kv_status_string(append.status));
        set_last_error("paged attention prompt allocation failed");
        end_paged_sequence();
        return false;
    }

    // Dense prefill writes logical row p directly. A freshly reset pool hands
    // out the lowest blocks, so checking the compact range endpoints is enough
    // to verify that those rows and the initial page table coincide.
    if (prompt_tokens > 0 &&
        (append.first.physical_token_index != append.first.logical_position ||
         append.last.physical_token_index != append.last.logical_position)) {
        std::fprintf(stderr,
            "[paged-attention] non-identity prefill allocation in [%u, %u]\n",
            append.first.logical_position, append.last.logical_position);
        set_last_error("paged attention prefill allocation is not contiguous");
        end_paged_sequence();
        return false;
    }

    // Upload the prefill portion of the block table once. The device tensor
    // is a persistent cache allocation, so each decode step afterwards
    // appends at most one fresh entry.
    PagedKvSequenceSnapshot sequence;
    status = paged_kv_pool_->sequence(handle, sequence);
    if (status != PagedKvStatus::Ok) {
        std::fprintf(stderr,
            "[paged-attention] sequence snapshot failed: %s\n",
            paged_kv_status_string(status));
        set_last_error("paged attention sequence snapshot failed");
        end_paged_sequence();
        return false;
    }
    const size_t table_capacity = (size_t)cache_.paged_block_table->ne[0];
    if (sequence.block_table.size() > table_capacity) {
        std::fprintf(stderr,
            "[paged-attention] block table needs %zu entries, capacity is %zu\n",
            sequence.block_table.size(), table_capacity);
        set_last_error("paged attention block table capacity exceeded");
        end_paged_sequence();
        return false;
    }
    if (!sequence.block_table.empty()) {
        std::vector<int32_t> table(sequence.block_table.size());
        for (size_t i = 0; i < table.size(); ++i) {
            table[i] = (int32_t)sequence.block_table[i];
        }
        ggml_backend_tensor_set(cache_.paged_block_table, table.data(), 0,
                                table.size() * sizeof(table[0]));
    }
    return true;
}

bool Qwen35Backend::prepare_paged_decode_step(uint32_t logical_position,
                                              int64_t & out_kv_row) {
    auto fail = [&](const std::string & msg) {
        std::fprintf(stderr, "[paged-attention] %s\n", msg.c_str());
        set_last_error("paged attention " + msg);
        return false;
    };
    if (!paged_kv_pool_ || !paged_sequence_ ||
        !cache_.paged_block_table || !cache_.paged_kv_seq_lens) {
        return fail("decode step is missing paging state");
    }

    PagedKvAppendResult append = paged_kv_pool_->append(
        *paged_sequence_, /*token_count=*/1,
        /*only_first_last_slots=*/true);
    if (!append || append.token_count != 1 ||
        append.last.logical_position != logical_position) {
        return fail("decode allocation failed at " +
                    std::to_string(logical_position) + ": " +
                    paged_kv_status_string(append.status));
    }

    const PagedKvWriteSlot & slot = append.last;
    // The feature gate bounds max_ctx and the pool caps its token capacity at
    // uint32, so these can only trip on allocator corruption.
    GGML_ASSERT(slot.physical_token_index <= (uint64_t)INT64_MAX);
    GGML_ASSERT(logical_position < (uint32_t)INT32_MAX);
    GGML_ASSERT(slot.physical_block <= (uint32_t)INT32_MAX);
    out_kv_row = (int64_t)slot.physical_token_index;

    // begin_paged_sequence uploaded the prefill table entries; a step that
    // grows the sequence into a fresh block appends exactly one entry, and
    // steps inside a block upload nothing.
    if (slot.block_offset == 0) {
        const size_t logical_block = logical_position / PAGED_BLOCK_SIZE;
        const size_t table_capacity = (size_t)cache_.paged_block_table->ne[0];
        if (logical_block >= table_capacity) {
            return fail("block index " + std::to_string(logical_block) +
                        " exceeds table capacity " +
                        std::to_string(table_capacity));
        }
        const int32_t entry = (int32_t)slot.physical_block;
        ggml_backend_tensor_set(cache_.paged_block_table, &entry,
                                logical_block * sizeof(entry), sizeof(entry));
    }

    const int32_t kv_seq_len = (int32_t)logical_position + 1;
    ggml_backend_tensor_set(cache_.paged_kv_seq_lens, &kv_seq_len, 0,
                            sizeof(kv_seq_len));
    return true;
}

void Qwen35Backend::end_paged_sequence() {
    if (paged_kv_pool_ && paged_sequence_) {
        const PagedKvStatus status =
            paged_kv_pool_->release(*paged_sequence_);
        if (status != PagedKvStatus::Ok &&
            status != PagedKvStatus::StaleHandle) {
            std::fprintf(stderr, "[paged-attention] release failed: %s\n",
                         paged_kv_status_string(status));
        }
    }
    paged_sequence_.reset();
}

// ── Concurrent slot serving (--max-concurrency N) ──────────────────────
//
// The engine lives in concurrency/qwen35_seq_engine.cpp; what stays here is the
// model-level policy it borrows — EOS identity and the min-tokens floor,
// both shared with the AR decode path.

bool Qwen35Backend::token_is_eos(int32_t token) const {
    return IS_EOS_TOK(token, w_);
}

int32_t Qwen35Backend::apply_min_tokens_floor(int32_t tok, int generated,
                                              size_t logits_row_offset) {
    // MIN_TOKENS_BEFORE_EOS (env LUCE_MIN_TOKENS, default off): same
    // policy as do_ar_decode — if the slot would stop before emitting the
    // floor, substitute the best non-EOS token. The logits row is fetched
    // on demand so the (common) GPU-argmax path pays nothing when the
    // floor is off or not triggered.
    const int floor = dflash_min_tokens_floor();
    if (floor <= 0 || generated >= floor || !IS_EOS_TOK(tok, w_)) return tok;
    const int vocab = w_.n_vocab;
    std::vector<float> buf((size_t)vocab);
    ggml_backend_tensor_get(sg_.logits, buf.data(), logits_row_offset,
                            sizeof(float) * (size_t)vocab);
    int alt = -1;
    float best = -1e30f;
    for (int v = 0; v < vocab; v++) {
        if (IS_EOS_TOK(v, w_)) continue;
        if (buf[(size_t)v] > best) { best = buf[(size_t)v]; alt = v; }
    }
    if (alt < 0) return tok;
    FILE * log = open_dflash_floor_log();
    if (log) {
        std::fprintf(log, "[floor] eos@%d -> alt=%d\n", generated, alt);
        std::fclose(log);
    }
    return alt;
}

SeqEngine * Qwen35Backend::seq_engine() {
    return seq_engine_.get();
}


// ── print_ready_banner ──────────────────────────────────────────────────

void Qwen35Backend::print_ready_banner() const {
    std::printf("[daemon] ready\n");
    std::fflush(stdout);
}

// ── Park / unpark ───────────────────────────────────────────────────────

bool Qwen35Backend::park(ParkTarget target) {
    materialize_live_snapshot();
    live_snapshot_slot_ = -1;
    const bool want_draft_model = park_target_includes_draft_model(target);
    const bool want_target_model = park_target_includes_target_model(target);
    const bool use_remote_draft = cfg_.remote_draft.enabled();

    if (want_draft_model && !draft_parked_) {
        if (use_remote_draft) {
            remote_draft_.close();
        } else {
            step_graph_destroy(draft_sg_);
            if (seq_engine_) seq_engine_->release_draft_graphs();
            draft_kv_free(draft_kv_);
            free_draft_weights(dw_);
        }
        draft_parked_ = true;
        std::printf("[park] draft released\n"); std::fflush(stdout);
    }
    if (want_target_model && !target_parked_) {
        step_graph_destroy(sg_);
        step_graph_destroy(proj_sg_);
        dflash2_selector_graph_invalidate();
        free_target_weights(w_);
        vision_.reset();
        target_parked_ = true;
        std::printf("[park] target released\n"); std::fflush(stdout);
    }
    return true;
}

bool Qwen35Backend::unpark(ParkTarget target) {
    materialize_live_snapshot();
    live_snapshot_slot_ = -1;
    const bool want_target_model = park_target_includes_target_model(target);
    const bool want_draft_model = park_target_includes_draft_model(target);
    const bool use_remote_draft = cfg_.remote_draft.enabled();

    if (want_target_model && target_parked_) {
        if (!load_target_model(target_backend_, w_)) {
            std::fprintf(stderr, "[unpark] target: %s\n", luce_last_error());
            return false;
        }
        if (!load_vision()) {
            // Stay parked, so a retry starts from released weights.
            free_target_weights(w_);
            return false;
        }
        kvflash_drafter_failed_ = false;   // fresh VRAM: allow a retry
        target_parked_ = false;
        std::printf("[unpark] target restored\n"); std::fflush(stdout);
    }
    if (want_draft_model && draft_parked_ && cfg_.draft_path) {
        if (use_remote_draft) {
            const int cap = cfg_.remote_draft.ring_cap > 0
                ? std::min(cfg_.remote_draft.ring_cap, cfg_.device.max_ctx)
                : std::min(cfg_.device.max_ctx, cfg_.draft_ctx_max);
            if (!remote_draft_.start(
                    cfg_.remote_draft.ipc_bin, *cfg_.draft_path,
                                     cfg_.draft_gpu, cap,
                                     cfg_.remote_draft.work_dir)) {
                std::fprintf(stderr, "[unpark] remote draft failed\n");
                return false;
            }
        } else {
            const std::string & dp = *cfg_.draft_path;
            bool draft_ok = (dp.size() >= 5 && dp.substr(dp.size() - 5) == ".gguf")
                ? load_draft_gguf(*cfg_.draft_path, draft_backend_, dw_, &w_)
                : load_draft_safetensors(
                      *cfg_.draft_path, draft_backend_, dw_, &w_);
            if (!draft_ok) {
                std::fprintf(stderr, "[unpark] draft: %s\n", luce_last_error());
                return false;
            }
            apply_drafter_capture_layer_ids(dw_, w_);
            // Re-apply rope overrides after reload.
            if (dw_.rope_theta != w_.rope_theta && w_.rope_theta > 0.0f)
                dw_.rope_theta = w_.rope_theta;
            if (dw_.rope_ext_factor == 0.0f && dw_.n_layer == 8 && dw_.n_embd == 2048) {
                float yf = cfg_.draft_yarn_factor > 1.0f ? cfg_.draft_yarn_factor : 64.0f;
                dw_.rope_freq_scale = 1.0f / yf;
                dw_.rope_ext_factor = 1.0f; dw_.rope_attn_factor = 1.0f;
                dw_.rope_beta_fast = cfg_.draft_yarn_beta_fast;
                dw_.rope_beta_slow = cfg_.draft_yarn_beta_slow;
                dw_.rope_n_ctx_orig = cfg_.draft_yarn_orig_ctx;
            }
            if (cfg_.draft_swa_window > 0) {
                const DraftSwaOverrideResult swa =
                    apply_draft_swa_window_override(dw_, cfg_.draft_swa_window);
                std::printf("[unpark] draft SWA layers: %d/%d (window=%d)\n",
                            swa.swa_layers, swa.total_layers, swa.effective_window);
            }
            // Re-apply the runtime block-size override: without this a
            // park/unpark cycle silently reverts to checkpoint metadata
            // (out-of-bounds rollback windows when narrowing, a silent
            // no-op when widening).
            if (cfg_.draft_block_size != 0 &&
                draft_block_size_override_supported(cfg_.draft_block_size,
                                                    dw_.block_size)) {
                dw_.block_size = cfg_.draft_block_size;
            }
        }
        // A reloaded drafter can select different target capture layers.
        // Rebuild any retained target graph against the refreshed mapping.
        step_graph_free(sg_);
        draft_parked_ = false;
        std::printf("[unpark] draft restored\n"); std::fflush(stdout);
    }
    return true;
}

// ── Snapshots ───────────────────────────────────────────────────────────

// Save the pooled KV as a pager blob. The recurrent state (SSM/conv) is
// captured from the live cache, so the snapshot position must be the live
// position: `snap_boundary` has to equal cache_.cur_pos (a chunk multiple at a
// pooled-prefill ubatch boundary, the prompt end otherwise). Only the chunks
// covering [0, snap_boundary) are serialized; a partial last chunk is included
// whole. Anything else would pair KV and recurrent state from different
// positions, so it is refused.
bool Qwen35Backend::snapshot_save_pooled_at(int slot, int snap_boundary) {
    if (slot < 0 || slot >= PREFIX_SLOTS || !kvflash_active()) return false;
    if (snap_boundary <= 0 || snap_boundary != cache_.cur_pos) return false;
    const int chunk = kvflash_pager_.chunk_tokens();
    const int n_chunks = (snap_boundary + chunk - 1) / chunk;
    const std::vector<uint8_t> blob = kvflash_pager_.serialize(n_chunks);
    return snapshot_target_cache(w_, cache_, snap_backend_,
                                 prefix_snapshots_[slot],
                                 /*skip_kv=*/true, &blob);
}

void Qwen35Backend::state_ladder_clear() {
    for (auto & kv : state_ladder_) free_prefix_snapshot(kv.second.snap);
    state_ladder_.clear();
    state_ladder_ckpt_bytes_ = 0;
    state_ladder_head_ = -1;
    kvflash_gen_ckpt_pos_ = -1;
}

// State-only checkpoint (ssm + conv, no KV; plus target_feat when a drafter
// is configured) of the live cache at `pos`, then LRU eviction down to
// LUCE_KVFLASH_STATE_LADDER_MIB (default 3072).
bool Qwen35Backend::state_ladder_capture(int pos) {
    if (pos <= 0 || pos != cache_.cur_pos) return false;
    StateCheckpoint & ck = state_ladder_[pos];
    if (!snapshot_target_cache(w_, cache_, snap_backend_, ck.snap, /*skip_kv=*/true,
                              /*kvflash_blob=*/nullptr,
                              /*skip_target_feat=*/!cfg_.draft_path)) {
        free_prefix_snapshot(ck.snap);
        state_ladder_.erase(pos);
        return false;
    }
    ck.tick = ++state_ladder_tick_;
    size_t bytes = 0;
    for (const auto * v : {&ck.snap.ssm_state_snap, &ck.snap.conv_state_snap}) {
        for (const ggml_tensor * t : *v) if (t) bytes += ggml_nbytes(t);
    }
    if (ck.snap.target_feat_snap) bytes += ggml_nbytes(ck.snap.target_feat_snap);
    state_ladder_ckpt_bytes_ = bytes;

    static const size_t budget = [] {
        const char * v = std::getenv("LUCE_KVFLASH_STATE_LADDER_MIB");
        const long mib = v ? std::atol(v) : 3072;
        return (size_t)(mib > 0 ? mib : 3072) << 20;
    }();
    std::vector<KvflashLadderEntry> entries;
    for (const auto & kv : state_ladder_) entries.push_back({kv.first, kv.second.tick});
    for (int p : kvflash_ladder_evict(entries, budget, bytes, state_ladder_head_)) {
        free_prefix_snapshot(state_ladder_[p].snap);
        state_ladder_.erase(p);
    }
    std::printf("[kvflash] state ladder: ckpt @%d (%.1f MiB each, %zu held, %.1f MiB)\n",
                pos, (double)bytes / (1 << 20), state_ladder_.size(),
                (double)(state_ladder_.size() * bytes) / (1 << 20));
    std::fflush(stdout);
    return true;
}

// Cut a live restore of `prompt` would resume from, or -1. No side effects.
int Qwen35Backend::kvflash_live_cut_peek(const std::vector<int32_t> & prompt) const {
    if (!kvflash_session_enabled() || !kvflash_active() || cfg_.paged_attention ||
        prompt.empty() || kvflash_pager_.chunk_tokens() <= 0) return -1;
    std::vector<int> rungs;
    for (const auto & kv : state_ladder_) rungs.push_back(kv.first);
    return kvflash_live_restore_decide(prompt, kvflash_history_, cache_.cur_pos, rungs).C;
}

// LUCE_KVFLASH_PREAMBLE_ROWS: pin the system+tools preamble. Pins live in the pager and are cleared
// by reset() / deserialize() and clipped by truncate(), so every path that does one of those calls this.
int Qwen35Backend::kvflash_apply_preamble_pin(const std::vector<int32_t> * ids, bool log) {
    if (kvflash_preamble_rows_ <= 0 || !kvflash_active()) return 0;
    if (ids) kvflash_preamble_end_ = luce::common::kvflash_preamble_end(*ids, w_.im_start_id, w_.system_id);
    const int E = kvflash_preamble_end_;
    if (log && luce::common::kvflash_preamble_exceeds(E, kvflash_preamble_rows_))
        std::fprintf(stderr, "[kvflash] preamble %d tok exceeds reserve %d\n", E, kvflash_preamble_rows_);
    const int pinned = luce::common::kvflash_pin_preamble(kvflash_pager_, E, kvflash_preamble_rows_);
    if (log)
        std::fprintf(stderr, "[kvflash] preamble pin end=%d pinned_chunks=%d reserve=%d\n",
                     E, pinned, kvflash_preamble_rows_);
    return pinned;
}

int Qwen35Backend::kvflash_live_restore(const std::vector<int32_t> & prompt) {
    std::vector<int> rungs;
    for (const auto & kv : state_ladder_) rungs.push_back(kv.first);
    const KvflashLiveRestore d =
        kvflash_live_restore_decide(prompt, kvflash_history_, cache_.cur_pos, rungs);
    // S9: report whether the re-rendered history still covers the generation-end rung.
    if (kvflash_gen_ckpt_pos_ > 0) {
        std::printf("[kvflash] gen ckpt check: G=%d L=%d L>=G=%d", kvflash_gen_ckpt_pos_, d.L,
                    d.L >= kvflash_gen_ckpt_pos_ ? 1 : 0);
        if (d.L < kvflash_gen_ckpt_pos_) {
            std::printf(" first-diff@%d prompt=[", d.L);
            for (size_t i = d.L; i < prompt.size() && i < (size_t)d.L + 8; i++) std::printf("%s%d", i > (size_t)d.L ? "," : "", prompt[i]);
            std::printf("] history=[");
            for (size_t i = d.L; i < kvflash_history_.size() && i < (size_t)d.L + 8; i++) std::printf("%s%d", i > (size_t)d.L ? "," : "", kvflash_history_[i]);
            std::printf("]");
        }
        std::printf("\n");
        std::fflush(stdout);
    }
    // Rungs above the common prefix describe tokens that no longer match.
    for (auto it = state_ladder_.upper_bound(d.discard_above); it != state_ladder_.end();) {
        free_prefix_snapshot(it->second.snap);
        it = state_ladder_.erase(it);
    }
    // C need not be chunk-aligned: the S9 gen rung is off-grid, and
    // pager.truncate (S5) zeroes the rows >= C of a partial chunk, which the
    // replay then rewrites through slot_for.
    const int ct = kvflash_pager_.chunk_tokens();
    int C = d.C;
    if (C > 0 && (ct <= 0 || state_ladder_.find(C) == state_ladder_.end())) C = -1;
    std::printf("[kvflash] live restore: L=%d C=%d replay=%d\n", d.L, C,
                C > 0 ? (int)prompt.size() - C : (int)prompt.size());
    std::fflush(stdout);
    if (C <= 0) return -1;

    // No clear-then-restore here: the pool rows below C are the session.
    kvflash_pager_.truncate(C);
    kvflash_apply_preamble_pin(&prompt, false);
    if (!restore_target_cache(state_ladder_[C].snap, cache_, /*skip_kv=*/true)) {
        std::fprintf(stderr, "[kvflash] live restore: state restore @%d failed; cold prefill\n", C);
        return -1;   // the cold prefill resets pager, ladder and history
    }
    kvflash_scores_.clear();
    kvflash_mask_epoch_ = (uint64_t)-1;
    kvflash_pager_.zero_free_blocks();
    kvflash_history_.resize((size_t)C);
    if (kvflash_qk_policy_) kvflash_qk_pooled_upto_ = std::min(kvflash_qk_pooled_upto_, C / ct);
    cache_.cur_pos = C;

    // Re-sync the draft-side feature mirror, as restore_and_generate does.
    if (!draft_parked_) {
        const int ring_cap = remote_draft_.active() ? remote_draft_.ring_cap()
                                                     : feature_mirror_.cap;
        const int n = std::min(C, ring_cap);
        bool ok = true;
        if (n > 0) {
            if (remote_draft_.active()) ok = sync_remote_draft_features(C - n, n);
            else if (feature_mirror_.target_feat && cache_.target_feat)
                ok = sync_local_draft_features(C - n, n);
        }
        if (!ok) {
            std::fprintf(stderr, "[kvflash] live restore: draft feature sync failed; cold prefill\n");
            return -1;
        }
    }
    return C;
}

bool Qwen35Backend::snapshot_save(int slot) {
    if (cfg_.paged_attention) {
        static bool warned = false;
        if (!warned) {
            std::fprintf(stderr,
                "[paged-attention] the classic single-sequence snapshot API "
                "is unavailable; continuous batching uses copied paged "
                "checkpoints\n");
            warned = true;
        }
        return false;
    }
    if (slot < 0 || slot >= PREFIX_SLOTS) return false;
    // kvflash: snapshots right-size to cur_pos, which is a LOGICAL position
    // that can exceed the physical pool once decode has paged, and dense
    // snapshots copy rows assuming the identity layout, which pooled prefill /
    // eviction breaks. Pooled state is saved as a pager blob instead
    // (LUCE_KVFLASH_RESTORE_CONSUME=1); without it the old refusal stands.
    if (kvflash_active() &&
        (cache_.cur_pos > kvflash_tokens_ || !kvflash_pager_.is_identity())) {
        if (!luce::common::kvflash_restore_consume_enabled()) {
            static bool warned = false;
            if (!warned) {
                std::fprintf(stderr, "[kvflash] snapshot skipped: cur_pos %d exceeds "
                                     "pool %d (set LUCE_KVFLASH_RESTORE_CONSUME=1 to save "
                                     "pooled snapshots)\n",
                             cache_.cur_pos, kvflash_tokens_);
                warned = true;
            }
            return false;
        }
        live_snapshot_slot_ = -1;
        return snapshot_save_pooled_at(slot, cache_.cur_pos);
    }
    // A deferred snapshot is the live state; copy it out while it still is.
    // Copying it out already saves this slot when it is the deferred one.
    const bool deferred_here = live_snapshot_deferred_ &&
        slot == live_snapshot_slot_ && cache_.cur_pos == live_snapshot_pos_;
    materialize_live_snapshot();
    const bool ok = deferred_here && prefix_snapshots_[slot].ctx
        ? true
        : snapshot_target_cache(w_, cache_, snap_backend_, prefix_snapshots_[slot]);
    live_snapshot_slot_ = ok && !generating_ ? slot : -1;
    return ok;
}

int Qwen35Backend::snapshot_granularity() const {
    return kQwen35OffGridLeadTokens;
}

bool Qwen35Backend::snapshot_save_deferred(int slot) {
    if (generating_ || cfg_.paged_attention || kvflash_active() ||
        slot < 0 || slot >= PREFIX_SLOTS || cache_.cur_pos <= 0) {
        return snapshot_save(slot);
    }
    materialize_live_snapshot();
    free_prefix_snapshot(prefix_snapshots_[slot]);
    live_snapshot_slot_ = slot;
    live_snapshot_pos_ = cache_.cur_pos;
    live_snapshot_deferred_ = true;
    return true;
}

void Qwen35Backend::materialize_live_snapshot() {
    if (!live_snapshot_deferred_) return;
    live_snapshot_deferred_ = false;
    const int slot = live_snapshot_slot_;
    if (cache_.cur_pos != live_snapshot_pos_ ||
        !snapshot_target_cache(w_, cache_, snap_backend_,
                               prefix_snapshots_[slot])) {
        free_prefix_snapshot(prefix_snapshots_[slot]);
        live_snapshot_slot_ = -1;
    }
}

void Qwen35Backend::snapshot_free(int slot) {
    if (slot < 0 || slot >= PREFIX_SLOTS) return;
    if (slot == live_snapshot_slot_) {
        live_snapshot_slot_ = -1;
        live_snapshot_deferred_ = false;
    }
    free_prefix_snapshot(prefix_snapshots_[slot]);
}

bool Qwen35Backend::snapshot_used(int slot) const {
    if (slot < 0 || slot >= PREFIX_SLOTS) return false;
    return prefix_snapshots_[slot].ctx != nullptr ||
           (live_snapshot_deferred_ && slot == live_snapshot_slot_);
}

bool Qwen35Backend::restore_target_cache_from_snapshot(int slot) {
    materialize_live_snapshot();
    live_snapshot_slot_ = -1;
    if (slot < 0 || slot >= PREFIX_SLOTS || !prefix_snapshots_[slot].ctx) return false;
    return restore_target_cache(prefix_snapshots_[slot], cache_);
}

int Qwen35Backend::snapshot_cur_pos(int slot) const {
    if (slot < 0 || slot >= PREFIX_SLOTS) return 0;
    if (live_snapshot_deferred_ && slot == live_snapshot_slot_) {
        return live_snapshot_pos_;
    }
    return prefix_snapshots_[slot].cur_pos;
}

// The arguments are the ones restore_and_generate_impl gives kvflash_restore_plan
// and kvflash_live_cut_peek, so the number reported is the cut that restore takes.
int Qwen35Backend::restorable_prefix(const std::vector<int32_t> & prompt, int slot,
                                     int slot_tokens) const {
    if (!kvflash_active()) return slot_tokens;
    const bool have = slot >= 0 && snapshot_used(slot);
    return kvflash_restorable_tokens(
        have, kvflash_restore_consume_enabled(),
        have && prefix_snapshots_[slot].is_pooled,
        have ? prefix_snapshots_[slot].cur_pos : 0, slot_tokens,
        kvflash_pager_.chunk_tokens(), (int)prompt.size(), kvflash_tokens_,
        kvflash_session_enabled(), kvflash_live_cut_peek(prompt));
}

size_t Qwen35Backend::snapshot_bytes_estimate(int tokens) const {
    // Paged serving keeps no single-sequence snapshots.
    if (cfg_.paged_attention || !snap_backend_ || tokens <= 0) return 0;
    return estimate_target_cache_snapshot_bytes(
        w_, cache_, tokens, ggml_backend_get_default_buffer_type(snap_backend_));
}

ModelBackend::MemoryReport Qwen35Backend::memory_report() const {
    MemoryReport report =
        memory_report_for(cache_, prefix_snapshots_, PREFIX_SLOTS);
    // Paged serving keeps sequences in blocks; cur_pos is not one of them.
    if (cfg_.paged_attention) report.cache.live_tokens = -1;
    // KVFlash bookkeeping and the DFlash feature mirror's staging buffer live
    // in host memory beside the cache.
    report.cache.host_state_bytes =
        kvflash_history_.capacity() * sizeof(int32_t) +
        kvflash_scores_.capacity() * sizeof(float) +
        kvflash_mask_buf_.capacity() * sizeof(uint16_t) +
        kvflash_qk_pool_.host_bytes() + feature_mirror_.staging_bytes;
    return report;
}

ModelBackend::MemoryReport Qwen35Backend::memory_report_for(
        const TargetCache & cache, const PrefixSnapshot * snapshots,
        int n_snapshots) {
    MemoryReport report;
    if (!cache.base_buf) return report;
    const auto bytes = [](const ggml_tensor * t) {
        return t ? (uint64_t) ggml_nbytes(t) : (uint64_t) 0;
    };
    report.available = true;
    report.cache.host = ggml_backend_buffer_is_host(cache.base_buf);
    report.cache.capacity_tokens = cache.max_ctx;
    report.cache.live_tokens = cache.slot_pool ? -1 : cache.cur_pos;
    for (size_t i = 0; i < cache.attn_k.size(); ++i) {
        report.cache.kv_bytes += bytes(cache.attn_k[i]) + bytes(cache.attn_v[i]);
    }
    for (size_t i = 0; i < cache.ssm_state.size(); ++i) {
        report.cache.recurrent_bytes +=
            bytes(cache.ssm_state[i]) + bytes(cache.conv_state[i]);
    }
    report.cache.draft_feature_bytes = bytes(cache.target_feat);
    uint64_t allocated = ggml_backend_buffer_get_size(cache.base_buf);
    if (cache.rollback_buf) {
        allocated += ggml_backend_buffer_get_size(cache.rollback_buf);
    }
    const uint64_t named = report.cache.kv_bytes + report.cache.recurrent_bytes +
                           report.cache.draft_feature_bytes;
    report.cache.other_bytes = allocated > named ? allocated - named : 0;
    for (int slot = 0; slot < n_snapshots; ++slot) {
        const auto & snap = snapshots[slot];
        if (!snap.ctx || !snap.buf) continue;
        report.snapshots.push_back({slot, snap.cur_pos,
                                    ggml_backend_buffer_get_size(snap.buf),
                                    ggml_backend_buffer_is_host(snap.buf)});
    }
    return report;
}

ModelBackend::SnapshotRef Qwen35Backend::snapshot_ref(int slot) const {
    SnapshotRef ref;
    if (slot < 0 || slot >= PREFIX_SLOTS) return ref;
    const auto & snap = prefix_snapshots_[slot];
    if (!snap.ctx) return ref;
    ref.ctx      = snap.ctx;
    ref.buf      = snap.buf;
    ref.cur_pos  = snap.cur_pos;
    ref.last_tok = snap.last_tok;
    ref.pooled   = snap.is_pooled;
    return ref;
}

// Hand a PrefixSnapshot's payload to an OwnedSnapshot and leave it empty.
// Only full states (dense, paged) are useful on disk; a thin KV-only one is
// freed instead, and so is a pooled one (its KV lives in the pager blob, which
// an OwnedSnapshot cannot carry). A payload outside host memory is copied to
// host first.
static OwnedSnapshot take_prefix_snapshot(PrefixSnapshot & snap) {
    if (!snap.ctx || !snap.buf || !snap.disk_persistable()) {
        free_prefix_snapshot(snap);
        return {};
    }
    OwnedSnapshot out;
    if (ggml_backend_buffer_is_host(snap.buf)) {
        out = OwnedSnapshot(snap.ctx, snap.buf, snap.cur_pos, snap.last_tok);
        snap.ctx = nullptr;
        snap.buf = nullptr;
    } else {
        out = OwnedSnapshot::clone_to_host(snap.ctx, snap.cur_pos, snap.last_tok);
    }
    free_prefix_snapshot(snap);
    return out;
}

OwnedSnapshot Qwen35Backend::snapshot_detach(int slot) {
    if (slot < 0 || slot >= PREFIX_SLOTS) return {};
    // A deferred snapshot is still the live state: copy it out first.
    if (live_snapshot_deferred_ && slot == live_snapshot_slot_) {
        materialize_live_snapshot();
    }
    if (slot == live_snapshot_slot_) {
        live_snapshot_slot_ = -1;
        live_snapshot_deferred_ = false;
    }
    return take_prefix_snapshot(prefix_snapshots_[slot]);
}

OwnedSnapshot Qwen35Backend::snapshot_take_displaced(int slot) {
    if (slot < 0 || slot >= PREFIX_SLOTS) return {};
    return take_prefix_snapshot(displaced_snapshots_[slot]);
}

bool Qwen35Backend::adopt_paged_snapshot(int slot, ggml_context * ctx,
                                         ggml_backend_buffer_t buf,
                                         int cur_pos) {
    snapshot_free(slot);
    auto & snap = prefix_snapshots_[slot];
    snap.attn_k_snap.assign(cache_.attn_k.size(), nullptr);
    snap.attn_v_snap.assign(cache_.attn_v.size(), nullptr);
    snap.ssm_state_snap.assign(cache_.ssm_state.size(), nullptr);
    snap.conv_state_snap.assign(cache_.conv_state.size(), nullptr);
    snap.target_feat_snap = nullptr;
    const char * mismatch = nullptr;
    for (ggml_tensor * t = ggml_get_first_tensor(ctx); t && !mismatch;
         t = ggml_get_next_tensor(ctx, t)) {
        size_t idx = 0;
        ggml_tensor ** dst = nullptr;
        if (std::sscanf(t->name, "snap_cache_k_%zu", &idx) == 1 &&
            idx < snap.attn_k_snap.size()) {
            dst = &snap.attn_k_snap[idx];
        } else if (std::sscanf(t->name, "snap_cache_v_%zu", &idx) == 1 &&
                   idx < snap.attn_v_snap.size()) {
            dst = &snap.attn_v_snap[idx];
        } else if (std::sscanf(t->name, "snap_ssm_state_%zu", &idx) == 1 &&
                   idx < snap.ssm_state_snap.size()) {
            dst = &snap.ssm_state_snap[idx];
        } else if (std::sscanf(t->name, "snap_conv_state_%zu", &idx) == 1 &&
                   idx < snap.conv_state_snap.size()) {
            dst = &snap.conv_state_snap[idx];
        }
        if (!dst || *dst) mismatch = "unexpected or duplicate tensor";
        else *dst = t;
    }
    // Every live layer needs its tensor, and KV rows must equal cur_pos;
    // restore_paged_target_cache() checks types and shapes against the
    // cache and the page table before it copies anything.
    for (size_t i = 0; i < cache_.attn_k.size() && !mismatch; ++i) {
        const bool present = cache_.attn_k[i] != nullptr;
        if ((snap.attn_k_snap[i] != nullptr) != present ||
            (snap.attn_v_snap[i] != nullptr) != present ||
            (present && (snap.attn_k_snap[i]->ne[1] != cur_pos ||
                         snap.attn_v_snap[i]->ne[1] != cur_pos))) {
            mismatch = "attention KV";
        }
    }
    for (size_t i = 0; i < cache_.ssm_state.size() && !mismatch; ++i) {
        const bool present = cache_.ssm_state[i] != nullptr;
        if ((snap.ssm_state_snap[i] != nullptr) != present ||
            (snap.conv_state_snap[i] != nullptr) != present) {
            mismatch = "recurrent state";
        }
    }
    if (mismatch || cur_pos <= 0 || cur_pos > cache_.max_ctx) {
        std::fprintf(stderr,
                     "[qwen35] paged snapshot adopt rejected slot=%d: %s\n",
                     slot, mismatch ? mismatch : "position");
        snap.attn_k_snap.clear(); snap.attn_v_snap.clear();
        snap.ssm_state_snap.clear(); snap.conv_state_snap.clear();
        return false;
    }
    snap.ctx = ctx;
    snap.buf = buf;
    snap.layout = PrefixSnapshot::Layout::paged;
    snap.cur_pos = cur_pos;
    snap.last_tok = -1;
    snap.kv_k_type = cache_.kv_k_type;
    snap.max_ctx = cache_.max_ctx;
    snap.target_feat_cap = 0;
    std::fprintf(stderr, "[qwen35] paged snapshot adopted slot=%d pos=%d\n",
                 slot, cur_pos);
    return true;
}

bool Qwen35Backend::snapshot_adopt(int slot, ggml_context * ctx,
                                   ggml_backend_buffer_t buf, int cur_pos,
                                   int32_t last_tok) {
    if (slot < 0 || slot >= PREFIX_SLOTS) return false;
    if (cfg_.paged_attention) {
        // Paged restores scatter from host memory (paged_snapshot_buffer_type).
        if (!ggml_backend_buffer_is_host(buf)) return false;
        return adopt_paged_snapshot(slot, ctx, buf, cur_pos);
    }
    snapshot_free(slot);

    auto & snap = prefix_snapshots_[slot];

    // Count expected tensor layout from weights.
    const int n_full_attn = w_.n_layer / w_.full_attention_interval;
    const int n_delta     = w_.n_layer - n_full_attn;

    snap.attn_k_snap.assign(n_full_attn, nullptr);
    snap.attn_v_snap.assign(n_full_attn, nullptr);
    snap.ssm_state_snap.assign(n_delta, nullptr);
    snap.conv_state_snap.assign(n_delta, nullptr);
    snap.target_feat_snap = nullptr;

    // Rebind tensors by name.
    for (ggml_tensor * t = ggml_get_first_tensor(ctx); t; t = ggml_get_next_tensor(ctx, t)) {
        if (!t->name[0]) continue;
        int idx = -1;
        if (std::sscanf(t->name, "snap_cache_k_%d", &idx) == 1 && idx >= 0 && idx < n_full_attn) {
            snap.attn_k_snap[idx] = t;
        } else if (std::sscanf(t->name, "snap_cache_v_%d", &idx) == 1 && idx >= 0 && idx < n_full_attn) {
            snap.attn_v_snap[idx] = t;
        } else if (std::sscanf(t->name, "snap_ssm_state_%d", &idx) == 1 && idx >= 0 && idx < n_delta) {
            snap.ssm_state_snap[idx] = t;
        } else if (std::sscanf(t->name, "snap_conv_state_%d", &idx) == 1 && idx >= 0 && idx < n_delta) {
            snap.conv_state_snap[idx] = t;
        } else if (std::strcmp(t->name, "snap_target_feat") == 0) {
            snap.target_feat_snap = t;
        }
    }

    // Validate all required tensors are present.
    for (int i = 0; i < n_full_attn; ++i) {
        if (!snap.attn_k_snap[i] || !snap.attn_v_snap[i]) {
            snap.attn_k_snap.clear(); snap.attn_v_snap.clear();
            snap.ssm_state_snap.clear(); snap.conv_state_snap.clear();
            snap.target_feat_snap = nullptr;
            return false;
        }
    }
    for (int i = 0; i < n_delta; ++i) {
        if (!snap.ssm_state_snap[i] || !snap.conv_state_snap[i]) {
            snap.attn_k_snap.clear(); snap.attn_v_snap.clear();
            snap.ssm_state_snap.clear(); snap.conv_state_snap.clear();
            snap.target_feat_snap = nullptr;
            return false;
        }
    }
    if (!snap.target_feat_snap) {
        snap.attn_k_snap.clear(); snap.attn_v_snap.clear();
        snap.ssm_state_snap.clear(); snap.conv_state_snap.clear();
        return false;
    }

    // Names alone do not prove compatibility: a snapshot taken with another
    // KV type or model shape would be restored as garbage. Every tensor must
    // match the live cache's type and per-row shape, and hold cur_pos rows.
    const auto same_rows = [](const ggml_tensor * s, const ggml_tensor * c) {
        return s && c && s->type == c->type && s->ne[0] == c->ne[0] &&
               s->ne[2] == c->ne[2] && s->ne[3] == c->ne[3];
    };
    const auto same_shape = [&](const ggml_tensor * s, const ggml_tensor * c) {
        return same_rows(s, c) && s->ne[1] == c->ne[1];
    };
    const char * mismatch = nullptr;
    for (int i = 0; i < n_full_attn && !mismatch; ++i) {
        const ggml_tensor * ck = i < (int)cache_.attn_k.size() ? cache_.attn_k[i] : nullptr;
        const ggml_tensor * cv = i < (int)cache_.attn_v.size() ? cache_.attn_v[i] : nullptr;
        if (!attn_snapshot_adoptable(snap.attn_k_snap[i], ck, cur_pos) ||
            !attn_snapshot_adoptable(snap.attn_v_snap[i], cv, cur_pos)) {
            mismatch = "attention KV";
        }
    }
    for (int i = 0; i < n_delta && !mismatch; ++i) {
        const ggml_tensor * cs = i < (int)cache_.ssm_state.size() ? cache_.ssm_state[i] : nullptr;
        const ggml_tensor * cc = i < (int)cache_.conv_state.size() ? cache_.conv_state[i] : nullptr;
        if (!same_shape(snap.ssm_state_snap[i], cs) ||
            !same_shape(snap.conv_state_snap[i], cc)) {
            mismatch = "recurrent state";
        }
    }
    if (!mismatch && cache_.target_feat &&
        (snap.target_feat_snap->type != cache_.target_feat->type ||
         snap.target_feat_snap->ne[0] != cache_.target_feat->ne[0] ||
         snap.target_feat_snap->ne[1] !=
             std::min<int64_t>(cur_pos, cache_.target_feat_cap))) {
        mismatch = "target features";
    }
    if (mismatch) {
        std::fprintf(stderr,
                     "[qwen35] snapshot adopt rejected slot=%d: %s type/shape "
                     "differs from the live cache\n", slot, mismatch);
        snap.attn_k_snap.clear(); snap.attn_v_snap.clear();
        snap.ssm_state_snap.clear(); snap.conv_state_snap.clear();
        snap.target_feat_snap = nullptr;
        return false;
    }

    snap.ctx     = ctx;
    snap.buf     = buf;
    snap.layout  = PrefixSnapshot::Layout::dense;  // restore_target_cache requires it
    snap.cur_pos = cur_pos;
    snap.last_tok        = last_tok;
    snap.kv_k_type       = cache_.kv_k_type;
    snap.max_ctx         = cache_.max_ctx;
    snap.target_feat_cap = cache_.target_feat_cap;
    std::fprintf(stderr, "[qwen35] snapshot adopted slot=%d pos=%d last_tok=%d\n", slot, cur_pos, last_tok);
    return true;
}

// KVFlash slots (C1.3-05): N slot pagers over one arena (slice i = rows
// [i*Q*64, (i+1)*Q*64), a scratch block after them), their shared page stream,
// the host reservation ledger, the SlotKv and the step driver, then the engine
// over them. Headroom 8 blocks (D2). The host budget is LUCE_KVFLASH_HOST_MIB
// (default 16384): reservations beyond it are refused at admission (R5).
bool Qwen35Backend::init_kvflash_slots(int n_slots, int quota_chunks, int max_ctx, int64_t scratch_row,
                                       int max_prefills, int mixed, int long_mixed,
                                       int long_threshold, int idle) {
    constexpr int kHeadroom = 8;
    using namespace luce::common;
    kvf_arena_ = std::make_unique<KvFlashArena>();
    std::vector<KvFlashSlotPager *> raw;
    for (int i = 0; i < n_slots; i++) {
        kvf_pagers_.push_back(std::make_unique<KvFlashSlotPager>());
        kvf_arena_->add(*kvf_pagers_.back());
        if (!kvf_pagers_.back()->attach_slice(i, n_slots, quota_chunks, KvFlashConfig{}, cache_.attn_k, cache_.attn_v)) {
            set_last_error("KVFlash slot " + std::to_string(i) + ": pager attach failed (quota " +
                           std::to_string(quota_chunks) + " chunks, headroom " + std::to_string(kHeadroom) + ")");
            return false;
        }
        raw.push_back(kvf_pagers_.back().get());
    }
    {   // P2-11: the host backing policy reaches every slot pager before any restore or eviction allocates
        const KvFlashHostBacking backing = kvflash_host_backing_from_env();
        kvflash_apply_host_backing(raw, backing);
        std::fprintf(stderr, "[kvflash] host backing policy: %s (%d slots)\n",
                     backing == KvFlashHostBacking::Pinned ? "pinned" : backing == KvFlashHostBacking::Pageable ? "pageable" : "auto",
                     (int)raw.size());
    }
    KvFlashSlotKvConfig kc;
    kc.quota_chunks = quota_chunks;
    kc.headroom_blocks = kHeadroom;
    kc.max_ctx = max_ctx;
    {   // host memory per request: the R state snapshot of a recall turn, the rungs (checkpoint, trunk and the third) of a
        // sticky session, and the disk bundle's staging at park (the end slab snapshot and its pack, the two rung slab
        // packs, the changed chunks, the head)
        int64_t slab = 0;
        for (const ggml_tensor * t : cache_.ssm_state)  if (t) slab += (int64_t)ggml_nbytes(t) / n_slots;
        for (const ggml_tensor * t : cache_.conv_state) if (t) slab += (int64_t)ggml_nbytes(t) / n_slots;
        const char * sticky = std::getenv("LUCE_KVFLASH_SLOT_STICKY");
        const bool sticky_on = !(sticky && sticky[0] == '0' && sticky[1] == '\0');
        const char * bdir = std::getenv("LUCE_KVFLASH_SLOT_BUNDLE_DIR");
        int64_t n_slab_tensors = 0;
        for (const ggml_tensor * t : cache_.ssm_state)  n_slab_tensors += t != nullptr;
        for (const ggml_tensor * t : cache_.conv_state) n_slab_tensors += t != nullptr;
        // the park staging (C1.5 F3, kvflash_park_staging): the end slab snapshot and its packed file, the packed checkpoint and
        // trunk slabs, the records of the changed chunks (at most every chunk), the ledger and the head, held until the writer
        // thread has the park on disk (one staged park per slot: save_bundle_ waits for the slot's earlier one)
        const int64_t chunk_tokens = raw.back()->chunk_tokens();
        const int64_t chunk_bytes = (int64_t)raw.back()->chunk_bytes();
        const KvFlashParkStaging staging = kvflash_park_staging(slab, n_slab_tensors, (int64_t)raw.back()->serialized_bytes(0), chunk_tokens, chunk_bytes);
        const bool staged = sticky_on && bdir && *bdir;
        kc.ladder_bytes = kvflash_ladder_bytes(slab, kvflash_slot_recall_profile() && !kvflash_pflash2_scorer_, sticky_on, staged, staging);
        kc.stage_chunk_bytes = staged ? staging.per_chunk_bytes : 0;   // eviction stage 0, F2: the session's own chunks, not nc_max of them (a 2K session was charged 4.97 GiB at --max-ctx 135168)
        std::fprintf(stderr, "[kvflash] per-slot ladder: %d rungs x slab %lld B + staging %lld B = %lld B (%.1f MiB)\n", sticky_on ? kKvFlashLadderRungs : 0,
                     (long long)slab, (long long)(staged ? staging.fixed_bytes : 0), (long long)kc.ladder_bytes, kc.ladder_bytes / 1048576.0);
        std::fprintf(stderr, "[kvflash] evict0 reservation: slab=%lld ladder=%lld stage_chunk=%lld (staging at max_ctx would add %lld) host_budget_mib=%d slots=%d\n",
                     (long long)slab, (long long)kc.ladder_bytes, (long long)kc.stage_chunk_bytes,
                     (long long)(staged ? kvflash_park_staging_max_bytes(staging, max_ctx, chunk_tokens) - staging.fixed_bytes : 0),
                     env_int_or_default("LUCE_KVFLASH_HOST_MIB", 16384), n_slots);
    }
    if (!kvflash_headroom_ok(quota_chunks, kHeadroom)) {
        set_last_error("KVFlash quota of " + std::to_string(quota_chunks) + " chunks leaves no evictable block beside a headroom of 8");
        return false;
    }
    kvf_ledger_ = std::make_unique<KvFlashHostLedger>(
        (int64_t)env_int_or_default("LUCE_KVFLASH_HOST_MIB", 16384) << 20, /*transient_reserve_bytes=*/0);
    kvf_kv_ = std::make_unique<KvFlashSlotKv>(raw, *kvf_ledger_, kc);
    kvf_driver_ = std::make_unique<KvFlashStepDriver>(raw, *kvf_arena_, kHeadroom);
    seq_engine_ = std::make_unique<Qwen35SeqEngine>(
        *this, *kvf_kv_, *kvf_driver_, max_ctx, scratch_row, max_prefills, mixed, long_mixed, long_threshold, idle);
    return true;
}

// ── Compress (pflash) ───────────────────────────────────────────────────

ModelBackend::CompressResult Qwen35Backend::compress(const CompressRequest & req) {
    const auto results = compress_batch({req});
    return results.empty() ? CompressResult{} : results.front();
}

std::vector<ModelBackend::CompressResult> Qwen35Backend::compress_batch(
        const std::vector<CompressRequest> & requests) {
    std::vector<CompressResult> results(requests.size());
    if (requests.empty()) return results;

    const CompressRequest * load_request = nullptr;
    for (const auto & request : requests) {
        if (request.input_ids.empty() || request.drafter_path.empty()) continue;
        if (load_request == nullptr) {
            load_request = &request;
        } else if (request.drafter_path != load_request->drafter_path ||
                   request.drafter_gpu != load_request->drafter_gpu ||
                   request.skip_park != load_request->skip_park ||
                   request.residency_action != load_request->residency_action) {
            // A residency window can host only one drafter. Preserve the base
            // API's per-request behavior for heterogeneous batches.
            return ModelBackend::compress_batch(requests);
        }
    }
    if (load_request == nullptr) return results;
    const bool should_park = !load_request->skip_park;
    const bool release_after_use =
        load_request->residency_action == DraftResidencyAction::ReleaseAfterUse;

    // Park target+draft to free VRAM for the drafter (unless skip_park).
    // A FlowKV request may contain many aged messages. Keep this residency
    // window around the whole batch so those messages do not reload all three
    // models independently.
    const bool was_target_parked = target_parked_;
    const bool was_draft_parked  = draft_parked_;
    if (should_park) {
        step_graph_destroy(sg_);
        if (!target_parked_) park(ParkTarget::TargetModel);
        if (!draft_parked_)  park(ParkTarget::DraftModel);
    }

    // Synchronize all backends to flush any outstanding async CUDA work
    // before loading the drafter. Without this, pending operations on
    // target/draft streams can corrupt the drafter's allocations. Only
    // parking frees their memory: a drafter beside resident models
    // allocates its own, and concurrent serving scores on a thread of its
    // own while the scheduler keeps the target busy.
    if (should_park) {
        ggml_backend_synchronize(target_backend_);
        if (draft_backend_) ggml_backend_synchronize(draft_backend_);
    }

    // Load drafter with its OWN backend (not target_backend_).
    // Matches test_dflash.cpp: separate backend supports multi-GPU
    // (drafter on GPU A, target on GPU B or CPU).
    // The drafter stays loaded across compress calls — only the first call
    // creates the backend + loads weights; subsequent calls reuse them.
    if (!drafter_loaded_) {
        // drafter_ctx_.backend == nullptr → load_drafter creates its own
        std::fprintf(stderr, "[compress] loading drafter from %s ...\n",
                     load_request->drafter_path.c_str());
        if (!load_drafter(load_request->drafter_path, /*gpu_layers=*/999,
                          load_request->drafter_gpu, drafter_ctx_)) {
            std::fprintf(stderr, "[compress] drafter init failed: %s\n",
                         luce_last_error());
            if (should_park) {
                if (!was_target_parked) unpark(ParkTarget::TargetModel);
                if (!was_draft_parked)  unpark(ParkTarget::DraftModel);
            }
            return results;
        }
        drafter_loaded_ = true;
        drafter_ctx_.engine_max_ctx = cfg_.device.max_ctx;   // scoring sessions are never sized past it
        std::fprintf(stderr, "[compress] drafter ready\n");
        // pflash + kvflash synergy: the drafter doubles as the pool's
        // Memory Indexer (tau-step reselect). Pager stays LRU without it.
        if (kvflash_active() && !kvflash_scorer_) {
            kvflash_scorer_ = std::make_unique<KvFlashDrafterScorer>(&drafter_ctx_);
            std::fprintf(stderr, "[kvflash] drafter scorer attached (tau=%d)\n",
                         kvflash_tau_);
        }
    }

    for (size_t index = 0; index < requests.size(); ++index) {
        const auto & request = requests[index];
        if (request.input_ids.empty() || request.drafter_path.empty()) continue;

        auto & result = results[index];
        // score_query_end < 0 is the legacy "tail window" request value;
        // the qwen35 scorer requires an explicit end.
        const int score_query_end = request.score_query_end >= 0
            ? request.score_query_end : (int)request.input_ids.size();
        result.compressed_ids = drafter_score_and_compress(
            drafter_ctx_, request.input_ids, request.keep_ratio,
            /*chunk_size=*/32, request.score_query_tokens, /*pool_kernel=*/13,
            score_query_end, request.required_instruction_spans,
            request.query_suffix_candidates, request.history_query_spans,
            request.turn_query_span, request.extra_query_spans,
            &request.request_messages, request.keep_query_spans);
        result.ok = !result.compressed_ids.empty();
        if (result.ok) result.kept_spans = pflash_last_kept_spans();
        if (result.ok) {
            const auto & scoring = pflash_last_scoring_stats();
            result.scorer_resume = scoring.resume;
            result.scorer_new_tokens = scoring.new_tokens;
            result.scorer_forward_s = scoring.forward_s;
            const auto & stage2 = pflash_last_stage2_outcome();
            result.budget_widened = stage2.widened;
            result.raw_fits = stage2.raw_fits;
            result.stage2_budget = stage2.budget;
            result.stage2_budget_widened = stage2.budget_widened;
            const auto & spread = pflash_last_relevance_spread();
            result.spread_valid = spread.valid;
            result.spread = spread.spread;
            result.spread_k = spread.k;
            for (const auto & candidate : pflash_last_candidate_lifts()) {
                result.candidate_lifts.push_back({candidate.span, candidate.lift});
            }
            result.probe_pins = pflash_last_probe_pins();
        }
        if (result.ok) {
            std::fprintf(stderr, "[compress] %zu -> %zu tokens\n",
                         request.input_ids.size(), result.compressed_ids.size());
        }
    }

    if (release_after_use) {
        // Scoring sessions outlive the release in host RAM; the next scoring
        // call restores them once the drafter is loaded again.
        if (drafter_loaded_) luce::common::drafter_offload_sessions(drafter_ctx_);
        free_drafter();
    } else if (drafter_loaded_ && drafter_ctx_.backend &&
               drafter_ctx_.backend != target_backend_ &&
               drafter_ctx_.backend != draft_backend_) {
        // A resident drafter owns its backend: on non-VMM devices that
        // backend's pool keeps every operator temporary of the last scoring
        // run (GiBs after a long prompt) until it is trimmed, and
        // release_scratch() trims only the target and draft pools. Scoring
        // sessions are tensors of their own and are unaffected.
        const char * keep = std::getenv("PFLASH_DRAFTER_TRIM_POOL");
        if (!(keep && std::string(keep) == "0")) {
            const size_t trimmed = ggml_backend_cuda_trim_pool(drafter_ctx_.backend);
            std::fprintf(stderr, "[compress] trimmed %.1f MiB from the drafter pool\n",
                         trimmed / (1024.0 * 1024.0));
        }
    }

    // Restore park state
    if (should_park) {
        if (!was_target_parked) unpark(ParkTarget::TargetModel);
        if (!was_draft_parked)  unpark(ParkTarget::DraftModel);
    }

    return results;
}

bool Qwen35Backend::handle_compress(const std::string & line, const DaemonIO & io) {
    materialize_live_snapshot();
    live_snapshot_slot_ = -1;
    // Check for "nopark" suffix (must be a separate token, not part of a path)
    bool skip_park = (line.size() >= 16 &&
                      line.compare(line.size() - 7, 7, " nopark") == 0);

    // Parse: "compress <path> <keep_x1000> <drafter_gguf> [nopark]"
    char ppath[1024];
    int  keep_x1000 = 0;
    char drafter_path[1024] = {0};
    const int n = std::sscanf(line.c_str() + 9, "%1023s %d %1023s",
                               ppath, &keep_x1000, drafter_path);
    if (n < 2) {
        std::fprintf(stderr, "[compress] bad args\n");
        io.emit(-1);
        return false;
    }

    CompressRequest req;
    req.input_ids = read_int32_file(ppath);
    req.keep_ratio = (float)keep_x1000 / 1000.0f;
    req.drafter_path = (n >= 3 && drafter_path[0])
        ? drafter_path
        : "/opt/lucebox/models/drafter/Qwen3.5-0.8B-BF16.gguf";
    {
        size_t total_vram = 0;
        int dev = 0;
        cudaGetDevice(&dev);
        cudaDeviceProp prop{};
        if (cudaGetDeviceProperties(&prop, dev) == cudaSuccess)
            total_vram = prop.totalGlobalMem;
        const bool allowed = luce::common::skip_park_allowed(
            skip_park, total_vram, cfg_.device.max_ctx);
        if (skip_park && !allowed) {
            std::fprintf(stderr,
                "[server] --prefill-skip-park downgraded: <32GB GPU with max_ctx>65536"
                " (VMM VA-fragmentation guard)\n");
        }
        req.skip_park = allowed;
    }

    CompressResult result = compress(req);
    for (int32_t t : result.compressed_ids) io.emit(t);
    io.emit(-1);
    return result.ok;
}

void Qwen35Backend::free_drafter() {
    if (drafter_loaded_) {
        // The kvflash DRAFTER scorer borrows drafter_ctx_; drop it first.
        // The target-QK scorer is drafter-independent and survives.
        if (!kvflash_qk_scorer_) kvflash_scorer_.reset();
        kvflash_pflash2_.reset();
        // Drafter has its own backend — do a full free (weights + backend)
        luce::common::free_drafter(drafter_ctx_);
        drafter_loaded_ = false;
        std::printf("[drafter] freed\n"); std::fflush(stdout);
    }
}

// ── try_handle_command (arch-specific) ──────────────────────────────────

bool Qwen35Backend::try_handle_command(const std::string & line, const DaemonIO & io) {
    // SNAPSHOT_THIN <slot> — lightweight snapshot (SSM state only, no KV copy)
    if (line.compare(0, 14, "SNAPSHOT_THIN ") == 0) {
        if (cfg_.paged_attention) {
            std::fprintf(stderr,
                "[paged-attention] SNAPSHOT_THIN is not page-table aware\n");
            io.emit(-1);
            return true;
        }
        int slot = std::atoi(line.c_str() + 14);
        if (slot >= 0 && slot < PREFIX_SLOTS) {
            snapshot_free(slot);
            PrefixSnapshot & snap = prefix_snapshots_[slot];
            snapshot_target_cache_thin(w_, cache_, snap_backend_,
                                       /*kv_start=*/0, /*kv_end=*/cache_.cur_pos, snap);
            std::printf("[snapshot_thin] slot=%d pos=%d\n", slot, snap.cur_pos);
            std::fflush(stdout);
        }
        io.emit(-1);
        return true;
    }

    return false;
}

// ── DFlash spec decode target ────────────────────────────────────────────

DFlashTarget * Qwen35Backend::dflash_target() {
    if (cfg_.paged_attention) return nullptr;
    if (!dflash_target_) {
        dflash_target_ = std::make_unique<Qwen35DFlashTarget>(
            w_, cache_, target_backend_, sg_,
            cfg_.kq_stride_pad, cfg_.fa_window);
        auto * qt = static_cast<Qwen35DFlashTarget *>(dflash_target_.get());
        if (kvflash_active()) {
            qt->set_kvflash_pager(&kvflash_pager_);
        }
        qt->set_fast_rollback(cfg_.fast_rollback);
        qt->set_rope_offset(&rope_delta_);
    }
    return dflash_target_.get();
}

// ── Shutdown ────────────────────────────────────────────────────────────

void Qwen35Backend::shutdown() {
    const bool use_remote_draft = cfg_.remote_draft.enabled();
    seq_engine_.reset();
    end_paged_sequence();
    free_drafter();
    step_graph_destroy(sg_);
    step_graph_destroy(draft_sg_);
    step_graph_destroy(proj_sg_);
    remote_draft_.close();
    draft_kv_free(draft_kv_);
    draft_feature_mirror_free(feature_mirror_);
    for (int i = 0; i < PREFIX_SLOTS; i++) {
        free_prefix_snapshot(prefix_snapshots_[i]);
        free_prefix_snapshot(displaced_snapshots_[i]);
    }
    state_ladder_clear();
    if (!target_parked_) free_target_weights(w_);
    vision_.reset();  // its buffers belong to target_backend_, freed below
    if (!use_remote_draft && !draft_parked_) free_draft_weights(dw_);
    free_target_cache(cache_);
    kvflash_gpu_mass_.reset();   // its buffers belong to target_backend_, freed below
    if (kvflash_exact_pin_) { ggml_backend_buffer_free(kvflash_exact_pin_); kvflash_exact_pin_ = nullptr; }
    if (split_gpus_ && draft_backend_) {
        ggml_backend_free(draft_backend_);
        draft_backend_ = nullptr;
    }
    if (target_backend_) {
        ggml_backend_free(target_backend_);
        target_backend_ = nullptr;
    }
    tensor_parallel_.reset();
    if (snap_backend_) {
        free_snapshot_backend(snap_backend_, target_backend_);
        snap_backend_ = nullptr;
    }
}

// ── Release scratch buffers between requests ────────────────────────────

void Qwen35Backend::release_scratch() {
    // Target graph allocator: grows during large prefill batches, not needed
    // between requests. Will be lazily recreated on next build_target_step().
    if (sg_.alloc) {
        ggml_gallocr_free(sg_.alloc);
        sg_.alloc = nullptr;
    }
    step_graph_free(sg_);

    // LM-head projection allocator (same pattern).
    if (proj_sg_.alloc) {
        ggml_gallocr_free(proj_sg_.alloc);
        proj_sg_.alloc = nullptr;
    }
    step_graph_free(proj_sg_);

    // BSA persistent CUDA buffers (blockmask, head_mask_type, softmax_lse).
#ifdef LUCE_HAVE_BSA
    flashprefill::dflash_bsa_free_persistent();
#endif

    // Gallocr teardown does not release operator temporaries cached by the
    // legacy CUDA/HIP pools. On non-VMM devices a long prefill can otherwise
    // leave nearly all VRAM reserved and make the next differently shaped
    // request fail despite having no live scratch tensors.
    size_t trimmed = ggml_backend_cuda_trim_pool(target_backend_);
    if (draft_backend_ && draft_backend_ != target_backend_) {
        trimmed += ggml_backend_cuda_trim_pool(draft_backend_);
    }

    std::fprintf(stderr, "[vram] released scratch buffers; trimmed %.1f MiB from device pools\n",
                 trimmed / (1024.0 * 1024.0));
}

// ── Generate (speculative decode) ───────────────────────────────────────

GenerateResult Qwen35Backend::generate_impl(const GenerateRequest & req,
                                            const DaemonIO & io) {
    materialize_live_snapshot();
    live_snapshot_slot_ = -1;
    generating_ = true;
    struct GeneratingScope {
        bool & flag;
        ~GeneratingScope() { flag = false; }
    } generating_scope{generating_};
    GenerateResult result;
    DaemonIO out_io = io.with_token_callback(req.on_token);
    if (seq_engine_wanted()) {
        // begin_paged_sequence would reset the shared pool under every live
        // slot; the scheduler must use the seq_* API instead.
        result.fail(GenerateErrorCode::BackendSpecific,
                    "generate() is unavailable with --max-concurrency; "
                    "use the concurrent slot API");
        return result;
    }
    sampler_ = req.sampler;
    if (req.do_sample && sampler_.seed != 0) {
        sampler_rng_.seed(sampler_.seed);
    }

    // Zero delta-net recurrent state (SSM + conv) so a fresh prompt doesn't
    // inherit stale hidden state from the previous request. KV cache is
    // position-addressed and will be overwritten during prefill.
    // S8a: resume from the live session when a ladder rung covers the common
    // prefix; the state checkpoint replaces the zeroing.
    ReselectRequestScope reselect_scope(kvflash_pager_, kvflash_scores_);
    int live_cut = -1;
    if (kvflash_session_enabled() && kvflash_active() && !cfg_.paged_attention &&
        !req.images) {
        live_cut = kvflash_live_restore(req.prompt);
    }
    if (live_cut < 0) reset_recurrent_state(cache_);

    if (cfg_.paged_attention) {
        // The pool's block rounding can hold up to PAGED_BLOCK_SIZE-1 tokens
        // beyond max_ctx; gate on max_ctx itself so no logical (RoPE)
        // position ever exceeds the configured context.
        if (req.prompt.size() > (uint64_t)cfg_.device.max_ctx) {
            result.fail(GenerateErrorCode::ContextOverflow,
                        "prompt exceeds max_ctx for paged attention");
            return result;
        }
        if (!begin_paged_sequence((uint32_t)req.prompt.size())) {
            result.fail(GenerateErrorCode::BackendSpecific,
                        "paged KV sequence initialization failed");
            return result;
        }
    }
    // Release the paged sequence on every exit (no-op in dense mode).
    struct PagedSequenceGuard {
        Qwen35Backend * backend;
        ~PagedSequenceGuard() { backend->end_paged_sequence(); }
    } paged_guard{this};

    // Prefill
    auto t_prefill_start = std::chrono::steady_clock::now();
    Qwen35ImageRows image_rows;
    if (req.images) {
        const auto * prompt = dynamic_cast<const Qwen35ImagePrompt *>(req.images.get());
        std::string image_error = "image binding does not match the prompt";
        if (!prompt || !prompt->matches(req.prompt) ||
            !encode_images(*prompt, image_rows, image_error)) {
            result.fail(GenerateErrorCode::BackendSpecific, image_error);
            return result;
        }
    }
    const bool has_images = image_rows.prompt != nullptr;
    int committed;
    if (live_cut > 0) {
        // Prefill only prompt[C:] on the grid, continuing the live pager.
        struct LiveScope {
            bool & live; bool & consume;
            ~LiveScope() { live = false; consume = false; }
        } live_scope{kvflash_live_restore_active_, kvflash_consume_restore_};
        kvflash_live_restore_active_ = true;
        kvflash_consume_restore_ = true;
        const std::vector<int32_t> delta(req.prompt.begin() + live_cut, req.prompt.end());
        committed = do_prefill(delta, out_io, req.snap_pos, req.snap_slot,
                               /*kv_offset=*/live_cut, nullptr, req.restore_points,
                               &req.prompt);
        // Publish the cut actually made (the server reports it as pt/prefix_len).
        result.restored_prefix_tokens = live_cut;
        result.restore_cut_reported = true;
    } else {
        committed = do_prefill(req.prompt, out_io, req.snap_pos, req.snap_slot,
                               /*kv_offset=*/0, has_images ? &image_rows : nullptr,
                               req.restore_points);
    }
    if (committed < 0) {
        result.fail(GenerateErrorCode::PrefillFailed);
        return result;
    }
    auto t_prefill_end = std::chrono::steady_clock::now();
    result.prefill_s = std::chrono::duration<double>(t_prefill_end - t_prefill_start).count();
    if (out_io.is_cancelled()) {
        result.succeed();
        return result;
    }

    // Decode (speculative)
    if (req.n_gen > 0) {
        auto t_decode_start = std::chrono::steady_clock::now();
        // One controller accounts for every output candidate across spec and AR.
        bool decode_ok = false;
        int ar_n_gen = req.n_gen;
        if (cfg_.paged_attention) {
            // The paged pool ends at the block-rounded max_ctx: an unclamped
            // budget would fail a mid-stream block allocation (and push RoPE
            // positions past max_ctx) instead of finishing cleanly. The HTTP
            // admission gate normally guarantees prompt+n_gen <= max_ctx, but
            // daemon-command callers are not required to.
            //
            // AR decode emits its first token from the prefill logits without
            // writing a K/V row, so the last position it forwards is
            // committed+n_gen-2: one token more than the remaining context
            // still fits.
            const int max_ar_n_gen = cfg_.device.max_ctx - committed + 1;
            if (max_ar_n_gen <= 0) {
                // do_ar_decode would return true on a non-positive count and
                // report an empty completion as success.
                result.fail(GenerateErrorCode::ContextOverflow,
                            "no context left for generation after prefill");
                return result;
            }
            if (ar_n_gen > max_ar_n_gen) {
                ar_n_gen = max_ar_n_gen;
                std::fprintf(stderr,
                    "[paged-attention] clamping n_gen %d -> %d "
                    "(committed=%d, max_ctx=%d)\n",
                    req.n_gen, ar_n_gen, committed, cfg_.device.max_ctx);
            }
        }
        ThinkingBudget budget(req.budget_hook, ar_n_gen);
        // Image requests speculate too: the verify target shifts its rotary
        // positions by rope_delta_, and the drafter only proposes tokens.
        if (cfg_.paged_attention || req.force_ar_decode) {
            decode_ok = do_ar_decode(committed, budget, result.tokens, out_io,
                                     &result.budget_forced_close,
                                     &result.degenerate_decode_close);
            out_io.emit(-1);
        } else {
            const auto * hint_tokens =
                req.hint_tokens.empty() ? nullptr : &req.hint_tokens;
            const auto * stall_prefix = req.stall_tool_prefix_tokens.empty()
                ? nullptr : &req.stall_tool_prefix_tokens;
            const auto * stall_suffix = req.stall_action_suffix_tokens.empty()
                ? nullptr : &req.stall_action_suffix_tokens;
            const auto * stall_skip = req.stall_skip_tokens.empty()
                ? nullptr : &req.stall_skip_tokens;
            decode_ok = do_spec_decode(committed, budget, result.tokens, out_io,
                                       result.accept_rate, result.spec_decode_ran,
                                       hint_tokens, stall_prefix, stall_suffix,
                                       stall_skip,
                                       &result.budget_forced_close,
                                       &result.degenerate_decode_close);
            if (decode_ok) {
                result.empty_visible_output =
                    qwen35_empty_visible_output(result.tokens, w_);
            }
        }
        if (!decode_ok) {
            result.fail(GenerateErrorCode::DecodeFailed);
            return result;
        }
        result.decode_s = std::chrono::duration<double>(
            std::chrono::steady_clock::now() - t_decode_start).count();
    }

    // S9: checkpoint the DeltaNet state at the generation end so the next turn
    // resumes after the answer. Speculative rollback and the early-stop repair
    // (EOS, cancel inside a verified block) leave cur_pos at the committed
    // position, where KV and state hold exactly history[0:G].
    if (kvflash_session_enabled() && kvflash_active() && !cfg_.paged_attention &&
        !has_images && req.n_gen > 0) {
        const int G = kvflash_gen_ckpt_pos(cache_.cur_pos, kvflash_history_.size());
        if (G > 0 && state_ladder_capture(G)) {
            kvflash_gen_ckpt_pos_ = G;
            std::printf("[kvflash] gen ckpt @%d\n", G);
            std::fflush(stdout);
        } else {
            std::fprintf(stderr, "[kvflash] gen ckpt skipped (cur_pos=%d history=%zu)\n",
                         cache_.cur_pos, kvflash_history_.size());
        }
    }

    result.succeed();
    return result;
}

// ── Restore + generate ──────────────────────────────────────────────────

GenerateResult Qwen35Backend::restore_and_generate_impl(int slot,
                                                        const GenerateRequest & req,
                                                        const DaemonIO & io) {
    // The live cache already is this snapshot when it was saved (or kept,
    // deferred) after the previous generation and nothing ran since.
    const bool live_restore = slot >= 0 && slot == live_snapshot_slot_ &&
        !kvflash_active() &&
        (live_snapshot_deferred_
            ? cache_.cur_pos == live_snapshot_pos_
            : prefix_snapshots_[slot].ctx &&
              cache_.cur_pos == prefix_snapshots_[slot].cur_pos);
    GenerateResult result;
    DaemonIO out_io = io.with_token_callback(req.on_token);
    ReselectRequestScope reselect_scope(kvflash_pager_, kvflash_scores_);
    if (cfg_.paged_attention) {
        result.fail(GenerateErrorCode::BackendSpecific,
                    "paged-attention snapshots are not yet supported");
        out_io.emit(-1);
        return result;
    }
    if (req.images) {
        // A snapshot is keyed by tokens, and pad tokens do not identify an image.
        result.fail(GenerateErrorCode::BackendSpecific,
                    "image requests cannot resume from a snapshot");
        out_io.emit(-1);
        return result;
    }
    if (slot < 0 || slot >= PREFIX_SLOTS ||
        (!live_restore && !prefix_snapshots_[slot].ctx)) {
        result.fail(GenerateErrorCode::InvalidSnapshotSlot);
        out_io.emit(-1);
        return result;
    }
    // A restore that other state overwrites copies a deferred snapshot out
    // first. Continuing the live state keeps it until prefill or decode
    // changes that state, so a failure before then leaves it restorable.
    if (!live_restore) {
        materialize_live_snapshot();
        live_snapshot_slot_ = -1;
    }
    generating_ = true;
    struct GeneratingScope {
        bool & flag;
        ~GeneratingScope() { flag = false; }
    } generating_scope{generating_};
    // An exact snapshot hit decodes without a prefill, so the offset a
    // previous image request left behind must not survive into this one.
    rope_delta_ = 0;
    const int snap_pos = live_restore ? cache_.cur_pos
                                      : prefix_snapshots_[slot].cur_pos;

    // A KVFlash pooled snapshot either feeds the suffix-only prefill (consume,
    // LUCE_KVFLASH_RESTORE_CONSUME=1) or is dropped for a full re-prefill from 0.
    // It is never read as a dense snapshot and never half-applied.
    const bool snap_pooled = !live_restore && prefix_snapshots_[slot].is_pooled;
    const KvflashRestorePlan plan = kvflash_restore_plan(
        kvflash_restore_consume_enabled(), snap_pooled, snap_pos,
        kvflash_active() ? kvflash_pager_.chunk_tokens() : 0,
        !req.prompt.empty(), (int)req.prompt.size(),
        kvflash_active() ? kvflash_tokens_ : 0,
        kvflash_active() && kvflash_session_enabled());

    // The live KVFlash session may cover more of the prompt than the server's
    // snapshot (or the snapshot cannot be used at all): take the larger cut,
    // never a full re-prefill while the live session matches.
    {
        const int live_cut = kvflash_live_cut_peek(req.prompt);
        const int server_cut = kvflash_server_cut(plan, snap_pos);
        if (kvflash_pick_resume(server_cut, live_cut) == KvflashResume::Live) {
            std::printf("[kvflash] resume: live cut %d beats server cut %d (slot %d)\n",
                        live_cut, server_cut, slot);
            std::fflush(stdout);
            return generate_impl(req, io);
        }
    }
    if (plan == KvflashRestorePlan::Reject) {
        result.fail(GenerateErrorCode::BackendSpecific,
                    "kvflash: snapshot cannot be restored (misaligned pooled or longer than the pool, or "
                    "consume off) and the request carries no prompt to re-prefill");
        out_io.emit(-1);
        return result;
    }
    const bool consume = plan == KvflashRestorePlan::Consume;
    const bool adopt = plan == KvflashRestorePlan::AdoptDense;
    const bool full_reprefill = plan == KvflashRestorePlan::FullReprefill;

    if (live_restore) {
        // The live cache already is this snapshot (saved after the previous
        // generation, nothing ran since). Only the rows past it differ from
        // a restore: rejected drafts, which the padded decode span reads.
        clear_kv_rows_from(cache_, cache_.cur_pos);
    } else {
        // Clear-then-restore: the step-invariant decode reads a 256-padded,
        // mask-less FA span, so rows beyond the restored prefix must be ZERO,
        // not leftovers from the previous request. cudaMemset is ~0.2ms.
        if (cache_.base_buf) ggml_backend_buffer_clear(cache_.base_buf, 0);
        if (full_reprefill) {
            std::fprintf(stderr,
                "[kvflash] snapshot (pos=%d) %s: full re-prefill "
                "of %zu prompt tokens\n", snap_pos,
                snap_pooled ? "not consumable" : "dense prefix cannot continue past the pool",
                req.prompt.size());
            reset_recurrent_state(cache_);
        } else {
            // Decoding on a cache the restore did not fill yields garbage.
            const bool restored = consume
                ? (restore_target_cache(prefix_snapshots_[slot], cache_, /*skip_kv=*/true) &&
                   kvflash_restore_pooled(prefix_snapshots_[slot], req.prompt))
                : restore_target_cache(prefix_snapshots_[slot], cache_);
            if (!restored) {
                std::fprintf(stderr, "[qwen35] restore slot=%d failed: %s\n",
                             slot, luce_last_error());
                cache_.cur_pos = 0;
                result.fail(consume ? GenerateErrorCode::BackendSpecific
                                    : GenerateErrorCode::InvalidSnapshotSlot,
                            consume ? std::string("kvflash: pooled snapshot restore failed")
                                    : std::string("snapshot restore failed: ") + luce_last_error());
                out_io.emit(-1);
                return result;
            }
            // Rungs belong to the live session, not to this snapshot's lineage
            // (a consuming restore replaces the pager and history too).
            if (kvflash_active() && kvflash_session_enabled()) state_ladder_clear();
            if (!consume) {
                if (adopt) {
                    // Dense rows [0, snap_pos) already sit in the pool tensor with
                    // chunk c in block c: register them as pager chunks, no copy.
                    kvflash_sync_prefill(snap_pos, {}, snap_pos, &req.prompt);
                    std::printf("[kvflash] adopt dense: %d rows, %d chunks, no copy\n", snap_pos,
                                kvflash_pager_.n_chunks());
                    std::fflush(stdout);
                }
            }
        }
    }

    // Now generate from restored state
    sampler_ = req.sampler;
    if (req.do_sample && sampler_.seed != 0) {
        sampler_rng_.seed(sampler_.seed);
    }

    cache_.cur_pos = full_reprefill ? 0 : snap_pos;
    result.restored_prefix_tokens = full_reprefill ? 0 : snap_pos;
    result.restore_cut_reported = true;

    // FIX(prefix-cache + spec-decode): restore_target_cache brings back KV /
    // recurrent state / target_feat, but the draft-side feature mirror is left
    // stale. Without re-syncing it the draft emits -1 -> verify_batch embed
    // fails -> empty output on every cache hit. Mirror what do_prefill does.
    if (!draft_parked_) {
        const int ring_cap = remote_draft_.active() ? remote_draft_.ring_cap()
                                                     : feature_mirror_.cap;
        const int n = std::min(cache_.cur_pos, ring_cap);
        if (n > 0) {
            const int start = cache_.cur_pos - n;
            if (remote_draft_.active()) {
                if (!sync_remote_draft_features(start, n)) {
                    result.fail(GenerateErrorCode::BackendSpecific,
                                "prefix remote feature sync");
                    return result;
                }
            } else if (feature_mirror_.target_feat && cache_.target_feat) {
                if (!sync_local_draft_features(start, n)) {
                    result.fail(GenerateErrorCode::BackendSpecific,
                                "prefix feature mirror sync");
                    return result;
                }
            }
        }
    }

    // Prefill and decode change the live state from here on, which consumes
    // a deferred snapshot this restore continues.
    live_snapshot_slot_ = -1;
    live_snapshot_deferred_ = false;

    // Daemon receives the FULL prompt; slice off the cached prefix and prefill
    // only the delta at KV positions [snap_pos, snap_pos + delta.size()).
    int committed = snap_pos;
    const int prompt_len = (int)req.prompt.size();
    if (full_reprefill) {
        auto t_prefill_start = std::chrono::steady_clock::now();
        committed = do_prefill(req.prompt, out_io, req.snap_pos, req.snap_slot,
                               /*kv_offset=*/0, nullptr, req.restore_points);
        if (committed < 0) {
            result.fail(GenerateErrorCode::PrefillFailed);
            return result;
        }
        result.prefill_s = std::chrono::duration<double>(
            std::chrono::steady_clock::now() - t_prefill_start).count();
    } else if (prompt_len > snap_pos) {
        auto t_prefill_start = std::chrono::steady_clock::now();
        std::vector<int32_t> delta = restore_prompt_delta(req.prompt, snap_pos);
        if (consume) {
            std::printf("[kvflash] pooled restore: consume kv_offset=%d suffix=%d "
                        "prompt=%d\n", snap_pos, (int)delta.size(), prompt_len);
            std::fflush(stdout);
        }
        struct ConsumeScope {
            bool & flag; bool & live;
            ~ConsumeScope() { flag = false; live = false; }
        } consume_scope{kvflash_consume_restore_, kvflash_live_restore_active_};
        // An adopted prefix ends mid-chunk like a live cut does; the partial
        // chunk stays resident and the pooled prefill continues in it.
        kvflash_consume_restore_ = consume || adopt;
        kvflash_live_restore_active_ = adopt;
        committed = do_prefill(delta, out_io, req.snap_pos, req.snap_slot,
                               /*kv_offset=*/snap_pos, nullptr, req.restore_points,
                               (consume || adopt || kvflash_active()) ? &req.prompt : nullptr);
        if (committed < 0) {
            result.fail(GenerateErrorCode::PrefillFailed);
            return result;
        }
        result.prefill_s = std::chrono::duration<double>(
            std::chrono::steady_clock::now() - t_prefill_start).count();
    } else if (prompt_len > 0 && prompt_len < snap_pos) {
        // The slot's snapshot covers more KV than the new prompt. This is
        // routine with agent clients (Letta, Hermes, ...) that edit or
        // summarize their history between turns. Fall back to a fresh full
        // prefill instead of failing the request with zero tokens.
        std::fprintf(stderr,
            "[pc] snapshot longer than prompt (snap=%d > prompt=%d) — "
            "fresh prefill fallback\n", snap_pos, prompt_len);
        reset_recurrent_state(cache_);
        cache_.cur_pos = 0;
        result.restored_prefix_tokens = 0;
        result.restore_cut_reported = true;
        auto t_prefill_start = std::chrono::steady_clock::now();
        committed = do_prefill(req.prompt, out_io, req.snap_pos, req.snap_slot,
                               /*kv_offset=*/0, nullptr, req.restore_points);
        if (committed < 0) {
            result.fail(GenerateErrorCode::PrefillFailed);
            return result;
        }
        result.prefill_s = std::chrono::duration<double>(
            std::chrono::steady_clock::now() - t_prefill_start).count();
    } else {
        // Exact full-prompt cache hit (prompt_len == snap_pos): no prefill ran,
        // so the per-request decode step-graph sg_ was never (re)built. The
        // first decode step (do_ar_decode / do_spec_decode) writes sg_.inp_embed
        // BEFORE its own build_target_step, so a null/freed graph tensor aborts
        // in ggml_backend_tensor_set. Build a single-token decode step graph at
        // the restored position now, mirroring do_ar_decode's per-step build.
        const bool pool = kvflash_active();
        if (!build_target_step(sg_, w_, cache_, target_backend_,
                               /*kv_start=*/cache_.cur_pos, /*n_tokens=*/1,
                               /*with_mask=*/pool || w_.is_bailingmoe3,
                               /*capture=*/false,
                               /*capture_delta_intermediate=*/false,
                               /*fa_window=*/0,
                               /*logits_tail_rows=*/0,
                               cfg_.kq_stride_pad,
                               should_capture_moe_router(),
                               /*kvflash_mask=*/pool,
                               /*capture_qk=*/pool && kvflash_qk_policy_)) {
            result.fail(GenerateErrorCode::BackendSpecific,
                             "restore step-graph build");
            return result;
        }
    }
    if (out_io.is_cancelled()) {
        result.succeed();
        return result;
    }

    // Decode
    if (req.n_gen > 0) {
        auto t_decode_start = std::chrono::steady_clock::now();
        ThinkingBudget budget(req.budget_hook, req.n_gen);
        bool decode_ok = false;
        if (req.force_ar_decode) {
            decode_ok = do_ar_decode(committed, budget, result.tokens, out_io,
                                     &result.budget_forced_close,
                                     &result.degenerate_decode_close);
            out_io.emit(-1);
        } else {
            const auto * hint_tokens =
                req.hint_tokens.empty() ? nullptr : &req.hint_tokens;
            const auto * stall_prefix = req.stall_tool_prefix_tokens.empty()
                ? nullptr : &req.stall_tool_prefix_tokens;
            const auto * stall_suffix = req.stall_action_suffix_tokens.empty()
                ? nullptr : &req.stall_action_suffix_tokens;
            const auto * stall_skip = req.stall_skip_tokens.empty()
                ? nullptr : &req.stall_skip_tokens;
            decode_ok = do_spec_decode(committed, budget, result.tokens, out_io,
                                       result.accept_rate, result.spec_decode_ran,
                                       hint_tokens, stall_prefix, stall_suffix,
                                       stall_skip,
                                       &result.budget_forced_close,
                                       &result.degenerate_decode_close);
            if (decode_ok) {
                result.empty_visible_output =
                    qwen35_empty_visible_output(result.tokens, w_);
            }
        }
        if (!decode_ok) {
            result.fail(GenerateErrorCode::DecodeFailed);
            return result;
        }
        result.decode_s = std::chrono::duration<double>(
            std::chrono::steady_clock::now() - t_decode_start).count();
    }

    result.succeed();
    return result;
}

// ═══════════════════════════════════════════════════════════════════════════
// INTERNAL HELPERS — will be fleshed out when the spec-decode loop is
// migrated from test_dflash.cpp. For now, these are stubs that produce an
// error so the build succeeds and the interface is validated.
// ═══════════════════════════════════════════════════════════════════════════

static int kvflash_read_captured_queries(const TargetCache & cache, const KvFlashQkDims & d,
                                        int batch_tokens, std::vector<float> & q_all);

int Qwen35Backend::do_prefill(const std::vector<int32_t> & tokens,
                               const DaemonIO & io,
                               int snap_pos, int snap_slot,
                               int kv_offset,
                               const Qwen35ImageRows * images,
                               const std::vector<int> & restore_points,
                               const std::vector<int32_t> * full_prompt) {
    if (images && kv_offset != 0) {
        std::fprintf(stderr, "prefill: an image prompt must start at position 0\n");
        return -1;
    }
    if (images) {
        // Snapshots are found again by their tokens, and pads say nothing
        // about which image they stood for.
        snap_pos = -1;
        snap_slot = -1;
    }
    rope_delta_ = images ? images->prompt->positions.next - (int)tokens.size() : 0;
    // A finite --fa-window caps the full-attention layers to a sliding
    // window, so anything earlier than the window is invisible to them. That
    // is silent: the model still answers, it just cannot see the head of a
    // long prompt, which reads as a model quality problem rather than a
    // configuration one. Say so once, the first time a prompt actually
    // outgrows the window.
    if (cfg_.fa_window > 0 &&
        (int)tokens.size() + kv_offset > cfg_.fa_window) {
        static std::atomic<bool> s_fa_window_warned{false};
        if (!s_fa_window_warned.exchange(true)) {
            std::fprintf(stderr,
                "[qwen35] WARNING: prompt is %d tokens but --fa-window is %d: "
                "full-attention layers see only the last %d tokens, so content "
                "before that cannot be retrieved. Drop --fa-window for "
                "long-context work.\n",
                (int)tokens.size() + kv_offset, cfg_.fa_window, cfg_.fa_window);
        }
    }
    const int hidden = w_.n_embd;
    const int vocab  = w_.n_vocab;
    int prefill_ubatch = std::max(kQwen35MinChunkTokens,
                                  qwen35_prefill_ubatch(kQwen35PrefillUbatch));
    const int prompt_len = (int)tokens.size();
    prefill_last_logits_valid_ = false;

    // The HIP legacy pool retains freed graph temporaries. During a long,
    // shape-changing prefill those cached blocks can fragment VRAM until the
    // next allocation fails even though a large part of the pool is idle.
    // Trimming is deliberately opt-in and performed only between completed
    // chunks: ggml_backend_cuda_trim_pool() synchronizes the backend and
    // retires captured graphs before returning cached blocks to the driver.
    static const int prefill_pool_trim_tokens = []() {
        const char * value = std::getenv("LUCE_PREFILL_POOL_TRIM_TOKENS");
        if (value == nullptr || value[0] == '\0') return 0;
        char * end = nullptr;
        const long parsed = std::strtol(value, &end, 10);
        if (end == value || *end != '\0' || parsed <= 0 || parsed > INT_MAX) {
            std::fprintf(stderr,
                "[vram] ignoring invalid LUCE_PREFILL_POOL_TRIM_TOKENS=%s\n",
                value);
            return 0;
        }
        return (int)parsed;
    }();
    int64_t next_prefill_pool_trim = prefill_pool_trim_tokens > 0
        ? ((int64_t)kv_offset / prefill_pool_trim_tokens + 1) * prefill_pool_trim_tokens
        : INT64_MAX;

    // kvflash: a prompt that fits the pool prefills contiguously (identity
    // mapping, normal chunking). A LARGER prompt switches to POOLED CHUNKED
    // PREFILL: pager-chunk-sized batches whose KV rows are slot-mapped via
    // set_rows, with a slot-space mask per chunk and live eviction as the
    // pool fills (constant VRAM, linear time). Restore offsets are not
    // supported in the pooled path (a relocated prefix cannot be restored
    // identity-style in the first place).
    // A consuming restore (kvflash_restore_pooled already rebuilt the pager
    // from the snapshot blob) continues the pooled prefill at kv_offset.
    const bool kvf_consume = kvflash_active() && kvflash_consume_restore_ && kv_offset > 0;
    if (kvf_consume && !kvflash_live_restore_active_ &&
        kv_offset % kvflash_pager_.chunk_tokens() != 0) {
        std::fprintf(stderr, "[kvflash] consuming restore at unaligned offset %d "
                             "(chunk %d)\n", kv_offset, kvflash_pager_.chunk_tokens());
        set_last_error("kvflash: unaligned pooled restore offset");
        return -1;
    }
    const bool kvf_paged = kvflash_active() && (kvf_consume ||
        kv_offset + prompt_len > kvflash_tokens_ - kvflash_pager_.chunk_tokens());
    if (kvf_paged && kv_offset != 0 && !kvf_consume) {
        std::fprintf(stderr,
            "[kvflash] restored prefix (%d) + prompt (%d) exceeds pool %d; "
            "pooled prefill requires a fresh request\n",
            kv_offset, prompt_len, kvflash_tokens_);
        set_last_error("kvflash: restore + pooled prefill unsupported");
        return -1;
    }
    // Full prompt ids [0, kv_offset + prompt_len) for the gated preamble pin / user-question capture
    // (empty when the restored prefix is not known, or both features are off).
    std::vector<int32_t> kvf_ids;
    if (kvflash_active() && (kvflash_preamble_rows_ > 0 || kvflash_uq_queries_ > 0 || kvflash_prefill_reselect_ == 5) &&
        (kv_offset == 0 || (full_prompt && (int)full_prompt->size() >= kv_offset)))
        kvf_ids = kvflash_restore_history(tokens, kv_offset, full_prompt);
    if (kvf_paged) {
        // Chunk-aligned but not one-chunk ubatches; see kvflash_pooled_ubatch.
        const int kvf_chunk = kvflash_pager_.chunk_tokens();
        prefill_ubatch = kvflash_pooled_ubatch(prefill_ubatch, kvf_chunk, kvflash_tokens_);
        if (!kvf_consume) {   // a consuming restore keeps the restored pager
            kvflash_pager_.reset();
            if (kvflash_qk_policy_) {
                kvflash_qk_pool_.reset(kvflash_qk_pool_.dims());
                kvflash_qk_pooled_upto_ = 0;
            }
        }
        // reset() cleared the pins and a restore may have clipped them: pin on every pooled request.
        kvflash_apply_preamble_pin(&kvf_ids, true);
        std::printf("[kvflash] pooled prefill: %d tokens through a %d-token pool "
                    "(%d-token chunks, ubatch=%d, evicting)\n",
                    prompt_len, kvflash_tokens_, kvf_chunk, prefill_ubatch);
        std::fflush(stdout);
    } else if (kvflash_active() && kv_offset == 0) {
        // Dense prefill rebuilds the pager only at completion. A boundary
        // snapshot before then must not see the previous request's mapping.
        kvflash_pager_.reset();
    }

    // Skip KV-cache migration when resuming from a snapshot — the cache was
    // already migrated when the snapshot was taken; re-running migrate would
    // clobber the restored state. The rollback buffers (2.43 GiB on Qwen3.8-27B)
    // serve only spec decode, which needs a draft model (can_spec).
    if (kv_offset == 0 && cfg_.draft_path) {
        const int max_verify_tokens = cfg_.ddtree_mode
            ? std::max<int>(dw_.block_size, cfg_.ddtree_budget + 1)
            : dw_.block_size;
        const bool enable_specla = cfg_.specla_mode && cfg_.fast_rollback &&
            !cfg_.device.is_tensor_parallel() && !kvflash_active();
        if (!migrate_prefill_cache(w_, cfg_.device.max_ctx,
                                   max_verify_tokens,
                                   target_backend_, cache_, enable_specla)) {
            std::fprintf(stderr, "prefill: rollback cache migration failed: %s\n",
                         luce_last_error());
            return -1;
        }
    }

    const bool restored_off_grid = kv_offset > 0 &&
        !std::binary_search(restore_points.begin(), restore_points.end(), kv_offset);

    // S7 state ladder: checkpoints are keyed by position only, so a cold
    // prefill starts a fresh ladder. The head rung is the server's snapshot cut
    // (snap_pos); ponytail: no separate system-anchor length is plumbed, so
    // without a snap_pos the head rung is simply absent.
    std::vector<int> ladder;
    int turn_cut = -1;
    std::vector<int> turn_cuts;
    if (kvflash_active() && !images && kvflash_session_enabled()) {
        // Every prefill (cold included, ladder or not) cuts after every assistant
        // header, so a restore at a turn rung replays the batches a cold prefill ran.
        turn_cuts = kvflash_turn_cuts(tokens, w_.im_start_id, w_.assistant_id, w_.newline_id);
        for (int & c : turn_cuts) c += kv_offset;
    }
    // A cold prefill replaces the live session, image or not: drop its rungs.
    if (kvflash_active() && kvflash_session_enabled() && kv_offset == 0) state_ladder_clear();
    if (kvflash_active() && !images && kvflash_session_enabled()) {
        // A live restore keeps the head rung unless the server names a new cut.
        if (!kvflash_live_restore_active_ || snap_pos > 0) {
            state_ladder_head_ = snap_pos > 0 ? snap_pos / kQwen35PrefillUbatch * kQwen35PrefillUbatch : -1;
        }
        // S9b: an exact rung just after the last assistant header (Pend - k).
        if (!turn_cuts.empty()) turn_cut = turn_cuts.back();
        ladder = kvflash_state_ladder_positions(snap_pos, kv_offset + prompt_len,
                                                kQwen35PrefillUbatch, 8192, turn_cut);
    }

    // LUCE_KVFLASH_PREFILL_RESELECT: R becomes a batch boundary (local cut
    // copy; turn_cuts itself still drives the ladder rung and restore).
    const bool kvf_reselect = kvf_paged && kvflash_active() && kvflash_qk_policy_ &&
                              kvflash_prefill_reselect_ != 0 && !images;
    // Modes 3 and 4 (rewind): the pass over [R, end) runs twice, the second time
    // under the reselected pool. Mode 3: R = end-1. Mode 4: R = start of the last
    // user turn (clamped to the prefill start). A strict no-op when the pool holds
    // the whole prompt: recall and compaction are active only when the pool is full.
    struct RewindPlan { int r = -1; bool run = false; int reserve = -1; } rw;
    // The first (scoring) pass must see OFF's plain-LRU pool, not the previous
    // request's recalled blocks: recency-reselect on every pooled request, after
    // every restore path and before the rw.run decision.
    kvflash_m4_ = {};
    if (kvf_paged) {
        kvflash_m4_.t_start = KvflashM4Phases::Clock::now();
        kvflash_m4_.dma0 = kvflash_pager_.dma_counters();
        kvflash_m4_.ins0 = kvflash_pager_.stats().page_ins;
        kvflash_m4_.outs0 = kvflash_pager_.stats().page_outs;
    }
    if (kvf_paged && kvflash_score_pool_recent_) {
        const int64_t ins0 = kvflash_pager_.stats().page_ins;
        const int events = kvflash_pager_.reselect_recent();
        kvflash_m4_.reset_async = kvflash_pager_.page_work_pending();   // no sync closed this phase
        kvflash_m4_.reset = std::chrono::duration<double, std::milli>(KvflashM4Phases::Clock::now() - kvflash_m4_.t_start).count();
        const int64_t ins = kvflash_pager_.stats().page_ins - ins0;
        std::fprintf(stderr, "[kvflash] prefill reselect recent-pool @%d: %d page events; "
                     "%d in (%.1f MiB H2D)\n", kv_offset, events, (int)ins,
                     (double)ins * (double)kvflash_pager_.chunk_bytes() / 1048576.0);
    }
    kvflash_m4_.t_after_reset = KvflashM4Phases::Clock::now();
    const bool rewind_mode = kvf_reselect && (kvflash_prefill_reselect_ == 3 || kvflash_prefill_reselect_ == 4);
    if (rewind_mode) {
        const int end_abs = kv_offset + prompt_len;
        const int ctk = kvflash_pager_.chunk_tokens();
        if (kvflash_prefill_reselect_ == 3) {
            const auto p3 = luce::common::kvflash_mode3_plan(kv_offset, end_abs);
            rw = {p3.r, p3.run, p3.reserve_chunks};
        } else {
            const auto p4 = luce::common::kvflash_mode4_plan(tokens, kv_offset, w_.im_start_id, w_.user_id);
            rw = {p4.r, p4.run, p4.reserve_chunks};
        }
        if (rw.run) {
            const int reserve = rw.reserve >= 0 ? rw.reserve : (end_abs - rw.r + ctk - 1) / ctk + 1;
            if (luce::common::kvflash_pool_fits(end_abs, ctk, reserve, kvflash_pager_.n_pool_blocks())) {
                std::fprintf(stderr, "[kvflash] prefill reselect mode=%d: skip, fits (%d chunks + %d reserve <= %d blocks)\n",
                             kvflash_prefill_reselect_, (end_abs + ctk - 1) / ctk, reserve,
                             kvflash_pager_.n_pool_blocks());
                rw.run = false;
            } else if (kvflash_prefill_reselect_ == 4) {
                std::fprintf(stderr, "[kvflash] prefill reselect mode=4: R4=%d (end %d, re-prefill %d tokens)\n",
                             rw.r, end_abs, end_abs - rw.r);
            }
        }
    }
    // Mode 5: the drafter scores the prompt ids now, before the 27B runs; the picks are applied when the prefill
    // reaches R (start of the last user turn) and the turn runs once: no first pass, rewind, replay or R snapshot.
    int m5_r = -1, m5_reserve = -1;
    double m5_ms = 0.0;
    if (kvf_reselect && kvflash_prefill_reselect_ == 5) {
        const int end_abs = kv_offset + prompt_len, ctk = kvflash_pager_.chunk_tokens();
        const auto p4 = luce::common::kvflash_mode4_plan(tokens, kv_offset, w_.im_start_id, w_.user_id);
        const int reserve = (end_abs - p4.r + ctk - 1) / ctk + 1;
        const char * skip = !p4.run || p4.r / ctk == 0 ? "no sealed chunk before R"
            : luce::common::kvflash_pool_fits(end_abs, ctk, reserve, kvflash_pager_.n_pool_blocks()) ? "fits"
            : (int)kvf_ids.size() != end_abs ? "prompt ids unknown"
            : !kvflash_pflash2_scores(kvf_ids, ctk, kvflash_scores_, m5_ms) ? "drafter scoring failed" : nullptr;
        if (skip) {
            std::fprintf(stderr, "[kvflash] prefill reselect mode=5: skip, %s (LRU)\n", skip);
        } else {
            m5_r = p4.r;
            m5_reserve = reserve;
            std::fprintf(stderr, "[kvflash] prefill reselect mode=5: R=%d (end %d, prefill %d tokens once)\n", m5_r, end_abs, end_abs - m5_r);
        }
    }
    const int reselect_r = rw.run ? rw.r
        : m5_r >= 0 ? m5_r
        : (kvf_reselect && kvflash_prefill_reselect_ <= 2)
            ? luce::common::kvflash_prefill_reselect_point(turn_cuts, kv_offset, kv_offset + prompt_len,
                                                      kvflash_prefill_reselect_ == 2 || kvflash_prefill_reselect_ >= 3
                                                          ? 2 : 1)
            : -1;
    // LUCE_KVFLASH_UQ_QUERIES: the last N queries of the latest human question. A stored copy of this
    // session's prompt prefix is reused; otherwise the prefill cuts at the question's end and captures.
    luce::common::KvFlashUqPlan uqp;
    bool uq_capturing = false, uq_done = false;
    int uq_got = 0, uq_cut = -1;
    std::vector<float> uq_rows;
    kvflash_uq_req_hi_ = -1;
    kvflash_uq_miss_.clear();
    if (kvflash_uq_queries_ > 0) {
        if (!kvflash_active() || images || !kvflash_qk_policy_) kvflash_uq_miss_ = "uq-inactive";
        else if (kvf_ids.empty()) kvflash_uq_miss_ = "no-prompt-ids";
        else {
            const auto t = luce::common::kvflash_last_human_turn(kvf_ids, w_.im_start_id, w_.user_id, w_.newline_id,
                                                                 w_.im_end_id, w_.tool_response_id);
            if (!t.valid()) {
                kvflash_uq_miss_ = "no-human-turn";
            } else {
                kvflash_uq_req_hi_ = t.hi;
                kvflash_uq_req_hash_ = luce::common::kvflash_uq_hash(kvf_ids, t.hi);
                uqp = luce::common::kvflash_uq_plan(t, kvflash_uq_queries_, kv_offset, kv_offset + prompt_len);
                uq_cut = luce::common::kvflash_uq_cut(t, kvflash_uq_queries_, kv_offset, kv_offset + prompt_len);
                const int hit = luce::common::kvflash_uq_find(kvflash_uq_stores_, t.hi, kvflash_uq_req_hash_);
                if (hit >= 0) luce::common::kvflash_uq_touch(kvflash_uq_stores_, hit);
                else if (uqp.capture) { uq_capturing = true; }
                else kvflash_uq_miss_ = "window-not-in-prefill";
            }
        }
    }
    std::vector<int> batch_cuts = reselect_r >= 0
        ? luce::common::kvflash_cuts_with(turn_cuts, reselect_r) : turn_cuts;
    // The cut follows the question, not the store: identical requests get identical batch schedules.
    if (uq_cut >= 0) batch_cuts = luce::common::kvflash_cuts_with(batch_cuts, uq_cut);
    // The dense (non-pooled) path chunks on restore points; the question end is one more.
    const std::vector<int> uq_dense_points = uq_cut >= 0 && !kvf_paged
        ? luce::common::kvflash_cuts_with(restore_points, uq_cut) : std::vector<int>();
    const std::vector<int> & chunk_points = uq_dense_points.empty() ? restore_points : uq_dense_points;
    // The last q_keep prompt queries, oldest first, in kvflash_q_tail_: selector v2's
    // (SELECT_QUERIES, rewind run only).
    const int q_keep = rw.run ? kvflash_select_queries_ : 0;
    kvflash_q_tail_.clear();
    // Scratch state snapshot at R (not a ladder rung), freed on every exit by the guard.
    struct M3Scratch {
        PrefixSnapshot snap;
        ~M3Scratch() { free_prefix_snapshot(snap); }
    } m3_scratch;
    bool rw_snapped = false, rw_done = false, rw_failed = false, rw_rerun_pending = false;
    const bool rw_zero_state = rw.run && rw.r == 0;   // R4 = 0: the state to restore is the zero state
    std::vector<float> rw_first_logits;   // first-pass final-token logits row

    if (kvf_paged && kvflash_log_resident_) {
        std::vector<int> res;
        for (int c = 0; c < kvflash_pager_.n_chunks(); c++)
            if (kvflash_pager_.is_resident(c)) res.push_back(c);
        std::fprintf(stderr, "[kvflash] resident @%d: n=%d chunks=%s\n", kv_offset, (int)res.size(),
                     luce::common::kvflash_format_ranges(res).c_str());
    }

    // Chunked prefill
    std::vector<float> embed_buf((size_t)hidden * prefill_ubatch);
    int committed = kv_offset;
    for (int start = 0; start < prompt_len;) {
        if (io.is_cancelled()) break;
        const int kv_pos = kv_offset + start;

        // Every restore point starts a chunk, in a cold prefill and after a
        // restore alike, so both cut the prompt into the same chunks and
        // produce the same state (FIX(bug2): chunk boundaries change the
        // numerics). A snapshot requested at a restore point is then exact.
        // Between restore points chunks keep to the ubatch grid: attention
        // over chunks off it is markedly slower.
        // Pooled kvflash prefill runs on an absolute 512-token grid (see
        // kvflash_prefill_batch_end) and never snapshots mid-batch.
        const int n_tokens = kvf_paged
            ? kvflash_prefill_batch_end(kv_pos, kv_offset + prompt_len,
                                        kQwen35PrefillUbatch, prefill_ubatch, batch_cuts) - kv_pos
            : qwen35_prefill_chunk_tokens(
                  kv_pos, prompt_len - start, prefill_ubatch, chunk_points,
                  start == 0 && restored_off_grid ? kQwen35OffGridLeadTokens
                                                  : kQwen35MinChunkTokens);
        // When snap_pos falls inside this chunk (a cut that is not a restore
        // point), snapshot at the chunk START kv_pos: the largest chunk
        // boundary <= snap_pos, strictly within the requested prefix, so a
        // later request that shares only that prefix still restores a valid
        // hit. (Rounding UP would include tokens past the cut, e.g. a user
        // message another request does not share.)
        if (snap_slot >= 0 && snap_pos >= 0 &&
            kv_pos <= snap_pos && snap_pos < kv_pos + n_tokens) {
            if (kv_pos > kv_offset && !kvf_paged) {   // skip degenerate / relocated
                cache_.cur_pos = kv_pos;
                if (snapshot_save(snap_slot)) {
                    std::printf("[snap] boundary slot=%d cur_pos=%d (req snap_pos=%d)\n",
                                snap_slot, kv_pos, snap_pos);
                    std::fflush(stdout);
                }
            } else if (kvf_paged && kv_pos > kv_offset &&
                       luce::common::kvflash_restore_consume_enabled()) {
                // kv_pos starts a pooled ubatch (chunk multiple); cur_pos is
                // already kv_pos. Serialize the pager chunks [0, kv_pos/chunk).
                cache_.cur_pos = kv_pos;
                if (snapshot_save_pooled_at(snap_slot, kv_pos)) {
                    std::printf("[snap] pooled boundary slot=%d cur_pos=%d "
                                "(req snap_pos=%d chunks=%d)\n", snap_slot,
                                kv_pos, snap_pos, kv_pos / kvflash_pager_.chunk_tokens());
                    std::fflush(stdout);
                } else {
                    std::fprintf(stderr, "[kvflash] pooled boundary snapshot failed "
                                         "at kv_pos=%d\n", kv_pos);
                }
            } else if (kvf_paged) {
                std::fprintf(stderr, "[kvflash] boundary snapshot skipped: pooled "
                                     "prefill relocates chunks\n");
            }
            snap_pos = -1;
            snap_slot = -1;
        }
        const bool with_mask = kvf_paged ||
            (cfg_.kq_stride_pad > KQ_MASK_PAD) || (n_tokens > 1);

        // Reselect point R: q_cap holds token R-1's query (the previous batch
        // ended at R and was built with capture_qk); refill the pool before
        // this batch's slots and mask are computed from the pager state.
        if (kvf_reselect && kvflash_prefill_reselect_ < 3 && reselect_r >= 0 && kv_pos == reselect_r)
            kvflash_prefill_reselect_now(kv_pos, kv_offset + prompt_len);
        if (m5_r >= 0 && kv_pos == m5_r)
            kvflash_prefill_apply(kv_pos, kv_offset + prompt_len, m5_reserve, kv_pos - 1, m5_ms, "; pflash2");

        // Modes 3/4: the state at R is about to run [R, end) under the old pool;
        // keep a scratch copy so the pass can be rewound.
        const bool is_final_batch = start + n_tokens >= prompt_len;
        if (rw.run && !rw_done && !rw_failed && !rw_snapped && kv_pos == rw.r) {
            const auto t_snap0 = KvflashM4Phases::Clock::now();
            kvflash_m4_.pre = std::chrono::duration<double, std::milli>(t_snap0 - kvflash_m4_.t_after_reset).count();
            if (cache_.cur_pos != kv_pos ||
                (!rw_zero_state &&
                 !snapshot_target_cache(w_, cache_, snap_backend_, m3_scratch.snap,
                                        /*skip_kv=*/true, /*kvflash_blob=*/nullptr,
                                        /*skip_target_feat=*/!cfg_.draft_path))) {
                std::fprintf(stderr, "[kvflash] prefill reselect mode=%d: scratch snapshot @%d failed; "
                                     "no reselect for this request\n", kvflash_prefill_reselect_, kv_pos);
                rw_failed = true;
            } else {
                rw_snapped = true;
            }
            kvflash_m4_.t_first0 = KvflashM4Phases::Clock::now();
            kvflash_m4_.snap = std::chrono::duration<double, std::milli>(kvflash_m4_.t_first0 - t_snap0).count();
        }
        const bool rw_pass1 = rw.run && !rw_done && !rw_failed && rw_snapped;   // first pass over [R, end)
        const bool rw_capture = rw_pass1 && is_final_batch;                      // its final batch: q(end-1)

        // kvflash pooled prefill: allocate this chunk's slots up front
        // (evicting the lowest-priority resident chunk once the pool fills).
        std::vector<int> kvf_slots;
        if (kvf_paged) {
            kvf_slots.resize((size_t)n_tokens);
            bool ok = true;
            for (int i = 0; i < n_tokens; i++) {
                kvf_slots[(size_t)i] = kvflash_pager_.slot_for(kv_pos + i);
                if (kvf_slots[(size_t)i] < 0) { ok = false; break; }
            }
            if (!ok) {
                std::fprintf(stderr, "[kvflash] pooled prefill: slot alloc failed @%d\n", kv_pos);
                set_last_error("kvflash: no evictable pool block");
                return -1;
            }
        }

        // Prefill always uses full attention (fa_window=0) so that all
        // positions encode the complete context — critical for tool
        // definitions at prompt start to propagate into KV values that
        // decode-time windowed attention will later read.
        static const bool prefill_timing = std::getenv("LUCE_PREFILL_TIMING") != nullptr;
        const auto t_build0 = std::chrono::steady_clock::now();
        if (!build_target_step(sg_, w_, cache_, target_backend_,
                               /*kv_start=*/kv_pos, /*n_tokens=*/n_tokens,
                               with_mask, /*capture=*/true,
                               /*capture_delta_intermediate=*/false,
                               /*fa_window=*/0,
                               /*logits_tail_rows=*/
                                   (start + n_tokens < prompt_len ? 1 : 0),
                               cfg_.kq_stride_pad,
                               should_capture_moe_router(),
                               /*kvflash_mask=*/kvf_paged,
                               /*capture_qk=*/(reselect_r >= 0 && kv_pos + n_tokens == reselect_r) || rw_capture ||
                                   (q_keep > 0 && kv_pos + n_tokens + q_keep > kv_offset + prompt_len) ||
                                   (uq_capturing && !uq_done && kv_pos < uqp.cut && kv_pos + n_tokens > uqp.win_lo))) {
            std::fprintf(stderr, "prefill build @%d\n", kv_pos);
            return -1;
        }
        if (kvf_paged) {
            if (!sg_.kv_write_rows) {
                std::fprintf(stderr, "[kvflash] pooled prefill requires the set_rows path\n");
                return -1;
            }
            // [n_tokens, n_head_kv] ne0-major (see verify_batch).
            std::vector<int64_t> rows((size_t)n_tokens * w_.n_head_kv);
            for (int h = 0; h < w_.n_head_kv; h++) {
                for (int i = 0; i < n_tokens; i++) {
                    rows[(size_t)h * n_tokens + i] = kvf_slots[(size_t)i];
                }
            }
            ggml_backend_tensor_set(sg_.kv_write_rows, rows.data(), 0,
                                    sizeof(int64_t) * rows.size());
        }

        // Embed
        if (!w_.embedder.embed(tokens.data() + start, n_tokens, embed_buf.data())) {
            return -1;
        }
        if (images) images->overwrite(embed_buf.data(), start, n_tokens, hidden);
        ggml_backend_tensor_set(sg_.inp_embed, embed_buf.data(), 0,
                                sizeof(float) * (size_t)hidden * n_tokens);

        // Positions (M-RoPE)
        std::vector<int32_t> pos_buf((size_t)4 * n_tokens, 0);
        if (images) {
            images->prompt->positions.fill(pos_buf.data(), start, n_tokens);
        } else {
            fill_qwen35_mrope_positions(pos_buf.data(), kv_pos, n_tokens);
        }
        ggml_backend_tensor_set(sg_.positions, pos_buf.data(), 0,
                                sizeof(int32_t) * pos_buf.size());

        // Mask — full attention during prefill (no windowing)
        if (sg_.attn_mask && kvf_paged) {
            // Slot-space mask (same recipe as verify_batch): row q attends
            // (a) the slots of resident chunks holding positions < kv_pos
            // and (b) this chunk's own slots, causally.
            constexpr uint16_t F16_ZERO = 0x0000, F16_NEG_INF = 0xFC00;
            const size_t kvd = (size_t)sg_.attn_mask->ne[0];
            const int q_pad = (int)sg_.attn_mask->ne[1];
            std::vector<uint16_t> mask_buf((size_t)kvd * q_pad, F16_NEG_INF);
            const int ct = kvflash_pager_.chunk_tokens();
            for (int c = 0; c < kvflash_pager_.n_chunks(); c++) {
                const int blk = kvflash_pager_.block_of(c);
                if (blk < 0) continue;
                for (int i = 0; i < ct; i++) {
                    if ((int64_t)c * ct + i >= kv_pos) break;
                    mask_buf[(size_t)blk * ct + i] = F16_ZERO;
                }
            }
            for (int q = 1; q < n_tokens; q++) {
                std::memcpy(mask_buf.data() + (size_t)q * kvd, mask_buf.data(), kvd * 2);
            }
            for (int q = 0; q < n_tokens; q++) {
                for (int i = 0; i <= q; i++) {
                    mask_buf[(size_t)q * kvd + kvf_slots[(size_t)i]] = F16_ZERO;
                }
            }
            ggml_backend_tensor_set(sg_.attn_mask, mask_buf.data(), 0,
                                    sizeof(uint16_t) * mask_buf.size());
        } else if (sg_.attn_mask) {
            upload_qwen35_causal_mask(
                sg_.attn_mask, kv_pos, n_tokens, cfg_.kq_stride_pad);
        }

        // Compute
        const auto t_comp0 = std::chrono::steady_clock::now();
        auto st = ggml_backend_graph_compute(target_backend_, sg_.gf);
        if (prefill_timing) {
            ggml_backend_synchronize(target_backend_);
            const auto t_comp1 = std::chrono::steady_clock::now();
            std::fprintf(stderr,
                "[prefill-timing] tokens=%d nodes=%d build+alloc=%.1fms compute=%.1fms\n",
                n_tokens, ggml_graph_n_nodes(sg_.gf),
                std::chrono::duration<double, std::milli>(t_comp0 - t_build0).count(),
                std::chrono::duration<double, std::milli>(t_comp1 - t_comp0).count());
        }
        if (st != GGML_STATUS_SUCCESS) {
            std::fprintf(stderr, "prefill compute @%d failed\n", kv_pos);
            return -1;
        }
        // The discarded first pass of modes 3/4 must not count in the MoE routing stats.
        if (!rw_pass1) after_target_compute(sg_, kv_pos, n_tokens);
        // Keep the first pass's final-token logits row; the re-run is compared with it.
        if (rw_capture || (rw_rerun_pending && is_final_batch)) {
            std::vector<float> row((size_t)vocab);
            ggml_backend_tensor_get(sg_.logits, row.data(),
                                    (size_t)(n_tokens - 1) * (size_t)vocab * sizeof(float),
                                    row.size() * sizeof(float));
            if (rw_capture) {
                kvflash_m4_.first = std::chrono::duration<double, std::milli>(KvflashM4Phases::Clock::now() - kvflash_m4_.t_first0).count();
                rw_first_logits = std::move(row);
            } else {
                const double dmax = (double)luce::common::kvflash_max_abs_diff(row, rw_first_logits);
                if (kvflash_prefill_reselect_ == 4) {
                    std::fprintf(stderr, "[kvflash] prefill reselect mode=4 @%d: re-prefill %d tokens ok, "
                                 "logits max|diff| %g vs first pass (argmax %d / %d)\n", rw.r,
                                 kv_offset + prompt_len - rw.r, dmax,
                                 luce::common::kvflash_argmax(row), luce::common::kvflash_argmax(rw_first_logits));
                } else {
                    std::fprintf(stderr, "[kvflash] prefill reselect mode=3 @%d: rerun ok, logits max|diff| %g "
                                 "vs first pass (argmax %d / %d)\n", kv_pos, dmax,
                                 luce::common::kvflash_argmax(row), luce::common::kvflash_argmax(rw_first_logits));
                }
                rw_rerun_pending = false;
            }
        }

        int32_t last_tok = -1;
        const bool is_final_chunk = (start + n_tokens >= prompt_len);
        const size_t argmax_off =
            is_final_chunk ? sizeof(int32_t) * (size_t)(n_tokens - 1) : 0;
        ggml_backend_tensor_get(sg_.argmax_tokens, &last_tok, argmax_off, sizeof(int32_t));
        cache_.last_tok = last_tok;
        if (is_final_chunk) {
            prefill_last_logits_offset_ = (size_t)(n_tokens - 1) * (size_t)vocab * sizeof(float);
            prefill_last_logits_valid_ = true;
        }

        committed = kv_pos + n_tokens;
        cache_.cur_pos = committed;

        // Every batch holding any of the last q_keep prompt tokens is captured;
        // append its queries and keep the last q_keep, oldest first.
        if (q_keep > 0 && !rw_done && committed + q_keep > kv_offset + prompt_len) {
            const KvFlashQkDims d = kvflash_exact_dims();
            std::vector<float> q_all;
            if (kvflash_read_captured_queries(cache_, d, n_tokens, q_all) > 0) {
                auto & tail = kvflash_q_tail_;
                tail.insert(tail.end(), q_all.begin(), q_all.end());
                const size_t keep = (size_t)q_keep * d.n_layers * d.n_q_heads * d.head_dim;
                if (tail.size() > keep) tail.erase(tail.begin(), tail.end() - keep);
            }
        }

        // LUCE_KVFLASH_UQ_QUERIES: this batch holds part of the question's window; keep those rows.
        if (uq_capturing && !uq_done && kv_pos < uqp.cut && committed > uqp.win_lo) {
            std::vector<float> q_all;
            const KvFlashQkDims d = kvflash_exact_dims();
            const int n_q = kvflash_read_captured_queries(cache_, d, n_tokens, q_all);
            const size_t qsz = (size_t)d.n_layers * d.n_q_heads * d.head_dim;
            const auto tk = luce::common::kvflash_uq_batch_take(kv_pos, n_tokens, n_q, uqp.win_lo, uqp.cut);
            if (tk.count > 0 && qsz > 0) {
                uq_rows.insert(uq_rows.end(), q_all.begin() + (long)((size_t)tk.first_row * qsz),
                               q_all.begin() + (long)((size_t)(tk.first_row + tk.count) * qsz));
                uq_got += tk.count;
            }
            if (committed >= uqp.cut) {
                uq_done = true;
                if (uq_got == uqp.n && uq_rows.size() == (size_t)uqp.n * qsz) {
                    luce::common::KvFlashUqStore st;
                    st.lo = uqp.win_lo; st.hi = uqp.cut; st.n = uqp.n; st.hash = kvflash_uq_req_hash_; st.rows = std::move(uq_rows);
                    for (int ev : luce::common::kvflash_uq_put(kvflash_uq_stores_, std::move(st), luce::common::kKvFlashUqMaxStores))
                        std::fprintf(stderr, "[kvflash] uq store evicted hi=%d (%zu kept)\n", ev, kvflash_uq_stores_.size());
                } else {
                    kvflash_uq_miss_ = "capture-incomplete";
                }
            }
        }

        // Modes 3/4: q_cap = q(end-1) under the old pool. Reselect, rewind to R and
        // run [R, end) again under the new pool (its result replaces this one).
        if (rw_capture) {
            const int rc = kvflash_rewind_turn(m3_scratch.snap, rw_zero_state, rw.r,
                                               kv_offset + prompt_len, rw.reserve);
            rw_done = true;
            if (rc < 0) return -1;
            if (rc > 0) {
                kvflash_m4_.rewound = true;
                kvflash_m4_.t_replay0 = KvflashM4Phases::Clock::now();
                committed = rw.r;
                start = rw.r - kv_offset;
                rw_rerun_pending = true;
                // The rewind reaches back over the question window: its rows from the old pool are
                // stale. They were used for the scoring just done; the second pass replaces them.
                if (kvflash_uq_req_hi_ > 0 && luce::common::kvflash_uq_rewind_recaptures(uqp, rw.r)) {
                    luce::common::kvflash_uq_erase(kvflash_uq_stores_, kvflash_uq_req_hi_, kvflash_uq_req_hash_);
                    uq_rows.clear(); uq_got = 0; uq_done = false; uq_capturing = true;
                    kvflash_uq_miss_.clear();
                }
                continue;
            }
        }

        // S7: batch ends are grid points, so ladder rungs coincide with them.
        if (!ladder.empty() && std::binary_search(ladder.begin(), ladder.end(), committed) &&
            !state_ladder_capture(committed)) {
            std::fprintf(stderr, "[kvflash] state ladder capture failed @%d\n", committed);
        }

        // QK policy: pool the post-RoPE keys of chunks this batch sealed
        // (they are resident — sealed inside the protected tail window).
        if (kvflash_active() && kvflash_qk_policy_) kvflash_qk_pool_to(committed);
        // Sync draft-side features if active.
        if (remote_draft_.active() && !draft_parked_) {
            if (!sync_remote_draft_features(kv_pos, n_tokens)) return -1;
        } else if (feature_mirror_.target_feat && !draft_parked_) {
            if (!sync_local_draft_features(kv_pos, n_tokens)) return -1;
        }

        start += n_tokens;

        if (prefill_pool_trim_tokens > 0 && start < prompt_len &&
            (int64_t)committed >= next_prefill_pool_trim &&
            ggml_backend_cuda_has_legacy_pool(target_backend_)) {
            const size_t freed = ggml_backend_cuda_trim_pool(target_backend_);
            std::fprintf(stderr,
                "[vram] prefill pool trim at token %d: freed %.1f MiB\n",
                committed, (double)freed / (1024.0 * 1024.0));
            while (next_prefill_pool_trim <= (int64_t)committed) {
                if (next_prefill_pool_trim > INT64_MAX - prefill_pool_trim_tokens) {
                    next_prefill_pool_trim = INT64_MAX;
                    break;
                }
                next_prefill_pool_trim += prefill_pool_trim_tokens;
            }
        }
    }

    kvflash_log_m4_phases();

    if (kvflash_active()) {
        if (kvf_paged) {
            // The pager mapping was built live during the pooled prefill;
            // only the history / hygiene parts of the sync apply.
            kvflash_history_ = kvflash_restore_history(tokens, kv_offset, full_prompt);
            kvflash_pager_.zero_free_blocks();
            kvflash_mask_epoch_ = (uint64_t)-1;
        } else {
            kvflash_sync_prefill(committed, tokens, kv_offset, full_prompt);
        }
    }

    // End-of-prefill snapshot: scoped disk-cache saves (auto/fixed policy)
    // request snap_pos == prompt end, which never falls inside a chunk so the
    // boundary branch above cannot fire. Taking the snapshot here changes
    // nothing about the prefill computation; it only persists the final state
    // (cache_.cur_pos == committed).
    if (snap_slot >= 0 && snap_pos == committed) {
        if (snapshot_save(snap_slot)) {
            std::printf("[snap] end-of-prefill slot=%d cur_pos=%d\n",
                        snap_slot, committed);
            std::fflush(stdout);
        }
    }

    if (kvflash_active() && std::getenv("LUCE_KVFLASH_DEBUG_RESIDENT")) {
        std::printf("[kvflash-dbg] after prefill kv_offset=%d committed=%d resident=%s\n", kv_offset, committed,
                    kvflash_pager_.debug_resident(committed / kvflash_pager_.chunk_tokens()).c_str());
        std::fflush(stdout);
    }
    return committed;
}

// ── kvflash helpers ─────────────────────────────────────────────────

void Qwen35Backend::kvflash_sync_prefill(int committed,
                                         const std::vector<int32_t> & tokens,
                                         int kv_offset,
                                         const std::vector<int32_t> * full_prompt) {
    kvflash_require_base_zero(kvflash_pager_, "kvflash_sync_prefill");
    // Prefill (and snapshot restore) place rows physically contiguous at
    // [0, committed): rebuild the pager mapping identity-style and reset
    // the token history to match.
    kvflash_pager_.reset();
    for (int p = 0; p < committed; p++) {
        const int slot = kvflash_pager_.slot_for(p);
        if (slot != p) {
            // Cannot happen while prompt <= pool (blocks are handed out in
            // order from a freshly reset pager); guard against future
            // changes to the hand-out order.
            std::fprintf(stderr, "[kvflash] prefill slot mismatch %d != %d\n", slot, p);
        }
    }
    // Restored-prefix ids are real when the caller knows the full prompt,
    // zeros otherwise.
    kvflash_history_ = kvflash_restore_history(tokens, kv_offset, full_prompt);
    kvflash_apply_preamble_pin(&kvflash_history_, false);   // reset() above dropped the pins
    // Slots past the prompt still hold the previous request's rows; the
    // maskless qwen35moe pipelined decode reads the whole padded pool span.
    kvflash_pager_.zero_free_blocks();
    kvflash_mask_epoch_ = (uint64_t)-1;
    if (kvflash_qk_policy_) {
        kvflash_qk_pool_.reset(kvflash_qk_pool_.dims());
        kvflash_qk_pooled_upto_ = 0;
        kvflash_qk_pool_to(committed);
    }
}

// Rebuild the pager from a pooled snapshot's blob so a suffix-only prefill can
// continue it: resident chunks return to the pool, the rest stay host-backed.
// QK policy: the pooled keys are not in the blob, so the resident chunks are
// re-pooled from the restored rows; evicted chunks score as missing (0) until
// paged back in (rebuilding them from the blob's score ledger is a follow-up).
bool Qwen35Backend::kvflash_restore_pooled(const PrefixSnapshot & snap,
                                           const std::vector<int32_t> & prompt) {
    const int ct = kvflash_pager_.chunk_tokens();
    if (!kvflash_active() || ct <= 0 || snap.kvflash_blob.empty()) return false;
    if (!kvflash_pager_.deserialize(snap.kvflash_blob.data(), snap.kvflash_blob.size())) {
        std::fprintf(stderr, "[kvflash] pager deserialize failed\n");
        return false;
    }
    if (kvflash_pager_.n_chunks() != (snap.cur_pos + ct - 1) / ct) {
        std::fprintf(stderr, "[kvflash] snapshot blob has %d chunks for pos %d\n",
                     kvflash_pager_.n_chunks(), snap.cur_pos);
        return false;
    }
    kvflash_pager_.zero_free_blocks();
    kvflash_mask_epoch_ = (uint64_t)-1;
    kvflash_scores_.clear();
    kvflash_history_ = kvflash_restore_history({}, snap.cur_pos, &prompt);
    kvflash_apply_preamble_pin(&prompt, false);
    if (kvflash_qk_policy_) {
        kvflash_qk_pool_.reset(kvflash_qk_pool_.dims());
        const int sealed = snap.cur_pos / ct;
        for (int c = 0; c < sealed; c++) {
            const int blk = kvflash_pager_.block_of(c);
            if (blk >= 0) kvflash_qk_pool_.pool_chunk(cache_.attn_k, blk, ct, c);
        }
        kvflash_qk_pooled_upto_ = sealed;
    }
    return true;
}

// Pool post-RoPE keys for chunks sealed before `committed` (QK policy).
// At seal time a chunk sits inside the protected tail window, so it is
// resident; a non-resident chunk here means we were called late (e.g.
// restored state) — it is skipped and scores missing (0) until repooled.
void Qwen35Backend::kvflash_qk_pool_to(int committed) {
    const int ct = kvflash_pager_.chunk_tokens();
    const auto range = kvflash_qk_pool_range(kvflash_qk_pooled_upto_, committed, ct);
    const int sealed = range.to;
    static const bool kTrace = std::getenv("LUCE_KVFLASH_TRACE_POOL") != nullptr;
    if (kTrace && sealed > range.from)
        std::fprintf(stderr, "[kvflash] qk pool: committed=%d chunks [%d,%d)\n",
                     committed, range.from, sealed);
    for (int c = range.from; c < sealed; c++) {
        const int blk = kvflash_pager_.block_of(c);
        if (blk < 0 || !kvflash_qk_pool_.pool_chunk(cache_.attn_k, blk, ct, c)) {
            std::fprintf(stderr, "[kvflash-qk] pool_chunk failed for chunk %d "
                                 "(block %d); chunk scores as missing\n", c, blk);
        }
    }
    kvflash_qk_pooled_upto_ = std::max(kvflash_qk_pooled_upto_, sealed);
}

// Regroup the last n_tok tokens' queries of the batch just computed from
// q_cap_all's [fa][head][T_alloc][hd] layout to the scorer's per-token
// [fa][head][hd]. Returns the token count (0 when not captured).
static int kvflash_read_captured_queries(const TargetCache & cache, const KvFlashQkDims & d,
                                        int batch_tokens, std::vector<float> & q_all) {
    if (!cache.q_cap_all) return 0;
    const int T = cache.q_cap_all_tokens;
    const int n_q = std::min(batch_tokens, T);
    const size_t qsize = (size_t)d.n_layers * d.n_q_heads * d.head_dim;
    std::vector<float> raw((size_t)ggml_nelements(cache.q_cap_all));
    ggml_backend_tensor_get(cache.q_cap_all, raw.data(), 0, raw.size() * sizeof(float));
    q_all.resize((size_t)n_q * qsize);
    const size_t planes = (size_t)d.n_layers * d.n_q_heads;
    for (int t = 0; t < n_q; t++)
        for (size_t p = 0; p < planes; p++)
            std::memcpy(q_all.data() + (size_t)t * qsize + p * d.head_dim,
                        raw.data() + (p * T + t) * d.head_dim, d.head_dim * sizeof(float));
    return n_q;
}

// C1.4: of the `captured` rows the graph wrote (the last `captured` tokens before
// q_capture_end), the LAST n_wanted — a slot's own queries in a packed batch.
int kvflash_read_captured_queries_tail(const TargetCache & cache, const KvFlashQkDims & d,
                                       int captured, int n_wanted, std::vector<float> & q_all) {
    if (!cache.q_cap_all) return 0;
    const int T = cache.q_cap_all_tokens;
    captured = std::min(captured, T);
    const int n_q = std::max(0, std::min(n_wanted, captured));
    const size_t qsize = (size_t)d.n_layers * d.n_q_heads * d.head_dim;
    std::vector<float> raw((size_t)ggml_nelements(cache.q_cap_all));
    ggml_backend_tensor_get(cache.q_cap_all, raw.data(), 0, raw.size() * sizeof(float));
    q_all.resize((size_t)n_q * qsize);
    const size_t planes = (size_t)d.n_layers * d.n_q_heads;
    for (int t = 0; t < n_q; t++)
        for (size_t p = 0; p < planes; p++)
            std::memcpy(q_all.data() + (size_t)t * qsize + p * d.head_dim,
                        raw.data() + (p * T + (size_t)(captured - n_q + t)) * d.head_dim, d.head_dim * sizeof(float));
    return n_q;
}

void Qwen35Backend::kvflash_upload_mask(int cur_pos) {
    if (!sg_.attn_mask) return;
    kvflash_require_base_zero(kvflash_pager_, "kvflash_upload_mask");
    const size_t need = (size_t)sg_.attn_mask->ne[0] * sg_.attn_mask->ne[1];
    // The decode step at cur_pos attends rows <= cur_pos only: rows above it hold
    // rejected drafts or an earlier turn's tokens and must stay hidden.
    if (kvflash_mask_buf_.size() != need || kvflash_pager_.epoch() != kvflash_mask_epoch_ ||
        cur_pos != kvflash_mask_pos_) {
        kvflash_mask_buf_.assign(need, F16_NEG_INF);
        kvflash_pager_.fill_slot_mask(kvflash_mask_buf_.data(), (int64_t)cur_pos + 1);   // q row 0
        kvflash_mask_epoch_ = kvflash_pager_.epoch();
        kvflash_mask_pos_ = cur_pos;
    }
    // Upload before EVERY compute: the input tensor's buffer region is
    // reused by graph execution, so a stale upload reads back as garbage.
    ggml_backend_tensor_set(sg_.attn_mask, kvflash_mask_buf_.data(), 0,
                            need * sizeof(uint16_t));
}

// Attach the drafter as the residency scorer outside the pflash compress
// path: with `--kvflash --prefill-drafter <gguf>` but compression off, the
// drafter would otherwise never load and the pool would silently run
// recency-only LRU. Loads lazily on the first reselect that needs it (and
// re-attaches after a draft-residency release frees the drafter).
void Qwen35Backend::kvflash_ensure_scorer() {
    if (kvflash_scorer_ || kvflash_drafter_path_.empty() || kvflash_drafter_failed_ || !kvflash_load_drafter()) {
        return;
    }
    kvflash_scorer_ = std::make_unique<KvFlashDrafterScorer>(&drafter_ctx_);
    std::fprintf(stderr, "[kvflash] drafter scorer attached (tau=%d)\n", kvflash_tau_);
}

// The residency drafter, loaded beside the target on first use; false after a failed load (not retried).
bool Qwen35Backend::kvflash_load_drafter() {
    if (drafter_loaded_) return true;
    if (kvflash_drafter_path_.empty() || kvflash_drafter_failed_) return false;
    ggml_backend_synchronize(target_backend_);
    if (draft_backend_) ggml_backend_synchronize(draft_backend_);
    std::fprintf(stderr, "[kvflash] loading drafter for residency scoring: %s\n",
                 kvflash_drafter_path_.c_str());
    if (!load_drafter(kvflash_drafter_path_, /*gpu_layers=*/999,
                      cfg_.device.gpu, drafter_ctx_)) {
        std::fprintf(stderr, "[kvflash] drafter load failed (%s); staying on "
                             "LRU residency\n", luce_last_error());
        kvflash_drafter_failed_ = true;
        return false;
    }
    drafter_loaded_ = true;
    drafter_ctx_.engine_max_ctx = cfg_.device.max_ctx;   // scoring sessions are never sized past it
    return true;
}

bool Qwen35Backend::kvflash_pflash2_scores(const std::vector<int32_t> & ids, int chunk_tokens,
                                           std::vector<float> & out, double & ms) {
    if (!kvflash_load_drafter()) return false;
    if (!kvflash_pflash2_) {
        kvflash_pflash2_ = std::make_unique<KvFlashPFlash2Scorer>(
            &drafter_ctx_, KvFlashPFlash2Scorer::Roles{w_.im_start_id, w_.user_id, w_.newline_id, w_.im_end_id, w_.tool_response_id, w_.assistant_id, w_.system_id});
    }
    // LUCE_KVFLASH_PFLASH2_DUMP_DIR: each scored prompt as counted int32 (the pflash daemon's input), for an offline replay.
    if (const char * dir = std::getenv("LUCE_KVFLASH_PFLASH2_DUMP_DIR"); dir && *dir) {
        static int n_dump = 0;
        const std::string path = std::string(dir) + "/ids_" + std::to_string(n_dump++) + ".bin";
        if (FILE * f = std::fopen(path.c_str(), "wb")) {
            const int32_t n = (int32_t)ids.size();
            std::fwrite(&n, sizeof n, 1, f);
            std::fwrite(ids.data(), sizeof(int32_t), ids.size(), f);
            std::fclose(f);
            std::fprintf(stderr, "[kvflash] pflash2 ids dumped: %s\n", path.c_str());
        }
    }
    const auto t0 = std::chrono::steady_clock::now();
    const bool ok = kvflash_pflash2_->score_chunks(ids, chunk_tokens, out);
    ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
    const PFlashScoringStats & st = kvflash_pflash2_->last_stats();
    std::fprintf(stderr, "[kvflash] pflash2 scorer: %s, %d tokens, %d segments, session resume %d, ran %d tokens, %.1f ms\n",
                 ok ? "ok" : "failed", (int)ids.size(), kvflash_pflash2_->last_segments(), st.resume, st.new_tokens, ms);
    return ok;
}

KvFlashQkDims Qwen35Backend::kvflash_exact_dims() const {
    KvFlashQkDims d;
    d.n_layers   = (int)cache_.attn_k.size();
    d.n_q_heads  = w_.n_head;
    d.n_kv_heads = w_.n_head_kv;
    d.head_dim   = w_.n_embd_head_k;
    return d;
}

// Fast path: one bulk device-to-host copy per layer (the whole pool tensor), the fused
// AVX2 scorer (kvflash_exact_fast.h) over it. Host-backed chunks are read in place.
bool Qwen35Backend::kvflash_exact_scores(int sealed, const std::vector<const float *> & qs,
                                         std::vector<std::vector<float>> & outs,
                                         std::vector<luce::common::KvFlashMassPlanes> * planes) {
    return kvflash_exact_scores_on(kvflash_pager_, sealed, qs, outs, planes);
}

// LUCE_KVFLASH_DUMP_QK=path: what the exact scorer is handed, for engine-vs-engine diffs. Per call: path.N.json with the
// per-plane norms of every query; on the first call of a session (sealed fell below half the last call's) also the full
// last 2 queries (path.N.q.bin, float32) and the raw K bytes of head 0, layers {0, n/2, n-1}, chunks {10, sealed/2, sealed-1}
// (path.N.k.bin: int32 c, l, evicted, nbytes, then the bytes). At most LUCE_KVFLASH_DUMP_QK_N calls (default 100).
void Qwen35Backend::kvflash_dump_qk(KvFlashPager & p, int sealed, const std::vector<const float *> & qs) {
    const char * path = std::getenv("LUCE_KVFLASH_DUMP_QK");
    if (!path || !*path || qs.empty() || cache_.attn_k.empty()) return;
    static int calls = 0, last_sealed = 0;
    const char * nmax = std::getenv("LUCE_KVFLASH_DUMP_QK_N");
    if (calls >= (nmax && *nmax ? std::atoi(nmax) : 100)) return;
    const int call = calls++;
    const bool heavy = call == 0 || sealed < last_sealed / 2;
    last_sealed = sealed;
    const KvFlashQkDims d = kvflash_exact_dims();
    const size_t planes = (size_t)d.n_layers * d.n_q_heads, qsize = planes * d.head_dim;
    const std::string base = std::string(path) + "." + std::to_string(call);
    if (FILE * f = std::fopen((base + ".json").c_str(), "w")) {
        std::fprintf(f, "{\"call\":%d,\"sealed\":%d,\"n_q\":%zu,\"planes\":%zu,\"heavy\":%d,\"norms\":[", call, sealed, qs.size(), planes, heavy ? 1 : 0);
        for (size_t i = 0; i < qs.size(); i++) {
            const auto n = luce::common::kvflash_q_plane_norms(qs[i], planes, (size_t)d.head_dim);
            std::fprintf(f, "%s[", i ? "," : "");
            for (size_t j = 0; j < n.size(); j++) std::fprintf(f, "%s%.7g", j ? "," : "", (double)n[j]);
            std::fprintf(f, "]");
        }
        std::fprintf(f, "]}\n");
        std::fclose(f);
    }
    if (!heavy) return;
    p.synchronize_paging();
    if (FILE * f = std::fopen((base + ".q.bin").c_str(), "wb")) {
        for (size_t i = qs.size() >= 2 ? qs.size() - 2 : 0; i < qs.size(); i++) std::fwrite(qs[i], sizeof(float), qsize, f);
        std::fclose(f);
    }
    const int ct = p.chunk_tokens();
    const ggml_tensor * k0 = cache_.attn_k[0];
    const size_t seg = (size_t)ct * k0->nb[1];
    std::vector<uint8_t> tmp(seg);
    FILE * fk = std::fopen((base + ".k.bin").c_str(), "wb");
    if (!fk) return;
    std::vector<int> chunks = {std::min(10, sealed - 1), sealed / 2, sealed - 1};
    chunks.erase(std::unique(chunks.begin(), chunks.end()), chunks.end());
    for (int c : chunks) for (int l : {0, d.n_layers / 2, d.n_layers - 1}) {
        const int b = p.block_of(c);
        const uint8_t * src = b < 0 ? p.host_k_seg(c, l, 0) : nullptr;
        if (b >= 0) {
            const ggml_tensor * t = cache_.attn_k[(size_t)l];
            ggml_backend_tensor_get(t, tmp.data(), (size_t)b * ct * t->nb[1], seg);
            src = tmp.data();
        }
        const int32_t hdr[4] = {c, l, b < 0 ? 1 : 0, src ? (int32_t)seg : 0};
        std::fwrite(hdr, sizeof hdr, 1, fk);
        if (src) std::fwrite(src, 1, seg, fk);
    }
    std::fclose(fk);
}

bool Qwen35Backend::kvflash_exact_scores_on(KvFlashPager & p, int sealed, const std::vector<const float *> & qs,
                                            std::vector<std::vector<float>> & outs,
                                            std::vector<luce::common::KvFlashMassPlanes> * planes) {
    const int ct = p.chunk_tokens();
    p.synchronize_paging();
    kvflash_dump_qk(p, sealed, qs);
    const KvFlashQkDims d = kvflash_exact_dims();
    if (ct <= 0 || sealed <= 0 || d.n_layers <= 0 || cache_.attn_k.empty()) return false;
    const ggml_tensor * k0 = cache_.attn_k[0];
    const size_t nseg = (size_t)d.n_layers * d.n_kv_heads;
    const auto tc0 = std::chrono::steady_clock::now();
    int n_res = 0;
    for (int c = 0; c < sealed; c++) n_res += p.block_of(c) >= 0;
    // Resident K: layer l -> buf[l * layer_bytes ...]. One async D2H per layer into a persistent pinned
    // buffer (no page faults, full PCIe rate); falls back to a pageable vector if the device has no host buffer type.
    const size_t layer_bytes = n_res ? ggml_nbytes(k0) : 0;
    const size_t need = layer_bytes * d.n_layers;
    uint8_t * buf = nullptr;
    bool pinned = false;
    if (need) {
        if (kvflash_exact_pin_ && ggml_backend_buffer_get_size(kvflash_exact_pin_) < need) {
            ggml_backend_buffer_free(kvflash_exact_pin_); kvflash_exact_pin_ = nullptr;
        }
        if (!kvflash_exact_pin_ && !kvflash_exact_pin_failed_) {
            ggml_backend_dev_t dev = ggml_backend_get_device(target_backend_);
            ggml_backend_buffer_type_t bt = dev ? ggml_backend_dev_host_buffer_type(dev) : nullptr;
            kvflash_exact_pin_ = bt ? ggml_backend_buft_alloc_buffer(bt, need) : nullptr;
            kvflash_exact_pin_failed_ = !kvflash_exact_pin_;
        }
        if (kvflash_exact_pin_) { buf = (uint8_t *)ggml_backend_buffer_get_base(kvflash_exact_pin_); pinned = true; }
        else {
            if (kvflash_exact_buf_.size() < need) kvflash_exact_buf_.resize(need);
            buf = kvflash_exact_buf_.data();
        }
        for (int l = 0; l < d.n_layers; l++) {
            if (pinned) ggml_backend_tensor_get_async(target_backend_, cache_.attn_k[(size_t)l], buf + (size_t)l * layer_bytes, 0, layer_bytes);
            else        ggml_backend_tensor_get(cache_.attn_k[(size_t)l], buf + (size_t)l * layer_bytes, 0, layer_bytes);
        }
        if (pinned) ggml_backend_synchronize(target_backend_);
    }
    const auto tc1 = std::chrono::steady_clock::now();
    std::vector<int> blk((size_t)sealed);
    for (int c = 0; c < sealed; c++) blk[(size_t)c] = p.block_of(c);
    luce::common::KvFlashExactKeys keys;
    keys.n_chunks = sealed;
    keys.chunk_tokens = ct;
    keys.type = k0->type;
    keys.seg = [&](int c, int l, int h) -> const uint8_t * {
        const int b = blk[(size_t)c];
        if (b < 0) return p.host_k_seg(c, l, h);
        const ggml_tensor * t = cache_.attn_k[(size_t)l];
        return buf + (size_t)l * layer_bytes + (size_t)b * ct * t->nb[1] + (size_t)h * t->nb[2];
    };
    luce::common::KvFlashExactTiming tm;
    const bool ok = luce::common::kvflash_exact_mass_scores_fast_multi(keys, d, qs, outs, 0, &tm, planes);
    const auto tc2 = std::chrono::steady_clock::now();
    std::fprintf(stderr, "[kvflash] exact-scorer fast split: n_q=%d sealed=%d resident=%d copy %.1f ms (%d bulk copies, %.0f MB), math %.1f ms "
                 "[cpu-ms summed over threads: dequant %.1f dot %.1f exp %.1f]\n", (int)qs.size(), sealed, n_res,
                 std::chrono::duration<double, std::milli>(tc1 - tc0).count(), n_res ? d.n_layers : 0,
                 (double)layer_bytes * (n_res ? d.n_layers : 0) / 1e6,
                 std::chrono::duration<double, std::milli>(tc2 - tc1).count(),
                 tm.dequant_ns.load() / 1e6, tm.dot_ns.load() / 1e6, tm.exp_ns.load() / 1e6);
    (void)nseg;
    return ok;
}

// Selector v2 on the target backend (kvflash_gpu_mass.h): the pool K tensors are graph leaves, evicted chunks are
// staged from the pager's pinned host copies. Pool blocks no sealed chunk maps to (free, or a chunk >= sealed: this
// turn's first pass) are masked in the graph. The first failure (alloc, free-VRAM guard, compute) is logged once and
// sends this and every later call to the CPU scorer.
bool Qwen35Backend::kvflash_exact_scores_gpu(int sealed, const std::vector<const float *> & qs,
                                             std::vector<std::vector<float>> & outs,
                                             std::vector<luce::common::KvFlashMassPlanes> & planes) {
    return kvflash_exact_scores_gpu_on(kvflash_pager_, sealed, qs, outs, planes);
}

bool Qwen35Backend::kvflash_exact_scores_gpu_on(KvFlashPager & p, int sealed, const std::vector<const float *> & qs,
                                                std::vector<std::vector<float>> & outs,
                                                std::vector<luce::common::KvFlashMassPlanes> & planes) {
    if (kvflash_gpu_failed_) return false;
    const int ct = p.chunk_tokens();
    p.synchronize_paging();   // queued D2H page-outs and H2D page-ins must land before any host/pool read
    kvflash_dump_qk(p, sealed, qs);
    const KvFlashQkDims d = kvflash_exact_dims();
    std::string why;
    if (ct <= 0 || sealed <= 0 || d.n_layers <= 0 || cache_.attn_k.empty() || !target_backend_) why = "no pooled K";
    else {
        if (!kvflash_gpu_mass_) kvflash_gpu_mass_ = std::make_unique<luce::common::KvFlashGpuMass>(target_backend_);
        std::vector<int> blk((size_t)sealed);
        int n_res = 0;
        for (int c = 0; c < sealed; c++) { blk[(size_t)c] = p.block_of(c); n_res += blk[(size_t)c] >= 0; }
        luce::common::KvFlashGpuMassTiming tm;
        const std::vector<ggml_tensor *> pool_k(cache_.attn_k.begin(), cache_.attn_k.end());
        if (kvflash_gpu_mass_->score(pool_k, blk, [&](int c, int l, int h) { return p.host_k_seg(c, l, h); },
                                     ct, d, qs, outs, &planes, &tm, &why)) {
            std::fprintf(stderr, "[kvflash] gpu scorer: n_q=%d sealed=%d resident=%d gather %.1f h2d %.1f graph %.1f d2h %.1f agg %.1f total %.1f ms batches=%d peak=%.0f MiB\n",
                         (int)qs.size(), sealed, n_res, tm.gather, tm.h2d, tm.graph, tm.d2h, tm.agg, tm.total, tm.batches, tm.peak_bytes / 1048576.0);
            return true;
        }
    }
    kvflash_gpu_failed_ = true;
    std::fprintf(stderr, "[kvflash] gpu scorer unavailable: %s; using the CPU scorer\n", why.c_str());
    return false;
}

// LUCE_KVFLASH_UQ_QUERIES: append the stored rows of the latest human question to the selector's queries
// when the store's key matches this prompt (and, on the GPU scorer, the call still fits in VRAM).
// Logs one hit or miss line per reselect.
void Qwen35Backend::kvflash_uq_append(std::vector<const float *> & qs, int sealed, size_t qsize, size_t n_v2) {
    const int found = luce::common::kvflash_uq_find(kvflash_uq_stores_, kvflash_uq_req_hi_, kvflash_uq_req_hash_);
    static const luce::common::KvFlashUqStore kNone;
    const auto & st = found >= 0 ? kvflash_uq_stores_[(size_t)found] : kNone;
    const char * why = nullptr;
    std::string reason;
    if (n_v2 == 0) why = "no-v2-selector";
    else if (kvflash_uq_req_hi_ <= 0 || st.empty() || st.rows.size() != (size_t)st.n * qsize)
        why = kvflash_uq_miss_.empty() ? "no-store" : kvflash_uq_miss_.c_str();
    else if (kvflash_scorer_gpu_ && !kvflash_gpu_failed_ && target_backend_ && !cache_.attn_k.empty()) {
        if (!kvflash_gpu_mass_) kvflash_gpu_mass_ = std::make_unique<luce::common::KvFlashGpuMass>(target_backend_);
        const int ct = kvflash_pager_.chunk_tokens();
        const ggml_tensor * k0 = cache_.attn_k[0];
        int n_ev = 0;
        for (int c = 0; c < sealed; c++) n_ev += kvflash_pager_.block_of(c) < 0;
        size_t need = 0;
        if (!kvflash_gpu_mass_->fits_now(kvflash_exact_dims(), k0->type, ct, (int)(k0->ne[1] / ct), n_ev,
                                         (int)qs.size() + st.n, &need)) {
            reason = "gpu-vram need " + std::to_string(need >> 20) + " MiB";
            why = reason.c_str();
        }
    }
    if (why) {
        std::fprintf(stderr, "[kvflash] uq miss reason=%s\n", why);
        return;
    }
    luce::common::kvflash_uq_append_queries(qs, st, qsize);
    std::fprintf(stderr, "[kvflash] uq hit lo=%d hi=%d n=%d\n", st.lo, st.hi, st.n);
}

// LUCE_KVFLASH_PREFILL_RESELECT: score the chunks sealed before `sealed_tokens`
// with the query held in cache_.q_cap.
bool Qwen35Backend::kvflash_prefill_score(int sealed_tokens, double & score_ms) {
    const int ct = kvflash_pager_.chunk_tokens();
    const KvFlashQkDims & d = kvflash_qk_pool_.dims();
    if (ct <= 0 || d.n_layers <= 0 || !cache_.q_cap) return false;
    const size_t qsize = (size_t)d.n_layers * d.n_q_heads * d.head_dim;
    if ((size_t)ggml_nelements(cache_.q_cap) != qsize) return false;
    std::vector<float> q(qsize);
    const auto tq0 = KvflashM4Phases::Clock::now();
    ggml_backend_tensor_get(cache_.q_cap, q.data(), 0, qsize * sizeof(float));
    kvflash_m4_.qread = std::chrono::duration<double, std::milli>(KvflashM4Phases::Clock::now() - tq0).count();
    const int sealed = sealed_tokens / ct;
    const auto t0 = std::chrono::steady_clock::now();
    // Selector v2: the last kvflash_select_queries_ prompt queries of the first pass
    // (q_cap's is the newest), max over queries of the top-plane mass.
    const size_t n_v2 = kvflash_q_tail_.size() % qsize ? 0
        : std::min(kvflash_q_tail_.size() / qsize, (size_t)kvflash_select_queries_);
    bool gpu_scored = false;
    std::vector<const float *> qs = {q.data()};
    if (kvflash_exact_scorer_) {
        if (n_v2 > 0) {
            qs.clear();
            for (size_t o = kvflash_q_tail_.size() - n_v2 * qsize; o < kvflash_q_tail_.size(); o += qsize)
                qs.push_back(kvflash_q_tail_.data() + o);
        }
        if (kvflash_uq_queries_ > 0) kvflash_uq_append(qs, sealed, qsize, n_v2);
        std::vector<std::vector<float>> o;
        std::vector<luce::common::KvFlashMassPlanes> planes;
        gpu_scored = n_v2 > 0 && kvflash_scorer_gpu_ && kvflash_exact_scores_gpu(sealed, qs, o, planes);
        if (!gpu_scored && !kvflash_exact_scores(sealed, qs, o, n_v2 > 0 ? &planes : nullptr)) {
            std::fprintf(stderr, "[kvflash] exact scorer failed @%d; keeping the pool\n", sealed_tokens);
            return false;
        }
        kvflash_scores_ = n_v2 > 0 ? luce::common::kvflash_select_scores(planes) : std::move(o[0]);
        if (kvflash_select_queries_ > 0)
            std::fprintf(stderr, "[kvflash] selector v2: %zu of %d queries (max over queries, top-%d planes)\n",
                         n_v2, kvflash_select_queries_, luce::common::kExactTopPlanes);
    } else {
        std::vector<const float *> pk((size_t)sealed, nullptr);
        for (int c = 0; c < sealed; c++) pk[(size_t)c] = kvflash_qk_pool_.data(c);
        kvflash_qk_chunk_scores(pk, q.data(), d, kvflash_scores_);
    }
    score_ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
    return true;
}

// With LUCE_KVFLASH_RECALL_BLOCKS=k the pool becomes an exact LRU window plus
// k recalled 10-chunk blocks (R-LRU). reserve_chunks is the pool room kept for
// the rest of the prefill (derived from `end` when negative).
void Qwen35Backend::kvflash_prefill_apply(int committed, int end, int reserve_chunks, int q_tok,
                                          double score_ms, const char * note) {
    kvflash_prefill_apply_on(kvflash_pager_, kvflash_scores_, committed, end, reserve_chunks, q_tok, score_ms, note, -1);
}

void Qwen35Backend::kvflash_prefill_apply_on(KvFlashPager & p, std::vector<float> & scores, int committed, int end,
                                             int reserve_chunks, int q_tok, double score_ms, const char * note, int slot,
                                             int slot_pins) {
    const int ct = p.chunk_tokens();
    int tail = -1;   // the pager's configured tail unless recall derives one
    std::vector<int> picked;
    std::vector<float> raw_dump;   // LUCE_KVFLASH_DUMP_SCORES: the same, for R's serial path too
    std::vector<float> raw;   // the scores before the picks and the window are promoted (logged per slot)
    int first_window = 0;
    if (kvflash_recall_blocks_ > 0) {
        constexpr int kCpb = 10;   // budget: kCpb * k chunks, picked in units of `unit` chunks
        const int unit = kvflash_select_unit_;
        if (slot >= 0) raw = scores;
        if (std::getenv("LUCE_KVFLASH_DUMP_SCORES")) raw_dump = scores;
        // The pin inputs: a slot's own accepted pins (P2-01), else the serial pager's. The serial pins sit in the
        // extra capacity LUCE_KVFLASH_PREAMBLE_ROWS added (0 when off): the history budget (tail + recall + reserve)
        // is the feature-off one, and the pinned chunks are not recall candidates. A slot's pins sit inside its
        // slice, so they come off the history budget and the combined budget is checked (kvflash_recall_plan).
        luce::common::KvFlashRecallIn in;
        in.n_pool_blocks = p.n_pool_blocks();
        in.cur_chunk = p.cur_chunk();
        in.recall_chunks = kCpb * kvflash_recall_blocks_;
        in.unit = unit;
        in.reserve = reserve_chunks >= 0 ? reserve_chunks : (end - committed + ct - 1) / ct + 1;
        in.pin_reserve_chunks = kvflash_preamble_reserve_ / ct;   // the history budget never depends on k
        in.pinned_chunks = slot >= 0 ? slot_pins
            : luce::common::kvflash_pin_chunks(kvflash_preamble_end_, kvflash_preamble_rows_, ct);
        in.sink_chunks = p.sink_chunks();
        in.min_tail_chunks = p.tail_window_chunks();
        in.slot_pins = slot >= 0;
        luce::common::KvFlashRecallPlan plan = luce::common::kvflash_recall_plan(scores, in);
        if (plan.clamped || !plan.fits)
            std::fprintf(stderr, "[kvflash] slot %d recall budget: %d pinned + reserve %d + sink %d + tail window %d "
                         "leave %d recall blocks of %d asked in a pool of %d%s\n", slot, slot_pins, in.reserve,
                         in.sink_chunks, in.min_tail_chunks, plan.k_units, kCpb * kvflash_recall_blocks_ / unit,
                         in.n_pool_blocks, plan.fits ? "" : " (pins and reserve alone do not fit)");
        tail = plan.tail;
        first_window = plan.first_window;
        picked = std::move(plan.picked);
        for (int b : picked)
            for (int c = b * unit; c < (b + 1) * unit; c++) scores[(size_t)c] = 1e30f;
        // The window is kept exactly: it must also survive evictions in the
        // rest of the prefill, so pin it in the ledger the hook reads.
        for (int c = std::max(0, first_window); c < (int)scores.size(); c++)
            scores[(size_t)c] = 1e30f;
    }
    p.score_hook = [&scores](int c) {
        return c < (int)scores.size() ? scores[c] : 1e30f;
    };
    const char * dump_path = std::getenv("LUCE_KVFLASH_DUMP_SCORES");
    std::vector<int> dump_res;
    if (dump_path && *dump_path) for (int c = 0; c < (int)raw_dump.size(); c++) if (p.block_of(c) >= 0) dump_res.push_back(c);
    const int64_t ins0 = p.stats().page_ins;
    const int events = p.reselect(tail);
    const int64_t ins = p.stats().page_ins - ins0;
    std::string ids;
    for (int b : picked) ids += (ids.empty() ? "" : ",") + std::to_string(b);
    std::fprintf(stderr, "[kvflash]%s prefill reselect mode=%d scorer=%s k=%d tail=%d @%d (q tok %d) "
                 "picked=[%s]: %d page events; %d in (%.1f MiB H2D) (resident %d/%d blocks) scorer %.1f ms%s\n",
                 slot >= 0 ? (" slot " + std::to_string(slot)).c_str() : "",
                 kvflash_prefill_reselect_, kvflash_pflash2_scorer_ ? "pflash2" : kvflash_exact_scorer_ ? "exact" : "cosine",
                 kvflash_recall_blocks_, tail, committed, q_tok, ids.c_str(), events,
                 (int)ins, (double)ins * (double)p.chunk_bytes() / 1048576.0,
                 p.resident_blocks(), p.n_pool_blocks(), score_ms, note);
    if (dump_path && *dump_path && !picked.empty()) {
        static int dump_seq = 0;
        if (FILE * f = std::fopen(dump_path, "a")) {
            std::fprintf(f, "%s\n", luce::common::kvflash_dump_line(dump_seq, committed, first_window, kvflash_select_unit_, picked, dump_res, raw_dump).c_str());
            std::fclose(f);
        }
        std::fprintf(stderr, "[kvflash] score dump #%d\n", dump_seq++);
    }
    if (slot >= 0 && !picked.empty()) {   // C1.4 parity: the picks' unit scores and the best runner-up, so a flipped pick shows its margin
        const int unit = kvflash_select_unit_;
        auto unit_sum = [&](int b) { double s = 0; for (int c = b * unit; c < (b + 1) * unit; c++) s += raw[(size_t)c]; return s; };
        std::string sc;
        for (int b : picked) { char buf[48]; std::snprintf(buf, sizeof buf, "%s%d=%.4g", sc.empty() ? "" : ",", b, unit_sum(b)); sc += buf; }
        int rb = -1; double rs = 0;
        for (int b = 1; (b + 1) * unit <= std::min(first_window, (int)raw.size()); b++)
            if (std::find(picked.begin(), picked.end(), b) == picked.end() && (rb < 0 || unit_sum(b) > rs)) { rb = b; rs = unit_sum(b); }
        std::fprintf(stderr, "[kvflash] slot %d pick scores=[%s] runner-up %d=%.4g\n", slot, sc.c_str(), rb, rs);
    }
    if (kvflash_select_unit_ != 10)
        std::fprintf(stderr, "[kvflash] prefill reselect picked units of %d chunks\n", kvflash_select_unit_);
}

void Qwen35Backend::kvflash_prefill_reselect_now(int committed, int end) {
    double ms = 0.0;
    if (!kvflash_prefill_score(committed, ms)) return;
    kvflash_prefill_apply(committed, end, -1, committed - 1, ms, "");
}

// Rewind of the pass over [r, end) (modes 3 and 4). Everything that can fail
// without touching state (scoring) runs first. Then the state restore, which is
// the same call live restore uses with skip_kv (or the zero state when r == 0);
// if it fails the request errors out rather than continuing on a half-restored
// cache.
int Qwen35Backend::kvflash_rewind_turn(const PrefixSnapshot & snap, bool zero_state, int r, int end,
                                       int reserve_chunks) {
    const int ct = kvflash_pager_.chunk_tokens();
    double ms = 0.0;
    if (!kvflash_prefill_score(r, ms)) {
        std::fprintf(stderr, "[kvflash] prefill reselect mode=%d @%d: scoring failed; "
                             "keeping the pre-rewind result\n", kvflash_prefill_reselect_, r);
        return 0;
    }
    kvflash_m4_.score = ms;
    const auto tr0 = KvflashM4Phases::Clock::now();
    if (zero_state) {
        reset_recurrent_state(cache_);
    } else if (!restore_target_cache(snap, cache_, /*skip_kv=*/true)) {
        std::fprintf(stderr, "[kvflash] prefill reselect mode=%d @%d: state restore failed\n",
                     kvflash_prefill_reselect_, r);
        set_last_error("kvflash: rewind state restore failed");
        return -1;
    }
    // Rows >= r are dropped; the chunk holding r keeps rows < r. kvflash_scores_
    // is NOT cleared: the pick must survive until the re-run allocates.
    kvflash_pager_.truncate(r);
    kvflash_m4_.rewind = std::chrono::duration<double, std::milli>(KvflashM4Phases::Clock::now() - tr0).count();
    kvflash_apply_preamble_pin(nullptr, false);   // truncate() below the preamble clips the pins
    {
        const auto ta0 = KvflashM4Phases::Clock::now();
        const luce::common::KvFlashDmaCounters d0 = kvflash_pager_.dma_counters();
        kvflash_prefill_apply(r, end, reserve_chunks, end - 1, ms, "; rewind");
        kvflash_m4_.apply = std::chrono::duration<double, std::milli>(KvflashM4Phases::Clock::now() - ta0).count();
        kvflash_m4_.page_submit = kvflash_pager_.dma_counters().submit_ms - d0.submit_ms;
        kvflash_m4_.page_wait = kvflash_pager_.dma_counters().wait_ms - d0.wait_ms;
    }
    cache_.cur_pos = r;
    kvflash_mask_epoch_ = (uint64_t)-1;
    if (kvflash_qk_policy_) kvflash_qk_pooled_upto_ = std::min(kvflash_qk_pooled_upto_, r / ct);
    return 1;
}

// One line per rewound mode-4 request. Phases are disjoint host wall intervals;
// `pre` (prefill before R) and `resid` (the rest of total) close the sum.
// Every boundary is closed by a host sync already in the code (tensor get/set/copy
// or a pager synchronize) except the reset phase when `async=reset` is logged:
// its page-outs may still be in flight and then land in the next phase.
void Qwen35Backend::kvflash_log_m4_phases() {
    KvflashM4Phases & m = kvflash_m4_;
    if (!m.rewound) return;
    const auto now = KvflashM4Phases::Clock::now();
    m.replay = std::chrono::duration<double, std::milli>(now - m.t_replay0).count();
    const double total = std::chrono::duration<double, std::milli>(now - m.t_start).count();
    const double resid = total - (m.reset + m.pre + m.snap + m.first + m.qread + m.score +
                                  m.rewind + m.apply + m.replay);
    std::fprintf(stderr, "[kvflash] mode4 phases ms: reset=%.1f snap=%.1f first=%.1f qread=%.1f score=%.1f "
                 "rewind=%.1f page_submit=%.1f page_wait=%.1f replay=%.1f total=%.1f "
                 "(pre=%.1f apply=%.1f resid=%.1f%s)\n",
                 m.reset, m.snap, m.first, m.qread, m.score, m.rewind, m.page_submit, m.page_wait,
                 m.replay, total, m.pre, m.apply, resid, m.reset_async ? " async=reset" : "");
    const luce::common::KvFlashDmaCounters & d = kvflash_pager_.dma_counters();
    std::fprintf(stderr, "[kvflash] mode4 calls: memcpy_d2h=%lld memcpy_h2d=%lld memset=%lld "
                 "page_ins=%lld page_outs=%lld\n",
                 (long long)(d.memcpy_d2h - m.dma0.memcpy_d2h), (long long)(d.memcpy_h2d - m.dma0.memcpy_h2d),
                 (long long)(d.memset - m.dma0.memset),
                 (long long)(kvflash_pager_.stats().page_ins - m.ins0),
                 (long long)(kvflash_pager_.stats().page_outs - m.outs0));
}

bool Qwen35Backend::kvflash_q_cap_from_stage(int new_committed) {
    const int row = kvflash_q_stage_row(cache_.q_stage_base, cache_.q_stage_n, new_committed);
    if (row < 0 || !cache_.q_stage || !cache_.q_cap) return false;
    const size_t bytes = ggml_nbytes(cache_.q_cap);
    if (cache_.q_stage->nb[3] != bytes) return false;
    std::vector<uint8_t> q(bytes);
    ggml_backend_tensor_get(cache_.q_stage, q.data(), (size_t)row * bytes, bytes);
    ggml_backend_tensor_set(cache_.q_cap, q.data(), 0, bytes);
    return true;
}

void Qwen35Backend::kvflash_maybe_reselect(int generated) {
    if (kvflash_tau_ <= 0) return;
    // Adaptive tau: a rescore costs ~0.11 ms per history token (full 0.6B
    // re-prefill; measured 0.9 s @8K, ~46 s bisected @256K), while decode
    // produces ~30 tok/s. Capping rescore overhead at ~15% of decode time
    // gives tau ~= history/45. The configured tau is the floor.
    const int tau = std::max<int>(kvflash_tau_, (int)(kvflash_history_.size() / 45));
    if (generated % tau != 0) return;
    // Lazy-load the drafter only when a rescore is actually due, so the
    // first tokens of the first request never pay the load.
    if (!kvflash_scorer_) kvflash_ensure_scorer();
    if (!kvflash_scorer_) return;
    if (kvflash_qk_scorer_) {
        // Feed the last decode step's captured query (post-RoPE/-rotation,
        // [n_fa, n_head, head_dim] f32) before scoring.
        if (!cache_.q_cap) return;
        std::vector<float> q((size_t)ggml_nelements(cache_.q_cap));
        ggml_backend_tensor_get(cache_.q_cap, q.data(), 0, q.size() * sizeof(float));
        kvflash_qk_scorer_->set_query(q.data(), q.size());
        if (std::getenv("LUCE_KVFLASH_TRACE_POOL")) {
            double sum = 0.0, l2 = 0.0;
            for (float x : q) { sum += x; l2 += (double)x * x; }
            std::fprintf(stderr, "[kvflash] reselect query @gen=%d cur_pos=%d hist=%zu sum=%.4f l2=%.4f\n",
                         generated, cache_.cur_pos, kvflash_history_.size(), sum, std::sqrt(l2));
        }
    }
    if (!kvflash_scorer_->score_chunks(kvflash_history_, kvflash_pager_.chunk_tokens(), kvflash_scores_)) {
        return;  // scorer failure -> keep LRU behavior this round
    }
    kvflash_pager_.score_hook = [this](int c) {
        return c < (int)kvflash_scores_.size() ? kvflash_scores_[c] : 1e30f;
    };
    const int events = kvflash_pager_.reselect();
    // Logged also for an unchanged selection, so a skipped reselect cannot hide.
    std::fprintf(stderr, "[kvflash] reselect @gen=%d: %d page events "
                 "(resident %d/%d blocks)\n",
                 generated, events, kvflash_pager_.resident_blocks(),
                 kvflash_tokens_ / kvflash_pager_.chunk_tokens());
}

bool Qwen35Backend::do_ar_decode(int committed, ThinkingBudget & budget,
                                  std::vector<int32_t> & out_tokens,
                                  const DaemonIO & io,
                                  bool * forced_close_out,
                                  bool * degenerate_close_out,
                                  size_t reselect_origin) {
    const size_t out_tokens_at_entry = out_tokens.size();
    // A spec step that hands off passes the request's output origin, so the
    // reselect boundaries stay global; AR entered directly counts from here.
    const size_t reselect_from = reselect_origin == SIZE_MAX ? out_tokens_at_entry : reselect_origin;
    auto apply_budget = [&](int32_t & tok) {
        const int remaining = budget.remaining();
        const bool was_forced = budget.forced_close();
        const int32_t sampled = tok;
        tok = budget.apply(sampled).token;
        if (budget.forced_close() && !was_forced) {
            std::fprintf(stderr,
                "[budget-hook] force-close at remaining=%d: overriding "
                "sampled token %d with close[0]=%d\n",
                remaining, sampled, tok);
            if (forced_close_out) *forced_close_out = true;
        }
    };
    // Logit-equivalence gate hooks: inert unless the env names a file.
    static const luce::common::LogitsDump logits_dump =
        luce::common::LogitsDump::from_env(std::getenv("LUCE_LOGITS_DUMP"));
    static const luce::common::TeacherForce teacher =
        luce::common::TeacherForce::from_env(std::getenv("LUCE_TEACHER_FORCE"));
    auto force_token = [&](int32_t & tok) {
        const int32_t f = teacher.at(out_tokens.size() - out_tokens_at_entry);
        if (f >= 0) tok = f;
    };
    if (budget.remaining() <= 0) return true;

    auto t_dec0_ar = std::chrono::steady_clock::now();
    static const int _repeat_guard = []{
        const int explicit_guard =
            env_int_or_default("LUCE_DEGENERATE_RUN_TOKENS", -1);
        if (explicit_guard >= 0) return explicit_guard;
        return dflash_min_tokens_floor() > 0 ? 32 : 0;
    }();

    const int hidden = w_.n_embd;
    const int vocab  = w_.n_vocab;
    std::vector<float> logits_buf(vocab);
    std::vector<float> embed_buf_vec(hidden);
    float * embed_buf = embed_buf_vec.data();

    // First token: consume the final prefill position.  Do not derive this
    // offset from committed/KV position: restore paths can prefill a delta at
    // nonzero KV offsets, and committed then no longer describes chunk size.
    //
    // Continuation mode keeps out_tokens.back() pending at `committed`.
    // Forwarding it below updates KV, but never emits or accounts it again.
    if (out_tokens.empty()) {
        int32_t first_tok;
        if (logits_dump.enabled() && prefill_last_logits_valid_) {
            ggml_backend_tensor_get(sg_.logits, logits_buf.data(), prefill_last_logits_offset_,
                                    sizeof(float) * vocab);
            logits_dump.append(logits_buf.data(), vocab);
        }
        if (sampler_.needs_logit_processing()) {
            if (!prefill_last_logits_valid_) return false;
            ggml_backend_tensor_get(sg_.logits, logits_buf.data(), prefill_last_logits_offset_,
                                    sizeof(float) * vocab);
            first_tok = sample_logits(logits_buf.data(), vocab, sampler_,
                                      out_tokens, sampler_rng_);
        } else {
            first_tok = cache_.last_tok;
        }
        force_token(first_tok);
        apply_budget(first_tok);
        out_tokens.push_back(first_tok);
        io.emit(first_tok);
        if (kvflash_active()) kvflash_history_.push_back(first_tok);
        if (io.is_cancelled() || IS_EOS_TOK(first_tok, w_)) return true;
        // The first token is pending: the prefill logits produced it, but its
        // K/V row has not been written yet. The first loop iteration below
        // forwards it at `committed`; only that compute may advance the cache.
    }

    // AR decode loop for remaining tokens
    while (budget.remaining() > 0) {
        int32_t tok = out_tokens.back();

        if (!w_.embedder.embed(&tok, 1, embed_buf)) return false;

        // kvflash: graph carries a slot-validity mask alongside the
        // step-invariant set_rows write; the FA span clamps to the pool.
        const bool pool = kvflash_active();
        const bool paged = cfg_.paged_attention;
        if (!build_target_step(sg_, w_, cache_, target_backend_,
                               /*kv_start=*/committed, /*n_tokens=*/1,
                               /*with_mask=*/pool || w_.is_bailingmoe3,
                               /*capture=*/false,
                               /*capture_delta_intermediate=*/false,
                               /*fa_window=*/0,
                               /*logits_tail_rows=*/0,
                               cfg_.kq_stride_pad,
                               should_capture_moe_router(),
                               /*kvflash_mask=*/pool,
                               /*capture_qk=*/pool && kvflash_qk_policy_,
                               /*paged_attention=*/paged)) {
            return false;
        }

        // Step inputs go in after the build: build_target_step frees and
        // reallocates the graph, so inputs written before it land in the old
        // graph's buffer. The step shape repeats between AR steps, which hid
        // that; the first AR step after a prefill or a verify block changes
        // the shape and read stale positions (K rope dims of that row wrong).
        ggml_backend_tensor_set(sg_.inp_embed, embed_buf, 0, sizeof(float) * hidden);
        const int32_t rope_pos = committed + rope_delta_;
        const int32_t pos4[4] = {rope_pos, rope_pos, rope_pos, 0};
        ggml_backend_tensor_set(sg_.positions, pos4, 0, sizeof(int32_t) * 4);

        // Fill kv_write_rows with this step's cache slot for set_rows: the
        // paged append row, its pool slot in kvflash mode, or the logical
        // position directly.
        if (sg_.kv_write_rows) {
            int64_t slot = committed;
            if (paged) {
                if (!prepare_paged_decode_step((uint32_t)committed, slot)) {
                    return false;
                }
            } else if (pool) {
                slot = (int64_t)kvflash_pager_.slot_for(committed);
                if (slot < 0) {
                    std::fprintf(stderr, "[kvflash] no pool slot at pos %d "
                                         "(pool %d exhausted)\n",
                                 committed, kvflash_tokens_);
                    set_last_error("kvflash: no evictable pool block");
                    return false;
                }
            }
            const int n_head_kv = w_.n_head_kv;
            std::vector<int64_t> row_vals(n_head_kv, slot);
            ggml_backend_tensor_set(sg_.kv_write_rows, row_vals.data(), 0,
                                    sizeof(int64_t) * n_head_kv);
        }
        if (pool) {
            kvflash_upload_mask(committed);
        } else if (w_.is_bailingmoe3) {
            upload_qwen35_causal_mask(
                sg_.attn_mask, committed, 1, cfg_.kq_stride_pad);
        }

        auto st = ggml_backend_graph_compute(target_backend_, sg_.gf);
        if (st != GGML_STATUS_SUCCESS) return false;

        after_target_compute(sg_, committed, 1);

        if (logits_dump.enabled()) {
            ggml_backend_tensor_get(sg_.logits, logits_buf.data(), 0, sizeof(float) * vocab);
            logits_dump.append(logits_buf.data(), vocab);
        }

        // GPU argmax: read 4 bytes, skip the 970 KB logit D2H. Escape: LUCE_GPU_ARGMAX=0.
        static const bool kGpuArgmaxAR = []() {
            const char * v = std::getenv("LUCE_GPU_ARGMAX");
            return v == nullptr || v[0] != '0';
        }();
        int32_t next_tok;
        if (sampler_.needs_logit_processing()) {
#ifdef LUCE_HAVE_GPU_SAMPLER
            // GPU sample straight from the device logits tensor, skipping the
            // full ~vocab-wide D2H copy (the same payoff LUCE_GPU_ARGMAX gets
            // for greedy). Falls back to the CPU chain on -1.
            int g_tok = -1;
            if (gpu_sampler_enabled() && gpu_sampler_supports(sampler_) &&
                sg_.logits && sg_.logits->data &&
                !ggml_backend_buft_is_meta(
                    ggml_backend_get_default_buffer_type(target_backend_))) {
                // Draw the uniform from a copy of the RNG so the real stream is
                // not advanced yet; we only commit that single draw if the GPU
                // path succeeds (below). On fallback the stream is untouched and
                // sample_logits() draws the identical value it would with the
                // GPU disabled, so the token stream is reproducible either way.
                double r = 0.0;
                if (sampler_.temp > 0.0f) {
                    std::mt19937_64 rng_peek = sampler_rng_;
                    std::uniform_real_distribution<double> u(0.0, 1.0);
                    r = u(rng_peek);
                }
                g_tok = geometric_sample_logits_cuda(
                    static_cast<const float *>(sg_.logits->data), vocab, sampler_,
                    out_tokens, r, /*logits_on_device=*/true);
            }
            if (g_tok >= 0) {
                // Commit the single uniform the GPU draw consumed, so the RNG
                // advances by exactly one draw — matching the CPU/GPU-off path.
                if (sampler_.temp > 0.0f) {
                    std::uniform_real_distribution<double> u(0.0, 1.0);
                    (void)u(sampler_rng_);
                }
                next_tok = g_tok;
            } else
#endif
            {
                ggml_backend_tensor_get(sg_.logits, logits_buf.data(), 0,
                                        sizeof(float) * vocab);
                next_tok = sample_logits(logits_buf.data(), vocab, sampler_,
                                          out_tokens, sampler_rng_);
            }
        } else if (kGpuArgmaxAR && sg_.argmax_tokens) {
            int32_t tok_i = 0;
            ggml_backend_tensor_get(sg_.argmax_tokens, &tok_i, 0, sizeof(int32_t));
            next_tok = tok_i;
        } else {
            ggml_backend_tensor_get(sg_.logits, logits_buf.data(), 0,
                                    sizeof(float) * vocab);
            next_tok = 0;
            float best = logits_buf[0];
            for (int j = 1; j < vocab; j++) {
                if (logits_buf[j] > best) { best = logits_buf[j]; next_tok = j; }
            }
        }

        next_tok = apply_min_tokens_floor(
            next_tok, (int)out_tokens.size(), /*logits_row_offset=*/0);

        force_token(next_tok);
        apply_budget(next_tok);

        out_tokens.push_back(next_tok);
        io.emit(next_tok);
        committed++;
        cache_.cur_pos = committed;
        if (pool) {
            kvflash_history_.push_back(next_tok);
            if (kvflash_qk_policy_) kvflash_qk_pool_to(committed);
            kvflash_maybe_reselect(kvflash_ar_reselect_g(out_tokens.size(), reselect_from));
        }
        if (io.is_cancelled()) break;

        if (IS_EOS_TOK(next_tok, w_)) break;

        if (_repeat_guard > 0 && (int)out_tokens.size() >= _repeat_guard) {
            int run = 1;
            for (int j = (int)out_tokens.size() - 2; j >= 0; --j) {
                if (out_tokens[j] != next_tok) break;
                run++;
            }
            if (run >= _repeat_guard) {
                std::fprintf(stderr,
                    "[degenerate-decode] token %d repeated %d times - "
                    "breaking AR loop at committed=%d\n",
                    next_tok, run, committed);
                if (degenerate_close_out) *degenerate_close_out = true;
                break;
            }
        }

        // Degenerate-decode watchdog. Once we're past the budget-hook's
        // close sequence (model in post-`</think>` content phase), watch
        // for repetition loops. The aime2025-02 case at think_max=4k
        // produces a ~50-token phrase that repeats verbatim until
        // max_tokens — pure waste.
        //
        // Sweep several common loop periods. For each period P we check
        // if the last P tokens equal the previous P tokens (one full
        // repeat). One match is enough; the model has already burned 2P
        // tokens at that point and isn't getting out. The minimum-3
        // bar would catch tighter cycles but waits ~3P tokens to fire,
        // which is wasteful for P ≥ 32. Periods are tuned to common
        // failure modes: short loops (16-24) for "we have X, X, X"
        // patterns, longer (48-64) for full-sentence restates like the
        // aime02 case.
        if (budget.forced_close() && !budget.injecting())
        {
            // Sweep contiguous periods 12..80. Any P where the last P
            // tokens equal the previous P tokens means a loop of that
            // period. Stop early on first match. Fixed periods missed
            // the aime02 case which loops with period ~50; dense sweep
            // covers any period in this range.
            auto end = out_tokens.end();
            const int avail = (int)out_tokens.size();
            for (int P = 12; P <= 80; P++) {
                if (avail < 2 * P) break;  // larger P also won't have data
                if (std::equal(end - 2*P, end - P, end - P)) {
                    std::fprintf(stderr,
                        "[degenerate-decode] post-close period=%d repeated — "
                        "breaking AR loop at committed=%d, content_tokens=%zu\n",
                        P, committed,
                        out_tokens.size() - out_tokens_at_entry);
                    if (degenerate_close_out) *degenerate_close_out = true;
                    goto degenerate_break;
                }
            }
        }
        if (false) { degenerate_break: break; }
    }

    auto t_dec1_ar = std::chrono::steady_clock::now();
    const double ar_decode_s = std::chrono::duration<double>(t_dec1_ar - t_dec0_ar).count();
    const int ar_tokens = (int)(out_tokens.size() - out_tokens_at_entry);
    std::fprintf(stderr, "[ar-decode] tokens=%d time=%.3f s speed=%.2f tok/s\n",
                 ar_tokens, ar_decode_s,
                 ar_tokens > 0 && ar_decode_s > 0 ? ar_tokens / ar_decode_s : 0.0);
    return true;
}

bool Qwen35Backend::sync_remote_draft_features(int start_pos, int n_tokens) {
    if (!remote_draft_.active() || !cache_.target_feat || n_tokens <= 0) return true;
    if (cache_.target_feat_cap <= 0) return false;

    const int n_capture = w_.n_capture_layers;
    const int feat_hidden = w_.n_embd;
    const size_t src_stride = cache_.target_feat->nb[1];
    std::vector<float> slice((size_t)n_tokens * (size_t)feat_hidden);
    std::vector<uint16_t> bf16(feat_hidden);
    ggml_backend_synchronize(target_backend_);
    for (int cap_idx = 0; cap_idx < n_capture; ++cap_idx) {
        for (int t = 0; t < n_tokens; ++t) {
            const int slot = (start_pos + t) % cache_.target_feat_cap;
            const size_t src_offset = (size_t)slot * src_stride +
                (size_t)cap_idx * (size_t)feat_hidden * sizeof(uint16_t);
            ggml_backend_tensor_get(cache_.target_feat, bf16.data(),
                                    src_offset,
                                    sizeof(uint16_t) * (size_t)feat_hidden);
            float * dst = slice.data() + (size_t)t * feat_hidden;
            for (int h = 0; h < feat_hidden; ++h) {
                dst[h] = bf16_bits_to_f32(bf16[h]);
            }
        }
        if (!remote_draft_.send_feature_slice(cap_idx, start_pos, n_tokens, slice)) {
            std::fprintf(stderr,
                "spec-decode: remote feature sync failed capture=%d\n",
                cap_idx);
            return false;
        }
    }
    return true;
}

bool Qwen35Backend::sync_local_draft_features(int start_pos, int n_tokens) {
    if (n_tokens <= 0 || draft_parked_ || !feature_mirror_.target_feat ||
        !cache_.target_feat) {
        return true;
    }
    if (draft_feature_mirror_sync_range(
            cache_.target_feat, cache_.target_feat_cap,
            feature_mirror_, start_pos, n_tokens)) {
        return true;
    }
    std::fprintf(stderr,
        "spec-decode: local feature sync failed start=%d n=%d\n",
        start_pos, n_tokens);
    return false;
}

// ── DFlash speculative decode loop ─────────────────────────────────────

static bool qwen35_dspark_enabled() {
    static const bool kEnabled = []() {
        const char * e = std::getenv("LUCE_QWEN35_DSPARK");
        return e == nullptr || std::string(e) != "0";
    }();
    return kEnabled;
}

// Confidence-gate threshold for adaptive block length (0 = gate off, verify
// the full drafted block). The drafter's AcceptRatePredictor scores each
// draft position; the chain is truncated at the first position below the
// threshold and only the confident prefix is verified.
static float qwen35_dspark_confidence_threshold() {
    static const float kThreshold = []() {
        const char * e = std::getenv("LUCE_QWEN35_DSPARK_CONFIDENCE_THRESHOLD");
        if (!e) return 0.0f;
        float threshold = (float)std::atof(e);
        if (threshold < 0.0f) threshold = 0.0f;
        if (threshold > 1.0f) threshold = 1.0f;
        return threshold;
    }();
    return kThreshold;
}

// Resolve the target lm_head to a tensor whose data is readable on the
// draft backend. In tensor-parallel mode the lm_head is a meta tensor whose
// data pointer is a placeholder; the real data is mirrored (a full copy) on
// every rank. Return the simple tensor of the rank that matches the draft
// GPU (a local read on the draft backend), else nullptr so the caller falls
// back to the host chain.
static ggml_tensor * resolve_fused_lm_head(ggml_tensor * lm_head,
                                           const DevicePlacement & device,
                                           int draft_gpu) {
    if (!lm_head || !lm_head->buffer) return nullptr;
    const ggml_backend_buffer_type_t lm_head_buft =
        ggml_backend_buffer_get_type(lm_head->buffer);
    if (!ggml_backend_buft_is_meta(lm_head_buft)) {
        // A simple tensor is safe only when it is local to the draft backend,
        // or when peer access was successfully enabled for the actual target
        // and draft GPU pair. The configuration flag alone is not proof that
        // the hardware/driver accepted peer access.
        const ggml_backend_dev_t tensor_dev =
            ggml_backend_buft_get_device(lm_head_buft);
        const ggml_backend_dev_t draft_dev =
            ggml_backend_buft_get_device(
                ggml_backend_cuda_buffer_type(draft_gpu));
        if (tensor_dev != draft_dev) {
            const ggml_backend_dev_t target_dev =
                ggml_backend_buft_get_device(
                    ggml_backend_cuda_buffer_type(device.gpu));
            if (tensor_dev != target_dev ||
                !cross_device_peer_memcpy_ok(device.gpu, draft_gpu)) {
                return nullptr;
            }
        }
        return lm_head;
    }
    // Meta tensor: the lm_head is mirrored (full copy) on every rank. Read
    // it from the rank that matches the draft GPU so the fused graph does a
    // local read on the draft backend.
    for (size_t j = 0; j < device.layer_split_gpus.size(); ++j) {
        if (device.layer_split_gpus[j] == draft_gpu) {
            ggml_tensor * simple = ggml_backend_meta_simple_tensor(lm_head, j);
            if (!simple || !simple->data) return nullptr;
            GGML_ASSERT(ggml_are_same_shape(simple, lm_head));
            return simple;
        }
    }
    // Draft GPU is not a target rank; fall back to the host chain.
    return nullptr;
}

// Adaptive speculation policy: a spec step (draft + heads + width-q verify)
// costs about LUCE_QWEN35_SPEC_STEP_RATIO plain-decode steps, so it only
// pays off while the drafter gets more than (ratio - 1) of its tokens
// accepted per step. Below that (low-acceptance prose) the loop runs
// LUCE_QWEN35_AR_BURST plain-decode steps inside the spec loop (target
// forward on the seed token only, features still captured for the drafter),
// then probes with one spec step. Set LUCE_QWEN35_SPEC_STEP_RATIO=0 to
// disable the policy.
struct Qwen35AdaptiveSpecPolicy {
    float step_ratio = 1.9f;   // spec/plain step cost used until both step kinds have been timed
                               // (measured 54-55 vs 28.6 ms on gfx1201 for width-8 and width-16 verify)
    int   burst      = 40;     // plain-decode steps per burst (each burst ends with one spec probe step)
    float ema_alpha  = 0.1f;   // slow EMA: ~10-step memory so bursty acceptance does not flap
    bool  enabled() const { return step_ratio > 1.0f && burst > 0; }
    // Enter a burst only clearly below break-even (hysteresis against noise).
    // `ratio` is the live spec/plain step-time ratio once measured.
    float accept_threshold(float ratio) const { return 0.8f * (ratio - 1.0f); }
    float accept_threshold() const { return accept_threshold(step_ratio); }
};

static Qwen35AdaptiveSpecPolicy qwen35_adaptive_spec_policy() {
    static const Qwen35AdaptiveSpecPolicy kPolicy = []() {
        Qwen35AdaptiveSpecPolicy p;
        if (const char * e = std::getenv("LUCE_QWEN35_SPEC_STEP_RATIO")) {
            p.step_ratio = (float)std::atof(e);
        }
        if (const char * e = std::getenv("LUCE_QWEN35_AR_BURST")) {
            p.burst = std::atoi(e);
        }
        return p;
    }();
    return kPolicy;
}

bool Qwen35Backend::do_spec_decode(int committed, ThinkingBudget & budget,
                                    std::vector<int32_t> & out_tokens,
                                    const DaemonIO & io,
                                    float & out_accept_rate,
                                    bool & out_spec_ran,
                                    const std::vector<int32_t> * hint_tokens,
                                    const std::vector<int32_t> * stall_tool_prefix_tokens,
                                    const std::vector<int32_t> * stall_action_suffix_tokens,
                                    const std::vector<int32_t> * stall_skip_tokens,
                                    bool * forced_close_out,
                                    bool * degenerate_close_out) {
    out_accept_rate = 0.0f;
    out_spec_ran    = false;
    // The caller's controller owns the seed, every accepted block and AR tail.
    // [TAG_DRAFT_KV] the drafter ring persists across requests but its rows
    // belong to the previous conversation; start every request empty (the
    // first begin_step bulk-appends the live window from the feature mirror).
    if (draft_kv_.gf) draft_kv_reset(draft_kv_);
    const int hidden = w_.n_embd;

    // First token: use the argmax that do_prefill already sampled and stored.
    // Reading sg_.argmax_tokens with a computed offset is fragile: when
    // restore_and_generate calls do_prefill with kv_offset != 0, committed
    // reflects total KV position but the last chunk size was
    // delta.size() % ubatch, making (committed % 512) wrong and causing an
    // out-of-bounds tensor read.  cache_.last_tok is always correct.
    int32_t last_tok = cache_.last_tok;

    // Sampled-verify: spec decode with an active sampler. Each chain
    // position is verified against a token drawn from the target's own
    // sampler chain instead of its argmax, so every committed token is an
    // exact target sample — the output distribution is identical to AR
    // sampling. Acceptance drops vs greedy but stays far above the AR
    // floor. Opt in with LUCE_SAMPLED_VERIFY=1; without it, sampling
    // requests fall back to AR decode (zero behavior change by default).
    static const bool kSampledVerify = []() {
        const char * e = std::getenv("LUCE_SAMPLED_VERIFY");
        return e != nullptr && std::string(e) == "1";
    }();
    // Sampled-verify additionally requires full attention in the verify
    // path. With a finite --fa-window the verify batch applies one
    // window-start to the whole batch (unlike the AR step graph, which is
    // hardcoded to full attention): the argmax stays robust, so greedy
    // verification is unaffected, but the logit TAIL drifts at long
    // context and top-k sampling draws degenerate tokens from it
    // (reproduced at 24K: 0/12 tool calls with fa-window 2048, 4/4 with 0).
    const bool sampled_verify = kSampledVerify &&
        sampler_.needs_logit_processing() &&
        cfg_.fa_window == 0;

    // Check if we can use speculative decode:
    // - draft model loaded and not parked
    // - feature mirror initialized
    // - greedy decoding (no logit processing) — spec decode uses argmax verification
    // - sampled-verify enabled — allows spec decode with active sampling
    // - kvflash: verify_batch is slot-mapped (Qwen35DFlashTarget pooled
    //   path), and that covers --ddtree too: in the daemon, ddtree_mode
    //   configures larger verify intermediates + fast_rollback, whose
    //   snapshot_kv/restore_kv only touch DeltaNet/conv state (pool-
    //   neutral); generation runs this same chain loop either way. The
    //   tree-verify graphs exist only in the test harness (test_dflash).
    const bool can_spec = cfg_.draft_path
        && !draft_parked_
        && (cfg_.remote_draft.enabled()
            ? remote_draft_.active()
            : feature_mirror_.target_feat != nullptr)
        && (!sampler_.needs_logit_processing() || sampled_verify);

    if (!can_spec) {
        // AR consumes the prefill candidate and retains this controller.
        bool ok = do_ar_decode(committed, budget, out_tokens, io,
                                forced_close_out, degenerate_close_out);
        io.emit(-1);
        return ok;
    }

    out_spec_ran = true;

    // KVFlash history: every committed id (accepted drafts, bonus, replay and
    // injected tokens) lands in out_tokens in order; mirror them into
    // kvflash_history_ on every exit so live restore sees the answer.
    // Verify passes stage each row's query for q_cap (see reselect_before_step).
    struct QStageGuard {
        TargetCache & c;
        QStageGuard(TargetCache & cache, bool on) : c(cache) { c.q_stage_on = on && c.q_stage; c.q_stage_n = 0; c.q_stage_base = -1; }
        ~QStageGuard() { c.q_stage_on = false; c.q_stage_n = 0; c.q_stage_base = -1; }
    } q_stage_guard{cache_, kvflash_active() && kvflash_qk_policy_};
    int n_span_clamped = 0;
    struct HistoryGuard {
        Qwen35Backend * self; const std::vector<int32_t> & out;
        size_t base, entry;
        ~HistoryGuard() {
            if (!self->kvflash_active()) return;
            kvflash_history_set_generated(self->kvflash_history_, base, out, entry);
            const size_t hs = self->kvflash_history_.size();
            const size_t cp = (size_t)std::max(self->cache_.cur_pos, 0);
            if (hs < cp || hs > cp + 1)   // cur_pos, +1 for a pending token
                std::fprintf(stderr, "[kvflash] WARNING: history size %zu != cur_pos %zu "
                             "after spec decode\n", hs, cp);
        }
    } history_guard{this, out_tokens, kvflash_history_.size(), out_tokens.size()};

    // Sampled-verify: cache_.last_tok is do_prefill's argmax, and the spec
    // loop commits it verbatim as the first generated token. The first token
    // is the highest-entropy decision of the whole generation (e.g. "answer
    // with text" vs "open a tool call"), so it must be sampled like every
    // other committed token — mirror do_ar_decode's first-token sampling.
    if (sampled_verify && out_tokens.empty() && prefill_last_logits_valid_) {
        std::vector<float> first_logits(w_.n_vocab);
        ggml_backend_tensor_get(sg_.logits, first_logits.data(),
                                prefill_last_logits_offset_,
                                sizeof(float) * (size_t)w_.n_vocab);
        if (std::getenv("LUCE_SV_DEBUG")) {
            int am = 0; float best = first_logits[0];
            for (int v = 1; v < w_.n_vocab; v++)
                if (first_logits[v] > best) { best = first_logits[v]; am = v; }
            std::fprintf(stderr,
                "[sv-debug] first-token: logits_argmax=%d cache_last_tok=%d "
                "(match=%d) top_logit=%.3f\n",
                am, cache_.last_tok, am == cache_.last_tok, best);
        }
        last_tok = sample_logits(first_logits.data(), w_.n_vocab, sampler_,
                                 out_tokens, sampler_rng_);
        cache_.last_tok = last_tok;
    }

    const int _min_floor = dflash_min_tokens_floor();

    // ── DFlash spec-decode: draft → verify → accept → replay ──────────

    DFlashTarget * target = dflash_target();
    auto finish_speculative_state = [&]() {
        if (target->finish_speculative_state()) return true;
        std::fprintf(stderr, "spec-decode: final SpecLA state flush failed\n");
        return false;
    };
    const bool use_remote_draft = cfg_.remote_draft.enabled() && remote_draft_.active();
    // Long-context draft width. Verify attention runs through ggml's tile
    // kernel (the vector kernel only covers a single query row), and that
    // kernel's cost scales with the number of query rows AND with KV length.
    // So a wide draft block, which is a clear win on short prompts, turns
    // into a liability once the KV is large: the extra rows multiply a term
    // that is itself growing.
    //
    // Measured on one R9700 with Qwen3.8-27B IQ4_XS + DFlash2 (decode tok/s):
    //     prompt   verify 16   verify 8
    //      6,208         63.2           71.1
    //     13,148         51.9           54.9
    //     26,728         43.4           46.9
    //     39,861         32.7           45.5
    // Narrowing inside a block-16 drafter also holds acceptance better than
    // configuring the drafter at block 8 outright (0.414 vs 0.375 at 27K).
    // Below the crossover the wide block still wins by a lot (HumanEval-10
    // 204 vs 143.6 decode), so narrow only past it. Halving also raises
    // commit depth there (0.462*8+1 = 4.70 vs 0.200*16+1 = 4.20): fewer rows
    // and deeper commits at the same time.
    //
    // The rule is a function of context length only. It deliberately does not
    // look at the observed acceptance rate, so the width for a given prompt
    // is reproducible and does not chase a moving target.
    // 8192 is deliberately above every prompt in the short-context benchmarks
    // this engine is tuned for, so high-acceptance completion workloads, where
    // the wide block earns its keep, are untouched.
    constexpr int kLongCtxNarrowTokens = 8192;
    // Narrowing floor, chosen to match the shipped DFlash2 checkpoint's
    // published block of 8 (clamped to q_len below for narrower checkpoints).
    constexpr int kLongCtxMinVerify = 8;
    const int q_len = dw_.block_size > 0 ? dw_.block_size : LUCE_DRAFT_BLOCK_SIZE;
    // This caps the VERIFY batch only; the drafter still proposes a full q_len
    // block. Narrowing the draft instead would leave the tail rows of the
    // drafter's non-causal block unwritten while its mask still makes them
    // visible to the rows we keep, and would break the fixed block_size
    // contract the IPC drafter validates against.
    // Clamped to q_len: a checkpoint whose published block is below the
    // narrowing floor never widens past its own block. Acceptance accounting
    // uses the actual offered width, including confidence-trimmed drafts.
    const int verify_cap = committed >= kLongCtxNarrowTokens
                               ? std::min(q_len, std::max(kLongCtxMinVerify, q_len / 2))
                               : q_len;
    const bool shared_feedback_width =
        !cfg_.ddtree_mode && adaptive_spec_width_globally_enabled();
    AdaptiveSpecWidth width_controller(q_len, 2, shared_feedback_width);
    if (verify_cap != q_len) {
        static std::atomic<bool> s_narrowed_logged{false};
        if (!s_narrowed_logged.exchange(true)) {
            std::fprintf(stderr,
                "[qwen35-spec] context %d >= %d: capping verify width %d -> %d "
                "(wide blocks lose to verify-attention cost at long context)\n",
                committed, kLongCtxNarrowTokens, q_len, verify_cap);
        }
    }
    const int max_verify_tokens = cfg_.ddtree_mode
        ? std::max<int>(dw_.block_size, cfg_.ddtree_budget + 1)
        : dw_.block_size;
    if ((cfg_.fast_rollback || cfg_.ddtree_mode) && !cache_.rollback_ctx) {
        const bool enable_specla = cfg_.specla_mode && cfg_.fast_rollback &&
            !cfg_.device.is_tensor_parallel() && !kvflash_active();
        if (!migrate_prefill_cache(w_, cfg_.device.max_ctx,
                                   max_verify_tokens,
                                   target_backend_, cache_, enable_specla)) {
            std::fprintf(stderr, "spec-decode: rollback cache migration failed: %s\n",
                         luce_last_error());
            return false;
        }
    }

    StepGraph draft_sg;

    std::vector<float>   noise_embed((size_t)hidden * q_len);
    std::vector<int32_t> noise_ids(q_len);
    std::vector<int32_t> draft_tok(q_len);
    std::vector<int32_t> target_tok(q_len);
    std::vector<float>   verify_logits;   // sampled-verify: [q_len x vocab]
    std::vector<int32_t> verify_history;  // sampled-verify: penalty history
    std::vector<int32_t> pos_q(q_len);
    std::vector<int32_t> pos_k;
    std::vector<float>   local_hidden;

    int n_generated     = 0;
    int n_draft_steps   = 0;
    SpecAcceptanceStats acceptance;
    int n_hint_proposed = 0;
    int n_hint_accepted = 0;
    int target_forwards = 0;
    const ChainRollbackPolicy rollback_policy =
        resolve_chain_rollback_policy(cfg_.device.is_tensor_parallel(),
                                      target->exact_fast_rollback());
    const int fast_rollback_threshold =
        rollback_policy.fast_rollback_threshold;
    RollbackDiag rollback_diag;

    const char * tp_profile_env = std::getenv("LUCE_TP_PROFILE");
    const bool tp_profile = tp_profile_env && std::strcmp(tp_profile_env, "0") != 0;
    double profile_draft_s = 0.0;
    double profile_project_s = 0.0;
    double profile_snapshot_s = 0.0;
    double profile_verify_s = 0.0;
    double profile_rollback_s = 0.0;
    double profile_replay_s = 0.0;
    double profile_feature_s = 0.0;
    auto profile_start = [&]() {
        return tp_profile
            ? std::chrono::steady_clock::now()
            : std::chrono::steady_clock::time_point{};
    };
    auto profile_add = [&](double & total,
                           std::chrono::steady_clock::time_point start) {
        if (tp_profile) {
            total += std::chrono::duration<double>(
                std::chrono::steady_clock::now() - start).count();
        }
    };

    auto log_target_forward_stats = [&]() {
        std::fprintf(stderr, "[spec-decode] target_forwards=%d forwards_per_token=%.6f forwards_per_step=%.3f\n",
                     target_forwards,
                     n_generated > 0 ? (double)target_forwards / n_generated : 0.0,
                     n_draft_steps > 0 ? (double)target_forwards / n_draft_steps : 0.0);
        rollback_diag.print(rollback_policy, stderr);
        if (kvflash_active())
            std::fprintf(stderr, "[kvflash] spec verify span clamps=%d (reselect point or unallocated chunk)\n",
                         n_span_clamped);
    };

    // kvflash: an in-pool prompt prefills contiguously without registering
    // chunks in the pager. Map the prefix now so the pager stays consistent
    // with the tree path's direct writes and hands off cleanly to slot-mapped
    // paging once the context outgrows the pool. (A >pool prompt already paged
    // during prefill, leaving the pager non-identity, so this is skipped.)
    if (kvflash_active() && kvflash_pager_.is_identity()) {
        (void)kvflash_pager_.alloc_span(0, committed);
    }

    auto t_dec0 = std::chrono::steady_clock::now();

    // Adaptive speculation state (see Qwen35AdaptiveSpecPolicy).
    const Qwen35AdaptiveSpecPolicy adaptive = qwen35_adaptive_spec_policy();
    // Start well above the burst threshold so an unlucky opening does not
    // park a predictable stream in plain decode. The probe step that ends a
    // burst updates the EMA with a fast alpha (see below) so a stream that
    // turned predictable leaves plain decode quickly.
    float accepted_ema = 2.0f * adaptive.accept_threshold();
    int   ar_burst_left = 0;
    int   n_ar_burst_steps = 0;
    bool  probe_step = false;   // first spec step after a burst
    // Live step-time EMAs (seconds) for the break-even ratio; 0 = not yet measured.
    double t_spec_step_ema = 0.0;
    double t_ar_step_ema   = 0.0;
    auto live_step_ratio = [&]() {
        return (t_spec_step_ema > 0.0 && t_ar_step_ema > 0.0)
            ? (float)(t_spec_step_ema / t_ar_step_ema) : adaptive.step_ratio;
    };

    // QK pooling has one call site per settled position: the top of every
    // iteration (all commit paths, chain and tree, end up there) and every exit.
    // It must never run inside a step: a forced budget close moves the position
    // back after the emit loop, and a chunk pooled from the unrepaired block
    // would keep the replaced token's key.
    auto pool_settled = [&]() {
        if (!kvflash_active()) return;
        if (kvflash_qk_policy_) kvflash_qk_pool_to(committed);
    };
    // Decode-time reselection: AR reselects right after it emits generated
    // token g (g % tau == 0), before it forwards that token. At a step start
    // the seed (last_tok) is token g0 + 1, pending at `committed`, so the
    // reselect for it runs here, from q_cap = the query of the last consumed
    // row (committed - 1, taken from the verify stage). The verify span is
    // clamped (kvflash_verify_span_cap) so no later point falls inside a step:
    // every reselect runs at a step start, with the resident set and the query
    // AR has there.
    int reselect_done_g = 0;
    auto reselect_before_step = [&]() {
        if (!kvflash_active()) return;
        const int g_seed = (int)(out_tokens.size() - history_guard.entry) + 1;
        const std::vector<int> pts =
            kvflash_reselect_points(history_guard.base, reselect_done_g, g_seed, kvflash_tau_);
        reselect_done_g = std::max(reselect_done_g, g_seed);
        if (pts.empty()) return;
        if (kvflash_qk_policy_) kvflash_q_cap_from_stage(committed);
        for (int g : pts) {
            kvflash_reselect_history(kvflash_history_, history_guard.base, out_tokens,
                                     history_guard.entry, g, g_seed, budget, last_tok);
            kvflash_maybe_reselect(g);
        }
    };

    // Every handoff to AR from the spec loop: AR continues the same request, so
    // it gets the history with every emitted token and the request's output
    // origin (its reselect boundaries are global, not counted from the handoff).
    auto ar_tail = [&]() -> bool {
        if (kvflash_active())
            kvflash_history_set_generated(kvflash_history_, history_guard.base, out_tokens,
                                          history_guard.entry);
        return do_ar_decode(committed, budget, out_tokens, io, forced_close_out,
                            degenerate_close_out, history_guard.entry);
    };
    while (budget.remaining() > 0) {
        pool_settled();
        reselect_before_step();
        const int need_commit_budget = budget.remaining();
        // Plain-decode step inside the spec loop: no drafter forward, verify
        // the seed token only. Features are still captured, so the drafter
        // resumes cleanly on the next probe step.
        bool ar_step = adaptive.enabled() && ar_burst_left > 0;
        // Pending tool-call hints verify at near-100% acceptance; a plain
        // decode burst would ignore them for up to `burst` tokens, so end
        // the burst as soon as hints are available.
        if (ar_step && hint_tokens && n_generated < (int)hint_tokens->size()) {
            ar_burst_left = 0;
            ar_step = false;
        }
        if (ar_step) {
            ar_burst_left--;
            n_ar_burst_steps++;
            probe_step = (ar_burst_left == 0);
        }
        const auto t_step_start = std::chrono::steady_clock::now();

        // Budget hook: no tail-off here. The close-token injection fires
        // during the emit phase (step 8) after acceptance+replay, mirroring
        // the existing floor_to_ar pattern. This keeps spec-decode active
        // through the close boundary instead of tearing down to AR early.

        if (last_tok < 0 && !out_tokens.empty()) {
            std::fprintf(stderr,
                "[spec-decode] invalid draft seed %d after %d emitted tokens; "
                "switching to AR\n",
                last_tok, (int)out_tokens.size());
            step_graph_destroy(draft_sg);
            cache_.last_tok = out_tokens.back();
            if (!finish_speculative_state()) return false;
            bool ok = ar_tail();
            log_target_forward_stats();
            io.emit(-1);
            return ok;
        }

        // KVFlash: a spec step never starts on a seed that opens a chunk the
        // pool has not allocated. That seed takes the AR single-token step:
        // emit, then forward, which allocates the chunk exactly as AR does (a
        // request that ends on the emission never allocates it; restore_kv()
        // could not undo a verify's allocation). The next spec step starts
        // after it. (LUCE_MIN_TOKENS with KVFlash is refused at startup, so no
        // floor case reaches here.)
        if (kvflash_active() && last_tok >= 0 &&
            kvflash_seed_takes_ar_step(
                committed, kvflash_pager_.chunk_tokens(),
                [&](int c) { return kvflash_pager_.is_resident(c); })) {
            // AR's state when the request ends or closes here: the last emitted
            // token is pending, not forwarded.
            auto hand_to_ar = [&](bool ended) -> bool {
                cache_.cur_pos = committed;
                cache_.last_tok = out_tokens.back();
                pool_settled();
                step_graph_destroy(draft_sg);
                out_accept_rate = acceptance.rate();
                if (!finish_speculative_state()) return false;
                if (ended) {
                    log_target_forward_stats();
                    io.emit(-1);
                    return true;
                }
                bool ok = ar_tail();
                log_target_forward_stats();
                io.emit(-1);
                return ok;
            };
            const int remaining = budget.remaining();
            const auto decision = budget.apply(last_tok);
            if (decision.cut) {
                std::fprintf(stderr,
                    "[budget-hook] spec-decode force-close at remaining=%d: "
                    "overriding token %d with close token %d\n",
                    remaining, last_tok, decision.token);
                if (forced_close_out) *forced_close_out = true;
            }
            out_tokens.push_back(decision.token);
            io.emit(decision.token);
            const bool ended = budget.remaining() <= 0 || io.is_cancelled() ||
                               IS_EOS_TOK(decision.token, w_);
            if (ended || decision.cut) return hand_to_ar(ended);
            // Forward the emitted token (features captured for the drafter).
            const std::vector<int32_t> hop_tok(1, decision.token);
            int hop_last_tok = -1;
            if (!target->verify_batch(hop_tok, committed, hop_last_tok, &target_tok,
                                      /*capture_ssm_intermediates=*/true)) {
                if (target->verify_pool_exhausted()) {
                    std::fprintf(stderr,
                        "[kvflash] spec AR step: pool exhausted at committed=%d, "
                        "finishing the request with AR\n", committed);
                    return hand_to_ar(false);
                }
                std::fprintf(stderr, "spec-decode: AR step forward failed\n");
                step_graph_destroy(draft_sg);
                return false;
            }
            target_forwards++;
            if (use_remote_draft && cache_.target_feat) {
                if (!sync_remote_draft_features(committed, 1)) {
                    step_graph_destroy(draft_sg);
                    return false;
                }
            } else if (feature_mirror_.target_feat && cache_.target_feat) {
                if (!sync_local_draft_features(committed, 1)) {
                    step_graph_destroy(draft_sg);
                    return false;
                }
            }
            if (sampled_verify && target->read_verify_logits(1, verify_logits)) {
                last_tok = sample_logits(verify_logits.data(), (int)verify_logits.size(),
                                         sampler_, out_tokens, sampler_rng_);
            } else {
                last_tok = target_tok[0];
            }
            committed += 1;
            cache_.cur_pos = committed;
            n_generated += 1;
            continue;
        }
        // end of the AR hop

        // 1. Build noise input for draft. The drafter GGUF's own MASK id
        // wins over the target-side default (the loader resolved the
        // precedence into dw_); remote drafters keep the target default
        // because dw_ is not populated for them.
        const int32_t noise_mask_id =
            (!use_remote_draft && dw_.mask_token_id >= 0)
                ? dw_.mask_token_id : target->mask_token_id();
        noise_ids[0] = last_tok;
        for (int i = 1; i < q_len; i++) noise_ids[i] = noise_mask_id;
        if (!target->embed_tokens(noise_ids.data(), q_len, noise_embed.data())) {
            std::fprintf(stderr, "spec-decode: noise embed failed (last_tok=%d mask=%d q_len=%d)\n",
                         last_tok, target->mask_token_id(), q_len);
            step_graph_destroy(draft_sg);
            return false;
        }

        // 2. Draft compute (skipped on plain-decode burst steps)
        constexpr int DRAFT_CTX_MAX_DEFAULT = 2048;
        bool used_draft_kv = false;
        if (!ar_step) {
            const int ring_cap = use_remote_draft ? remote_draft_.ring_cap() : feature_mirror_.cap;
            const int draft_ctx = std::min(committed,
                std::min(ring_cap, std::max(DRAFT_CTX_MAX_DEFAULT, cfg_.draft_ctx_max)));
            const int draft_start = committed - draft_ctx;
            int mirror_slot0 = 0;
            const bool use_mirror_view =
                !use_remote_draft &&
                draft_feature_mirror_can_view(feature_mirror_, committed, draft_ctx, mirror_slot0);

            const auto profile_draft_start = profile_start();
            if (use_remote_draft) {
                local_hidden.clear();
                if (!remote_draft_.propose(committed, draft_ctx, noise_embed, local_hidden)) {
                    std::fprintf(stderr, "spec-decode: remote draft propose failed\n");
                    step_graph_destroy(draft_sg);
                    return false;
                }
            } else {
                // [TAG_DRAFT_KV] ring-cached drafter context KV: append newly
                // committed rows instead of re-encoding the whole feature window.
                static const bool draft_kv_on = []() {
                    const char * e = std::getenv("LUCE_DRAFT_KV");
                    return !(e && e[0] == '0' && e[1] == '\0');
                }();
                bool use_draft_kv = draft_kv_on && feature_mirror_.target_feat != nullptr;
                if (use_draft_kv && draft_kv_.gf &&
                    draft_kv_.built_for != (const void *)&dw_) {
                    draft_kv_free(draft_kv_);
                }
                if (use_draft_kv && !draft_kv_.gf) {
                    const int kv_cap = std::min(ring_cap,
                        std::max(DRAFT_CTX_MAX_DEFAULT, cfg_.draft_ctx_max));
                    if (!draft_kv_init(draft_kv_, dw_, draft_backend_, kv_cap, nullptr)) {
                        draft_kv_free(draft_kv_);
                        use_draft_kv = false;
                        std::fprintf(stderr,
                            "spec-decode: draft-kv init failed; using legacy draft path\n");
                    }
                }
                if (use_draft_kv) {
                    used_draft_kv = true;
                    if (!draft_kv_begin_step(draft_kv_, dw_, draft_backend_,
                                             feature_mirror_, committed)) {
                        std::fprintf(stderr, "spec-decode: draft-kv step prep failed\n");
                        step_graph_destroy(draft_sg);
                        return false;
                    }
                    ggml_backend_tensor_set(draft_kv_.inp_embed, noise_embed.data(), 0,
                                            sizeof(float) * noise_embed.size());
                    if (ggml_backend_graph_compute(draft_backend_, draft_kv_.gf) !=
                        GGML_STATUS_SUCCESS) {
                        std::fprintf(stderr, "spec-decode: draft-kv compute failed\n");
                        step_graph_destroy(draft_sg);
                        return false;
                    }
                    local_hidden.resize((size_t)hidden * q_len);
                    ggml_backend_tensor_get(draft_kv_.hidden_states, local_hidden.data(), 0,
                                            sizeof(float) * local_hidden.size());
                } else {
                    if (!build_draft_step(draft_sg, dw_, /*lm_head=*/nullptr, draft_backend_,
                                          draft_ctx, use_mirror_view ? &feature_mirror_ : nullptr,
                                          committed,
                                          /*ctx_len_max=*/std::min(ring_cap, std::max(DRAFT_CTX_MAX_DEFAULT, cfg_.draft_ctx_max)))) {
                        std::fprintf(stderr, "spec-decode: draft build failed\n");
                        step_graph_destroy(draft_sg);
                        return false;
                    }
                    if (!use_mirror_view &&
                        !copy_feature_ring_range_to_tensor(feature_mirror_, draft_sg.target_hidden_cat,
                                                           draft_start, draft_ctx)) {
                        std::fprintf(stderr, "spec-decode: feature copy failed\n");
                        step_graph_destroy(draft_sg);
                        return false;
                    }
                    ggml_backend_tensor_set(draft_sg.inp_embed, noise_embed.data(), 0,
                                            sizeof(float) * noise_embed.size());
                    pos_k.resize((size_t)draft_ctx + q_len);
                    for (int i = 0; i < q_len; i++) pos_q[i] = draft_ctx + i;
                    for (int i = 0; i < draft_ctx + q_len; i++) pos_k[i] = i;
                    ggml_backend_tensor_set(draft_sg.positions, pos_q.data(), 0,
                                            sizeof(int32_t) * pos_q.size());
                    ggml_backend_tensor_set(draft_sg.positions_k, pos_k.data(), 0,
                                            sizeof(int32_t) * pos_k.size());

                    auto st = ggml_backend_graph_compute(draft_backend_, draft_sg.gf);
                    if (st != GGML_STATUS_SUCCESS) {
                        std::fprintf(stderr, "spec-decode: draft compute failed\n");
                        step_graph_destroy(draft_sg);
                        return false;
                    }

                    // Read draft hidden states to host for LM-head projection.
                    local_hidden.resize((size_t)hidden * q_len);
                    ggml_backend_tensor_get(draft_sg.hidden_states, local_hidden.data(), 0,
                                            sizeof(float) * local_hidden.size());
                }
            }
            profile_add(profile_draft_s, profile_draft_start);
        }  // !ar_step

        // ── DDTree tree-structured verify ────────────────────────────────
        // When --ddtree is on and the target supports tree verify, build a
        // draft tree from per-position top-K, verify all nodes in one
        // ancestor-masked target forward, walk the verified path, then roll
        // recurrent/KV state forward to the accepted path. Higher acceptance
        // per step than chain verify. Local draft only. On any failure we
        // fall through is unsafe (draft graph already built for chain), so we
        // bail to false.
        // Until the controller settles, use chain verify so every marker token
        // is observed and a forced close can restore/replay its accepted prefix.
        // Tool hints, stall recovery and the min-tokens floor also require chain.
        const bool tree_special_inactive =
            budget.settled() &&
            !(hint_tokens && n_generated < (int)hint_tokens->size()) &&
            stall_tool_prefix_tokens == nullptr &&
            (int)out_tokens.size() >= _min_floor;
        // kvflash: only take the non-paged tree path while the read prefix
        // [0, committed) is identity-resident and the write span (+FA padding)
        // fits the resident pool. Full-attention only (windowed FA padding could
        // index past the pool). Otherwise the slot-mapped chain verify handles
        // it. Non-kvflash is unaffected.
        const bool kvflash_tree_ok =
            !kvflash_active() ||
            (cfg_.fa_window == 0 &&
             committed + cfg_.ddtree_budget + 1 + cfg_.kq_stride_pad <= kvflash_tokens_ &&
             kvflash_pager_.identity_prefix_covers(committed));
        const bool use_tree_verify =
            cfg_.ddtree_mode && target->supports_tree_verify() && kvflash_tree_ok &&
            !use_remote_draft && q_len > 1 && tree_special_inactive && !ar_step;

        // Chain-verify length for this step. The DSpark confidence gate may
        // truncate the drafted block (adaptive block length); q_len stays the
        // buffer-sizing upper bound.
        int v_len = q_len;
        // DDTree consumes top-K rows directly. Avoid projecting the same
        // hidden block once for argmax and again for top-K on every step.
        if (ar_step) {
            draft_tok.assign(1, last_tok);
            v_len = 1;
        } else if (!use_tree_verify) {
            const auto profile_project_start = profile_start();
            // DFlash 2 selector (top-k candidates + low-rank path score) when
            // the drafter ships it.
            bool used_dspark = false;
            if (dw_.selector.enabled && q_len > 1 && !sampled_verify && !use_remote_draft) {
                static std::atomic<bool> s_sel_logged{false};
                if (!s_sel_logged.exchange(true)) {
                    std::fprintf(stderr,
                        "[qwen35-spec] DFlash 2 selector active for greedy chain decode "
                        "(rank=%d top_k=%d)\n", dw_.selector.rank, dw_.selector.top_k);
                }
                if (dflash2_select_chain(dw_, draft_backend_, *target,
                                         local_hidden.data(), q_len, last_tok, draft_tok)) {
                    used_dspark = true;
                    v_len = std::max(1, (int)draft_tok.size());
                } else {
                    static std::atomic<bool> s_sel_warned{false};
                    if (!s_sel_warned.exchange(true)) {
                        std::fprintf(stderr,
                            "[qwen35-spec] DFlash 2 selector failed; falling back to "
                            "base DFlash projection\n");
                    }
                }
            }
            // DSpark heads (markov bigram correction + optional confidence
            // gate) when the drafter ships them; mirrors the laguna hook.
            if (!used_dspark && qwen35_dspark_enabled() && dw_.dspark.enabled &&
                q_len > 1 && !sampled_verify && !use_remote_draft) {
                static std::atomic<bool> s_dspark_logged{false};
                if (!s_dspark_logged.exchange(true)) {
                    std::fprintf(stderr,
                        "[qwen35-spec] DSpark Markov head active for greedy chain decode "
                        "(rank=%d vocab=%d confidence_dim=%d)\n",
                        dw_.dspark.markov_rank, dw_.dspark.vocab_size,
                        dw_.dspark.confidence_dim);
                }
                static const bool fused_dspark = []() {
                    const char * e = std::getenv("LUCE_QWEN35_FUSED_DSPARK");
                    return !(e && e[0] == '0' && e[1] == '\0');
                }();
                bool ds_ok = false;
                const float conf_threshold = qwen35_dspark_confidence_threshold();
                if (fused_dspark) {
                    // One graph for every candidate: markov-corrected tokens
                    // plus (when gated) the confidence score per position.
                    // The lm_head must be readable on the draft backend; in
                    // tensor-parallel mode that is the simple tensor of the
                    // rank matching the draft GPU (lm_head is mirrored on
                    // every rank).
                    std::vector<float> conf_scores;
                    ggml_tensor * fused_lm_head =
                        resolve_fused_lm_head(target->lm_head_tensor(),
                                              cfg_.device, cfg_.draft_gpu);
                    if (fused_lm_head) {
                        ds_ok = dspark_markov_correct_greedy_chain_fused(
                            dw_, draft_backend_, fused_lm_head,
                            local_hidden.data(), q_len, last_tok, draft_tok,
                            conf_threshold > 0.0f ? &conf_scores : nullptr);
                    }
                    if (ds_ok && conf_threshold > 0.0f) {
                        // Truncate the chain at the first low-confidence
                        // position: draft_tok[0] is the seed, candidate i
                        // scores conf_scores[i-1].
                        size_t keep = 1;
                        while (keep < draft_tok.size() &&
                               keep - 1 < conf_scores.size() &&
                               conf_scores[keep - 1] >= conf_threshold) {
                            ++keep;
                        }
                        static const bool conf_debug = []() {
                            const char * e = std::getenv("LUCE_QWEN35_DSPARK_CONF_DEBUG");
                            return e && e[0] == '1';
                        }();
                        if (conf_debug) {
                            std::fprintf(stderr, "[dspark-conf] keep=%zu/%zu:", keep, draft_tok.size());
                            for (float c : conf_scores) std::fprintf(stderr, " %.3f", c);
                            std::fprintf(stderr, "\n");
                        }
                        draft_tok.resize(keep);
                    }
                }
                if (!ds_ok) {
                    ds_ok = dspark_markov_correct_greedy_chain(dw_, draft_backend_, *target,
                                                       local_hidden.data(), q_len,
                                                       last_tok, conf_threshold,
                                                       draft_tok);
                }
                if (ds_ok) {
                    used_dspark = true;
                    // Confidence gate truncates the drafted chain: verify
                    // only the confident prefix this step.
                    v_len = std::max(1, (int)draft_tok.size());
                } else {
                    static std::atomic<bool> s_dspark_warned{false};
                    if (!s_dspark_warned.exchange(true)) {
                        std::fprintf(stderr,
                            "[qwen35-spec] DSpark Markov head failed; falling back to "
                            "base DFlash projection\n");
                    }
                }
            }
            if (!used_dspark) {
                if (!target->project_hidden_to_tokens(local_hidden.data(), q_len, draft_tok)) {
                    std::fprintf(stderr, "spec-decode: projection failed\n");
                    step_graph_destroy(draft_sg);
                    return false;
                }
                draft_tok[0] = last_tok;
            }
            profile_add(profile_project_s, profile_project_start);
        }

        if (use_tree_verify) {
            const int L = q_len - 1;
            // The paper's end-to-end tree route uses top-k=4.  Wider top-k
            // spends projection and scheduling work on low-ranked siblings;
            // keep the legacy width only outside SpecLA.
            const int K = (cfg_.ddtree_budget > L)
                ? (target->exact_fast_rollback()
                    ? std::min(cfg_.specla_top_k, w_.n_vocab)
                    : 8)
                : 1;
            DDTree tree;
            if (target->exact_fast_rollback() && K > 1 &&
                specla_conditional_draft_enabled()) {
                // SpecLA §6.2: every expanded branch is proposed from that
                // branch's exact token prefix, avoiding the old one-spine
                // approximation. This is experimental for the current
                // five-layer drafter because every prefix requires a rerun.
                bool conditional_ok = true;
                std::vector<float> row_hidden((size_t)hidden);
                DDTreeConditionalTopK conditional_topk =
                    [&](const std::vector<int32_t> & prefix, int next_depth,
                        std::vector<float> & top_lp,
                        std::vector<int32_t> & top_ids) -> bool {
                    if (!conditional_ok || next_depth < 1 || next_depth > L ||
                        (int)prefix.size() != next_depth - 1) {
                        conditional_ok = false;
                        return false;
                    }

                    if (prefix.empty()) {
                        std::memcpy(row_hidden.data(),
                                    local_hidden.data() + (size_t)hidden,
                                    sizeof(float) * (size_t)hidden);
                    } else {
                        noise_ids[0] = last_tok;
                        for (int i = 1; i < q_len; ++i) {
                            noise_ids[(size_t)i] = i <= (int)prefix.size()
                                ? prefix[(size_t)i - 1]
                                : noise_mask_id;
                        }
                        if (!target->embed_tokens(noise_ids.data(), q_len,
                                                  noise_embed.data())) {
                            conditional_ok = false;
                            return false;
                        }
                        const auto branch_draft_start = profile_start();
                        ggml_tensor * branch_input = used_draft_kv
                            ? draft_kv_.inp_embed : draft_sg.inp_embed;
                        ggml_cgraph * branch_graph = used_draft_kv
                            ? draft_kv_.gf : draft_sg.gf;
                        ggml_tensor * branch_hidden = used_draft_kv
                            ? draft_kv_.hidden_states : draft_sg.hidden_states;
                        if (!branch_input || !branch_graph || !branch_hidden) {
                            conditional_ok = false;
                            return false;
                        }
                        ggml_backend_tensor_set(branch_input, noise_embed.data(), 0,
                                                sizeof(float) * noise_embed.size());
                        if (ggml_backend_graph_compute(draft_backend_, branch_graph) !=
                            GGML_STATUS_SUCCESS) {
                            conditional_ok = false;
                            return false;
                        }
                        ggml_backend_tensor_get(
                            branch_hidden, row_hidden.data(),
                            (size_t)next_depth * hidden * sizeof(float),
                            sizeof(float) * (size_t)hidden);
                        profile_add(profile_draft_s, branch_draft_start);
                    }

                    const auto branch_project_start = profile_start();
                    const bool ok = target->project_hidden_to_topk(
                        row_hidden.data(), 1, K, cfg_.ddtree_temp,
                        top_lp, top_ids);
                    profile_add(profile_project_s, branch_project_start);
                    conditional_ok = conditional_ok && ok;
                    return ok;
                };
                tree = build_ddtree_conditional(
                    conditional_topk, L, K, cfg_.ddtree_budget,
                    cfg_.ddtree_chain_seed, cfg_.ddtree_tau);
                if (!conditional_ok || tree.n_nodes == 0) {
                    std::fprintf(stderr,
                        "spec-decode: conditioned ddtree proposal failed\n");
                    step_graph_destroy(draft_sg);
                    return false;
                }
            } else {
                // DFlash2 selector-scored tree (DARTree-style): branch scores
                // are logp + selector compatibility with the branch's actual
                // parent, so the tree at worst degenerates to the selector
                // chain. LUCE_QWEN35_DFLASH2_TREE=0 falls back to raw top-k.
                static const bool dflash2_tree = []() {
                    const char * e = std::getenv("LUCE_QWEN35_DFLASH2_TREE");
                    return !(e && e[0] == '0' && e[1] == '\0');
                }();
                bool selector_tree_ok = false;
                if (dflash2_tree && dw_.selector.enabled && !use_remote_draft) {
                    Dflash2TreeScores sc;
                    const auto profile_project_start = profile_start();
                    if (dflash2_score_candidates(dw_, draft_backend_, *target,
                                                 local_hidden.data(), q_len, last_tok,
                                                 cfg_.ddtree_temp, sc)) {
                        static std::atomic<bool> s_seltree_logged{false};
                        if (!s_seltree_logged.exchange(true)) {
                            std::fprintf(stderr,
                                "[qwen35-spec] DFlash2 selector scores active for DDTree candidates\n");
                        }
                        DDTreeConditionalTopK selector_topk =
                            [&](const std::vector<int32_t> & prefix, int next_depth,
                                std::vector<float> & lp, std::vector<int32_t> & ids) -> bool {
                            if (!sc.topk(prefix, next_depth, lp, ids)) return false;
                            if ((int)lp.size() > K) { lp.resize((size_t)K); ids.resize((size_t)K); }
                            return true;
                        };
                        tree = build_ddtree_conditional(
                            selector_topk, L, K, cfg_.ddtree_budget,
                            cfg_.ddtree_chain_seed, cfg_.ddtree_tau);
                        selector_tree_ok = tree.n_nodes > 0;
                    }
                    profile_add(profile_project_s, profile_project_start);
                }
                if (!selector_tree_ok) {
                std::vector<float>   top_lp;
                std::vector<int32_t> top_ids;
                const auto profile_project_start = profile_start();
                static const bool dspark_tree = []() {
                    const char * e = std::getenv("LUCE_QWEN35_DSPARK_TREE");
                    return !(e && e[0] == '0' && e[1] == '\0');
                }();
                bool topk_ok = false;
                if (dspark_tree && qwen35_dspark_enabled() && dw_.dspark.enabled) {
                    static std::atomic<bool> s_dstree_logged{false};
                    if (!s_dstree_logged.exchange(true)) {
                        std::fprintf(stderr,
                            "[qwen35-spec] DSpark Markov head active for DDTree candidates\n");
                    }
                    ggml_tensor * fused_lm_head =
                        resolve_fused_lm_head(target->lm_head_tensor(),
                                              cfg_.device, cfg_.draft_gpu);
                    if (fused_lm_head) {
                        topk_ok = dspark_markov_project_topk(dw_, draft_backend_,
                                                             fused_lm_head,
                                                             local_hidden.data(), q_len, K,
                                                             cfg_.ddtree_temp, last_tok,
                                                             top_lp, top_ids);
                    }
                }
                if (!topk_ok &&
                    !target->project_hidden_to_topk(local_hidden.data(), q_len, K,
                                                    cfg_.ddtree_temp,
                                                    top_lp, top_ids)) {
                    std::fprintf(stderr,
                        "spec-decode: ddtree topk projection failed\n");
                    step_graph_destroy(draft_sg);
                    return false;
                }
                profile_add(profile_project_s, profile_project_start);
                tree = build_ddtree(
                    top_lp.data() + (size_t)K,
                    top_ids.data() + (size_t)K,
                    L, K, cfg_.ddtree_budget, cfg_.ddtree_chain_seed,
                    cfg_.ddtree_tau);
                }
            }
            // SpecLA schedules the retained topology directly. Never execute
            // fake padding nodes: confidence pruning must reduce target work,
            // and this loop already rebuilds the topology-dependent graph.
            const int N = target->exact_fast_rollback()
                ? 1 + tree.n_nodes
                : (std::isfinite(cfg_.ddtree_tau)
                    ? 1 + tree.n_nodes
                    : cfg_.ddtree_budget + 1);

            std::vector<int32_t> flat_tokens((size_t)N, 0);
            flat_tokens[0] = last_tok;
            for (int i = 0; i < tree.n_nodes; i++) flat_tokens[1 + i] = tree.token_ids[i];

            if (!sampled_verify) {
                const auto profile_snapshot_start = profile_start();
                if (!target->snapshot_kv()) {
                    step_graph_destroy(draft_sg);
                    return false;
                }
                profile_add(profile_snapshot_s, profile_snapshot_start);
            }

            std::vector<int32_t> posterior;
            std::vector<float>   node_logits;
            const auto profile_verify_start = profile_start();
            if (!target->verify_tree(committed, tree, flat_tokens, N, posterior,
                                     sampled_verify ? &node_logits : nullptr)) {
                std::fprintf(stderr, "spec-decode: verify_tree failed\n");
                step_graph_destroy(draft_sg);
                return false;
            }
            profile_add(profile_verify_s, profile_verify_start);
            target_forwards++;

            int next_token = -1, bonus_node = 0;
            std::vector<int> accepted;
            if (!sampled_verify) {
                accepted =
                    follow_verified_tree(tree, posterior.data(), next_token, &bonus_node);
            } else {
                accepted.reserve((size_t)tree.n_nodes + 1);
                accepted.push_back(0);
                std::vector<int32_t> hist = out_tokens;
                hist.push_back(last_tok);
                const int vocab = w_.n_vocab;
                int cur = 0;
                int stok = sample_logits(node_logits.data() + (size_t)cur * vocab,
                                         vocab, sampler_, hist, sampler_rng_);
                while (true) {
                    auto it = tree.child_maps[cur].find(stok);
                    if (it == tree.child_maps[cur].end()) break;
                    cur = it->second;
                    accepted.push_back(cur);
                    hist.push_back(stok);
                    stok = sample_logits(node_logits.data() + (size_t)cur * vocab,
                                         vocab, sampler_, hist, sampler_rng_);
                }
                next_token = stok;
            }

            int accepted_n = (int)accepted.size();        // root + accepted children
            if (accepted_n > need_commit_budget) accepted_n = need_commit_budget;
            if (accepted_n <= 0) { step_graph_destroy(draft_sg); break; }

            // Emit the accepted path: slot 0 = last_tok (pending from prev iter),
            // each subsequent accepted node = its tree token.
            bool hit_eos = false;
            int accepted_emitted = 0;
            for (int i = 0; i < accepted_n; i++) {
                const int dfs = accepted[i];
                const int32_t tok = budget.apply(
                    (dfs == 0) ? last_tok : tree.token_ids[dfs - 1]).token;
                out_tokens.push_back(tok);
                io.emit(tok);
                accepted_emitted++;
                if (io.is_cancelled()) { hit_eos = true; break; }
                if (target->is_eos(tok)) { hit_eos = true; break; }
            }

            // Seed-inclusive, like chain telemetry; exclude graph padding.
            acceptance.record_tree(accepted_emitted, tree.n_nodes, need_commit_budget);

            if (accepted_emitted <= 0) { step_graph_destroy(draft_sg); break; }

            if (!sampled_verify) {
                const int root_last_tok = last_tok;
                const bool use_tree_fast_rollback =
                    target->supports_fast_rollback() &&
                    accepted_emitted >= fast_rollback_threshold;

                if (use_tree_fast_rollback) {
                    // Fast greedy production path: restore to the accepted path
                    // from tree captures and defer the bonus as next step's root.
                    std::vector<int> accepted_committed(accepted.begin(),
                                                        accepted.begin() + accepted_emitted);
                    const auto profile_rollback_start = profile_start();
                    if (!target->rollback_to_tree(committed, tree, accepted_committed)) {
                        std::fprintf(stderr, "spec-decode: rollback_to_tree failed\n");
                        step_graph_destroy(draft_sg);
                        return false;
                    }
                    profile_add(profile_rollback_s, profile_rollback_start);
                    last_tok = next_token;

                    if (feature_mirror_.target_feat && !draft_parked_) {
                        const auto profile_feature_start = profile_start();
                        if (!sync_local_draft_features(committed, accepted_emitted)) {
                            step_graph_destroy(draft_sg);
                            return false;
                        }
                        profile_add(profile_feature_s, profile_feature_start);
                    }

                    committed   += accepted_emitted;
                    cache_.cur_pos = committed;
                    n_generated += accepted_emitted;
                    n_draft_steps++;
                    if (hit_eos || io.is_cancelled() || budget.remaining() <= 0 ||
                        last_tok < 0 || target->is_eos(last_tok)) {
                        break;
                    }
                    continue;
                }

                // Low-accept greedy path: mirror the chain's exact replay so the
                // next step starts from replayed F32 recurrent state. The accepted
                // path has already been emitted above; only emit the bonus here.
                int total_emitted = accepted_emitted;
                const bool can_commit_bonus =
                    !hit_eos && !io.is_cancelled() && next_token >= 0 &&
                    total_emitted < need_commit_budget;

                std::vector<int32_t> replay_batch;
                replay_batch.reserve((size_t)accepted_emitted + (can_commit_bonus ? 1 : 0));
                for (int i = 0; i < accepted_emitted; i++) {
                    const int dfs = accepted[i];
                    replay_batch.push_back((dfs == 0) ? root_last_tok : tree.token_ids[dfs - 1]);
                }
                if (can_commit_bonus) replay_batch.push_back(next_token);

                const auto profile_replay_start = profile_start();
                if (!target->restore_kv()) {
                    step_graph_destroy(draft_sg);
                    return false;
                }
                int replay_last_tok = -1;
                if (!target->verify_batch(replay_batch, committed, replay_last_tok, nullptr)) {
                    std::fprintf(stderr, "spec-decode: tree replay failed\n");
                    step_graph_destroy(draft_sg);
                    return false;
                }
                profile_add(profile_replay_s, profile_replay_start);
                target_forwards++;

                if (can_commit_bonus) {
                    next_token = budget.apply(next_token).token;
                    out_tokens.push_back(next_token);
                    io.emit(next_token);
                    total_emitted++;
                    if (io.is_cancelled()) {
                        hit_eos = true;
                    } else if (target->is_eos(next_token)) {
                        hit_eos = true;
                    }
                }

                last_tok = replay_last_tok;
                if (feature_mirror_.target_feat && !draft_parked_) {
                    if (!sync_local_draft_features(committed, total_emitted)) {
                        step_graph_destroy(draft_sg);
                        return false;
                    }
                }

                committed   += total_emitted;
                cache_.cur_pos = committed;
                n_generated += total_emitted;
                n_draft_steps++;
                if (hit_eos || io.is_cancelled() || budget.remaining() <= 0 ||
                    last_tok < 0 || target->is_eos(last_tok)) {
                    break;
                }
                continue;
            }

            // Sampled path keeps the distribution-preserving bonus replay but
            // still needs tree rollback for the accepted prefix first.
            std::vector<int> accepted_committed(accepted.begin(),
                                                accepted.begin() + accepted_emitted);
            if (!target->rollback_to_tree(committed, tree, accepted_committed)) {
                std::fprintf(stderr, "spec-decode: rollback_to_tree failed\n");
                step_graph_destroy(draft_sg);
                return false;
            }

            int total_emitted = accepted_emitted;
            int bonus_last_tok = -1;
            std::vector<float> bonus_logits;
            const bool can_commit_bonus =
                !hit_eos && !io.is_cancelled() && next_token >= 0 &&
                total_emitted < need_commit_budget;
            if (can_commit_bonus) {
                const int bonus_pos = committed + total_emitted;
                std::vector<int32_t> bonus_vec(1, next_token);
                // A delayed-state target must consume the accepted tree path
                // inside this verify and capture the committed bonus as the
                // next pending factor. A normal target has already materialized
                // the tree rollback and may keep the ordinary writeback path.
                const bool delayed_bonus = target->exact_fast_rollback();
                if (!target->verify_batch(bonus_vec, bonus_pos, bonus_last_tok,
                                          nullptr, delayed_bonus)) {
                    std::fprintf(stderr, "spec-decode: tree bonus replay failed\n");
                    step_graph_destroy(draft_sg);
                    return false;
                }
                target_forwards++;
                if (!target->read_verify_logits(1, bonus_logits)) {
                    std::fprintf(stderr, "spec-decode: tree bonus logits read failed\n");
                    step_graph_destroy(draft_sg);
                    return false;
                }
                if (bonus_logits.empty()) {
                    std::fprintf(stderr, "spec-decode: tree bonus logits empty\n");
                    step_graph_destroy(draft_sg);
                    return false;
                }
                if (delayed_bonus && !target->rollback_to(bonus_pos, 1)) {
                    std::fprintf(stderr,
                        "spec-decode: tree bonus factor commit failed\n");
                    step_graph_destroy(draft_sg);
                    return false;
                }

                next_token = budget.apply(next_token).token;
                out_tokens.push_back(next_token);
                io.emit(next_token);
                total_emitted++;
                if (io.is_cancelled()) {
                    hit_eos = true;
                } else if (target->is_eos(next_token)) {
                    hit_eos = true;
                }

                const int vocab_v = (int)bonus_logits.size();
                last_tok = sample_logits(bonus_logits.data(), vocab_v,
                                         sampler_, out_tokens, sampler_rng_);
            } else {
                last_tok = next_token;
            }

            // Sampled path commits the bonus in-step, so sync accepted+bonus.
            if (feature_mirror_.target_feat && !draft_parked_) {
                if (!sync_local_draft_features(committed, total_emitted)) {
                    step_graph_destroy(draft_sg);
                    return false;
                }
            }

            committed   += total_emitted;
            cache_.cur_pos = committed;
            n_generated += total_emitted;
            n_draft_steps++;
            if (hit_eos || io.is_cancelled() || budget.remaining() <= 0 || last_tok < 0) {
                break;
            }
            continue;
        }

        // 3a-ter. Long-context verify cap. Drafting is done; trim the block
        // we actually check. Hint injection below fills draft_tok[1..v_len-1],
        // so it sees the capped width and stays consistent.
        if (!ar_step && v_len > verify_cap) {
            v_len = verify_cap;
            if ((int)draft_tok.size() > verify_cap) {
                draft_tok.resize((size_t)verify_cap);
            }
        }
        if (!ar_step) {
            const int feedback_width = width_controller.next_width(v_len);
            if (feedback_width < v_len) {
                v_len = feedback_width;
                draft_tok.resize((size_t)v_len);
            }
        }

        // 3a-quater. KVFlash: keep the span AR-equivalent. No reselect point
        // inside the span but its first row, and no chunk allocated ahead of
        // the first row's own (a rejected tail or a terminal repair would leave
        // the allocation and its eviction behind).
        if (kvflash_active()) {
            const int cap = kvflash_step_span_cap(
                history_guard.base, history_guard.entry, out_tokens.size(), committed, v_len,
                kvflash_tau_, kvflash_pager_.chunk_tokens(),
                [&](int c) { return kvflash_pager_.is_resident(c); });
            if (cap < v_len) {
                v_len = cap;
                if ((int)draft_tok.size() > cap) draft_tok.resize((size_t)cap);
                n_span_clamped++;
            }
        }

        // 3b. Tool call hint injection: override draft tokens with pre-known
        // structural tokens for near-100% acceptance.
        int hint_fill = 0;
        if (hint_tokens && n_generated < (int)hint_tokens->size()) {
            const int hint_avail = (int)hint_tokens->size() - n_generated;
            hint_fill = std::min(hint_avail, v_len - 1);
            for (int i = 0; i < hint_fill; i++) {
                draft_tok[1 + i] = (*hint_tokens)[n_generated + i];
            }
        }

        // Notify observer with draft tokens for this step.
        if (io.observer) {
            io.observer("draft", draft_tok);
        }

        // 4. Verify: snapshot KV, run target forward over draft tokens. Every
        //    step snapshots, plain-decode ones included: io.emit can end the
        //    request (cancel, stop string) after the verify, and the terminal
        //    repair needs the pre-step state to leave the last token pending.
        const auto profile_snapshot_start = profile_start();
        if (!target->snapshot_kv()) {
            step_graph_destroy(draft_sg);
            return false;
        }
        profile_add(profile_snapshot_s, profile_snapshot_start);

        int verify_last_tok = -1;
        const auto profile_verify_start = profile_start();
        // Test hook (G3c P1 repro): LUCE_TEST_VERIFY_EXHAUST_AT=N makes the
        // verify of the (N+1)th step of a request behave as a pool-slot
        // allocation failure, which happens before any compute.
        static const int kExhaustAt = []() {
            const char * e = std::getenv("LUCE_TEST_VERIFY_EXHAUST_AT");
            return e ? std::atoi(e) : -1;
        }();
        const bool forced_exhaust = kExhaustAt >= 0 && n_draft_steps == kExhaustAt && !ar_step;
        if (forced_exhaust || !target->verify_batch(draft_tok, committed, verify_last_tok, &target_tok,
                                   /*capture_ssm_intermediates=*/true)) {
            // KVFlash pool exhaustion is not a spec failure: no state was
            // touched, so the rest of the request runs as plain decode.
            if (forced_exhaust || target->verify_pool_exhausted()) {
                std::fprintf(stderr,
                    "[kvflash] spec verify: pool exhausted at committed=%d, "
                    "finishing the request with AR\n", committed);
                if (!target->restore_kv()) {
                    step_graph_destroy(draft_sg);
                    return false;
                }
                step_graph_destroy(draft_sg);
                cache_.cur_pos = committed;
                // Spec convention at a step boundary: last_tok is the verified
                // next token, pending at `committed` and not emitted yet (the
                // emit lags one step); out_tokens.back() is the previous seed,
                // already forwarded at committed - 1. AR keeps its pending
                // token emitted and last in out_tokens, so emit it now;
                // do_ar_decode then forwards it at `committed`, once.
                if (!out_tokens.empty()) {
                    if (last_tok < 0) {
                        std::fprintf(stderr, "spec-decode: verify failed (no pending token)\n");
                        return false;
                    }
                    const bool was_forced = budget.forced_close();
                    const int32_t pend = budget.apply(last_tok).token;
                    if (budget.forced_close() && !was_forced && forced_close_out) *forced_close_out = true;
                    out_tokens.push_back(pend);
                    io.emit(pend);
                    cache_.last_tok = pend;
                    if (budget.remaining() <= 0 || io.is_cancelled() || IS_EOS_TOK(pend, w_)) {
                        pool_settled();
                        if (!finish_speculative_state()) return false;
                        log_target_forward_stats();
                        io.emit(-1);
                        return true;
                    }
                } else {
                    cache_.last_tok = last_tok;
                }
                pool_settled();
                if (!finish_speculative_state()) return false;
                bool ok = ar_tail();
                log_target_forward_stats();
                io.emit(-1);
                return ok;
            }
            std::fprintf(stderr, "spec-decode: verify failed\n");
            target->restore_kv();
            step_graph_destroy(draft_sg);
            return false;
        }
        profile_add(profile_verify_s, profile_verify_start);
        target_forwards++;

        // 5. Acceptance. Greedy: longest matching prefix between draft and
        // target argmax. Sampled-verify: walk the chain drawing each next
        // token from the target's sampler chain; accept while the draft
        // guessed the drawn token, and the first mismatch becomes the bonus
        // token (it is already a valid target sample at that position).
        int accept_n = 1;
        int bonus_tok = -1;
        if (sampled_verify) {
            if (!target->read_verify_logits(v_len, verify_logits)) {
                std::fprintf(stderr, "spec-decode: verify logits read failed\n");
                target->restore_kv();
                step_graph_destroy(draft_sg);
                return false;
            }
            const int vocab_v = (int)(verify_logits.size() / (size_t)v_len);
            static const bool kSvDebug = []() {
                const char * e = std::getenv("LUCE_SV_DEBUG");
                return e != nullptr && std::string(e) == "1";
            }();
            if (kSvDebug) {
                // Row-alignment check: CPU argmax over each bulk-read row must
                // equal the GPU argmax (target_tok). Divergence = misaligned
                // or stale bulk read.
                for (int i = 0; i < v_len; i++) {
                    const float * row = verify_logits.data() + (size_t)i * vocab_v;
                    int am = 0; float best = row[0];
                    for (int v = 1; v < vocab_v; v++)
                        if (row[v] > best) { best = row[v]; am = v; }
                    if (am != target_tok[i]) {
                        std::fprintf(stderr,
                            "[sv-debug] ROW MISMATCH i=%d cpu_argmax=%d (%.3f) "
                            "gpu_argmax=%d (%.3f) vocab_v=%d\n",
                            i, am, best, target_tok[i],
                            target_tok[i] < vocab_v ? row[target_tok[i]] : -999.0f,
                            vocab_v);
                        break;
                    }
                }
            }
            // Penalty history must match AR exactly: when AR samples the
            // token after X, X is already in out_tokens. The seed
            // draft_tok[0] is committed by this step's replay but not yet
            // in out_tokens, so add it before the walk — without it the
            // repetition penalty never sees the seed and the sampled
            // distribution drifts from AR whenever penalties are active.
            verify_history = out_tokens;
            verify_history.push_back(draft_tok[0]);
            bool mismatched = false;
            for (int i = 0; i < v_len - 1; i++) {
                const int s = sample_logits(
                    verify_logits.data() + (size_t)i * vocab_v, vocab_v,
                    sampler_, verify_history, sampler_rng_);
                if (kSvDebug && n_draft_steps < 3 && i < 4) {
                    std::fprintf(stderr,
                        "[sv-debug] step=%d pos=%d seed/draft0=%d draft=%d "
                        "sampled=%d\n",
                        n_draft_steps, i, draft_tok[0], draft_tok[i + 1], s);
                }
                if (draft_tok[i + 1] == s) {
                    accept_n++;
                    verify_history.push_back(s);
                } else {
                    bonus_tok = s;
                    mismatched = true;
                    break;
                }
            }
            (void)mismatched;
        } else {
            for (int i = 0; i < v_len - 1; i++) {
                if (draft_tok[i + 1] == target_tok[i]) accept_n++;
                else break;
            }
            bonus_tok = (accept_n < v_len) ? target_tok[accept_n - 1] : -1;
        }
        if (!ar_step) {
            width_controller.observe(accept_n, v_len);
        }
        // Track hint acceptance telemetry.
        if (hint_fill > 0) {
            n_hint_proposed += hint_fill;
            n_hint_accepted += std::min(hint_fill, accept_n - 1);
        }
        int commit_n  = spec_commit_n(accept_n, bonus_tok >= 0, need_commit_budget);
        if (commit_n <= accept_n) bonus_tok = -1;

        // 6. Fix state: adaptive fast-rollback vs legacy replay.
        //    Fast-rollback (implicit bonus, skip replay) is profitable when
        //    accept_n is large enough that skipping the replay saves more compute
        //    than the cost of deferring the bonus to the next step. TP uses
        //    device-local rollback; other paths use the configurable policy.
        rollback_diag.record_accept(accept_n);
        const bool use_fast_rollback =
            target->supports_fast_rollback() &&
            (accept_n >= rollback_policy.fast_rollback_threshold);

        int replay_last_tok = -1;
        bool fast_rolled_back = false;
        if (ar_step) {
            // Seed-only verify: the recurrent state already sits after the
            // one committed token; nothing to restore.
            bonus_tok = -1;
            commit_n = std::min(accept_n, need_commit_budget);
            replay_last_tok = target_tok[commit_n - 1];
            fast_rolled_back = true;
        } else if (use_fast_rollback) {
            // Fast rollback: restore SSM from captured intermediates, skip replay.
            // Implicit bonus: target_tok[commit_n-1] seeds next draft as draft_tok[0],
            // always accepted on next step.
            bonus_tok = -1;
            // Respect the generation budget: accept_n may exceed the remaining
            // budget (need_commit_budget), so committing accept_n would emit
            // more tokens than requested. commit_n was already clamped above.
            commit_n = std::min(accept_n, need_commit_budget);
            const auto profile_rollback_start = profile_start();
            const bool rolled = target->rollback_to(committed, commit_n);
            profile_add(profile_rollback_s, profile_rollback_start);
            if (rolled) {
                replay_last_tok = target_tok[commit_n - 1];
                fast_rolled_back = true;
                rollback_diag.record_fast_rollback(accept_n);
            } else {
                if (!target->rollback_failure_is_recoverable()) {
                    std::fprintf(stderr, "spec-decode: rollback_to failed after "
                                         "an in-place commit attempt; aborting\n");
                    step_graph_destroy(draft_sg);
                    return false;
                }
                // The pre-verify snapshot is still valid, so degrade to the
                // legacy restore+replay path below.
                std::fprintf(stderr, "spec-decode: rollback_to failed; "
                                     "falling back to restore+replay\n");
                rollback_diag.record_failed_fallback();
            }
        }
        if (!fast_rolled_back) {
            rollback_diag.record_legacy_replay();
            // Legacy replay: restore SSM snapshot, replay accepted + bonus tokens.
            // (When falling back from fast-rollback, bonus_tok is -1 and commit_n
            //  is the budget-clamped accepted count.)
            if (!target->restore_kv()) {
                step_graph_destroy(draft_sg);
                return false;
            }
            std::vector<int32_t> replay_batch((size_t)commit_n);
            for (int i = 0; i < commit_n; i++) {
                replay_batch[i] = (i < accept_n) ? draft_tok[i] : bonus_tok;
            }
            const auto profile_replay_start = profile_start();
            if (!target->verify_batch(replay_batch, committed, replay_last_tok, nullptr)) {
                std::fprintf(stderr, "spec-decode: replay failed\n");
                step_graph_destroy(draft_sg);
                return false;
            }
            profile_add(profile_replay_s, profile_replay_start);
            target_forwards++;
        }

        // Build replay_tok for emitting committed tokens.
        std::vector<int32_t> replay_tok((size_t)commit_n);
        for (int i = 0; i < commit_n; i++) {
            replay_tok[i] = (i < accept_n) ? draft_tok[i] : bonus_tok;
        }

        // 7. Sync features for replayed range to mirror (needed for next draft step)
        if (use_remote_draft && cache_.target_feat) {
            if (!sync_remote_draft_features(committed, commit_n)) {
                step_graph_destroy(draft_sg);
                return false;
            }
        } else if (feature_mirror_.target_feat && cache_.target_feat) {
            const auto profile_feature_start = profile_start();
            if (!sync_local_draft_features(committed, commit_n)) {
                step_graph_destroy(draft_sg);
                return false;
            }
            profile_add(profile_feature_s, profile_feature_start);
        }

        // 8. Emit committed tokens (stop at EOS)
        const int committed_at_step = committed;
        bool hit_eos = false;
        bool floor_to_ar = false;
        bool inject_tool_prefix = false;
        bool budget_close_fired = false;
        constexpr size_t kActionSuffixLookback = 16;
        constexpr size_t kSkipSequenceLookback = 64;
        int emitted = 0;
        for (int i = 0; i < commit_n; i++) {
            if (_min_floor > 0 && (int)out_tokens.size() < _min_floor &&
                IS_EOS_TOK(replay_tok[i], w_)) {
                // Action preambles often end as "I'll check:\n\n" before EOS.
                // Tokenization makes the colon several tokens back, so keep a
                // modest trailing window while still requiring a recent action
                // suffix token and no nearby completion phrase.
                const bool can_inject_tool =
                    stall_tool_prefix_tokens && !stall_tool_prefix_tokens->empty() &&
                    stall_action_suffix_tokens && !stall_action_suffix_tokens->empty() &&
                    tokens_have_recent_any(out_tokens, *stall_action_suffix_tokens,
                                           kActionSuffixLookback) &&
                    !(stall_skip_tokens &&
                      tokens_contain_recent_sequence(out_tokens,
                                                     *stall_skip_tokens,
                                                     kSkipSequenceLookback));
                if (can_inject_tool) {
                    // Debug-only diagnostic, same LUCE_MIN_TOKENS gating as the
                    // AR-path floor log above; silent in the default lane.
                    FILE* _d = open_dflash_floor_log();
                    if (_d) {
                        std::fprintf(_d,
                            "[spec-tool-floor] eos@%d committed=%d emitted=%d prefix=%zu -> ar\n",
                            (int)out_tokens.size(), committed, emitted,
                            stall_tool_prefix_tokens->size());
                        std::fclose(_d);
                    }
                    floor_to_ar = true;
                    inject_tool_prefix = true;
                    break;
                }
            }
            // Account only emitted candidates, including the first seed and
            // tokens after settlement. Replayed prefixes never pass here again.
            const int remaining = budget.remaining();
            const auto decision = budget.apply(replay_tok[i]);
            if (decision.cut) {
                std::fprintf(stderr,
                    "[budget-hook] spec-decode force-close at remaining=%d: "
                    "overriding token %d with close token %d\n",
                    remaining, replay_tok[i], decision.token);
                budget_close_fired = true;
                if (forced_close_out) *forced_close_out = true;
            }
            replay_tok[i] = decision.token;
            out_tokens.push_back(replay_tok[i]);
            io.emit(replay_tok[i]);
            emitted++;
            if (io.is_cancelled()) break;
            if (budget_close_fired) break;
            if (IS_EOS_TOK(replay_tok[i], w_)) { hit_eos = true; break; }
        }
        int injected = 0;
        SpecStepSettle settle{};
        if (floor_to_ar) {
            if (!target->restore_kv()) {
                step_graph_destroy(draft_sg);
                return false;
            }
            cache_.cur_pos = committed;
            if (emitted > 0) {
                std::vector<int32_t> replay_prefix(replay_tok.begin(),
                                                   replay_tok.begin() + emitted);
                int prefix_last_tok = -1;
                if (!target->verify_batch(replay_prefix, committed,
                                          prefix_last_tok, nullptr)) {
                    std::fprintf(stderr, "spec-decode: floor prefix replay failed\n");
                    step_graph_destroy(draft_sg);
                    return false;
                }
                target_forwards++;
            }
            committed += emitted;
            cache_.cur_pos = committed;
            if (inject_tool_prefix) {
                std::vector<int32_t> tool_prefix;
                tool_prefix.reserve(std::min(
                    stall_tool_prefix_tokens->size(), (size_t)budget.remaining()));
                for (int32_t candidate : *stall_tool_prefix_tokens) {
                    if (budget.remaining() <= 0) break;
                    const auto decision = budget.apply(candidate);
                    const int32_t tok = decision.token;
                    if (budget.forced_close() && forced_close_out) {
                        *forced_close_out = true;
                    }
                    tool_prefix.push_back(tok);
                    out_tokens.push_back(tok);
                    io.emit(tok);
                    hit_eos = IS_EOS_TOK(tok, w_);
                    if (io.is_cancelled() || hit_eos || decision.cut) break;
                }
                injected = (int)tool_prefix.size();
                // Like a budget cut, leave the last emitted token pending for
                // AR. Replay only the prefix; these are not new candidates.
                if (injected > 1) {
                    tool_prefix.pop_back();
                    int tool_prefix_last_tok = -1;
                    if (!target->verify_batch(tool_prefix, committed,
                                              tool_prefix_last_tok, nullptr)) {
                        std::fprintf(stderr, "spec-decode: tool prefix replay failed\n");
                        step_graph_destroy(draft_sg);
                        return false;
                    }
                    target_forwards++;
                    committed += injected - 1;
                    cache_.cur_pos = committed;
                }
            }
        } else {
            // Normal (non-floor) path: carry the replay's last token into the
            // next draft step. The floor_to_ar path never reaches the next
            // iteration — it sets cache_.last_tok directly below and returns —
            // so last_tok is intentionally left untouched when flooring.
            //
            // Sampled-verify: the seed is committed as-is by the next step
            // (draft_tok[0]), so it must itself be a sample from the target
            // distribution. replay_last_tok is the argmax — seeding with it
            // injects one greedy token per step, which biases the output and
            // locks long generations into repetition loops.
            if (sampled_verify && !replay_tok.empty() &&
                target->read_verify_logits((int)replay_tok.size(), verify_logits)) {
                const int vocab_v =
                    (int)(verify_logits.size() / replay_tok.size());
                last_tok = sample_logits(
                    verify_logits.data() +
                        (replay_tok.size() - 1) * (size_t)vocab_v,
                    vocab_v, sampler_, out_tokens, sampler_rng_);
            } else {
                last_tok = replay_last_tok;
            }
            // The target ran all commit_n tokens. A step that ends the request
            // (EOS, cancel, stop string, last budget token) must end in AR's
            // state, the last emitted token pending: restore the pre-step
            // snapshot and replay the emitted tokens but the last. A budget
            // close is repaired below.
            const bool terminal_step = hit_eos || io.is_cancelled() || budget.remaining() <= 0;
            settle = spec_step_settle(committed, commit_n, emitted, budget_close_fired, terminal_step);
            if (settle.repair && !budget_close_fired) {
                // Evidence line: the G3c cases count these to prove the path ran.
                if (kvflash_active() && spec_early_stop_needs_repair(commit_n, emitted, false)) {
                    std::fprintf(stderr,
                        "[kvflash] spec early-stop repair #%d: commit_n=%d emitted=%d "
                        "committed=%d (cancel=%d eos=%d)\n",
                        ++spec_early_stop_repairs_, commit_n, emitted, committed,
                        (int)io.is_cancelled(), (int)hit_eos);
                }
                if (!target->restore_kv()) {
                    step_graph_destroy(draft_sg);
                    return false;
                }
                cache_.cur_pos = committed;
                if (settle.replay_n > 0) {
                    std::vector<int32_t> replay_prefix(replay_tok.begin(),
                                                       replay_tok.begin() + settle.replay_n);
                    int prefix_last_tok = -1;
                    if (!target->verify_batch(replay_prefix, committed,
                                              prefix_last_tok, nullptr)) {
                        std::fprintf(stderr, "spec-decode: early-stop prefix replay failed\n");
                        step_graph_destroy(draft_sg);
                        return false;
                    }
                    target_forwards++;
                }
            }
            // A budget close is settled below, after the snapshot restore.
            if (!budget_close_fired) committed = settle.new_committed;
        }
        cache_.cur_pos = committed;
        n_generated += emitted + injected;
        // Only speculative steps enter the accept-rate
        // accounting; 1-token burst steps would otherwise dilute the rate
        // that steers the PFlash residency bandit.
        if (!ar_step) {
            acceptance.record_chain(accept_n, v_len, emitted, need_commit_budget);
        }
        n_draft_steps++;

        // Adaptive policy update on real spec steps: EMA of accepted draft
        // tokens (the seed is always accepted); a low EMA schedules a burst
        // of plain-decode steps, the step after the burst is a spec probe.
        if (adaptive.enabled()) {
            const double t_step = std::chrono::duration<double>(
                std::chrono::steady_clock::now() - t_step_start).count();
            double & t_ema = ar_step ? t_ar_step_ema : t_spec_step_ema;
            t_ema = (t_ema > 0.0) ? 0.9 * t_ema + 0.1 * t_step : t_step;
        }
        if (adaptive.enabled() && !ar_step) {
            const float accepted_drafts = (float)std::max(0, accept_n - 1);
            // A probe (first spec step after a burst) weighs its result
            // heavily: it is the only evidence about the current text.
            const float alpha = probe_step ? 0.5f : adaptive.ema_alpha;
            accepted_ema = (1.0f - alpha) * accepted_ema + alpha * accepted_drafts;
            probe_step = false;
            // A plain-decode burst commits greedy argmax tokens without
            // passing through the sampler chain, so bursts stay off under
            // sampled-verify to keep its output-distribution guarantee.
            if (!sampled_verify &&
                accepted_ema < adaptive.accept_threshold(live_step_ratio())) {
                ar_burst_left = adaptive.burst;
            }
        }

        // Notify observer with accepted tokens for this step.
        if (io.observer) {
            io.observer("verify", replay_tok);
        }

        // A cancelled forced-close emission still needs the prefix repair
        // below: the verified block contains the original token, not close[0].
        if (io.is_cancelled() && !budget_close_fired) break;
        if (floor_to_ar) {
            step_graph_destroy(draft_sg);
            pool_settled();
            cache_.last_tok = out_tokens.empty() ? last_tok : out_tokens.back();
            out_accept_rate = acceptance.rate();
            if (budget.remaining() <= 0 || io.is_cancelled() || hit_eos) {
                if (!finish_speculative_state()) return false;
                log_target_forward_stats();
                io.emit(-1);
                return true;
            }
            if (!finish_speculative_state()) return false;
            bool ok = ar_tail();
            log_target_forward_stats();
            io.emit(-1);
            return ok;
        }
        // Budget hook close: close[0] was emitted as this step's last token.
        // Roll back to the pre-step snapshot and replay only the tokens
        // emitted before it. close[0] stays pending, the AR convention:
        // do_ar_decode forwards out_tokens.back() at `committed` first, then
        // injects the rest of the sequence through the same controller.
        if (budget_close_fired) {
            if (!target->restore_kv()) {
                step_graph_destroy(draft_sg);
                return false;
            }
            committed = committed_at_step;
            cache_.cur_pos = committed;
            if (kvflash_active())
                std::fprintf(stderr, "[kvflash] spec budget-close repair: step_start=%d emitted=%d "
                                     "pooled_upto=%d chunk_tokens=%d\n",
                             committed_at_step, emitted, kvflash_qk_pooled_upto_,
                             kvflash_pager_.chunk_tokens());
            if (settle.replay_n > 0) {
                std::vector<int32_t> replay_prefix(replay_tok.begin(),
                                                   replay_tok.begin() + settle.replay_n);
                int prefix_last_tok = -1;
                if (!target->verify_batch(replay_prefix, committed,
                                          prefix_last_tok, nullptr)) {
                    std::fprintf(stderr, "spec-decode: budget-close prefix replay failed\n");
                    step_graph_destroy(draft_sg);
                    return false;
                }
                target_forwards++;
            }
            committed = settle.new_committed;
            cache_.cur_pos = committed;
            pool_settled();
            step_graph_destroy(draft_sg);
            cache_.last_tok = out_tokens.empty() ? last_tok : out_tokens.back();
            out_accept_rate = acceptance.rate();
            if (budget.remaining() <= 0 || io.is_cancelled() ||
                IS_EOS_TOK(out_tokens.back(), w_)) {
                if (!finish_speculative_state()) return false;
                log_target_forward_stats();
                io.emit(-1);
                return true;
            }
            if (!finish_speculative_state()) return false;
            bool ok = ar_tail();
            log_target_forward_stats();
            io.emit(-1);
            return ok;
        }
        if (hit_eos) break;
    }

    pool_settled();
    if (kvflash_active() && kvflash_qk_policy_ && std::getenv("LUCE_KVFLASH_TRACE_POOL"))
        std::fprintf(stderr, "[kvflash] spec end: committed=%d pooled_upto=%d chunk_tokens=%d\n",
                     committed, kvflash_qk_pooled_upto_, kvflash_pager_.chunk_tokens());
    if (!finish_speculative_state()) {
        step_graph_destroy(draft_sg);
        return false;
    }
    step_graph_destroy(draft_sg);

    auto t_dec1 = std::chrono::steady_clock::now();
    const double decode_s = std::chrono::duration<double>(t_dec1 - t_dec0).count();
    const double accept_pct = 100.0 * acceptance.rate();
    out_accept_rate = acceptance.rate();
    std::fprintf(stderr, "[spec-decode] tokens=%d time=%.3f s speed=%.2f tok/s "
                 "steps=%d accepted=%d/%d (%.1f%%) avg_commit=%.2f\n",
                 n_generated, decode_s,
                 n_generated > 0 ? n_generated / decode_s : 0.0,
                 n_draft_steps, acceptance.accepted(), acceptance.offered(), accept_pct,
                 n_draft_steps > 0 ? (double)n_generated / (double)n_draft_steps : 0.0);
    if (n_ar_burst_steps > 0) {
        std::fprintf(stderr, "[spec-decode] adaptive: %d of %d steps ran as plain decode "
                             "(step ratio %.2f, accept threshold %.2f drafts/step, burst %d)\n",
                     n_ar_burst_steps, n_draft_steps, live_step_ratio(),
                     adaptive.accept_threshold(live_step_ratio()), adaptive.burst);
    }
    if (tp_profile) {
        std::fprintf(stderr,
            "[spec-profile] draft=%.3fs project=%.3fs snapshot=%.3fs "
            "verify=%.3fs rollback=%.3fs replay=%.3fs feature=%.3fs accounted=%.3fs\n",
            profile_draft_s, profile_project_s, profile_snapshot_s,
            profile_verify_s, profile_rollback_s, profile_replay_s, profile_feature_s,
            profile_draft_s + profile_project_s + profile_snapshot_s +
            profile_verify_s + profile_rollback_s + profile_replay_s + profile_feature_s);
    }
    log_target_forward_stats();
    if (n_hint_proposed > 0) {
        std::fprintf(stderr, "[spec-decode] hint tokens: %d/%d accepted (%.1f%%)\n",
                     n_hint_accepted, n_hint_proposed,
                     100.0 * (double)n_hint_accepted / (double)n_hint_proposed);
    }

    io.emit(-1);
    return true;
}

int Qwen35Backend::verify_chain(int committed, const int32_t * draft_tok, int q_len) {
    (void)committed; (void)draft_tok; (void)q_len;
    return 0;
}

int Qwen35Backend::verify_tree(int committed, const DDTree & tree) {
    (void)committed; (void)tree;
    return 0;
}

}  // namespace luce::common
