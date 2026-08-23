#pragma once

#include "common/model_backend.h"
#include "common/dflash_feature_ring.h"
#include "common/moe_hybrid_routing_stats.h"
#include "common/moe_hybrid_stream.h"
#include "common/moe_storage_policy.h"
#include "internal.h"
#include "kimi_k3_internal.h"
#include "kimi_k3_prefill.h"
#include "placement/placement_config.h"

#include <algorithm>
#include <cstddef>
#include <random>
#include <memory>
#include <string>

struct ggml_backend;

namespace dflash::common {

enum class KimiK3CorePlacement {
    Accelerator,
    Cpu,
};

inline const char * kimi_k3_core_placement_name(
        KimiK3CorePlacement placement) {
    return placement == KimiK3CorePlacement::Cpu ? "cpu" : "accelerator";
}

inline bool parse_kimi_k3_core_placement(
        const std::string & value, KimiK3CorePlacement & out) {
    if (value == "accelerator") {
        out = KimiK3CorePlacement::Accelerator;
        return true;
    }
    if (value == "cpu") {
        out = KimiK3CorePlacement::Cpu;
        return true;
    }
    return false;
}

// Initialize the backend that owns KDA, MLA, shared experts, and the output
// head. Routed experts may independently use the exact NVMe stream engine on
// an accelerator.
ggml_backend_t init_kimi_k3_core_backend(
    KimiK3CorePlacement placement, int gpu, std::string * error = nullptr);

struct KimiK3BackendConfig {
    const char * model_path = nullptr;
    const char * draft_path = nullptr;
    DevicePlacement device;
    KimiK3CorePlacement core_placement = KimiK3CorePlacement::Accelerator;
    int draft_gpu = 0;
    int draft_ctx_max = 4096;
    bool fast_rollback = true;
    int stream_fd = -1;
    // Optional research trace. When set, ordinary autoregressive generation
    // atomically writes every full-vocabulary logit row produced by the
    // target. Null keeps production behavior and overhead unchanged.
    const char * logits_trace_path = nullptr;
    // -1 resolves DFLASH_MOE_TP_GPU and otherwise keeps the primary device
    // index. DFLASH_MOE_TP_BACKEND may select a different in-process runtime
    // (for example CUDA beside a HIP primary). A different backend or device
    // becomes the secondary capacity owner; routed work is partitioned while
    // dense KDA/MLA, recurrent state, and sampling remain primary-owned.
    int expert_gpu = -1;
    // Auto uses Kimi's capacity-safe file-backed routed experts. Resident is
    // retained as a deterministic oracle for small architecture fixtures.
    MoeStoragePolicy moe_storage = MoeStoragePolicy::Auto;
    // Opt-in capacity for diagnostic/oracle verification without requiring a
    // draft checkpoint. Zero preserves the ordinary runtime configuration.
    int oracle_verify_tokens = 0;
    bool oracle_layer_diagnostics = false;
};

struct KimiK3OracleVerifyResult {
    int width = 0;
    double sequential_seconds = 0.0;
    double verify_seconds = 0.0;
    double commit_seconds = 0.0;
    uint64_t sequential_storage_bytes = 0;
    uint64_t verify_storage_bytes = 0;
    uint64_t sequential_logical_provider_bytes = 0;
    uint64_t verify_logical_provider_bytes = 0;
    uint64_t sequential_compact_attempted = 0;
    uint64_t sequential_compact_completed = 0;
    uint64_t sequential_compact_fallbacks = 0;
    uint64_t sequential_compact_invalid = 0;
    uint64_t verify_compact_attempted = 0;
    uint64_t verify_compact_completed = 0;
    uint64_t verify_compact_fallbacks = 0;
    uint64_t verify_compact_invalid = 0;
    bool logits_bit_equal = false;
    bool argmax_bit_equal = false;
    bool recurrent_state_hash_equal = false;
    bool mla_rows_hash_equal = false;
    double logits_max_abs = 0.0;
    double logits_rel_l2 = 0.0;
    uint64_t sequential_recurrent_hash = 0;
    uint64_t verify_recurrent_hash = 0;
    uint64_t sequential_mla_hash = 0;
    uint64_t verify_mla_hash = 0;
    int first_hidden_mismatch_layer = -1;
    int first_hidden_mismatch_token = -1;
    double first_hidden_max_abs = 0.0;
    double first_hidden_rel_l2 = 0.0;
    int first_conv_state_mismatch_layer = -1;
    int first_ssm_state_mismatch_layer = -1;
    int first_mla_row_mismatch_layer = -1;
    std::vector<uint64_t> sequential_conv_layer_hashes;
    std::vector<uint64_t> verify_conv_layer_hashes;
    std::vector<uint64_t> sequential_ssm_layer_hashes;
    std::vector<uint64_t> verify_ssm_layer_hashes;
    std::vector<uint64_t> sequential_mla_layer_hashes;
    std::vector<uint64_t> verify_mla_layer_hashes;
};

class KimiK3Backend final : public ModelBackend {
public:
    explicit KimiK3Backend(const KimiK3BackendConfig & cfg);
    ~KimiK3Backend() override;

    bool init();

    // S0 research-only oracle ceiling. Both arms start from a freshly rebuilt
    // prompt state. The sequential arm processes oracle_tokens one row at a
    // time; the verify arm processes the same rows in one causal batch and
    // commits its recurrent KDA state through ReplaySSM. Default generation
    // never calls this entry point.
    bool benchmark_oracle_verify(
        const std::vector<int32_t> & prompt,
        const std::vector<int32_t> & oracle_tokens,
        KimiK3OracleVerifyResult & result,
        std::string * error = nullptr);

    void print_ready_banner() const override;
    bool park(ParkTarget target) override;
    bool unpark(ParkTarget target) override;
    bool is_target_parked() const override { return parked_; }

    GenerateResult generate_impl(const GenerateRequest & req,
                                 const DaemonIO & io) override;
    GenerateResult restore_and_generate_impl(int slot,
                                             const GenerateRequest & req,
                                             const DaemonIO & io) override;

    bool snapshot_save(int slot) override;
    void snapshot_free(int slot) override;
    bool snapshot_used(int slot) const override;
    int snapshot_cur_pos(int slot) const override;

    bool handle_compress(const std::string & line,
                         const DaemonIO & io) override;
    void free_drafter() override;
    bool supports_dflash_spec_decode() const override;
    DFlashTarget * dflash_target() override;
    void shutdown() override;

private:
    bool init_streaming();
    bool init_draft();
    void release_expert_backend();
    void maybe_save_routing_stats();
    bool write_logits_trace(const GenerateRequest & request,
                            const GenerateResult & result,
                            const std::vector<float> & rows,
                            const char * destination_path = nullptr) const;

    int32_t choose_token(const std::vector<float> & logits,
                         const SamplerCfg & sampler,
                         const std::vector<int32_t> & history);

    KimiK3BackendConfig cfg_;
    ggml_backend_t backend_ = nullptr;
    ggml_backend_t draft_backend_ = nullptr;
    ggml_backend_t expert_backend_ = nullptr;
    PlacementBackend expert_backend_kind_ = PlacementBackend::Auto;
    int expert_gpu_ = -1;
    KimiK3Weights weights_;
    KimiK3MoeCoreOffload moe_core_offload_;
    KimiK3Cache cache_;
    DraftWeights draft_weights_;
    DraftFeatureMirror feature_ring_;
    std::unique_ptr<class KimiK3DFlashTarget> dflash_target_;
    MoeHybridStreamEngine stream_engine_;
    MoeHybridStreamEngine secondary_stream_engine_;
    std::unique_ptr<class KimiK3RoutedOutputProvider> routed_output_provider_;
    MoeStreamDualOwnerExecutor dual_stream_executor_;
    MoeHybridPlacement stream_placement_;
    MoeStreamDualOwnerPolicy stream_owner_policy_;
    std::shared_ptr<MoeHybridRoutingStats> routing_stats_;
    std::string routing_stats_out_path_;
    int prefill_chunk_ = 1;
    bool p58_exact_multirow_ = false;
    bool prefill_census_ = false;
    bool parked_ = false;
    std::mt19937_64 rng_{std::random_device{}()};
};

} // namespace dflash::common
