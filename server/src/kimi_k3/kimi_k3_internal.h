// Native Kimi-K3 text-model support for Lucebox Hub.
//
// This is intentionally split into three model-neutral boundaries:
//   * GGUF loading owns tensor metadata/storage only;
//   * KimiK3Cache owns recurrent/attention state only; and
//   * kimi_k3_forward owns the architecture graph only.
//
// Routed-expert placement can therefore replace the resident expert tensors
// with the common MoE stream engine without changing KDA, MLA, AttnRes, or the
// public ModelBackend contract.

#pragma once

#include "common/gguf_mmap.h"
#include "common/moe_hybrid_stream.h"

#include "ggml.h"
#include "ggml-backend.h"

#include <cmath>
#include <cstdint>
#include <string>
#include <vector>

namespace dflash::common {

struct MoeHybridRoutingStats;

class MoeHybridStreamEngine;
class MoeStreamDualOwnerExecutor;
class MoeStreamExpertObserver;
class KimiK3RoutedOutputProvider;
struct MoeStreamDualOwnerPolicy;

struct KimiK3Layer {
    bool recurrent = false;

    ggml_tensor * attn_norm       = nullptr;
    ggml_tensor * ffn_norm        = nullptr;
    ggml_tensor * attn_res_score  = nullptr;
    ggml_tensor * ffn_res_score   = nullptr;

    // KDA (recurrent) attention.
    ggml_tensor * wq              = nullptr;
    ggml_tensor * wk              = nullptr;
    ggml_tensor * wv              = nullptr;
    ggml_tensor * wo              = nullptr;
    ggml_tensor * ssm_q_conv      = nullptr;
    ggml_tensor * ssm_k_conv      = nullptr;
    ggml_tensor * ssm_v_conv      = nullptr;
    ggml_tensor * ssm_f_a         = nullptr;
    ggml_tensor * ssm_f_b         = nullptr;
    ggml_tensor * ssm_beta        = nullptr;
    ggml_tensor * ssm_a           = nullptr;
    ggml_tensor * ssm_dt_b        = nullptr;
    ggml_tensor * ssm_g           = nullptr;
    ggml_tensor * ssm_o_norm      = nullptr;

    // MLA attention. Kimi-K3 uses the absorbed K-only cache when wk_b/wv_b
    // are present, which is the layout emitted by the official converter.
    ggml_tensor * wq_a            = nullptr;
    ggml_tensor * wq_a_norm       = nullptr;
    ggml_tensor * wq_b            = nullptr;
    ggml_tensor * wkv_a_mqa       = nullptr;
    ggml_tensor * wkv_a_norm      = nullptr;
    ggml_tensor * wk_b            = nullptr;
    ggml_tensor * wv_b            = nullptr;
    ggml_tensor * wkv_b           = nullptr;
    ggml_tensor * wqkv_gate       = nullptr;

    // Dense FFN (leading dense blocks).
    ggml_tensor * ffn_gate        = nullptr;
    ggml_tensor * ffn_up          = nullptr;
    ggml_tensor * ffn_down        = nullptr;

    // Latent routed MoE + full-width shared expert.
    ggml_tensor * ffn_gate_inp    = nullptr;
    ggml_tensor * ffn_exp_probs_b = nullptr;
    ggml_tensor * ffn_gate_exps   = nullptr;
    ggml_tensor * ffn_up_exps     = nullptr;
    ggml_tensor * ffn_down_exps   = nullptr;
    ggml_tensor * ffn_routed_down = nullptr;
    ggml_tensor * ffn_routed_up   = nullptr;
    ggml_tensor * ffn_routed_norm = nullptr;
    ggml_tensor * ffn_gate_shexp  = nullptr;
    ggml_tensor * ffn_up_shexp    = nullptr;
    ggml_tensor * ffn_down_shexp  = nullptr;
};

struct KimiK3Weights {
    // ctx/buf alias the first entries for compatibility with the original
    // single-file loader. Split GGUFs retain one metadata context and optional
    // resident backend buffer per shard.
    ggml_context *        ctx     = nullptr;
    ggml_backend_t        backend = nullptr;
    ggml_backend_buffer_t buf     = nullptr;
    std::vector<ggml_context *> contexts;
    std::vector<ggml_backend_buffer_t> buffers;
    // CPU-core capacity mode binds resident tensors directly to immutable
    // GGUF mappings. Buffers are non-owning; mappings outlive every tensor.
    std::vector<GgufMmap> mapped_shards;
    std::vector<std::string> shard_paths;

    // The routed stacks may remain file-backed. Regions use MoE-layer-local
    // indices [0, n_layer-n_dense_lead), independent of model layer numbers.
    std::vector<LayerExpertRegions> streamed_layer_regions;
    size_t max_streamed_expert_bytes = 0;
    bool routed_experts_streamed = false;

    ggml_tensor * tok_embd         = nullptr;
    ggml_tensor * output_norm      = nullptr;
    ggml_tensor * output           = nullptr;
    ggml_tensor * output_res_score = nullptr;
    std::vector<KimiK3Layer> layers;

    int n_layer       = 0;
    int n_embd        = 0;
    int n_ff          = 0;
    int n_vocab       = 0;
    int n_ctx_train   = 0;
    int n_head        = 0;
    int n_expert      = 0;
    int n_expert_used = 0;
    int n_ff_exp      = 0;
    int n_expert_latent = 0;
    int n_expert_shared = 0;
    int n_dense_lead  = 0;

    int ssm_d_conv    = 0;
    int kda_head_dim  = 0;
    int q_lora_rank   = 0;
    int kv_lora_rank  = 0;
    int mla_k_head_dim = 0;
    int mla_v_head_dim = 0;
    int rope_dim       = 0;
    int attn_res_block_size = 0;

    float rms_eps              = 1.0e-5f;
    float kda_gate_lower_bound = -INFINITY;
    float expert_weights_scale = 1.0f;
    bool  expert_weights_norm  = true;
    int   expert_gating_func   = 2; // sigmoid
    float situ_beta            = 4.0f;
    float situ_linear_beta     = 25.0f;
    int32_t eos_token_id       = 2;
};

inline MoeStreamExpertSpec make_kimi_k3_stream_spec(
        const KimiK3Weights & weights, const KimiK3Layer & layer) {
    MoeStreamExpertSpec spec;
    spec.input_dim = weights.n_expert_latent;
    spec.intermediate_dim = weights.n_ff_exp;
    spec.output_dim = weights.n_expert_latent;
    spec.gate_type = layer.ffn_gate_exps->type;
    spec.up_type = layer.ffn_up_exps->type;
    spec.down_type = layer.ffn_down_exps->type;
    spec.gated_activation = MoeGatedActivation::Situ;
    spec.situ_beta = weights.situ_beta;
    spec.situ_linear_beta = weights.situ_linear_beta;
    return spec;
}

// Optional accelerator copy of the always-used MoE core.  Routed expert
// stacks remain governed by the ordinary stream provider; this structure owns
// only the native router, latent projections, and full-width shared expert for
// each routed layer.  The CPU tensors remain immutable and are the default
// path.  This copy is enabled only by the explicit research environment gate.
struct KimiK3MoeCoreOffloadLayer {
    // Present only for layers selected by
    // DFLASH_KIMI_COMPLETE_PREP_LAYERS.  These tensors and recurrent states
    // keep the complete pre-expert half of that routed layer on one backend.
    ggml_tensor * attn_norm       = nullptr;
    ggml_tensor * ffn_norm        = nullptr;
    ggml_tensor * attn_res_score  = nullptr;
    ggml_tensor * ffn_res_score   = nullptr;
    ggml_tensor * wq              = nullptr;
    ggml_tensor * wk              = nullptr;
    ggml_tensor * wv              = nullptr;
    ggml_tensor * wo              = nullptr;
    ggml_tensor * ssm_q_conv      = nullptr;
    ggml_tensor * ssm_k_conv      = nullptr;
    ggml_tensor * ssm_v_conv      = nullptr;
    ggml_tensor * ssm_f_a         = nullptr;
    ggml_tensor * ssm_f_b         = nullptr;
    ggml_tensor * ssm_beta        = nullptr;
    ggml_tensor * ssm_a           = nullptr;
    ggml_tensor * ssm_dt_b        = nullptr;
    ggml_tensor * ssm_g           = nullptr;
    ggml_tensor * ssm_o_norm      = nullptr;
    ggml_tensor * conv_state      = nullptr;
    ggml_tensor * ssm_state       = nullptr;
    ggml_tensor * ffn_gate_inp    = nullptr;
    ggml_tensor * ffn_exp_probs_b = nullptr;
    ggml_tensor * ffn_routed_down = nullptr;
    ggml_tensor * ffn_routed_up   = nullptr;
    ggml_tensor * ffn_routed_norm = nullptr;
    ggml_tensor * ffn_gate_shexp  = nullptr;
    ggml_tensor * ffn_up_shexp    = nullptr;
    ggml_tensor * ffn_down_shexp  = nullptr;
};

struct KimiK3MoeCoreOffload {
    ggml_context * ctx = nullptr;
    ggml_backend_buffer_t buf = nullptr;
    ggml_backend_t backend = nullptr; // non-owning
    std::vector<KimiK3MoeCoreOffloadLayer> layers;
    size_t weight_bytes = 0;
    size_t state_bytes = 0;
    // Execution families selected by DFLASH_KIMI_MOE_CORE_OFFLOAD.  Keeping
    // these explicit lets the research path leave the numerically sensitive
    // router on the CPU while independently measuring latent and shared-MoE
    // placement.  "1" and "all" continue to select all three families.
    bool router = false;
    bool latent = false;
    bool shared = false;
    std::vector<uint8_t> complete_preparation;

    bool enabled() const {
        return backend && buf && !layers.empty();
    }

    bool preparation_enabled() const {
        return enabled() && (router || latent || shared);
    }

    bool join_enabled() const {
        return enabled() && latent;
    }

    bool complete_preparation_enabled(int model_layer) const {
        return enabled() && model_layer >= 0 &&
            model_layer < static_cast<int>(complete_preparation.size()) &&
            complete_preparation[static_cast<size_t>(model_layer)] != 0;
    }
};

struct KimiK3LayerCache {
    ggml_tensor * conv_state = nullptr; // [d_conv-1, 3*d_inner], F32
    ggml_tensor * ssm_state  = nullptr; // [head_dim, head_dim, n_head], F32
    ggml_tensor * mla_k      = nullptr; // [kv_rank+rope_dim, 1, max_ctx], F16

    // Speculative-decode state.  ReplaySSM captures the much smaller
    // pre-KDA activation for every verify row, then re-runs only the accepted
    // recurrent transitions at commit time.  The snapshots are a failure-safe
    // for the commit graph; ordinary rejected verification leaves the live
    // recurrent tensors untouched and therefore needs no restore copy.
    ggml_tensor * conv_state_snap = nullptr;
    ggml_tensor * ssm_state_snap  = nullptr;
    ggml_tensor * replay_input    = nullptr; // [hidden, max_verify_tokens], F32
};

struct KimiK3Cache {
    ggml_context *        ctx = nullptr;
    ggml_backend_buffer_t buf = nullptr;
    std::vector<KimiK3LayerCache> layers;
    int max_ctx = 0;
    int cur_pos = 0;
    int max_verify_tokens = 0;
    int snapshot_pos = -1;
    int replay_base_pos = -1;
    int replay_n_tokens = 0;
    bool snapshot_valid = false;
    bool replay_valid = false;
    bool recurrent_state_pristine = false;
    // P58 records each replay row through the native one-row KDA graph. Its
    // later ReplaySSM commit must retain that same row boundary.
    bool replay_exact_rows = false;
    // Opaque graph-lifetime owner for the opt-in one-token persistent routed
    // preparation path.  The concrete type remains private to graph.cpp.
    void * persistent_routed_preparation = nullptr;
};

// Model-neutral forward result shape used by the Kimi DFlash adapter.  Capture
// rows are capture-major, then token-major:
//   [capture_layer][token][hidden].
struct KimiK3MoePanelCapture {
    int layer = -1;          // zero-indexed model layer
    int base_pos = 0;
    int n_tokens = 0;
    int latent_dimension = 0;
    int top_k = 0;
    std::vector<float> latent;       // token-major exact z = W_down h
    std::vector<int32_t> expert_ids; // token-major native router IDs
    std::vector<float> router_weights;
};

// Retained exact recurrent-micro / routed-macro discriminator widths. These
// are explicit research envelopes, not a generic chunking promise.
inline bool kimi_k3_exact_multirow_width(size_t width) {
    return width == 8 || width == 64 || width == 1024;
}

struct KimiK3ForwardOptions {
    const std::vector<int> * capture_layer_ids = nullptr;
    bool capture_replay = false;
    bool read_logits = false;
    bool read_argmax = true;
    // A non-negative model-layer index stops after its exact latent/router
    // preparation and before any routed expert for that layer is requested.
    int stop_before_moe_layer = -1;
    KimiK3MoePanelCapture * panel_capture = nullptr;
    // Optional bounded full-forward capture. Each requested routed layer is
    // captured at the same pre-expert boundary as panel_capture, while exact
    // execution continues through the model. The result vector follows the
    // caller-provided layer order and is replaced on every forward call.
    const std::vector<int> * panel_capture_layer_ids = nullptr;
    std::vector<KimiK3MoePanelCapture> * panel_captures = nullptr;
    MoeStreamExpertObserver * expert_observer = nullptr;
    KimiK3RoutedOutputProvider * routed_output_provider = nullptr;
    // P58-only semantic discriminator: evolve KDA/MLA and all core arithmetic
    // one row at a time, while exposing one bounded routed-provider macro per
    // layer. The public backend validates the complete fail-closed envelope.
    bool exact_multirow_core = false;
    // Experimental CPU-core / accelerator-MoE split. Null preserves the
    // established single-backend arithmetic and placement.
    KimiK3MoeCoreOffload * moe_core_offload = nullptr;
};

struct KimiK3ForwardResult {
    std::vector<float> logits;
    std::vector<int32_t> argmax;
    std::vector<float> captured_hidden;
};

// Isolated research measurement for one recurrent KDA layer.  This copies
// only the selected layer's KDA tensors to the accelerator and compares the
// same one-token graph against the mapped CPU tensors.  It does not alter the
// production placement or model state.
struct KimiK3KdaLayerBenchmarkResult {
    int model_layer = -1;
    int iterations = 0;
    size_t weight_bytes = 0;
    double cpu_median_ms = 0.0;
    double accelerator_median_ms = 0.0;
    double speedup = 0.0;
    double relative_l2 = 0.0;
    double cosine = 0.0;
    double max_abs = 0.0;
};

bool benchmark_kimi_k3_kda_layer(
    ggml_backend_t cpu_backend,
    ggml_backend_t accelerator_backend,
    const KimiK3Weights & weights,
    int model_layer,
    int iterations,
    KimiK3KdaLayerBenchmarkResult & result,
    std::string * error = nullptr);

// Isolated ceiling for the complete pre-expert half of one recurrent routed
// layer.  Unlike KimiK3KdaLayerBenchmarkResult this includes both AttnRes
// mixes, their norms, KDA, the native router/latent projection, and the shared
// expert.  It therefore measures the real CPU -> streamed-expert boundary:
// prefix, routed latent, route IDs/weights, and shared output are read back.
// Production execution remains unchanged.
struct KimiK3RoutedPreparationBenchmarkResult {
    int model_layer = -1;
    int checkpoint_count = 0;
    int iterations = 0;
    size_t weight_bytes = 0;
    double cpu_median_ms = 0.0;
    double accelerator_median_ms = 0.0;
    double speedup = 0.0;
    double persistent_accelerator_median_ms = 0.0;
    double persistent_speedup_vs_transient = 0.0;
    size_t persistent_compute_buffer_bytes = 0;
    size_t persistent_metadata_bytes = 0;
    int persistent_graph_nodes = 0;
    bool persistent_prefix_byte_equal = false;
    bool persistent_routed_byte_equal = false;
    bool persistent_shared_byte_equal = false;
    bool persistent_route_weight_byte_equal = false;
    bool persistent_selected_id_equal = false;
    double persistent_max_abs = 0.0;
    double prefix_relative_l2 = 0.0;
    double routed_relative_l2 = 0.0;
    double shared_relative_l2 = 0.0;
    double route_weight_relative_l2 = 0.0;
    double max_abs = 0.0;
    int selected_id_agreement = 0;
    int selected_id_count = 0;
};

bool benchmark_kimi_k3_routed_preparation(
    ggml_backend_t cpu_backend,
    ggml_backend_t accelerator_backend,
    const KimiK3Weights & weights,
    int model_layer,
    int iterations,
    KimiK3RoutedPreparationBenchmarkResult & result,
    std::string * error = nullptr);

struct KimiK3LoadOptions {
    bool stream_routed_experts = false;
    // Bind non-routed tensors directly to read-only GGUF mappings instead of
    // allocating and copying them. This is valid for the CPU core backend and
    // keeps the full 57.94 GiB core executable under a smaller RAM ceiling.
    bool mmap_resident_tensors = false;
    // When non-negative, allocate only the tensors required to reach this
    // layer's native router and routed-down projection.  The layer itself is
    // not evaluated beyond that boundary.  This is a research capture mode;
    // ordinary full-model loading leaves it at -1.
    int stop_before_moe_layer = -1;
};

bool kimi_k3_capture_tensor_required(const std::string & name,
                                     int stop_before_moe_layer);

bool load_kimi_k3_gguf(const std::string & path,
                       ggml_backend_t backend,
                       KimiK3Weights & out,
                       const KimiK3LoadOptions & options);
bool load_kimi_k3_gguf(const std::string & path,
                       ggml_backend_t backend,
                       KimiK3Weights & out,
                       bool stream_routed_experts = false);
void free_kimi_k3_weights(KimiK3Weights & w);

bool init_kimi_k3_moe_core_offload(ggml_backend_t accelerator_backend,
                                   const KimiK3Weights & weights,
                                   KimiK3MoeCoreOffload & out,
                                   std::string * error = nullptr);
void free_kimi_k3_moe_core_offload(KimiK3MoeCoreOffload & offload);
void reset_kimi_k3_moe_core_offload_state(KimiK3MoeCoreOffload & offload);

// Decode selected embedding rows with the format's canonical scalar decoder.
// This is the fallback used when a device backend does not implement GET_ROWS
// for the checkpoint's quantized embedding type.
bool kimi_k3_read_token_embeddings_on_host(
    const KimiK3Weights & w,
    const std::vector<int32_t> & tokens,
    std::vector<float> & hidden);

bool create_kimi_k3_cache(ggml_backend_t backend,
                          const KimiK3Weights & w,
                          int max_ctx,
                          KimiK3Cache & out,
                          int max_verify_tokens = 0);
void reset_kimi_k3_cache(KimiK3Cache & cache);
void free_kimi_k3_cache(KimiK3Cache & cache);

// Batch forward used for target verification.  With capture_replay=true the
// recurrent state is read-only: KDA inputs are persisted for a later
// kimi_k3_replay_commit(), while MLA writes remain position-indexed and become
// invisible simply by restoring cur_pos.
bool kimi_k3_forward(ggml_backend_t backend,
                     const KimiK3Weights & w,
                     KimiK3Cache & cache,
                     const std::vector<int32_t> & tokens,
                     int base_pos,
                     const KimiK3ForwardOptions & options,
                     KimiK3ForwardResult & result,
                     MoeHybridStreamEngine * stream_engine = nullptr,
                     MoeStreamDualOwnerExecutor * dual_stream_executor = nullptr,
                     const MoeStreamDualOwnerPolicy * stream_owner_policy = nullptr,
                     MoeHybridRoutingStats * routing_stats = nullptr);

bool kimi_k3_replay_snapshot(ggml_backend_t backend, KimiK3Cache & cache);
bool kimi_k3_replay_restore(ggml_backend_t backend, KimiK3Cache & cache);
bool kimi_k3_replay_commit(ggml_backend_t backend,
                           const KimiK3Weights & w,
                           KimiK3Cache & cache,
                           int base_pos,
                           int commit_n);

// Compatibility wrapper for the ordinary one-token AR path. Speculative
// verification calls kimi_k3_forward directly with a bounded token batch.
bool kimi_k3_step(ggml_backend_t backend,
                  const KimiK3Weights & w,
                  KimiK3Cache & cache,
                  int32_t token,
                  int position,
                  std::vector<float> & logits,
                  MoeHybridStreamEngine * stream_engine = nullptr,
                  MoeStreamDualOwnerExecutor * dual_stream_executor = nullptr,
                  const MoeStreamDualOwnerPolicy * stream_owner_policy = nullptr,
                  MoeHybridRoutingStats * routing_stats = nullptr,
                  KimiK3RoutedOutputProvider * routed_output_provider = nullptr,
                  KimiK3MoeCoreOffload * moe_core_offload = nullptr);

} // namespace dflash::common
