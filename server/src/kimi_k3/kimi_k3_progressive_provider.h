#pragma once

#include "common/moe_hybrid_stream.h"

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace dflash::common {

// Optional prompt-phase service.  Decode providers remain one-row oriented;
// only implementations with an explicitly qualified macro contract expose
// this interface.
class KimiK3RoutedPrefillService {
public:
    virtual ~KimiK3RoutedPrefillService() = default;

    virtual bool supports_width(size_t width) const = 0;
    virtual bool evaluate_layer(
        int model_layer,
        int base_pos,
        const MoeStreamExpertSpec & exact_spec,
        const MoeStreamRouteBatch & native_routes,
        MoeHybridStreamEngine & exact_engine,
        std::vector<float> & output,
        std::string * err = nullptr) = 0;
};

// Monotonic routed-provider counters used to attribute one bounded generation
// phase without adding per-route logging to the measured hot path. Providers
// that do not expose these counters return an all-zero snapshot.
struct KimiK3RoutedRuntimeStats {
    uint64_t logical_provider_bytes = 0;
    uint64_t explicit_read_bytes = 0;
    uint64_t physical_direct_read_bytes = 0;
    uint64_t direct_io_ns = 0;
    uint64_t payload_h2d_bytes = 0;
    uint64_t metadata_h2d_bytes = 0;
    uint64_t compact_pack_ns = 0;
    uint64_t expert_graph_ns = 0;
    uint64_t expert_readback_ns = 0;
    uint64_t compact_attempted = 0;
    uint64_t compact_completed = 0;
    uint64_t compact_fallbacks = 0;
    uint64_t compact_invalid = 0;
    uint64_t async_begins = 0;
    uint64_t async_jobs = 0;
    uint64_t async_h2d_calls = 0;
    uint64_t async_h2d_bytes = 0;
    uint64_t async_input_d2d_copies = 0;
    uint64_t async_input_d2d_bytes = 0;
    uint64_t async_graph_enqueues = 0;
    uint64_t async_layer_flushes = 0;
    uint64_t async_abort_syncs = 0;
    uint64_t ordered_expert_d2d_copies = 0;
    uint64_t ordered_expert_d2d_bytes = 0;
    uint64_t ordered_join_launches = 0;
    uint64_t ordered_output_d2d_copies = 0;
    uint64_t ordered_output_d2d_bytes = 0;
    uint64_t p40_requested_slabs = 0;
    uint64_t p40_resident_before_slabs = 0;
    uint64_t p40_hits = 0;
    uint64_t p40_extensions = 0;
    uint64_t p40_cold = 0;
    uint64_t p40_unavailable = 0;
    uint64_t p40_completed = 0;
    uint64_t p40_aborted = 0;
    uint64_t p40_fallbacks = 0;
    uint64_t p40_evictions = 0;
    uint64_t p40_h2d_bytes = 0;
    uint64_t p40_scatter_calls = 0;
    uint64_t p40_scatter_avoided = 0;
};

// Research-only routed boundary used by H16. The exact stream engine remains
// the immutable teacher; implementations may replace one model layer only.
class KimiK3RoutedOutputProvider {
public:
    virtual ~KimiK3RoutedOutputProvider() = default;

    virtual bool handles_layer(int model_layer) const = 0;
    virtual KimiK3RoutedPrefillService * prefill_service() { return nullptr; }
    virtual bool evaluate(
        int model_layer,
        int base_pos,
        const MoeStreamExpertSpec & exact_spec,
        const MoeStreamRouteBatch & native_routes,
        MoeHybridStreamEngine & exact_engine,
        std::vector<float> & output,
        std::string * err = nullptr) = 0;

    virtual bool requires_device_output() const { return false; }
    virtual bool evaluate_device(
        int,
        int,
        const MoeStreamExpertSpec &,
        const MoeStreamRouteBatch &,
        MoeHybridStreamEngine &,
        ggml_backend_t,
        std::string * err = nullptr) {
        if (err) *err = "provider does not implement device output";
        return false;
    }
    virtual bool copy_device_output(
        ggml_backend_t,
        ggml_tensor *,
        std::string * err = nullptr) {
        if (err) *err = "provider has no pending device output";
        return false;
    }
    virtual void discard_device_output() {}
    virtual KimiK3RoutedRuntimeStats runtime_stats() const { return {}; }
};

// An unset or "exact" DFLASH_KIMI_LAYER1_PROVIDER returns success with a null
// provider. "slabs" and "whole" require the registered H16 runtime artifacts.
bool create_kimi_k3_progressive_provider_from_env(
    ggml_backend_t expert_backend,
    ggml_backend_t destination_backend,
    std::unique_ptr<KimiK3RoutedOutputProvider> & out,
    std::string * err = nullptr);

// H22 uses a strict, provenance-hashable text file rather than a long
// environment string.  The file must contain exactly one "layer budget" row
// for routed layers 1..92.  Budgets are drawn from the preregistered
// progressive ladder 24,48,72,...,192.  The returned vector is indexed by
// model_layer - 1.
bool parse_kimi_k3_layer_budget_table(
    const std::string & path,
    std::vector<int32_t> & budgets,
    std::string * err = nullptr);

// Deterministic policy helpers kept public for GPU-free regression tests.
std::vector<int32_t> select_kimi_k3_slab_prefix_ids(
    const int32_t * expert_ids,
    const float * router_weights,
    int top_k,
    const float * ordered_importance,
    int expert_count,
    int slabs_per_expert,
    int budget);

std::vector<int32_t> select_kimi_k3_whole_expert_routes(
    const int32_t * expert_ids,
    const float * router_weights,
    int top_k,
    const float * expert_importance,
    int expert_count,
    int budget);

// Select whole routes first, then retain an equal calibrated prefix from each
// selected expert. Returned IDs use expert * slabs_per_expert + rank.
std::vector<int32_t> select_kimi_k3_route_slab_prefix_ids(
    const int32_t * expert_ids,
    const float * router_weights,
    int top_k,
    const float * expert_importance,
    int expert_count,
    int slabs_per_expert,
    int route_budget,
    int slabs_per_route);

struct KimiK3CalibratedSlabPlan {
    int requested_budget = 0;
    std::vector<int32_t> selected_slab_ids;
    std::vector<int32_t> exact_route_indices;
};

// Experts whose calibrated flag is zero never enter the slab selector.  They
// are returned as native route indices for exact evaluation, and the actual
// selected count may therefore be below the requested nominal budget.
KimiK3CalibratedSlabPlan plan_kimi_k3_calibrated_slabs(
    const int32_t * expert_ids,
    const float * router_weights,
    int top_k,
    const float * ordered_importance,
    const uint8_t * calibrated_experts,
    int expert_count,
    int slabs_per_expert,
    int requested_budget);

// As above, but first chooses whole calibrated routes using the native expert
// importance and then keeps an equal calibrated slab prefix from each.  Routes
// without sufficient calibration are returned for exact evaluation.
KimiK3CalibratedSlabPlan plan_kimi_k3_calibrated_route_prefixes(
    const int32_t * expert_ids,
    const float * router_weights,
    int top_k,
    const float * expert_importance,
    const uint8_t * calibrated_experts,
    int expert_count,
    int slabs_per_expert,
    int route_budget,
    int slabs_per_route);

// A calibrated route with an empty selected prefix emits no authoritative
// sidecar read. Exact-fallback routes remain physical requests even though
// their slab list is empty in the P28 trace representation.
bool kimi_k3_prefetch_route_has_physical_request(
    bool calibrated, int selected_slab_count);

enum class KimiK3SparseDeliveryPolicy : uint8_t { BufferedSlabs, DirectSlabs, CompactPageable, CompactPinned, DirectPinnedCompact };
enum class KimiK3SparseUpload : uint8_t { SlabCopies, PageableCompact, PinnedCompact, PrepackedCompact };
KimiK3SparseUpload kimi_k3_sparse_upload_for_call(KimiK3SparseDeliveryPolicy, bool has_prepacked_payload);

// Convert rank-space route selection into the natural 12-bit cache mask and
// suppress ranks already covered by a resident variant. Invalid natural IDs
// are ignored here and rejected by artifact validation before execution.
uint16_t kimi_k3_selected_natural_slab_mask(
    const uint16_t * natural_by_rank,
    const uint8_t * selected_by_rank,
    int slab_count);
void kimi_k3_suppress_resident_slab_ranks(
    const uint16_t * natural_by_rank,
    uint16_t missing_mask,
    uint8_t * selected_by_rank,
    int slab_count);

// Validate one physical sparse payload and derive its exact natural-slab mask.
// Every ID must be in [0, 12) and occur exactly once.
bool kimi_k3_sparse_natural_mask(
    const uint16_t * naturals,
    int slab_count,
    uint16_t * mask);

struct KimiK3CompactWireLayout {
    size_t metadata_bytes = 32;
    size_t gate_offset = 0;
    size_t up_offset = 0;
    size_t down_offset = 0;
    size_t total_bytes = 0;
};

// Phase-2 compact executor wire contract: a fixed 32-byte natural-ID header,
// then all selected gate slabs, all selected up slabs, and all selected down
// slabs. Returns false on invalid counts, zero extents, or size overflow.
bool kimi_k3_compact_wire_layout(
    int slab_count,
    size_t gate_slab_bytes,
    size_t up_slab_bytes,
    size_t down_slab_bytes,
    KimiK3CompactWireLayout * layout);

} // namespace dflash::common
