#pragma once

#include "common/moe_hybrid_stream.h"

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace dflash::common {

// Prompt-only service boundary. Decode remains one-row; a provider exposes
// this interface only for explicitly qualified macro widths.
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
        std::string * error = nullptr) = 0;
};

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

// Exact routed-expert boundary shared by decode and prompt macro execution.
// Device output is optional and guarded: an unconsumed pending output is
// discarded on every failure path.
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
        std::string * error = nullptr) = 0;

    virtual bool requires_device_output() const { return false; }
    virtual bool evaluate_device(
        int,
        int,
        const MoeStreamExpertSpec &,
        const MoeStreamRouteBatch &,
        MoeHybridStreamEngine &,
        ggml_backend_t,
        std::string * error = nullptr) {
        if (error) *error = "provider does not implement device output";
        return false;
    }
    virtual bool copy_device_output(
        ggml_backend_t,
        ggml_tensor *,
        std::string * error = nullptr) {
        if (error) *error = "provider has no pending device output";
        return false;
    }
    virtual void discard_device_output() {}
    virtual KimiK3RoutedRuntimeStats runtime_stats() const { return {}; }
};

} // namespace dflash::common
