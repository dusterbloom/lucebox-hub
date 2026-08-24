#pragma once

#include "common/model_backend.h"
#include "common/moe_hybrid_stream.h"
#include "kimi_k3_calibrated_provider.h"
#include "kimi_k3_internal.h"
#include "kimi_k3_prefill.h"
#include "placement/placement_config.h"

#include <array>
#include <memory>
#include <random>
#include <string>
#include <vector>

namespace dflash::common {

struct KimiK3BackendConfig {
    const char * model_path = nullptr;
    DevicePlacement device;
};

// Production Kimi-K3 is deliberately one owner: core, routed expert compute,
// and canonical joins all use one HIP backend. The routed bank remains
// file-backed and is serviced by the shared NVMe stream engine.
class KimiK3Backend final : public ModelBackend {
public:
    explicit KimiK3Backend(const KimiK3BackendConfig & cfg);
    ~KimiK3Backend() override;

    KimiK3Backend(const KimiK3Backend &) = delete;
    KimiK3Backend & operator=(const KimiK3Backend &) = delete;

    bool init();

    void print_ready_banner() const override;
    bool park(ParkTarget target) override;
    bool unpark(ParkTarget target) override;
    bool is_target_parked() const override { return false; }

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
    void free_drafter() override {}
    void shutdown() override;

private:
    bool resolve_prefill_policy(std::string & error);
    bool init_streaming(std::string & error);
    GenerateResult generate_from_state(
        const GenerateRequest & req,
        const DaemonIO & io,
        const KimiK3PrefixSnapshot * snapshot);
    bool capture_last_logits(std::string & error) const;
    int32_t choose_token(const std::vector<float> & logits,
                         const SamplerCfg & sampler,
                         bool do_sample,
                         const std::vector<int32_t> & history);

    KimiK3BackendConfig cfg_;

    // Manual shutdown follows the same dependency order as reverse member
    // destruction: provider -> stream -> snapshots -> cache -> weights -> backend.
    ggml_backend_t backend_ = nullptr;
    ggml_backend_t snapshot_backend_ = nullptr;
    KimiK3Weights weights_;
    KimiK3Cache cache_;
    std::array<KimiK3PrefixSnapshot, ModelBackend::kMaxSlots>
        prefix_snapshots_;
    size_t snapshot_bytes_ = 0;
    std::vector<float> last_logits_;
    int last_logits_pos_ = -1;
    MoeHybridStreamEngine stream_engine_;
    std::unique_ptr<KimiK3RoutedOutputProvider> routed_output_provider_;

    KimiK3PrefillPolicy prefill_policy_;
    bool initialized_ = false;
    std::mt19937_64 rng_{std::random_device{}()};
};

} // namespace dflash::common
