#pragma once

#include "ggml.h"

#include <array>
#include <cstdint>
#include <string>
#include <vector>

namespace dflash::common {

inline constexpr std::array<char, 8> kKimiK3PanelCaptureMagic = {
    'K', '3', 'P', 'N', 'L', '0', '0', '1'};
inline constexpr uint32_t kKimiK3PanelCaptureVersion = 1;

struct KimiK3PanelCaptureHeader {
    std::array<char, 8> magic = kKimiK3PanelCaptureMagic;
    uint32_t version = kKimiK3PanelCaptureVersion;
    int32_t model_layer = -1;
    uint32_t latent_dimension = 0;
    uint32_t top_k = 0;
    uint64_t sequence_count = 0;
    uint64_t token_count = 0;
    uint32_t latent_storage = 1; // 1 = bfloat16
    uint32_t route_weight_storage = 0; // 0 = float32
    std::array<uint64_t, 4> reserved{};
};
static_assert(sizeof(KimiK3PanelCaptureHeader) == 80,
              "panel capture header must remain byte-stable");

struct KimiK3PanelCaptureRecord {
    std::string id;
    uint8_t split = 2; // 0 = calibration, 1 = validation
    std::vector<int32_t> tokens;
    std::vector<ggml_bf16_t> latent;
    std::vector<int32_t> expert_ids;
    std::vector<float> router_weights;
};

struct KimiK3PanelCaptureArtifact {
    KimiK3PanelCaptureHeader header;
    std::vector<KimiK3PanelCaptureRecord> records;
};

bool read_kimi_k3_panel_capture(
    const std::string & path,
    KimiK3PanelCaptureArtifact & artifact,
    std::string * error = nullptr);

} // namespace dflash::common
