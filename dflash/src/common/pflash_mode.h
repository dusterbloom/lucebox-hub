// PFlash mode enum and string parser — no other dflash dependencies.
// Included by model_backend.h, qwen3_drafter.h, and http_server.h.

#pragma once

#include <optional>
#include <string>

namespace dflash27b {

enum class PFlashMode { OFF, AUTO, ALWAYS };

// Parse "off" / "auto" / "always" → typed enum.
// Returns nullopt for any other string — caller should return HTTP 400.
inline std::optional<PFlashMode> parse_pflash_mode_str(const std::string & s) {
    if (s == "off")    return PFlashMode::OFF;
    if (s == "auto")   return PFlashMode::AUTO;
    if (s == "always") return PFlashMode::ALWAYS;
    return std::nullopt;
}

}  // namespace dflash27b
