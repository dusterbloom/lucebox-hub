#pragma once

#include <tuple>

namespace dflash::common {

struct Qwen35VerifyGraphKey {
    int width = 0;
    bool capture_intermediates = false;
    bool need_mask = false;

    constexpr bool operator<(const Qwen35VerifyGraphKey & other) const noexcept {
        return std::tie(width, capture_intermediates, need_mask) <
               std::tie(other.width, other.capture_intermediates,
                        other.need_mask);
    }
};

}  // namespace dflash::common
