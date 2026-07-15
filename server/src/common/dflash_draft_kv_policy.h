#pragma once

namespace dflash::common {

constexpr int draft_kv_append_capacity(
        int draft_rows,
        bool proposals_only) noexcept {
    if (proposals_only) {
        return draft_rows > 0 ? draft_rows + 1 : 1;
    }
    return draft_rows > 0 ? 2 * draft_rows + 2 : 2;
}

}  // namespace dflash::common
