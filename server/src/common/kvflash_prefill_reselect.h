// LUCE_KVFLASH_PREFILL_RESELECT: pure helpers for reselecting the kvflash pool
// by target-QK score before the final prefill batch (no ggml, no GPU).

#pragma once

#include <algorithm>
#include <vector>

namespace luce::common {

// Absolute position R at which the pool is reselected: the last turn cut when
// it lies strictly inside (kv_offset, end), else end-1 when that is past
// kv_offset, else -1 (no reselect for this request).
inline int kvflash_prefill_reselect_point(const std::vector<int> & turn_cuts,
                                          int kv_offset, int end) {
    if (!turn_cuts.empty() && turn_cuts.back() > kv_offset && turn_cuts.back() < end)
        return turn_cuts.back();
    return end - 1 > kv_offset ? end - 1 : -1;
}

// Sorted copy of `cuts` with r inserted; unchanged if r < 0 or already present.
inline std::vector<int> kvflash_cuts_with(const std::vector<int> & cuts, int r) {
    std::vector<int> out = cuts;
    if (r >= 0 && std::find(out.begin(), out.end(), r) == out.end()) out.push_back(r);
    std::sort(out.begin(), out.end());
    return out;
}

}  // namespace luce::common
