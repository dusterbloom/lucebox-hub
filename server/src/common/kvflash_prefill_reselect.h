// LUCE_KVFLASH_PREFILL_RESELECT: pure helpers for reselecting the kvflash pool
// by target-QK score before the final prefill batch (no ggml, no GPU).

#pragma once

#include <algorithm>
#include <cmath>
#include <vector>

namespace luce::common {

// Absolute position R at which the pool is reselected. mode 1: the last turn
// cut when it lies strictly inside (kv_offset, end), else end-1 when that is
// past kv_offset, else -1 (no reselect). mode 2: end-1 when past kv_offset,
// else -1; cuts are ignored.
inline int kvflash_prefill_reselect_point(const std::vector<int> & turn_cuts,
                                          int kv_offset, int end, int mode = 1) {
    if (mode < 2 && !turn_cuts.empty() && turn_cuts.back() > kv_offset && turn_cuts.back() < end)
        return turn_cuts.back();
    return end - 1 > kv_offset ? end - 1 : -1;
}

// Mode 3 (LUCE_KVFLASH_PREFILL_RESELECT=3): route with the LAST prompt token's
// query. R = end-1 is a batch boundary; the state at R is snapshotted, the
// final token runs once to produce q, the pool is reselected, the token is
// rewound and re-run under the new pool. `run` is false when there is no
// earlier state to snapshot (end-1 <= kv_offset). reserve_chunks: pool room
// left for the one re-run token.
struct KvFlashMode3Plan {
    int  r = -1;
    bool run = false;
    int  reserve_chunks = 1;
};
inline KvFlashMode3Plan kvflash_mode3_plan(int kv_offset, int end) {
    KvFlashMode3Plan p;
    p.r = kvflash_prefill_reselect_point({}, kv_offset, end, 3);
    p.run = p.r >= 0;
    return p;
}

// Mode-3 rerun check: max |a-b| over two logits rows (-1 on a size mismatch),
// and the argmax of a row (-1 when empty).
inline float kvflash_max_abs_diff(const std::vector<float> & a, const std::vector<float> & b) {
    if (a.size() != b.size()) return -1.0f;
    float m = 0.0f;
    for (size_t i = 0; i < a.size(); i++) m = std::max(m, std::fabs(a[i] - b[i]));
    return m;
}
inline int kvflash_argmax(const std::vector<float> & a) {
    return a.empty() ? -1 : (int)(std::max_element(a.begin(), a.end()) - a.begin());
}

// Sorted copy of `cuts` with r inserted; unchanged if r < 0 or already present.
inline std::vector<int> kvflash_cuts_with(const std::vector<int> & cuts, int r) {
    std::vector<int> out = cuts;
    if (r >= 0 && std::find(out.begin(), out.end(), r) == out.end()) out.push_back(r);
    std::sort(out.begin(), out.end());
    return out;
}

// R-LRU recall: block b = chunks [cpb*b, cpb*b + cpb); its score is the sum of
// its chunk scores. Eligible blocks lie entirely below first_window_chunk, lie
// entirely inside chunk_scores, and exclude chunk 0 (the sink). Returns the
// top-k block ids, best first, ties newer-first.
inline std::vector<int> kvflash_recall_pick(const std::vector<float> & chunk_scores,
                                            int first_window_chunk, int k,
                                            int chunks_per_block = 10) {
    std::vector<int> ids;
    if (k <= 0 || chunks_per_block <= 0) return ids;
    const int limit = std::min(first_window_chunk, (int)chunk_scores.size());
    std::vector<double> bs;
    for (int b = 1; (b + 1) * chunks_per_block <= limit; b++) {
        double s = 0.0;
        for (int c = b * chunks_per_block; c < (b + 1) * chunks_per_block; c++) s += chunk_scores[(size_t)c];
        bs.resize((size_t)b + 1);
        bs[(size_t)b] = s;
        ids.push_back(b);
    }
    std::stable_sort(ids.begin(), ids.end(), [&](int a, int b) {
        return bs[(size_t)a] != bs[(size_t)b] ? bs[(size_t)a] > bs[(size_t)b] : a > b;
    });
    if ((int)ids.size() > k) ids.resize((size_t)k);
    return ids;
}

}  // namespace luce::common
