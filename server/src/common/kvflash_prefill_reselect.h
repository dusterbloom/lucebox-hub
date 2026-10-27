// LUCE_KVFLASH_PREFILL_RESELECT: pure helpers for reselecting the kvflash pool
// by target-QK score before the final prefill batch (no ggml, no GPU).

#pragma once

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <string>
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

// Position of the last `<|im_start|>` immediately followed by `user`, or -1.
inline int kvflash_last_user_turn(const std::vector<int32_t> & prompt, int32_t im_start, int32_t user) {
    if (im_start < 0 || user < 0) return -1;
    for (int i = (int)prompt.size() - 2; i >= 0; i--)
        if (prompt[(size_t)i] == im_start && prompt[(size_t)i + 1] == user) return i;
    return -1;
}

// Mode 4 ("rewind the turn"): R4 = start of the last user turn of this
// request's tokens (absolute = kv_offset + index), or the prefill start when
// the tokens hold none. Never below kv_offset. reserve_chunks < 0: derived
// from the tokens left to prefill at apply time.
struct KvFlashMode4Plan {
    int  r = -1;
    bool run = false;
    int  reserve_chunks = -1;
};
inline KvFlashMode4Plan kvflash_mode4_plan(const std::vector<int32_t> & tokens, int kv_offset,
                                           int32_t im_start, int32_t user) {
    KvFlashMode4Plan p;
    if (tokens.empty()) return p;
    const int idx = kvflash_last_user_turn(tokens, im_start, user);
    p.r = kv_offset + (idx > 0 ? idx : 0);
    p.run = true;
    return p;
}

// Recall and compaction are active only when the pool is full: the whole
// prompt plus the room reserved for what follows must NOT fit.
inline bool kvflash_pool_fits(int end, int chunk_tokens, int reserve_chunks, int n_pool_blocks) {
    if (chunk_tokens <= 0) return false;
    return (end + chunk_tokens - 1) / chunk_tokens + reserve_chunks <= n_pool_blocks;
}

// Sorted ascending ids as ranges: {0,3,4,5,9} -> "0,3-5,9" ("" when empty).
inline std::string kvflash_format_ranges(const std::vector<int> & ids) {
    std::string out;
    for (size_t i = 0; i < ids.size();) {
        size_t j = i;
        while (j + 1 < ids.size() && ids[j + 1] == ids[j] + 1) j++;
        if (!out.empty()) out += ',';
        out += std::to_string(ids[i]);
        if (j > i) out += "-" + std::to_string(ids[j]);
        i = j + 1;
    }
    return out;
}

// Sorted copy of `cuts` with r inserted; unchanged if r < 0 or already present.
inline std::vector<int> kvflash_cuts_with(const std::vector<int> & cuts, int r) {
    std::vector<int> out = cuts;
    if (r >= 0 && std::find(out.begin(), out.end(), r) == out.end()) out.push_back(r);
    std::sort(out.begin(), out.end());
    return out;
}

// R-LRU recall: block b = chunks [cpb*b, cpb*b + cpb); its score is the sum of
// its chunk scores. Eligible blocks lie entirely in [first_chunk, first_window_chunk)
// and inside chunk_scores; the default first_chunk keeps v1's 10-chunk sink block
// (chunk 0 and its neighbours) out at every block size. Returns the top-k block
// ids, best first, ties newer-first.
inline std::vector<int> kvflash_recall_pick(const std::vector<float> & chunk_scores,
                                            int first_window_chunk, int k,
                                            int chunks_per_block = 10, int first_chunk = 10) {
    std::vector<int> ids;
    if (k <= 0 || chunks_per_block <= 0) return ids;
    const int limit = std::min(first_window_chunk, (int)chunk_scores.size());
    std::vector<double> bs;
    for (int b = std::max(1, (first_chunk + chunks_per_block - 1) / chunks_per_block);
         (b + 1) * chunks_per_block <= limit; b++) {
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

// LUCE_KVFLASH_DUMP_QK: the L2 norm of each (layer, head) plane of one query vector (planes x head_dim floats).
inline std::vector<float> kvflash_q_plane_norms(const float * q, size_t planes, size_t head_dim) {
    std::vector<float> n(planes);
    for (size_t p = 0; p < planes; p++) {
        double s = 0; for (size_t i = 0; i < head_dim; i++) s += (double)q[p * head_dim + i] * q[p * head_dim + i];
        n[p] = (float)std::sqrt(s);
    }
    return n;
}

// LUCE_KVFLASH_DUMP_SCORES=path: one JSON line per reselect, for diffing two engines' decisions. `resident` lists the
// chunk ids that had a pool block before the reselect; `scores` is the scorer's per-chunk vector before any promotion.
inline std::string kvflash_dump_line(int seq, int committed, int first_window, int unit, const std::vector<int> & picked,
                                     const std::vector<int> & resident, const std::vector<float> & scores) {
    auto ints = [](const std::vector<int> & v) { std::string o = "["; for (size_t i = 0; i < v.size(); i++) o += (i ? "," : "") + std::to_string(v[i]); return o + "]"; };
    std::string o = "{\"seq\":" + std::to_string(seq) + ",\"c\":" + std::to_string(committed) + ",\"fw\":" + std::to_string(first_window) +
                    ",\"u\":" + std::to_string(unit) + ",\"picked\":" + ints(picked) + ",\"res\":" + ints(resident) + ",\"sc\":[";
    char buf[32];
    for (size_t i = 0; i < scores.size(); i++) { std::snprintf(buf, sizeof buf, "%s%.9g", i ? "," : "", (double)scores[i]); o += buf; }
    return o + "]}";
}


// ---- LUCE_KVFLASH_PREAMBLE_ROWS: pin the system+tools preamble outside the history budget ----

// R (tokens) from the env; 0 = off (unset, non-numeric or negative).
inline int kvflash_preamble_rows_from_env() {
    const char * e = std::getenv("LUCE_KVFLASH_PREAMBLE_ROWS");
    if (!e || !*e) return 0;
    char * end = nullptr;
    const long v = std::strtol(e, &end, 10);
    return (*end || v <= 0 || v > 1L << 30) ? 0 : (int)v;
}

// R rounded up to a whole number of chunks (0 when off or the chunk size is unusable).
inline int kvflash_preamble_reserve_tokens(int rows, int chunk_tokens) {
    if (rows <= 0 || chunk_tokens <= 0) return 0;
    return (rows + chunk_tokens - 1) / chunk_tokens * chunk_tokens;
}

// Pool capacity with the preamble reserve on top of the history budget: rounded up to the 256-token
// stride kvflash_pool_from_env keeps (FA vec-kernel eligibility) and clamped to max_ctx like it. A pool
// of 0 (KVFlash off) and a reserve of 0 return pool_tokens untouched.
inline int kvflash_pool_with_preamble(int pool_tokens, int reserve_tokens, int max_ctx, int chunk_tokens) {
    if (pool_tokens <= 0 || reserve_tokens <= 0 || chunk_tokens <= 0) return pool_tokens;
    const int total = (pool_tokens + reserve_tokens + 255) / 256 * 256;
    return std::min(total, std::max(pool_tokens, max_ctx / 256 * 256));
}

// End of the preamble: when the prompt starts with `<|im_start|>system`, the index of the next
// `<|im_start|>` (the Qwen template puts the tool schemas inside the system block, so this is
// system + tools); otherwise 0 (no system block, an unknown id, or no turn after it).
inline int kvflash_preamble_end(const std::vector<int32_t> & ids, int32_t im_start, int32_t system_role) {
    if (im_start < 0 || system_role < 0 || ids.size() < 2 || ids[0] != im_start || ids[1] != system_role) return 0;
    for (size_t i = 2; i < ids.size(); i++)
        if (ids[i] == im_start) return (int)i;
    return 0;
}

// Why the preamble pin must stay off for the process ("" = ids present, on).
inline std::string kvflash_preamble_gate(int32_t im_start, int32_t system_role) {
    if (im_start < 0) return "<|im_start|> id missing";
    if (system_role < 0) return "system id missing";
    return "";
}

// Rows to pin: the preamble, never more than the reserve.
inline int kvflash_preamble_pin_rows(int preamble_end, int reserve_rows) {
    return preamble_end <= 0 || reserve_rows <= 0 ? 0 : std::min(preamble_end, reserve_rows);
}
inline bool kvflash_preamble_exceeds(int preamble_end, int reserve_rows) {
    return reserve_rows > 0 && preamble_end > reserve_rows;
}

// Chunks the pin covers (an upper bound on what the pager accepts).
inline int kvflash_pin_chunks(int preamble_end, int reserve_rows, int chunk_tokens) {
    const int rows = kvflash_preamble_pin_rows(preamble_end, reserve_rows);
    return rows <= 0 || chunk_tokens <= 0 ? 0 : (rows + chunk_tokens - 1) / chunk_tokens;
}

// Pins the preamble chunks of `pager` (anything with pin_range / is_pinned / chunk_tokens); returns
// the number of chunks actually pinned (the pager refuses a span that would deadlock eviction).
template <class Pager>
inline int kvflash_pin_preamble(Pager & pager, int preamble_end, int reserve_rows) {
    const int rows = kvflash_preamble_pin_rows(preamble_end, reserve_rows);
    const int ct = pager.chunk_tokens();
    if (rows <= 0 || ct <= 0) return 0;
    pager.pin_range(0, rows);
    int n = 0;
    for (int c = 0; c < (rows + ct - 1) / ct; c++) n += pager.is_pinned(c) ? 1 : 0;
    return n;
}

// Recall eligibility floor (chunks): v1's 10-chunk sink block, or past the pinned preamble.
inline int kvflash_recall_first_chunk(int pinned_chunks) { return std::max(10, pinned_chunks); }

// Chunks of exact tail window next to `recall_chunks` of recalled blocks and `reserve` chunks of room.
// `pin_reserve_chunks` (the preamble reserve, a pool capacity added on top at load) is taken off the
// pool first, so the history budget is the one the feature-off pool has.
inline int kvflash_recall_tail_chunks(int n_pool_blocks, int recall_chunks, int reserve, int pin_reserve_chunks) {
    return std::max(0, n_pool_blocks - pin_reserve_chunks - 1 - recall_chunks - reserve - 1);
}

// ---- per-slot recall plan (P2-01) ----
//
// The recall step of kvflash_prefill_apply_on, pure: how many chunks of exact tail, which blocks are
// picked, under which pin inputs. `pin_reserve_chunks` is pool capacity taken off the history budget
// (the serial path's LUCE_KVFLASH_PREAMBLE_ROWS capacity, or a slot's own accepted pins inside its
// slice); `pinned_chunks` are the leading chunks excluded from the candidates. `slot_pins` marks a
// slot's pins (k > 0): only then is the combined budget checked, so serial and k = 0 runs reproduce
// the pre-P1 numbers exactly (the old formula clamped the tail to zero and nothing else).
struct KvFlashRecallIn {
    int  n_pool_blocks = 0;      // Q: the slot's slice, or the serial pool
    int  cur_chunk = 0;
    int  recall_chunks = 0;      // chunks the recall asks for (10 x RECALL_BLOCKS)
    int  unit = 10;              // chunks per picked block
    int  reserve = 0;            // pool room kept for the replay
    int  pin_reserve_chunks = 0;
    int  pinned_chunks = 0;
    int  sink_chunks = 0;        // protected leading chunks (pager config)
    int  min_tail_chunks = 0;    // protected trailing chunks (pager config)
    bool slot_pins = false;
};
struct KvFlashRecallPlan {
    int tail = 0;
    int k_units = 0;
    int first_chunk = 10;
    int first_window = 0;
    bool clamped = false;        // k_units was reduced to fit
    bool fits = true;            // pins + reserve (+ sink + tail window) alone fit Q
    std::vector<int> picked;
};
inline KvFlashRecallPlan kvflash_recall_plan(const std::vector<float> & scores, const KvFlashRecallIn & in) {
    KvFlashRecallPlan r;
    r.k_units = in.unit > 0 ? in.recall_chunks / in.unit : 0;
    int recall = in.recall_chunks;
    if (in.slot_pins && in.pin_reserve_chunks > 0) {
        // sink + tail window + pins + reserve + picked <= Q, with the pager's two spare blocks (the +2 of
        // kvflash_recall_tail_chunks and of pin_range's guard). ponytail: sink and pins are added even though
        // pins overlap the sink when k >= sink, the pager's own guard does the same; the ceiling is
        // sink chunks of unused recall.
        const int room = in.n_pool_blocks - in.sink_chunks - in.min_tail_chunks - in.pin_reserve_chunks - in.reserve - 2;
        r.fits = room >= 0;
        const int max_units = in.unit > 0 ? std::max(0, room) / in.unit : 0;
        if (r.k_units > max_units) { r.k_units = max_units; r.clamped = true; }
        recall = r.k_units * in.unit;
    }
    r.first_chunk = kvflash_recall_first_chunk(in.pinned_chunks);
    r.tail = kvflash_recall_tail_chunks(in.n_pool_blocks, recall, in.reserve, in.pin_reserve_chunks);
    r.first_window = in.cur_chunk - r.tail;
    r.picked = kvflash_recall_pick(scores, r.first_window, r.k_units, in.unit, r.first_chunk);
    return r;
}

}  // namespace luce::common
