// KVFlash C1.4: per-slot mode-4 turn bookkeeping and session matching, pure.
//
// A slot's turn is the single-slot mode 4 (qwen35_backend.cpp) over the step
// engine: a first pass over [kv_offset, end) under LRU with the state snapshot
// at R (start of the last user turn) and the last T prompt queries captured,
// then score, rewind to R, apply the picks, replay [R, end). The engine keeps
// one KvFlashSlotTurn per slot and asks these helpers what the next cut is and
// whether the step that runs it snapshots or captures.
#pragma once

#include "kvflash_prefill_reselect.h"
#include "kvflash_slot_ladder.h"

#include <algorithm>
#include <cstdint>
#include <vector>

namespace luce::common {

enum class KvFlashTurnPhase : uint8_t { Lru, FirstPass, Replay };

struct KvFlashSlotTurn {
    KvFlashTurnPhase phase = KvFlashTurnPhase::Lru;
    int  kv_offset = 0;             // resume point: tokens [0, kv_offset) are the session
    int  r = -1;                    // rewind point
    int  end = 0;                   // prompt end
    int  reserve_chunks = 0;        // pool room kept for the replay (single-slot rule)
    std::vector<int> cuts;          // first-pass cut ends, ascending
    bool recall() const { return phase != KvFlashTurnPhase::Lru; }
    bool first_pass() const { return phase == KvFlashTurnPhase::FirstPass; }
    bool snapshot_at_r() const { return recall() && r > kv_offset; }
};

// First-pass cut ends over [kv_offset, end): r is a cut end (the state snapshot
// is taken there), cuts are U long after it, and the last one is at least
// min(T, end - r) long (its predecessor shrinks) so one step captures all T queries.
inline std::vector<int> kvflash_slot_turn_cuts(int kv_offset, int r, int end, int cut_u, int n_queries) {
    std::vector<int> cuts;
    if (cut_u <= 0 || end <= kv_offset) return cuts;
    r = std::clamp(r, kv_offset, end);
    for (int p = kv_offset; p < r;) { p = std::min(p + cut_u, r); cuts.push_back(p); }
    const size_t tail0 = cuts.size();
    for (int p = r; p < end;) { p = std::min(p + cut_u, end); cuts.push_back(p); }
    const int need = std::min(std::max(n_queries, 1), end - r);
    if (cuts.size() - tail0 >= 2 && cuts.back() - cuts[cuts.size() - 2] < need) cuts[cuts.size() - 2] = end - need;
    return cuts;
}

// The slot's turn for a prompt resumed at kv_offset. Mode 4 runs when the recall
// profile is on and the prompt with the replay's reserve does not fit the slot's
// pool (kvflash_pool_fits, the single-slot rule); otherwise the slot stays LRU.
inline KvFlashSlotTurn kvflash_slot_turn_plan(const std::vector<int32_t> & prompt, int kv_offset,
                                              int32_t im_start, int32_t user, int chunk_tokens,
                                              int n_pool_blocks, int cut_u, int n_queries, bool profile_on) {
    KvFlashSlotTurn t;
    t.kv_offset = std::clamp(kv_offset, 0, (int)prompt.size());
    t.end = (int)prompt.size();
    if (!profile_on || chunk_tokens <= 0 || t.end <= t.kv_offset) return t;
    const std::vector<int32_t> suffix(prompt.begin() + t.kv_offset, prompt.end());
    const KvFlashMode4Plan p4 = kvflash_mode4_plan(suffix, t.kv_offset, im_start, user);
    if (!p4.run) return t;
    t.r = p4.r;
    t.reserve_chunks = p4.reserve_chunks >= 0 ? p4.reserve_chunks : (t.end - t.r + chunk_tokens - 1) / chunk_tokens + 1;
    // fits, or no sealed chunk before r to score: LRU turn
    if (kvflash_pool_fits(t.end, chunk_tokens, t.reserve_chunks, n_pool_blocks) || t.r / chunk_tokens == 0) { t.r = -1; t.reserve_chunks = 0; return t; }
    t.phase = KvFlashTurnPhase::FirstPass;
    t.cuts = kvflash_slot_turn_cuts(t.kv_offset, t.r, t.end, cut_u, n_queries);
    return t;
}

// Chunks the slot's next step allocates: what [cur_pos, cur_pos + n_next) needs
// past the chunks the pager already holds. The step driver settles to this.
inline int kvflash_slot_need_chunks(int cur_pos, int n_next, int chunk_tokens, int n_chunks) {
    if (chunk_tokens <= 0 || n_next <= 0) return 0;
    return std::max(0, (cur_pos + n_next + chunk_tokens - 1) / chunk_tokens - n_chunks);
}

// Tokens of the next cut from cur_pos: the planned cut in the first pass, a
// plain U cut in the replay or an LRU turn. 0 at or past the end.
inline int kvflash_slot_turn_next_cut(const KvFlashSlotTurn & t, int cur_pos, int cut_u) {
    if (cur_pos >= t.end) return 0;
    if (t.first_pass())
        for (int c : t.cuts) if (c > cur_pos) return c - cur_pos;
    return kvflash_next_cut(cur_pos, t.end, cut_u) - cur_pos;
}

// What the step that runs the cut [cur_pos, cur_pos + n) must do for this slot.
struct KvFlashTurnStep {
    bool snapshot = false;      // after the graph: capture the slot's recurrent state (it is the R state)
    bool capture = false;       // build the graph with q capture ending at this slot's segment
};
inline KvFlashTurnStep kvflash_slot_turn_step(const KvFlashSlotTurn & t, int cur_pos, int n) {
    KvFlashTurnStep s;
    if (!t.first_pass() || n <= 0) return s;
    s.snapshot = t.snapshot_at_r() && cur_pos + n == t.r;
    s.capture = cur_pos + n == t.end;
    return s;
}

// Sticky sessions: a parked slot and the tokens it holds (history.size() == its cur_pos).
struct KvFlashParkedSession {
    int slot = -1;
    const std::vector<int32_t> * history = nullptr;
    uint64_t tick = 0;          // park order; newest wins a tie
};
struct KvFlashSessionPick {
    int slot = -1;
    int lcp = 0;
    bool exact() const { return slot >= 0; }
};

// The parked session with the longest common prefix with `prompt` (at least
// min_lcp tokens, ties to the newest); slot -1 when none qualifies.
inline KvFlashSessionPick kvflash_pick_session(const std::vector<KvFlashParkedSession> & parked,
                                               const std::vector<int32_t> & prompt, int min_lcp) {
    KvFlashSessionPick best;
    uint64_t best_tick = 0;
    for (const KvFlashParkedSession & s : parked) {
        if (s.slot < 0 || !s.history) continue;
        const size_t cap = std::min(prompt.size(), s.history->size());
        size_t l = 0;
        while (l < cap && prompt[l] == (*s.history)[l]) l++;
        if ((int)l < std::max(min_lcp, 1)) continue;
        if ((int)l > best.lcp || ((int)l == best.lcp && s.tick > best_tick)) { best = {s.slot, (int)l}; best_tick = s.tick; }
    }
    return best;
}

// The oldest parked session (the one to drop when a slot is needed); -1 when none.
inline int kvflash_oldest_session(const std::vector<KvFlashParkedSession> & parked) {
    int slot = -1;
    uint64_t tick = 0;
    for (const KvFlashParkedSession & s : parked)
        if (s.slot >= 0 && (slot < 0 || s.tick < tick)) { slot = s.slot; tick = s.tick; }
    return slot;
}

}  // namespace luce::common
