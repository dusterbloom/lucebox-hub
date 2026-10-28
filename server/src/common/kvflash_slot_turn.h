// KVFlash C1.4: per-slot mode-4 turn bookkeeping and session matching, pure.
//
// A slot's turn is the single-slot mode 4 (qwen35_backend.cpp) over the step
// engine: a first pass over [kv_offset, end) under LRU with the state snapshot
// at R (start of the last user turn) and the last T prompt queries captured,
// then score, rewind to R, apply the picks, replay [R, end). The engine keeps
// one KvFlashSlotTurn per slot and asks these helpers what the next cut is and
// whether the step that runs it snapshots or captures.
#pragma once
#include <cstdlib>

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
    int  ckpt = -1;                 // the prompt-end checkpoint: the last <|im_start|> (the assistant header)
    bool end_rung = false;          // C1.5 F7: a recall turn whose replay stops at ckpt and checkpoints there too
    int  apply_at = -1;             // mode 5: the cut at which the drafter's picks are applied (one pass, no snapshot at R); -1: none
    int  capture_at = -1;           // sessions P1b: the cut at which the preamble entry is captured (64k); -1: none
    std::vector<int> cuts;          // first-pass cut ends, ascending
    bool recall() const { return phase != KvFlashTurnPhase::Lru; }
    bool first_pass() const { return phase == KvFlashTurnPhase::FirstPass; }
    bool snapshot_at_r() const { return recall() && r > kv_offset; }
    // The pass that checkpoints at ckpt: an LRU turn's only pass, a recall turn's replay with the end rung.
    bool ckpt_pass() const { return !recall() || (end_rung && phase == KvFlashTurnPhase::Replay); }
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
// end_rung_min (LUCE_KVFLASH_END_RUNG_MIN): a recall turn also checkpoints at the
// header when end - r reaches it; < 0: never (the R rung alone, C1.4).
// pflash2 (mode 5): the picks come from the drafter's scores of the prompt ids, so the turn stays one
// pass (the LRU turn's checkpoint rule) and only records where to apply them (apply_at = R).
// capture_at (sessions P1b): a cut the turn must stop at so the engine can capture the preamble entry there;
// honoured inside (kv_offset, end) and, in a recall turn, only before R (the recall is planned around it).
inline KvFlashSlotTurn kvflash_slot_turn_plan(const std::vector<int32_t> & prompt, int kv_offset,
                                              int32_t im_start, int32_t user, int chunk_tokens,
                                              int n_pool_blocks, int cut_u, int n_queries, bool profile_on,
                                              int end_rung_min = -1, bool pflash2 = false, int capture_at = -1) {
    KvFlashSlotTurn t;
    t.kv_offset = std::clamp(kv_offset, 0, (int)prompt.size());
    t.end = (int)prompt.size();
    if (capture_at > t.kv_offset && capture_at < t.end) t.capture_at = capture_at;
    // The next prompt re-renders everything before the generation prompt's assistant header
    // unchanged (the reply after it is re-tokenized from text): the sticky checkpoint is taken there.
    for (int i = t.end - 1; i > t.kv_offset; --i) if (prompt[(size_t)i] == im_start) { t.ckpt = i; break; }
    if (!profile_on || chunk_tokens <= 0 || t.end <= t.kv_offset) return t;
    const std::vector<int32_t> suffix(prompt.begin() + t.kv_offset, prompt.end());
    const KvFlashMode4Plan p4 = kvflash_mode4_plan(suffix, t.kv_offset, im_start, user);
    if (!p4.run) return t;
    t.r = p4.r;
    t.reserve_chunks = p4.reserve_chunks >= 0 ? p4.reserve_chunks : (t.end - t.r + chunk_tokens - 1) / chunk_tokens + 1;
    // fits, or no sealed chunk before r to score: LRU turn
    if (kvflash_pool_fits(t.end, chunk_tokens, t.reserve_chunks, n_pool_blocks) || t.r / chunk_tokens == 0) { t.r = -1; t.reserve_chunks = 0; return t; }
    if (t.capture_at > t.r) t.capture_at = -1;   // a recall turn: only a cut before R
    if (pflash2) { t.apply_at = t.r; return t; }
    t.phase = KvFlashTurnPhase::FirstPass;
    t.cuts = kvflash_slot_turn_cuts(t.kv_offset, t.r, t.end, cut_u, n_queries);
    if (t.capture_at > 0 && std::find(t.cuts.begin(), t.cuts.end(), t.capture_at) == t.cuts.end())
        t.cuts.insert(std::upper_bound(t.cuts.begin(), t.cuts.end(), t.capture_at), t.capture_at);
    t.end_rung = end_rung_min >= 0 && t.ckpt > t.r && t.end - t.r >= end_rung_min;
    return t;
}

// Chunks the slot's next step allocates: what [cur_pos, cur_pos + n_next) needs
// past the chunks the pager already holds. The step driver settles to this.
inline int kvflash_slot_need_chunks(int cur_pos, int n_next, int chunk_tokens, int n_chunks) {
    if (chunk_tokens <= 0 || n_next <= 0) return 0;
    return std::max(0, (cur_pos + n_next + chunk_tokens - 1) / chunk_tokens - n_chunks);
}

// Where a sticky session checkpoints this turn: a recall turn keeps its R snapshot (the
// first-pass state at R, what a cold turn has there: the replay's state, shaped by the picks,
// drifted the next turn's queries and cost recall), an LRU turn the state at the assistant
// header from its only pass. -1: nothing to keep from this turn.
inline int kvflash_sticky_ckpt_pos(const KvFlashSlotTurn & t) { return t.recall() ? t.r : t.ckpt; }

// Tokens of the next cut from cur_pos: the planned cut in the first pass (its
// last cut carries the captured queries), a plain U cut in the replay, a plain U
// cut stopping at the checkpoint in the pass that takes it. 0 at or past the end.
// tail (G-D): the cut that would end the turn stops `tail` tokens short, so the last prompt tokens are a
// cut of their own and the engine can stash the state before them (the first pass keeps its capture cut
// whole: a replay or an LRU pass always ends the turn). 0: off.
inline int kvflash_slot_turn_next_cut(const KvFlashSlotTurn & t, int cur_pos, int cut_u, int tail = 0) {
    if (cur_pos >= t.end) return 0;
    if (t.first_pass())
        for (int c : t.cuts) if (c > cur_pos) return c - cur_pos;
    int n = kvflash_next_cut(cur_pos, t.end, cut_u) - cur_pos;
    if (t.apply_at > cur_pos) n = std::min(n, t.apply_at - cur_pos);
    if (t.capture_at > cur_pos) n = std::min(n, t.capture_at - cur_pos);
    if (t.ckpt_pass() && t.ckpt > cur_pos) n = std::min(n, t.ckpt - cur_pos);
    if (tail > 0 && n > tail && cur_pos + n == t.end) n -= tail;
    return n;
}

// Mode 5: the prefill reached the point where the drafter's picks are applied (the engine then clears apply_at).
inline bool kvflash_slot_turn_apply_due(const KvFlashSlotTurn & t, int cur_pos) {
    return t.apply_at >= 0 && cur_pos >= t.apply_at;
}

// What the step that runs the cut [cur_pos, cur_pos + n) must do for this slot.
struct KvFlashTurnStep {
    bool snapshot = false;      // after the graph: capture the slot's recurrent state (it is the R state)
    bool capture = false;       // build the graph with q capture ending at this slot's segment (the first pass's last cut)
    int  q_rows = 0;            // first pass: how many of this cut's last rows are among the last n_queries prompt tokens (to capture)
    bool ckpt = false;          // after the graph: the slot's recurrent state is the sticky checkpoint at the header
    bool preamble = false;      // after the graph: the slot holds exactly the preamble prefix (the capture cut)
};
inline KvFlashTurnStep kvflash_slot_turn_step(const KvFlashSlotTurn & t, int cur_pos, int n, int n_queries = 0) {
    KvFlashTurnStep s;
    if (n <= 0) return s;
    s.ckpt = t.ckpt_pass() && t.ckpt > 0 && cur_pos + n == t.ckpt;
    s.preamble = t.capture_at > 0 && cur_pos + n == t.capture_at;
    if (!t.first_pass()) return s;
    s.snapshot = t.snapshot_at_r() && cur_pos + n == t.r;
    s.capture = cur_pos + n == t.end;
    s.q_rows = n_queries > 0 ? std::clamp(cur_pos + n - (t.end - n_queries), 0, n) : 0;
    return s;
}

// The serial engine's selector-v2 queries: the last `keep` prompt queries over all the first pass's batches,
// oldest first (qwen35_backend.cpp kvflash_q_tail_). Append one cut's rows (n_rows of row_size floats) and trim.
inline void kvflash_q_tail_push(std::vector<float> & tail, const float * rows, int n_rows, size_t row_size, int keep) {
    tail.insert(tail.end(), rows, rows + (size_t)n_rows * row_size);
    const size_t cap = (size_t)std::max(keep, 0) * row_size;
    if (tail.size() > cap) tail.erase(tail.begin(), tail.end() - cap);
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

// C1.5 F1: a session keeps two rungs, `ckpt` (its prompt-end checkpoint: the state before the
// reply, which the client re-renders and may re-tokenize differently) and `trunk` below it (the
// checkpoint before that: where a branch forked). Either may be absent (<= 0).
struct KvFlashRungs { int ckpt = -1, trunk = -1; };

// Rule B. Where a parked session of `len` tokens resumes for a prompt sharing `lcp` tokens with
// its history: the highest of len, ckpt and trunk the prompt covers; -1 when none (cold). A
// resume always leaves new tokens to prefill: the position must be below prompt_len.
inline int kvflash_resume_point(int lcp, int len, int ckpt, int trunk, int prompt_len) {
    int best = -1;
    for (int p : {len, ckpt, trunk}) if (p > 0 && lcp >= p && p < prompt_len) best = std::max(best, p);
    return best;
}

// G-D: where a session of `len` fed tokens parks. The state after the last fed token has no
// logits for a next token and a hybrid model cannot step back from it, so a prompt equal to the
// history could only resume at a rung. A slot that kept an earlier state (stash_pos: the token count
// it covers, at most `tail` tokens back) parks there, short of len: the history is stash_pos tokens
// and an identical prompt resumes at stash_pos and prefills the rest (len - stash_pos tokens, the
// reselect's queries). Never below a rung (the rung would be orphaned) and never at 0.
inline int kvflash_park_point(int len, int stash_pos, int ckpt, int trunk, int tail = 1) {
    return stash_pos > 0 && stash_pos < len && stash_pos >= len - tail && stash_pos >= ckpt && stash_pos >= trunk ? stash_pos : len;
}

// The park-short switches (env strings, null when unset). LUCE_KVFLASH_PARK_SHORT=0: everything off. The prefill-only
// tail stash (one copy per turn) is on by default with tail 4 (R's replay window); LUCE_KVFLASH_PARK_DECODE=1 adds the
// per-step decode stash (a copy before every decode step, ~4-5% decode), off by default.
struct KvFlashParkCfg { bool tail_on = false; bool decode_on = false; int tail = 4; bool any() const { return tail_on || decode_on; } };
inline KvFlashParkCfg kvflash_park_cfg(bool sticky, const char * park_short, const char * tail, const char * decode) {
    KvFlashParkCfg c;
    const bool off = park_short && park_short[0] == '0' && !park_short[1];
    c.tail_on = sticky && !off;
    c.decode_on = c.tail_on && decode && decode[0] == '1' && !decode[1];
    if (tail && *tail) { const int n = std::atoi(tail); c.tail = n < 1 ? 1 : n > 64 ? 64 : n; }
    return c;
}

// Rule B's bookkeeping. The rungs a session keeps when it resumes at L: the rung at L (or the
// highest below it) is the fallback `ckpt`, a rung below that the `trunk`; rungs above L are gone
// with their rows. At L == len both stay.
inline KvFlashRungs kvflash_rungs_after_resume(int ckpt, int trunk, int L) {
    KvFlashRungs r;
    if (ckpt > 0 && ckpt <= L) { r.ckpt = ckpt; if (trunk > 0 && trunk < ckpt) r.trunk = trunk; }
    else if (trunk > 0 && trunk <= L) r.ckpt = trunk;
    return r;
}

// Rule A. The rungs after a checkpoint at p: the checkpoint below p becomes the trunk (the old
// trunk goes), one at p is replaced (the trunk stays); a trunk at or above p goes.
inline KvFlashRungs kvflash_rungs_after_ckpt(int ckpt, int trunk, int p) {
    KvFlashRungs r;
    r.ckpt = p;
    if (ckpt > 0 && ckpt < p) r.trunk = ckpt;
    else if (trunk > 0 && trunk < p) r.trunk = trunk;
    return r;
}

// The oldest parked session (the one to drop when a slot is needed); -1 when none.
inline int kvflash_oldest_session(const std::vector<KvFlashParkedSession> & parked) {
    int slot = -1;
    uint64_t tick = 0;
    for (const KvFlashParkedSession & s : parked)
        if (s.slot >= 0 && (slot < 0 || s.tick < tick)) { slot = s.slot; tick = s.tick; }
    return slot;
}

// Frees the oldest parked session other than `keep`: `to_disk(slot)` is the session's move to disk (false: its park
// was refused, it stays), `drop(slot)` forgets it (its next turn prefills cold). A refused park is dropped, not skipped:
// the parked set is a cache and never blocks a request. -1 only when nothing is parked.
struct KvFlashEvicted { int slot = -1; bool dropped = false; };
template <class ToDisk, class Drop>
inline KvFlashEvicted kvflash_evict_oldest(const std::vector<KvFlashParkedSession> & parked, ToDisk to_disk, Drop drop, int keep = -1) {
    std::vector<KvFlashParkedSession> list;
    for (const KvFlashParkedSession & s : parked) if (s.slot != keep) list.push_back(s);
    KvFlashEvicted r;
    r.slot = kvflash_oldest_session(list);
    if (r.slot < 0) return r;
    if (!to_disk(r.slot)) { drop(r.slot); r.dropped = true; }
    return r;
}

// Restoring a session from disk reserves host bytes like any admit. Busy (the ledger lacks room) evicts the oldest
// parked session and tries again; Refused (the bundle or the pager cannot take it) is final. Returns Busy only
// when `evict()` has nothing left to free.
enum class KvFlashAdopt { Ok, Busy, Refused };
// `release()` runs before every eviction: the returning bundle's memory is freed (the retry re-reads it) so no allocation the
// ledger has not charged is live while a victim is staged.
template <class TryAdopt, class Release, class Evict>
inline KvFlashAdopt kvflash_adopt_with_evict(TryAdopt try_adopt, Release release, Evict evict) {
    for (;;) {
        const KvFlashAdopt r = try_adopt();
        if (r != KvFlashAdopt::Busy) return r;
        release();
        if (!evict()) return r;
    }
}
template <class TryAdopt, class Evict>
inline KvFlashAdopt kvflash_adopt_with_evict(TryAdopt try_adopt, Evict evict) {
    return kvflash_adopt_with_evict(try_adopt, [] {}, evict);
}

// A returning session's restore from disk: `probe()` answers from the ledger alone whether the host reservation fits
// (Busy = it does not; anything else = cannot decide or fits), `load()` reads the bundle (false: unreadable), `try_adopt`
// reserves and installs it, `release` frees the loaded bundle before a victim is staged, `evict_one` frees the oldest
// parked session (false: nothing left). `load_failed` says the bundle could not be read (first load or the reload).
struct KvFlashRestoreResult { KvFlashAdopt adopt = KvFlashAdopt::Refused; bool load_failed = false; };
template <class Probe, class HasVictim, class Validate, class Load, class TryAdopt, class Release, class EvictOne>
inline KvFlashRestoreResult kvflash_restore_adopt(Probe probe, HasVictim has_victim, Validate validate, Load load, TryAdopt try_adopt, Release release, EvictOne evict_one, bool evict_for_room) {
    (void)has_victim; (void)validate;
    // The ledger first: a reservation that does not fit is Busy without a read, or one eviction at a time until it fits.
    while (probe() == KvFlashAdopt::Busy)
        if (!evict_for_room || !evict_one()) return {KvFlashAdopt::Busy, false};
    if (!load()) return {KvFlashAdopt::Refused, true};
    bool reload_failed = false;
    const KvFlashAdopt got = kvflash_adopt_with_evict(try_adopt, release, [&] {
        if (!evict_for_room || !evict_one()) return false;
        if (!load()) { reload_failed = true; return false; }
        return true;
    });
    return {got, reload_failed};
}

}  // namespace luce::common
