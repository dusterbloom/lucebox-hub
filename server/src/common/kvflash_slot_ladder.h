// Pure policy for the exact per-slot state ladder (Track C, C1.3-04).
// The recurrent (SSM + conv) state of a slot cannot be sliced back to an
// earlier position, so rewinding a KVFlash slot to P needs a state rung at P,
// or the newest rung C < P plus a replay of [C, P) through the same cuts.
// Device capture/restore of the slabs lives with the target cache; this header
// is the bookkeeping: which rungs a slot holds, who may evict what, and what a
// rewind has to do. No GPU, no IO.

#pragma once

#include <algorithm>
#include <cstdint>
#include <unordered_map>
#include <vector>

namespace luce::common {

// ── D1: deterministic cuts ─────────────────────────────────────────────
// The next prefill batch of a slot ends at the next multiple of U after `pos`,
// or at `end`. A function of position only, so idle, mixed and cancelled steps
// all produce the same boundaries and rungs always sit on them.
inline int kvflash_next_cut(int pos, int end, int cut_u) {
    const int next = (pos / cut_u + 1) * cut_u;
    return next < end ? next : end;
}
inline std::vector<int> kvflash_cut_sequence(int start, int end, int cut_u) {
    std::vector<int> out;
    for (int pos = start; pos < end;) { pos = kvflash_next_cut(pos, end, cut_u); out.push_back(pos); }
    return out;
}

// ── Rewind plan ────────────────────────────────────────────────────────
enum class KvFlashRewindKind { Exact, Replay, FromScratch };

struct KvFlashRewindPlan {
    KvFlashRewindKind kind = KvFlashRewindKind::FromScratch;
    int rung = -1;                       // rung to restore; -1 = zero state
    int replay_from = 0;                 // first position to recompute
    std::vector<int> replay_cuts;        // batch ends for [replay_from, target), on the D1 cuts
};

// ── The rung store ─────────────────────────────────────────────────────
// Keyed (slot, generation, position). Each slot holds at most `max_rungs`
// rungs: a capture beyond that evicts the slot's own least recently used rung,
// never the head rung, never the newest, and never another slot's. A request
// is a generation; rungs of any other generation neither restore nor capture.
class KvFlashSlotLadder {
public:
    explicit KvFlashSlotLadder(int max_rungs_per_slot) : max_rungs_(std::max(1, max_rungs_per_slot)) {}

    // New request on the slot: everything of the previous generation is dropped.
    void begin(int slot, uint64_t generation, int head_pos) {
        Slot & s = slots_[slot];
        s = Slot{};
        s.generation = generation;
        s.live = true;
        s.head = head_pos;
    }
    void release(int slot) { slots_.erase(slot); }

    // Returns the positions evicted to make room (empty when none or refused).
    std::vector<int> capture(int slot, uint64_t generation, int pos) {
        std::vector<int> evicted;
        Slot * s = live_(slot, generation);
        if (!s) return evicted;
        for (Rung & r : s->rungs) if (r.pos == pos) { r.tick = ++s->clock; return evicted; }
        s->rungs.push_back({pos, ++s->clock});
        while ((int)s->rungs.size() > max_rungs_) {
            const uint64_t newest = newest_tick_(*s);
            auto victim = s->rungs.end();
            for (auto it = s->rungs.begin(); it != s->rungs.end(); ++it) {
                if (it->pos == s->head || it->tick == newest) continue;
                if (victim == s->rungs.end() || it->tick < victim->tick) victim = it;
            }
            if (victim == s->rungs.end()) break;     // only head and newest remain: over the cap rather than losing either
            evicted.push_back(victim->pos);
            s->rungs.erase(victim);
        }
        return evicted;
    }

    // Greatest rung <= pos, or -1. Using a rung refreshes it.
    int pick(int slot, uint64_t generation, int pos) {
        Slot * s = live_(slot, generation);
        if (!s) return -1;
        Rung * best = nullptr;
        for (Rung & r : s->rungs) if (r.pos <= pos && (!best || r.pos > best->pos)) best = &r;
        if (!best) return -1;
        best->tick = ++s->clock;
        return best->pos;
    }
    bool has(int slot, uint64_t generation, int pos) const {
        const Slot * s = live_(slot, generation);
        if (!s) return false;
        for (const Rung & r : s->rungs) if (r.pos == pos) return true;
        return false;
    }
    std::vector<int> positions(int slot) const {
        std::vector<int> out;
        const auto it = slots_.find(slot);
        if (it == slots_.end()) return out;
        for (const Rung & r : it->second.rungs) out.push_back(r.pos);
        std::sort(out.begin(), out.end());
        return out;
    }

    // What a rewind of the slot to `target` has to do. Read-only.
    KvFlashRewindPlan plan_rewind(int slot, uint64_t generation, int target, int cut_u) const {
        KvFlashRewindPlan p;
        int best = -1;
        if (const Slot * s = live_(slot, generation))
            for (const Rung & r : s->rungs) if (r.pos <= target && r.pos > best) best = r.pos;
        p.rung = best;
        p.replay_from = best < 0 ? 0 : best;
        p.kind = best == target ? KvFlashRewindKind::Exact
               : best < 0       ? KvFlashRewindKind::FromScratch
                                : KvFlashRewindKind::Replay;
        p.replay_cuts = kvflash_cut_sequence(p.replay_from, target, cut_u);
        return p;
    }
    // The rewind happened: the abandoned future's rungs are gone.
    void commit_rewind(int slot, uint64_t generation, int target) {
        Slot * s = live_(slot, generation);
        if (!s) return;
        s->rungs.erase(std::remove_if(s->rungs.begin(), s->rungs.end(),
                                      [&](const Rung & r) { return r.pos > target; }), s->rungs.end());
    }

private:
    struct Rung { int pos; uint64_t tick; };
    struct Slot { uint64_t generation = 0; bool live = false; int head = -1; uint64_t clock = 0; std::vector<Rung> rungs; };

    Slot * live_(int slot, uint64_t generation) {
        const auto it = slots_.find(slot);
        return it != slots_.end() && it->second.live && it->second.generation == generation ? &it->second : nullptr;
    }
    const Slot * live_(int slot, uint64_t generation) const {
        const auto it = slots_.find(slot);
        return it != slots_.end() && it->second.live && it->second.generation == generation ? &it->second : nullptr;
    }
    static uint64_t newest_tick_(const Slot & s) {
        uint64_t t = 0;
        for (const Rung & r : s.rungs) t = std::max(t, r.tick);
        return t;
    }

    int max_rungs_;
    std::unordered_map<int, Slot> slots_;
};

}  // namespace luce::common
