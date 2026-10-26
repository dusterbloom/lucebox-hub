// KVFlash 2.0 sessions P1a: the preamble store core (SESSIONS_DESIGN.md 3.2), pure.
//
// The shared system+tools prefix of an agent harness, held on the host as chunk-aligned KV plus the recurrent
// slab at its end, keyed by a chain of chunk-prefix hashes. This header has the decisions and the bookkeeping
// only: hashing, the boundary rule, the capture bound, refcount + LRU within a byte budget, and the re-pin
// helper. The KV and slab bytes, the attach and the capture cut live in the engine (P1b).
#pragma once

#include <algorithm>
#include <cstdint>
#include <numeric>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

namespace luce::common {

// The grid E_c and k live on: whole chunks, and with `slice` > 0 whole prefill slices too (kKvFlashCutU in the engine).
// A cut off the slice grid changes the prefill's segment shapes and so its float rounding; on the grid the legacy prefill,
// the capture cut and an attach are the same computation.
inline int kvflash_preamble_unit(int ct, int slice) {
    return ct <= 0 ? 1 : slice > ct ? std::lcm(ct, slice) : ct;
}

// E_c = floor(E_role / unit) * unit: the pin boundary an entry can have never exceeds the role boundary.
inline int kvflash_preamble_ec(int e_role, int ct, int slice = 0) {
    const int u = kvflash_preamble_unit(ct, slice);
    return e_role <= 0 || ct <= 0 ? 0 : (e_role / u) * u;
}

// H_k = fnv1a continued from H_{k-1} over ids[ct (k-1), ct k)) for k = 1 .. n_chunks (clamped to the whole chunks `ids` holds).
// Entry i of the result is H_{i+1}.
inline std::vector<uint64_t> kvflash_preamble_hashes(const std::vector<int32_t> & ids, int n_chunks, int ct) {
    std::vector<uint64_t> h;
    if (ct <= 0 || n_chunks <= 0) return h;
    const int n = std::min(n_chunks, (int)(ids.size() / (size_t)ct));
    uint64_t acc = 1469598103934665603ULL;
    for (int k = 0; k < n; k++) {
        for (int i = k * ct; i < (k + 1) * ct; i++) {
            const uint32_t t = (uint32_t)ids[(size_t)i];
            for (int b = 0; b < 4; b++) { acc ^= (t >> (8 * b)) & 0xff; acc *= 1099511628211ULL; }
        }
        h.push_back(acc);
    }
    return h;
}

// The longest chunk-aligned prefix two token sequences share, in chunks, looking at no more than `limit_tokens`.
inline int kvflash_preamble_shared_chunks(const std::vector<int32_t> & a, const std::vector<int32_t> & b, int ct,
                                          int limit_tokens) {
    if (ct <= 0) return 0;
    const size_t n = std::min({a.size(), b.size(), (size_t)std::max(0, limit_tokens)});
    size_t i = 0;
    while (i < n && a[i] == b[i]) i++;
    return (int)(i / (size_t)ct);
}

// The largest k a slot may pin or attach (P2-04), 0 when none. Two limits:
//  - pool: sink + tail_window + k + reserve_max + recall_chunks + 2 <= Q. The +2 is pin_range's own guard
//    (kvflash_pager.h, "would deadlock eviction"), which counts sink and pins additively; the recall term is
//    kvflash_recall_plan's budget, so a pinned preamble never starves the recall.
//  - host: k * chunk_bytes + slab_bytes <= store_free_bytes (free after protected entries).
inline int kvflash_capture_bound(int Q, int sink, int tail_window, int reserve_max, int recall_chunks,
                                 int64_t store_free_bytes, int64_t slab_bytes, int64_t chunk_bytes) {
    const int pool = Q - sink - tail_window - reserve_max - recall_chunks - 2;
    if (pool <= 0 || chunk_bytes <= 0 || store_free_bytes < slab_bytes) return 0;
    const int64_t host = (store_free_bytes - slab_bytes) / chunk_bytes;
    return (int)std::min<int64_t>(pool, host);
}

// Refcounted, LRU-evicted entries within a byte budget. Entries are keyed by (H_k, exec_id): an entry is only
// ever attached under the execution identity that captured it. Not thread-safe: the engine thread owns it.
// ponytail: an entry is charged k * chunk_bytes + slab_bytes even when its chunk records are shared with a
// shorter entry, so the budget overcounts shared prefixes by at most (entries - 1) * chunk records.
class KvFlashPreambleStore {
public:
    struct Entry {
        int k = 0;
        std::string exec_id;
        std::vector<uint64_t> path;   // H_1 .. H_k
        int refcount = 0;
        uint64_t last_used = 0;
        int64_t bytes = 0;
        // The bytes (engine-filled after the capture cut, or loaded from preambles/): the pager's blob header at
        // k chunks, the k chunk payloads back to back, the packed DeltaNet/conv slab at 64k.
        std::vector<uint8_t> hdr, chunks, slab;
    };
    struct Evicted { uint64_t hash; std::string exec_id; };
    enum Capture { Stored, Exists, Declined };

    KvFlashPreambleStore(int64_t budget_bytes, int64_t slab_bytes, int64_t chunk_bytes)
        : budget_(budget_bytes), slab_(slab_bytes), chunk_(chunk_bytes) {}

    int64_t budget_bytes() const { return budget_; }
    int64_t used_bytes() const { return used_; }
    int count() const { return (int)entries_.size(); }
    int declined() const { return declined_; }
    int64_t entry_bytes(int k) const { return (int64_t)k * chunk_ + slab_; }
    int64_t protected_bytes() const {
        int64_t b = 0;
        for (const auto & kv : entries_) if (kv.second.refcount > 0) b += kv.second.bytes;
        return b;
    }
    // Room a new capture can have: everything but the protected entries (the unprotected ones can be evicted).
    int64_t free_for_capture() const { return std::max<int64_t>(0, budget_ - protected_bytes()); }

    const Entry * find(uint64_t hash, const std::string & exec_id) const {
        const auto it = entries_.find(key_(hash, exec_id));
        return it == entries_.end() ? nullptr : &it->second;
    }
    // Largest k <= n with an entry under exec_id (path[i] = H_{i+1}); 0 when none.
    int largest_hit(const std::vector<uint64_t> & path, const std::string & exec_id, int max_k) const {
        for (int k = std::min<int>(max_k, (int)path.size()); k >= 1; k--)
            if (find(path[(size_t)k - 1], exec_id)) return k;
        return 0;
    }
    // Chunks `path` shares with the best entry under exec_id (the partial-hit length).
    int shared_with_entries(const std::vector<uint64_t> & path, const std::string & exec_id) const {
        int best = 0;
        for (const auto & kv : entries_) {
            if (kv.second.exec_id != exec_id) continue;
            int j = 0;
            while (j < (int)path.size() && j < kv.second.k && path[(size_t)j] == kv.second.path[(size_t)j]) j++;
            best = std::max(best, j);
        }
        return best;
    }

    // Stores the entry for `path` (k = path.size()). Evicts least-recently-used unreferenced entries to make room;
    // when the protected entries leave no room, nothing is evicted and the capture is declined.
    Capture capture(const std::vector<uint64_t> & path, const std::string & exec_id) {
        if (path.empty()) return Declined;
        const uint64_t key = key_(path.back(), exec_id);
        if (entries_.count(key)) return Exists;
        const int64_t need = entry_bytes((int)path.size());
        if (need > free_for_capture()) { declined_++; return Declined; }
        while (used_ + need > budget_ && evict_lru_()) {}
        Entry e;
        e.k = (int)path.size(); e.exec_id = exec_id; e.path = path; e.bytes = need; e.last_used = ++clock_;
        entries_.emplace(key, std::move(e));
        used_ += need;
        return Stored;
    }
    // The entry's bytes; false when there is no such entry under exec_id.
    bool set_payload(uint64_t hash, const std::string & exec_id, std::vector<uint8_t> hdr, std::vector<uint8_t> chunks,
                     std::vector<uint8_t> slab) {
        const auto it = entries_.find(key_(hash, exec_id));
        if (it == entries_.end() || it->second.exec_id != exec_id) return false;
        it->second.hdr = std::move(hdr); it->second.chunks = std::move(chunks); it->second.slab = std::move(slab);
        return true;
    }
    // Removes an unprotected entry (a capture whose bytes could not be read); false when absent or protected.
    bool erase(uint64_t hash, const std::string & exec_id) {
        const auto it = entries_.find(key_(hash, exec_id));
        if (it == entries_.end() || it->second.refcount > 0) return false;
        used_ -= it->second.bytes;
        entries_.erase(it);
        return true;
    }
    // The entries the LRU evicted since the last call (the engine removes their files).
    std::vector<Evicted> take_evicted() { return std::exchange(evicted_, {}); }
    void touch(uint64_t hash, const std::string & exec_id) {
        const auto it = entries_.find(key_(hash, exec_id));
        if (it != entries_.end()) it->second.last_used = ++clock_;
    }
    bool acquire(uint64_t hash, const std::string & exec_id) {
        const auto it = entries_.find(key_(hash, exec_id));
        if (it == entries_.end()) return false;
        it->second.refcount++;
        it->second.last_used = ++clock_;
        return true;
    }
    bool release(uint64_t hash, const std::string & exec_id) {
        const auto it = entries_.find(key_(hash, exec_id));
        if (it == entries_.end() || it->second.refcount <= 0) return false;
        it->second.refcount--;
        return true;
    }

private:
    static uint64_t key_(uint64_t hash, const std::string & exec_id) {
        uint64_t k = hash;
        for (unsigned char c : exec_id) { k ^= c; k *= 1099511628211ULL; }
        return k;
    }
    bool evict_lru_() {
        auto victim = entries_.end();
        for (auto it = entries_.begin(); it != entries_.end(); ++it)
            if (it->second.refcount == 0 && (victim == entries_.end() || it->second.last_used < victim->second.last_used))
                victim = it;
        if (victim == entries_.end()) return false;   // capture() checked free_for_capture() first
        used_ -= victim->second.bytes;
        evicted_.push_back({victim->second.path.back(), victim->second.exec_id});
        entries_.erase(victim);
        return true;
    }
    int64_t budget_, slab_, chunk_, used_ = 0;
    uint64_t clock_ = 0;
    int declined_ = 0;
    std::unordered_map<uint64_t, Entry> entries_;
    std::vector<Evicted> evicted_;
};

// What the admit does with this prompt's preamble (3.2): attach a stored entry, cold-admit and capture one, or
// plain cold. `k` chunks for Attach/Capture. `k_max` is this slot's kvflash_capture_bound.
struct KvFlashPreambleDecision {
    enum Kind { Cold, Capture, Attach } kind = Cold;
    int k = 0;
};
inline KvFlashPreambleDecision kvflash_preamble_decide(const KvFlashPreambleStore & st, const std::vector<int32_t> & ids,
                                                       int e_role, int ct, const std::string & exec_id, int k_max, int slice = 0) {
    KvFlashPreambleDecision d;
    const int uc = kvflash_preamble_unit(ct, slice) / std::max(1, ct);   // chunks per grid step
    const int least = std::max(2, uc);                                // two chunks, and one whole slice when a slice is given
    const int n = kvflash_preamble_ec(e_role, ct, slice) / std::max(1, ct);
    if (n < least || k_max < least) return d;
    const std::vector<uint64_t> path = kvflash_preamble_hashes(ids, n, ct);
    if ((int)path.size() < n) return d;
    const int hit = st.largest_hit(path, exec_id, k_max);
    if (hit >= 1) { d.kind = KvFlashPreambleDecision::Attach; d.k = hit; return d; }
    // No entry this slot can attach: capture the longest cut it can hold. A partial hit (the prompt shares
    // j < k' chunks with an entry) captures at j; no entry at all captures at E_c.
    const int shared = st.shared_with_entries(path, exec_id);
    const int want = shared >= 2 ? shared : n;
    d.k = std::min(want, k_max) / uc * uc;
    d.kind = d.k >= least ? KvFlashPreambleDecision::Capture : KvFlashPreambleDecision::Cold;
    if (d.kind == KvFlashPreambleDecision::Cold) d.k = 0;
    return d;
}

// ---- re-pin (P2-02) ----
//
// A slot's binding {k chunks pinned from chunk 0} is lost by reset() and deserialize() (they clear every pin)
// and cut by truncate() below k. The slot holds the prefix while it has k chunks (n_chunks(), not cur_chunk(),
// which is the last index: an attach leaves exactly k). Every engine path that reselects or allocates calls this first. It re-applies
// the pins, reads back how many the pager accepted (pin_range refuses a span that would deadlock eviction), and
// releases the binding (k = 0, pins cleared) when the slot's length fell below the prefix or fewer than k pins
// were accepted: the request then runs as if cold. `accepted` is the slot's pin count for recall.
// ponytail: unpin_all() on release is right because hard pins are only ever the preamble's; a second user of
// pin_range on a slot pager must move to a per-owner pin before it relies on this.
struct KvFlashRepin {
    int accepted = 0;
    bool released = false;
};
template <class Pager>
inline KvFlashRepin kvflash_repin_preamble(Pager & pager, int & bound_k) {
    KvFlashRepin r;
    if (bound_k <= 0) return r;
    const int ct = pager.chunk_tokens();
    if (ct > 0 && pager.n_chunks() >= bound_k) {
        pager.pin_range(0, (int64_t)bound_k * ct);
        for (int c = 0; c < bound_k; c++) r.accepted += pager.is_pinned(c) ? 1 : 0;
        if (r.accepted == bound_k) return r;
    }
    pager.unpin_all();
    bound_k = 0;
    r.accepted = 0;
    r.released = true;
    return r;
}

}  // namespace luce::common
