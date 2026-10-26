// KVFlash 2.0 sessions P1b: the preamble store wired to slots (SESSIONS_DESIGN.md 3.2, 3.3), engine-free.
//
//  - KvFlashPreambleBinder: the slot -> {entry hash, k} bindings, the entry refcount they hold, and the one
//    re-pin (kvflash_repin_preamble) with the accepted count recall reads. KvFlashSlotKv calls ensure() after
//    every operation that resets, truncates or replaces a slot's pager, so no engine path can forget it (P2-02).
//  - capture / attach / rebind: the bounded capture of the first k chunks + slab, the blob an attach adopts,
//    and the re-binding of a restored session. Templates over the pager and the allocator, so the CPU tests
//    drive the same code the engine runs.
//  - persistence under preambles/: one file per entry, written through a rename, read back only under the
//    exec_id that wrote it (P2-03).
#pragma once

#include "kvflash_preamble_store.h"
#include "kvflash_slot_bundle.h"

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <string>
#include <type_traits>
#include <vector>

namespace luce::common {

// The execution identity (P2-03): what must be equal for an entry's KV and slab to be the bytes this server
// would have computed. The parts are model/KV geometry, backend, cut size, numeric flags, ggml build.
inline std::string kvflash_preamble_exec_id(const std::vector<std::string> & parts) {
    uint64_t h = 1469598103934665603ULL;
    auto mix = [&](unsigned char c) { h ^= c; h *= 1099511628211ULL; };
    for (const std::string & p : parts) {
        const uint64_t n = p.size();
        for (int b = 0; b < 8; b++) mix((unsigned char)(n >> (8 * b)));   // length-prefixed: no concatenation ambiguity
        for (unsigned char c : p) mix(c);
    }
    char buf[17];
    std::snprintf(buf, sizeof buf, "%016llx", (unsigned long long)h);
    return buf;
}

struct KvFlashPreambleBinding {
    uint64_t hash = 0;
    int k = 0;   // 0: not bound
};

class KvFlashPreambleBinder {
public:
    KvFlashPreambleBinder(KvFlashPreambleStore & store, std::string exec_id, int n_slots)
        : st_(store), exec_(std::move(exec_id)), b_((size_t)std::max(0, n_slots)), pins_((size_t)std::max(0, n_slots), 0) {}

    const std::string & exec_id() const { return exec_; }
    KvFlashPreambleStore & store() { return st_; }
    KvFlashPreambleBinding binding(int slot) const { return ok_(slot) ? b_[(size_t)slot] : KvFlashPreambleBinding{}; }
    // The slot's accepted pins: what kvflash_prefill_apply_on gets as slot_pins.
    int pins(int slot) const { return ok_(slot) ? pins_[(size_t)slot] : 0; }

    // The slot's session now uses entry (hash, k): refcount +1, the earlier binding's -1. False (and unbound)
    // when the store has no such entry.
    bool bind(int slot, uint64_t hash, int k) {
        if (!ok_(slot)) return false;
        unbind(slot);
        if (k <= 0 || !st_.acquire(hash, exec_)) return false;
        b_[(size_t)slot] = {hash, k};
        return true;
    }
    // The session left the slot or lost its prefix: refcount -1, no pins. Idempotent.
    void unbind(int slot) {
        if (!ok_(slot)) return;
        KvFlashPreambleBinding & b = b_[(size_t)slot];
        if (b.k > 0) st_.release(b.hash, exec_);
        b = {};
        pins_[(size_t)slot] = 0;
    }
    // Re-applies the slot's pins on its pager and records the accepted count; a binding the pager cannot hold
    // (length below the prefix, a refused span) is released with its refcount. Returns the accepted count.
    template <class Pager>
    int ensure(int slot, Pager & pager) {
        if (!ok_(slot)) return 0;
        KvFlashPreambleBinding & b = b_[(size_t)slot];
        if (b.k <= 0) { pins_[(size_t)slot] = 0; return 0; }
        const KvFlashRepin r = kvflash_repin_preamble(pager, b.k);
        pins_[(size_t)slot] = r.accepted;
        if (r.released) { st_.release(b.hash, exec_); b = {}; }
        return r.accepted;
    }

private:
    bool ok_(int slot) const { return slot >= 0 && (size_t)slot < b_.size(); }
    KvFlashPreambleStore & st_;
    std::string exec_;
    std::vector<KvFlashPreambleBinding> b_;
    std::vector<int> pins_;
};

// ---- the bytes an attach adopts ---------------------------------------------------------------

// KvFlashPager::serialize(k) of a pager whose chunks 0..k-1 are all resident with score 0: header, the k chunk
// payloads, a ledger of {resident, materialized, score 0}. The deserialize of it places chunk c in block c.
inline std::vector<uint8_t> kvflash_preamble_blob(const KvFlashPreambleStore::Entry & e, size_t chunk_bytes) {
    std::vector<uint8_t> out;
    out.reserve(e.hdr.size() + e.chunks.size() + (size_t)e.k * 8);
    out.insert(out.end(), e.hdr.begin(), e.hdr.end());
    out.insert(out.end(), e.chunks.begin(), e.chunks.end());
    (void)chunk_bytes;
    for (int c = 0; c < e.k; c++) { const uint8_t led[8] = {1, 1, 0, 0, 0, 0, 0, 0}; out.insert(out.end(), led, led + 8); }
    return out;
}

struct KvFlashPreambleCapture {
    enum Kind { Stored, Exists, Declined, Refused } kind = Refused;
    uint64_t hash = 0;
    int k = 0;
};

// The capture cut ran: the slot's pager holds chunks 0..k-1 of `ids` and the caller has the slab at 64k.
// Refused (nothing stored, nothing bound) past k_max (P2-04), for k < 2, ids shorter than k chunks, or a pager
// that does not hold k chunks. Stored/Exists bind the slot to the entry and pin it; Declined (the protected
// entries leave no room) leaves the slot as it was.
template <class Pager>
inline KvFlashPreambleCapture kvflash_preamble_capture(KvFlashPreambleBinder & bd, int slot, Pager & pager,
                                                       const std::vector<int32_t> & ids, int k, int k_max,
                                                       std::vector<uint8_t> slab) {
    KvFlashPreambleCapture r;
    const int ct = pager.chunk_tokens();
    if (k < 2 || k > k_max || ct <= 0 || pager.n_chunks() < k || (int)(ids.size() / (size_t)ct) < k) return r;
    const std::vector<uint64_t> path = kvflash_preamble_hashes(ids, k, ct);
    if ((int)path.size() < k) return r;
    r.hash = path.back();
    r.k = k;
    KvFlashPreambleStore & st = bd.store();
    const auto c = st.capture(path, bd.exec_id());
    if (c == KvFlashPreambleStore::Declined) { r.kind = KvFlashPreambleCapture::Declined; return r; }
    if (c == KvFlashPreambleStore::Stored) {
        pager.synchronize_paging();
        const size_t cb = pager.chunk_bytes();
        std::vector<uint8_t> chunks((size_t)k * cb);
        for (int i = 0; i < k; i++) pager.read_chunk(i, chunks.data() + (size_t)i * cb);
        st.set_payload(r.hash, bd.exec_id(), pager.blob_header(k), std::move(chunks), std::move(slab));
    }
    r.kind = c == KvFlashPreambleStore::Stored ? KvFlashPreambleCapture::Stored : KvFlashPreambleCapture::Exists;
    if (bd.bind(slot, r.hash, k)) bd.ensure(slot, pager);
    return r;
}

// Attach entry (hash) to the free slot: bind, adopt the blob as a parked session of 64k tokens, and check that
// the pager kept the pins. False leaves the slot free and unbound (the request goes cold). `slab` is the
// packed slab to restore at 64k. The caller then resumes the slot at 64k.
template <class Kv>
inline bool kvflash_preamble_attach(Kv & kv, KvFlashPreambleBinder & bd, int slot, uint64_t ledger_id, uint64_t hash,
                                    int k, std::vector<uint8_t> & slab) {
    const KvFlashPreambleStore::Entry * e = bd.store().find(hash, bd.exec_id());
    if (!e || e->k != k || e->hdr.empty() || e->slab.empty()) return false;
    if (!bd.bind(slot, hash, k)) return false;
    const std::vector<uint8_t> blob = kvflash_preamble_blob(*e, 0);
    const uint32_t len = (uint32_t)k * kv.block_size();
    const auto adopted = kv.adopt_parked(slot, ledger_id, len, blob.data(), blob.size());
    if (adopted != std::decay_t<decltype(adopted)>::Ok || bd.binding(slot).k != k) {
        if (kv.parked(slot)) kv.drop_parked(slot);   // adopted, but the pins did not hold
        bd.unbind(slot);
        return false;
    }
    slab = e->slab;
    return true;
}

// A restored session (restart, evicted-to-disk): honour its persisted binding when the history still starts
// with the entry's prefix and the store still has the entry under this exec_id.
template <class Kv>
inline bool kvflash_preamble_rebind(Kv & kv, KvFlashPreambleBinder & bd, int slot, const std::vector<int32_t> & history,
                                    uint64_t hash, int k) {
    if (k < 2) return false;
    const int ct = (int)kv.block_size();
    const std::vector<uint64_t> path = kvflash_preamble_hashes(history, k, ct);
    if ((int)path.size() < k || path.back() != hash) return false;
    if (!bd.bind(slot, hash, k)) return false;
    bd.ensure(slot, kv.pager(slot));
    return bd.binding(slot).k == k;
}

// ---- persistence under preambles/ -----------------------------------------------------------------

namespace detail {
inline std::string preamble_hex(uint64_t v) {
    char b[17];
    std::snprintf(b, sizeof b, "%016llx", (unsigned long long)v);
    return b;
}
inline std::string preamble_file(const std::string & root, uint64_t hash, const std::string & exec_id) {
    return root + "/" + preamble_hex(hash) + "-" + preamble_hex(kvflash_fnv1a(exec_id)) + ".kvp";
}
constexpr char kPreambleMagic[8] = {'K', 'V', 'P', 'R', 'E', '0', '0', '1'};
}  // namespace detail

// One file per entry: magic, k, exec_id, the hash path, hdr, chunks, slab, then a crc64 of everything before it.
inline bool kvflash_preamble_save(const std::string & root, const KvFlashPreambleStore::Entry & e) {
    if (e.hdr.empty() || e.k <= 0 || (int)e.path.size() != e.k) return false;
    std::error_code ec;
    std::filesystem::create_directories(root, ec);
    std::vector<uint8_t> body;
    auto put = [&](const void * d, size_t n) { const uint8_t * p = (const uint8_t *)d; body.insert(body.end(), p, p + n); };
    auto u64 = [&](uint64_t v) { put(&v, 8); };
    put(detail::kPreambleMagic, 8);
    u64((uint64_t)e.k);
    u64(e.exec_id.size()); put(e.exec_id.data(), e.exec_id.size());
    for (uint64_t h : e.path) u64(h);
    u64(e.hdr.size()); put(e.hdr.data(), e.hdr.size());
    u64(e.chunks.size()); put(e.chunks.data(), e.chunks.size());
    u64(e.slab.size()); put(e.slab.data(), e.slab.size());
    u64(kvflash_crc64(body.data(), body.size()));
    return kvflash_bundle_write_stream(detail::preamble_file(root, e.path.back(), e.exec_id), {},
                                       [&](auto && w) { return w(body.data(), body.size()); });
}

inline void kvflash_preamble_remove(const std::string & root, uint64_t hash, const std::string & exec_id) {
    std::error_code ec;
    std::filesystem::remove(detail::preamble_file(root, hash, exec_id), ec);
}

struct KvFlashPreambleLoad {
    int loaded = 0;
    int foreign = 0;    // another execution identity: listed, left on disk, never attached
    int bad = 0;        // torn, resized or of another geometry: left on disk, never attached
    int declined = 0;   // does not fit the byte budget
};

// Adopts every entry file of this exec_id into `store` (refcount 0, payload filled). chunk_bytes is this
// server's per-chunk size: a file whose chunk payload is not k x chunk_bytes is bad.
inline KvFlashPreambleLoad kvflash_preamble_load_all(const std::string & root, KvFlashPreambleStore & store,
                                                     const std::string & exec_id, size_t chunk_bytes) {
    KvFlashPreambleLoad rep;
    std::error_code ec;
    if (!std::filesystem::is_directory(root, ec)) return rep;
    std::vector<std::string> files;
    for (const auto & d : std::filesystem::directory_iterator(root, ec))
        if (d.is_regular_file() && d.path().extension() == ".kvp") files.push_back(d.path().string());
    std::sort(files.begin(), files.end());
    for (const std::string & f : files) {
        std::vector<uint8_t> raw;
        if (!kvflash_bundle_read(f, raw, (size_t)1 << 34) || raw.size() < 8 + 8 + 8 + 8) { rep.bad++; continue; }
        size_t at = 0;
        bool ok = std::memcmp(raw.data(), detail::kPreambleMagic, 8) == 0;
        at = 8;
        auto u64 = [&](uint64_t & v) { if (at + 8 > raw.size()) { ok = false; return false; } std::memcpy(&v, raw.data() + at, 8); at += 8; return true; };
        auto bytes = [&](std::vector<uint8_t> & out, uint64_t n) {
            if (!ok || n > raw.size() - at) { ok = false; return false; }
            out.assign(raw.begin() + (long)at, raw.begin() + (long)(at + n)); at += (size_t)n; return true;
        };
        uint64_t k = 0, n = 0;
        std::vector<uint8_t> exec_b, hdr, chunks, slab;
        std::vector<uint64_t> path;
        ok = ok && u64(k) && k > 0 && k < (1u << 20) && u64(n) && bytes(exec_b, n);
        for (uint64_t i = 0; ok && i < k; i++) { uint64_t h = 0; ok = u64(h); path.push_back(h); }
        ok = ok && u64(n) && bytes(hdr, n) && u64(n) && bytes(chunks, n) && u64(n) && bytes(slab, n);
        uint64_t crc = 0;
        const size_t body = at;
        ok = ok && u64(crc) && at == raw.size() && crc == kvflash_crc64(raw.data(), body) && chunks.size() == (size_t)k * chunk_bytes;
        if (!ok) { rep.bad++; continue; }
        const std::string exec(exec_b.begin(), exec_b.end());
        if (exec != exec_id) { rep.foreign++; continue; }
        // a load never evicts what it loaded before: an entry past the budget is declined
        if (store.used_bytes() + store.entry_bytes((int)k) > store.budget_bytes() || store.capture(path, exec) != KvFlashPreambleStore::Stored) { rep.declined++; continue; }
        store.set_payload(path.back(), exec, std::move(hdr), std::move(chunks), std::move(slab));
        rep.loaded++;
    }
    return rep;
}

}  // namespace luce::common
