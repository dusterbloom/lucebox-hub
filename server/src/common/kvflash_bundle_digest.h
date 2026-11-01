// Digest of a session's whole saved state (R3 digest mode, LUCE_KVFLASH_BUNDLE_DIGEST=1): one FNV-1a (the slot-state
// hash) per component and one over the six, from host bytes alone, so the save (the bytes being staged) and the
// restore (the live slot, read back through the same staging) are compared on equal terms.
//
// Digested: the token history, the end slab (ssm + conv pack), every materialized chunk's KV bytes in chunk order
// (each folded with its index), the ledger, the pager header (magic, chunk count, chunk tokens, head count, K/V segment
// bytes, chunk bytes, K/V types: geometry only, all fixed for a session), and the rungs a next turn can rewind to
// (ckpt, trunk, third: a presence byte, the position and the slab pack; then the pin position).
// Excluded, because a later park of the same state legitimately differs there: the head's gen, log_gen, end_gen and
// rung gens (a counter per park), log_bytes and the chunk map offsets (where a record landed in an appended or
// compacted log), the record tags' park generation and crc, and the slab file ids. A chunk the ledger does not
// materialize is zeros at load and is not digested.
#pragma once

#include "kvflash_slot_bundle.h"
#include "kvflash_slot_state.h"

#include <cstdint>
#include <cstdio>
#include <string>
#include <vector>

namespace luce::common {

// The ladder a restored session can rewind to: ckpt, trunk and third rung (a position and its slab pack; no slab or a
// non-positive position: absent) and the pin position (0: none).
struct KvFlashDigestRungs {
    struct Rung { int32_t pos = 0; const std::vector<uint8_t> * slab = nullptr; };
    Rung r[3];
    int32_t pin = 0;
};

struct KvFlashStateDigest {
    uint64_t all = 0, hist = 0, slab = 0, kv = 0, ledger = 0, pager = 0, rung = 0;
    uint64_t bytes = 0;   // bytes digested, the chunk index folds not counted
};

// chunk_at(ci, dst) writes chunk_bytes of chunk ci (KvFlashSlotPager::read_chunk, or a slice of a blob).
template <class ChunkAt>
inline KvFlashStateDigest kvflash_state_digest(const std::vector<int32_t> & hist, const std::vector<uint8_t> & slab, const std::vector<uint8_t> & pager_hdr,
                                               const std::vector<uint8_t> & ledger, const KvFlashDigestRungs & rungs, size_t chunk_bytes, ChunkAt && chunk_at) {
    KvFlashStateDigest d;
    d.hist = kvflash_fnv_update(kKvFlashFnvBasis, (const uint8_t *)hist.data(), hist.size() * 4);
    d.slab = kvflash_slot_state_hash(slab);
    d.ledger = kvflash_slot_state_hash(ledger);
    d.pager = kvflash_slot_state_hash(pager_hdr);
    d.bytes = hist.size() * 4 + slab.size() + ledger.size() + pager_hdr.size();
    d.rung = kKvFlashFnvBasis;
    for (const auto & r : rungs.r) {
        const uint8_t present = r.pos > 0 && r.slab;
        d.rung = kvflash_fnv_update(d.rung, &present, 1);
        if (!present) continue;
        d.rung = kvflash_fnv_update(d.rung, (const uint8_t *)&r.pos, 4);
        d.rung = kvflash_fnv_update(d.rung, r.slab->data(), r.slab->size());
        d.bytes += r.slab->size();
    }
    const uint8_t pinned = rungs.pin > 0;
    d.rung = kvflash_fnv_update(d.rung, &pinned, 1);
    if (pinned) d.rung = kvflash_fnv_update(d.rung, (const uint8_t *)&rungs.pin, 4);
    d.kv = kKvFlashFnvBasis;
    std::vector<uint8_t> buf(chunk_bytes);
    for (size_t ci = 0; ci * kKvFlashLedgerEntry + 1 < ledger.size(); ++ci) {
        if (ledger[ci * kKvFlashLedgerEntry + 1] == 0) continue;
        chunk_at((int)ci, buf.data());
        const uint32_t idx = (uint32_t)ci;
        d.kv = kvflash_fnv_update(d.kv, (const uint8_t *)&idx, 4);
        d.kv = kvflash_fnv_update(d.kv, buf.data(), buf.size());
        d.bytes += chunk_bytes;
    }
    const uint64_t parts[6] = {d.hist, d.slab, d.kv, d.ledger, d.pager, d.rung};
    d.all = kvflash_fnv_update(kKvFlashFnvBasis, (const uint8_t *)parts, sizeof parts);
    return d;
}

// The one log line of both sides; tag is "save" or "restore", sid the session's store key.
inline std::string kvflash_state_digest_line(const char * tag, const std::string & sid, uint32_t gen, const KvFlashStateDigest & d, double ms) {
    char b[448];
    std::snprintf(b, sizeof b, "[kvflash-digest] %s sid=%s gen=%u all=%016llx hist=%016llx slab=%016llx kv=%016llx ledger=%016llx pager=%016llx rung=%016llx bytes=%llu ms=%.1f",
                  tag, sid.c_str(), gen, (unsigned long long)d.all, (unsigned long long)d.hist, (unsigned long long)d.slab, (unsigned long long)d.kv,
                  (unsigned long long)d.ledger, (unsigned long long)d.pager, (unsigned long long)d.rung, (unsigned long long)d.bytes, ms);
    return b;
}

}  // namespace luce::common
