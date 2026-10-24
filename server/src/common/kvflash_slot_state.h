// Byte-exact capture/restore of one slot's recurrent slabs (Track C, C1.3-04):
// the SSM state [S_v, S_v, H_v, n_slots] and conv state [(k-1), channels, n_slots]
// of every delta-net layer, no KV. A slot's slab is the contiguous block at its
// index of the trailing axis, so each copy is one ggml_backend_tensor_get/set
// range per tensor; the same code serves CPU and device tensors.

#pragma once

#include "ggml.h"
#include "ggml-backend.h"

#include <cstdint>
#include <vector>

namespace luce::common {

namespace kvflash_slot_state_detail {
// Bytes of one slot's slab of `t` whose slot axis is `axis` (3 for ssm, 2 for conv); 0 if unusable.
inline size_t slab_bytes(const ggml_tensor * t, int axis, int slot) {
    if (!t || slot < 0 || t->ne[axis] <= slot || !ggml_is_contiguous(t)) return 0;
    return t->nb[axis];
}
template <class F>
inline bool each(const std::vector<ggml_tensor *> & ssm, const std::vector<ggml_tensor *> & conv, int slot, F && fn) {
    size_t at = 0;
    for (int pass = 0; pass < 2; pass++)
        for (ggml_tensor * t : pass == 0 ? ssm : conv) {
            const int axis = pass == 0 ? 3 : 2;
            const size_t n = slab_bytes(t, axis, slot);
            if (n == 0) return false;
            fn(t, (size_t)slot * t->nb[axis], at, n);
            at += n;
        }
    return true;
}
}  // namespace kvflash_slot_state_detail

// Bytes of one slot's state across all layers (0 for an empty or unusable set).
inline size_t kvflash_slot_state_bytes(const std::vector<ggml_tensor *> & ssm, const std::vector<ggml_tensor *> & conv) {
    size_t n = 0;
    for (ggml_tensor * t : ssm)  { const size_t b = kvflash_slot_state_detail::slab_bytes(t, 3, 0); if (!b) return 0; n += b; }
    for (ggml_tensor * t : conv) { const size_t b = kvflash_slot_state_detail::slab_bytes(t, 2, 0); if (!b) return 0; n += b; }
    return n;
}

inline bool kvflash_slot_state_capture(const std::vector<ggml_tensor *> & ssm, const std::vector<ggml_tensor *> & conv,
                                       int slot, std::vector<uint8_t> & out) {
    const size_t total = kvflash_slot_state_bytes(ssm, conv);
    std::vector<uint8_t> tmp(total);
    const bool ok = total > 0 && kvflash_slot_state_detail::each(ssm, conv, slot,
        [&](ggml_tensor * t, size_t off, size_t at, size_t n) { ggml_backend_tensor_get(t, tmp.data() + at, off, n); });
    if (ok) out = std::move(tmp);
    return ok;
}

// All-or-nothing: the blob size and slot are checked before any byte is written.
inline bool kvflash_slot_state_restore(const std::vector<ggml_tensor *> & ssm, const std::vector<ggml_tensor *> & conv,
                                       int slot, const std::vector<uint8_t> & blob) {
    if (blob.size() != kvflash_slot_state_bytes(ssm, conv) || blob.empty()) return false;
    if (!kvflash_slot_state_detail::each(ssm, conv, slot, [](ggml_tensor *, size_t, size_t, size_t) {})) return false;
    return kvflash_slot_state_detail::each(ssm, conv, slot,
        [&](ggml_tensor * t, size_t off, size_t at, size_t n) { ggml_backend_tensor_set(t, blob.data() + at, off, n); });
}

// FNV-1a over the blob: the box test compares hashes before and after.
inline uint64_t kvflash_slot_state_hash(const std::vector<uint8_t> & blob) {
    uint64_t h = 1469598103934665603ull;
    for (uint8_t b : blob) { h ^= b; h *= 1099511628211ull; }
    return h;
}

}  // namespace luce::common
