// Digest of a session's whole saved state (R3 digest mode, LUCE_KVFLASH_BUNDLE_DIGEST=1). STUB.
#pragma once

#include <cstdint>
#include <cstdio>
#include <string>
#include <vector>

namespace luce::common {

struct KvFlashStateDigest {
    uint64_t all = 0, hist = 0, slab = 0, kv = 0, ledger = 0, pager = 0;
    uint64_t bytes = 0;
};

template <class ChunkAt>
inline KvFlashStateDigest kvflash_state_digest(const std::vector<int32_t> &, const std::vector<uint8_t> &, const std::vector<uint8_t> &,
                                               const std::vector<uint8_t> &, size_t, ChunkAt &&) {
    return {};
}

inline std::string kvflash_state_digest_line(const char *, const std::string &, uint32_t, const KvFlashStateDigest &, double) { return ""; }

}  // namespace luce::common
