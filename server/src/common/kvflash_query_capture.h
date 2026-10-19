// Per-token query capture for selector v2 and the user-question queries.
//
// The prefill graph can keep the last N tokens' queries of every batch
// (TargetCache::q_cap_all). How many it keeps comes from two environment
// variables, read here so the graph, the memory estimate and the backend agree.

#pragma once

#include <algorithm>
#include <cstdlib>

namespace luce::common {

// Most per-token queries the graph captures per batch.
constexpr int kQueryCaptureMaxTokens = 512;

// A query count from env var `name`: 0 when unset or invalid, at most kQueryCaptureMaxTokens.
inline int kvflash_env_queries(const char * name) {
    const char * e = std::getenv(name);
    if (!e || !*e) return 0;
    char * end = nullptr;
    const long v = std::strtol(e, &end, 10);
    return (*end || v <= 0) ? 0 : (int)std::min<long>(v, kQueryCaptureMaxTokens);
}

// LUCE_KVFLASH_SELECT_QUERIES=N (rewind modes): selector v2 scores with the last N prompt queries.
inline int kvflash_select_queries() { return kvflash_env_queries("LUCE_KVFLASH_SELECT_QUERIES"); }

// LUCE_KVFLASH_UQ_QUERIES=N: the latest human question's last N queries join the selector's.
inline int kvflash_uq_queries() { return kvflash_env_queries("LUCE_KVFLASH_UQ_QUERIES"); }

// Per-token queries the graph captures per batch (0 = none).
inline int kvflash_q_capture_all_tokens() {
    return std::max(kvflash_select_queries(), kvflash_uq_queries());
}

} // namespace luce::common
