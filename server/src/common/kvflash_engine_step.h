// What the seq engine decides on its own for KVFlash slots (Track C, C1.3-05),
// kept pure so the unit suite covers it without a device.

#pragma once
#include <cstdlib>
#include <algorithm>

#include "concurrency/seq_engine.h"
#include "kvflash_slot_ladder.h"
#include "kvflash_slot_pager.h"

#include <algorithm>

namespace luce::common {

// D1: a KVFlash slot prefills in cuts of U logical positions (8 chunks), never
// shorter because of peers.
constexpr int kKvFlashCutU = 512;
// LUCE_KVFLASH_CUT_U (a noise-floor knob, default 512): a multiple of the 64-token chunk in [64, 4096]; anything else is the default.
inline int kvflash_parse_cut_u(const char * v) {
    const int n = v && *v ? std::atoi(v) : 0;
    return n < 64 ? kKvFlashCutU : (std::min(n, 4096) / 64) * 64;
}
inline int kvflash_cut_u() { static const int u = kvflash_parse_cut_u(std::getenv("LUCE_KVFLASH_CUT_U")); return u; }

// The scheduler grants a KVFlash slice whole cuts only: each slice gets U (the
// allocation quantum), and the step's total is a whole number of them.
inline StepPlanLimits kvflash_step_limits(StepPlanLimits l, int cut_u = kvflash_cut_u()) {
    l.max_prefill_tokens_per_sequence = std::min(l.max_prefill_tokens_per_sequence, cut_u);
    l.prefill_allocation_quantum = cut_u;
    l.max_prefill_tokens_total = std::max(cut_u, l.max_prefill_tokens_total / cut_u * cut_u);
    return l;
}

// Entries of a slot's block-table column that cover kv_len tokens: 4 per
// 64-token chunk at 16-token blocks, clamped to the table height.
inline int kvflash_column_entries(int kv_len, int table_height, int chunk_tokens = 64, int block_tokens = 16) {
    const int chunks = (kv_len + chunk_tokens - 1) / chunk_tokens;
    return std::min(table_height, chunks * kvflash_table_entries(chunk_tokens, block_tokens));
}

}  // namespace luce::common
