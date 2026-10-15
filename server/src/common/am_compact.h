// KVFlash 2.0b M7: automatic Attention Matching (AM) compaction at the seal of a
// 640-token prefill block. This header holds everything about the feature that
// does not need a model:
//   * AutoConfig / the seal-trigger policy (which sealed blocks compact);
//   * the per-block pipeline on tensors that already live on a backend
//     (M4 top-64 selection per KV head, then the M5 beta / C2 fit);
//   * the row writer's pure half (gather the kept rows, order them).
// The engine (qwen35_backend.cpp) owns the part that touches the KV cache and
// the pager: it reads the block back, calls compact_block(), writes the rows
// into the head chunk and marks it compacted.
//
// Layout contract (all from M3..M6):
//   q  F16 [head_dim, 640, n_head]       post-RoPE (and post-FWHT under Q4_0 K)
//                                        queries of the block, one plane of the capture ring
//   k  F32 [head_dim, 640, n_head_kv]    the block's cached keys, dequantized, same basis as q
//   v  F32 [head_dim, 640, n_head_kv]    the block's cached values, dequantized
//   sel[h*64 + r]                        kept key index (0..639) of KV head h, ascending
//   beta[h*64 + r], c2[(h*64 + r)*head_dim + c]   M5 output (FitResult)
#pragma once

#include "common/am_fit.h"
#include "common/am_fit_solve.h"

#include "ggml.h"
#include "ggml-backend.h"

#include <cstdint>
#include <string>
#include <vector>

namespace luce::am {

inline constexpr int kAutoBlockTokens = 640;   // tokens per sealed block (10 pager chunks)
inline constexpr int kAutoSlots       = 64;    // rows a compacted block keeps (one pager chunk)

// --kvflash-am-ratio / -prefix-blocks / -recent-blocks. Compaction is off unless
// ratio > 0. A compacted block is one 64-row pager chunk, so the only ratio the
// engine supports is kAutoSlots / kAutoBlockTokens = 0.10.
struct AutoConfig {
    double ratio         = 0.0;
    int    prefix_blocks = 0;   // P: the first P blocks of a prompt are never compacted
    int    recent_blocks = 0;   // W: the last W sealed blocks of a prompt are never compacted
    bool enabled() const { return ratio > 0.0; }
};

// Parses the three option strings (null or empty: default). Returns the config
// and, on a bad value, fills *err (the config is then disabled).
AutoConfig auto_config_from_strings(const char * ratio, const char * prefix_blocks,
                                    const char * recent_blocks, std::string * err);

// The engine runs the automatic path only when it is enabled, the pager is on
// and the architecture is the dense single-device Qwen3.5 family.
bool auto_active(const AutoConfig & cfg, bool kvflash_active, bool arch_supported);

// What the seal-trigger policy sees about the block that just sealed.
struct SealInfo {
    int  block              = 0;      // block index: tokens [640 b, 640 b + 640)
    int  prompt_len         = 0;      // tokens of the prompt being prefilled
    int  exact_tail_tokens  = 0;      // tokens at the end of the prompt that stay exact (0: none)
    bool pinned             = false;  // any pager chunk of the block is pinned
};

// True when the sealed block should be compacted: enabled, not in the exact prefix
// (block >= P), not among the last W sealed blocks of the prompt, not touching the
// exact tail, not pinned. The open tail (< 640 tokens) never seals, so it is exact
// by construction.
bool should_compact(const AutoConfig & cfg, const SealInfo & s);

// Device memory the automatic path adds on top of the cache: the capture ring
// (ring_bytes, M3) plus the scratch of one block fit (K and V as F32 for one
// layer, the Gram accumulators). Pure; the planner reserves this.
int64_t auto_reserve_bytes(int64_t ring_bytes, int head_dim, int n_head_kv);

// ---- per-block pipeline --------------------------------------------------------

struct BlockFit {
    std::vector<int32_t>  sel;     // [n_head_kv * slots], head-major, ascending within a head
    luce_am::FitResult    fit;     // beta [64, n_head_kv], c2 [head_dim, 64, n_head_kv]
    double ms_select = 0, ms_fit = 0;
};

// Select the top-64 keys of every KV head (M4), order them by position, fit beta and
// C2 (M5). q, k, v are tensors in buffers of `backend` (see the layout contract above;
// q may be a view of a ring plane). Returns false, with *err set, when any step fails
// or any KV head's solve fails; nothing is to be published then.
bool compact_block(ggml_backend_t backend, luce_am::GramAccumulator & acc, ggml_tensor * q,
                   ggml_tensor * k, ggml_tensor * v, int slots, const luce_am::NnlsParams & params,
                   BlockFit & out, std::string * err);

// Host-array form (tests, tools): uploads q/k/v to `backend` and calls the form above.
// q: F16 [head_dim, n_tokens, n_head]; k, v: F32 [head_dim, n_keys, n_head_kv]. n_keys
// is any value >= slots; the kept count per KV head is `slots`.
bool compact_block_host(ggml_backend_t backend, luce_am::GramAccumulator & acc, int head_dim,
                        int n_tokens, int n_head, int n_head_kv, int n_keys, int slots,
                        const ggml_fp16_t * q, const float * k, const float * v,
                        const luce_am::NnlsParams & params, BlockFit & out, std::string * err);

// ---- pure row helpers ------------------------------------------------------------

// Sorts every head's kept indices ascending (sel is [n_head_kv][slots]).
void sort_sel_ascending(std::vector<int32_t> & sel, int n_head_kv, int slots);

// Gather the kept rows of one KV head. `rows` holds the block's rows of that head,
// row_bytes each, in position order (n_keys rows); out receives slots rows.
void gather_rows(const uint8_t * rows, size_t row_bytes, const int32_t * sel, int slots, uint8_t * out);

}  // namespace luce::am
