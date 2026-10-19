// LUCE_KVFLASH_UQ_QUERIES: the latest human question as extra selector queries. Pure helpers
// (no ggml, no GPU): the human-turn span, the capture plan, the per-session store with its key.

#pragma once

#include "kvflash_qk_dump.h"

#include <algorithm>
#include <cstdint>
#include <string>
#include <vector>

namespace luce::common {

// Content span [lo, hi) of a user turn: lo is after `<|im_start|> user \n`, hi the `<|im_end|>`.
struct KvFlashHumanTurn {
    int lo = -1, hi = -1;
    bool valid() const { return lo >= 0 && hi > lo; }
};

// The last `<|im_start|>user` turn that is not a tool result. Qwen templates render tool results as
// user turns whose content starts with `<tool_response>` (tool_response < 0: not distinguishable, so
// every user turn counts as human). A turn with no closing `<|im_end|>` is skipped.
inline KvFlashHumanTurn kvflash_last_human_turn(const std::vector<int32_t> & ids, int32_t im_start, int32_t user,
                                                int32_t newline, int32_t im_end, int32_t tool_response) {
    if (im_start < 0 || user < 0 || im_end < 0) return {};
    for (int i = (int)ids.size() - 2; i >= 0; i--) {
        if (ids[(size_t)i] != im_start || ids[(size_t)i + 1] != user) continue;
        int lo = i + 2;
        if (newline >= 0 && lo < (int)ids.size() && ids[(size_t)lo] == newline) lo++;
        if (tool_response >= 0 && lo < (int)ids.size() && ids[(size_t)lo] == tool_response) continue;
        int hi = lo;
        while (hi < (int)ids.size() && ids[(size_t)hi] != im_end) hi++;
        if (hi < (int)ids.size() && hi > lo) return {lo, hi};
    }
    return {};
}

struct KvFlashUqPlan {
    bool capture = false;   // this prefill covers the whole window: cut at `cut`, capture rows
    int  n = 0;             // rows wanted: min(N, question tokens)
    int  win_lo = 0, cut = 0;
};
inline KvFlashUqPlan kvflash_uq_plan(const KvFlashHumanTurn & t, int n_queries, int kv_offset, int end) {
    KvFlashUqPlan p;
    if (n_queries <= 0 || !t.valid()) return p;
    p.n = std::min(n_queries, t.hi - t.lo);
    p.win_lo = t.hi - p.n;
    p.cut = t.hi;
    p.capture = t.hi <= end && kv_offset <= p.win_lo;
    return p;
}

// A batch [batch_start, batch_start + batch_tokens) captured its last `captured_rows` queries. Which of
// them fall in the window [win_lo, win_hi): the first row index and the count.
struct KvFlashUqTake { int first_row = 0, count = 0; };
inline KvFlashUqTake kvflash_uq_batch_take(int batch_start, int batch_tokens, int captured_rows, int win_lo, int win_hi) {
    KvFlashUqTake t;
    const int end = batch_start + batch_tokens, rows_lo = end - std::min(captured_rows, batch_tokens);
    const int lo = std::max(win_lo, rows_lo), hi = std::min(win_hi, end);
    if (hi <= lo) return t;
    t.first_row = lo - rows_lo;
    t.count = hi - lo;
    return t;
}

// The captured rows of one session, keyed by {hi, hash of ids[0, hi)}. Not persisted into snapshots.
struct KvFlashUqStore {
    int lo = -1, hi = -1, n = 0;    // the n rows are the queries of positions [hi - n, hi)
    uint64_t hash = 0;
    std::vector<float> rows;        // n * (layers * q heads * head dim), oldest first
    bool empty() const { return n <= 0; }
    void clear() { lo = hi = -1; n = 0; hash = 0; rows.clear(); }
};

inline uint64_t kvflash_uq_hash(const std::vector<int32_t> & ids, int upto) {
    uint64_t h = 1469598103934665603ULL;
    for (int i = 0; i < upto && i < (int)ids.size(); i++) h = (h ^ (uint64_t)(uint32_t)ids[(size_t)i]) * 1099511628211ULL;
    return h;
}
inline bool kvflash_uq_matches(const KvFlashUqStore & s, const std::vector<int32_t> & ids, int hi) {
    return !s.empty() && s.hi == hi && hi > 0 && hi <= (int)ids.size() && s.hash == kvflash_uq_hash(ids, hi);
}
// The history was cut to `len` tokens: a store whose question ends past it is gone.
inline void kvflash_uq_truncate(KvFlashUqStore & s, size_t len) {
    if (!s.empty() && (size_t)s.hi > len) s.clear();
}
// The history was replaced: keep the store only if it still starts with the same ids[0, hi).
inline void kvflash_uq_revalidate(KvFlashUqStore & s, const std::vector<int32_t> & ids) {
    if (!s.empty() && !kvflash_uq_matches(s, ids, s.hi)) s.clear();
}

// Append the stored rows to the scorer's query list (no-op for an empty store).
inline void kvflash_uq_append_queries(std::vector<const float *> & qs, const KvFlashUqStore & s, size_t qsize) {
    if (s.empty() || qsize == 0 || s.rows.size() != (size_t)s.n * qsize) return;
    for (int i = 0; i < s.n; i++) qs.push_back(s.rows.data() + (size_t)i * qsize);
}

// Why the feature must stay off for the process ("" = ids present, on). Tool results can only be told
// from human turns by the single <tool_response> token, so without it the feature fails closed.
inline std::string kvflash_uq_gate(int32_t im_start, int32_t user, int32_t im_end, int32_t tool_response) {
    if (im_start < 0) return "<|im_start|> id missing";
    if (user < 0) return "user id missing";
    if (im_end < 0) return "<|im_end|> id missing";
    if (tool_response < 0) return "<tool_response> is not a single token";
    return "";
}

}  // namespace luce::common
