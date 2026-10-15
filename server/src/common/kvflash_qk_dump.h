// KVFLASH_QK_DUMP: diagnostic dump of the target-QK chunk scores at the end of
// every prefill (opt-in; the env var unset means nothing here runs).
//
// One JSON line per request is appended to <dir>/qk_dump.jsonl:
//   seq, prompt_hash, prompt_len, kv_offset, committed, chunk_tokens,
//   n_chunks (sealed chunks), n_pooled (chunks with pooled keys),
//   resident (chunk indices holding a pool block), n_q_tokens,
//   scores_last (query of the last prefill token),
//   scores_max  (elementwise max over the per-token queries of the prefill
//                batches ending within the last 512 prompt tokens; empty when
//                none were captured).
// Unscorable chunks carry the -2 sentinel of kvflash_qk_chunk_scores.
// Floats print with %.9g, which round-trips exactly.
//
// Pure (no ggml): the scores come from kvflash_qk_chunk_scores itself.

#pragma once

#include "kvflash_qk.h"

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <string>
#include <thread>
#include <vector>

namespace luce::common {

// Tokens per batch whose queries are captured (the last N of the batch).
constexpr int kQkDumpQueryTokens = 512;
// Only batches ending within this many tokens of the prompt end fold their queries.
constexpr int kQkDumpWindowTokens = 512;

struct KvFlashQkDumpMeta {
    uint64_t seq         = 0;
    uint64_t prompt_hash = 0;
    int      prompt_len  = 0;   // full prompt tokens when known, else the prefilled tokens
    int      kv_offset   = 0;   // tokens restored before this prefill
    int      committed   = 0;   // KV position at prefill end
    int      chunk_tokens = 64;
};

// Directory from KVFLASH_QK_DUMP, empty when unset.
inline std::string kvflash_qk_dump_dir() {
    const char * e = std::getenv("KVFLASH_QK_DUMP");
    return (e && *e) ? std::string(e) : std::string();
}

// FNV-1a 64 over the token ids.
inline uint64_t kvflash_qk_prompt_hash(const std::vector<int32_t> & ids) {
    uint64_t h = 1469598103934665603ULL;
    for (int32_t t : ids) {
        const uint32_t u = (uint32_t)t;
        for (int b = 0; b < 4; b++) {
            h ^= (uint64_t)((u >> (8 * b)) & 0xFF);
            h *= 1099511628211ULL;
        }
    }
    return h;
}

inline void kvflash_qk_dump_floats(std::string & s, const char * key, const std::vector<float> & v) {
    s += ",\""; s += key; s += "\":[";
    char buf[32];
    for (size_t i = 0; i < v.size(); i++) {
        std::snprintf(buf, sizeof buf, "%.9g", (double)v[i]);
        if (i) s += ',';
        s += buf;
    }
    s += ']';
}

// Fold the scores of n_q queries (n_q * qsize floats back to back) into
// `running`, an elementwise max over the sealed chunks. `running` grows to
// pooled_keys.size() with the -2 missing sentinel; entries only ever rise.
inline void kvflash_qk_fold_max(const std::vector<const float *> & pooled_keys,
                                const float * queries, int n_q,
                                const KvFlashQkDims & d, std::vector<float> & running) {
    const size_t qsize = (size_t)d.n_layers * d.n_q_heads * d.head_dim;
    if (running.size() < pooled_keys.size()) running.resize(pooled_keys.size(), -2.0f);
    if (n_q <= 0) return;
    // Queries are independent and the fold is a max, so threads give the same result.
    const int nt = std::max(1, std::min<int>({n_q, 12, (int)std::thread::hardware_concurrency()}));
    std::vector<std::vector<float>> part((size_t)nt, running);
    auto work = [&](int w) {
        std::vector<float> tmp;
        for (int t = w; t < n_q; t += nt) {
            kvflash_qk_chunk_scores(pooled_keys, queries + (size_t)t * qsize, d, tmp);
            for (size_t c = 0; c < tmp.size(); c++) part[(size_t)w][c] = std::max(part[(size_t)w][c], tmp[c]);
        }
    };
    std::vector<std::thread> th;
    for (int w = 1; w < nt; w++) th.emplace_back(work, w);
    work(0);
    for (auto & x : th) x.join();
    for (const auto & p : part)
        for (size_t c = 0; c < running.size(); c++) running[c] = std::max(running[c], p[c]);
}

// scores_last: scores of the last prefill token's query over the sealed chunks.
// scores_max: running max over the folded per-token queries (empty when n_q = 0).
// resident[c] != 0 when chunk c holds a pool block. n_pooled: chunks with pooled keys.
inline std::string kvflash_qk_dump_line(const KvFlashQkDumpMeta & m, int n_pooled,
                                        const std::vector<float> & scores_last,
                                        const std::vector<float> & scores_max, int n_q,
                                        const std::vector<uint8_t> & resident) {
    char head[320];
    std::snprintf(head, sizeof head,
        "{\"seq\":%llu,\"prompt_hash\":\"%016llx\",\"prompt_len\":%d,\"kv_offset\":%d,"
        "\"committed\":%d,\"chunk_tokens\":%d,\"n_chunks\":%d,\"n_pooled\":%d,\"n_q_tokens\":%d",
        (unsigned long long)m.seq, (unsigned long long)m.prompt_hash, m.prompt_len, m.kv_offset,
        m.committed, m.chunk_tokens, (int)scores_last.size(), n_pooled, n_q);
    std::string s = head;
    s += ",\"resident\":[";
    bool first = true;
    for (size_t c = 0; c < resident.size(); c++) {
        if (!resident[c]) continue;
        if (!first) s += ',';
        s += std::to_string(c);
        first = false;
    }
    s += ']';
    kvflash_qk_dump_floats(s, "scores_last", scores_last);
    kvflash_qk_dump_floats(s, "scores_max", n_q > 0 ? scores_max : std::vector<float>());
    s += '}';
    return s;
}

// Append one line to <dir>/qk_dump.jsonl. An empty dir is a no-op.
inline bool kvflash_qk_dump_append(const std::string & dir, const std::string & line) {
    if (dir.empty()) return false;
    std::error_code ec;
    std::filesystem::create_directories(dir, ec);
    std::ofstream f(std::filesystem::path(dir) / "qk_dump.jsonl", std::ios::app);
    if (!f) return false;
    f << line << '\n';
    return (bool)f;
}

} // namespace luce::common
