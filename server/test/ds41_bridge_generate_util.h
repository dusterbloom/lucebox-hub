// Pure logic for ds41_bridge_generate (no GPU, no ggml, no IO beyond read_file):
// DS41ROWS parsing and shape validation, DSML tool-call block detection on
// token ids, the next-step comparison, tool-name extraction, sha256, and the
// small JSON field readers used for resume and the summary.
//
// PRIVACY: nothing here formats session or generated content. Callers emit
// counts, indices, booleans and sha256 hashes only. Tested by
// test/unit/test_ds41_generate_util.cpp (host-only, g++ -std=c++17).
#pragma once

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

namespace ds41gen {

// ── sha256 ──────────────────────────────────────────────────────────────

inline std::string sha256_hex(const uint8_t * data, size_t len) {
    static const uint32_t K[64] = {
        0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5, 0x3956c25b, 0x59f111f1, 0x923f82a4, 0xab1c5ed5,
        0xd807aa98, 0x12835b01, 0x243185be, 0x550c7dc3, 0x72be5d74, 0x80deb1fe, 0x9bdc06a7, 0xc19bf174,
        0xe49b69c1, 0xefbe4786, 0x0fc19dc6, 0x240ca1cc, 0x2de92c6f, 0x4a7484aa, 0x5cb0a9dc, 0x76f988da,
        0x983e5152, 0xa831c66d, 0xb00327c8, 0xbf597fc7, 0xc6e00bf3, 0xd5a79147, 0x06ca6351, 0x14292967,
        0x27b70a85, 0x2e1b2138, 0x4d2c6dfc, 0x53380d13, 0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85,
        0xa2bfe8a1, 0xa81a664b, 0xc24b8b70, 0xc76c51a3, 0xd192e819, 0xd6990624, 0xf40e3585, 0x106aa070,
        0x19a4c116, 0x1e376c08, 0x2748774c, 0x34b0bcb5, 0x391c0cb3, 0x4ed8aa4a, 0x5b9cca4f, 0x682e6ff3,
        0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208, 0x90befffa, 0xa4506ceb, 0xbef9a3f7, 0xc67178f2};
    uint32_t h[8] = {0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a,
                     0x510e527f, 0x9b05688c, 0x1f83d9ab, 0x5be0cd19};
    std::vector<uint8_t> m(data, data + len);
    m.push_back(0x80);
    while (m.size() % 64 != 56) m.push_back(0);
    const uint64_t bits = (uint64_t) len * 8;
    for (int i = 7; i >= 0; --i) m.push_back((uint8_t) (bits >> (8 * i)));
    const auto rotr = [](uint32_t x, int n) { return (x >> n) | (x << (32 - n)); };
    for (size_t off = 0; off < m.size(); off += 64) {
        uint32_t w[64];
        for (int i = 0; i < 16; ++i)
            w[i] = (uint32_t) m[off + 4 * i] << 24 | (uint32_t) m[off + 4 * i + 1] << 16 |
                   (uint32_t) m[off + 4 * i + 2] << 8 | (uint32_t) m[off + 4 * i + 3];
        for (int i = 16; i < 64; ++i) {
            const uint32_t s0 = rotr(w[i - 15], 7) ^ rotr(w[i - 15], 18) ^ (w[i - 15] >> 3);
            const uint32_t s1 = rotr(w[i - 2], 17) ^ rotr(w[i - 2], 19) ^ (w[i - 2] >> 10);
            w[i] = w[i - 16] + s0 + w[i - 7] + s1;
        }
        uint32_t a = h[0], b = h[1], c = h[2], d = h[3], e = h[4], f = h[5], g = h[6], hh = h[7];
        for (int i = 0; i < 64; ++i) {
            const uint32_t S1 = rotr(e, 6) ^ rotr(e, 11) ^ rotr(e, 25);
            const uint32_t ch = (e & f) ^ (~e & g);
            const uint32_t t1 = hh + S1 + ch + K[i] + w[i];
            const uint32_t S0 = rotr(a, 2) ^ rotr(a, 13) ^ rotr(a, 22);
            const uint32_t mj = (a & b) ^ (a & c) ^ (b & c);
            const uint32_t t2 = S0 + mj;
            hh = g; g = f; f = e; e = d + t1; d = c; c = b; b = a; a = t1 + t2;
        }
        h[0] += a; h[1] += b; h[2] += c; h[3] += d; h[4] += e; h[5] += f; h[6] += g; h[7] += hh;
    }
    char out[65];
    for (int i = 0; i < 8; ++i) std::snprintf(out + 8 * i, 9, "%08x", h[i]);
    return std::string(out, 64);
}

// sha256 of the ids as little-endian int32 (the bytes of the ids file).
inline std::string sha256_ids(const int32_t * ids, size_t n) {
    std::vector<uint8_t> b(n * 4);
    for (size_t i = 0; i < n; ++i)
        for (int k = 0; k < 4; ++k) b[4 * i + k] = (uint8_t) ((uint32_t) ids[i] >> (8 * k));
    return sha256_hex(b.data(), b.size());
}

// ── DS41ROWS ────────────────────────────────────────────────────────────
// Little-endian: "DS41ROWS", u32 version=1, u32 n_layers; per layer
// u32 layer, u32 ratio, u32 n_rows, u32 width; then float16 row-major
// [n_rows, width] per layer in header order.

struct RowsLayer {
    uint32_t layer = 0, ratio = 0, n_rows = 0, width = 0;
    std::vector<uint16_t> f16;  // n_rows * width raw half bits
};

inline float half_to_float(uint16_t h) {
    const int e = (h >> 10) & 0x1F, m = h & 0x3FF;
    const float v = e == 0 ? std::ldexp((float) m, -24)
                  : e == 31 ? (m ? NAN : INFINITY)
                  : std::ldexp((float) (m | 0x400), e - 25);
    return h & 0x8000 ? -v : v;
}

// ||b - a|| / ||a|| over n half values (a = reference).
inline double rel_l2_f16(const uint16_t * a, const uint16_t * b, size_t n) {
    double num = 0, den = 0;
    for (size_t i = 0; i < n; ++i) {
        const double x = half_to_float(a[i]), d = half_to_float(b[i]) - x;
        num += d * d;
        den += x * x;
    }
    return den > 0 ? std::sqrt(num / den) : (num > 0 ? INFINITY : 0.0);
}

struct RowsFile {
    std::vector<RowsLayer> layers;
    const RowsLayer * find(uint32_t layer) const {
        for (const RowsLayer & l : layers) if (l.layer == layer) return &l;
        return nullptr;
    }
};

inline bool parse_rows(const uint8_t * p, size_t n, RowsFile & out, std::string & err) {
    out.layers.clear();
    const auto u32 = [&](size_t off) {
        return (uint32_t) p[off] | (uint32_t) p[off + 1] << 8 | (uint32_t) p[off + 2] << 16 |
               (uint32_t) p[off + 3] << 24;
    };
    if (n < 16 || std::memcmp(p, "DS41ROWS", 8) != 0) { err = "bad magic"; return false; }
    if (u32(8) != 1) { err = "unsupported version"; return false; }
    const uint32_t nl = u32(12);
    if (nl > 4096) { err = "implausible n_layers"; return false; }
    size_t off = 16;
    if (n - off < (size_t) nl * 16) { err = "truncated layer headers"; return false; }
    out.layers.resize(nl);
    for (uint32_t i = 0; i < nl; ++i, off += 16) {
        RowsLayer & l = out.layers[i];
        l.layer = u32(off); l.ratio = u32(off + 4); l.n_rows = u32(off + 8); l.width = u32(off + 12);
        if (l.ratio == 0 || l.width == 0) { err = "zero ratio or width"; out.layers.clear(); return false; }
        for (uint32_t j = 0; j < i; ++j)
            if (out.layers[j].layer == l.layer) { err = "duplicate layer"; out.layers.clear(); return false; }
    }
    for (RowsLayer & l : out.layers) {
        const uint64_t count = (uint64_t) l.n_rows * l.width;
        if ((uint64_t) (n - off) / 2 < count) { err = "truncated data"; out.layers.clear(); return false; }
        l.f16.resize((size_t) count);
        for (uint64_t k = 0; k < count; ++k, off += 2)
            l.f16[(size_t) k] = (uint16_t) (p[off] | p[off + 1] << 8);
    }
    if (off != n) { err = "trailing bytes"; out.layers.clear(); return false; }
    return true;
}

inline bool read_file(const std::string & path, std::vector<uint8_t> & out) {
    FILE * f = std::fopen(path.c_str(), "rb");
    if (!f) return false;
    std::fseek(f, 0, SEEK_END);
    const long bytes = std::ftell(f);
    std::fseek(f, 0, SEEK_SET);
    out.resize(bytes < 0 ? 0 : (size_t) bytes);
    const bool ok = bytes >= 0 && std::fread(out.data(), 1, out.size(), f) == out.size();
    std::fclose(f);
    return ok;
}

// A rows layer against its snapshot tensor (ggml ne0 = width, ne1 = rows) and
// the session cut s. Exact equality: a file with fewer rows than the snapshot
// holds would leave stale native rows in place, so it fails like a mismatch.
inline bool validate_rows_layer(const RowsLayer & l, int64_t s, int64_t ne0, int64_t ne1,
                                std::string & err) {
    char b[200];
    if ((int64_t) l.width != ne0) {
        std::snprintf(b, sizeof b, "layer %u width %u != tensor ne0 %lld", l.layer, l.width, (long long) ne0);
        err = b; return false;
    }
    if ((int64_t) l.n_rows != ne1) {
        std::snprintf(b, sizeof b, "layer %u rows %u != tensor ne1 %lld", l.layer, l.n_rows, (long long) ne1);
        err = b; return false;
    }
    if (s % (int64_t) l.ratio != 0 || (int64_t) l.n_rows != s / (int64_t) l.ratio) {
        std::snprintf(b, sizeof b, "layer %u rows %u != s/ratio (%lld/%u)", l.layer, l.n_rows,
                      (long long) s, l.ratio);
        err = b; return false;
    }
    return true;
}

// ── DSML tool-call block on token ids ───────────────────────────────────

struct ToolSpan {
    int64_t begin = -1;      // first token of the open marker (-1: none seen)
    int64_t end = -1;        // one past the last token of the close marker
    bool complete = false;
};

inline bool ends_with(const std::vector<int32_t> & v, const std::vector<int32_t> & m) {
    return !m.empty() && v.size() >= m.size() &&
           std::equal(m.begin(), m.end(), v.end() - (std::ptrdiff_t) m.size());
}

// Incremental detector: feed tokens one at a time; push() returns true once a
// close marker completes after an open marker. Used as the generation stop.
class ToolBlockTracker {
public:
    ToolBlockTracker(std::vector<int32_t> open, std::vector<int32_t> close)
        : open_(std::move(open)), close_(std::move(close)) {}
    bool push(int32_t tok) {
        toks_.push_back(tok);
        if (span_.complete) return true;
        if (span_.begin < 0) {
            if (ends_with(toks_, open_)) span_.begin = (int64_t) toks_.size() - (int64_t) open_.size();
        } else if (ends_with(toks_, close_) &&
                   (int64_t) toks_.size() - (int64_t) close_.size() >= span_.begin + (int64_t) open_.size()) {
            span_.end = (int64_t) toks_.size();
            span_.complete = true;
        }
        return span_.complete;
    }
    const ToolSpan & span() const { return span_; }
    const std::vector<int32_t> & tokens() const { return toks_; }
private:
    std::vector<int32_t> open_, close_, toks_;
    ToolSpan span_;
};

// First complete open..close block of `ids`. If only an open marker is seen,
// begin is set and complete is false; if none, begin is -1.
inline ToolSpan find_tool_block(const std::vector<int32_t> & ids, const std::vector<int32_t> & open,
                                const std::vector<int32_t> & close) {
    ToolBlockTracker t(open, close);
    for (int32_t id : ids) if (t.push(id)) break;
    return t.span();
}

// Number of (non-overlapping, left to right) occurrences of `m` in `ids`.
inline int count_marker(const std::vector<int32_t> & ids, const std::vector<int32_t> & m) {
    int n = 0;
    if (m.empty()) return 0;
    for (size_t i = 0; i + m.size() <= ids.size();) {
        if (std::equal(m.begin(), m.end(), ids.begin() + (std::ptrdiff_t) i)) { ++n; i += m.size(); }
        else ++i;
    }
    return n;
}

// Name of the first invoke in a decoded DSML block. V4.1 spells the tag
// `<｜DSML｜ invoke name="X">`, V4 `<｜DSML｜invoke name="X">`; both contain
// `invoke name="`. Empty when absent or unterminated.
inline std::string extract_tool_name(const std::string & decoded) {
    static const std::string key = "invoke name=\"";
    const size_t p = decoded.find(key);
    if (p == std::string::npos) return {};
    const size_t b = p + key.size(), e = decoded.find('"', b);
    return e == std::string::npos ? std::string() : decoded.substr(b, e - b);
}

// ── next-step comparison ────────────────────────────────────────────────

struct GenInfo {
    std::vector<int32_t> tokens;  // every generated token (including a final EOS)
    ToolSpan span;                // tool block inside `tokens`
    std::string tool_name;        // from the in-memory decoded block; never written
};

struct StepCompare {
    bool same_step = false;
    bool same_tool = false;
    int64_t first_divergence = -1;  // -1: the two generations are identical
};

inline int64_t first_divergence(const std::vector<int32_t> & a, const std::vector<int32_t> & b) {
    const size_t n = std::min(a.size(), b.size());
    for (size_t i = 0; i < n; ++i) if (a[i] != b[i]) return (int64_t) i;
    return a.size() == b.size() ? -1 : (int64_t) n;
}

inline StepCompare compare_next_step(const GenInfo & ref, const GenInfo & arm) {
    StepCompare c;
    c.first_divergence = first_divergence(ref.tokens, arm.tokens);
    if (ref.span.complete && arm.span.complete) {
        const auto rb = ref.tokens.begin() + ref.span.begin, re = ref.tokens.begin() + ref.span.end;
        const auto ab = arm.tokens.begin() + arm.span.begin, ae = arm.tokens.begin() + arm.span.end;
        c.same_step = (re - rb) == (ae - ab) && std::equal(rb, re, ab);
        c.same_tool = !ref.tool_name.empty() && ref.tool_name == arm.tool_name;
    }
    return c;
}

enum class StopKind { Tool, Eos, Cap };
inline const char * stop_name(StopKind k) {
    return k == StopKind::Tool ? "tool" : k == StopKind::Eos ? "eos" : "cap";
}
inline StopKind classify_stop(const GenInfo & g, const std::vector<int32_t> & eos_ids) {
    if (g.span.complete) return StopKind::Tool;
    if (!g.tokens.empty() &&
        std::find(eos_ids.begin(), eos_ids.end(), g.tokens.back()) != eos_ids.end()) return StopKind::Eos;
    return StopKind::Cap;
}

// NLL of `tok` and whether it is the argmax (first max wins, like the
// engine's greedy sampler) for one full-vocab logit row, without copying it.
inline void score_step(const float * lg, size_t n, int32_t tok, double & nll, bool & top1) {
    double mx = -INFINITY;
    size_t arg = 0;
    for (size_t i = 0; i < n; ++i) if ((double) lg[i] > mx) { mx = lg[i]; arg = i; }
    double sum = 0.0;
    for (size_t i = 0; i < n; ++i) sum += std::exp((double) lg[i] - mx);
    nll = mx + std::log(sum) - (double) lg[(size_t) tok];
    top1 = (int32_t) arg == tok;
}

// ── tiny JSON helpers (flat objects written by this tool) ───────────────

// Raw value text of "key": up to the next ',' or '}' (quotes stripped).
inline std::string json_field(const std::string & line, const std::string & key) {
    const std::string pat = "\"" + key + "\":";
    const size_t p = line.find(pat);
    if (p == std::string::npos) return {};
    size_t b = p + pat.size();
    while (b < line.size() && (line[b] == ' ' || line[b] == '\n' || line[b] == '\t')) ++b;
    const size_t e = line.find_first_of(",}\n", b);
    std::string v = line.substr(b, e == std::string::npos ? std::string::npos : e - b);
    v.erase(std::remove(v.begin(), v.end(), '"'), v.end());
    while (!v.empty() && v.back() == ' ') v.pop_back();
    return v;
}

inline bool json_int(const std::string & text, const std::string & key, int64_t & out) {
    const std::string v = json_field(text, key);
    if (v.empty()) return false;
    char * end = nullptr;
    const long long x = std::strtoll(v.c_str(), &end, 10);
    if (end == v.c_str()) return false;
    out = x;
    return true;
}

// "null" for NaN/inf, else %.6f.
inline std::string json_num(double v) {
    if (!std::isfinite(v)) return "null";
    char b[64];
    std::snprintf(b, sizeof b, "%.6f", v);
    return b;
}

}  // namespace ds41gen
