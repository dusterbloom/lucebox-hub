// Reader for the AM reference fixtures (generator: optimizations/kvflash/lucebox/a1/am_fixture_dump.py).
//
// One directory per case (traj1_first, traj1_12k, article_r1e, ...) holds manifest.json, the
// per-case tokens.i32 / positions.i32 and one raw little-endian file per layer and tensor:
//   L<layer:02d>_<name>.f32 (or .i32 for idx), in ggml element order, fastest dimension first:
//     q [d, n, n_head]  k, v [d, n, n_kv]  scores [n, n_kv]  idx [slots, n_kv] (int32, rank order)
//     beta [slots, n_kv]  c2 [d, slots, n_kv]
// Header only, no JSON library: the manifest is searched for its "layers" array, and every file is
// checked against the expected element count by its size.
//
// manifest_int (M4 addition) reads head_dim / n_tokens / n_head / n_kv / slots from the manifest text.
// Used by the M5 fit gate (test_am_fit_solve) and meant for the M4 selection gate.
#pragma once

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

#include <dirent.h>

namespace am_fixture {

struct Case {
    std::string      dir;
    std::string      name;
    std::vector<int> layers;  // full-attention layer ids, in manifest order
};

inline bool read_file(const std::string & path, std::string & out) {
    std::ifstream f(path, std::ios::binary);
    if (!f) return false;
    std::ostringstream ss;
    ss << f.rdbuf();
    out = ss.str();
    return true;
}

// Loads dir/manifest.json. False (with *err) if it is missing or lists no layers.
inline bool load_case(const std::string & dir, Case & c, std::string * err = nullptr) {
    std::string m;
    if (!read_file(dir + "/manifest.json", m)) { if (err) *err = "no manifest.json in " + dir; return false; }
    c = Case{};
    c.dir = dir;
    const size_t slash = dir.find_last_of('/');
    c.name = slash == std::string::npos ? dir : dir.substr(slash + 1);
    const size_t key = m.find("\"layers\"");
    const size_t open = key == std::string::npos ? key : m.find('[', key);
    const size_t close = open == std::string::npos ? open : m.find(']', open);
    if (close == std::string::npos) { if (err) *err = "no layers array in manifest of " + dir; return false; }
    const char * p = m.c_str() + open + 1;
    const char * end = m.c_str() + close;
    while (p < end) {
        char * next = nullptr;
        const long v = std::strtol(p, &next, 10);
        if (next == p) { ++p; continue; }
        c.layers.push_back((int) v);
        p = next;
    }
    if (c.layers.empty()) { if (err) *err = "empty layers array in manifest of " + dir; return false; }
    return true;
}

// Integer value of a top-level-looking "key": N in the manifest text (first occurrence).
inline bool manifest_int(const std::string & manifest, const char * key, int64_t & out) {
    const std::string k = std::string("\"") + key + "\"";
    const size_t at = manifest.find(k);
    if (at == std::string::npos) return false;
    const char * p = manifest.c_str() + at + k.size();
    while (*p == ' ' || *p == ':' || *p == '\t' || *p == '\n') ++p;
    char * end = nullptr;
    const long long v = std::strtoll(p, &end, 10);
    if (end == p) return false;
    out = v;
    return true;
}

template <class T>
inline bool read_raw(const std::string & path, std::vector<T> & out, size_t expect_elems, std::string * err) {
    std::string bytes;
    if (!read_file(path, bytes)) { if (err) *err = "cannot read " + path; return false; }
    if (bytes.size() != expect_elems * sizeof(T)) {
        if (err) *err = path + ": " + std::to_string(bytes.size()) + " bytes, expected " + std::to_string(expect_elems * sizeof(T));
        return false;
    }
    out.resize(expect_elems);
    std::memcpy(out.data(), bytes.data(), bytes.size());  // little-endian host
    return true;
}

// Per-layer tensor L<layer>_<name>.f32 / .i32.
inline std::string layer_path(const Case & c, int layer, const char * name, const char * ext) {
    char b[64];
    std::snprintf(b, sizeof b, "/L%02d_%s.%s", layer, name, ext);
    return c.dir + b;
}
inline bool read_f32(const Case & c, int layer, const char * name, size_t n, std::vector<float> & out, std::string * err = nullptr) {
    return read_raw(layer_path(c, layer, name, "f32"), out, n, err);
}
inline bool read_i32(const Case & c, int layer, const char * name, size_t n, std::vector<int32_t> & out, std::string * err = nullptr) {
    return read_raw(layer_path(c, layer, name, "i32"), out, n, err);
}
// Per-case tensor <name>.i32 (tokens, positions).
inline bool read_case_i32(const Case & c, const char * name, size_t n, std::vector<int32_t> & out, std::string * err = nullptr) {
    return read_raw(c.dir + "/" + name + ".i32", out, n, err);
}

// Subdirectories of `root` that hold a manifest.json, sorted by name. Empty if root is unset.
inline std::vector<std::string> case_dirs(const char * root) {
    std::vector<std::string> dirs;
    if (!root || !*root) return dirs;
    if (DIR * d = opendir(root)) {
        while (const dirent * e = readdir(d)) {
            if (e->d_name[0] == '.') continue;
            const std::string sub = std::string(root) + "/" + e->d_name;
            std::ifstream probe(sub + "/manifest.json");
            if (probe) dirs.push_back(sub);
        }
        closedir(d);
    }
    std::sort(dirs.begin(), dirs.end());
    return dirs;
}

}  // namespace am_fixture
