// Reader for the AM reference fixtures written by
// optimizations/kvflash/lucebox/a1/am_fixture_dump.py (manifest.json plus raw
// little-endian tensor files, no header; shape_ggml lists the fastest
// dimension first), and the arithmetic the M3 cosine gate needs.
// Minimal on purpose: the M5 fit tests carry their own reader and the two
// unify at integration.
#pragma once

#include <nlohmann/json.hpp>

#include "ggml.h"

#include <algorithm>
#include <cctype>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <string>
#include <vector>

namespace luce::amfix {

struct File {
    int layer = -1;  // -1: not per layer
    std::string name, file, dtype;
    std::vector<int64_t> shape_ggml;
    size_t bytes = 0;
};

struct Fixture {
    std::string dir;
    nlohmann::json manifest;
    std::vector<File> files;

    // Parses manifest.json and checks every listed file exists with exactly
    // the byte count the manifest (and shape and dtype) claims.
    static bool open(const std::string & dir, Fixture & out, std::string & err) {
        namespace fs = std::filesystem;
        out = Fixture{};
        out.dir = dir;
        std::ifstream mf(fs::path(dir) / "manifest.json");
        if (!mf) { err = "cannot open " + dir + "/manifest.json"; return false; }
        try {
            out.manifest = nlohmann::json::parse(mf);
            for (const auto & e : out.manifest.at("files")) {
                File f;
                f.layer = e.at("layer").is_null() ? -1 : e.at("layer").get<int>();
                f.name = e.at("name").get<std::string>();
                f.file = e.at("file").get<std::string>();
                f.dtype = e.at("dtype").get<std::string>();
                f.shape_ggml = e.at("shape_ggml").get<std::vector<int64_t>>();
                f.bytes = e.at("bytes").get<size_t>();
                if (f.dtype != "float32" && f.dtype != "int32") { err = f.file + ": unsupported dtype " + f.dtype; return false; }
                // Every dimension positive and bounded, the product checked, before any size is trusted.
                int64_t n = 1;
                constexpr int64_t kMaxDim = int64_t(1) << 31, kMaxElems = int64_t(1) << 40;
                for (int64_t s : f.shape_ggml)
                    if (s < 1 || s > kMaxDim || __builtin_mul_overflow(n, s, &n) || n > kMaxElems) {
                        err = f.file + ": shape_ggml has a non-positive, oversized or overflowing dimension"; return false;
                    }
                if ((uint64_t)n * 4 != f.bytes) { err = f.file + ": manifest bytes disagree with shape"; return false; }
                std::error_code ec;
                const auto sz = fs::file_size(fs::path(dir) / f.file, ec);
                if (ec || sz != f.bytes) { err = f.file + ": missing or size differs from manifest"; return false; }
                out.files.push_back(std::move(f));
            }
        } catch (const std::exception & e) {
            err = std::string("manifest.json: ") + e.what();
            return false;
        }
        return true;
    }

    const File * find(const std::string & name, int layer = -1) const {
        for (const File & f : files) if (f.name == name && f.layer == layer) return &f;
        return nullptr;
    }

    std::vector<int> layers() const {
        std::vector<int> l;
        for (const auto & v : manifest.value("layers", nlohmann::json::array())) l.push_back(v.get<int>());
        return l;
    }

    bool read_f32(const std::string & name, int layer, std::vector<float> & out, std::string & err) const {
        return read(name, layer, "float32", out, err);
    }
    bool read_i32(const std::string & name, std::vector<int32_t> & out, std::string & err) const {
        return read(name, -1, "int32", out, err);
    }

private:
    template <class T>
    bool read(const std::string & name, int layer, const char * dtype, std::vector<T> & out, std::string & err) const {
        const File * f = find(name, layer);
        if (!f) { err = "no entry " + name + " layer " + std::to_string(layer); return false; }
        if (f->dtype != dtype) { err = f->file + ": dtype is " + f->dtype + ", asked for " + dtype; return false; }
        out.resize(f->bytes / sizeof(T));
        std::ifstream in(std::filesystem::path(dir) / f->file, std::ios::binary);
        in.read((char *)out.data(), (std::streamsize)f->bytes);
        if (!in) { err = f->file + ": short read"; return false; }
        return true;
    }
};

// ---- cosine arithmetic ----

struct HeadCosine {
    std::vector<double> mean;  // per head: mean over tokens of cos(ref row, engine row)
    std::vector<double> min;   // per head: worst token
    std::vector<double> ratio; // per head: ||engine|| / ||ref|| over the whole head
};

// ref: F32 [H][n][d] (the dump); eng: F16 [H][n][d] (the ring plane of one layer).
// A row with a zero norm on either side counts as cosine 0.
inline HeadCosine cosine_per_head(const float * ref, const uint16_t * eng, int H, int n, int d) {
    HeadCosine r;
    for (int h = 0; h < H; h++) {
        double sum = 0, lo = 1.0, ref2 = 0, eng2 = 0;
        for (int t = 0; t < n; t++) {
            const size_t o = ((size_t)h * n + t) * d;
            double dot = 0, na = 0, nb = 0;
            for (int i = 0; i < d; i++) {
                const double a = ref[o + i], b = ggml_fp16_to_fp32(eng[o + i]);
                dot += a * b; na += a * a; nb += b * b;
            }
            const double c = (na > 0 && nb > 0) ? dot / std::sqrt(na * nb) : 0.0;
            sum += c; lo = std::min(lo, c); ref2 += na; eng2 += nb;
        }
        r.mean.push_back(sum / n);
        r.min.push_back(lo);
        r.ratio.push_back(ref2 > 0 ? std::sqrt(eng2 / ref2) : 0.0);
    }
    return r;
}

// Every head must clear the gate on its own: the layer average hides one bad head
// (one negated head of 24 still averages 0.917), and cosine cannot see a scale error.
// Constants are tripwires (the plan fixes no number: "Q4-vs-BF16 noise floor"), to be
// replaced by the measured floor.
struct HeadGate { double min_cos = 0.9, ratio_lo = 0.95, ratio_hi = 1.05; };

inline std::string head_violation(const HeadCosine & r, HeadGate g = {}) {
    std::string out;
    for (size_t h = 0; h < r.mean.size(); h++) {
        const bool ok = r.mean[h] > g.min_cos && r.ratio[h] >= g.ratio_lo && r.ratio[h] <= g.ratio_hi;  // NaN fails
        if (!ok) {
            char b[96];
            std::snprintf(b, sizeof b, "head %zu: cos %.4f norm ratio %.4f; ", h, r.mean[h], r.ratio[h]);
            out += b;
        }
    }
    return out;
}

struct Summary { double min = 0, median = 0, mean = 0; };

inline Summary summarize(std::vector<double> v) {
    Summary s;
    if (v.empty()) return s;
    std::sort(v.begin(), v.end());
    s.min = v.front();
    s.median = v.size() % 2 ? v[v.size() / 2] : 0.5 * (v[v.size() / 2 - 1] + v[v.size() / 2]);
    double t = 0; for (double x : v) t += x;
    s.mean = t / (double)v.size();
    return s;
}

// ---- same base model? ----

struct GgufFacts {
    int n_layer = 0, full_attention_interval = 0, n_head = 0, head_dim = 0;
    std::vector<std::string> names;  // general.name, basename, ... as found
};

inline std::string alnum_lower(const std::string & s) {
    std::string o;
    for (char c : s) if (std::isalnum((unsigned char)c)) o += (char)std::tolower((unsigned char)c);
    return o;
}

// "" when the GGUF and the dump describe the same base model, else why not.
// Structure (head counts, head_dim, which layers are full attention) is always
// compared; the name only when both sides carry one.
inline std::string identity_mismatch(const nlohmann::json & manifest, const GgufFacts & g) {
    auto want = [&](const char * k, int have) -> std::string {
        if (manifest.contains(k) && manifest[k].get<int>() != have)
            return std::string(k) + ": dump " + std::to_string(manifest[k].get<int>()) + " vs gguf " + std::to_string(have);
        return "";
    };
    for (const std::string & e : {want("n_head", g.n_head), want("head_dim", g.head_dim)}) if (!e.empty()) return e;
    if (g.full_attention_interval > 0) {
        std::vector<int> expect;
        for (int l = 0; l < g.n_layer; l++) if (l % g.full_attention_interval == g.full_attention_interval - 1) expect.push_back(l);
        std::vector<int> have;
        for (const auto & v : manifest.value("layers", nlohmann::json::array())) have.push_back(v.get<int>());
        if (have != expect) return "full-attention layers differ: dump lists " + std::to_string(have.size()) + ", gguf has " + std::to_string(expect.size());
    }
    const auto & id = manifest.value("model_identity", nlohmann::json::object());
    std::string path;
    for (const char * k : {"_name_or_path", "model_path"})
        if (path.empty() && id.contains(k) && id[k].is_string()) path = id[k].get<std::string>();
    while (!path.empty() && path.back() == '/') path.pop_back();
    const std::string dump_name = alnum_lower(std::filesystem::path(path).filename().string());
    if (!dump_name.empty() && !g.names.empty()) {
        bool found = false;
        for (const std::string & n : g.names) found |= alnum_lower(n).find(dump_name) != std::string::npos;
        if (!found) return "model name differs: dump '" + dump_name + "' not in the gguf names";
    }
    return "";
}

}  // namespace luce::amfix
