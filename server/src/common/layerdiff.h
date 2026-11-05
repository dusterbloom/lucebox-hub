#pragma once

// LUCE_LAYERDIFF=<dir>: per-node, per-token-row bit hashes of a qwen35 target
// graph, to find the first op where spec-verify and plain decode disagree for
// the same token at the same position. Off (zero cost) unless the env is set.
//
// Run the server once as plain decode and once with --draft, each with
//   LUCE_LAYERDIFF=<dir> LUCE_LAYERDIFF_TAG=ar|spec [LUCE_LAYERDIFF_MAXCALLS=n]
// then compare with server/tools/layerdiff_compare.py <dir>/ar.tsv <dir>/spec.tsv.
//
// Every graph node is labelled "L<layer>/<op>#<k>" at build time and flagged as
// an output, so the allocator keeps it alive until it is read back. That also
// stops the CUDA backend fusing across it; compare with a run that has the
// flags off (LUCE_LAYERDIFF_KEEP=0 keeps fusion but only the last node of each
// fused group is trustworthy) if fusion is the suspect.

#include "ggml.h"
#include "ggml-backend.h"

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <mutex>
#include <string>
#include <unordered_map>
#include <vector>

namespace luce::layerdiff {

struct Row {
    int slot;
    int pos;
    int tok;
    int graph_row;   // index along the token dimension of the graph tensors
};

inline const char * dir() { return std::getenv("LUCE_LAYERDIFF"); }
inline bool on() { const char * d = dir(); return d && *d; }
inline bool keep_alive() {
    const char * k = std::getenv("LUCE_LAYERDIFF_KEEP");
    return !(k && k[0] == '0');
}

inline std::unordered_map<const ggml_tensor *, std::string> & labels() {
    static std::unordered_map<const ggml_tensor *, std::string> m;
    return m;
}
inline std::mutex & mu() { static std::mutex m; return m; }

// Expand gf up to `tip` and label every node added since `cursor` as layer `layer`.
inline void mark(ggml_cgraph * gf, ggml_tensor * tip, int layer, int & cursor) {
    if (!on()) return;
    if (tip) ggml_build_forward_expand(gf, tip);
    std::lock_guard<std::mutex> lock(mu());
    for (int i = cursor; i < ggml_graph_n_nodes(gf); ++i) {
        ggml_tensor * n = ggml_graph_node(gf, i);
        // ggml auto-names ("node_12", "x (view)") depend on graph position; use the op then.
        const bool coded = n->name[0] && !std::strstr(n->name, "node_") && !std::strstr(n->name, " (");
        const std::string base = coded ? n->name : ggml_op_name(n->op);
        char key[160];
        std::snprintf(key, sizeof(key), "L%02d/%s", layer, base.c_str());
        labels()[n] = key;
        if (keep_alive()) ggml_set_output(n);
    }
    cursor = ggml_graph_n_nodes(gf);
}

inline bool is_alias_op(ggml_op op) {
    return op == GGML_OP_VIEW || op == GGML_OP_RESHAPE || op == GGML_OP_PERMUTE ||
           op == GGML_OP_TRANSPOSE || op == GGML_OP_NONE;
}

struct Acc { uint64_t h = 1469598103934665603ull; double sum = 0, amax = 0; };

inline double elem_to_f(const uint8_t * p, ggml_type t) {
    if (t == GGML_TYPE_F32) { float v; std::memcpy(&v, p, 4); return v; }
    if (t == GGML_TYPE_F16) { ggml_fp16_t v; std::memcpy(&v, p, 2); return ggml_fp16_to_fp32(v); }
    if (t == GGML_TYPE_BF16) { ggml_bf16_t v; std::memcpy(&v, p, 2); return ggml_bf16_to_fp32(v); }
    return 0.0;
}

inline void feed(Acc & a, const uint8_t * p, size_t n) {
    size_t i = 0;
    for (; i + 8 <= n; i += 8) { uint64_t w; std::memcpy(&w, p + i, 8); a.h = (a.h ^ w) * 1099511628211ull; a.h ^= a.h >> 29; }
    for (; i < n; ++i) a.h = (a.h ^ p[i]) * 1099511628211ull;
}

inline std::string shape(const ggml_tensor * t) {
    char b[96];
    std::snprintf(b, sizeof(b), "%s:%lldx%lldx%lldx%lld", ggml_type_name(t->type),
        (long long) t->ne[0], (long long) t->ne[1], (long long) t->ne[2], (long long) t->ne[3]);
    return b;
}

// Reads back every labelled, row-shaped node of gf and appends one line per
// (row, node) to <dir>/<tag>.tsv. `n_rows` is the graph's token dimension.
inline void dump(ggml_cgraph * gf, const std::vector<Row> & rows, int n_rows) {
    if (!on() || rows.empty()) return;
    static int calls = 0;
    static const int max_calls = [] { const char * m = std::getenv("LUCE_LAYERDIFF_MAXCALLS"); return m ? std::atoi(m) : 64; }();
    if (calls >= max_calls) return;
    const int call = calls++;
    const char * tag_env = std::getenv("LUCE_LAYERDIFF_TAG");
    const std::string path = std::string(dir()) + "/" + (tag_env ? tag_env : "run") + ".tsv";
    std::FILE * f = std::fopen(path.c_str(), "a");
    if (!f) return;

    std::vector<int> col(rows.size()), ncols(rows.size());
    for (size_t i = 0; i < rows.size(); ++i) {
        for (size_t j = 0; j < rows.size(); ++j) if (rows[j].slot == rows[i].slot) {
            if (j < i) ++col[i];
            ++ncols[i];
        }
    }

    std::vector<uint8_t> buf;
    std::lock_guard<std::mutex> lock(mu());
    std::unordered_map<std::string, int> seen;
    for (int i = 0; i < ggml_graph_n_nodes(gf); ++i) {
        const ggml_tensor * n = ggml_graph_node(gf, i);
        auto it = labels().find(n);
        if (it == labels().end() || is_alias_op(n->op)) continue;
        if (n->type != GGML_TYPE_F32 && n->type != GGML_TYPE_F16 && n->type != GGML_TYPE_BF16 &&
            n->type != GGML_TYPE_I32) continue;
        int d = -1;
        for (int k = 3; k >= 1; --k) if (n->ne[k] == n_rows) { d = k; break; }
        // gated_delta_net packs [attn_out rows | final state | intermediates] along ne[1]; the first n_rows rows are the per-token output
        if (n->op == GGML_OP_GATED_DELTA_NET && n->ne[1] >= n_rows) d = 1;
        if (d < 0) continue;
        const size_t ts = ggml_type_size(n->type);
        if (n->nb[0] != ts) continue;
        // key: layer/op, output shape with the token dim as R, k-th such node in graph order
        std::string keyshape = ggml_type_name(n->type);
        keyshape += ":";   // the dims except the token one, so [C,1,1,R] and [C,R,1,1] layouts share a key
        for (int k = 0; k < 4; ++k) if (k != d && (k == 0 || n->ne[k] != 1)) keyshape += std::to_string((long long) n->ne[k]) + "x";
        const std::string base = it->second + "|" + keyshape;
        const std::string key = base + "#" + std::to_string(seen[base]++);
        buf.resize(ggml_nbytes(n));
        ggml_backend_tensor_get(n, buf.data(), 0, buf.size());
        std::string srcs;
        for (int s = 0; s < GGML_MAX_SRC && s < 3; ++s) if (n->src[s]) {
            if (!srcs.empty()) srcs += ';';
            srcs += shape(n->src[s]) + "@" + ggml_op_name(n->src[s]->op);
        }
        for (size_t ri = 0; ri < rows.size(); ++ri) {
            const Row & r = rows[ri];
            if (r.graph_row < 0 || r.graph_row >= n_rows) continue;
            Acc a;
            const int64_t ne[4] = {n->ne[0], n->ne[1], n->ne[2], n->ne[3]};
            int64_t idx[4] = {0, 0, 0, 0};
            idx[d] = r.graph_row;
            for (idx[3] = (d == 3 ? r.graph_row : 0); idx[3] < (d == 3 ? r.graph_row + 1 : ne[3]); ++idx[3])
            for (idx[2] = (d == 2 ? r.graph_row : 0); idx[2] < (d == 2 ? r.graph_row + 1 : ne[2]); ++idx[2])
            for (idx[1] = (d == 1 ? r.graph_row : 0); idx[1] < (d == 1 ? r.graph_row + 1 : ne[1]); ++idx[1]) {
                const uint8_t * p = buf.data() + idx[1] * n->nb[1] + idx[2] * n->nb[2] + idx[3] * n->nb[3];
                feed(a, p, (size_t) ne[0] * ts);
                if (n->type != GGML_TYPE_I32) {
                    for (int64_t e = 0; e < ne[0]; ++e) {
                        const double v = elem_to_f(p + e * ts, n->type);
                        a.sum += v;
                        a.amax = std::fmax(a.amax, std::fabs(v));
                    }
                }
            }
            std::fprintf(f, "C%d\t%d\t%d\t%d\t%d\t%d\t%s\t%s\t%s\t%s\t%016llx\t%.9g\t%.9g\t%s\n",
                call, r.slot, r.pos, r.tok, col[ri], ncols[ri], key.c_str(),
                n->name, ggml_op_name(n->op), shape(n).c_str(),
                (unsigned long long) a.h, a.sum, a.amax, srcs.c_str());
        }
    }
    std::fclose(f);
}

}  // namespace luce::layerdiff
