#include "common/am_fit.h"

#include "ggml-alloc.h"

#include <algorithm>
#include <cmath>
#include <numeric>

namespace luce::am {

namespace {

// Scores of one query chunk, summed over queries and over the G query heads of
// every KV head, added in place into `acc` [1, T, n_kv]. All softmax work is
// fp32. Layout trick: ggml_mul_mat broadcasts src0 (K, n_kv) over src1 (Q,
// n_head) by integer division of the head index, which is exactly the
// contiguous GQA grouping head -> head / G.
void build_chunk(ggml_context * ctx, ggml_cgraph * gf, const ScoreDims & D, ggml_tensor * q16,
                 ggml_tensor * kt, ggml_tensor * acc, int q0, int nq) {
    const int G = D.n_head / D.n_kv_head;
    ggml_tensor * qc = ggml_view_3d(ctx, q16, D.head_dim, nq, D.n_head, q16->nb[1], q16->nb[2],
                                    (size_t) q0 * q16->nb[1]);
    ggml_tensor * qf = ggml_cast(ctx, qc, GGML_TYPE_F32);                          // [d, nq, n_head]
    ggml_tensor * s  = ggml_mul_mat(ctx, kt, qf);                                  // [T, nq, n_head]
    ggml_tensor * w  = ggml_soft_max_ext(ctx, s, nullptr, 1.0f / std::sqrt((float) D.head_dim), 0.0f);
    ggml_tensor * w2 = ggml_sqr(ctx, w);
    ggml_tensor * wt = ggml_cont(ctx, ggml_permute(ctx, w2, 1, 0, 2, 3));          // [nq, T, n_head]
    ggml_tensor * r  = ggml_sum_rows(ctx, wt);                                     // [1, T, n_head]
    ggml_tensor * r2 = ggml_reshape_3d(ctx, r, D.n_keys, G, D.n_kv_head);          // head = h*G + g
    ggml_tensor * rt = ggml_cont(ctx, ggml_permute(ctx, r2, 1, 0, 2, 3));          // [G, T, n_kv]
    ggml_tensor * p  = ggml_sum_rows(ctx, rt);                                     // [1, T, n_kv]
    ggml_build_forward_expand(gf, ggml_add_inplace(ctx, acc, p));
}

}  // namespace

bool score_keys(ggml_backend_t backend, const ScoreDims & D, const ggml_fp16_t * q, ggml_type k_type,
                const void * k, int chunk_rows, std::vector<float> & scores, std::string * err) {
    auto fail = [&](const char * msg) { if (err) *err = msg; return false; };
    if (!backend || !q || !k) return fail("null argument");
    if (D.head_dim <= 0 || D.n_head <= 0 || D.n_kv_head <= 0 || D.n_keys <= 0 || D.n_queries <= 0 ||
        D.n_head % D.n_kv_head != 0) return fail("bad dims (n_head must be a multiple of n_kv_head)");
    if (k_type != GGML_TYPE_F16 && k_type != GGML_TYPE_F32) return fail("k_type must be F16 or F32");
    if (chunk_rows < 1) return fail("chunk_rows must be >= 1");

    ggml_init_params ip = { ggml_tensor_overhead() * 8, nullptr, true };
    ggml_context * pctx = ggml_init(ip);
    if (!pctx) return fail("ggml_init failed");
    ggml_tensor * q16 = ggml_new_tensor_3d(pctx, GGML_TYPE_F16, D.head_dim, D.n_queries, D.n_head);
    ggml_tensor * kt  = ggml_new_tensor_3d(pctx, k_type, D.head_dim, D.n_keys, D.n_kv_head);
    ggml_tensor * acc = ggml_new_tensor_3d(pctx, GGML_TYPE_F32, 1, D.n_keys, D.n_kv_head);
    ggml_backend_buffer_t pbuf = ggml_backend_alloc_ctx_tensors(pctx, backend);
    if (!pbuf) { ggml_free(pctx); return fail("persistent buffer allocation failed"); }

    scores.assign((size_t) D.n_keys * D.n_kv_head, 0.0f);
    ggml_backend_tensor_set(q16, q, 0, ggml_nbytes(q16));
    ggml_backend_tensor_set(kt, k, 0, ggml_nbytes(kt));
    ggml_backend_tensor_set(acc, scores.data(), 0, ggml_nbytes(acc));

    ggml_gallocr_t galloc = ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend));
    const int nq_chunk = std::max(1, chunk_rows / D.n_head);
    bool ok = true;
    for (int q0 = 0; ok && q0 < D.n_queries; q0 += nq_chunk) {
        const int nq = std::min(nq_chunk, D.n_queries - q0);
        ggml_init_params gp = { ggml_tensor_overhead() * 32 + ggml_graph_overhead(), nullptr, true };
        ggml_context * gctx = ggml_init(gp);
        ggml_cgraph * gf = ggml_new_graph(gctx);
        build_chunk(gctx, gf, D, q16, kt, acc, q0, nq);
        if (!ggml_gallocr_alloc_graph(galloc, gf)) { ok = false; if (err) *err = "graph allocation failed"; }
        else if (ggml_backend_graph_compute(backend, gf) != GGML_STATUS_SUCCESS) { ok = false; if (err) *err = "graph compute failed"; }
        ggml_free(gctx);
    }
    if (ok) ggml_backend_tensor_get(acc, scores.data(), 0, ggml_nbytes(acc));
    ggml_gallocr_free(galloc);
    ggml_backend_buffer_free(pbuf);
    ggml_free(pctx);
    return ok;
}

void select_top_k(const float * scores, int n, int k, int32_t * out) {
    std::vector<int32_t> idx(n);
    std::iota(idx.begin(), idx.end(), 0);
    auto better = [scores](int32_t a, int32_t b) {
        return scores[a] > scores[b] || (scores[a] == scores[b] && a < b);
    };
    std::partial_sort(idx.begin(), idx.begin() + k, idx.end(), better);
    std::copy(idx.begin(), idx.begin() + k, out);
}

bool select_keys(ggml_backend_t backend, const ScoreDims & D, const ggml_fp16_t * q, ggml_type k_type,
                 const void * k, int top_k, int chunk_rows, std::vector<int32_t> & out, std::string * err) {
    if (top_k < 1 || top_k > D.n_keys) { if (err) *err = "top_k must be in [1, n_keys]"; return false; }
    std::vector<float> scores;
    if (!score_keys(backend, D, q, k_type, k, chunk_rows, scores, err)) return false;
    for (float s : scores)
        if (!std::isfinite(s)) { if (err) *err = "non-finite key score"; return false; }
    out.resize((size_t) D.n_kv_head * top_k);
    for (int h = 0; h < D.n_kv_head; ++h)
        select_top_k(&scores[(size_t) h * D.n_keys], D.n_keys, top_k, &out[(size_t) h * top_k]);
    return true;
}

}  // namespace luce::am
