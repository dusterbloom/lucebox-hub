#include "common/am_fit.h"

#include "ggml-alloc.h"

#include <algorithm>
#include <cmath>
#include <memory>
#include <numeric>

namespace luce::am {

namespace {

struct CtxDel   { void operator()(ggml_context * c) const { ggml_free(c); } };
struct BufDel   { void operator()(ggml_backend_buffer_t b) const { ggml_backend_buffer_free(b); } };
struct GallocDel { void operator()(ggml_gallocr_t g) const { ggml_gallocr_free(g); } };
using CtxPtr    = std::unique_ptr<ggml_context, CtxDel>;
using BufPtr    = std::unique_ptr<ggml_backend_buffer, BufDel>;
using GallocPtr = std::unique_ptr<ggml_gallocr, GallocDel>;

ggml_backend_buffer_t buffer_of(const ggml_tensor * t) {
    return (t->view_src ? t->view_src : t)->buffer;
}

// An operand as a sum of F16 tensors: {x} for F16, {hi, lo} for F32 with x = hi + lo (22
// mantissa bits). CUDA runs F32 matmuls on TF32 tensor cores whatever the op precision says
// (cuBLAS math mode TF32, MMF kernels), which costs ~1e-3 relative in the logits; F16
// operands honour GGML_PREC_F32 (fp32 accumulation, exact F16 products), so F32 inputs are
// split and multiplied as F16 pieces. F16 inputs (the engine's feed) take the single term.
std::vector<ggml_tensor *> f16_parts(ggml_context * ctx, ggml_tensor * x) {
    if (x->type == GGML_TYPE_F16) return { x };
    ggml_tensor * hi = ggml_cast(ctx, x, GGML_TYPE_F16);
    ggml_tensor * lo = ggml_cast(ctx, ggml_sub(ctx, x, ggml_cast(ctx, hi, GGML_TYPE_F32)), GGML_TYPE_F16);
    return { hi, lo };
}

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
    ggml_tensor * s = nullptr;                                                     // [T, nq, n_head]
    const std::vector<ggml_tensor *> kp = f16_parts(ctx, kt);
    const std::vector<ggml_tensor *> qp = f16_parts(ctx, qc);
    for (size_t i = 0; i < kp.size(); ++i)
        for (size_t j = 0; j < qp.size(); ++j) {
            if (i == 1 && j == 1) continue;  // lo x lo is below 2^-22 relative
            ggml_tensor * t = ggml_mul_mat(ctx, kp[i], ggml_cast(ctx, qp[j], GGML_TYPE_F32));
            ggml_mul_mat_set_prec(t, GGML_PREC_F32);
            s = s ? ggml_add(ctx, s, t) : t;
        }
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

bool score_keys(ggml_backend_t backend, ggml_tensor * q, ggml_tensor * k, int chunk_rows,
                std::vector<float> & scores, std::string * err) {
    auto fail = [&](const char * msg) { if (err) *err = msg; return false; };
    if (!backend || !q || !k) return fail("null argument");
    if ((q->type != GGML_TYPE_F16 && q->type != GGML_TYPE_F32) || q->nb[0] != ggml_type_size(q->type) || q->ne[3] != 1)
        return fail("q must be a 3D F16 or F32 tensor");
    if (k->type != GGML_TYPE_F16 && k->type != GGML_TYPE_F32) return fail("k type must be F16 or F32");
    if (k->nb[0] != ggml_type_size(k->type) || k->ne[3] != 1) return fail("k must be a 3D tensor with contiguous rows");
    if (!buffer_of(q) || !buffer_of(k)) return fail("q and k must live in a backend buffer");
    ScoreDims D = { (int) q->ne[0], (int) q->ne[2], (int) k->ne[2], (int) k->ne[1], (int) q->ne[1] };
    if (D.head_dim <= 0 || D.n_head <= 0 || D.n_kv_head <= 0 || D.n_keys <= 0 || D.n_queries <= 0 ||
        k->ne[0] != q->ne[0] || D.n_head % D.n_kv_head != 0) return fail("bad dims (head_dim must match, n_head must be a multiple of n_kv_head)");
    if (chunk_rows < D.n_head) return fail("chunk_rows must be >= n_head (one query per head is the smallest chunk)");

    scores.assign((size_t) D.n_keys * D.n_kv_head, 0.0f);
    ggml_init_params ip = { ggml_tensor_overhead() * 4, nullptr, true };
    CtxPtr pctx(ggml_init(ip));
    if (!pctx) return fail("ggml_init failed");
    ggml_tensor * acc = ggml_new_tensor_3d(pctx.get(), GGML_TYPE_F32, 1, D.n_keys, D.n_kv_head);
    BufPtr pbuf(ggml_backend_alloc_ctx_tensors(pctx.get(), backend));
    if (!pbuf) return fail("accumulator allocation failed");
    ggml_backend_tensor_set(acc, scores.data(), 0, ggml_nbytes(acc));

    GallocPtr galloc(ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend)));
    const int nq_chunk = chunk_rows / D.n_head;  // >= 1
    for (int q0 = 0; q0 < D.n_queries; q0 += nq_chunk) {
        const int nq = std::min(nq_chunk, D.n_queries - q0);
        ggml_init_params gp = { ggml_tensor_overhead() * 128 + ggml_graph_overhead(), nullptr, true };
        CtxPtr gctx(ggml_init(gp));
        if (!gctx) return fail("ggml_init failed");
        ggml_cgraph * gf = ggml_new_graph(gctx.get());
        build_chunk(gctx.get(), gf, D, q, k, acc, q0, nq);
        if (!ggml_gallocr_alloc_graph(galloc.get(), gf)) return fail("graph allocation failed");
        if (ggml_backend_graph_compute(backend, gf) != GGML_STATUS_SUCCESS) return fail("graph compute failed");
    }
    ggml_backend_tensor_get(acc, scores.data(), 0, ggml_nbytes(acc));
    return true;
}

bool score_keys(ggml_backend_t backend, const ScoreDims & D, const ggml_fp16_t * q, ggml_type k_type,
                const void * k, int chunk_rows, std::vector<float> & scores, std::string * err) {
    if (!backend || !q || !k || D.head_dim <= 0 || D.n_head <= 0 || D.n_kv_head <= 0 || D.n_keys <= 0 || D.n_queries <= 0) {
        if (err) *err = "null argument or bad dims";
        return false;
    }
    if (k_type != GGML_TYPE_F16 && k_type != GGML_TYPE_F32) { if (err) *err = "k_type must be F16 or F32"; return false; }
    ggml_init_params ip = { ggml_tensor_overhead() * 4, nullptr, true };
    CtxPtr ctx(ggml_init(ip));
    if (!ctx) { if (err) *err = "ggml_init failed"; return false; }
    ggml_tensor * q16 = ggml_new_tensor_3d(ctx.get(), GGML_TYPE_F16, D.head_dim, D.n_queries, D.n_head);
    ggml_tensor * kt  = ggml_new_tensor_3d(ctx.get(), k_type, D.head_dim, D.n_keys, D.n_kv_head);
    BufPtr buf(ggml_backend_alloc_ctx_tensors(ctx.get(), backend));
    if (!buf) { if (err) *err = "input buffer allocation failed"; return false; }
    ggml_backend_tensor_set(q16, q, 0, ggml_nbytes(q16));
    ggml_backend_tensor_set(kt, k, 0, ggml_nbytes(kt));
    return score_keys(backend, q16, kt, chunk_rows, scores, err);
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

namespace {
bool finish_selection(const std::vector<float> & scores, const ScoreDims & D, int top_k,
                      std::vector<int32_t> & out, std::string * err) {
    for (float s : scores)
        if (!std::isfinite(s)) { if (err) *err = "non-finite key score"; return false; }
    out.resize((size_t) D.n_kv_head * top_k);
    for (int h = 0; h < D.n_kv_head; ++h)
        select_top_k(&scores[(size_t) h * D.n_keys], D.n_keys, top_k, &out[(size_t) h * top_k]);
    return true;
}
}  // namespace

bool select_keys(ggml_backend_t backend, ggml_tensor * q, ggml_tensor * k, int top_k, int chunk_rows,
                 std::vector<int32_t> & out, std::string * err) {
    std::vector<float> scores;
    if (!score_keys(backend, q, k, chunk_rows, scores, err)) return false;
    const ScoreDims D = { (int) q->ne[0], (int) q->ne[2], (int) k->ne[2], (int) k->ne[1], (int) q->ne[1] };
    if (top_k < 1 || top_k > D.n_keys) { if (err) *err = "top_k must be in [1, n_keys]"; return false; }
    return finish_selection(scores, D, top_k, out, err);
}

bool select_keys(ggml_backend_t backend, const ScoreDims & D, const ggml_fp16_t * q, ggml_type k_type,
                 const void * k, int top_k, int chunk_rows, std::vector<int32_t> & out, std::string * err) {
    if (top_k < 1 || top_k > D.n_keys) { if (err) *err = "top_k must be in [1, n_keys]"; return false; }
    std::vector<float> scores;
    if (!score_keys(backend, D, q, k_type, k, chunk_rows, scores, err)) return false;
    return finish_selection(scores, D, top_k, out, err);
}

}  // namespace luce::am
