// Parity test for the route-union MUL_MAT_ID kernel (mmvq.cu mul_mat_vec_q_moe_union; LUCE_MMID_UNION=0 keeps the
// per-pair kernels). Contract: every row of a T-token batch (the MTP verify window, T = 2..8) is byte-identical to the
// T=1 product of that token, on the qwen4exp expert shapes (gate/up Q4_K [2560, 640, E] fused with SWIGLU, down
// Q5_1 / Q8_0 [640, 2560, E], top-10 of E) and routings with heavy expert reuse across tokens (runs of up to 8 pairs
// per expert), all tokens identical, a duplicated slot inside one token, a -1 sentinel slot, and uniform random ids.
// The union launch counter must grow on every batch (no silent fallback). Batch-invariant MMVQ is switched on as
// the verify forward does. SKIP (77) without a GPU.
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-cuda.h"
#include "ggml.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <string>
#include <vector>

namespace {

constexpr int N_EMBD = 2560, N_FF = 640, N_EXPERT = 64, N_USED = 10;

struct Weights {
    ggml_context * ctx = nullptr;
    ggml_backend_buffer_t buf = nullptr;
    ggml_tensor * gate = nullptr, * up = nullptr, * down = nullptr;
};

void fill_quant(ggml_tensor * t, std::mt19937 & rng) {
    const int64_t rows = t->ne[1] * t->ne[2];
    std::vector<float> f((size_t) t->ne[0] * rows);
    std::uniform_real_distribution<float> d(-1.0f, 1.0f);
    for (float & v : f) v = d(rng);
    std::vector<uint8_t> q(ggml_nbytes(t));
    ggml_quantize_chunk(t->type, f.data(), q.data(), 0, rows, t->ne[0], nullptr);
    ggml_backend_tensor_set(t, q.data(), 0, q.size());
}

bool make_weights(ggml_backend_t gpu, ggml_type down_type, Weights & w) {
    ggml_init_params ip{ 3 * ggml_tensor_overhead(), nullptr, true };
    w.ctx = ggml_init(ip);
    w.gate = ggml_new_tensor_3d(w.ctx, GGML_TYPE_Q4_K, N_EMBD, N_FF, N_EXPERT);
    w.up   = ggml_new_tensor_3d(w.ctx, GGML_TYPE_Q4_K, N_EMBD, N_FF, N_EXPERT);
    w.down = ggml_new_tensor_3d(w.ctx, down_type, N_FF, N_EMBD, N_EXPERT);
    w.buf = ggml_backend_alloc_ctx_tensors(w.ctx, gpu);
    if (!w.buf) return false;
    std::mt19937 rng(20261015u + (unsigned) down_type);
    fill_quant(w.gate, rng); fill_quant(w.up, rng); fill_quant(w.down, rng);
    return true;
}

struct Out { std::vector<float> gu, down; };

// x [N_EMBD, T] f32, ids [N_USED, T] i32 -> gu [N_FF, N_USED, T], down [N_EMBD, N_USED, T] (the qwen4exp MoE graph).
bool run_moe(ggml_backend_t gpu, const Weights & w, const float * x, const int32_t * ids, int T, Out & out) {
    ggml_init_params ip{ 16 * 1024 * 1024, nullptr, true };
    ggml_context * c = ggml_init(ip);
    ggml_tensor * xt = ggml_new_tensor_3d(c, GGML_TYPE_F32, N_EMBD, 1, T);
    ggml_tensor * sel = ggml_new_tensor_2d(c, GGML_TYPE_I32, N_USED, T);
    ggml_set_input(xt); ggml_set_input(sel);
    ggml_tensor * gate = ggml_mul_mat_id(c, w.gate, xt, sel);
    ggml_tensor * up   = ggml_mul_mat_id(c, w.up, xt, sel);
    ggml_tensor * gu   = ggml_swiglu_split(c, gate, up);
    ggml_tensor * down = ggml_mul_mat_id(c, w.down, gu, sel);
    ggml_set_output(gu); ggml_set_output(down);
    ggml_cgraph * gf = ggml_new_graph(c);
    ggml_build_forward_expand(gf, down);
    ggml_gallocr_t alloc = ggml_gallocr_new(ggml_backend_get_default_buffer_type(gpu));
    bool ok = ggml_gallocr_alloc_graph(alloc, gf);
    if (ok) {
        ggml_backend_tensor_set(xt, x, 0, ggml_nbytes(xt));
        ggml_backend_tensor_set(sel, ids, 0, ggml_nbytes(sel));
        ok = ggml_backend_graph_compute(gpu, gf) == GGML_STATUS_SUCCESS;
    }
    if (ok) {
        out.gu.resize(ggml_nelements(gu)); out.down.resize(ggml_nelements(down));
        ggml_backend_tensor_get(gu, out.gu.data(), 0, ggml_nbytes(gu));
        ggml_backend_tensor_get(down, out.down.data(), 0, ggml_nbytes(down));
    }
    ggml_gallocr_free(alloc);
    ggml_free(c);
    return ok;
}

// Routings over T tokens. "window": token t takes experts (base + 2t + j) % 24, so consecutive tokens share 8 of
// 10 experts (runs of up to T pairs); "same": every token the same 10; "dup": window with token 0 slot 3 == slot 0;
// "sentinel": window with token 1 slot 5 = -1; "random": uniform over N_EXPERT.
std::vector<int32_t> routing(const std::string & kind, int T, std::mt19937 & rng) {
    std::vector<int32_t> ids((size_t) N_USED * T);
    std::uniform_int_distribution<int> d(0, N_EXPERT - 1);
    for (int t = 0; t < T; ++t) {
        for (int j = 0; j < N_USED; ++j) {
            int e = (3 + 2 * t + j) % 24;
            if (kind == "same") e = (5 + j) % N_EXPERT;
            if (kind == "random") e = d(rng);
            ids[(size_t) t * N_USED + j] = e;
        }
    }
    if (kind == "dup") ids[3] = ids[0];
    if (kind == "sentinel" && T > 1) ids[(size_t) N_USED + 5] = -1;
    return ids;
}

bool run_case(ggml_backend_t gpu, const Weights & w, const std::string & kind, int T, std::mt19937 & rng) {
    std::vector<float> x((size_t) N_EMBD * T);
    std::uniform_real_distribution<float> d(-1.0f, 1.0f);
    for (float & v : x) v = d(rng);
    const std::vector<int32_t> ids = routing(kind, T, rng);
    const size_t before = ggml_backend_cuda_get_mmvq_mmid_union_launch_count();
    Out batch;
    if (!run_moe(gpu, w, x.data(), ids.data(), T, batch)) return false;
    const size_t launches = ggml_backend_cuda_get_mmvq_mmid_union_launch_count() - before;
    const char * mode = std::getenv("LUCE_MMID_UNION");
    const bool expect_union = !(mode && std::strcmp(mode, "0") == 0);
    bool ok = expect_union ? launches >= 2 : launches == 0;   // gate/up (fused) + down
    size_t bad_gu = 0, bad_down = 0;
    for (int t = 0; t < T; ++t) {
        Out one;
        if (!run_moe(gpu, w, x.data() + (size_t) t * N_EMBD, ids.data() + (size_t) t * N_USED, 1, one)) return false;
        if (std::memcmp(one.gu.data(), batch.gu.data() + (size_t) t * N_FF * N_USED, one.gu.size() * sizeof(float)) != 0) ++bad_gu;
        if (std::memcmp(one.down.data(), batch.down.data() + (size_t) t * N_EMBD * N_USED, one.down.size() * sizeof(float)) != 0) ++bad_down;
    }
    ok = ok && bad_gu == 0 && bad_down == 0;
    std::printf("[mmid-union] down=%s %-8s T=%d union_launches=%zu rows_differ gu=%zu down=%zu %s\n",
                ggml_type_name(w.down->type), kind.c_str(), T, launches, bad_gu, bad_down, ok ? "PASS" : "FAIL");
    return ok;
}

}  // namespace

int main() {
    if (!std::getenv("LUCE_MMID_UNION")) setenv("LUCE_MMID_UNION", "1", 1);
    ggml_backend_t gpu = ggml_backend_cuda_init(0);
    if (!gpu) { std::printf("[mmid-union] SKIP: no GPU\n"); return 77; }
    ggml_backend_cuda_set_mmvq_batch_invariant(true);   // as the verify forward
    std::mt19937 rng(7u);
    bool ok = true;
    for (ggml_type down_type : { GGML_TYPE_Q5_1, GGML_TYPE_Q8_0 }) {
        Weights w;
        if (!make_weights(gpu, down_type, w)) { std::printf("[mmid-union] FAIL: weights\n"); return 1; }
        for (int T : { 2, 3, 4, 5, 8 }) {
            for (const char * kind : { "window", "same", "random" }) ok = run_case(gpu, w, kind, T, rng) && ok;
        }
        ok = run_case(gpu, w, "dup", 4, rng) && ok;
        ok = run_case(gpu, w, "sentinel", 4, rng) && ok;
        ggml_backend_buffer_free(w.buf);
        ggml_free(w.ctx);
    }
    ggml_backend_free(gpu);
    std::printf("[mmid-union] %s\n", ok ? "PASS" : "FAIL");
    return ok ? 0 : 1;
}
