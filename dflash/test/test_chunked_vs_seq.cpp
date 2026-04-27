// test_chunked_vs_seq.cpp
// Comparison test: runs the same q,k,v,g,beta,state through both
// ggml_gated_delta_net (sequential) and build_delta_net_chunked,
// compares output and final state element-by-element.
//
// Usage: test_chunked_vs_seq [n_tokens]
//   n_tokens defaults to 16 (typical verify length)

#include "internal.h"
#include "delta_net_chunked.h"

#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"
#include "ggml-cuda.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <vector>

using namespace dflash27b;

static void fill_random(std::vector<float> & data, unsigned seed) {
    srand(seed);
    for (auto & x : data) {
        x = 2.0f * (float)rand() / (float)RAND_MAX - 1.0f;
    }
}

static std::vector<float> read_tensor(ggml_tensor * t) {
    int64_t n = ggml_nelements(t);
    std::vector<float> data(n);
    ggml_backend_tensor_get(t, data.data(), 0, sizeof(float) * n);
    return data;
}

struct CmpResult {
    double rmse;
    float  max_abs;
    int64_t max_idx;
    bool   pass;
};

struct RefResult {
    std::vector<float> output;      // [S_v, H_v, n_tokens, n_seqs]
    std::vector<float> final_state; // [S_v, S_v, H_v, n_seqs]
    std::vector<float> state_hist;  // [n_tokens, S_v, S_v, H_v, n_seqs]
};

static inline int64_t idx_qk(int64_t s, int64_t h, int64_t t,
                             int64_t S_v, int64_t H_v) {
    return s + S_v * (h + H_v * t);
}

static inline int64_t idx_gb(int64_t h, int64_t t, int64_t H_v) {
    return h + H_v * t;
}

static inline int64_t idx_state(int64_t row, int64_t col, int64_t h,
                                int64_t S_v) {
    return row + S_v * (col + S_v * h);
}

static RefResult run_reference_delta_net(const std::vector<float> & h_q,
                                         const std::vector<float> & h_k,
                                         const std::vector<float> & h_v,
                                         const std::vector<float> & h_g,
                                         const std::vector<float> & h_b,
                                         const std::vector<float> & h_s,
                                         int64_t S_v,
                                         int64_t H_v,
                                         int64_t n_tokens) {
    RefResult r;
    r.output.resize((size_t)S_v * H_v * n_tokens);
    r.final_state = h_s;
    r.state_hist.resize((size_t)n_tokens * S_v * S_v * H_v);

    const float scale = 1.0f / sqrtf((float)S_v);
    std::vector<float> kv(S_v);
    std::vector<float> delta(S_v);
    std::vector<float> q_scaled(S_v);

    for (int64_t t = 0; t < n_tokens; t++) {
        for (int64_t h = 0; h < H_v; h++) {
            const float decay = expf(h_g[idx_gb(h, t, H_v)]);
            const float beta  = h_b[idx_gb(h, t, H_v)];

            float * s_h = r.final_state.data() + idx_state(0, 0, h, S_v);

            for (int64_t col = 0; col < S_v; col++) {
                for (int64_t row = 0; row < S_v; row++) {
                    s_h[idx_state(row, col, 0, S_v)] *= decay;
                }
            }

            for (int64_t col = 0; col < S_v; col++) {
                float acc = 0.0f;
                for (int64_t row = 0; row < S_v; row++) {
                    acc += s_h[idx_state(row, col, 0, S_v)] *
                           h_k[idx_qk(row, h, t, S_v, H_v)];
                }
                kv[col] = acc;
            }

            for (int64_t i = 0; i < S_v; i++) {
                delta[i] = (h_v[idx_qk(i, h, t, S_v, H_v)] - kv[i]) * beta;
            }

            for (int64_t row = 0; row < S_v; row++) {
                const float k_row = h_k[idx_qk(row, h, t, S_v, H_v)];
                for (int64_t col = 0; col < S_v; col++) {
                    s_h[idx_state(row, col, 0, S_v)] += k_row * delta[col];
                }
            }

            for (int64_t i = 0; i < S_v; i++) {
                q_scaled[i] = h_q[idx_qk(i, h, t, S_v, H_v)] * scale;
            }

            for (int64_t out = 0; out < S_v; out++) {
                float acc = 0.0f;
                for (int64_t row = 0; row < S_v; row++) {
                    acc += s_h[idx_state(row, out, 0, S_v)] * q_scaled[row];
                }
                r.output[(size_t)t * H_v * S_v + (size_t)h * S_v + out] = acc;
            }

            float * hist = r.state_hist.data() + (size_t)t * H_v * S_v * S_v + (size_t)h * S_v * S_v;
            for (int64_t col = 0; col < S_v; col++) {
                for (int64_t row = 0; row < S_v; row++) {
                    hist[(size_t)col * S_v + row] = s_h[idx_state(row, col, 0, S_v)];
                }
            }
        }
    }

    return r;
}

static CmpResult compare(const std::vector<float> & a,
                         const std::vector<float> & b,
                         float tol) {
    CmpResult r = {0, 0, 0, false};
    if (a.size() != b.size()) return r;
    double ss = 0;
    for (size_t i = 0; i < a.size(); i++) {
        float d = a[i] - b[i];
        ss += (double)d * d;
        if (fabsf(d) > r.max_abs) { r.max_abs = fabsf(d); r.max_idx = (int64_t)i; }
    }
    r.rmse = sqrt(ss / (double)a.size());
    r.pass = r.max_abs < tol;
    return r;
}

// Helper: set input + run graph, return output views
struct RunResult {
    std::vector<float> output;
    std::vector<float> state;
};

int main(int argc, char ** argv) {
    int n_tokens = (argc > 1) ? atoi(argv[1]) : 16;
    if (n_tokens < 2) { fprintf(stderr, "n_tokens must be >= 2\n"); return 1; }
    printf("# test_chunked_vs_seq  n_tokens=%d\n", n_tokens);

    const char * backend_name = std::getenv("DFLASH27B_BACKEND");
    ggml_backend_t backend = nullptr;
    if (backend_name && std::string(backend_name) == "cpu") {
        backend = ggml_backend_cpu_init();
    } else {
        backend = ggml_backend_cuda_init(0);
    }
    if (!backend) {
        fprintf(stderr, "%s backend init failed\n",
                (backend_name && std::string(backend_name) == "cpu") ? "CPU" : "CUDA");
        return 1;
    }

    // MoE delta-net dimensions (after Q/K repeat from H_k=16 to H_v=32)
    const int64_t S_v     = 128;   // head_v_dim
    const int64_t H_v     = 32;    // num_v_heads = dt_rank
    const int64_t n_seqs  = 1;

    const int64_t qk_n = S_v * H_v * n_tokens * n_seqs;
    const int64_t v_n  = S_v * H_v * n_tokens * n_seqs;
    const int64_t gb_n = H_v * n_tokens * n_seqs;
    const int64_t s_n  = S_v * S_v * H_v * n_seqs;

    std::vector<float> h_q(qk_n), h_k(qk_n), h_v(v_n);
    std::vector<float> h_g(gb_n), h_b(gb_n), h_s(s_n);

    fill_random(h_q, 100);
    fill_random(h_k, 200);
    fill_random(h_v, 300);
    fill_random(h_g, 400);
    fill_random(h_b, 500);
    fill_random(h_s, 600);
    for (auto & x : h_g) x = 0.1f * x;  // small non-zero gates
    for (auto & x : h_b) x = 0.5f + 0.5f * x;
    // h_s keeps random initial state (non-zero)

    auto ref = run_reference_delta_net(h_q, h_k, h_v, h_g, h_b, h_s,
                                       S_v, H_v, n_tokens);
    printf("ref: out=%zu  state=%zu  hist=%zu\n",
           ref.output.size(), ref.final_state.size(), ref.state_hist.size());

    // ── Sequential path ──────────────────────────────────────────────
    std::vector<float> seq_out, seq_st;
    {
        struct ggml_init_params p = {}; p.mem_size = 512 << 20; p.no_alloc = true;
        ggml_context * ctx = ggml_init(p);

        auto * q = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, S_v, H_v, n_tokens, n_seqs);
        auto * k = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, S_v, H_v, n_tokens, n_seqs);
        auto * v = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, S_v, H_v, n_tokens, n_seqs);
        auto * g = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, 1,   H_v, n_tokens, n_seqs);
        auto * b = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, 1,   H_v, n_tokens, n_seqs);
        auto * s = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, S_v, S_v, H_v, n_seqs);
        ggml_set_input(q); ggml_set_input(k); ggml_set_input(v);
        ggml_set_input(g); ggml_set_input(b); ggml_set_input(s);

        ggml_cgraph * gf = ggml_new_graph_custom(ctx, 8192, false);
        auto * result = ggml_gated_delta_net(ctx, q, k, v, g, b, s);
        ggml_set_output(result);
        ggml_build_forward_expand(gf, result);

        auto * alloc = ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend));
        ggml_gallocr_alloc_graph(alloc, gf);

        ggml_backend_tensor_set(q, h_q.data(), 0, sizeof(float) * qk_n);
        ggml_backend_tensor_set(k, h_k.data(), 0, sizeof(float) * qk_n);
        ggml_backend_tensor_set(v, h_v.data(), 0, sizeof(float) * v_n);
        ggml_backend_tensor_set(g, h_g.data(), 0, sizeof(float) * gb_n);
        ggml_backend_tensor_set(b, h_b.data(), 0, sizeof(float) * gb_n);
        ggml_backend_tensor_set(s, h_s.data(), 0, sizeof(float) * s_n);

        if (ggml_backend_graph_compute(backend, gf) != GGML_STATUS_SUCCESS) {
            fprintf(stderr, "FAIL: sequential compute\n"); return 1;
        }

        const int64_t seq_out_n = S_v * H_v * n_tokens * n_seqs;
        const int64_t seq_st_n   = S_v * S_v * H_v * n_seqs;
        std::vector<float> seq_all(ggml_nelements(result));
        ggml_backend_tensor_get(result, seq_all.data(), 0, sizeof(float) * seq_all.size());

        seq_out.assign(seq_all.begin(), seq_all.begin() + seq_out_n);
        seq_st.assign(seq_all.begin() + seq_out_n,
                      seq_all.begin() + seq_out_n + seq_st_n);

        printf("seq: out=%zu  state=%zu\n", seq_out.size(), seq_st.size());
        ggml_gallocr_free(alloc);
        ggml_free(ctx);
    }

    // ── Chunked path ─────────────────────────────────────────────────
    std::vector<float> chk_out, chk_st;
    {
        struct ggml_init_params p = {}; p.mem_size = 512 << 20; p.no_alloc = true;
        ggml_context * ctx = ggml_init(p);

        auto * q = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, S_v, H_v, n_tokens, n_seqs);
        auto * k = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, S_v, H_v, n_tokens, n_seqs);
        auto * v = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, S_v, H_v, n_tokens, n_seqs);
        auto * g = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, 1,   H_v, n_tokens, n_seqs);
        auto * b = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, 1,   H_v, n_tokens, n_seqs);
        auto * s = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, S_v, S_v, H_v, n_seqs);
        ggml_set_input(q); ggml_set_input(k); ggml_set_input(v);
        ggml_set_input(g); ggml_set_input(b); ggml_set_input(s);

        ggml_cgraph * gf = ggml_new_graph_custom(ctx, 8192, false);
        auto r = build_delta_net_chunked(ctx, gf, q, k, v, g, b, s);
        ggml_set_output(r.output);
        ggml_set_output(r.new_state);
        ggml_build_forward_expand(gf, r.output);
        ggml_build_forward_expand(gf, r.new_state);

        auto * alloc = ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend));
        ggml_gallocr_alloc_graph(alloc, gf);

        // Fill strictly-lower mask if present (non-KDA path)
        if (r.sl_mask) {
            const int64_t mCS   = r.sl_mask->ne[0];
            const int64_t mNC   = r.sl_mask->ne[2];
            const int64_t mHVS  = r.sl_mask->ne[3];
            const int64_t total = ggml_nelements(r.sl_mask);
            std::vector<float> mask_data(total, 1.0f); // DEBUG: all 1s
            // ggml column-major: offset(row, col, chunk, head) = row + CS*col + CS*CS*chunk + CS*CS*NC*head
            for (int64_t h = 0; h < mHVS; h++)
                for (int64_t c = 0; c < mNC; c++)
                    for (int64_t row = 0; row < mCS; row++)
                        for (int64_t col = row; col < mCS; col++) // on + above diagonal → 0
                            mask_data[h * mNC * mCS * mCS + c * mCS * mCS + col * mCS + row] = 0.0f;
            ggml_backend_tensor_set(r.sl_mask, mask_data.data(), 0,
                sizeof(float) * mask_data.size());
            // Verify: read back first few values
            std::vector<float> verify(std::min((int64_t)10, total));
            ggml_backend_tensor_get(r.sl_mask, verify.data(), 0, sizeof(float) * verify.size());
            printf("mask check: ");
            for (int i = 0; i < (int)verify.size(); i++) printf("%.0f ", verify[i]);
            printf("\n");
        }

        ggml_backend_tensor_set(q, h_q.data(), 0, sizeof(float) * qk_n);
        ggml_backend_tensor_set(k, h_k.data(), 0, sizeof(float) * qk_n);
        ggml_backend_tensor_set(v, h_v.data(), 0, sizeof(float) * v_n);
        ggml_backend_tensor_set(g, h_g.data(), 0, sizeof(float) * gb_n);
        ggml_backend_tensor_set(b, h_b.data(), 0, sizeof(float) * gb_n);
        ggml_backend_tensor_set(s, h_s.data(), 0, sizeof(float) * s_n);

        if (ggml_backend_graph_compute(backend, gf) != GGML_STATUS_SUCCESS) {
            fprintf(stderr, "FAIL: chunked compute\n"); return 1;
        }

        chk_out = read_tensor(r.output);
        chk_st  = read_tensor(r.new_state);

        printf("chk: out=%zu  state=%zu\n", chk_out.size(), chk_st.size());
        ggml_gallocr_free(alloc);
        ggml_free(ctx);
    }

    // ── Compare ──────────────────────────────────────────────────────
    printf("\n## Comparison (tol=0.01)\n");
    bool ok = true;

    auto seq_o_ref = compare(seq_out, ref.output, 0.01f);
    auto chk_o_ref = compare(chk_out, ref.output, 0.01f);
    auto seq_o_chk = compare(seq_out, chk_out, 0.01f);
    printf("  output seq-ref: rmse=%.3e max_abs=%.3e idx=%lld %s\n",
           seq_o_ref.rmse, seq_o_ref.max_abs, (long long)seq_o_ref.max_idx,
           seq_o_ref.pass ? "OK" : "FAIL");
    printf("  output chk-ref: rmse=%.3e max_abs=%.3e idx=%lld %s\n",
           chk_o_ref.rmse, chk_o_ref.max_abs, (long long)chk_o_ref.max_idx,
           chk_o_ref.pass ? "OK" : "FAIL");
    printf("  output seq-chk: rmse=%.3e max_abs=%.3e idx=%lld %s\n",
           seq_o_chk.rmse, seq_o_chk.max_abs, (long long)seq_o_chk.max_idx,
           seq_o_chk.pass ? "OK" : "FAIL");
    if (!seq_o_ref.pass || !chk_o_ref.pass || !seq_o_chk.pass) ok = false;

    auto seq_s_ref = compare(seq_st, ref.final_state, 0.01f);
    auto chk_s_ref = compare(chk_st, ref.final_state, 0.01f);
    auto seq_s_chk = compare(seq_st, chk_st, 0.01f);
    printf("  state  seq-ref: rmse=%.3e max_abs=%.3e idx=%lld %s\n",
           seq_s_ref.rmse, seq_s_ref.max_abs, (long long)seq_s_ref.max_idx,
           seq_s_ref.pass ? "OK" : "FAIL");
    printf("  state  chk-ref: rmse=%.3e max_abs=%.3e idx=%lld %s\n",
           chk_s_ref.rmse, chk_s_ref.max_abs, (long long)chk_s_ref.max_idx,
           chk_s_ref.pass ? "OK" : "FAIL");
    printf("  state  seq-chk: rmse=%.3e max_abs=%.3e idx=%lld %s\n",
           seq_s_chk.rmse, seq_s_chk.max_abs, (long long)seq_s_chk.max_idx,
           seq_s_chk.pass ? "OK" : "FAIL");
    if (!seq_s_ref.pass || !chk_s_ref.pass || !seq_s_chk.pass) ok = false;

    printf("\n  Per-token output max_abs vs ref:\n");
    for (int t = 0; t < n_tokens; t++) {
        float mt_seq = 0;
        float mt_chk = 0;
        for (int h = 0; h < (int)H_v; h++)
            for (int s = 0; s < S_v; s++) {
                int64_t idx = (int64_t)t*H_v*S_v + (int64_t)h*S_v + s;
                float d_seq = fabsf(seq_out[idx] - ref.output[idx]);
                float d_chk = fabsf(chk_out[idx] - ref.output[idx]);
                if (d_seq > mt_seq) mt_seq = d_seq;
                if (d_chk > mt_chk) mt_chk = d_chk;
            }
        printf("    t%02d seq=%.3e%s  chk=%.3e%s\n",
               t,
               mt_seq, mt_seq >= 0.01f ? " ***" : "",
               mt_chk, mt_chk >= 0.01f ? " ***" : "");
    }

    printf("\n  Per-head state max_abs vs ref:\n");
    for (int h = 0; h < (int)H_v; h++) {
        float mh_seq = 0;
        float mh_chk = 0;
        for (int i = 0; i < S_v; i++)
            for (int j = 0; j < S_v; j++) {
                int64_t idx = (int64_t)h*S_v*S_v + (int64_t)i*S_v + j;
                float d_seq = fabsf(seq_st[idx] - ref.final_state[idx]);
                float d_chk = fabsf(chk_st[idx] - ref.final_state[idx]);
                if (d_seq > mh_seq) mh_seq = d_seq;
                if (d_chk > mh_chk) mh_chk = d_chk;
            }
        printf("    h%02d seq=%.3e%s  chk=%.3e%s\n",
               h,
               mh_seq, mh_seq >= 0.01f ? " ***" : "",
               mh_chk, mh_chk >= 0.01f ? " ***" : "");
    }

    ggml_backend_free(backend);
    printf("\n%s\n", ok ? "ALL PASS" : "FAILURES DETECTED");
    return ok ? 0 : 1;
}
