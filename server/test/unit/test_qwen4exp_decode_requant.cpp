// Decode-only requantized copies (qwen4exp_loader.cpp, LUCE_DECODE_REQ): the env parser, byte parity of
// qwen4exp_requant_q8_0 with ggml's reference quantizer on the dequantized Q8_0 rows (Q5_1 / Q5_0 / Q5_K / Q6_K /
// Q4_K, block-size rejects), and the T=1 GEMV on a copy against the F64 product of the copy's own dequantized rows on
// the GPU backend (the mmvq vec_dot the decode graph runs) and on the CPU backend. GPU part SKIPs (77) without one.
#include "qwen4exp/qwen4exp_internal.h"

#include "ggml-backend.h"
#include "ggml-cpu.h"
#include "ggml-cuda.h"
#include "ggml.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <vector>

using luce::common::qwen4exp_decode_req_type;
using luce::common::qwen4exp_requant_q8_0;

static int failures = 0;
#define CHECK(cond) do { if (!(cond)) { std::fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, #cond); ++failures; } } while (0)

static std::vector<float> random_rows(int64_t ne0, int64_t ne1, uint32_t seed) {
    std::vector<float> v((size_t) (ne0 * ne1));
    uint32_t r = seed;
    for (float & x : v) { r = r * 1664525u + 1013904223u; x = 0.05f * (((r >> 8) & 0xFFFF) / 32768.0f - 1.0f); }
    for (int64_t i = 0; i < ne1; ++i) v[(size_t) (i * ne0 + (i * 37) % ne0)] *= 8.0f;   // one outlier per row
    return v;
}

static std::vector<uint8_t> q8_rows(const std::vector<float> & f32, int64_t ne0, int64_t ne1) {
    std::vector<uint8_t> q8(ggml_row_size(GGML_TYPE_Q8_0, ne0) * ne1);
    ggml_quantize_chunk(GGML_TYPE_Q8_0, f32.data(), q8.data(), 0, ne1, ne0, nullptr);
    return q8;
}

static void test_type_parser() {
    CHECK(qwen4exp_decode_req_type(nullptr) == GGML_TYPE_COUNT);
    CHECK(qwen4exp_decode_req_type("") == GGML_TYPE_COUNT);
    CHECK(qwen4exp_decode_req_type("0") == GGML_TYPE_COUNT);
    CHECK(qwen4exp_decode_req_type("bogus") == GGML_TYPE_COUNT);
    CHECK(qwen4exp_decode_req_type("q8_0") == GGML_TYPE_Q8_0);   // same-size control copy (placement A/B)
    CHECK(qwen4exp_decode_req_type("q5_1") == GGML_TYPE_Q5_1);
    CHECK(qwen4exp_decode_req_type("Q5_K") == GGML_TYPE_Q5_K);
    CHECK(qwen4exp_decode_req_type("q6_K") == GGML_TYPE_Q6_K);
    CHECK(qwen4exp_decode_req_type("q4_k") == GGML_TYPE_Q4_K);
}

static void test_requant_parity() {
    const int64_t ne0 = 2560, ne1 = 8;
    const std::vector<float> f32 = random_rows(ne0, ne1, 11u);
    const std::vector<uint8_t> q8 = q8_rows(f32, ne0, ne1);
    std::vector<float> deq((size_t) (ne0 * ne1));
    ggml_get_type_traits(GGML_TYPE_Q8_0)->to_float(q8.data(), deq.data(), ne0 * ne1);
    for (ggml_type t : { GGML_TYPE_Q5_1, GGML_TYPE_Q5_0, GGML_TYPE_Q5_K, GGML_TYPE_Q6_K, GGML_TYPE_Q4_K }) {
        std::vector<uint8_t> got(ggml_row_size(t, ne0) * ne1), ref(got.size());
        CHECK(qwen4exp_requant_q8_0(t, q8.data(), ne0, ne1, got.data()));
        ggml_quantize_chunk(t, deq.data(), ref.data(), 0, ne1, ne0, nullptr);
        CHECK(std::memcmp(got.data(), ref.data(), got.size()) == 0);
    }
    // Rows that are not a multiple of the K-quant super-block are rejected, 32-wide blocks still fit.
    const std::vector<uint8_t> q8_320 = q8_rows(random_rows(320, 2, 5u), 320, 2);
    std::vector<uint8_t> out(ggml_row_size(GGML_TYPE_Q5_1, 320) * 2);
    CHECK(!qwen4exp_requant_q8_0(GGML_TYPE_Q5_K, q8_320.data(), 320, 2, out.data()));
    CHECK(qwen4exp_requant_q8_0(GGML_TYPE_Q5_1, q8_320.data(), 320, 2, out.data()));
}

// y = W x on `backend` for a [ne0, ne1] weight of type t built from q8; compared with the F64 product of the
// copy's dequantized rows. The activation is q8_1-quantized by every backend's quantized GEMV, so the bound is
// the 8-bit activation error, not bit-exactness.
static bool gemv_matches(ggml_backend_t backend, const char * label, ggml_type t, const std::vector<uint8_t> & q8,
                         int64_t ne0, int64_t ne1) {
    std::vector<uint8_t> copy(ggml_row_size(t, ne0) * ne1);
    if (!qwen4exp_requant_q8_0(t, q8.data(), ne0, ne1, copy.data())) return false;
    std::vector<float> w((size_t) (ne0 * ne1));
    ggml_get_type_traits(t)->to_float(copy.data(), w.data(), ne0 * ne1);
    const std::vector<float> x = random_rows(ne0, 1, 23u);

    ggml_init_params ip{ ggml_tensor_overhead() * 8 + ggml_graph_overhead(), nullptr, true };
    ggml_context * c = ggml_init(ip);
    ggml_tensor * W = ggml_new_tensor_2d(c, t, ne0, ne1);
    ggml_tensor * X = ggml_new_tensor_2d(c, GGML_TYPE_F32, ne0, 1);
    ggml_tensor * Y = ggml_mul_mat(c, W, X);
    ggml_cgraph * gf = ggml_new_graph(c);
    ggml_build_forward_expand(gf, Y);
    ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors(c, backend);
    if (!buf) { ggml_free(c); return false; }
    ggml_backend_tensor_set(W, copy.data(), 0, copy.size());
    ggml_backend_tensor_set(X, x.data(), 0, ggml_nbytes(X));
    const bool computed = ggml_backend_graph_compute(backend, gf) == GGML_STATUS_SUCCESS;
    std::vector<float> y((size_t) ne1);
    if (computed) ggml_backend_tensor_get(Y, y.data(), 0, ggml_nbytes(Y));
    ggml_backend_buffer_free(buf);
    ggml_free(c);
    if (!computed) return false;

    double max_abs = 0.0, max_err = 0.0;
    for (int64_t r = 0; r < ne1; ++r) {
        double ref = 0.0;
        for (int64_t k = 0; k < ne0; ++k) ref += (double) w[(size_t) (r * ne0 + k)] * (double) x[(size_t) k];
        max_abs = std::max(max_abs, std::fabs(ref));
        max_err = std::max(max_err, std::fabs(ref - (double) y[(size_t) r]));
    }
    // Q5_1 / Q4_1 on FAST_FP16 GPUs form the min term m * sum(x) in half (vecdotq.cuh vec_dot_q5_1_q8_1_impl),
    // and y is the cancellation of that term against the unsigned-quant dot: measured 3.3% of max |y| at
    // [2560 x 64] on gfx1151 against 0.5% for Q5_K / Q6_K and the CPU path. The gate below is for the wiring; the
    // e2e quality of each type is the teacher-forced KL gate (bench/exact/decode-requant).
    const bool ok = max_err <= 5e-2 * max_abs;
    std::printf("[decode-requant] %s %s gemv [%lld x %lld]: max err %.3g vs max |y| %.3g %s\n", label,
                ggml_type_name(t), (long long) ne0, (long long) ne1, max_err, max_abs, ok ? "ok" : "FAIL");
    return ok;
}

int main() {
    test_type_parser();
    test_requant_parity();
    const int64_t ne0 = 2560, ne1 = 64;
    const std::vector<uint8_t> q8 = q8_rows(random_rows(ne0, ne1, 3u), ne0, ne1);
    ggml_backend_t cpu = ggml_backend_cpu_init();
    for (ggml_type t : { GGML_TYPE_Q5_1, GGML_TYPE_Q5_K, GGML_TYPE_Q6_K }) CHECK(gemv_matches(cpu, "cpu", t, q8, ne0, ne1));
    ggml_backend_free(cpu);
    if (failures) return 1;
    ggml_backend_t gpu = ggml_backend_cuda_init(0);
    if (!gpu) { std::printf("[decode-requant] cpu checks pass; SKIP gpu gemv: no GPU\n"); return 77; }
    for (ggml_type t : { GGML_TYPE_Q5_1, GGML_TYPE_Q5_K, GGML_TYPE_Q6_K }) CHECK(gemv_matches(gpu, "gpu", t, q8, ne0, ne1));
    ggml_backend_free(gpu);
    std::printf("[decode-requant] %s\n", failures ? "FAIL" : "all checks pass");
    return failures ? 1 : 0;
}
