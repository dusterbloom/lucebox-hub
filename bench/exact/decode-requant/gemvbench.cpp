// T=1 dense GEMV micro-benchmark through the ggml backend API: ncopies distinct [K x rows] weights of one type
// (> the 32 MiB MALL together, so every GEMV streams from DRAM), one graph of ncopies mul_mats per rep. Time the
// kernels with rocprofv3 --kernel-trace (ktsum.py); the wall here includes the eager launch gaps. Also checks each
// copy's y against the F64 product of its dequantized rows and reports the max error (the kernel-variant gate).
// usage: gemvbench <type> <K> <rows> [ncopies=8] [reps=20]
#include "ggml-backend.h"
#include "ggml-cpu.h"
#include "ggml-cuda.h"
#include "ggml.h"

#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

static std::vector<float> random_rows(int64_t ne0, int64_t ne1, uint32_t seed) {
    std::vector<float> v((size_t) (ne0 * ne1));
    uint32_t r = seed;
    for (float & x : v) { r = r * 1664525u + 1013904223u; x = 0.05f * (((r >> 8) & 0xFFFF) / 32768.0f - 1.0f); }
    for (int64_t i = 0; i < ne1; ++i) v[(size_t) (i * ne0 + (i * 37) % ne0)] *= 8.0f;
    return v;
}

int main(int argc, char ** argv) {
    if (argc < 4) { std::fprintf(stderr, "usage: gemvbench <type> <K> <rows> [ncopies] [reps]\n"); return 2; }
    ggml_type t = GGML_TYPE_COUNT;
    for (int i = 0; i < GGML_TYPE_COUNT; ++i) {
        const char * n = ggml_type_name((ggml_type) i);
        if (n && strcasecmp(n, argv[1]) == 0) t = (ggml_type) i;
    }
    if (t == GGML_TYPE_COUNT) { std::fprintf(stderr, "unknown type %s\n", argv[1]); return 2; }
    const int64_t K = atoll(argv[2]), rows = atoll(argv[3]);
    const int ncopies = argc > 4 ? atoi(argv[4]) : 8, reps = argc > 5 ? atoi(argv[5]) : 20;

    ggml_backend_t gpu = ggml_backend_cuda_init(0);
    if (!gpu) { std::fprintf(stderr, "no GPU\n"); return 1; }

    const size_t row_bytes = ggml_row_size(t, K);
    std::vector<std::vector<uint8_t>> w(ncopies);
    std::vector<std::vector<float>> wf(ncopies);
    for (int i = 0; i < ncopies; ++i) {
        const std::vector<float> f = random_rows(K, rows, 3u + i);
        w[i].resize(row_bytes * rows);
        ggml_quantize_chunk(t, f.data(), w[i].data(), 0, rows, K, nullptr);
        wf[i].resize((size_t) (K * rows));
        ggml_get_type_traits(t)->to_float(w[i].data(), wf[i].data(), K * rows);
    }
    const std::vector<float> x = random_rows(K, 1, 23u);

    ggml_init_params ip{ ggml_tensor_overhead() * (size_t) (3 * ncopies + 4) + ggml_graph_overhead(), nullptr, true };
    ggml_context * c = ggml_init(ip);
    ggml_tensor * X = ggml_new_tensor_2d(c, GGML_TYPE_F32, K, 1);
    std::vector<ggml_tensor *> W(ncopies), Y(ncopies);
    ggml_cgraph * gf = ggml_new_graph(c);
    for (int i = 0; i < ncopies; ++i) {
        W[i] = ggml_new_tensor_2d(c, t, K, rows);
        Y[i] = ggml_mul_mat(c, W[i], X);
        ggml_build_forward_expand(gf, Y[i]);
    }
    ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors(c, gpu);
    if (!buf) { std::fprintf(stderr, "alloc failed\n"); return 1; }
    for (int i = 0; i < ncopies; ++i) ggml_backend_tensor_set(W[i], w[i].data(), 0, w[i].size());
    ggml_backend_tensor_set(X, x.data(), 0, ggml_nbytes(X));

    if (ggml_backend_graph_compute(gpu, gf) != GGML_STATUS_SUCCESS) { std::fprintf(stderr, "compute failed\n"); return 1; }
    ggml_backend_synchronize(gpu);
    const auto t0 = std::chrono::steady_clock::now();
    for (int r = 0; r < reps; ++r) ggml_backend_graph_compute(gpu, gf);
    ggml_backend_synchronize(gpu);
    const double us = std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now() - t0).count() / (reps * ncopies);

    double max_abs = 0.0, max_err = 0.0;
    std::vector<float> y((size_t) rows);
    for (int i = 0; i < ncopies; ++i) {
        ggml_backend_tensor_get(Y[i], y.data(), 0, ggml_nbytes(Y[i]));
        for (int64_t r = 0; r < rows; ++r) {
            double ref = 0.0;
            const float * wr = wf[i].data() + (size_t) r * K;
            for (int64_t k = 0; k < K; ++k) ref += (double) wr[k] * (double) x[(size_t) k];
            max_abs = std::fmax(max_abs, std::fabs(ref));
            max_err = std::fmax(max_err, std::fabs(ref - (double) y[(size_t) r]));
        }
    }
    const double mb = row_bytes * (double) rows / 1e6;
    std::printf("gemvbench %s K=%lld rows=%lld bytes=%.2f MB wall %.2f us/gemv (%.1f GB/s incl. launch gaps) "
                "max err %.3g vs max |y| %.3g (%.2e rel)\n",
                ggml_type_name(t), (long long) K, (long long) rows, mb, us, mb / us * 1e3, max_err, max_abs, max_err / max_abs);
    ggml_backend_buffer_free(buf);
    ggml_free(c);
    ggml_backend_free(gpu);
    return 0;
}
