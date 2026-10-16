// Per-shape timing of the batch-invariant Q8_0 multi-column MMVQ launch (the MTP verify rows' dense trunk):
// every qwen4exp dense Q8_0 shape, or one K M, as a T-column MUL_MAT on the GPU backend with batch invariance on,
// `reps` graph computes each (synchronised), host wall us per compute after 5 warm-ups (median / min). The
// rows-per-wave and LDS-staging policy come from LUCE_MMVQ_INVARIANT_ROWS / LUCE_MMVQ_INVARIANT_LDS /
// LUCE_MMVQ_INVARIANT_LDS_BLOCKS; run under rocprofv3 --kernel-trace for the kernel time alone. The weights are
// replicated until the copies exceed 128 MiB and the computes rotate through them, so every launch streams its
// weights from DRAM (the 32 MB MALL would otherwise hold a 27 MiB matrix across repetitions).
// usage: bench_mmvq_invariant T [reps=50] [K M]
#include "ggml-backend.h"
#include "ggml-cuda.h"
#include "ggml.h"

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <vector>

namespace {
struct Shape { int64_t K, M; const char * name; };
const Shape k_shapes[] = {
    { 2560, 10240, "attn_qkv" },  { 2560, 6144, "attn_gate" }, { 6144, 2560, "ssm_out" },
    { 2560, 12288, "attn_q" },    { 2560, 512, "attn_k" },     { 640, 2560, "shexp_down" },
    { 2560, 640, "shexp_up" },    { 320, 10240, "hc_up" },     { 10240, 320, "hc_down" },
    { 2560, 248320, "lm_head" },
};
}  // namespace

int main(int argc, char ** argv) {
    if (argc < 2) { std::fprintf(stderr, "usage: bench_mmvq_invariant T [reps] [K M]\n"); return 2; }
    const int T = std::atoi(argv[1]);
    const int reps = argc > 2 ? std::atoi(argv[2]) : 50;
    std::vector<Shape> shapes(std::begin(k_shapes), std::end(k_shapes));
    if (argc > 4) shapes = { { std::atoll(argv[3]), std::atoll(argv[4]), "custom" } };
    ggml_backend_t gpu = ggml_backend_cuda_init(0);
    if (!gpu) { std::printf("no GPU\n"); return 77; }
    ggml_backend_cuda_set_mmvq_batch_invariant(true);
    uint32_t r = 7u;
    auto rnd = [&r]() { r = r * 1664525u + 1013904223u; return (float) (r >> 8) / (float) (1u << 24) * 2.0f - 1.0f; };
    std::printf("T=%d reps=%d LUCE_MMVQ_INVARIANT_ROWS=%s LUCE_MMVQ_INVARIANT_LDS=%s LUCE_MMVQ_INVARIANT_LDS_BLOCKS=%s\n", T, reps,
                std::getenv("LUCE_MMVQ_INVARIANT_ROWS") ? std::getenv("LUCE_MMVQ_INVARIANT_ROWS") : "-",
                std::getenv("LUCE_MMVQ_INVARIANT_LDS") ? std::getenv("LUCE_MMVQ_INVARIANT_LDS") : "-",
                std::getenv("LUCE_MMVQ_INVARIANT_LDS_BLOCKS") ? std::getenv("LUCE_MMVQ_INVARIANT_LDS_BLOCKS") : "-");
    for (const Shape & s : shapes) {
        std::vector<float> wf((size_t) s.K * s.M);
        for (float & v : wf) v = rnd();
        std::vector<uint8_t> wq(ggml_row_size(GGML_TYPE_Q8_0, s.K) * (size_t) s.M);
        ggml_quantize_chunk(GGML_TYPE_Q8_0, wf.data(), wq.data(), 0, s.M, s.K, nullptr);
        std::vector<float> x((size_t) s.K * T);
        for (float & v : x) v = rnd();
        const int ncopies = (int) std::min<size_t>(16, (128u * 1024 * 1024 + wq.size() - 1) / wq.size());
        ggml_init_params ip{ 64 * 1024 * 1024, nullptr, true };
        ggml_context * c = ggml_init(ip);
        ggml_tensor * src1 = ggml_new_tensor_2d(c, GGML_TYPE_F32, s.K, T);
        ggml_set_input(src1);
        std::vector<ggml_tensor *> ws(ncopies);
        std::vector<ggml_cgraph *> gfs(ncopies);
        for (int b = 0; b < ncopies; ++b) {
            ws[b] = ggml_new_tensor_2d(c, GGML_TYPE_Q8_0, s.K, s.M);
            ggml_set_input(ws[b]);
            gfs[b] = ggml_new_graph(c);
            ggml_tensor * o = ggml_mul_mat(c, ws[b], src1);
            ggml_set_output(o); ggml_build_forward_expand(gfs[b], o);
        }
        ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors(c, gpu);
        for (int b = 0; b < ncopies; ++b) ggml_backend_tensor_set(ws[b], wq.data(), 0, wq.size());
        ggml_backend_tensor_set(src1, x.data(), 0, x.size() * sizeof(float));
        std::vector<double> us;
        for (int i = 0; i < reps + 5; ++i) {
            const auto t0 = std::chrono::steady_clock::now();
            ggml_backend_graph_compute(gpu, gfs[i % ncopies]);
            ggml_backend_synchronize(gpu);
            const auto t1 = std::chrono::steady_clock::now();
            if (i >= 5) us.push_back(std::chrono::duration<double, std::micro>(t1 - t0).count());
        }
        std::sort(us.begin(), us.end());
        const double mib = (double) wq.size() / (1024.0 * 1024.0);
        std::printf("[mmvq-invariant] %-10s K=%-5lld M=%-6lld %6.1f MiB T=%d median %8.1f us min %8.1f us (%5.0f GB/s at median)\n",
                    s.name, (long long) s.K, (long long) s.M, mib, T, us[us.size() / 2], us[0],
                    (double) wq.size() / us[us.size() / 2] / 1e3);
        ggml_backend_buffer_free(buf); ggml_free(c);
    }
    ggml_backend_free(gpu);
    return 0;
}
