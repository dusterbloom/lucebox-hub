// Correctness test for the bounded (1024-route bitonic) mmid sort vs the
// classic mm_ids_helper it is meant to replace on the prefill-chunk path.
//
// Claim under test: for experts=512, used=10 (the only shape
// ggml_cuda_launch_mm_ids_bounded accepts), both kernels must produce
// byte-identical ids_src1 / ids_dst / expert_bounds for the same routing,
// because both order compact slots by ascending token*used+slot within each
// expert. If this is false, relaxing the bounded-path token gate (mmid.cu)
// would silently reorder expert inputs.
//
// mmid.cuh declares these with cudaStream_t; the HIP build maps that to
// hipStream_t (ggml/src/ggml-cuda/vendors/hip.h:152), so both are declared
// with the HIP spelling here instead of including the header.

#include "ggml-backend.h"
#include "ggml-backend-impl.h"
#include "ggml-cuda.h"
#include "ggml.h"

#include <hip/hip_runtime.h>

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <vector>

void ggml_cuda_launch_mm_ids_helper(
        const int32_t * ids, int32_t * ids_src1, int32_t * ids_dst, int32_t * expert_bounds,
        int n_experts, int n_tokens, int n_expert_used, int nchannels_y, int si1, int sis1,
        bool write_inverse, hipStream_t stream);

// Opaque here: the caller never dereferences it, only forwards the reference
// to the real definition in common.cuh (same pattern as mmid.cuh).
struct ggml_backend_cuda_context;
bool ggml_cuda_launch_mm_ids_bounded(
        ggml_backend_cuda_context & ctx, const int32_t * ids, int32_t * src_map,
        int32_t * dst_map, int32_t * bounds, int experts, int tokens, int used,
        int channels, int ids_stride, int token_stride, bool inverse, hipStream_t stream);

namespace {

bool check(bool cond, const char * what) {
    if (!cond) {
        std::fprintf(stderr, "  FAIL: %s\n", what);
    }
    return cond;
}

// Runs both kernels on identical input routing and asserts the three output
// arrays are byte-for-byte identical.
bool run_case(ggml_backend_cuda_context & ctx, int n_tokens, int n_channels, bool inverse) {
    constexpr int n_experts     = 512; // MM_IDS_ROUTE_EXPERTS, the only size the bounded path accepts
    constexpr int n_expert_used = 10;  // MM_IDS_ROUTE_USED
    const int n_routes = n_tokens * n_expert_used;
    const int ids_stride   = n_expert_used; // contiguous ids
    const int token_stride = n_channels;    // contiguous per-channel rows

    // Deterministic routing with masked (negative) routes sprinkled in, same
    // generator shape as test_ds4_mmid_regression.cpp.
    std::vector<int32_t> h_ids((size_t) n_routes);
    for (int t = 0; t < n_tokens; ++t) {
        for (int u = 0; u < n_expert_used; ++u) {
            int32_t v = (int32_t) ((t * 3 + u * 7 + u * u) % n_experts);
            if (((t + u) % 5) == 0 || ((t * 2 + u) % 11) == 0) {
                v = -1;
            }
            h_ids[(size_t) (t * n_expert_used + u)] = v;
        }
    }

    int32_t * d_ids = nullptr;
    int32_t * d_src1_helper = nullptr, * d_dst_helper = nullptr, * d_bounds_helper = nullptr;
    int32_t * d_src1_bounded = nullptr, * d_dst_bounded = nullptr, * d_bounds_bounded = nullptr;

    bool ok = true;
    ok = check(hipMalloc(&d_ids, sizeof(int32_t) * n_routes) == hipSuccess, "hipMalloc ids") && ok;
    ok = check(hipMalloc(&d_src1_helper, sizeof(int32_t) * n_routes) == hipSuccess, "hipMalloc src1 helper") && ok;
    ok = check(hipMalloc(&d_dst_helper, sizeof(int32_t) * n_routes) == hipSuccess, "hipMalloc dst helper") && ok;
    ok = check(hipMalloc(&d_bounds_helper, sizeof(int32_t) * (n_experts + 1)) == hipSuccess, "hipMalloc bounds helper") && ok;
    ok = check(hipMalloc(&d_src1_bounded, sizeof(int32_t) * n_routes) == hipSuccess, "hipMalloc src1 bounded") && ok;
    ok = check(hipMalloc(&d_dst_bounded, sizeof(int32_t) * n_routes) == hipSuccess, "hipMalloc dst bounded") && ok;
    ok = check(hipMalloc(&d_bounds_bounded, sizeof(int32_t) * (n_experts + 1)) == hipSuccess, "hipMalloc bounds bounded") && ok;
    if (!ok) {
        hipFree(d_ids); hipFree(d_src1_helper); hipFree(d_dst_helper); hipFree(d_bounds_helper);
        hipFree(d_src1_bounded); hipFree(d_dst_bounded); hipFree(d_bounds_bounded);
        return false;
    }

    hipMemcpy(d_ids, h_ids.data(), sizeof(int32_t) * n_routes, hipMemcpyHostToDevice);
    hipMemset(d_src1_helper, 0xFF, sizeof(int32_t) * n_routes);
    hipMemset(d_dst_helper,  0xFF, sizeof(int32_t) * n_routes);
    hipMemset(d_bounds_helper, 0xFF, sizeof(int32_t) * (n_experts + 1));
    hipMemset(d_src1_bounded, 0xFF, sizeof(int32_t) * n_routes);
    hipMemset(d_dst_bounded,  0xFF, sizeof(int32_t) * n_routes);
    hipMemset(d_bounds_bounded, 0xFF, sizeof(int32_t) * (n_experts + 1));

    // si1 == sis1 == nchannels_y here: the test drives ids_src1's forward map
    // as it*sis1 + iex%nchannels_y, matching the bounded path's
    // (route/used)*token_stride + (route%used)%channels when token_stride ==
    // sis1 and channels == nchannels_y.
    ggml_cuda_launch_mm_ids_helper(d_ids, d_src1_helper, d_dst_helper, d_bounds_helper,
        n_experts, n_tokens, n_expert_used, /*nchannels_y=*/n_channels,
        /*si1=*/ids_stride, /*sis1=*/token_stride, inverse, (hipStream_t) 0);

    const bool took_bounded = ggml_cuda_launch_mm_ids_bounded(ctx, d_ids,
        d_src1_bounded, d_dst_bounded, d_bounds_bounded,
        n_experts, n_tokens, n_expert_used, n_channels,
        ids_stride, token_stride, inverse, (hipStream_t) 0);
    hipDeviceSynchronize();

    ok = check(took_bounded, "bounded path accepted the shape (gate)") && ok;
    if (!ok) {
        hipFree(d_ids); hipFree(d_src1_helper); hipFree(d_dst_helper); hipFree(d_bounds_helper);
        hipFree(d_src1_bounded); hipFree(d_dst_bounded); hipFree(d_bounds_bounded);
        return false;
    }

    std::vector<int32_t> h_src1_helper((size_t) n_routes), h_dst_helper((size_t) n_routes), h_bounds_helper((size_t) n_experts + 1);
    std::vector<int32_t> h_src1_bounded((size_t) n_routes), h_dst_bounded((size_t) n_routes), h_bounds_bounded((size_t) n_experts + 1);
    hipMemcpy(h_src1_helper.data(), d_src1_helper, sizeof(int32_t) * n_routes, hipMemcpyDeviceToHost);
    hipMemcpy(h_dst_helper.data(),  d_dst_helper,  sizeof(int32_t) * n_routes, hipMemcpyDeviceToHost);
    hipMemcpy(h_bounds_helper.data(), d_bounds_helper, sizeof(int32_t) * (n_experts + 1), hipMemcpyDeviceToHost);
    hipMemcpy(h_src1_bounded.data(), d_src1_bounded, sizeof(int32_t) * n_routes, hipMemcpyDeviceToHost);
    hipMemcpy(h_dst_bounded.data(),  d_dst_bounded,  sizeof(int32_t) * n_routes, hipMemcpyDeviceToHost);
    hipMemcpy(h_bounds_bounded.data(), d_bounds_bounded, sizeof(int32_t) * (n_experts + 1), hipMemcpyDeviceToHost);

    hipFree(d_ids); hipFree(d_src1_helper); hipFree(d_dst_helper); hipFree(d_bounds_helper);
    hipFree(d_src1_bounded); hipFree(d_dst_bounded); hipFree(d_bounds_bounded);

    ok = check(h_bounds_helper == h_bounds_bounded, "expert_bounds byte-identical") && ok;
    ok = check(h_dst_helper == h_dst_bounded, "ids_dst byte-identical") && ok;
    ok = check(h_src1_helper == h_src1_bounded, "ids_src1 byte-identical") && ok;
    return ok;
}

}  // namespace

int main() {
#if defined(GGML_USE_HIP)
    int n_devices = 0;
    if (hipGetDeviceCount(&n_devices) != hipSuccess || n_devices == 0) {
        std::printf("[mmid-bounded-parity] SKIP: no HIP device\n");
        return 77;
    }
    ggml_backend_t backend = ggml_backend_cuda_init(0);
    if (!backend) {
        std::printf("[mmid-bounded-parity] FAIL: backend init\n");
        return 1;
    }
    ggml_backend_cuda_context * cuda_ctx = (ggml_backend_cuda_context *) backend->context;

    bool ok = true;
    for (int n_tokens : { 512, 1000, 4096, 4097 }) {
        for (int n_channels : { 1, 10 }) {
            for (int inverse = 0; inverse < 2; ++inverse) {
                const bool c = run_case(*cuda_ctx, n_tokens, n_channels, inverse != 0);
                std::printf("[mmid-bounded-parity] tokens=%d channels=%d inverse=%d %s\n",
                    n_tokens, n_channels, inverse, c ? "ok" : "FAIL");
                ok = c && ok;
            }
        }
    }
    ggml_backend_free(backend);
    std::printf("[mmid-bounded-parity] %s\n", ok ? "PASS" : "FAIL");
    return ok ? 0 : 1;
#else
    std::printf("[mmid-bounded-parity] SKIP: HIP-only\n");
    return 77;
#endif
}
