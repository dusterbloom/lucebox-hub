#include "moe-route.cuh"
#include <hip/hip_bf16.h>
#include <cfloat>

// Grid: (WGB, T) blocks, RTHREADS threads/block. WGB WGs compute 2 rows each of the [NE experts + 1 shexp logit]
// BF16/F32 router GEMV for token blockIdx.y; a last-WG-done atomic counter (per token) hands the tail (softmax,
// exact top-NU selection with ties -> lowest expert id, renormalise, sigmoid(shexp)) to whichever WG finishes
// last for that token. Kernel body generalizes server/test/bench/moe_route_kernels.cu's mr_route_kernel (T=1
// prototype, compile-time NE/NU) to runtime NE/NU and a token grid dimension; the per-row dot, softmax and
// tie-break arithmetic are unchanged.
namespace {

constexpr int RTHREADS = 256, RWARPS = RTHREADS / 32;

__device__ __forceinline__ float warp_sum(float v) {
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) v += __shfl_xor(v, o, 32);
    return v;
}
__device__ __forceinline__ float sigmoidf_(float x) { return 1.0f / (1.0f + expf(-x)); }

__device__ __forceinline__ float row_dot_bf16(const uint16_t * w_row, const float * x, int N, int tid, float * red) {
    float sum = 0.0f;
    for (int i = tid; i < N; i += RTHREADS) {
        const __hip_bfloat16 wb = *reinterpret_cast<const __hip_bfloat16 *>(w_row + i);
        sum = fmaf(__bfloat162float(wb), x[i], sum);
    }
    const int lane = tid & 31, warp = tid >> 5;
    sum = warp_sum(sum);
    if (lane == 0) red[warp] = sum;
    __syncthreads();
    float out = 0.0f;
    if (tid == 0) { for (int w = 0; w < RWARPS; ++w) out += red[w]; red[0] = out; }
    __syncthreads();
    return red[0];
}
__device__ __forceinline__ float row_dot_f32(const float * w_row, const float * x, int N, int tid, float * red) {
    float sum = 0.0f;
    for (int i = tid; i < N; i += RTHREADS) sum = fmaf(w_row[i], x[i], sum);
    const int lane = tid & 31, warp = tid >> 5;
    sum = warp_sum(sum);
    if (lane == 0) red[warp] = sum;
    __syncthreads();
    float out = 0.0f;
    if (tid == 0) { for (int w = 0; w < RWARPS; ++w) out += red[w]; red[0] = out; }
    __syncthreads();
    return red[0];
}

// strictly larger wins; exact tie -> LOWER index wins (matches k_argsort_f32_i32<DESC>'s stable tie-break).
__device__ __forceinline__ void amax_take(float & val, int & idx, float ov, int oidx) {
    if (ov > val || (ov == val && oidx < idx)) { val = ov; idx = oidx; }
}
__device__ __forceinline__ void amax_warp(float & val, int & idx) {
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) {
        const float ov = __shfl_xor(val, o, 32);
        const int   oi = __shfl_xor(idx, o, 32);
        amax_take(val, idx, ov, oi);
    }
}

__global__ void __launch_bounds__(RTHREADS) mr_route_kernel(
        const float * __restrict__ mixed, const uint16_t * __restrict__ w_router, const float * __restrict__ w_shexp,
        int N, int NE, int NU, int32_t * __restrict__ sel, float * __restrict__ wsel, float * __restrict__ sh_gate,
        float * __restrict__ logits, int32_t * __restrict__ counter) {
    extern __shared__ float work[];   // [NE]
    __shared__ float red[RWARPS];
    __shared__ float wv[RWARPS];
    __shared__ int   wi[RWARPS];
    __shared__ bool  last;

    const int tid = threadIdx.x, b = blockIdx.x, t = blockIdx.y;
    const int WGB = (NE + 1 + 1) / 2;
    const int row_a = 2 * b, row_b = 2 * b + 1;
    const float * x = mixed + (size_t) t * N;
    float * tok_logits = logits + (size_t) t * (NE + 1);
    int32_t * tok_sel = sel + (size_t) t * NU;
    float * tok_wsel = wsel + (size_t) t * NU;

    if (row_a < NE) {
        const float v = row_dot_bf16(w_router + (size_t) row_a * N, x, N, tid, red);
        if (tid == 0) tok_logits[row_a] = v;
    } else if (row_a == NE) {
        const float v = row_dot_f32(w_shexp, x, N, tid, red);
        if (tid == 0) tok_logits[row_a] = v;
    }
    if (row_b < NE) {
        const float v = row_dot_bf16(w_router + (size_t) row_b * N, x, N, tid, red);
        if (tid == 0) tok_logits[row_b] = v;
    } else if (row_b == NE) {
        const float v = row_dot_f32(w_shexp, x, N, tid, red);
        if (tid == 0) tok_logits[row_b] = v;
    }

    if (tid == 0) {
        const int done = atomicAdd(&counter[t], 1) + 1;
        last = (done == WGB);
    }
    __syncthreads();
    if (!last) return;

    for (int i = tid; i < NE; i += RTHREADS) work[i] = tok_logits[i];
    __syncthreads();

    for (int k = 0; k < NU; ++k) {
        float val = -FLT_MAX; int idx = -1;
        for (int i = tid; i < NE; i += RTHREADS) amax_take(val, idx, work[i], i);
        amax_warp(val, idx);
        const int lane = tid & 31, warp = tid >> 5;
        if (lane == 0) { wv[warp] = val; wi[warp] = idx; }
        __syncthreads();
        if (tid == 0) {
            float bv = wv[0]; int bi = wi[0];
            for (int w = 1; w < RWARPS; ++w) amax_take(bv, bi, wv[w], wi[w]);
            tok_sel[k] = bi;
            work[bi] = -FLT_MAX;
        }
        __syncthreads();
    }

    if (tid == 0) wv[0] = 0.0f;
    __syncthreads();
    float p = 0.0f;
    if (tid < NU) {
        float mx = -FLT_MAX;
        for (int k = 0; k < NU; ++k) mx = fmaxf(mx, tok_logits[tok_sel[k]]);
        p = expf(tok_logits[tok_sel[tid]] - mx);
        atomicAdd(&wv[0], p);
    }
    __syncthreads();
    if (tid < NU) {
        const float sum = fmaxf(wv[0], 6.103515625e-5f);
        tok_wsel[tid] = p / sum;
    }
    if (tid == 0) sh_gate[t] = sigmoidf_(tok_logits[NE]);
}

} // namespace

bool ggml_cuda_moe_route_shape_ok(int64_t N, int64_t NE, int64_t NU, int64_t T) {
    return N > 0 && NE > 0 && NU >= 1 && NU <= RWARPS * 32 && T >= 1 && T <= 8;
}

void ggml_cuda_op_moe_route(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * mixed    = dst->src[0];
    const ggml_tensor * w_router = dst->src[1];
    const ggml_tensor * w_shexp  = dst->src[2];

    const int32_t * ip = (const int32_t *) dst->op_params;
    const int NE = ip[0], NU = ip[1], T = ip[2];
    const int N = (int) mixed->ne[0];
    GGML_ASSERT(ggml_cuda_moe_route_shape_ok(N, NE, NU, T));

    char * base = (char *) dst->data;
    int32_t * sel     = (int32_t *) (base + ggml_moe_route_part_offset(dst, 0));
    float   * wsel    = (float *)   (base + ggml_moe_route_part_offset(dst, 1));
    float   * sh_gate = (float *)   (base + ggml_moe_route_part_offset(dst, 2));

    ggml_cuda_pool_alloc<float>   logits_alloc(ctx.pool(), (size_t) T * (NE + 1));
    ggml_cuda_pool_alloc<int32_t> counter_alloc(ctx.pool(), (size_t) T);
    CUDA_CHECK(cudaMemsetAsync(counter_alloc.get(), 0, (size_t) T * sizeof(int32_t), ctx.stream()));

    const int WGB = (NE + 2) / 2;
    const dim3 grid((unsigned) WGB, (unsigned) T, 1);
    const size_t shmem = (size_t) NE * sizeof(float);
    mr_route_kernel<<<grid, RTHREADS, shmem, ctx.stream()>>>(
        (const float *) mixed->data, (const uint16_t *) w_router->data, (const float *) w_shexp->data,
        N, NE, NU, sel, wsel, sh_gate, logits_alloc.get(), counter_alloc.get());
    CUDA_CHECK(cudaGetLastError());
}
