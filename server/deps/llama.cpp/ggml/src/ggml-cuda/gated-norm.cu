#include "gated-norm.cuh"

// same expression as op_sigmoid in unary.cu
static __device__ __forceinline__ float gn_sigmoid(const float x) {
    return 1.0f / (1.0f + expf(-x));
}

// same saturating conversion as mq_f2h_sat in mmb-q8f16.cuh
static __device__ __forceinline__ half gn_f2h_sat(const float v) {
    return __float2half(fabsf(v) > 65504.0f ? copysignf(65504.0f, v) : v);
}

// One block per (head, token) row, the thread mapping and reduction of rms_norm_f32<256, true>.
// NORM = false: sigmoid(z) * x only (the attention output gate), same product as unary_mul(sigmoid).
template <int block_size, bool NORM>
static __global__ void gated_rms_norm_f16_kernel(const float * x, const float * gamma, const float * z, half * dst,
        const int ncols, const int64_t x_s1, const int64_t x_s2, const int64_t z_sh, const int64_t z_st, const int gamma_rows,
        const float eps) {
    const int h = blockIdx.x, t = blockIdx.y, nh = gridDim.x, tid = threadIdx.x;
    const float * xr = x + t * x_s2 + h * x_s1;
    const float * zr = z + t * z_st + (int64_t) h * z_sh;
    half * d = dst + ((int64_t) t * nh + h) * ncols;
    if constexpr (!NORM) {
        for (int col = tid; col < ncols; col += block_size) d[col] = gn_f2h_sat(gn_sigmoid(zr[col]) * xr[col]);
        return;
    }
    float tmp = 0.0f;
    for (int col = tid; col < ncols; col += block_size) {
        const float xi = xr[col];
        tmp += xi * xi;
    }
    extern __shared__ float s_sum[];
    tmp = block_reduce<block_reduce_method::SUM, block_size>(tmp, s_sum);
    const float mean  = tmp / ncols;
    const float scale = rsqrtf(mean + eps);
    const float * g  = gamma + (int64_t) (h % gamma_rows) * ncols;
    for (int col = tid; col < ncols; col += block_size) {
        const float v = scale * xr[col] * g[col];
        d[col] = gn_f2h_sat(gn_sigmoid(zr[col]) * v);
    }
}

// Rows of <= 128 (GDN heads): two rows per 256-thread block, one per 128-thread half, so no thread idles. Each half
// reproduces block_reduce<SUM, 256> over [its 4 warp partials, 4 zero partials] exactly (same warp_reduce_sum on the
// same lane values), so the result is bit-identical to gated_rms_norm_f16_kernel<256, true>. WRITE_Q8 retains the
// F32 result and emits quantize_q8_1's exact 32-lane representation from the same value.
template <bool WRITE_Q8>
static __global__ void gated_rms_norm2_kernel(const float * x, const float * gamma, const float * z, void * dst,
        block_q8_1 * out_q8,
        const int ncols, const int nh, const int64_t x_s1, const int64_t x_s2, const int64_t z_sh, const int64_t z_st,
        const int gamma_rows, const float eps) {
    __shared__ float s_sum[8];
    const int tid = threadIdx.x, grp = tid >> 7, gt = tid & 127, lane = tid & 31, warp = tid >> 5;
    const int h = blockIdx.x * 2 + grp, t = blockIdx.y;
    const bool live = h < nh;
    const float * xr = x + t * x_s2 + (int64_t) (live ? h : 0) * x_s1;
    float tmp = 0.0f;
    if (live && gt < ncols) { const float xi = xr[gt]; tmp += xi * xi; }
    tmp = warp_reduce_sum(tmp);
    if (lane == 0) s_sum[warp] = tmp;
    __syncthreads();
    tmp = lane < 4 ? s_sum[grp * 4 + lane] : 0.0f;
    tmp = warp_reduce_sum(tmp);
    if (!live || gt >= ncols) return;
    const float mean  = tmp / ncols;
    const float scale = rsqrtf(mean + eps);
    const float * g  = gamma + (int64_t) (h % gamma_rows) * ncols;
    const float * zr = z + t * z_st + (int64_t) h * z_sh;
    const float v = scale * xr[gt] * g[gt];
    const float gated = gn_sigmoid(zr[gt]) * v;
    const int64_t di = ((int64_t) t * nh + h) * ncols + gt;
    if constexpr (WRITE_Q8) {
        ((float *) dst)[di] = gated;
        float amax = warp_reduce_max<QK8_1>(fabsf(gated));
        float sum  = warp_reduce_sum<QK8_1>(gated);
        const float d = amax / 127.0f;
        const int lane = gt & (QK8_1 - 1);
        block_q8_1 & q = out_q8[di / QK8_1];
        q.qs[lane] = amax == 0.0f ? 0 : (int8_t) roundf(gated / d);
        if (lane == 0) q.ds = make_half2(d, sum);
    } else {
        ((half *) dst)[di] = gn_f2h_sat(gated);
    }
}

void ggml_cuda_op_gated_rms_norm_f16(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * x = dst->src[0], * gamma = dst->src[1], * z = dst->src[2];
    const int ncols = (int) x->ne[0], nh = (int) x->ne[1], T = (int) x->ne[2];
    GGML_ASSERT(ncols <= 256 && dst->type == GGML_TYPE_F16 && ggml_is_contiguous(dst));
    float eps;
    memcpy(&eps, dst->op_params, sizeof(float));
    // z is [ncols*nh, T] (head stride ncols) or a [ncols, nh, T] view with its own strides.
    const int64_t z_sh = z->ne[0] == ncols * nh ? ncols : (int64_t) (z->nb[1] / sizeof(float));
    const int64_t z_st = z->ne[0] == ncols * nh ? (int64_t) (z->nb[1] / sizeof(float)) : (int64_t) (z->nb[2] / sizeof(float));
    const dim3 grid((unsigned) nh, (unsigned) T, 1);
    const int64_t x_s1 = (int64_t) (x->nb[1] / sizeof(float)), x_s2 = (int64_t) (x->nb[2] / sizeof(float));
    if (gamma && ncols <= 128) {
        gated_rms_norm2_kernel<false><<<dim3((unsigned) ((nh + 1) / 2), (unsigned) T, 1), 256, 0, ctx.stream()>>>(
            (const float *) x->data, (const float *) gamma->data, (const float *) z->data, dst->data, nullptr, ncols, nh,
            x_s1, x_s2, z_sh, z_st, (int) (ggml_nelements(gamma) / ncols), eps);
    } else if (gamma) {
        gated_rms_norm_f16_kernel<256, true><<<grid, 256, 32 * sizeof(float), ctx.stream()>>>(
            (const float *) x->data, (const float *) gamma->data, (const float *) z->data, (half *) dst->data, ncols,
            x_s1, x_s2, z_sh, z_st, (int) (ggml_nelements(gamma) / ncols), eps);
    } else {
        gated_rms_norm_f16_kernel<256, false><<<grid, 256, 0, ctx.stream()>>>(
            (const float *) x->data, nullptr, (const float *) z->data, (half *) dst->data, ncols,
            x_s1, x_s2, z_sh, z_st, 1, eps);
    }
    CUDA_CHECK(cudaGetLastError());
}

void ggml_cuda_gated_rms_norm_q8_1(ggml_backend_cuda_context & ctx,
        const ggml_tensor * x, const ggml_tensor * gamma, const ggml_tensor * z,
        ggml_tensor * dst, block_q8_1 * out_q8, float eps) {
    GGML_ASSERT(x && gamma && z && dst && out_q8);
    GGML_ASSERT(x->type == GGML_TYPE_F32 && gamma->type == GGML_TYPE_F32 &&
                z->type == GGML_TYPE_F32 && dst->type == GGML_TYPE_F32);
    GGML_ASSERT(x->ne[2] == 1 && x->ne[3] == 1 && dst->ne[2] == 1 && dst->ne[3] == 1);
    GGML_ASSERT(ggml_is_contiguous(dst) && ggml_nelements(dst) == ggml_nelements(x));
    const int ncols = (int) x->ne[0], nh = (int) x->ne[1];
    GGML_ASSERT(ncols == 128 && ncols % QK8_1 == 0);
    GGML_ASSERT(ggml_nelements(gamma) % ncols == 0 && ggml_nelements(z) == ggml_nelements(x));
    const int64_t z_sh = z->ne[0] == ncols * nh ? ncols : (int64_t) (z->nb[1] / sizeof(float));
    gated_rms_norm2_kernel<true><<<dim3((unsigned) ((nh + 1) / 2), 1, 1), 256, 0, ctx.stream()>>>(
        (const float *) x->data, (const float *) gamma->data, (const float *) z->data,
        dst->data, out_q8, ncols, nh, (int64_t) (x->nb[1] / sizeof(float)),
        (int64_t) (x->nb[2] / sizeof(float)), z_sh, (int64_t) (z->nb[2] / sizeof(float)),
        (int) (ggml_nelements(gamma) / ncols), eps);
    CUDA_CHECK(cudaGetLastError());
}

static bool gn_disjoint(const void * a, size_t an, const void * b, size_t bn) {
    const uintptr_t pa = (uintptr_t) a, pb = (uintptr_t) b;
    return pa + an <= pb || pb + bn <= pa;
}

extern "C" GGML_BACKEND_API int ggml_cuda_test_gdn_q8_producer(
        const float * x, const float * gamma, const float * z, float * dst,
        block_q8_1 * q8, int gamma_rows, float eps, void * raw_stream) {
    constexpr size_t fbytes = 6144*sizeof(float), qbytes = 6912;
    const size_t gbytes = (size_t) gamma_rows*128*sizeof(float);
    if (!x || !gamma || !z || !dst || !q8 || (gamma_rows != 1 && gamma_rows != 48) || eps < 0 ||
        (x != dst && !gn_disjoint(x, fbytes, dst, fbytes)) ||
        !gn_disjoint(gamma, gbytes, dst, fbytes) || !gn_disjoint(z, fbytes, dst, fbytes) ||
        !gn_disjoint(q8, qbytes, x, fbytes) || !gn_disjoint(q8, qbytes, gamma, gbytes) ||
        !gn_disjoint(q8, qbytes, z, fbytes) || !gn_disjoint(q8, qbytes, dst, fbytes)) return 0;
    cudaStream_t stream = (cudaStream_t) raw_stream;
    gated_rms_norm2_kernel<true><<<dim3(24,1,1),256,0,stream>>>(
        x, gamma, z, dst, q8, 128, 48, 128, 6144, 128, 6144, gamma_rows, eps);
    return cudaGetLastError() == cudaSuccess;
}

// GGML_OP_GDN_TAIL (fusion-design.md K4): rms_norm(x) * gamma * sigmoid(z) -> F32. One block per (head, token),
// ncols threads/block -- same grid/reduction shape as the validated gdn_post_kernel prototype
// (docs/handoffs/third-eye/fusion-exp3.md), just without its Q8_1 side-emit (the decode ssm_out consumer here
// reads the un-rounded F32 activation directly).
static __global__ void gdn_tail_kernel(const float * x, const float * gamma, const float * z, float * dst,
        const int ncols, const int nh, const int64_t x_s1, const int64_t x_s2, const int64_t z_sh, const int64_t z_st,
        const float eps) {
    const int h = blockIdx.x, t = blockIdx.y, tid = threadIdx.x;
    const float * xr = x + t * x_s2 + (int64_t) h * x_s1;
    const float xi = xr[tid];
    __shared__ float s_warp[4];
    const int lane = tid & 31, warp = tid >> 5;
    float sq = warp_reduce_sum(xi * xi);
    if (lane == 0) s_warp[warp] = sq;
    __syncthreads();
    const float total = s_warp[0] + s_warp[1] + s_warp[2] + s_warp[3];
    const float scale = rsqrtf(total / ncols + eps);
    const float v = scale * xi * gamma[tid];
    const float * zr = z + t * z_st + (int64_t) h * z_sh;
    dst[((int64_t) t * nh + h) * ncols + tid] = v / (1.0f + expf(-zr[tid]));
}

bool ggml_cuda_gdn_tail_shape_ok(int64_t ncols) {
    return ncols > 0 && ncols <= 128;
}

void ggml_cuda_op_gdn_tail(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * x = dst->src[0], * gamma = dst->src[1], * z = dst->src[2];
    const int ncols = (int) x->ne[0], nh = (int) x->ne[1], T = (int) x->ne[2];
    GGML_ASSERT(ggml_cuda_gdn_tail_shape_ok(ncols) && dst->type == GGML_TYPE_F32 && ggml_is_contiguous(dst));
    float eps;
    memcpy(&eps, dst->op_params, sizeof(float));
    const int64_t z_sh = z->ne[0] == ncols * nh ? ncols : (int64_t) (z->nb[1] / sizeof(float));
    const int64_t z_st = z->ne[0] == ncols * nh ? (int64_t) (z->nb[1] / sizeof(float)) : (int64_t) (z->nb[2] / sizeof(float));
    const int64_t x_s1 = (int64_t) (x->nb[1] / sizeof(float)), x_s2 = (int64_t) (x->nb[2] / sizeof(float));
    gdn_tail_kernel<<<dim3((unsigned) nh, (unsigned) T, 1), ncols, 0, ctx.stream()>>>(
        (const float *) x->data, (const float *) gamma->data, (const float *) z->data, (float *) dst->data,
        ncols, nh, x_s1, x_s2, z_sh, z_st, eps);
    CUDA_CHECK(cudaGetLastError());
}
