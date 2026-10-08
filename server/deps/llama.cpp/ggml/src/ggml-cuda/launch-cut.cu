#include "launch-cut.cuh"

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <set>
#include <string>

// Matches CUDA_QUANTIZE_BLOCK_SIZE in quantize.cuh (not included here, to keep
// this translation unit's reference path independent of quantize.cu/.cuh).
#define CUDA_QUANTIZE_BLOCK_SIZE_REF 256

// ---------------------------------------------------------------------------
// 0. Site-analysis instrumentation (see launch-cut.cuh for the full comment).
bool ggml_cuda_launch_cut_debug_enabled() {
    static const bool on = [] {
        const char * e = std::getenv("LUCE_QWEN_LAUNCH_CUT_DEBUG");
        return e && e[0] == '1' && e[1] == '\0';
    }();
    return on;
}

void ggml_cuda_launch_cut_debug_note(
        const ggml_tensor * dst, const ggml_tensor * src1, int src0_type, int64_t ne10) {
    static std::set<std::string> seen;
    char key[256];
    std::snprintf(key, sizeof(key), "dst=%s src0_type=%d ne10=%lld src1_op=%s src1=%s",
        dst && dst->name[0] ? dst->name : "?", src0_type, (long long) ne10,
        src1 ? ggml_op_name(src1->op) : "?", src1 && src1->name[0] ? src1->name : "?");
    if (seen.insert(key).second) {
        fprintf(stderr, "[LUCE_QWEN_LAUNCH_CUT_DEBUG] new site #%zu: %s\n", seen.size(), key);
    }
}

void ggml_cuda_launch_cut_debug_dump() {
    // The note() function already prints on first sight; this is a
    // lightweight marker for driver teardown hooks to confirm the
    // instrumentation ran at all (distinguishes "0 sites" from "never called").
    fprintf(stderr, "[LUCE_QWEN_LAUNCH_CUT_DEBUG] dump requested (see preceding 'new site' lines for the full table)\n");
}

// ---------------------------------------------------------------------------
// A. Fused kernel: rms_norm(x) * gamma, with the Q8_1 quantization of that
// same per-column value folded into the same launch.
//
// Norm half is bit-identical to rms_norm_f32<block_size, /*do_multiply=*/true>
// in norm.cu: one block per (row, channel, sample), block_reduce<SUM,block_size>
// over the sum of squares, scale = rsqrtf(mean + eps), second pass recomputes
// `scale * x[col] * mul[col]` from the same registers/global reads (no
// reassociation, no extra rounding step relative to the unfused path, which
// also recomputes x[col] fresh when quantize_q8_1 reads back the norm's F32
// output -- same value, same bits).
//
// Q8_1 half is bit-identical to quantize_q8_1's q8_1_store_lane in
// quantize.cu: per-32-lane (one hardware warp) amax/sum via
// warp_reduce_max<QK8_1>/warp_reduce_sum<QK8_1>, d = amax/127, q = round(v/d),
// ds = (d, sum). Requires ncols a multiple of QK8_1=32 (true for every
// embedding-dim-sized row in this model, 2560 % 32 == 0) and block_size a
// multiple of QK8_1 (so each warp's 32 lanes map to 32 column-contiguous
// elements, identical to quantize_q8_1's own thread-to-column mapping).
template <int block_size>
__launch_bounds__(block_size, 1)
static __global__ void rms_norm_mul_q8_1_f32(
        const float * __restrict__ x, const float * __restrict__ mul,
        float * __restrict__ dst, block_q8_1 * __restrict__ out_q8,
        const int ncols,
        const int64_t stride_row, const int64_t stride_channel, const int64_t stride_sample,
        const int64_t mul_stride_row, const int64_t mul_stride_channel, const int64_t mul_stride_sample,
        const float eps) {
    static_assert(block_size % QK8_1 == 0, "block_size must be a multiple of QK8_1");

    const int nrows = gridDim.x, nchannels = gridDim.y;
    const int row = blockIdx.x, channel = blockIdx.y, sample = blockIdx.z, tid = threadIdx.x;

    x   += sample*stride_sample + channel*stride_channel + row*stride_row;
    mul += sample*mul_stride_sample + channel*mul_stride_channel + row*mul_stride_row;
    dst += ((int64_t) (sample*nchannels + channel)*nrows + row) * (int64_t) ncols;
    out_q8 += (((int64_t) (sample*nchannels + channel)*nrows + row) * (int64_t) ncols) / QK8_1;

    float tmp = 0.0f;
    for (int col = tid; col < ncols; col += block_size) {
        const float xi = x[col];
        tmp += xi * xi;
    }

    extern __shared__ float s_sum[];
    tmp = block_reduce<block_reduce_method::SUM, block_size>(tmp, s_sum);
    const float mean  = tmp / ncols;
    const float scale = rsqrtf(mean + eps);

    for (int col = tid; col < ncols; col += block_size) {
        const float v = scale * x[col] * mul[col];
        dst[col] = v;

        const float amax = warp_reduce_max<QK8_1>(fabsf(v));
        const float sum   = warp_reduce_sum<QK8_1>(v);
        const float d     = amax / 127.0f;
        const int8_t q    = amax == 0.0f ? 0 : (int8_t) roundf(v / d);

        const int iqs = col & (QK8_1 - 1);
        block_q8_1 & blk = out_q8[col / QK8_1];
        blk.qs[iqs] = q;
        if (iqs == 0) {
            blk.ds = make_half2(d, sum);
        }
    }
}

void ggml_cuda_rms_norm_mul_q8_1_cuda(
        const float * x, const float * mul, float * dst, block_q8_1 * out_q8,
        const int ncols, const int nrows, const int nchannels, const int nsamples,
        const int64_t stride_row, const int64_t stride_channel, const int64_t stride_sample,
        const int64_t mul_stride_row, const int64_t mul_stride_channel, const int64_t mul_stride_sample,
        const float eps, cudaStream_t stream) {
    GGML_ASSERT(ncols % QK8_1 == 0);
    const dim3 blocks_num(nrows, nchannels, nsamples);
    if (ncols < 1024) {
        const dim3 block_dims(256, 1, 1);
        rms_norm_mul_q8_1_f32<256><<<blocks_num, block_dims, 32 * sizeof(float), stream>>>(
            x, mul, dst, out_q8, ncols, stride_row, stride_channel, stride_sample,
            mul_stride_row, mul_stride_channel, mul_stride_sample, eps);
    } else {
        const dim3 block_dims(1024, 1, 1);
        rms_norm_mul_q8_1_f32<1024><<<blocks_num, block_dims, 32 * sizeof(float), stream>>>(
            x, mul, dst, out_q8, ncols, stride_row, stride_channel, stride_sample,
            mul_stride_row, mul_stride_channel, mul_stride_sample, eps);
    }
}

// ---------------------------------------------------------------------------
// B. Reference baseline: a from-scratch reproduction of the two existing,
// unfused kernels (rms_norm_f32<block,true> in norm.cu, quantize_q8_1 in
// quantize.cu), duplicated here -- not called -- so the bit-identity test
// below is a pure kernel-vs-kernel comparison with no dependency on norm.cu
// or quantize.cu internals (both are `static`, file-local there), matching
// the hc_combine_norm test's isolation pattern (test_hc_cn_bitexact.cpp).
template <int block_size>
__launch_bounds__(block_size, 1)
static __global__ void ref_rms_norm_mul_f32(
        const float * __restrict__ x, const float * __restrict__ mul, float * __restrict__ dst,
        const int ncols, const int64_t stride_row, const int64_t mul_stride_row, const float eps) {
    const int row = blockIdx.x, tid = threadIdx.x;
    x   += (int64_t) row * stride_row;
    mul += (int64_t) row * mul_stride_row;
    dst += (int64_t) row * ncols;

    float tmp = 0.0f;
    for (int col = tid; col < ncols; col += block_size) {
        const float xi = x[col];
        tmp += xi * xi;
    }
    extern __shared__ float s_sum[];
    tmp = block_reduce<block_reduce_method::SUM, block_size>(tmp, s_sum);
    const float mean  = tmp / ncols;
    const float scale = rsqrtf(mean + eps);
    for (int col = tid; col < ncols; col += block_size) {
        dst[col] = scale * x[col] * mul[col];
    }
}

// Exact reproduction of quantize_q8_1's q8_1_store_lane (quantize.cu), applied
// to a single contiguous F32 row (i_cont == col, matching ne1==ne2==ne3==1).
__launch_bounds__(CUDA_QUANTIZE_BLOCK_SIZE_REF, 1)
static __global__ void ref_quantize_q8_1(
        const float * __restrict__ x, block_q8_1 * __restrict__ y, const int ncols) {
    const int col = blockIdx.x * CUDA_QUANTIZE_BLOCK_SIZE_REF + threadIdx.x;
    if (col >= ncols) {
        return;
    }
    const float xi   = x[col];
    const float amax = warp_reduce_max<QK8_1>(fabsf(xi));
    const float sum  = warp_reduce_sum<QK8_1>(xi);
    const float d     = amax / 127.0f;
    const int8_t q    = amax == 0.0f ? 0 : (int8_t) roundf(xi / d);
    const int iqs = col & (QK8_1 - 1);
    block_q8_1 & blk = y[col / QK8_1];
    blk.qs[iqs] = q;
    if (iqs == 0) {
        blk.ds = make_half2(d, sum);
    }
}

extern "C" GGML_BACKEND_API int ggml_cuda_test_launch_cut_rms_norm_mul_q8_1(
        const float * x, const float * gamma,
        float * dst_a, void * q8_a, float * dst_b, void * q8_b,
        const int ncols, const int nrows, const float eps, void * raw_stream) {
    if (!x || !gamma || !dst_a || !q8_a || !dst_b || !q8_b || ncols <= 0 || nrows <= 0 ||
        ncols % QK8_1 != 0) {
        return 0;
    }
    cudaStream_t stream = (cudaStream_t) raw_stream;

    // A: the fused candidate kernel.
    ggml_cuda_rms_norm_mul_q8_1_cuda(
        x, gamma, dst_a, (block_q8_1 *) q8_a, ncols, nrows, 1, 1,
        ncols, 0, 0, ncols, 0, 0, eps, stream);

    // B: the from-scratch two-kernel baseline reproduction.
    if (ncols < 1024) {
        ref_rms_norm_mul_f32<256><<<nrows, 256, 32 * sizeof(float), stream>>>(
            x, gamma, dst_b, ncols, ncols, ncols, eps);
    } else {
        ref_rms_norm_mul_f32<1024><<<nrows, 1024, 32 * sizeof(float), stream>>>(
            x, gamma, dst_b, ncols, ncols, ncols, eps);
    }
    const int ref_blocks = (ncols + CUDA_QUANTIZE_BLOCK_SIZE_REF - 1) / CUDA_QUANTIZE_BLOCK_SIZE_REF;
    for (int row = 0; row < nrows; ++row) {
        ref_quantize_q8_1<<<ref_blocks, CUDA_QUANTIZE_BLOCK_SIZE_REF, 0, stream>>>(
            dst_b + (int64_t) row * ncols, ((block_q8_1 *) q8_b) + (int64_t) row * (ncols / QK8_1), ncols);
    }

    return cudaGetLastError() == cudaSuccess;
}
