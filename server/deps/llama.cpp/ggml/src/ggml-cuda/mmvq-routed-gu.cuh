// Routed gate/up GEMV for one decode token (MUL_MAT_ID, ncols_dst == 1,
// SwiGLU fusion) on the Qwen4exp expert profile: 512 experts, 10 routed,
// K = 2560 -> M = 640, gate/up Q4_K or Q5_K.
//
// The legacy path (mul_mat_vec_q<..., row_warps = 8>) has one wave walk a row
// of gate AND the same row of up, so every wave drives two weight streams
// 1.5 GB apart.  Here a block is c_pairs pairs of waves: wave 2p streams
// c_rows adjacent rows of gate, wave 2p+1 the same rows of up, and the pair
// combines through LDS.  Each wave now streams one contiguous row set, which
// is what gufo's mul_mat_vec_q_moe<Q4_K, 2, true> does.
//
// Exactness: the lane -> K-block mapping (kbx = lane / (qi/vdr) + iter *
// blocks_per_iter), the per-lane accumulation order, warp_reduce_sum and the
// epilogue (up * silu(gate)) are the legacy kernel's, so the result is
// bit-identical to it.
#pragma once

#include "common.cuh"
#include "unary.cuh"
#include "vecdotq.cuh"

// vec_dot_q4_K_q8_1 with the sub-block scale/min pair formed branch-free: the
// twelve scale bytes are read as three words and both packings are built
// before a select.  With the branch on j (vecdotq.cuh), the compiler sinks the
// third word's load into the j >= 2 side and its wait then covers every load
// the wave has in flight.  Integer-only, same aux words, same float math:
// bit-identical to vec_dot_q4_K_q8_1 (gufo, MIT, vecdotq.hpp q4_K_scale_pair).
static __device__ __forceinline__ float vec_dot_q4_K_q8_1_scale_pair(
    const void * __restrict__ vbq, const block_q8_1 * __restrict__ bq8_1, const int & kbx, const int & iqs) {
    const block_q4_K * bq4_K = (const block_q4_K *) vbq + kbx;

    int   v[2];
    int   u[2*QR4_K];
    float d8[QR4_K];

    const int bq8_offset = QR4_K * ((iqs/2) / (QI8_1/2));
    const int * q4 = (const int *)(bq4_K->qs + 16 * bq8_offset + 4 * ((iqs/2)%4));
    v[0] = q4[0];
    v[1] = q4[4];

    const int j = bq8_offset/2;
    const uint32_t * words = (const uint32_t *) bq4_K->scales;
    const int shift = 16*(j & 1);
    const uint32_t lo  = (words[0] >> shift) & 0xffff; // scales[j % 2]
    const uint32_t mid = (words[1] >> shift) & 0xffff; // scales[j % 2 + 2]
    const uint32_t hi  = (words[2] >> shift) & 0xffff; // scales[j % 2 + 4]
    uint16_t aux[2];
    aux[0] = j < 2 ? (lo  & 0x3f3f) : (((hi >> 0) & 0x0f0f) | ((lo  & 0xc0c0) >> 2));
    aux[1] = j < 2 ? (mid & 0x3f3f) : (((hi >> 4) & 0x0f0f) | ((mid & 0xc0c0) >> 2));
    const uint8_t * sc = (const uint8_t *) aux;
    const uint8_t * m  = sc + 2;

#pragma unroll
    for (int i = 0; i < QR4_K; ++i) {
        const block_q8_1 * bq8i = bq8_1 + bq8_offset + i;
        d8[i] = __low2float(bq8i->ds);
        const int * q8 = (const int *) bq8i->qs + ((iqs/2)%4);
        u[2*i+0] = q8[0];
        u[2*i+1] = q8[4];
    }

    return vec_dot_q4_K_q8_1_impl_vmmq(v, u, sc, m, bq4_K->dm, d8);
}

template <ggml_type type, bool c_scale_pair> struct mmvq_routed_gu_traits;
template <bool c_scale_pair> struct mmvq_routed_gu_traits<GGML_TYPE_Q4_K, c_scale_pair> {
    static constexpr int vdr = VDR_Q4_K_Q8_1_MMVQ;
    static constexpr auto vec_dot = c_scale_pair ? vec_dot_q4_K_q8_1_scale_pair : vec_dot_q4_K_q8_1;
};
template <bool c_scale_pair> struct mmvq_routed_gu_traits<GGML_TYPE_Q5_K, c_scale_pair> {
    static constexpr int vdr = VDR_Q5_K_Q8_1_MMVQ;
    static constexpr auto vec_dot = vec_dot_q5_K_q8_1;
};

template <ggml_type type, int c_rows, int c_pairs, bool c_scale_pair = true>
__launch_bounds__(ggml_cuda_get_physical_warp_size()*2*c_pairs, 1)
static __global__ void mul_mat_vec_q_routed_gate_up(
        const void * __restrict__ vgate, const void * __restrict__ vup, const void * __restrict__ vy,
        const int32_t * __restrict__ ids, float * __restrict__ dst,
        const uint32_t ncols_x, const uint32_t stride_row_x,
        const uint32_t stride_channel_x, const uint32_t stride_channel_dst) {
    constexpr int qk  = ggml_cuda_type_traits<type>::qk;
    constexpr int qi  = ggml_cuda_type_traits<type>::qi;
    constexpr int vdr = mmvq_routed_gu_traits<type, c_scale_pair>::vdr;
    constexpr int warp_size = ggml_cuda_get_physical_warp_size();
    constexpr int blocks_per_iter = vdr*warp_size/qi;
    constexpr auto vec_dot_q_cuda = mmvq_routed_gu_traits<type, c_scale_pair>::vec_dot;
    static_assert(c_rows >= 1 && c_rows <= 8 && c_pairs >= 1 && c_pairs <= 8, "wave pair shape");

    const int  pair  = threadIdx.y >> 1;
    const bool is_up = (threadIdx.y & 1) != 0;
    const int  row0  = c_rows*(c_pairs*blockIdx.x + pair);
    const uint32_t channel_dst = blockIdx.y;

    // Negative MUL_MAT_ID entries are masked routes: no weight read, zero out.
    const int32_t id_raw = ids[channel_dst];
    const bool valid = id_raw >= 0;
    const uint32_t channel_x = valid ? (uint32_t) id_raw : 0u;

    const void * __restrict__ vx = is_up ? vup : vgate;
    const block_q8_1 * __restrict__ y = (const block_q8_1 *) vy;
    const int blocks_per_row_x = ncols_x / qk;
    const int kbx_base = channel_x*stride_channel_x + row0*stride_row_x;

    float tmp[c_rows] = {0.0f};
    if (valid) {
        for (int kbx = threadIdx.x / (qi/vdr); kbx < blocks_per_row_x; kbx += blocks_per_iter) {
            const int kby = kbx * (qk/QK8_1);
            const int kqs = vdr * (threadIdx.x % (qi/vdr));
#pragma unroll
            for (int i = 0; i < c_rows; ++i) {
                tmp[i] += vec_dot_q_cuda(vx, &y[kby], kbx_base + i*stride_row_x + kbx, kqs);
            }
        }
    }
#pragma unroll
    for (int i = 0; i < c_rows; ++i) {
        tmp[i] = warp_reduce_sum<warp_size>(tmp[i]);
    }

    __shared__ float gate_s[c_pairs][c_rows];
    if (!is_up && threadIdx.x < c_rows) {
        float g = 0.0f;
#pragma unroll
        for (int i = 0; i < c_rows; ++i) {
            if (threadIdx.x == i) g = tmp[i];
        }
        gate_s[pair][threadIdx.x] = g;
    }
    __syncthreads();
    if (is_up && threadIdx.x < c_rows) {
        float result = 0.0f;
        if (valid) {
#pragma unroll
            for (int i = 0; i < c_rows; ++i) {
                if (threadIdx.x == i) result = tmp[i];
            }
            result *= ggml_cuda_op_silu_single(gate_s[pair][threadIdx.x]);
        }
        dst[channel_dst*stride_channel_dst + row0 + threadIdx.x] = result;
    }
}

template <ggml_type type, int c_rows, int c_pairs, bool c_scale_pair = true>
static void launch_mul_mat_vec_q_routed_gate_up(
        const void * vgate, const void * vup, const void * vy, const int32_t * ids, float * dst,
        const uint32_t ncols_x, const uint32_t nrows_x, const uint32_t nchannels_dst,
        const uint32_t stride_row_x, const uint32_t stride_channel_x, const uint32_t stride_channel_dst,
        const int warp_size, cudaStream_t stream) {
    GGML_ASSERT(nrows_x % (c_rows*c_pairs) == 0);
    const dim3 grid(nrows_x/(c_rows*c_pairs), nchannels_dst, 1);
    const dim3 block(warp_size, 2*c_pairs, 1);
    mul_mat_vec_q_routed_gate_up<type, c_rows, c_pairs, c_scale_pair><<<grid, block, 0, stream>>>(
        vgate, vup, vy, ids, dst, ncols_x, stride_row_x, stride_channel_x, stride_channel_dst);
}
