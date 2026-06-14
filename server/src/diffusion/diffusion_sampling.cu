// CUDA sampling kernel for diffusion denoising. See diffusion_sampling.h.
//
// Kernel design (one block per canvas position):
//   - 1024 threads per block, each owns V/1024 vocab elements (V=262144 → 256 each).
//   - Pass 1: block-reduce to find max logit*temp_inv.
//   - Pass 2: each thread accumulates partial_Z and partial_H over its elements.
//     Block-reduce to get total Z. Each thread computes cumulative partial-Z offset
//     via warp-scan + block-scan, then walks its own 256 elements to locate the
//     CDF crossing (target = u*Z). Argmax tracked per-thread, reduced at end.
//
// No warp divergence on the hot path (vocab stride is warp-friendly).

#ifdef DFLASH27B_BACKEND_CUDA

#include "diffusion_sampling.h"

#include <cuda_runtime.h>
#include <math.h>
#include <stdint.h>
#include <limits.h>

namespace dflash::diffusion {

// ── Device helpers ────────────────────────────────────────────────────────

__device__ __forceinline__ float warp_reduce_max(float v) {
    for (int mask = 16; mask > 0; mask >>= 1)
        v = fmaxf(v, __shfl_xor_sync(0xFFFFFFFF, v, mask));
    return v;
}

__device__ __forceinline__ float warp_reduce_sum(float v) {
    for (int mask = 16; mask > 0; mask >>= 1)
        v += __shfl_xor_sync(0xFFFFFFFF, v, mask);
    return v;
}

// Warp-inclusive prefix sum (returns lane's inclusive partial sum).
__device__ __forceinline__ float warp_prefix_sum(float v) {
    for (int d = 1; d < 32; d <<= 1) {
        float t = __shfl_up_sync(0xFFFFFFFF, v, d);
        if ((threadIdx.x & 31) >= (unsigned)d) v += t;
    }
    return v;
}

// ── Main sampling kernel ──────────────────────────────────────────────────
//
// Grid: (C, 1, 1), Block: (1024, 1, 1).
// logits layout: [C, n_vocab] row-major (canvas position in outer dim).

__global__ void diffusion_sample_kernel(
    const float * __restrict__ logits,   // [C, n_vocab] F32
    const float * __restrict__ u,        // [C] uniform [0,1)
    float                      temp_inv,
    int                        n_vocab,
    int32_t * __restrict__     sampled,
    float   * __restrict__     entropy,
    int32_t * __restrict__     argmax_out)
{
    const int pos  = blockIdx.x;          // canvas position
    const int tid  = threadIdx.x;         // thread within block [0,1023]
    const int BLK  = blockDim.x;          // 1024

    const float * row = logits + (size_t)pos * n_vocab;

    // ── Shared memory ─────────────────────────────────────────────────
    extern __shared__ float smem[];       // BLK floats
    float * smax = smem;                  // BLK floats for block max reduce

    // Each thread's range: [v0, v1)
    // Use round-up division so all vocab elements are covered.
    const int chunk = (n_vocab + BLK - 1) / BLK;  // vocab elements per thread
    const int v0    = tid * chunk;
    const int v1    = min(v0 + chunk, n_vocab);

    // ── Pass 1: find max(row[v] * temp_inv) over all v ───────────────
    float tmax = -1e38f;
    int   amax = 0;
    for (int v = v0; v < v1; ++v) {
        float z = row[v] * temp_inv;
        if (z > tmax) { tmax = z; amax = v; }
    }

    // Warp reduce max
    float wmax = warp_reduce_max(tmax);
    // Pick argmax: need to find the thread holding the warp max
    // Use shfl: thread with tmax == wmax wins; ties broken by smallest lane.
    // We broadcast the amax from the lane that holds wmax.
    {
        // lane of wmax winner (lowest lane with tmax==wmax)
        unsigned mask = __ballot_sync(0xFFFFFFFF, tmax == wmax);
        int winner = __ffs((int)mask) - 1;  // lowest set bit
        amax = __shfl_sync(0xFFFFFFFF, amax, winner);
        tmax = wmax;
    }

    // Block reduce via shared memory (32 warps → 32 values → final reduce)
    const int warp_id = tid >> 5;
    const int lane    = tid & 31;
    if (lane == 0) smax[warp_id] = tmax;
    __syncthreads();

    // Thread 0 of each warp reduces the 32 warp-maxima
    float block_max;
    int   block_amax;
    if (tid < 32) {
        float bm = (tid < (BLK >> 5)) ? smax[tid] : -1e38f;
        block_max = warp_reduce_max(bm);
        // amax is only meaningful from tid < (BLK>>5) warps; pick from winner
        unsigned mask2 = __ballot_sync(0xFFFFFFFF, (tid < (BLK >> 5)) && (bm == block_max));
        int winner2 = __ffs((int)mask2) - 1;
        // winner warp's amax is stored in smax during pass 2; for now borrow register
        // We need the per-warp amax — store in separate smem on warp_id==0 lane.
        // Simplest: after block_max is known, do a second pass to find argmax.
        (void)winner2;
        block_amax = 0;  // filled below
    }
    if (tid == 0) smax[0] = block_max;
    __syncthreads();
    block_max = smax[0];
    __syncthreads();

    // Argmax: each thread checks if its local max == block_max, takes first such v
    int my_amax = -1;
    for (int v = v0; v < v1; ++v) {
        if (row[v] * temp_inv == block_max) { my_amax = v; break; }
    }
    // Block reduce: pick smallest valid amax
    // Use INT_MAX as "no candidate", reduce with min
    int cand = (my_amax >= 0) ? my_amax : INT_MAX;
    // Warp min
    for (int mask = 16; mask > 0; mask >>= 1)
        cand = min(cand, __shfl_xor_sync(0xFFFFFFFF, cand, mask));
    if (lane == 0) ((int*)smax)[warp_id] = cand;
    __syncthreads();
    if (tid < 32) {
        int bc = (tid < (BLK >> 5)) ? ((int*)smax)[tid] : INT_MAX;
        for (int mask2 = 16; mask2 > 0; mask2 >>= 1)
            bc = min(bc, __shfl_xor_sync(0xFFFFFFFF, bc, mask2));
        if (tid == 0) ((int*)smax)[0] = bc;
    }
    __syncthreads();
    block_amax = ((int*)smax)[0];
    __syncthreads();

    // ── Pass 2: partial_Z, partial_H per thread ───────────────────────
    float pZ = 0.0f;
    float pH = 0.0f;
    for (int v = v0; v < v1; ++v) {
        float e = __expf(row[v] * temp_inv - block_max);
        pZ += e;
        // -p*log(p) = -e/Z * log(e/Z) = (e/Z)*(log(Z) - (v*tinv - m))
        // We accumulate: pH_contrib = e * (block_max - row[v]*temp_inv)
        // because: -sum_v (e/Z)*log(e/Z) = -sum_v (e/Z)*(logit*tinv-m-logZ)
        //        = sum_v (e/Z)*(m + logZ - logit*tinv)
        // We'll compute log(Z) after the Z reduction; accumulate e*(m - logit*tinv) now.
        pH += e * (block_max - row[v] * temp_inv);
    }

    // Block-reduce Z
    float wZ = warp_reduce_sum(pZ);
    if (lane == 0) smax[warp_id] = wZ;
    __syncthreads();
    if (tid < 32) {
        float bz = (tid < (BLK >> 5)) ? smax[tid] : 0.0f;
        bz = warp_reduce_sum(bz);
        if (tid == 0) smax[0] = bz;
    }
    __syncthreads();
    float total_Z = smax[0];
    __syncthreads();

    // Block-reduce partial H (= sum_v e*(m - logit*tinv))
    float wH = warp_reduce_sum(pH);
    if (lane == 0) smax[warp_id] = wH;
    __syncthreads();
    if (tid < 32) {
        float bh = (tid < (BLK >> 5)) ? smax[tid] : 0.0f;
        bh = warp_reduce_sum(bh);
        if (tid == 0) smax[0] = bh;
    }
    __syncthreads();
    float total_H_sum = smax[0];
    __syncthreads();

    // Shannon entropy = (total_H_sum / Z) + log(Z)
    // Because: H = sum_v (e/Z)*(m - logit*tinv) + log(Z)
    //            = total_H_sum/Z + log(Z)
    float logZ   = __logf(total_Z > 0.0f ? total_Z : 1e-38f);
    float H      = (total_Z > 0.0f ? total_H_sum / total_Z : 0.0f) + logZ;

    // ── Pass 3: CDF walk to find sampled token ────────────────────────
    // target = u[pos] * total_Z
    float target = u[pos] * total_Z;

    // We need per-thread partial prefix sum of Z to know the CDF offset at v0.
    // Strategy: warp-prefix on pZ, then block-prefix on warp sums.
    // Step 1: warp inclusive prefix of pZ
    float warp_inc_prefix = warp_prefix_sum(pZ);
    // Store warp total (lane 31's value) = sum of this warp
    if (lane == 0) smax[warp_id] = 0.0f;  // init
    __syncthreads();
    if (lane == 31) smax[warp_id] = warp_inc_prefix;
    __syncthreads();
    // Step 2: prefix sum of warp totals (only 32 warps, do it in first warp)
    float warp_offset = 0.0f;
    if (tid < 32) {
        float wt = smax[tid];
        // exclusive prefix sum in first warp
        for (int d = 1; d < 32; d <<= 1) {
            float t = __shfl_up_sync(0xFFFFFFFF, wt, d);
            if (lane >= (unsigned)d) wt += t;
        }
        // wt is now inclusive; store exclusive prefix (wt - original)
        float orig = smax[tid];
        smax[tid] = wt - orig;
    }
    __syncthreads();
    warp_offset = smax[warp_id];  // exclusive sum of all warps before mine
    // My exclusive thread prefix = warp_offset + (warp_inc_prefix - pZ)
    float thread_prefix = warp_offset + (warp_inc_prefix - pZ);

    // Walk my [v0,v1) range; find where cumulative crosses target
    int32_t samp = -1;
    float cum = thread_prefix;
    for (int v = v0; v < v1; ++v) {
        float e = __expf(row[v] * temp_inv - block_max);
        cum += e;
        if (samp < 0 && cum >= target) samp = v;
    }
    // Each thread writes its candidate (or INT_MAX if not found)
    // Block-reduce: pick smallest samp >= 0
    int cand2 = (samp >= 0) ? samp : INT_MAX;
    for (int mask = 16; mask > 0; mask >>= 1)
        cand2 = min(cand2, __shfl_xor_sync(0xFFFFFFFF, cand2, mask));
    if (lane == 0) ((int*)smax)[warp_id] = cand2;
    __syncthreads();
    if (tid < 32) {
        int bc = (tid < (BLK >> 5)) ? ((int*)smax)[tid] : INT_MAX;
        for (int mask2 = 16; mask2 > 0; mask2 >>= 1)
            bc = min(bc, __shfl_xor_sync(0xFFFFFFFF, bc, mask2));
        if (tid == 0) ((int*)smax)[0] = bc;
    }
    __syncthreads();
    int32_t block_samp = ((int*)smax)[0];
    // Fallback: if no thread found a crossing (shouldn't happen; floating point),
    // use argmax (highest probability = minimum entropy = safest default).
    if (block_samp == INT_MAX) block_samp = (int32_t)block_amax;

    // ── Write outputs ─────────────────────────────────────────────────
    if (tid == 0) {
        sampled[pos]    = block_samp;
        entropy[pos]    = H > 0.0f ? H : 0.0f;
        argmax_out[pos] = (int32_t)block_amax;
    }
}

// ── Host launcher ─────────────────────────────────────────────────────────

void diffusion_sample_gpu(
    const float *  logits_dev,
    const float *  u_dev,
    float          temp_inv,
    int            C,
    int            n_vocab,
    int32_t *      sampled_dev,
    float *        entropy_dev,
    int32_t *      argmax_dev,
    cudaStream_t   stream)
{
    const int    BLK  = 1024;
    const size_t smem = BLK * sizeof(float);

    diffusion_sample_kernel<<<C, BLK, smem, stream>>>(
        logits_dev, u_dev, temp_inv, n_vocab,
        sampled_dev, entropy_dev, argmax_dev);
}

}  // namespace dflash::diffusion

#endif  // DFLASH27B_BACKEND_CUDA
