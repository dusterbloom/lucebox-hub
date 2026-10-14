#pragma once
// F16 single-launch Q4_K gate/up SwiGLU GEMM for gfx1151 (LUCE_MMB_GLU_F16=0 restores the bf16 kernel in
// mmb-routed.cuh). Included from mmb-routed.cuh; same block (64 gate + 64 up rows x BN tokens, 8 waves, the
// balanced wave map, the persistent atomic tile schedule), different operands and LDS plan:
//  - weights are dequantized once per block at LDS-store time into F16 with the 1024-magic trick (v_perm to
//    0x64|q = 1024+q as F16, pk_add -1024 = q exactly, pk_fma scale/-min): 15 VALU per 8 elements against ~70 for
//    the f32 fma + bf16 RNE pack path, and without the half-wave duplication a fragment-time dequant pays on wave32;
//  - activations are F16 (the bf16 values converted once per layer), WMMA f32_16x16x16_f16;
//  - LDS tiles are chunk-major, [16-byte chunk of the 64-wide K step][row]: a fragment read is 128 contiguous
//    bytes per eight lanes (conflict-free without the 16-B row pad) and the chunk is an immediate offset, so no
//    address registers; 32768 B (gate 8 K + up 8 K + tokens 16 K) and four blocks fit a 128 KB WGP (the padded
//    bf16 kernel is 41 KB, three); the work-item slot lives in the gate tile, written between the K loop and the
//    epilogue when that tile is dead.
// Tried and dropped (bench/exact/prefill/GUFO-PARITY.md, 'Routed gate/up F16'): a 128-row m slice with every wave on
// all eight token fragments (128 accumulators per lane spill at 256 VGPRs, 11.2 ms vs 9.6), branchless masked token
// loads (+0.7 ms: dead rows cost real loads), per-slice weight loads and a prefetched next item (+2 ms together),
// the fast silu (neutral).
// Not bit-exact with the bf16 kernels: F16 weights carry three more mantissa bits, so the error against an exact
// (double) reference is lower than the bf16 path's.
// Not bit-exact with the bf16 kernels: F16 weights carry three more mantissa bits, so the error against an exact
// (double) reference is lower (bench/exact/prefill/GUFO-PARITY.md, 'Routed gate/up F16').

typedef _Float16 mmbf_v16h __attribute__((ext_vector_type(16)));
typedef _Float16 mmbf_h2   __attribute__((ext_vector_type(2)));

// (d*sc, -dm*mn) of Q4_K sub-block I as a packed F16 pair (meta = the 16-byte block header: d|dm, 12 scale bytes).
template <int I>
__device__ __forceinline__ mmbf_h2 mmbf_q4k_sb(const uint4 meta) {
    auto byte = [&](const int i) -> uint32_t { const uint32_t w = i < 4 ? meta.y : i < 8 ? meta.z : meta.w; return (w >> (8 * (i & 3))) & 255u; };
    const uint32_t sc = I < 4 ? byte(I) & 63 : (byte(I + 4) & 15) | ((byte(I - 4) >> 6) << 4);
    const uint32_t mn = I < 4 ? byte(I + 4) & 63 : (byte(I + 4) >> 4) | ((byte(I) >> 6) << 4);
    const float d = mmb_h2f((uint16_t) meta.x), dm = mmb_h2f((uint16_t) (meta.x >> 16));
    return mmbf_h2{(_Float16) (d * (float) sc), (_Float16) (-(dm * (float) mn))};
}

// 16 code bytes q -> 16 low-nibble halves (lo0, lo1: K order of the bytes) and 16 high-nibble halves (hi0, hi1),
// sub-block pairs slo / shi.
__device__ __forceinline__ void mmbf_dq16(const uint4 q, const mmbf_h2 slo, const mmbf_h2 shi, uint4 & lo0, uint4 & lo1, uint4 & hi0, uint4 & hi1) {
    const mmbf_h2 m1024 = mmbf_h2{(_Float16) -1024.f, (_Float16) -1024.f};
    const mmbf_h2 sl = mmbf_h2{slo.x, slo.x}, bl = mmbf_h2{slo.y, slo.y}, sh = mmbf_h2{shi.x, shi.x}, bh = mmbf_h2{shi.y, shi.y};
    auto conv = [&](const uint32_t n, const mmbf_h2 s2, const mmbf_h2 b2, uint32_t & o0, uint32_t & o1) {
        // v_perm: {0x64, n_b1, 0x64, n_b0} and {.., n_b3, .., n_b2} = F16 1024 + q, exact
        const uint32_t p0 = __builtin_amdgcn_perm(n, 0x64646464u, 0x01050004u), p1 = __builtin_amdgcn_perm(n, 0x64646464u, 0x03070206u);
        o0 = __builtin_bit_cast(uint32_t, __builtin_elementwise_fma(__builtin_bit_cast(mmbf_h2, p0) + m1024, s2, b2));
        o1 = __builtin_bit_cast(uint32_t, __builtin_elementwise_fma(__builtin_bit_cast(mmbf_h2, p1) + m1024, s2, b2));
    };
    conv(q.x & 0x0F0F0F0Fu, sl, bl, lo0.x, lo0.y); conv(q.y & 0x0F0F0F0Fu, sl, bl, lo0.z, lo0.w);
    conv(q.z & 0x0F0F0F0Fu, sl, bl, lo1.x, lo1.y); conv(q.w & 0x0F0F0F0Fu, sl, bl, lo1.z, lo1.w);
    conv((q.x >> 4) & 0x0F0F0F0Fu, sh, bh, hi0.x, hi0.y); conv((q.y >> 4) & 0x0F0F0F0Fu, sh, bh, hi0.z, hi0.w);
    conv((q.z >> 4) & 0x0F0F0F0Fu, sh, bh, hi1.x, hi1.y); conv((q.w >> 4) & 0x0F0F0F0Fu, sh, bh, hi1.z, hi1.w);
}

// One (tile, 64-row m slice) work item. LDS tiles are chunk-major: element offset of 16-byte chunk c of row r is
// c * ROWS * 8 + r * 8. Thread roles: weight staging = (matrix tid>>7, 16-byte half (tid>>6)&1 of each 32-byte Q4_K
// slice, row tid&63: consecutive lanes write consecutive rows); token staging = eight consecutive lanes take eight
// consecutive rows at one chunk (conflict-free writes, 64 contiguous bytes per row per load). Wave w computes rows
// 16*(w&3) of gate and up against token fragments (w>>2) + 2 j. `fetch` (thread 0) returns the next work item; it
// is parked in the gate tile between the K loop and the epilogue.
template <int BN, typename XRowFn, typename DRowFn, typename FetchFn>
__device__ __forceinline__ void mmb_tile_gemm_glu_f16(const uint8_t * __restrict__ Wg, const uint8_t * __restrict__ Wu, const size_t wrow_bytes, const int a_rows,
        const _Float16 * __restrict__ Xh, const int K, XRowFn xrow, float * __restrict__ D, uint16_t * __restrict__ Dh, const bool store_f32, const int M, DRowFn drow, const int m0,
        const int n_cols, _Float16 * Ag, _Float16 * Au, _Float16 * Bs, FetchFn fetch) {
    constexpr int TN = BN / 32, B_ITEMS = (BN * 8) / MMB_NT, AP = 64 * 8, BP = BN * 8;   // halves per chunk plane
    static_assert(BN == 128, "token staging map below assumes 128 rows x 8 chunks = 4 chunks per thread");
    const int tid = threadIdx.x, lane = tid & 31, wave = __builtin_amdgcn_readfirstlane(tid >> 5);
    const int wr = wave & 3, wg = wave >> 2;
    // token chunk i of this thread: c = tid + 256 i; lane -> (row lane&7 + 8*(c>>5 & 15), chunk lane>>3 + 4*(c>>9))
    uint4 bst[B_ITEMS];
    int brow[B_ITEMS], bchunk[B_ITEMS], bslot[B_ITEMS];
#pragma unroll
    for (int i = 0; i < B_ITEMS; ++i) {
        const int c = tid + i * MMB_NT, row = (lane & 7) + 8 * ((c >> 5) & 15), chunk = (lane >> 3) + 4 * (c >> 9);
        brow[i] = xrow(row); bchunk[i] = chunk * 8; bslot[i] = chunk * BP + row * 8;
    }
    const int s_mat = tid >> 7, s_half = (tid >> 6) & 1, s_row = tid & 63;
    _Float16 * As = (s_mat ? Au : Ag) + s_row * 8;
    const bool wlive = s_row < a_rows;
    const uint8_t * wp = (s_mat ? Wu : Wg) + (size_t) s_row * wrow_bytes;
    uint4 sqm, sq0, sq1, sq2, sq3;
    auto load_w = [&](const int kb) {
        if (wlive) {
            const uint8_t * p = wp + (size_t) kb * sizeof(block_q4_K);
            sqm = *(const uint4 *) p;
            p += 16 + s_half * 16;
            sq0 = *(const uint4 *) p; sq1 = *(const uint4 *) (p + 32); sq2 = *(const uint4 *) (p + 64); sq3 = *(const uint4 *) (p + 96);
        } else { sqm = sq0 = sq1 = sq2 = sq3 = make_uint4(0, 0, 0, 0); }
    };
    auto load_b = [&](const int ks) {
#pragma unroll
        for (int i = 0; i < B_ITEMS; ++i)
            bst[i] = (brow[i] >= 0) ? *(const uint4 *) (Xh + (size_t) brow[i] * K + ks * 64 + bchunk[i]) : make_uint4(0, 0, 0, 0);
    };
    auto store_lds = [&](auto sc) {   // slice S (64 K) of the loaded Q4_K block + the token registers
        constexpr int S = decltype(sc)::value;
        const uint4 q = S == 0 ? sq0 : S == 1 ? sq1 : S == 2 ? sq2 : sq3;
        uint4 l0, l1, h0, h1;
        mmbf_dq16(q, mmbf_q4k_sb<2 * S>(sqm), mmbf_q4k_sb<2 * S + 1>(sqm), l0, l1, h0, h1);
        // low nibbles = elements 16*half .. +16 of the slice (chunks 2*half, 2*half+1), high nibbles = 32 + 16*half ..
        *(uint4 *) (As + (2 * s_half) * AP)     = l0; *(uint4 *) (As + (2 * s_half + 1) * AP) = l1;
        *(uint4 *) (As + (4 + 2 * s_half) * AP) = h0; *(uint4 *) (As + (5 + 2 * s_half) * AP) = h1;
#pragma unroll
        for (int i = 0; i < B_ITEMS; ++i) *(uint4 *) (Bs + bslot[i]) = bst[i];
    };
    const int nblk = K / 256, nks = nblk * 4;
    const int cm = lane & 15, cn = lane >> 4, col = m0 + wr * 16 + cm, r = lane & 15;
    const _Float16 * Agr = Ag + (wr * 16 + r) * 8, * Aur = Au + (wr * 16 + r) * 8, * Bsr = Bs + (wg * 16 + r) * 8;
    auto kloop = [&](auto njc) __attribute__((always_inline)) {
        constexpr int NJ = decltype(njc)::value;
        v8f accg[NJ > 0 ? NJ : 1], accu[NJ > 0 ? NJ : 1];
#pragma unroll
        for (int j = 0; j < NJ; ++j)
#pragma unroll
            for (int e = 0; e < 8; ++e) { accg[j][e] = 0.f; accu[j][e] = 0.f; }
        load_w(0); load_b(0); store_lds(std::integral_constant<int, 0>{}); __syncthreads();
        auto step = [&](const int kb, auto sc) __attribute__((always_inline)) {
            constexpr int S = decltype(sc)::value;
            const int ks = kb * 4 + S;
            const bool more = ks + 1 < nks;
            if (S == 3 && kb + 1 < nblk) load_w(kb + 1);   // slice 3 is already in LDS: the registers are free
            if (more) load_b(ks + 1);
#pragma unroll
            for (int kk = 0; kk < 4; ++kk) {
                const mmbf_v16h ag = __builtin_bit_cast(mmbf_v16h, (uint4[2]){*(const uint4 *) (Agr + (2 * kk) * AP), *(const uint4 *) (Agr + (2 * kk + 1) * AP)});
                const mmbf_v16h au = __builtin_bit_cast(mmbf_v16h, (uint4[2]){*(const uint4 *) (Aur + (2 * kk) * AP), *(const uint4 *) (Aur + (2 * kk + 1) * AP)});
#pragma unroll
                for (int j = 0; j < NJ; ++j) {
                    const _Float16 * p = Bsr + (2 * j * 16) * 8;
                    const mmbf_v16h b = __builtin_bit_cast(mmbf_v16h, (uint4[2]){*(const uint4 *) (p + (2 * kk) * BP), *(const uint4 *) (p + (2 * kk + 1) * BP)});
                    accg[j] = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(b, ag, accg[j]);
                    accu[j] = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(b, au, accu[j]);
                    asm volatile("" : "+v"(accg[j]), "+v"(accu[j]));   // keep the WMMAs in this K step (see mmb-routed.cuh)
                }
                asm volatile("" ::: "memory");
            }
            __syncthreads();
            if (more) { store_lds(std::integral_constant<int, (S + 1) & 3>{}); __syncthreads(); }
        };
        for (int kb = 0; kb < nblk; ++kb) {
            step(kb, std::integral_constant<int, 0>{}); step(kb, std::integral_constant<int, 1>{});
            step(kb, std::integral_constant<int, 2>{}); step(kb, std::integral_constant<int, 3>{});
        }
        if (tid == 0) *(int *) Ag = fetch();   // the gate tile is dead from here to the next item's first store
        // Epilogue: silu(g)*u. Lane holds tokens 2e+cn of each fragment at column cm; the bf16 tile is staged in Bs
        // (chunk-major like the K tiles) so every token row leaves as 16-byte chunks.
        uint16_t * Bh = (uint16_t *) Bs;
#pragma unroll
        for (int j = 0; j < NJ; ++j) {
#pragma unroll
            for (int e = 0; e < 8; ++e) {
                const int n = (wg + 2 * j) * 16 + 2 * e + cn;
                const float v = ggml_cuda_op_silu_single(accg[j][e]) * accu[j][e];
                if (Dh) Bh[(2 * wr + (cm >> 3)) * BP + n * 8 + (cm & 7)] = mmb_f2bf(v);
                if (store_f32 && col < M) { const int dr = drow(n); if (dr >= 0) D[(size_t) dr * M + col] = v; }
            }
        }
    };
    const int nlive = (n_cols + 15) / 16, nj = min(TN, max(0, (nlive - wg + 1) / 2));   // fragments wg, wg+2, ... < nlive
    switch (nj) {
        case 0: kloop(std::integral_constant<int, 0>{}); break;
        case 1: kloop(std::integral_constant<int, 1>{}); break;
        case 2: kloop(std::integral_constant<int, 2>{}); break;
        case 3: kloop(std::integral_constant<int, 3>{}); break;
        default: kloop(std::integral_constant<int, TN>{}); break;
    }
    __syncthreads();
    if (Dh) {
#pragma unroll
        for (int i = 0; i < B_ITEMS; ++i) {   // coalesced row stores: eight lanes cover one 128-byte row
            const int c = tid + i * MMB_NT, n = c >> 3, k = (c & 7) * 8, dr = drow(n);
            if (dr >= 0 && m0 + k + 8 <= M) *(uint4 *) (Dh + (size_t) dr * M + m0 + k) = *(const uint4 *) (Bs + (c & 7) * BP + n * 8);
        }
        __syncthreads();
    }
}

#ifndef MMBF_WAVES_PER_EU
#define MMBF_WAVES_PER_EU 8   // four 32 KB blocks per WGP = 8 waves per SIMD: 192 VGPRs
#endif
template <int BN, int WTYPE>
__global__ void __launch_bounds__(MMB_NT, MMBF_WAVES_PER_EU)
mmb_routed_glu_f16_kernel(const uint8_t * __restrict__ Wg, const uint8_t * __restrict__ Wu, const size_t expert_bytes, const _Float16 * __restrict__ Xh,
        float * __restrict__ D, uint16_t * __restrict__ Dh, const bool store_f32,
        const int32_t * __restrict__ ids_src, const int32_t * __restrict__ ids_dst, const int32_t * __restrict__ bounds,
        const uint32_t * __restrict__ tiles, int * __restrict__ ctl, const int M, const int K) {
    static_assert(WTYPE == 32 + GGML_TYPE_Q4_K, "Q4_K experts only");
    __shared__ __align__(16) _Float16 Ag[64 * 64];
    __shared__ __align__(16) _Float16 Au[64 * 64];
    __shared__ __align__(16) _Float16 Bs[BN * 64];
    const int mt = (M + 63) / 64, n_items = ctl[0] * mt;
    const size_t wrow_bytes = mmb_row_bytes<WTYPE>(K);
    auto fetch = [&]() { return atomicAdd(ctl + 1, 1); };
    if (threadIdx.x == 0) *(int *) Ag = fetch();
    __syncthreads();
    for (;;) {
        const int item = __builtin_amdgcn_readfirstlane(*(const int *) Ag);
        __syncthreads();   // every wave has the item before the first weight store lands on the slot
        if (item >= n_items) return;
        const uint32_t dsc = tiles[item / mt];
        const int e = dsc & 0xffff, jt = dsc >> 16, m0 = (item % mt) * 64;
        const int r0 = bounds[e] + jt * BN, cnt = bounds[e + 1] - r0;
        mmb_tile_gemm_glu_f16<BN>(Wg + (size_t) e * expert_bytes + (size_t) m0 * wrow_bytes, Wu + (size_t) e * expert_bytes + (size_t) m0 * wrow_bytes, wrow_bytes, M - m0, Xh, K,
            [&](int i) { return (i < cnt) ? ids_src[r0 + i] : -1; }, D, Dh, store_f32, M, [&](int i) { return (i < cnt) ? ids_dst[r0 + i] : -1; }, m0, cnt, Ag, Au, Bs, fetch);
    }
}
