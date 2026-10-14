#pragma once
// Routed (MUL_MAT_ID) gate+up SwiGLU GEMM for gfx1151: the fused BF16 WMMA tile, its kernels and the per-expert
// tile schedule. Included by mmb.cu inside its anonymous namespace (needs the MMB_* constants, mmb_pack2/mmb_h2f,
// mq_f2h_sat, ggml_cuda_op_silu_single and the mmb-quant.cuh dequant helpers defined before it); the same header
// builds standalone in the moe microbench and the LUCE_TESTS parity test.

template <bool DH_F16 = false, typename DRowFn>
__device__ __forceinline__ void mmb_store_tile(const v8f & acc, float * __restrict__ stg, float * __restrict__ D, uint16_t * __restrict__ Dh,
        const bool store_f32, const int M, DRowFn drow, const int n_base, const int m_base, const int lane) {
    const int cm = lane & 15, cn = lane >> 4;
#pragma unroll
    for (int e = 0; e < 8; ++e) { stg[(2 * e + cn) * 16 + cm] = acc[e]; }
    __syncthreads();
    const int n = lane >> 1, half = lane & 1;
    const int dr = drow(n_base + n);
    if (dr >= 0) {
        const float4 v0 = *(const float4 *)(stg + n * 16 + half * 8), v1 = *(const float4 *)(stg + n * 16 + half * 8 + 4);
        const size_t base = (size_t)dr * M + m_base + half * 8;
        const bool full = m_base + 16 <= M;
        if (store_f32) {
            if (full && (M & 3) == 0) { *(float4 *)(D + base) = v0; *(float4 *)(D + base + 4) = v1; }
            else { const float vv[8] = {v0.x, v0.y, v0.z, v0.w, v1.x, v1.y, v1.z, v1.w};
#pragma unroll
                   for (int k = 0; k < 8; ++k) { if (m_base + half * 8 + k < M) D[base + k] = vv[k]; } }
        }
        if (Dh) {
            // DH_F16: the half copy is F16 (saturating, as the Q8F16 activations) instead of bf16.
            auto pk = [](const float a, const float b) -> uint32_t {
                if constexpr (DH_F16) return (uint32_t) __builtin_bit_cast(uint16_t, mq_f2h_sat(a)) | ((uint32_t) __builtin_bit_cast(uint16_t, mq_f2h_sat(b)) << 16);
                else return mmb_pack2(a, b);
            };
            if (full && (M & 7) == 0) { *(uint4 *)(Dh + base) = make_uint4(pk(v0.x, v0.y), pk(v0.z, v0.w), pk(v1.x, v1.y), pk(v1.z, v1.w)); }
            else { const float vv[8] = {v0.x, v0.y, v0.z, v0.w, v1.x, v1.y, v1.z, v1.w};
#pragma unroll
                   for (int k = 0; k < 8; ++k) { if (m_base + half * 8 + k < M) Dh[base + k] = (uint16_t) (pk(vv[k], 0.0f) & 0xffffu); } }
        }
    }
    __syncthreads();
}

template <int BM, int BN, int WTM, int WTN, int WTYPE, bool TAIL, typename XRowFn, typename DRowFn>
__device__ __forceinline__ void mmb_tile_gemm_glu(const uint8_t * __restrict__ Wg, const uint8_t * __restrict__ Wu, const size_t wrow_bytes, const int a_rows,
        const uint16_t * __restrict__ Xh, const int K, XRowFn xrow, float * __restrict__ D, uint16_t * __restrict__ Dh, const bool store_f32, const int M, DRowFn drow, const int m0,
        const int n_cols, uint16_t * Ag, uint16_t * Au, uint16_t * Bs) {
    constexpr int WAVES_M = BM / WTM, TM = WTM / 16, TN = WTN / 16;
    constexpr int A_ITEMS = (BM + MMB_NT - 1) / MMB_NT;
    constexpr int B_ITEMS = (BN * 8) / MMB_NT;
    const int tid = threadIdx.x, lane = tid & 31, wave = tid >> 5;
    const int wm = wave % WAVES_M, wn = wave / WAVES_M;
    uint4 g0[A_ITEMS], g1[A_ITEMS], u0[A_ITEMS], u1[A_ITEMS]; uint32_t g2[A_ITEMS], u2[A_ITEMS];
    uint4 gm[A_ITEMS], um[A_ITEMS];
    uint4 bst[B_ITEMS];
    int brow[B_ITEMS];
#pragma unroll
    for (int i = 0; i < B_ITEMS; ++i) { const int c = tid + i * MMB_NT; brow[i] = xrow(c >> 3); }
    int weight_ks = 0;
    // Q4_K: two threads per gate/up row, each owning 16 packed bytes of every 64-value slice. Keep the full
    // compressed block in registers across four K steps: 128 payload + 2*16 header bytes instead of 4*2*(32+16).
    // BF16 tiles and the per-output WMMA K order stay unchanged; no expanded weights go through DRAM.
    constexpr bool Q4K_SPLIT = WTYPE == 32 + GGML_TYPE_Q4_K && 4 * BM == MMB_NT;
    const int s_mat = tid / (2 * BM), s_row = (tid >> 1) % BM, s_half = tid & 1;
    uint4 sq0, sq1, sq2, sq3, sqm;
    auto load_regs = [&](const int ks) {
        weight_ks = ks;
        if constexpr (Q4K_SPLIT) {
            if ((ks & 3) == 0) {
                if (s_row < a_rows) {
                    const uint8_t * p = (s_mat ? Wu : Wg) + (size_t)s_row * wrow_bytes + (size_t)(ks / 4) * sizeof(block_q4_K);
                    sqm = *(const uint4 *)p;
                    p += 16 + s_half * 16;
                    sq0 = *(const uint4 *)(p);      sq1 = *(const uint4 *)(p + 32);
                    sq2 = *(const uint4 *)(p + 64); sq3 = *(const uint4 *)(p + 96);
                } else { sqm = sq0 = sq1 = sq2 = sq3 = make_uint4(0,0,0,0); }
            }
        } else
#pragma unroll
        for (int i = 0; i < A_ITEMS; ++i) {
            const int row = tid + i * MMB_NT;
            if constexpr (WTYPE == 0) {
            if (row < BM && row < a_rows) {
                const uint8_t * pg = Wg + (size_t)row * wrow_bytes + (size_t)ks * 36;
                const uint8_t * pu = Wu + (size_t)row * wrow_bytes + (size_t)ks * 36;
                g0[i] = *(const uint4 *)(pg); g1[i] = *(const uint4 *)(pg + 16); g2[i] = *(const uint32_t *)(pg + 32);
                u0[i] = *(const uint4 *)(pu); u1[i] = *(const uint4 *)(pu + 16); u2[i] = *(const uint32_t *)(pu + 32);
            } else { g0[i] = g1[i] = u0[i] = u1[i] = make_uint4(0,0,0,0); g2[i] = u2[i] = 0; }
            } else if constexpr (WTYPE == 32 + GGML_TYPE_Q4_K) {
                if (row < BM && row < a_rows) {
                    const uint8_t * pg = Wg + (size_t)row * wrow_bytes + (size_t)(ks / 4) * sizeof(block_q4_K);
                    const uint8_t * pu = Wu + (size_t)row * wrow_bytes + (size_t)(ks / 4) * sizeof(block_q4_K);
                    gm[i] = *(const uint4 *)pg; um[i] = *(const uint4 *)pu;
                    const int offset = 16 + (ks & 3) * 32;
                    g0[i] = *(const uint4 *)(pg + offset); g1[i] = *(const uint4 *)(pg + offset + 16);
                    u0[i] = *(const uint4 *)(pu + offset); u1[i] = *(const uint4 *)(pu + offset + 16);
                } else { gm[i] = um[i] = g0[i] = g1[i] = u0[i] = u1[i] = make_uint4(0,0,0,0); }
            }
        }
#pragma unroll
        for (int i = 0; i < B_ITEMS; ++i) {
            const int c = tid + i * MMB_NT; const int off = (c & 7) * 8;
            bst[i] = (brow[i] >= 0) ? *(const uint4 *)(Xh + (size_t)brow[i] * K + ks * MMB_BK + off) : make_uint4(0,0,0,0);
        }
    };
    auto store_lds = [&]() {
        if constexpr (WTYPE != 0 && WTYPE != 32 + GGML_TYPE_Q4_K) {
            constexpr int LOAD_TYPE = WTYPE == 1 ? 32 + GGML_TYPE_Q8_0 : WTYPE;
            mmb_load_quant_tile<LOAD_TYPE, BM, MMB_LDS_STRIDE>(Wg, wrow_bytes, a_rows, weight_ks, Ag);
            mmb_load_quant_tile<LOAD_TYPE, BM, MMB_LDS_STRIDE>(Wu, wrow_bytes, a_rows, weight_ks, Au);
        }

        if constexpr (Q4K_SPLIT) {
            const int slice = weight_ks & 3;
            const uint4 q = slice == 0 ? sq0 : slice == 1 ? sq1 : slice == 2 ? sq2 : sq3;
            mmb_dq_q4k_stripe(q, sqm, slice, (uint32_t *)((s_mat ? Au : Ag) + s_row * MMB_LDS_STRIDE) + 8 * s_half);
        } else
#pragma unroll
        for (int i = 0; i < A_ITEMS; ++i) {
            const int row = tid + i * MMB_NT;
            if (row < BM) {
                if constexpr (WTYPE == 0) {
                    mmb_dq_row36(g0[i], g1[i], g2[i], (uint32_t *)(Ag + row * MMB_LDS_STRIDE));
                    mmb_dq_row36(u0[i], u1[i], u2[i], (uint32_t *)(Au + row * MMB_LDS_STRIDE));
                } else if constexpr (WTYPE == 32 + GGML_TYPE_Q4_K) {
                    mmb_dq_q4k_slice(g0[i], g1[i], gm[i], weight_ks & 3, (uint32_t *)(Ag + row * MMB_LDS_STRIDE));
                    mmb_dq_q4k_slice(u0[i], u1[i], um[i], weight_ks & 3, (uint32_t *)(Au + row * MMB_LDS_STRIDE));
                }
            }
        }
#pragma unroll
        for (int i = 0; i < B_ITEMS; ++i) { const int c = tid + i * MMB_NT; *(uint4 *)(Bs + (c >> 3) * MMB_LDS_STRIDE + (c & 7) * 8) = bst[i]; }
    };
    v8f accg[TM][TN], accu[TM][TN];
#pragma unroll
    for (int i = 0; i < TM; ++i)
#pragma unroll
        for (int j = 0; j < TN; ++j)
#pragma unroll
            for (int e = 0; e < 8; ++e) { accg[i][j][e] = 0.f; accu[i][j][e] = 0.f; }
    bool jact[TN];
#pragma unroll
    for (int j = 0; j < TN; ++j) jact[j] = !TAIL || (wn * WTN + j * 16) < n_cols;
    const int nks = K / MMB_BK;
    load_regs(0); store_lds(); __syncthreads();
    for (int ks = 0; ks < nks; ++ks) {
        if (ks + 1 < nks) load_regs(ks + 1);
#pragma unroll
        for (int kk = 0; kk < MMB_BK; kk += 16) {
            v16s ag[TM], au[TM], b[TN]; const int r = lane & 15;
#pragma unroll
            for (int i = 0; i < TM; ++i) { const int off = (wm * WTM + i * 16 + r) * MMB_LDS_STRIDE + kk;
                ag[i] = __builtin_bit_cast(v16s, (uint4[2]){*(const uint4 *)(Ag + off), *(const uint4 *)(Ag + off + 8)});
                au[i] = __builtin_bit_cast(v16s, (uint4[2]){*(const uint4 *)(Au + off), *(const uint4 *)(Au + off + 8)}); }
#pragma unroll
            for (int j = 0; j < TN; ++j) { if constexpr (TAIL) { if (!jact[j]) continue; } const uint16_t * p = Bs + (wn * WTN + j * 16 + r) * MMB_LDS_STRIDE + kk;
                b[j] = __builtin_bit_cast(v16s, (uint4[2]){*(const uint4 *)p, *(const uint4 *)(p + 8)}); }
#pragma unroll
            for (int i = 0; i < TM; ++i)
#pragma unroll
                for (int j = 0; j < TN; ++j) {
                    if constexpr (TAIL) { if (!jact[j]) continue; }
                    accg[i][j] = __builtin_amdgcn_wmma_f32_16x16x16_bf16_w32(b[j], ag[i], accg[i][j]);
                    accu[i][j] = __builtin_amdgcn_wmma_f32_16x16x16_bf16_w32(b[j], au[i], accu[i][j]);
                }
        }
        __syncthreads();
        if (ks + 1 < nks) store_lds();
        __syncthreads();
    }
    float * stg = (float *)((BN * MMB_LDS_STRIDE * 2 >= MMB_NT / 32 * 1024) ? Bs : Ag) + wave * 256;
#pragma unroll
    for (int i = 0; i < TM; ++i) {
        const int ml = wm * WTM + i * 16; const bool ok = ml < a_rows;
#pragma unroll
        for (int j = 0; j < TN; ++j) {
            if (ok && jact[j]) {
                v8f v;
#pragma unroll
                for (int e = 0; e < 8; ++e) { v[e] = ggml_cuda_op_silu_single(accg[i][j][e]) * accu[i][j][e]; }
                mmb_store_tile(v, stg, D, Dh, store_f32, M, drow, wn * WTN + j * 16, m0 + ml, lane);
            } else { __syncthreads(); __syncthreads(); }
        }
    }
}

template <int BM, int BN, int WTM, int WTN, int WTYPE = 0, bool EXPERT_MAJOR = false>
__global__ void __launch_bounds__(MMB_NT, 2)
mmb_routed_glu_kernel(const uint8_t * __restrict__ Wg, const uint8_t * __restrict__ Wu, const size_t expert_bytes, const uint16_t * __restrict__ Xh,
        float * __restrict__ D, uint16_t * __restrict__ Dh, const bool store_f32,
        const int32_t * __restrict__ ids_src, const int32_t * __restrict__ ids_dst, const int32_t * __restrict__ bounds,
        const uint32_t * __restrict__ desc, const int M, const int K) {
    __shared__ __align__(16) uint16_t Ag[BM * MMB_LDS_STRIDE];
    __shared__ __align__(16) uint16_t Au[BM * MMB_LDS_STRIDE];
    __shared__ __align__(16) uint16_t Bs[BN * MMB_LDS_STRIDE];
    const uint32_t dsc = desc[blockIdx.y];
    if (dsc == UINT32_MAX) return;
    const int e = dsc & 0xffff, jt = dsc >> 16;
    const int r0 = bounds[e] + jt * BN, cnt = bounds[e + 1] - r0;
    const int m0 = blockIdx.x * BM;
    const size_t wrow_bytes = mmb_row_bytes<WTYPE>(K);
    mmb_tile_gemm_glu<BM, BN, WTM, WTN, WTYPE, true>(Wg + (size_t)e * expert_bytes + (size_t)m0 * wrow_bytes, Wu + (size_t)e * expert_bytes + (size_t)m0 * wrow_bytes, wrow_bytes, M - m0, Xh, K,
        [&](int i) { return (i < cnt) ? ids_src[r0 + i] : -1; }, D, Dh, store_f32, M, [&](int i) { return (i < cnt) ? (EXPERT_MAJOR ? r0 + i : ids_dst[r0 + i]) : -1; }, m0, cnt, Ag, Au, Bs);
}

// two tile classes: experts with >= thresh rows get BN_BIG-row tiles, the rest BN_SMALL-row tiles (fewer wasted rows on tiny experts)
__global__ void mmb_build_desc2(const int32_t * __restrict__ bounds, uint32_t * __restrict__ desc_big, uint32_t * __restrict__ desc_small,
        const int E, const int nbig_max, const int nsmall_max, const int BN_BIG, const int BN_SMALL, const int thresh) {
    __shared__ int sb[1024], ss[1024];
    const int e = threadIdx.x;
    for (int i = e; i < nbig_max;   i += blockDim.x) desc_big[i]   = UINT32_MAX;
    for (int i = e; i < nsmall_max; i += blockDim.x) desc_small[i] = UINT32_MAX;
    int cnt = (e < E) ? bounds[e + 1] - bounds[e] : 0;
    const bool big = cnt >= thresh;
    const int tb = big ? (cnt + BN_BIG - 1) / BN_BIG : 0;
    const int ts = big ? 0 : (cnt + BN_SMALL - 1) / BN_SMALL;
    sb[e] = tb; ss[e] = ts;
    __syncthreads();
    for (int off = 1; off < 1024; off <<= 1) {
        const int vb = (e >= off) ? sb[e - off] : 0, vs = (e >= off) ? ss[e - off] : 0;
        __syncthreads();
        sb[e] += vb; ss[e] += vs;
        __syncthreads();
    }
    const int bb = sb[e] - tb, bs = ss[e] - ts;
    for (int jt = 0; jt < tb; ++jt) { const int idx = bb + jt; if (idx < nbig_max)   desc_big[idx]   = (uint32_t)e | ((uint32_t)jt << 16); }
    for (int jt = 0; jt < ts; ++jt) { const int idx = bs + jt; if (idx < nsmall_max) desc_small[idx] = (uint32_t)e | ((uint32_t)jt << 16); }
}

// ---------------------------------------------------------------------------------------------------------------
// Single-launch Q4_K gate/up (LUCE_MMB_GLU_LEGACY=1 restores the two-launch kernels above).
// Block = 64 weight rows of gate AND up x BN tokens, 8 waves. Wave w owns rows 16*(w&3)..+16 of both matrices and
// the token fragments j = (w>>2), (w>>2)+2, ... so a tile with L live 16-token fragments costs ceil(L/2)/4 of a full
// one on every wave; the 2x4 layout idled whole waves on partial tiles, which is why 65..127-row experts needed a
// second 64-token kernel. Same BF16 weights (mmb_dq_q4k_stripe), same ascending-K WMMA order per output -> bit-exact.
// Tiles come from one list (expert order, BN-token tiles for every expert) and resident blocks pull work items
// (tile, 64-row m slice; m fastest so the 10 m slices of a token tile share the activations in L2) with an atomic
// counter: no empty grid rows, no second launch, no host sync.
// ponytail: Q4_K only (the qwen4exp expert type); other weight types stay on the legacy path.
template <int BN, typename XRowFn, typename DRowFn>
__device__ __forceinline__ void mmb_tile_gemm_glu_q4k(const uint8_t * __restrict__ Wg, const uint8_t * __restrict__ Wu, const size_t wrow_bytes, const int a_rows,
        const uint16_t * __restrict__ Xh, const int K, XRowFn xrow, float * __restrict__ D, uint16_t * __restrict__ Dh, const bool store_f32, const int M, DRowFn drow, const int m0,
        const int n_cols, uint16_t * Ag, uint16_t * Au, uint16_t * Bs) {
    constexpr int BM = 64, TN = BN / 32, B_ITEMS = (BN * 8) / MMB_NT;
    static_assert(4 * BM == MMB_NT && BN % 32 == 0, "two threads per gate/up row, two token-fragment groups");
    const int tid = threadIdx.x, lane = tid & 31, wave = __builtin_amdgcn_readfirstlane(tid >> 5);
    const int wr = wave & 3, wg = wave >> 2;   // weight rows 16*wr..+16 of both matrices; token fragments wg, wg+2, ...
    uint4 bst[B_ITEMS];
    int brow[B_ITEMS];
#pragma unroll
    for (int i = 0; i < B_ITEMS; ++i) { const int c = tid + i * MMB_NT; brow[i] = xrow(c >> 3); }
    int weight_ks = 0;
    const int s_mat = tid / (2 * BM), s_row = (tid >> 1) % BM, s_half = tid & 1;
    uint4 sq0, sq1, sq2, sq3, sqm;
    auto load_regs = [&](const int ks) {
        weight_ks = ks;
        if ((ks & 3) == 0) {
            if (s_row < a_rows) {
                const uint8_t * p = (s_mat ? Wu : Wg) + (size_t)s_row * wrow_bytes + (size_t)(ks / 4) * sizeof(block_q4_K);
                sqm = *(const uint4 *)p;
                p += 16 + s_half * 16;
                sq0 = *(const uint4 *)(p);      sq1 = *(const uint4 *)(p + 32);
                sq2 = *(const uint4 *)(p + 64); sq3 = *(const uint4 *)(p + 96);
            } else { sqm = sq0 = sq1 = sq2 = sq3 = make_uint4(0,0,0,0); }
        }
#pragma unroll
        for (int i = 0; i < B_ITEMS; ++i) {
            const int c = tid + i * MMB_NT; const int off = (c & 7) * 8;
            bst[i] = (brow[i] >= 0) ? *(const uint4 *)(Xh + (size_t)brow[i] * K + ks * MMB_BK + off) : make_uint4(0,0,0,0);
        }
    };
    auto store_lds = [&]() {
        const int slice = weight_ks & 3;
        const uint4 q = slice == 0 ? sq0 : slice == 1 ? sq1 : slice == 2 ? sq2 : sq3;
        mmb_dq_q4k_stripe(q, sqm, slice, (uint32_t *)((s_mat ? Au : Ag) + s_row * MMB_LDS_STRIDE) + 8 * s_half);
#pragma unroll
        for (int i = 0; i < B_ITEMS; ++i) { const int c = tid + i * MMB_NT; *(uint4 *)(Bs + (c >> 3) * MMB_LDS_STRIDE + (c & 7) * 8) = bst[i]; }
    };
    // The K loop and epilogue are specialized on this wave's live fragment count NJ (uniform: wg is readfirstlane'd
    // and the tile comes from an SGPR work item). Guarding each fragment with `if (jact[j])` instead left the WMMAs
    // in 16 conditional blocks per K step and the accumulators were copied at every merge (256 VGPRs, 250 B/lane
    // of spills); straight-line loops per NJ coalesce.
    const int nks = K / MMB_BK;
    const int cm = lane & 15, cn = lane >> 4, col = m0 + wr * 16 + cm;
    auto kloop = [&](auto njc) __attribute__((always_inline)) {
        constexpr int NJ = decltype(njc)::value;
        v8f accg[NJ > 0 ? NJ : 1], accu[NJ > 0 ? NJ : 1];
#pragma unroll
        for (int j = 0; j < NJ; ++j)
#pragma unroll
            for (int e = 0; e < 8; ++e) { accg[j][e] = 0.f; accu[j][e] = 0.f; }
        load_regs(0); store_lds(); __syncthreads();
        for (int ks = 0; ks < nks; ++ks) {
            if (ks + 1 < nks) load_regs(ks + 1);
#pragma unroll
            for (int kk = 0; kk < MMB_BK; kk += 16) {
                const int r = lane & 15, off = (wr * 16 + r) * MMB_LDS_STRIDE + kk;
                const v16s ag = __builtin_bit_cast(v16s, (uint4[2]){*(const uint4 *)(Ag + off), *(const uint4 *)(Ag + off + 8)});
                const v16s au = __builtin_bit_cast(v16s, (uint4[2]){*(const uint4 *)(Au + off), *(const uint4 *)(Au + off + 8)});
#pragma unroll
                for (int j = 0; j < NJ; ++j) {
                    const uint16_t * p = Bs + ((wg + 2 * j) * 16 + r) * MMB_LDS_STRIDE + kk;
                    const v16s b = __builtin_bit_cast(v16s, (uint4[2]){*(const uint4 *)p, *(const uint4 *)(p + 8)});
                    accg[j] = __builtin_amdgcn_wmma_f32_16x16x16_bf16_w32(b, ag, accg[j]);
                    accu[j] = __builtin_amdgcn_wmma_f32_16x16x16_bf16_w32(b, au, accu[j]);
                    // pin the WMMAs to this K step: LLVM sank the (pure) WMMA chain to the end of the loop body
                    // and hoisted all 24 fragment loads of the 64-wide step above it, 192 VGPRs on top of the
                    // accumulators, 400 B/lane of spills. Volatile asms keep program order among themselves.
                    asm volatile("" : "+v"(accg[j]), "+v"(accu[j]));
                }
                asm volatile("" ::: "memory");
            }
            __syncthreads();
            if (ks + 1 < nks) store_lds();
            __syncthreads();
        }
        // Epilogue: silu(g)*u. Lane holds tokens 2e+cn of each fragment at column cm. The bf16 tile is staged in Bs
        // so every token row leaves as one 128-byte line; the F32 copy (parity tests only) goes straight from registers.
#pragma unroll
        for (int j = 0; j < NJ; ++j) {
#pragma unroll
            for (int e = 0; e < 8; ++e) {
                const int n = (wg + 2 * j) * 16 + 2 * e + cn;
                const float v = ggml_cuda_op_silu_single(accg[j][e]) * accu[j][e];
                if (Dh) Bs[n * MMB_LDS_STRIDE + wr * 16 + cm] = mmb_f2bf(v);
                if (store_f32 && col < M) { const int dr = drow(n); if (dr >= 0) D[(size_t)dr * M + col] = v; }
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
    if (Dh) {
        __syncthreads();
#pragma unroll
        for (int i = 0; i < B_ITEMS; ++i) {
            const int c = tid + i * MMB_NT, n = c >> 3, k = (c & 7) * 8, dr = drow(n);
            if (dr >= 0 && m0 + k + 8 <= M) *(uint4 *)(Dh + (size_t)dr * M + m0 + k) = *(const uint4 *)(Bs + n * MMB_LDS_STRIDE + k);
        }
    }
    __syncthreads();
}

template <int BN, int WTYPE>
__global__ void __launch_bounds__(MMB_NT, 2)
mmb_routed_glu_q4k_kernel(const uint8_t * __restrict__ Wg, const uint8_t * __restrict__ Wu, const size_t expert_bytes, const uint16_t * __restrict__ Xh,
        float * __restrict__ D, uint16_t * __restrict__ Dh, const bool store_f32,
        const int32_t * __restrict__ ids_src, const int32_t * __restrict__ ids_dst, const int32_t * __restrict__ bounds,
        const uint32_t * __restrict__ tiles, int * __restrict__ ctl, const int M, const int K) {
    __shared__ __align__(16) uint16_t Ag[64 * MMB_LDS_STRIDE];
    __shared__ __align__(16) uint16_t Au[64 * MMB_LDS_STRIDE];
    __shared__ __align__(16) uint16_t Bs[BN * MMB_LDS_STRIDE];
    __shared__ int s_item;
    const int mt = (M + 63) / 64, n_items = ctl[0] * mt;
    const size_t wrow_bytes = mmb_row_bytes<WTYPE>(K);
    if (threadIdx.x == 0) s_item = atomicAdd(ctl + 1, 1);
    __syncthreads();
    for (;;) {
        // readfirstlane: an LDS broadcast is a VGPR value, so without it the tile (e, jt, cnt) and every jact branch
        // around the WMMAs became divergent -- the compiler masked each WMMA through copies and spilled 500 B/lane.
        const int item = __builtin_amdgcn_readfirstlane(s_item);
        if (item >= n_items) return;
        const uint32_t dsc = tiles[item / mt];
        const int e = dsc & 0xffff, jt = dsc >> 16, m0 = (item % mt) * 64;
        const int r0 = bounds[e] + jt * BN, cnt = bounds[e + 1] - r0;
        mmb_tile_gemm_glu_q4k<BN>(Wg + (size_t)e * expert_bytes + (size_t)m0 * wrow_bytes, Wu + (size_t)e * expert_bytes + (size_t)m0 * wrow_bytes, wrow_bytes, M - m0, Xh, K,
            [&](int i) { return (i < cnt) ? ids_src[r0 + i] : -1; }, D, Dh, store_f32, M, [&](int i) { return (i < cnt) ? ids_dst[r0 + i] : -1; }, m0, cnt, Ag, Au, Bs);
        if (threadIdx.x == 0) s_item = atomicAdd(ctl + 1, 1);
        __syncthreads();
    }
}

// tiles[] = (e | jt<<16) for every BN-token tile of every expert, in expert order; ctl = {n_tiles, 0 (work counter)}.
__global__ void mmb_build_tiles(const int32_t * __restrict__ bounds, uint32_t * __restrict__ tiles, int * __restrict__ ctl, const int E, const int n_max, const int BN) {
    __shared__ int s[1024];
    const int e = threadIdx.x;
    const int t = (e < E) ? (bounds[e + 1] - bounds[e] + BN - 1) / BN : 0;
    s[e] = t;
    __syncthreads();
    for (int off = 1; off < 1024; off <<= 1) {
        const int v = (e >= off) ? s[e - off] : 0;
        __syncthreads();
        s[e] += v;
        __syncthreads();
    }
    const int base = s[e] - t;
    for (int jt = 0; jt < t; ++jt) { if (base + jt < n_max) tiles[base + jt] = (uint32_t)e | ((uint32_t)jt << 16); }
    if (e == 1023) { ctl[0] = min(s[e], n_max); ctl[1] = 0; }
}
