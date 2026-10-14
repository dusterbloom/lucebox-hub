// File-scope kernels for f16exp.cuh (included before main in moe2_bench.hip).
__global__ void cvt_bf16_f16_kernel(const uint16_t * __restrict__ x, _Float16 * __restrict__ y, const size_t n) {
    const size_t i = ((size_t) blockIdx.x * blockDim.x + threadIdx.x) * 8;
    if (i >= n) return;
    const uint4 v = *(const uint4 *) (x + i);
    const uint32_t w[4] = {v.x, v.y, v.z, v.w};
    uint32_t o[4];
#pragma unroll
    for (int j = 0; j < 4; ++j) {
        const float a = __uint_as_float(w[j] << 16), b = __uint_as_float(w[j] & 0xffff0000u);
        o[j] = (uint32_t) __builtin_bit_cast(uint16_t, mq_f2h_sat(a)) | ((uint32_t) __builtin_bit_cast(uint16_t, mq_f2h_sat(b)) << 16);
    }
    *(uint4 *) (y + i) = make_uint4(o[0], o[1], o[2], o[3]);
}

// Double reference of the routed gate/up SwiGLU: block = REF_COLS weight rows x REF_POS routed rows (lanes of a wave
// share the weight row and span routed rows), the 32 tokens' 256 activations of each Q4_K block staged in LDS.
constexpr int REF_COLS = 8, REF_POS = 32, REF_XS = 264;
__global__ void ref_f64_kernel(const uint8_t * __restrict__ wg, const uint8_t * __restrict__ wu, const size_t expert_bytes, const size_t row_bytes,
        const uint16_t * __restrict__ xb, const int32_t * __restrict__ ids_src, const int32_t * __restrict__ ids_dst, const int32_t * __restrict__ ids_e,
        const int n_rows, const int M, const int K, double * __restrict__ out) {
    __shared__ __align__(16) uint16_t xs[REF_POS * REF_XS];
    const int tid = threadIdx.x, pl = tid & 31, cl = tid >> 5;
    const int pos = blockIdx.y * REF_POS + pl, col = blockIdx.x * REF_COLS + cl;
    const bool live = pos < n_rows;
    const int e = live ? ids_e[pos] : 0;
    const uint8_t * rg = wg + (size_t) e * expert_bytes + (size_t) col * row_bytes, * ru = wu + (size_t) e * expert_bytes + (size_t) col * row_bytes;
    double g = 0, u = 0;
    for (int b = 0; b < K / 256; ++b) {
        __syncthreads();
        for (int i = tid; i < REF_POS * 32; i += REF_COLS * REF_POS) {
            const int p = i / 32, c = i % 32, pp = blockIdx.y * REF_POS + p;
            const int t = pp < n_rows ? ids_src[pp] : 0;
            *(uint4 *) (xs + p * REF_XS + c * 8) = *(const uint4 *) (xb + (size_t) t * K + b * 256 + c * 8);
        }
        __syncthreads();
        auto blk = [&](const uint8_t * r, double & acc) {
            const uint8_t * p = r + (size_t) b * 144;
            const double d = (double) (float) __builtin_bit_cast(_Float16, (uint16_t) (p[0] | (p[1] << 8)));
            const double dm = (double) (float) __builtin_bit_cast(_Float16, (uint16_t) (p[2] | (p[3] << 8)));
            const uint8_t * sc = p + 4, * qs = p + 16;
            for (int j = 0; j < 8; ++j) {
                uint32_t s, m;
                if (j < 4) { s = sc[j] & 63; m = sc[j + 4] & 63; } else { s = (sc[j + 4] & 15) | ((sc[j - 4] >> 6) << 4); m = (sc[j + 4] >> 4) | ((sc[j] >> 6) << 4); }
                const double ds = d * (double) s, ms = dm * (double) m;
                for (int kk = 0; kk < 32; ++kk) {
                    const int k = j * 32 + kk, slice = k / 64, w = k % 64;
                    const uint8_t q = qs[32 * slice + (w % 32)];
                    const int v = w < 32 ? (q & 15) : (q >> 4);
                    const double x = (double) __uint_as_float((uint32_t) xs[pl * REF_XS + k] << 16);
                    acc += ((double) v * ds - ms) * x;
                }
            }
        };
        blk(rg, g); blk(ru, u);
    }
    if (live) out[(size_t) ids_dst[pos] * M + col] = g / (1.0 + exp(-g)) * u;
}
