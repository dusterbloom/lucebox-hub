// C shim over gufo's HcCombine / HcCombineMoeF16 (MIT), the prefill hyper-connection combine launchers its
// executor calls (inject_parts = 1, streams = 4, gate_stride = used + 1 as in the executor).
#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include <cstdint>

namespace gufo::models::qwen38_flash_next::rocm {
void HcCombine(float * res, const float * block_out, const float * inject, std::uint32_t inject_parts, const float * gamma,
               float * xn, std::uint32_t n_tokens, std::uint32_t hidden, std::uint32_t streams, float eps, hipStream_t stream,
               bool decode);
bool HcCombineMoeF16(void * res, const __half * expert_out, const float * weights, const float * shared_out, const float * gate,
                     std::uint32_t gate_stride, std::uint32_t used, const float * inject, std::uint32_t inject_parts,
                     const float * gamma, __half * xn, void * xn_q8, std::uint32_t n_tokens, std::uint32_t hidden,
                     std::uint32_t streams, float eps, hipStream_t stream, bool res_half);
}

extern "C" void gufo_hc_combine(float * res, const float * block_out, const float * inject, const float * gamma, float * xn,
                                uint32_t n_tokens, uint32_t hidden, float eps, hipStream_t st) {
    gufo::models::qwen38_flash_next::rocm::HcCombine(res, block_out, inject, 1, gamma, xn, n_tokens, hidden, 4, eps, st, false);
}
extern "C" int gufo_hc_combine_moe(void * res, const uint16_t * expert_out, const float * weights, const float * shared_out,
                                   const float * gate, uint32_t used, const float * inject, const float * gamma, uint16_t * xn,
                                   void * xn_q8, uint32_t n_tokens, uint32_t hidden, float eps, hipStream_t st, int res_half) {
    return gufo::models::qwen38_flash_next::rocm::HcCombineMoeF16(res, (const __half *) expert_out, weights, shared_out, gate, 1, used,
        inject, 1, gamma, (__half *) xn, xn_q8, n_tokens, hidden, 4, eps, st, res_half != 0) ? 1 : 0;
}
