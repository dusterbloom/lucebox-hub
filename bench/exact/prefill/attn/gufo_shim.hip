// C shim over gufo's WmmaCausalAttention (MIT), the prefill attention launcher its executor calls:
// sparse <4,16,true> when a mask is given, dense <16,16,false> otherwise. gate = nullptr (no output gate).
#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include <cstdint>

namespace gufo::models::qwen38_flash_next::rocm {
bool WmmaCausalAttention(const float * q, const float * gate, const __half * k_cache, const __half * v_cache,
                         const std::uint32_t * mask, std::uint32_t mask_words, float * out, std::uint32_t n_tokens,
                         std::uint32_t start_pos, std::uint32_t heads, std::uint32_t kv_heads, std::uint32_t d,
                         std::uint32_t ratio, hipStream_t stream, bool last_only);
}

extern "C" int gufo_wmma_attention(const float * q, const uint16_t * k, const uint16_t * v, const uint32_t * mask,
                                   uint32_t mask_words, float * out, uint32_t n_tokens, uint32_t start_pos, hipStream_t st) {
    return gufo::models::qwen38_flash_next::rocm::WmmaCausalAttention(
        q, nullptr, (const __half *) k, (const __half *) v, mask, mask_words, out, n_tokens, start_pos, 24, 2, 256, 4, st, false) ? 1 : 0;
}
