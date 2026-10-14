// C shim over gufo's mul_mat_vec_moe_gated (its ggml_type/block_q8_1 live in namespace qfn_mmq).
#include "qfn_mmq_prelude.h"
namespace qfn_mmq {
#include "mmvq.hpp"
}
extern "C" void gufo_gated_q4k(const void * gate, const void * up, const void * y, const int32_t * ids, float * out,
                               int k, int rows, int experts_used, int input_stride, hipStream_t stream) {
    qfn_mmq::mul_mat_vec_moe_gated(gate, up, qfn_mmq::GGML_TYPE_Q4_K, (const qfn_mmq::block_q8_1 *) y, ids, nullptr, out,
                                   k, rows, 1, experts_used, input_stride, stream);
}
