#include "common.cuh"

#define MMVQ_MAX_BATCH_SIZE 8 // Max. batch size for which to use MMVQ kernels.
// Max. batch size for MUL_MAT_ID via the dedicated multi-token MoE kernel
// (mul_mat_vec_q_moe takes ncols_dst at runtime). Staying on this path also
// keeps CUDA-graph capture enabled ([TAG_MUL_MAT_ID_CUDA_GRAPHS]); measured
// on sm_86 the fallback costs ~9ms/step for 9-16 token spec-verify batches.
#define MMVQ_MAX_MOE_BATCH_SIZE 16

// Returns the maximum batch size for which MMVQ should be used for MUL_MAT_ID,
// based on the quantization type and GPU architecture (compute capability).
// Batch-invariant mode also admits tokenwise MMID up to MMVQ_MAX_BATCH_SIZE.
// A zero ceiling (unsupported type) is never raised.
int get_mmvq_mmid_max_batch(ggml_type type, int cc);

bool ggml_cuda_mmvq_mmid_grouped_enabled(
    ggml_type type, int cc, int64_t ncols_dst, int64_t routed_pairs);

// Set by ggml_backend_cuda_set_mmvq_batch_invariant (calling thread).
bool ggml_cuda_mmvq_batch_invariant();

// Set by ggml_backend_cuda_set_mmvq_invariant_rows_override (calling thread): 0 = env / policy.
int ggml_cuda_mmvq_invariant_rows_override();
int ggml_cuda_mmvq_invariant_lds_override();

void ggml_cuda_mul_mat_vec_q(ggml_backend_cuda_context & ctx,
    const ggml_tensor * src0, const ggml_tensor * src1, const ggml_tensor * ids, ggml_tensor * dst, const ggml_cuda_mm_fusion_args_host * fusion = nullptr);

void ggml_cuda_op_mul_mat_vec_q(
    ggml_backend_cuda_context & ctx,
    const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst, const char * src0_dd_i, const float * src1_ddf_i,
    const char * src1_ddq_i, float * dst_dd_i, const int64_t row_low, const int64_t row_high, const int64_t src1_ncols,
    const int64_t src1_padded_row_size, cudaStream_t stream);

// Private graph probe: scoped stream-1 scratch, never a cross-stream memo.
void ggml_cuda_mmvq_fixed_q8_for_capture(char * ptr, size_t bytes, bool prequantized = false);

// Dense trunk weight types the qwen4exp T=1 producer / memo handoffs accept: the GGUF's Q8_0 or Q6_K and the
// decode-only requantized copies (qwen4exp_loader.cpp, LUCE_DECODE_REQ). The handoffs carry q8_1 activations, which
// every mmvq weight type consumes; the weight type only selects the vec_dot.
static inline bool ggml_cuda_dense_decode_weight_ok(ggml_type t) {
    return t == GGML_TYPE_Q8_0 || t == GGML_TYPE_Q6_K || t == GGML_TYPE_Q5_K || t == GGML_TYPE_Q4_K ||
           t == GGML_TYPE_Q5_1 || t == GGML_TYPE_Q5_0 || t == GGML_TYPE_Q4_1 || t == GGML_TYPE_Q4_0;
}
