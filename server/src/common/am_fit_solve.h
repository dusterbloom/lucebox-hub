// Attention Matching (AM) fit solver, KVFlash 2.0b milestone M5.
//
// A sealed block of T keys is compacted to t slots per KV head. Given the
// block's own prefill queries, the selected key indices (M4) and the block's
// K/V, this fits
//   * beta[slot]  - the additive logit bias that goes into the FLASH_ATTN_EXT
//                   kv_bias input (src[9], F32 [n_kv, n_head_kv]);
//   * C2[slot]    - the replacement values.
// It follows the reference algorithms/base.py (`_nnls_pg`, `_compute_C2`) and
// highest_attention_keys.py, non-causal and span-local:
//   s_qj   = scale * q . k_j                    (scale = 1/sqrt(head_dim))
//   E_qj   = exp(s_qj - max_j s_qj)             (max over ALL T keys of the block)
//   M      = E[:, sel]   y = rowsum(E)          beta = log w, w = argmin |M w - y|^2
//                                               w in [lo, hi], 2 projected-gradient steps
//   A      = softmax_sel(s[:, sel] + beta)      Y = softmax_all(s) V
//   C2     = argmin |A C2 - Y|^2
// All query heads that share a KV head are stacked as rows of that KV head's
// problem (GQA grouping, group-major, as per_layer_head_on_policy.py).
//
// The split: the backend (CPU, CUDA, HIP, anything ggml runs) only accumulates
// the Gram matrices M^T M, M^T y, A^T A, A^T Y, chunked over queries; the host
// solves the t x t systems in fp64. The solvers are pure functions of the Gram
// matrices and carry no ggml dependency.
#pragma once

#include "ggml.h"
#include "ggml-backend.h"

#include <cmath>
#include <cstdint>
#include <string>
#include <vector>

namespace luce_am {

enum class SolveStatus {
    Ok,         // plain Cholesky solve
    RidgeUsed,  // the plain solve failed or was ill-conditioned; ridge was added
    Failed,     // not finite even with the ridge; outputs are the neutral fit (beta 0)
};

struct NnlsParams {
    double lo = std::exp(-3.0);  // box on w = exp(beta), [e^-3, e^3]
    double hi = std::exp(3.0);
    int    pg_iters = 2;         // projected-gradient steps
    int    power_iters = 3;      // power iterations for L (reference: 3)
    double ridge = 1e-6;         // absolute diagonal ridge of the fallback (reference: 1e-6)
    double chol_rel_tol = 1e-12; // a Cholesky pivot below tol * diag counts as failure
};

// ---- Pure host solvers (fp64) ------------------------------------------------

// beta = log(w) for w = box-constrained NNLS of min |M w - y|^2 given
// G = M^T M (t x t, row-major, symmetric) and b = M^T y (t).
SolveStatus solve_beta(const double * G, const double * b, int t, const NnlsParams & p, double * beta_out);

// C2 = (A^T A)^-1 A^T Y. AtA is t x t, AtY and C2_out are row-major [t][d].
// Ridge fallback p.ridge on the diagonal when the plain solve fails.
SolveStatus solve_values(const double * AtA, const double * AtY, int t, int d, const NnlsParams & p, double * C2_out);

// ---- Backend Gram accumulation ------------------------------------------------

struct BlockShape {
    int   head_dim = 0;
    int   n_tokens = 0;      // queries per head in the block (640)
    int   n_head = 0;        // query heads
    int   n_head_kv = 0;     // KV heads, n_head % n_head_kv == 0
    int   n_keys = 0;        // T, keys in the block (640)
    int   t = 0;             // kept slots per head (64)
    float scale = 0.0f;      // logit scale, 1/sqrt(head_dim)
    // Queries (tokens) per backend graph. Tensor-core GEMMs accumulate with truncation, so the
    // error grows with the accumulation depth chunk_tokens * n_rep: at 256 tokens the real-shape
    // Gram error is 2e-5, at 64 it is 7e-6, at 16 it is 3e-6 (RTX 3090).
    int   chunk_tokens = 64;
    // Backends that run F32 matmuls on TF32 tensor cores (CUDA on Ampere and newer) lose ~2e-4 in the
    // Gram sums. Compensated mode splits each operand into two F16-exact halves and sums three
    // products, which restores fp32 accuracy at ~3x the matmul work. -1: on for every non-CPU
    // backend, 0: off, 1: on.
    int   compensated = -1;
};

// Plain host inputs. Layouts are ggml layouts (ne0 fastest):
//   q    [head_dim, n_tokens, n_head]  F16  post-RoPE queries (M3 output)
//   k    [head_dim, n_keys,   n_head_kv] F32
//   v    [head_dim, n_keys,   n_head_kv] F32
//   sel  [t, n_head_kv] int32 indices into [0, n_keys), M4 output
struct BlockInputs {
    const ggml_fp16_t * q = nullptr;
    const float *       k = nullptr;
    const float *       v = nullptr;
    const int32_t *     sel = nullptr;
};

// Gram sums per KV head, row-major fp64.
struct BetaGrams {
    std::vector<double> MtM;  // [n_head_kv][t*t]
    std::vector<double> Mty;  // [n_head_kv][t]
};
struct ValueGrams {
    std::vector<double> AtA;  // [n_head_kv][t*t]
    std::vector<double> AtY;  // [n_head_kv][t*head_dim], row-major [slot][dim]
};

// Holds one block's inputs on the backend and accumulates Gram matrices.
class GramAccumulator {
public:
    GramAccumulator() = default;
    ~GramAccumulator();
    GramAccumulator(const GramAccumulator &) = delete;
    GramAccumulator & operator=(const GramAccumulator &) = delete;

    // Uploads the block (q must stay valid until the Gram calls return). Reuses the device buffers and
    // graphs of the previous call when backend and shape are unchanged. False (with *err) on a bad shape or backend failure.
    bool init(ggml_backend_t backend, const BlockShape & shape, const BlockInputs & in, std::string * err);

    // M^T M and M^T y, summed over all queries (all heads of each KV group).
    bool beta_grams(BetaGrams & out, std::string * err);

    // A^T A and A^T Y with beta [t, n_head_kv] (F32, beta[g*t + slot]).
    bool value_grams(const float * beta, ValueGrams & out, std::string * err);

private:
    struct Impl;
    Impl * impl_ = nullptr;
};

// ---- Whole-block fit ------------------------------------------------------------

struct FitResult {
    std::vector<float>       beta;    // [t, n_head_kv]: beta[g*t + slot], the kv_bias block slice
    std::vector<float>       c2;      // [head_dim, t, n_head_kv]: c2[(g*t + slot)*head_dim + dim]
    std::vector<SolveStatus> status;  // per KV head, worst of the beta and value solves
    // Wall time of the phases, ms: upload + graph build, beta Grams, beta solve, value Grams, value solve.
    double ms_init = 0, ms_beta_grams = 0, ms_beta_solve = 0, ms_value_grams = 0, ms_value_solve = 0;
};

bool fit_block(ggml_backend_t backend, const BlockShape & shape, const BlockInputs & in,
               const NnlsParams & p, FitResult & out, std::string * err);

// Same, reusing `acc`: when backend and shape match the previous block, the device buffers and
// compiled graphs are kept (the engine holds one accumulator per device and calls it once per
// layer and sealed block).
bool fit_block(GramAccumulator & acc, ggml_backend_t backend, const BlockShape & shape, const BlockInputs & in,
               const NnlsParams & p, FitResult & out, std::string * err);

}  // namespace luce_am
