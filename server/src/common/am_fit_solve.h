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
//   M      = E[:, sel]   y = rowsum(E)          w = lstsq(M, y) (QR), clamp to [lo, hi],
//                                               2 projected-gradient steps at 1/L, beta = log w
//   A      = softmax_sel(s[:, sel] + beta)      Y = softmax_all(s) V
//   C2     = lstsq(A, Y) (QR)
// All query heads that share a KV head are stacked as rows of that KV head's
// problem (GQA grouping, group-major, as per_layer_head_on_policy.py).
//
// The split: the backend (CPU, CUDA, HIP, anything ggml runs) computes QK,
// softmax and exp and either reads the design matrices M, y, A, Y back
// (route Qr, the reference's semantics) or accumulates the Gram matrices
// M^T M, M^T y, A^T A, A^T Y (route Gram, cheaper, squares the condition
// number). The host solves in fp64. The solvers are pure functions and carry
// no ggml dependency.
//
// Failure: fit_block returns false when any KV head's solve failed (neither
// the plain nor the ridge solve produced finite numbers). The FitResult is then
// empty and the caller keeps the block exact.
#pragma once

#include "ggml.h"
#include "ggml-backend.h"

#include <cmath>
#include <cstdint>
#include <string>
#include <vector>

namespace luce_am {

enum class SolveStatus {
    Ok,         // plain solve
    RidgeUsed,  // the plain solve failed (singular or non-finite); the ridge fallback was used
    Failed,     // not finite even with the ridge
};

enum class SolveRoute {
    Qr,    // design matrices to the host, Householder QR least squares (reference: torch lstsq gels)
    Gram,  // Gram matrices from the backend, Cholesky on the normal equations
};

struct NnlsParams {
    double lo = std::exp(-3.0);  // box on w = exp(beta), [e^-3, e^3]
    double hi = std::exp(3.0);
    int    pg_iters = 2;         // projected-gradient steps
    int    power_iters = 3;      // power iterations for L (reference: 3, from a random vector;
                                 // here from the normalised ones vector, deterministic)
    double ridge = 1e-6;         // absolute diagonal ridge of the fallback (reference: 1e-6)
    // Route Gram: a Cholesky pivot not above chol_rel_tol * diagonal is a failure (0: pivot must be > 0).
    double chol_rel_tol = 0.0;
    // Route Qr: the plain solve is rejected when min|R_kk| <= qr_rank_tol * max|R_kk|. 0 (default) is the
    // reference semantics: LAPACK gels only fails on an exactly zero R_kk; a tiny but nonzero pivot (a
    // softmax-tail column) is solved, not regularised. The ridge is for exact singularity and non-finite results.
    double qr_rank_tol = 0.0;
    SolveRoute route = SolveRoute::Qr;
    // The reference stores beta in the model dtype before fitting C2 (bf16 for the Qwen3.8 runs).
    bool   round_beta_bf16 = false;
};

// ---- Pure host solvers (fp64) ------------------------------------------------

// Route Gram. beta = log(w) for G = M^T M (t x t, row-major) and b = M^T y (t).
SolveStatus solve_beta(const double * G, const double * b, int t, const NnlsParams & p, double * beta_out);

// Route Gram. C2 = (A^T A)^-1 A^T Y. AtY and C2_out are row-major [t][d].
SolveStatus solve_values(const double * AtA, const double * AtY, int t, int d, const NnlsParams & p, double * C2_out);

// Route Qr. M_cm is column-major [t][n] (column i contiguous, n rows), y has n entries.
SolveStatus solve_beta_qr(const double * M_cm, const double * y, int n, int t, const NnlsParams & p, double * beta_out);

// Route Qr. A_cm column-major [t][n], Y_cm column-major [d][n]. C2_out row-major [t][d].
SolveStatus solve_values_qr(const double * A_cm, const double * Y_cm, int n, int t, int d, const NnlsParams & p,
                            double * C2_out);

// Narrows doubles to F32 for publication. False (dst unspecified) if any value is non-finite or
// outside the F32 range, or becomes non-finite after the conversion.
bool narrow_f32(const double * src, size_t n, float * dst);

// ---- Backend accumulation -----------------------------------------------------

struct BlockShape {
    int   head_dim = 0;
    int   n_tokens = 0;      // queries per head in the block (640)
    int   n_head = 0;        // query heads
    int   n_head_kv = 0;     // KV heads, n_head % n_head_kv == 0
    int   n_keys = 0;        // T, keys in the block (640)
    int   t = 0;             // kept slots per head (64)
    float scale = 0.0f;      // logit scale, 1/sqrt(head_dim)
    // Queries (tokens) per backend graph. Tensor-core GEMMs accumulate with truncation, so the
    // Gram error grows with the accumulation depth chunk_tokens * n_rep: at 256 tokens the real-shape
    // error is 2e-5, at 64 it is 7e-6, at 16 it is 3e-6 (RTX 3090).
    int   chunk_tokens = 64;
    // Backends that run F32 matmuls on TF32 tensor cores (CUDA on Ampere and newer) lose ~2e-4.
    // Compensated mode splits each operand into two TF32-exact halves (Veltkamp, exponent kept) and
    // sums three products. -1: on for every non-CPU backend, 0: off, 1: on.
    int   compensated = -1;
    // The reference computes q.k in the model dtype (bf16), so the raw scores are rounded to bf16
    // before the scale. Set to reproduce a bf16 reference run.
    bool  bf16_scores = false;
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

// Device inputs: tensors that already live in buffers of the accumulator's backend. No host round
// trip except the t * n_head_kv selected indices.
//   q  F16 [head_dim, n_tokens, n_head], nb[0] = 2, any nb[1], nb[2]. For M3 this is a layer plane of
//      cache.q_blk ([d, 640, n_head, n_fa]): ggml_view_3d(q_blk, d, 640, n_head, nb1, nb2, layer * nb3).
//   k, v  F16 or F32 [head_dim, n_keys, n_head_kv], nb[0] = type size, any nb[1], nb[2].
// None of them is modified. They must stay valid until the last Gram/design call returns.
struct BlockTensors {
    ggml_tensor *   q = nullptr;
    ggml_tensor *   k = nullptr;
    ggml_tensor *   v = nullptr;
    const int32_t * sel = nullptr;
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
// Design matrices per KV head, fp64, column-major per head: M[(g*t + i) * rows + r].
struct BetaDesign {
    int                 rows = 0;  // n_tokens * n_head / n_head_kv
    std::vector<double> M;         // [n_head_kv][t][rows]
    std::vector<double> y;         // [n_head_kv][rows]
};
struct ValueDesign {
    int                 rows = 0;
    std::vector<double> A;         // [n_head_kv][t][rows]
    std::vector<double> Y;         // [n_head_kv][head_dim][rows]
};

struct TransferStats {
    int    h2d_calls = 0, d2h_calls = 0;
    size_t h2d_bytes = 0, d2h_bytes = 0;
};

// Holds one block's inputs on the backend and accumulates Gram / design matrices.
class GramAccumulator {
public:
    GramAccumulator() = default;
    ~GramAccumulator();
    GramAccumulator(const GramAccumulator &) = delete;
    GramAccumulator & operator=(const GramAccumulator &) = delete;

    // Device form (the one implementation). Shape fields must match the tensors. Reuses device
    // buffers and compiled graphs of the previous call when backend and shape are unchanged.
    bool init(ggml_backend_t backend, const BlockShape & shape, const BlockTensors & in, std::string * err);

    // Host form: uploads q, k, v into staging tensors of the backend and calls the device form.
    bool init(ggml_backend_t backend, const BlockShape & shape, const BlockInputs & in, std::string * err);

    // Route Gram: M^T M and M^T y, summed over all queries (all heads of each KV group).
    bool beta_grams(BetaGrams & out, std::string * err);
    // Route Gram: A^T A and A^T Y with beta [t, n_head_kv] (F32, beta[g*t + slot]).
    bool value_grams(const float * beta, ValueGrams & out, std::string * err);
    // Route Qr: the design matrices M, y and A, Y.
    bool beta_design(BetaDesign & out, std::string * err);
    bool value_design(const float * beta, ValueDesign & out, std::string * err);

    // Cumulative host<->device traffic since construction.
    TransferStats transfers() const;

private:
    struct Impl;
    Impl * impl_ = nullptr;
};

// ---- Whole-block fit ------------------------------------------------------------

struct FitResult {
    std::vector<float>       beta;    // [t, n_head_kv]: beta[g*t + slot], the kv_bias block slice
    std::vector<float>       c2;      // [head_dim, t, n_head_kv]: c2[(g*t + slot)*head_dim + dim]
    std::vector<SolveStatus> status;  // per KV head, worst of the beta and value solves
    // Wall time of the phases, ms: upload + prep graph, beta Grams/design, beta solve, value Grams/design, value solve.
    double ms_init = 0, ms_beta_grams = 0, ms_beta_solve = 0, ms_value_grams = 0, ms_value_solve = 0;
    TransferStats transfers;          // host<->device traffic of this fit
};

// Returns false (and leaves `out` empty) when the inputs are bad, a backend call fails, or any KV
// head's solve is Failed; *err names the head. Nothing is to be published from a false return.
bool fit_block(ggml_backend_t backend, const BlockShape & shape, const BlockInputs & in,
               const NnlsParams & p, FitResult & out, std::string * err);

// Same, reusing `acc` (one per device in the engine, called once per layer and sealed block).
bool fit_block(GramAccumulator & acc, ggml_backend_t backend, const BlockShape & shape, const BlockInputs & in,
               const NnlsParams & p, FitResult & out, std::string * err);

// Device-tensor form: no host copy of q, k or v.
bool fit_block(GramAccumulator & acc, ggml_backend_t backend, const BlockShape & shape, const BlockTensors & in,
               const NnlsParams & p, FitResult & out, std::string * err);

}  // namespace luce_am
