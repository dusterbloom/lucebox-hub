// KVFlash 2.0b M4: HighestAttnKeys scoring for attention-matching (AM)
// compaction. Selects, for every KV head, the top-k keys of one sealed block
// by the repo's "rms" score, computed from the block's own prefill queries.
//
// Reference algorithm: compaction/algorithms/highest_attention_keys.py
// (_select_keys_highest_attention, score_method 'rms'), with the GQA grouping
// of compaction/query_generation/query_generator.py: KV head h owns the query
// heads h*G .. h*G+G-1 (G = n_head / n_kv_head); their queries are pooled into
// one query set of G*n_queries rows for that KV head. For every pooled query
// row the softmax runs over all keys of the block (non-causal, no mask):
//     s = (q . k) / sqrt(head_dim)          fp32
//     w = exp(s - max s) / sum exp(s - max s)
// and the key score is sqrt(mean over rows of w^2). The mean and the sqrt are
// monotone and identical for every key of a head, so this code ranks keys by
// the raw sum of w^2 (the sqrt can only merge distinct sums, never reorder).
//
// Tie rule (see select_top_k): higher score first; among exactly equal
// scores the LOWER key index wins. torch.topk leaves tie order unspecified
// (CPU and CUDA kernels differ), so this is the deterministic choice, equal to
// torch.sort(stable=True, descending=True)[:k]. The same rule applies on every
// backend because selection runs on the host over the fp32 scores.
#pragma once

#include "ggml.h"
#include "ggml-backend.h"

#include <cstdint>
#include <string>
#include <vector>

namespace luce::am {

struct ScoreDims {
    int head_dim;   // d
    int n_head;     // query heads, a multiple of n_kv_head
    int n_kv_head;
    int n_keys;     // T, keys of the block
    int n_queries;  // n, queries of the block, per query head
};

// Per-key score sum_q w^2 for every KV head, on `backend`.
//   q: F16, contiguous [head_dim, n_queries, n_head] (M3 capture layout)
//   k: k_type (F16 or F32), contiguous [head_dim, n_keys, n_kv_head]
//   scores: resized to n_kv_head * n_keys, head-major (scores[h*n_keys + j])
// The graph is chunked over queries: at most `chunk_rows` softmax rows
// (n_head * queries) are live at once, so chunk_rows must be >= n_head (one
// query per head is the smallest chunk); smaller values are rejected.
// Returns false and fills *err on failure. This host-pointer form uploads q
// and k to `backend` and calls the tensor form below.
bool score_keys(ggml_backend_t backend, const ScoreDims & dims, const ggml_fp16_t * q,
                ggml_type k_type, const void * k, int chunk_rows,
                std::vector<float> & scores, std::string * err);

// Tensor form, no host round trip: q and k already live in buffers of
// `backend` (for M3, a plane view of cache.q_blk and the block's K).
//   q: F16 or F32 [head_dim, n_queries, n_head]; contiguous rows, any nb[1], nb[2]
//      (F32 skips the cast; the engine feeds F16, the R3 gate also runs F32)
//   k: F16 or F32 [head_dim, n_keys, n_kv_head]; nb[0] = type size, any nb[1], nb[2]
// Dimensions come from the tensors. Neither tensor is modified.
bool score_keys(ggml_backend_t backend, ggml_tensor * q, ggml_tensor * k, int chunk_rows,
                std::vector<float> & scores, std::string * err);

// Pure host selection: indices of the k largest scores, ordered by score
// descending, exactly equal scores by index ascending. out has k entries.
void select_top_k(const float * scores, int n, int k, int32_t * out);

// score_keys + select_top_k per KV head. out: n_kv_head * top_k int32,
// head-major (out[h*top_k + r] is the r-th ranked key index of head h).
bool select_keys(ggml_backend_t backend, const ScoreDims & dims, const ggml_fp16_t * q,
                 ggml_type k_type, const void * k, int top_k, int chunk_rows,
                 std::vector<int32_t> & out, std::string * err);

// Tensor-form score_keys + select_top_k per KV head.
bool select_keys(ggml_backend_t backend, ggml_tensor * q, ggml_tensor * k, int top_k, int chunk_rows,
                 std::vector<int32_t> & out, std::string * err);

}  // namespace luce::am
