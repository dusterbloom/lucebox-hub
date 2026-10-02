// Qwen4Exp forward graph.
//
// qwen4exp_forward() runs n_tokens new tokens starting at cache position pos0
// through the 48-layer hybrid trunk and writes last-token logits. The optional
// batched decode entry point below handles independent one-token slot rows.
// Ported from upstream llama.cpp src/models/qwen4exp.cpp
// (hyper-connections, gated delta net linear attention, dense full attention,
// 512-expert top-10 MoE, per-layer n-gram embedding) into Luzebox's ggml graph
// style.
//
// Single sequence (n_seqs = 1). Past the indexer's block budget, full attention uses QSA selected attention
// (gfx1151: prefill chunks of >= 128 tokens and decode); below it QSA equals dense attention. The PLE table is
// read through Qwen4ExpPleReader and is never uploaded in full.

#pragma once

#include "qwen4exp_internal.h"
#include "qwen4exp_cache.h"

#include "ggml.h"
#include "ggml-backend.h"

#include <cstdint>
#include <vector>

namespace luce::common {

// QSA indexer block pooling: [idim, nb*r] keys -> [idim, nb], block b = mean of tokens r*b .. r*b+r-1.
ggml_tensor * qwen4exp_pool_blocks(ggml_context * c, ggml_tensor * keys, int64_t r);

// K/V span of the stable T=1 decode graph at kv_len. The sequence's first span (`base`, set on first use) leaves at
// least one 256-token window past its kv_len; later spans grow from it in 512-token steps. These are the spans a
// token-by-token decode rebuilds with, as a function of kv_len alone, so a verify row attends over the same span
// (same attention numerics) as plain decode at that position.
int64_t qwen4exp_stable_kv_span(int64_t & base, int64_t max_ctx, int64_t kv_len);

struct Qwen4ExpForwardResult {
    bool ok = false;
    int  n_tokens = 0;
    int  pos0 = 0;
};

// One independent sequence span in a packed forward graph. Tokens within a
// segment are consecutive for this cache; segments never share recurrent, PLE,
// or KV state. The result returns one logits row for each segment's final
// token.
struct Qwen4ExpForwardSegment {
    Qwen4ExpCache * cache = nullptr;
    const int32_t * tokens = nullptr;
    int n_tokens = 0;
    int pos0 = 0;
};

// Run the trunk. `tokens` has n_tokens entries, processed as one contiguous
// single-sequence span at positions [pos0, pos0 + n_tokens). On success the
// cache is advanced to pos0 + n_tokens and out_logits holds n_vocab floats
// for the final token. out_hidden, when set, receives every token's final HC
// residual (n_embd * n_hc floats each) for the MTP draft head.
//
// verify (MTP speculation, n_tokens == 2, qwen4exp_verify_supported): each
// token is computed exactly as a T=1 forward at its position computes it
// (batch-invariant matmuls, per-token attention), out_logits holds both rows,
// and the cache keeps the state after the first token for
// qwen4exp_verify_rollback.
Qwen4ExpForwardResult qwen4exp_forward(ggml_backend_t backend,
                                       const Qwen4ExpWeights & w,
                                       Qwen4ExpCache & cache,
                                       const int32_t * tokens,
                                       int n_tokens,
                                       int pos0,
                                       std::vector<float> & out_logits,
                                       std::vector<float> * out_hidden = nullptr,
                                       bool verify = false,
                                       bool qsa_rebuild_reference = false); // smoke oracle only

// The cache was created with `mtp` and the graph is the default one (not QWEN4EXP_UPSTREAM / QWEN4EXP_DUMP).
bool qwen4exp_verify_supported(const Qwen4ExpCache & cache);

// After a verify forward at pos0 whose second token was rejected: return the cache to the state right after the
// first token (recurrent, conv and PLE state, PLE n-gram tail, QSA pooled-block count).
bool qwen4exp_verify_rollback(ggml_backend_t backend, const Qwen4ExpWeights & w, Qwen4ExpCache & cache, int pos0);

// MTP draft step over (trunk hidden h_p, token x_{p+1}) pairs at positions [pos0, pos0 + n): runs the sidecar's
// nextn projection and layer, writes the draft layer's K/V there, and returns the logits of the last pair (its
// argmax drafts x_{p+2}). `hidden` holds n rows of the trunk's final HC residual (n_embd * n_hc floats each), as
// returned by qwen4exp_forward's out_hidden.
bool qwen4exp_mtp_forward(ggml_backend_t backend, const Qwen4ExpWeights & w, Qwen4ExpCache & cache,
                          const int32_t * tokens, const float * hidden, int n, int pos0,
                          std::vector<float> & out_logits);

// Decode one next token for each independent slot. `caches[s]` owns that
// sequence's KV and recurrent state; `tokens[s]` and `positions[s]` are never
// interpreted as a common time axis. The shared workspace must outlive calls
// and is normally owned by the sequence-engine/model instance.
// Enabled only when QWEN4EXP_BATCHED_DECODE=1 and never under UPSTREAM.
Qwen4ExpForwardResult qwen4exp_forward_batched(
                                       ggml_backend_t backend,
                                       const Qwen4ExpWeights & w,
                                       Qwen4ExpCache * const * caches,
                                       const int32_t * tokens,
                                       const int32_t * positions,
                                       int n_slots,
                                       Qwen4ExpBatchedDecodeWorkspace & workspace,
                                       std::vector<std::vector<float>> & out_logits);

// Packed independent-sequence forward for concurrent prefill/decode. Shared
// dense/HC/MoE operations use the concatenated token rows; stateful operators
// are built separately for each segment against its own cache.
Qwen4ExpForwardResult qwen4exp_forward_packed(
                                       ggml_backend_t backend,
                                       const Qwen4ExpWeights & w,
                                       const Qwen4ExpForwardSegment * segments,
                                       int n_segments,
                                       Qwen4ExpBatchedDecodeWorkspace & workspace,
                                       std::vector<std::vector<float>> & out_logits);

}  // namespace luce::common
