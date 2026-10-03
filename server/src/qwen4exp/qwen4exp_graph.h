// Qwen4Exp forward graph.
//
// qwen4exp_forward() runs n_tokens new tokens starting at cache position pos0
// through the 48-layer hybrid trunk and writes last-token logits. The multi-slot
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

struct Qwen4ExpForwardResult {
    bool ok = false;
    int  n_tokens = 0;
    int  pos0 = 0;
};

// One independent sequence span for batch eligibility and per-slot solo fallback.
struct Qwen4ExpForwardSegment {
    Qwen4ExpCache * cache = nullptr;
    const int32_t * tokens = nullptr;
    int n_tokens = 0;
    int pos0 = 0;
};

// Pure decision for validated spans: at most four one-token rows, all dense in the solo path.
// use_qsa is the cached gfx1151 capability; reference caches always run solo.
bool qwen4exp_can_batch(const Qwen4ExpWeights & w,
                       const Qwen4ExpForwardSegment * segments, int n_segments, bool use_qsa);

// Run the trunk. `tokens` has n_tokens entries, processed as one contiguous
// single-sequence span at positions [pos0, pos0 + n_tokens). On success the
// cache is advanced to pos0 + n_tokens and out_logits holds n_vocab floats
// for the final token.
Qwen4ExpForwardResult qwen4exp_forward(ggml_backend_t backend,
                                       const Qwen4ExpWeights & w,
                                       Qwen4ExpCache & cache,
                                       const int32_t * tokens,
                                       int n_tokens,
                                       int pos0,
                                       std::vector<float> & out_logits,
                                       bool dump = false);  // test-only activation summaries

// Decode one next token for each independent slot. `caches[s]` owns that
// sequence's KV and recurrent state; `tokens[s]` and `positions[s]` are never
// interpreted as a common time axis. The shared workspace must outlive calls
// and is normally owned by the sequence-engine/model instance.
// If any slot needs QSA or more than four slots are active, use per-slot solo forwards.
// Reference caches also use per-slot solo forwards. The server admits at most four slots.
Qwen4ExpForwardResult qwen4exp_forward_batched(
                                       ggml_backend_t backend,
                                       const Qwen4ExpWeights & w,
                                       Qwen4ExpCache * const * caches,
                                       const int32_t * tokens,
                                       const int32_t * positions,
                                       int n_slots,
                                       Qwen4ExpBatchedDecodeWorkspace & workspace,
                                       std::vector<std::vector<float>> & out_logits);

}  // namespace luce::common
