#pragma once

#include "dflash_target.h"
#include "internal.h"

#include <cstdint>
#include <vector>

namespace dflash::common {

// Append native DSpark's sequential Markov correction to an existing draft
// graph. Every base-logit row is a proposal: row 0 is corrected from the
// accepted anchor token, and each later row is corrected from the actual
// argmax selected at the previous row.
bool build_dspark_native_proposal_graph(
    ggml_context * ctx,
    ggml_cgraph * gf,
    const DraftWeights & dw,
    ggml_tensor * base_logits,
    ggml_tensor * seed_token,
    std::vector<ggml_tensor *> & proposal_tokens);

bool dspark_markov_correct_greedy_chain(const DraftWeights & dw,
                                        ggml_backend_t backend,
                                        DFlashTarget & target,
                                        const float * local_hidden,
                                        int q_len,
                                        int32_t last_tok,
                                        float confidence_threshold,
                                        std::vector<int32_t> & draft_tok);

// Fused variant: base logits (one lm_head matmul over all candidates) +
// unrolled Markov correction chain + in-graph argmax feeding the next
// step's get_rows, all in ONE graph on the draft backend. No host logits
// round-trip. Does not implement the confidence gate; callers wanting
// confidence-prefix truncation must use the unfused path.
bool dspark_markov_correct_greedy_chain_fused(const DraftWeights & dw,
                                              ggml_backend_t backend,
                                              ggml_tensor * lm_head,
                                              const float * local_hidden,
                                              int q_len,
                                              int32_t last_tok,
                                              std::vector<int32_t> & draft_tok);

// DDTree candidate generation with the Markov correction: base logits for
// all n_tokens positions in ONE lm_head matmul; rows 1..n-1 get the low-rank
// previous-token bias chained along the main (argmax) path; top-K extracted
// on host via extract_draft_topk. Output contract matches
// DFlashTarget::project_hidden_to_topk (row 0 = seed position, uncorrected).
bool dspark_markov_project_topk(const DraftWeights & dw,
                                ggml_backend_t backend,
                                ggml_tensor * lm_head,
                                const float * hidden,
                                int n_tokens, int K, float temperature,
                                int32_t last_tok,
                                std::vector<float> & top_log_probs,
                                std::vector<int32_t> & top_token_ids);

}  // namespace dflash::common
