#pragma once

#include "internal.h"

#include <vector>

namespace dflash::common {

// Append a proposals-only DSpark Markov chain to an existing draft graph.
// Row zero is corrected from the accepted anchor token, and every later row
// is corrected from the actual argmax selected at the previous row.
bool build_dspark_proposal_chain(
    ggml_context * ctx,
    ggml_cgraph * gf,
    const DraftDSparkWeights & head,
    int proposal_count,
    ggml_tensor * base_logits,
    ggml_tensor * seed_token,
    std::vector<ggml_tensor *> & proposal_tokens);

}  // namespace dflash::common
