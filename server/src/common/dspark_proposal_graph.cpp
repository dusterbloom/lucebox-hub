#include "dspark_proposal_graph.h"

namespace dflash::common {

bool build_dspark_proposal_chain(
        ggml_context * ctx,
        ggml_cgraph * gf,
        const DraftDSparkWeights & head,
        int proposal_count,
        ggml_tensor * base_logits,
        ggml_tensor * seed_token,
        std::vector<ggml_tensor *> & proposal_tokens) {
    proposal_tokens.clear();
    if (!ctx || !gf || !base_logits || !seed_token ||
        !head.enabled || !head.markov_w1 || !head.markov_w2) {
        return false;
    }
    const int vocab = (int)base_logits->ne[0];
    if (proposal_count <= 0 || base_logits->ne[1] != proposal_count ||
        (head.vocab_size > 0 && vocab != head.vocab_size)) {
        return false;
    }

    ggml_tensor * prev = seed_token;
    proposal_tokens.reserve((size_t)proposal_count);
    for (int row = 0; row < proposal_count; ++row) {
        ggml_tensor * prev_emb = ggml_get_rows(ctx, head.markov_w1, prev);
        ggml_tensor * bias = ggml_mul_mat(ctx, head.markov_w2, prev_emb);
        ggml_tensor * base_i = ggml_view_2d(
            ctx, base_logits, vocab, 1, base_logits->nb[1],
            (size_t)row * base_logits->nb[1]);
        ggml_tensor * corrected = ggml_add(ctx, base_i, bias);
        ggml_tensor * tok = ggml_argmax(ctx, corrected);
        ggml_set_output(tok);
        ggml_build_forward_expand(gf, tok);
        proposal_tokens.push_back(tok);
        prev = tok;
    }
    return true;
}

}  // namespace dflash::common
