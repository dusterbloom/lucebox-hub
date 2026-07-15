#include "common/dspark_proposal_graph.h"

#include "ggml.h"

#include <cstdio>
#include <vector>

using namespace dflash::common;

namespace {

int failures = 0;

void check(bool condition, const char * expression, int line) {
    if (condition) return;
    ++failures;
    std::fprintf(stderr, "check failed at line %d: %s\n", line, expression);
}

}  // namespace

#define CHECK(expr) check((expr), #expr, __LINE__)

int main() {
    ggml_init_params params{};
    params.mem_size = 1024 * 1024;
    params.no_alloc = true;
    ggml_context * ctx = ggml_init(params);
    CHECK(ctx != nullptr);
    if (!ctx) return 1;

    ggml_cgraph * graph = ggml_new_graph_custom(ctx, 128, false);
    constexpr int rank = 2;
    constexpr int vocab = 8;
    constexpr int proposals = 4;

    DraftDSparkWeights head{};
    head.enabled = true;
    head.markov_rank = rank;
    head.vocab_size = vocab;
    head.markov_w1 = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, rank, vocab);
    head.markov_w2 = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, rank, vocab);

    ggml_tensor * logits =
        ggml_new_tensor_2d(ctx, GGML_TYPE_F32, vocab, proposals);
    ggml_tensor * seed = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, 1);
    std::vector<ggml_tensor *> tokens;

    CHECK(build_dspark_proposal_chain(
        ctx, graph, head, proposals, logits, seed, tokens));
    CHECK(tokens.size() == proposals);
    if (tokens.size() == proposals) {
        for (int row = 0; row < proposals; ++row) {
            ggml_tensor * corrected = tokens[(size_t)row]->src[0];
            ggml_tensor * bias = corrected ? corrected->src[1] : nullptr;
            ggml_tensor * prev_embedding = bias ? bias->src[1] : nullptr;
            CHECK(tokens[(size_t)row]->op == GGML_OP_ARGMAX);
            CHECK(prev_embedding != nullptr);
            CHECK(prev_embedding && prev_embedding->op == GGML_OP_GET_ROWS);
            CHECK(prev_embedding && prev_embedding->src[1] ==
                (row == 0 ? seed : tokens[(size_t)row - 1]));
        }
    }

    CHECK(!build_dspark_proposal_chain(
        ctx, graph, head, 0, logits, seed, tokens));
    CHECK(tokens.empty());
    CHECK(!build_dspark_proposal_chain(
        ctx, graph, head, proposals - 1, logits, seed, tokens));
    CHECK(tokens.empty());

    ggml_free(ctx);
    return failures == 0 ? 0 : 1;
}
