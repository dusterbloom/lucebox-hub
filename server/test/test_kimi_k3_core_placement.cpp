#include "kimi_k3/kimi_k3_backend.h"
#include "kimi_k3/kimi_k3_prefill_plan.h"

#include <cstdio>
#include <cstdlib>
#include <vector>

using namespace dflash::common;

#define REQUIRE(condition) do {                                           \
    if (!(condition)) {                                                   \
        std::fprintf(stderr, "requirement failed at %s:%d: %s\n",       \
                     __FILE__, __LINE__, #condition);                     \
        std::exit(1);                                                     \
    }                                                                     \
} while (0)

int main() {
    KimiK3CorePlacement placement = KimiK3CorePlacement::Accelerator;
    REQUIRE(parse_kimi_k3_core_placement("cpu", placement));
    REQUIRE(placement == KimiK3CorePlacement::Cpu);
    REQUIRE(std::string(kimi_k3_core_placement_name(placement)) == "cpu");
    REQUIRE(parse_kimi_k3_core_placement("accelerator", placement));
    REQUIRE(placement == KimiK3CorePlacement::Accelerator);
    REQUIRE(!parse_kimi_k3_core_placement("gpu", placement));

    int prefill_chunk = 0;
    REQUIRE(parse_kimi_k3_prefill_chunk(nullptr, prefill_chunk));
    REQUIRE(prefill_chunk == 1);
    REQUIRE(parse_kimi_k3_prefill_chunk("", prefill_chunk));
    REQUIRE(prefill_chunk == 1);
    REQUIRE(parse_kimi_k3_prefill_chunk("1", prefill_chunk));
    REQUIRE(prefill_chunk == 1);
    REQUIRE(parse_kimi_k3_prefill_chunk("2", prefill_chunk));
    REQUIRE(prefill_chunk == 2);
    REQUIRE(parse_kimi_k3_prefill_chunk("4", prefill_chunk));
    REQUIRE(prefill_chunk == 4);
    REQUIRE(parse_kimi_k3_prefill_chunk("8", prefill_chunk));
    REQUIRE(prefill_chunk == 8);
    REQUIRE(!parse_kimi_k3_prefill_chunk("0", prefill_chunk));
    REQUIRE(!parse_kimi_k3_prefill_chunk("2x", prefill_chunk));

    REQUIRE(kimi_k3_prefill_chunk_size(16, 8, true) == 8);
    REQUIRE(kimi_k3_prefill_chunk_size(8, 8, true) == 8);
    REQUIRE(kimi_k3_prefill_chunk_size(7, 8, true) == 1);
    REQUIRE(kimi_k3_prefill_chunk_size(3, 2, false) == 2);
    REQUIRE(kimi_k3_prefill_chunk_size(1, 4, false) == 1);
    REQUIRE(kimi_k3_p58_configuration_valid(1, false));
    REQUIRE(kimi_k3_p58_configuration_valid(4, false));
    REQUIRE(kimi_k3_p58_configuration_valid(8, true));
    REQUIRE(!kimi_k3_p58_configuration_valid(8, false));
    REQUIRE(!kimi_k3_p58_configuration_valid(4, true));
    REQUIRE(kimi_k3_p58_oracle_candidate(true, 8, true));
    REQUIRE(!kimi_k3_p58_oracle_candidate(true, 8, false));
    REQUIRE(!kimi_k3_p58_oracle_candidate(true, 4, true));
    REQUIRE(!kimi_k3_p58_oracle_candidate(false, 8, true));

    const int32_t prefill_experts[] = {3, 1, 3, 2};
    const float prefill_weights[] = {0.4f, 0.6f, 0.7f, 0.3f};
    const uint8_t prefill_partitions[] = {0, 0, 1, 0};
    const uint16_t prefill_masks[] = {
        static_cast<uint16_t>((1u << 0) | (1u << 2)),
        static_cast<uint16_t>(1u << 1),
        static_cast<uint16_t>((1u << 2) | (1u << 3)),
        static_cast<uint16_t>(1u << 0),
    };
    KimiK3PrefillLayerPlan prefill_plan;
    REQUIRE(plan_kimi_k3_layer_major_prefill(
        2, 2, 4, 4, 8192, 4096, 4096, prefill_experts,
        prefill_weights, prefill_partitions, prefill_masks, prefill_plan));
    REQUIRE(prefill_plan.width == 2);
    REQUIRE(prefill_plan.top_k == 2);
    REQUIRE(prefill_plan.requested_slab_records == 6);
    REQUIRE(prefill_plan.physical_reads.size() == 5);
    REQUIRE(prefill_plan.expert_groups.size() == 3);
    REQUIRE(prefill_plan.canonical_routes == std::vector<int>({1, 0, 3, 2}));
    REQUIRE(prefill_plan.expert_groups[0].expert == 1);
    REQUIRE(prefill_plan.expert_groups[1].expert == 2);
    REQUIRE(prefill_plan.expert_groups[2].expert == 3);
    REQUIRE(prefill_plan.expert_groups[2].routes.size() == 2);
    REQUIRE(prefill_plan.expert_groups[2].union_natural_slab_mask ==
        static_cast<uint16_t>((1u << 0) | (1u << 2) | (1u << 3)));
    REQUIRE(prefill_plan.physical_reads.front().aligned_offset ==
        8192 + static_cast<uint64_t>(1 * 4 + 1) * 4096);
    const int32_t invalid_prefill_experts[] = {4};
    REQUIRE(!plan_kimi_k3_layer_major_prefill(
        1, 1, 4, 4, 0, 4096, 4096, invalid_prefill_experts,
        prefill_weights, prefill_partitions, prefill_masks, prefill_plan));

    std::string error;
    ggml_backend_t cpu = init_kimi_k3_core_backend(
        KimiK3CorePlacement::Cpu, 0, &error);
    REQUIRE(cpu != nullptr);

    ggml_init_params params{};
    params.mem_size = 4096;
    params.no_alloc = true;
    ggml_context * ctx = ggml_init(params);
    REQUIRE(ctx != nullptr);
    KimiK3Weights embedding_weights;
    embedding_weights.n_embd = 4;
    embedding_weights.n_vocab = 2;
    embedding_weights.tok_embd = ggml_new_tensor_2d(
        ctx, GGML_TYPE_F16, embedding_weights.n_embd,
        embedding_weights.n_vocab);
    ggml_backend_buffer_t embedding_buffer =
        ggml_backend_alloc_ctx_tensors(ctx, cpu);
    REQUIRE(embedding_buffer != nullptr);
    const ggml_fp16_t embedding_values[] = {
        ggml_fp32_to_fp16(1.0f), ggml_fp32_to_fp16(2.0f),
        ggml_fp32_to_fp16(3.0f), ggml_fp32_to_fp16(4.0f),
        ggml_fp32_to_fp16(5.0f), ggml_fp32_to_fp16(6.0f),
        ggml_fp32_to_fp16(7.0f), ggml_fp32_to_fp16(8.0f),
    };
    ggml_backend_tensor_set(
        embedding_weights.tok_embd, embedding_values, 0,
        sizeof(embedding_values));
    std::vector<float> embedding_row(
        static_cast<size_t>(embedding_weights.n_embd));
    REQUIRE(kimi_k3_read_token_embeddings_on_host(
        embedding_weights, {1}, embedding_row));
    REQUIRE(embedding_row ==
        std::vector<float>({5.0f, 6.0f, 7.0f, 8.0f}));
    std::vector<float> invalid_embedding_row;
    REQUIRE(!kimi_k3_read_token_embeddings_on_host(
        embedding_weights, {1}, invalid_embedding_row));
    ggml_backend_buffer_free(embedding_buffer);
    ggml_free(ctx);
    ggml_backend_free(cpu);

    std::printf("Kimi K3 core placement test passed\n");
    return 0;
}
