#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-cuda.h"
#include "ggml.h"

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <limits>
#include <numeric>
#include <random>
#include <vector>

namespace {

constexpr int block_k = 256;
constexpr int virtual_blocks = 12;
constexpr int virtual_k = block_k*virtual_blocks;
constexpr int default_nrows = 64;

struct quantized_corpus {
    ggml_type type;
    int nrows;
    size_t block_bytes;
    std::vector<uint8_t> weights;
    std::vector<float> activations;
};

quantized_corpus make_corpus(ggml_type type, int nrows = default_nrows) {
    ggml_quantize_init(type);
    const size_t block_bytes = ggml_row_size(type, block_k);
    quantized_corpus corpus{
        type,
        nrows,
        block_bytes,
        std::vector<uint8_t>((size_t) virtual_blocks*nrows*block_bytes),
        std::vector<float>((size_t) virtual_blocks*block_k),
    };

    std::mt19937 rng(20260819u + (unsigned) type*1009u);
    std::uniform_real_distribution<float> weight_dist(-1.0f, 1.0f);
    std::uniform_real_distribution<float> activation_dist(-2.0f, 2.0f);
    std::vector<float> block(block_k);
    std::vector<float> importance(block_k, 1.0f);

    for (int natural = 0; natural < virtual_blocks; ++natural) {
        for (int i = 0; i < block_k; ++i) {
            corpus.activations[(size_t) natural*block_k + i] =
                activation_dist(rng);
        }
        for (int row = 0; row < corpus.nrows; ++row) {
            for (float & value : block) {
                value = weight_dist(rng);
            }
            uint8_t * dst = corpus.weights.data() +
                ((size_t) natural*nrows + row)*block_bytes;
            const size_t written = ggml_quantize_chunk(
                type, block.data(), dst, 0, 1, block_k, importance.data());
            if (written != block_bytes) {
                std::fprintf(stderr,
                    "quantization size mismatch type=%s natural=%d row=%d got=%zu expected=%zu\n",
                    ggml_type_name(type), natural, row, written, block_bytes);
                std::abort();
            }
        }
    }
    return corpus;
}

bool contains(const std::vector<int> & values, int value) {
    return std::find(values.begin(), values.end(), value) != values.end();
}

bool run_case(
        ggml_backend_t backend,
        const quantized_corpus & corpus,
        const std::vector<int> & requested,
        const std::vector<int> & resident,
        const char * label) {
    ggml_init_params params = {4*1024*1024, nullptr, true};
    ggml_context * ctx = ggml_init(params);
    if (!ctx) {
        return false;
    }

    ggml_tensor * full_weights =
        ggml_new_tensor_2d(ctx, corpus.type, virtual_k, corpus.nrows);
    ggml_tensor * full_x = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, virtual_k);
    ggml_tensor * compact_weights = ggml_new_tensor_3d(
        ctx, corpus.type, block_k, corpus.nrows, resident.size());
    ggml_tensor * compact_x = ggml_new_tensor_2d(
        ctx, GGML_TYPE_F32, block_k, resident.size());
    ggml_tensor * natural_to_compact =
        ggml_new_tensor_1d(ctx, GGML_TYPE_I32, virtual_blocks);
    for (ggml_tensor * input :
         {full_weights, full_x, compact_weights, compact_x, natural_to_compact}) {
        ggml_set_input(input);
    }

    ggml_tensor * reference = ggml_mul_mat(ctx, full_weights, full_x);
    ggml_tensor * sparse = ggml_mul_mat_sparse_k_blocks(
        ctx, compact_weights, compact_x, natural_to_compact, virtual_k);
    ggml_set_output(reference);
    ggml_set_output(sparse);
    ggml_cgraph * graph = ggml_new_graph(ctx);
    ggml_build_forward_expand(graph, reference);
    ggml_build_forward_expand(graph, sparse);

    ggml_gallocr_t alloc =
        ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend));
    if (!ggml_gallocr_alloc_graph(alloc, graph)) {
        ggml_gallocr_free(alloc);
        ggml_free(ctx);
        return false;
    }

    std::vector<uint8_t> full_weights_h(ggml_nbytes(full_weights), 0);
    std::vector<float> full_x_h(virtual_k, 0.0f);
    std::vector<uint8_t> compact_weights_h(ggml_nbytes(compact_weights));
    std::vector<float> compact_x_h((size_t) resident.size()*block_k);
    std::array<int32_t, virtual_blocks> map;
    map.fill(-1);

    for (size_t slot = 0; slot < resident.size(); ++slot) {
        const int natural = resident[slot];
        if (natural < 0 || natural >= virtual_blocks) {
            return false;
        }
        for (int row = 0; row < corpus.nrows; ++row) {
            const uint8_t * src = corpus.weights.data() +
                ((size_t) natural*corpus.nrows + row)*corpus.block_bytes;
            uint8_t * compact_dst = compact_weights_h.data() +
                (slot*corpus.nrows + row)*corpus.block_bytes;
            std::memcpy(compact_dst, src, corpus.block_bytes);
            if (contains(requested, natural)) {
                uint8_t * full_dst = full_weights_h.data() +
                    ((size_t) row*virtual_blocks + natural)*corpus.block_bytes;
                std::memcpy(full_dst, src, corpus.block_bytes);
            }
        }
        const float * x_src = corpus.activations.data() + (size_t) natural*block_k;
        std::copy_n(x_src, block_k, compact_x_h.data() + slot*block_k);
        if (contains(requested, natural)) {
            std::copy_n(x_src, block_k, full_x_h.data() + natural*block_k);
            map[natural] = (int32_t) slot;
        }
    }
    for (int natural : requested) {
        if (!contains(resident, natural)) {
            return false;
        }
    }

    ggml_backend_tensor_set(
        full_weights, full_weights_h.data(), 0, full_weights_h.size());
    ggml_backend_tensor_set(
        full_x, full_x_h.data(), 0, full_x_h.size()*sizeof(float));
    ggml_backend_tensor_set(
        compact_weights, compact_weights_h.data(), 0, compact_weights_h.size());
    ggml_backend_tensor_set(
        compact_x, compact_x_h.data(), 0, compact_x_h.size()*sizeof(float));
    ggml_backend_tensor_set(
        natural_to_compact, map.data(), 0, map.size()*sizeof(int32_t));

    const ggml_status status = ggml_backend_graph_compute(backend, graph);
    ggml_backend_synchronize(backend);
    std::vector<float> reference_h(corpus.nrows);
    std::vector<float> sparse_h(corpus.nrows);
    if (status == GGML_STATUS_SUCCESS) {
        ggml_backend_tensor_get(
            reference, reference_h.data(), 0, reference_h.size()*sizeof(float));
        ggml_backend_tensor_get(
            sparse, sparse_h.data(), 0, sparse_h.size()*sizeof(float));
    }

    const bool exact = status == GGML_STATUS_SUCCESS &&
        std::memcmp(
            reference_h.data(), sparse_h.data(), reference_h.size()*sizeof(float)) == 0;
    if (!exact) {
        size_t first = reference_h.size();
        for (size_t i = 0; i < reference_h.size(); ++i) {
            if (std::memcmp(&reference_h[i], &sparse_h[i], sizeof(float)) != 0) {
                first = i;
                break;
            }
        }
        std::fprintf(stderr,
            "sparse-K mismatch type=%s case=%s rows=%d status=%d row=%zu reference=%a sparse=%a\n",
            ggml_type_name(corpus.type), label, corpus.nrows, (int) status, first,
            first < reference_h.size() ? reference_h[first] : 0.0f,
            first < sparse_h.size() ? sparse_h[first] : 0.0f);
    }

    ggml_gallocr_free(alloc);
    ggml_free(ctx);
    return exact;
}

} // namespace

int main() {
    int device = 0;
    if (const char * value = std::getenv("DFLASH_TEST_GPU")) {
        device = std::max(0, std::atoi(value));
    }
    ggml_backend_t backend = ggml_backend_cuda_init(device);
    if (!backend) {
        std::printf(
            "[sparse-k-mmvq-test] SKIP: CUDA/HIP backend unavailable device=%d\n",
            device);
        return 77;
    }

    bool ok = true;
    const std::array<ggml_type, 2> types = {
        GGML_TYPE_IQ1_S,
        GGML_TYPE_IQ2_XXS,
    };
    for (ggml_type type : types) {
        const quantized_corpus corpus = make_corpus(type);
        for (int natural = 0; natural < virtual_blocks; ++natural) {
            const std::vector<int> singleton = {natural};
            ok = run_case(
                backend, corpus, singleton, singleton, "singleton") && ok;
        }
        ok = run_case(
            backend, corpus, {0, 5, 11}, {0, 5, 11}, "three-slab") && ok;
        ok = run_case(
            backend, corpus, {0, 4}, {0, 4}, "same-lane-two") && ok;
        ok = run_case(
            backend, corpus, {0, 4, 8}, {0, 4, 8}, "same-lane-three") && ok;
        ok = run_case(
            backend, corpus, {0, 1, 2, 3}, {0, 1, 2, 3}, "first-four") && ok;
        ok = run_case(
            backend, corpus, {0, 2, 4, 6, 8, 10},
            {0, 2, 4, 6, 8, 10}, "six-slab") && ok;
        std::vector<int> all(virtual_blocks);
        std::iota(all.begin(), all.end(), 0);
        ok = run_case(backend, corpus, all, all, "all-twelve") && ok;
        std::vector<int> reversed = all;
        std::reverse(reversed.begin(), reversed.end());
        ok = run_case(
            backend, corpus, all, reversed, "all-twelve-reversed") && ok;
        for (int resident_count = 1; resident_count <= virtual_blocks;
             ++resident_count) {
            std::vector<int> resident(resident_count);
            std::iota(resident.begin(), resident.end(), 0);
            ok = run_case(
                backend, corpus, {0}, resident, "resident-count") && ok;
        }
        ok = run_case(
            backend, corpus, {0, 2, 4, 6, 8, 10},
            {9, 0, 10, 5, 2, 8, 4, 6}, "resident-superset") && ok;
        quantized_corpus edge = corpus;
        float * edge_x = edge.activations.data() + 7*block_k;
        for (int i = 0; i < block_k; ++i) {
            edge_x[i] = (i & 1) ? -0.0f : 0.0f;
        }
        edge_x[31] = std::numeric_limits<float>::quiet_NaN();
        ok = run_case(backend, edge, {7}, {7}, "signed-zero-nan") && ok;
        const quantized_corpus production = make_corpus(type, 3584);
        ok = run_case(
            backend, production, all, reversed, "production-reversed") && ok;
    }

    ggml_quantize_free();
    ggml_backend_free(backend);
    std::printf(
        "[sparse-k-mmvq-test] device=%d cases=68 parity=%s\n",
        device, ok ? "PASS" : "FAIL");
    return ok ? 0 : 1;
}
