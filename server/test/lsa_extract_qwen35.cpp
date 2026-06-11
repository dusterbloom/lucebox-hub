// Offline Qwen3.5 LSA teacher extractor.
//
// Usage:
//   lsa_extract_qwen35 <model.gguf> <tokens.i32> <output-dir> [boundaries.txt]
//
// Token input is native little-endian int32. Optional boundaries are one
// block-aligned committed-token position per line. Without a file, up to eight
// positions are distributed across the sequence.

#include "dflash27b.h"
#include "internal.h"
#include "qwen35/qwen35_lsa_raw_writer.h"
#include "qwen35/qwen35_lsa_teacher.h"

#include "ggml-backend.h"
#include "ggml-cuda.h"

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <set>
#include <string>
#include <vector>

using namespace dflash::common;

namespace {

constexpr int BLOCK_SIZE = 64;
constexpr int RECENT_TOKENS = 8192;

bool read_tokens(const char * path, std::vector<int32_t> & tokens) {
    std::ifstream input(path, std::ios::binary | std::ios::ate);
    if (!input) return false;
    const std::streamsize bytes = input.tellg();
    if (bytes <= 0 || bytes % static_cast<std::streamsize>(sizeof(int32_t))) {
        return false;
    }
    input.seekg(0);
    tokens.resize(static_cast<size_t>(bytes) / sizeof(int32_t));
    return static_cast<bool>(input.read(
        reinterpret_cast<char *>(tokens.data()), bytes));
}

bool read_boundaries(const char * path, std::set<int> & boundaries) {
    std::ifstream input(path);
    int position = 0;
    while (input >> position) boundaries.insert(position);
    return input.eof() && !boundaries.empty();
}

std::set<int> default_boundaries(int token_count) {
    std::set<int> result;
    const int first = std::min(
        token_count - BLOCK_SIZE,
        ((RECENT_TOKENS + 2 * BLOCK_SIZE - 1) / BLOCK_SIZE) * BLOCK_SIZE);
    const int last = ((token_count - BLOCK_SIZE) / BLOCK_SIZE) * BLOCK_SIZE;
    if (first < BLOCK_SIZE || last < first) return result;
    constexpr int count = 8;
    for (int i = 0; i < count; ++i) {
        const int64_t numerator =
            static_cast<int64_t>(last - first) * i;
        const int position =
            first + static_cast<int>(numerator / std::max(1, count - 1));
        result.insert((position / BLOCK_SIZE) * BLOCK_SIZE);
    }
    return result;
}

bool validate_boundaries(const std::set<int> & boundaries, int token_count) {
    return !boundaries.empty() && std::all_of(
        boundaries.begin(), boundaries.end(),
        [token_count](int position) {
            return position >= BLOCK_SIZE &&
                   position % BLOCK_SIZE == 0 &&
                   position + BLOCK_SIZE <= token_count;
        });
}

std::string model_fingerprint(const std::filesystem::path & path) {
    std::error_code error;
    const uintmax_t bytes = std::filesystem::file_size(path, error);
    if (error) return path.filename().string();
    return path.filename().string() + ":" + std::to_string(bytes);
}

}  // namespace

int main(int argc, char ** argv) {
    if (argc < 4 || argc > 5) {
        std::fprintf(
            stderr,
            "usage: %s <model.gguf> <tokens.i32> <output-dir> "
            "[boundaries.txt]\n",
            argv[0]);
        return 2;
    }

    std::vector<int32_t> tokens;
    if (!read_tokens(argv[2], tokens)) {
        std::fprintf(stderr, "failed to read token file: %s\n", argv[2]);
        return 1;
    }
    std::set<int> boundaries =
        argc == 5 ? std::set<int>{}
                  : default_boundaries(static_cast<int>(tokens.size()));
    if (argc == 5 && !read_boundaries(argv[4], boundaries)) {
        std::fprintf(stderr, "failed to read boundary file: %s\n", argv[4]);
        return 1;
    }
    if (!validate_boundaries(boundaries, static_cast<int>(tokens.size()))) {
        std::fprintf(
            stderr,
            "boundaries must be positive multiples of 64 with a full "
            "64-token future window\n");
        return 1;
    }

    ggml_backend_t backend = ggml_backend_cuda_init(0);
    if (!backend) {
        std::fprintf(stderr, "CUDA backend initialization failed\n");
        return 1;
    }

    TargetWeights weights;
    TargetCache cache;
    Qwen35LsaTeacherStep step;
    Qwen35LsaRawWriter writer;
    int result = 1;
    std::string error;

    if (!load_target_gguf(argv[1], backend, weights)) {
        std::fprintf(stderr, "model load failed: %s\n", dflash27b_last_error());
        goto cleanup;
    }
    if (!create_target_cache(
            weights, static_cast<int>(tokens.size()), 0, backend, cache,
            /*prefill_only=*/true)) {
        std::fprintf(stderr, "cache creation failed: %s\n",
                     dflash27b_last_error());
        goto cleanup;
    }

    {
        const Qwen35LsaCaptureConfig capture =
            qwen35_lsa_capture_config(weights);
        Qwen35LsaRawWriterConfig writer_config;
        writer_config.output_dir = argv[3];
        writer_config.model_fingerprint = model_fingerprint(argv[1]);
        writer_config.hidden_size = weights.n_embd;
        writer_config.kv_heads = weights.n_head_kv;
        writer_config.query_heads = weights.n_head;
        writer_config.head_dim = weights.n_embd_head_k;
        writer_config.block_size = BLOCK_SIZE;
        writer_config.lookahead_horizon = BLOCK_SIZE;
        const auto key_layer = std::find_if(
            capture.qk_layers.begin(), capture.qk_layers.end(),
            [&capture](int layer) {
                return layer > capture.hidden_layer;
            });
        writer_config.key_layer =
            key_layer == capture.qk_layers.end()
                ? capture.qk_layers.back()
                : *key_layer;
        writer_config.oracle_layers = capture.qk_layers;
        if (!writer.open(writer_config, error)) {
            std::fprintf(stderr, "raw writer: %s\n", error.c_str());
            goto cleanup;
        }

        for (int start = 0; start < static_cast<int>(tokens.size());
             start += BLOCK_SIZE) {
            const int count = std::min(
                BLOCK_SIZE, static_cast<int>(tokens.size()) - start);
            if (!build_qwen35_lsa_teacher_step(
                    step, weights, cache, backend, start, count, capture,
                    /*kq_stride_pad=*/32, error)) {
                std::fprintf(stderr, "teacher build @%d: %s\n",
                             start, error.c_str());
                goto cleanup;
            }
            Qwen35LsaCaptureBatch batch;
            if (!execute_qwen35_lsa_teacher_step(
                    step, weights, cache, backend, tokens.data() + start,
                    batch, error)) {
                std::fprintf(stderr, "teacher compute @%d: %s\n",
                             start, error.c_str());
                goto cleanup;
            }
            if (!writer.append(
                    start, batch, boundaries.count(start) != 0, error)) {
                std::fprintf(stderr, "raw write @%d: %s\n",
                             start, error.c_str());
                goto cleanup;
            }
            destroy_qwen35_lsa_teacher_step(step);
            std::fprintf(stderr, "\rLSA teacher: %d/%zu tokens",
                         start + count, tokens.size());
            std::fflush(stderr);
        }
        std::fprintf(stderr, "\n");
    }

    if (!writer.finalize(error)) {
        std::fprintf(stderr, "raw finalize: %s\n", error.c_str());
        goto cleanup;
    }
    result = 0;

cleanup:
    destroy_qwen35_lsa_teacher_step(step);
    free_target_cache(cache);
    free_target_weights(weights);
    ggml_backend_free(backend);
    return result;
}
