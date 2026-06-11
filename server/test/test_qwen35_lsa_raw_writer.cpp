#include "qwen35/qwen35_lsa_raw_writer.h"

#include <cstdio>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <string>
#include <vector>

using namespace dflash::common;

namespace {

int failures = 0;

#define CHECK(cond)                                                            \
    do {                                                                       \
        if (!(cond)) {                                                         \
            std::fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, #cond); \
            ++failures;                                                        \
        }                                                                      \
    } while (0)

Qwen35LsaCaptureBatch make_batch(float base) {
    Qwen35LsaCaptureBatch batch;
    batch.n_tokens = 2;
    batch.hidden = {base + 1, base + 2, base + 3, base + 4};
    for (int layer : {3, 7}) {
        Qwen35LsaLayerCapture capture;
        capture.layer = layer;
        capture.k_pre_rope = {
            base + layer + 1, base + layer + 2,
            base + layer + 3, base + layer + 4};
        capture.k_post_rope = capture.k_pre_rope;
        capture.q_post_rope = {
            base + 1, base + 2, base + 3, base + 4,
            base + 5, base + 6, base + 7, base + 8};
        batch.layers.push_back(std::move(capture));
    }
    return batch;
}

void test_sparse_boundary_writer() {
    const auto directory =
        std::filesystem::temp_directory_path() / "qwen35-lsa-raw-writer-test";
    std::filesystem::remove_all(directory);

    Qwen35LsaRawWriterConfig config;
    config.output_dir = directory.string();
    config.model_fingerprint = "unit-test";
    config.hidden_size = 2;
    config.kv_heads = 1;
    config.query_heads = 2;
    config.head_dim = 2;
    config.block_size = 2;
    config.lookahead_horizon = 2;
    config.key_layer = 3;
    config.oracle_layers = {3, 7};

    Qwen35LsaRawWriter writer;
    std::string error;
    CHECK(writer.open(config, error));
    CHECK(writer.append(0, make_batch(0), false, error));
    CHECK(writer.append(2, make_batch(10), true, error));
    CHECK(writer.finalize(error));
    CHECK(!writer.is_open());

    CHECK(std::filesystem::file_size(
              directory / "chunk_tokens.i32") == 2 * sizeof(int32_t));
    CHECK(std::filesystem::file_size(
              directory / "boundary_pos.i32") == sizeof(int32_t));
    CHECK(std::filesystem::file_size(
              directory / "query_hidden.bf16") == 2 * sizeof(uint16_t));
    CHECK(std::filesystem::file_size(
              directory / "key_pre.f16") == 8 * sizeof(uint16_t));
    CHECK(std::filesystem::file_size(
              directory / "layer_03.key_post.f16") ==
          8 * sizeof(uint16_t));
    CHECK(std::filesystem::file_size(
              directory / "layer_03.query_post.f16") ==
          8 * sizeof(uint16_t));

    int32_t boundary = -1;
    {
        std::ifstream input(
            directory / "boundary_pos.i32", std::ios::binary);
        input.read(reinterpret_cast<char *>(&boundary), sizeof(boundary));
    }
    CHECK(boundary == 2);

    std::string manifest;
    {
        std::ifstream input(directory / "manifest.json");
        manifest.assign(std::istreambuf_iterator<char>(input),
                        std::istreambuf_iterator<char>());
    }
    CHECK(manifest.find("\"tokens\": 4") != std::string::npos);
    CHECK(manifest.find("\"examples\": 1") != std::string::npos);
    CHECK(manifest.find("\"model_fingerprint\": \"unit-test\"") !=
          std::string::npos);

    std::filesystem::remove_all(directory);
}

}  // namespace

int main() {
    test_sparse_boundary_writer();
    if (failures != 0) {
        std::fprintf(stderr, "%d Qwen LSA raw-writer test(s) failed\n",
                     failures);
        return 1;
    }
    std::printf("Qwen LSA raw-writer tests passed\n");
    return 0;
}
