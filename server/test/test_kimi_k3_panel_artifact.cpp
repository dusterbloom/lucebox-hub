#include "kimi_k3/kimi_k3_panel_artifact.h"

#include <array>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <string>
#include <vector>

#if defined(_WIN32)
#include <process.h>
#else
#include <unistd.h>
#endif

using namespace dflash::common;

#define REQUIRE(condition) do {                                           \
    if (!(condition)) {                                                   \
        std::fprintf(stderr, "requirement failed at %s:%d: %s\n",       \
                     __FILE__, __LINE__, #condition);                     \
        std::exit(1);                                                     \
    }                                                                     \
} while (0)

template <typename T>
void write_value(std::ofstream & output, const T & value) {
    output.write(reinterpret_cast<const char *>(&value), sizeof(value));
}

template <typename T>
void write_values(std::ofstream & output, const std::vector<T> & values) {
    output.write(reinterpret_cast<const char *>(values.data()),
                 static_cast<std::streamsize>(values.size() * sizeof(T)));
}

int main() {
#if defined(_WIN32)
    const int process_id = _getpid();
#else
    const int process_id = static_cast<int>(getpid());
#endif
    const std::string path = "/tmp/kimi-panel-artifact-" +
        std::to_string(process_id) + ".bin";

    KimiK3PanelCaptureHeader header;
    header.model_layer = 1;
    header.latent_dimension = 3;
    header.top_k = 2;
    header.sequence_count = 1;
    header.token_count = 2;
    const uint32_t id_bytes = 4;
    const uint8_t split = 1;
    const std::array<uint8_t, 3> reserved{};
    const uint32_t token_count = 2;
    const std::string id = "test";
    const std::vector<int32_t> tokens = {11, 12};
    std::vector<ggml_bf16_t> latent(6);
    for (size_t i = 0; i < latent.size(); ++i) {
        latent[i] = ggml_fp32_to_bf16(static_cast<float>(i) * 0.25f);
    }
    const std::vector<int32_t> experts = {1, 2, 3, 4};
    const std::vector<float> weights = {0.6f, 0.4f, 0.7f, 0.3f};
    {
        std::ofstream output(path, std::ios::binary);
        REQUIRE(output.good());
        write_value(output, header);
        write_value(output, id_bytes);
        write_value(output, split);
        write_value(output, reserved);
        write_value(output, token_count);
        output.write(id.data(), static_cast<std::streamsize>(id.size()));
        write_values(output, tokens);
        write_values(output, latent);
        write_values(output, experts);
        write_values(output, weights);
        REQUIRE(output.good());
    }

    KimiK3PanelCaptureArtifact artifact;
    std::string error;
    REQUIRE(read_kimi_k3_panel_capture(path, artifact, &error));
    REQUIRE(artifact.header.model_layer == 1);
    REQUIRE(artifact.header.token_count == 2);
    REQUIRE(artifact.records.size() == 1);
    REQUIRE(artifact.records[0].id == id);
    REQUIRE(artifact.records[0].split == split);
    REQUIRE(artifact.records[0].tokens == tokens);
    REQUIRE(artifact.records[0].expert_ids == experts);
    REQUIRE(artifact.records[0].router_weights == weights);
    REQUIRE(artifact.records[0].latent.size() == latent.size());
    REQUIRE(std::remove(path.c_str()) == 0);
    std::printf("Kimi K3 panel artifact test passed\n");
    return 0;
}
