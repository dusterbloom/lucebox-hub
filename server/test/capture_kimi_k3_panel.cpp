#include "internal.h"
#include "kimi_k3/kimi_k3_internal.h"
#include "kimi_k3/kimi_k3_panel_artifact.h"
#include "server/tokenizer.h"

#include "ggml-cuda.h"

#include <nlohmann/json.hpp>

#include <algorithm>
#include <array>
#include <cerrno>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <limits>
#include <string>
#include <vector>

#if defined(_WIN32)
#include <process.h>
#else
#include <unistd.h>
#endif

using json = nlohmann::json;
using namespace dflash::common;

namespace {

struct InputSequence {
    std::string id;
    std::string split;
    std::string text;
};

uint8_t split_code(const std::string & split) {
    if (split == "calibration") return 0;
    if (split == "validation") return 1;
    return 2;
}

bool write_all(FILE * file, const void * data, size_t bytes) {
    return bytes == 0 || std::fwrite(data, 1, bytes, file) == bytes;
}

bool read_sequences(const std::string & path,
                    std::vector<InputSequence> & sequences,
                    std::string & error) {
    std::ifstream input(path);
    if (!input) {
        error = "cannot open sequence file " + path;
        return false;
    }
    std::string line;
    size_t line_number = 0;
    while (std::getline(input, line)) {
        ++line_number;
        if (line.empty()) continue;
        try {
            const json row = json::parse(line);
            if (!row.is_object() || !row.contains("id") ||
                !row.contains("split") || !row.contains("text") ||
                !row["id"].is_string() || !row["split"].is_string() ||
                !row["text"].is_string()) {
                error = "sequence line " + std::to_string(line_number) +
                    " must contain string id, split, and text fields";
                return false;
            }
            InputSequence sequence;
            sequence.id = row["id"].get<std::string>();
            sequence.split = row["split"].get<std::string>();
            sequence.text = row["text"].get<std::string>();
            if (sequence.id.empty() || sequence.text.empty() ||
                (sequence.split != "calibration" &&
                 sequence.split != "validation")) {
                error = "sequence line " + std::to_string(line_number) +
                    " has an empty value or unsupported split";
                return false;
            }
            sequences.push_back(std::move(sequence));
        } catch (const std::exception & exception) {
            error = "cannot parse sequence line " +
                std::to_string(line_number) + ": " + exception.what();
            return false;
        }
    }
    if (sequences.empty()) {
        error = "sequence file contains no usable rows";
        return false;
    }
    return true;
}

bool parse_positive_int(const char * raw, int & value) {
    if (!raw || !*raw) return false;
    char * end = nullptr;
    errno = 0;
    const long parsed = std::strtol(raw, &end, 10);
    if (errno != 0 || !end || *end != '\0' || parsed <= 0 ||
        parsed > std::numeric_limits<int>::max()) {
        return false;
    }
    value = static_cast<int>(parsed);
    return true;
}

bool parse_nonnegative_int(const char * raw, int & value) {
    if (!raw || !*raw) return false;
    char * end = nullptr;
    errno = 0;
    const long parsed = std::strtol(raw, &end, 10);
    if (errno != 0 || !end || *end != '\0' || parsed < 0 ||
        parsed > std::numeric_limits<int>::max()) {
        return false;
    }
    value = static_cast<int>(parsed);
    return true;
}

std::string partial_path_for(const std::string & output) {
#if defined(_WIN32)
    const int process_id = _getpid();
#else
    const int process_id = static_cast<int>(getpid());
#endif
    return output + ".partial." + std::to_string(process_id);
}

} // namespace

int main(int argc, char ** argv) {
    if (argc < 4) {
        std::fprintf(stderr,
            "usage: %s <first-model-shard.gguf> <sequences.jsonl> "
            "<capture.bin> [gpu=0] [layer=1] [total_tokens=2048] "
            "[max_context=4096] [chunk_tokens=128]\n",
            argv[0]);
        return 2;
    }

    const std::string model_path = argv[1];
    const std::string sequences_path = argv[2];
    const std::string output_path = argv[3];
    int gpu = 0;
    int model_layer = 1;
    int total_token_limit = 2048;
    int max_context = 4096;
    int chunk_tokens = 128;
    if ((argc > 4 && !parse_nonnegative_int(argv[4], gpu)) ||
        (argc > 5 && !parse_positive_int(argv[5], model_layer)) ||
        (argc > 6 && !parse_positive_int(argv[6], total_token_limit)) ||
        (argc > 7 && !parse_positive_int(argv[7], max_context)) ||
        (argc > 8 && !parse_positive_int(argv[8], chunk_tokens))) {
        std::fprintf(stderr, "[kimi-panel-capture] invalid numeric argument\n");
        return 2;
    }
    if (gpu < 0 || model_layer < 0 || chunk_tokens > max_context) {
        std::fprintf(stderr, "[kimi-panel-capture] invalid capture bounds\n");
        return 2;
    }

    std::vector<InputSequence> sequences;
    std::string error;
    if (!read_sequences(sequences_path, sequences, error)) {
        std::fprintf(stderr, "[kimi-panel-capture] %s\n", error.c_str());
        return 1;
    }

    Tokenizer tokenizer;
    if (!tokenizer.load_from_gguf(model_path.c_str())) {
        std::fprintf(stderr, "[kimi-panel-capture] tokenizer load failed\n");
        return 1;
    }

    ggml_backend_t backend = ggml_backend_cuda_init(gpu);
    if (!backend) {
        std::fprintf(stderr,
                     "[kimi-panel-capture] graphics backend init failed\n");
        return 1;
    }

    KimiK3Weights weights;
    KimiK3LoadOptions load_options;
    load_options.stream_routed_experts = true;
    load_options.stop_before_moe_layer = model_layer;
    if (!load_kimi_k3_gguf(
            model_path, backend, weights, load_options)) {
        std::fprintf(stderr, "[kimi-panel-capture] selective load failed: %s\n",
                     dflash27b_last_error());
        ggml_backend_free(backend);
        return 1;
    }

    KimiK3Cache cache;
    if (!create_kimi_k3_cache(
            backend, weights, max_context, cache)) {
        std::fprintf(stderr, "[kimi-panel-capture] cache allocation failed\n");
        free_kimi_k3_weights(weights);
        ggml_backend_free(backend);
        return 1;
    }

    const std::string temporary_path = partial_path_for(output_path);
    FILE * output = std::fopen(temporary_path.c_str(), "wb+");
    if (!output) {
        std::fprintf(stderr, "[kimi-panel-capture] cannot create %s: %s\n",
                     temporary_path.c_str(), std::strerror(errno));
        free_kimi_k3_cache(cache);
        free_kimi_k3_weights(weights);
        ggml_backend_free(backend);
        return 1;
    }

    KimiK3PanelCaptureHeader header;
    header.model_layer = model_layer;
    header.latent_dimension = static_cast<uint32_t>(weights.n_expert_latent);
    header.top_k = static_cast<uint32_t>(weights.n_expert_used);
    bool ok = write_all(output, &header, sizeof(header));
    json index;
    index["schema"] = "kimi-k3-panel-capture-v1";
    index["model_path"] = model_path;
    index["model_layer"] = model_layer;
    index["latent_dimension"] = weights.n_expert_latent;
    index["top_k"] = weights.n_expert_used;
    index["latent_storage"] = "bfloat16";
    index["router_weight_storage"] = "float32";
    index["sequences"] = json::array();

    uint64_t written_tokens = 0;
    uint64_t written_sequences = 0;
    for (const InputSequence & sequence : sequences) {
        if (!ok || written_tokens >= static_cast<uint64_t>(total_token_limit)) {
            break;
        }
        std::vector<int32_t> tokens = tokenizer.encode(sequence.text);
        if (tokens.empty()) continue;
        const size_t remaining = static_cast<size_t>(total_token_limit) -
            static_cast<size_t>(written_tokens);
        tokens.resize(std::min({tokens.size(), remaining,
                               static_cast<size_t>(max_context)}));
        if (tokens.empty()) continue;

        reset_kimi_k3_cache(cache);
        std::vector<ggml_bf16_t> latent;
        std::vector<int32_t> expert_ids;
        std::vector<float> router_weights;
        latent.reserve(tokens.size() * static_cast<size_t>(weights.n_expert_latent));
        expert_ids.reserve(tokens.size() * static_cast<size_t>(weights.n_expert_used));
        router_weights.reserve(tokens.size() * static_cast<size_t>(weights.n_expert_used));

        for (size_t begin = 0; ok && begin < tokens.size();
             begin += static_cast<size_t>(chunk_tokens)) {
            const size_t end = std::min(
                tokens.size(), begin + static_cast<size_t>(chunk_tokens));
            const std::vector<int32_t> chunk(
                tokens.begin() + static_cast<ptrdiff_t>(begin),
                tokens.begin() + static_cast<ptrdiff_t>(end));
            KimiK3MoePanelCapture capture;
            KimiK3ForwardOptions forward_options;
            forward_options.read_logits = false;
            forward_options.read_argmax = false;
            forward_options.stop_before_moe_layer = model_layer;
            forward_options.panel_capture = &capture;
            KimiK3ForwardResult result;
            ok = kimi_k3_forward(
                backend, weights, cache, chunk, static_cast<int>(begin),
                forward_options, result);
            if (!ok) {
                error = dflash27b_last_error();
                break;
            }
            const size_t latent_values = chunk.size() *
                static_cast<size_t>(weights.n_expert_latent);
            const size_t route_values = chunk.size() *
                static_cast<size_t>(weights.n_expert_used);
            if (capture.layer != model_layer ||
                capture.base_pos != static_cast<int>(begin) ||
                capture.n_tokens != static_cast<int>(chunk.size()) ||
                capture.latent_dimension != weights.n_expert_latent ||
                capture.top_k != weights.n_expert_used ||
                capture.latent.size() != latent_values ||
                capture.expert_ids.size() != route_values ||
                capture.router_weights.size() != route_values) {
                ok = false;
                error = "capture returned an inconsistent shape";
                break;
            }
            const size_t old_latent_size = latent.size();
            latent.resize(old_latent_size + latent_values);
            ggml_fp32_to_bf16_row(
                capture.latent.data(), latent.data() + old_latent_size,
                static_cast<int64_t>(latent_values));
            expert_ids.insert(expert_ids.end(),
                              capture.expert_ids.begin(),
                              capture.expert_ids.end());
            router_weights.insert(router_weights.end(),
                                  capture.router_weights.begin(),
                                  capture.router_weights.end());
        }
        if (!ok) break;

        const uint32_t id_bytes = static_cast<uint32_t>(sequence.id.size());
        const uint8_t split = split_code(sequence.split);
        const std::array<uint8_t, 3> record_reserved{};
        const uint32_t token_count = static_cast<uint32_t>(tokens.size());
        ok = write_all(output, &id_bytes, sizeof(id_bytes)) &&
             write_all(output, &split, sizeof(split)) &&
             write_all(output, record_reserved.data(), record_reserved.size()) &&
             write_all(output, &token_count, sizeof(token_count)) &&
             write_all(output, sequence.id.data(), sequence.id.size()) &&
             write_all(output, tokens.data(), tokens.size() * sizeof(int32_t)) &&
             write_all(output, latent.data(),
                       latent.size() * sizeof(ggml_bf16_t)) &&
             write_all(output, expert_ids.data(),
                       expert_ids.size() * sizeof(int32_t)) &&
             write_all(output, router_weights.data(),
                       router_weights.size() * sizeof(float));
        if (!ok) {
            error = "failed while writing capture record";
            break;
        }

        written_tokens += tokens.size();
        ++written_sequences;
        index["sequences"].push_back({
            {"id", sequence.id},
            {"split", sequence.split},
            {"tokens", tokens.size()},
        });
        std::fprintf(stderr,
            "[kimi-panel-capture] sequence=%s split=%s tokens=%zu "
            "total=%llu/%d\n",
            sequence.id.c_str(), sequence.split.c_str(), tokens.size(),
            static_cast<unsigned long long>(written_tokens), total_token_limit);
    }

    header.sequence_count = written_sequences;
    header.token_count = written_tokens;
    if (ok && std::fseek(output, 0, SEEK_SET) == 0) {
        ok = write_all(output, &header, sizeof(header));
    } else {
        ok = false;
    }
    if (std::fflush(output) != 0) ok = false;
#if !defined(_WIN32)
    if (ok && fsync(fileno(output)) != 0) ok = false;
#endif
    if (std::fclose(output) != 0) ok = false;

    if (ok && written_sequences > 0 && written_tokens > 0) {
        if (std::rename(temporary_path.c_str(), output_path.c_str()) != 0) {
            ok = false;
            error = "cannot publish capture: " +
                std::string(std::strerror(errno));
        }
    } else if (error.empty()) {
        error = "capture produced no records";
    }

    if (ok) {
        index["sequence_count"] = written_sequences;
        index["token_count"] = written_tokens;
        index["capture_path"] = output_path;
        std::ofstream sidecar(output_path + ".json");
        if (!sidecar) {
            ok = false;
            error = "cannot create capture index";
        } else {
            sidecar << index.dump(2) << '\n';
            ok = static_cast<bool>(sidecar);
            if (!ok) error = "cannot write capture index";
        }
    }

    if (!ok) {
        (void) std::remove(temporary_path.c_str());
        std::fprintf(stderr, "[kimi-panel-capture] failed: %s\n",
                     error.c_str());
    } else {
        std::fprintf(stderr,
            "[kimi-panel-capture] wrote %llu sequences, %llu tokens to %s\n",
            static_cast<unsigned long long>(written_sequences),
            static_cast<unsigned long long>(written_tokens),
            output_path.c_str());
    }

    free_kimi_k3_cache(cache);
    free_kimi_k3_weights(weights);
    ggml_backend_free(backend);
    return ok ? 0 : 1;
}
