#include "internal.h"
#include "kimi_k3/kimi_k3_backend.h"
#include "kimi_k3/kimi_k3_internal.h"
#include "kimi_k3/kimi_k3_panel_artifact.h"
#include "common/moe_hybrid_stream.h"
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
#include <filesystem>
#include <iomanip>
#include <limits>
#include <sstream>
#include <string>
#include <vector>

#if defined(_WIN32)
#include <fcntl.h>
#include <process.h>
#include <sys/stat.h>
#else
#include <fcntl.h>
#include <sys/stat.h>
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

void close_descriptor(int descriptor) {
#if defined(_WIN32)
    _close(descriptor);
#else
    close(descriptor);
#endif
}

bool open_model_sources(const std::vector<std::string> & paths,
                        std::vector<int> & descriptors,
                        std::vector<MoeNvmeSource> & sources,
                        std::string & error) {
    for (const std::string & path : paths) {
#if defined(_WIN32)
        const int descriptor = _open(path.c_str(), _O_RDONLY | _O_BINARY);
        struct _stat64 status{};
        const bool stat_ok = descriptor >= 0 &&
            _fstat64(descriptor, &status) == 0 && status.st_size > 0;
#else
        const int descriptor = open(path.c_str(), O_RDONLY | O_CLOEXEC);
        struct stat status{};
        const bool stat_ok = descriptor >= 0 &&
            fstat(descriptor, &status) == 0 && status.st_size > 0;
#endif
        if (!stat_ok) {
            if (descriptor >= 0) close_descriptor(descriptor);
            error = "cannot open model source " + path;
            return false;
        }
        descriptors.push_back(descriptor);
        sources.push_back({nullptr, static_cast<size_t>(status.st_size),
                           descriptor});
    }
    return true;
}

void maybe_release_mapped_pages(const KimiK3Weights & weights) {
    const char * raw = std::getenv("DFLASH_KIMI_MMAP_DROP_PAGES");
    if (!raw || !*raw || std::strcmp(raw, "0") == 0) return;
    for (const GgufMmap & mapping : weights.mapped_shards) {
        mapping.advise_dontneed();
    }
}

struct CaptureOutput {
    int model_layer = -1;
    std::string path;
    std::string temporary_path;
    FILE * file = nullptr;
    KimiK3PanelCaptureHeader header;
    json index;
};

std::string all_layer_capture_path(const std::string & directory,
                                   int model_layer,
                                   int total_tokens) {
    std::ostringstream name;
    name << "kimi_layer" << std::setw(2) << std::setfill('0')
         << model_layer << '_' << total_tokens << ".bin";
    return (std::filesystem::path(directory) / name.str()).string();
}

bool open_capture_output(const std::string & path,
                         const std::string & model_path,
                         int model_layer,
                         const KimiK3Weights & weights,
                         CaptureOutput & output,
                         std::string & error) {
    output = CaptureOutput{};
    output.model_layer = model_layer;
    output.path = path;
    output.temporary_path = partial_path_for(path);
    output.file = std::fopen(output.temporary_path.c_str(), "wb+");
    if (!output.file) {
        error = "cannot create " + output.temporary_path + ": " +
            std::strerror(errno);
        return false;
    }
    output.header.model_layer = model_layer;
    output.header.latent_dimension =
        static_cast<uint32_t>(weights.n_expert_latent);
    output.header.top_k = static_cast<uint32_t>(weights.n_expert_used);
    if (!write_all(output.file, &output.header, sizeof(output.header))) {
        error = "cannot write capture header " + output.temporary_path;
        return false;
    }
    output.index["schema"] = "kimi-k3-panel-capture-v1";
    output.index["model_path"] = model_path;
    output.index["model_layer"] = model_layer;
    output.index["latent_dimension"] = weights.n_expert_latent;
    output.index["top_k"] = weights.n_expert_used;
    output.index["latent_storage"] = "bfloat16";
    output.index["router_weight_storage"] = "float32";
    output.index["sequences"] = json::array();
    return true;
}

bool append_capture_record(CaptureOutput & output,
                           const InputSequence & sequence,
                           const std::vector<int32_t> & tokens,
                           const std::vector<ggml_bf16_t> & latent,
                           const std::vector<int32_t> & expert_ids,
                           const std::vector<float> & router_weights,
                           std::string & error) {
    const size_t expected_latent = tokens.size() *
        static_cast<size_t>(output.header.latent_dimension);
    const size_t expected_routes = tokens.size() *
        static_cast<size_t>(output.header.top_k);
    if (!output.file || latent.size() != expected_latent ||
        expert_ids.size() != expected_routes ||
        router_weights.size() != expected_routes) {
        error = "capture record shape disagrees at model layer " +
            std::to_string(output.model_layer);
        return false;
    }
    const uint32_t id_bytes = static_cast<uint32_t>(sequence.id.size());
    const uint8_t split = split_code(sequence.split);
    const std::array<uint8_t, 3> reserved{};
    const uint32_t token_count = static_cast<uint32_t>(tokens.size());
    const bool ok =
        write_all(output.file, &id_bytes, sizeof(id_bytes)) &&
        write_all(output.file, &split, sizeof(split)) &&
        write_all(output.file, reserved.data(), reserved.size()) &&
        write_all(output.file, &token_count, sizeof(token_count)) &&
        write_all(output.file, sequence.id.data(), sequence.id.size()) &&
        write_all(output.file, tokens.data(), tokens.size() * sizeof(int32_t)) &&
        write_all(output.file, latent.data(),
                  latent.size() * sizeof(ggml_bf16_t)) &&
        write_all(output.file, expert_ids.data(),
                  expert_ids.size() * sizeof(int32_t)) &&
        write_all(output.file, router_weights.data(),
                  router_weights.size() * sizeof(float));
    if (!ok) {
        error = "failed while writing capture record at model layer " +
            std::to_string(output.model_layer);
        return false;
    }
    output.index["sequences"].push_back({
        {"id", sequence.id},
        {"split", sequence.split},
        {"tokens", tokens.size()},
    });
    return true;
}

bool finalize_capture_output(CaptureOutput & output,
                             uint64_t sequence_count,
                             uint64_t token_count,
                             std::string & error) {
    if (!output.file || sequence_count == 0 || token_count == 0) {
        error = "capture produced no records at model layer " +
            std::to_string(output.model_layer);
        return false;
    }
    output.header.sequence_count = sequence_count;
    output.header.token_count = token_count;
    bool ok = std::fseek(output.file, 0, SEEK_SET) == 0 &&
        write_all(output.file, &output.header, sizeof(output.header)) &&
        std::fflush(output.file) == 0;
#if !defined(_WIN32)
    if (ok && fsync(fileno(output.file)) != 0) ok = false;
#endif
    if (std::fclose(output.file) != 0) ok = false;
    output.file = nullptr;
    if (!ok) {
        error = "cannot finalize capture " + output.temporary_path;
        return false;
    }
    if (std::rename(output.temporary_path.c_str(), output.path.c_str()) != 0) {
        error = "cannot publish capture " + output.path + ": " +
            std::strerror(errno);
        return false;
    }
    output.index["sequence_count"] = sequence_count;
    output.index["token_count"] = token_count;
    output.index["capture_path"] = output.path;
    std::ofstream sidecar(output.path + ".json");
    if (!sidecar) {
        error = "cannot create capture index " + output.path + ".json";
        return false;
    }
    sidecar << output.index.dump(2) << '\n';
    if (!sidecar) {
        error = "cannot write capture index " + output.path + ".json";
        return false;
    }
    return true;
}

void abandon_capture_outputs(std::vector<CaptureOutput> & outputs) {
    for (CaptureOutput & output : outputs) {
        if (output.file) {
            (void)std::fclose(output.file);
            output.file = nullptr;
        }
        if (!output.temporary_path.empty()) {
            (void)std::remove(output.temporary_path.c_str());
        }
    }
}

} // namespace

int main(int argc, char ** argv) {
    if (argc < 4) {
        std::fprintf(stderr,
            "usage: %s <first-model-shard.gguf> <sequences.jsonl> "
            "<capture.bin|output-dir> [gpu=0] [layer=1|all] "
            "[total_tokens=2048] "
            "[max_context=4096] [chunk_tokens=128] "
            "[core=accelerator|cpu]\n",
            argv[0]);
        return 2;
    }

    const std::string model_path = argv[1];
    const std::string sequences_path = argv[2];
    const std::string output_path = argv[3];
    int gpu = 0;
    int model_layer = 1;
    const bool capture_all_layers =
        argc > 5 && std::strcmp(argv[5], "all") == 0;
    int total_token_limit = 2048;
    int max_context = 4096;
    int chunk_tokens = 128;
    KimiK3CorePlacement core_placement =
        KimiK3CorePlacement::Accelerator;
    if ((argc > 4 && !parse_nonnegative_int(argv[4], gpu)) ||
        (argc > 5 && !capture_all_layers &&
         !parse_positive_int(argv[5], model_layer)) ||
        (argc > 6 && !parse_positive_int(argv[6], total_token_limit)) ||
        (argc > 7 && !parse_positive_int(argv[7], max_context)) ||
        (argc > 8 && !parse_positive_int(argv[8], chunk_tokens)) ||
        (argc > 9 && !parse_kimi_k3_core_placement(
            argv[9], core_placement))) {
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

    ggml_backend_t backend = init_kimi_k3_core_backend(
        core_placement, gpu, &error);
    if (!backend) {
        std::fprintf(stderr,
                     "[kimi-panel-capture] core backend init failed: %s\n",
                     error.c_str());
        return 1;
    }

    KimiK3Weights weights;
    KimiK3LoadOptions load_options;
    load_options.stream_routed_experts = true;
    load_options.mmap_resident_tensors =
        core_placement == KimiK3CorePlacement::Cpu;
    load_options.stop_before_moe_layer =
        capture_all_layers ? -1 : model_layer;
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

    // An interior-layer capture must evaluate every preceding routed layer
    // exactly. Reuse the production file-backed evaluator so the capture stays
    // on the same numerical and storage path as ordinary Kimi inference.
    ggml_backend_t expert_backend = nullptr;
    MoeHybridStreamEngine stream_engine;
    std::vector<int> source_descriptors;
    std::vector<MoeNvmeSource> model_sources;
    if ((!capture_all_layers &&
         (model_layer < weights.n_dense_lead ||
          model_layer >= weights.n_layer)) ||
        (capture_all_layers && weights.n_dense_lead >= weights.n_layer)) {
        std::fprintf(stderr,
                     "[kimi-panel-capture] capture layer is outside routed layers\n");
        free_kimi_k3_cache(cache);
        free_kimi_k3_weights(weights);
        ggml_backend_free(backend);
        return 2;
    }
    const bool needs_streaming = capture_all_layers ||
        model_layer > weights.n_dense_lead;
    if (needs_streaming) {
        expert_backend = core_placement == KimiK3CorePlacement::Cpu
            ? ggml_backend_cuda_init(gpu) : backend;
        if (!expert_backend) {
            std::fprintf(stderr,
                         "[kimi-panel-capture] expert backend init failed\n");
            free_kimi_k3_cache(cache);
            free_kimi_k3_weights(weights);
            ggml_backend_free(backend);
            return 1;
        }
        MoeStreamConfig stream_config = MoeStreamConfig::from_env();
        if (!open_model_sources(weights.shard_paths, source_descriptors,
                                model_sources, error) ||
            !stream_engine.init(expert_backend,
                                weights.max_streamed_expert_bytes,
                                stream_config, &error) ||
            !stream_engine.bind_sources(
                model_sources, weights.streamed_layer_regions, &error)) {
            std::fprintf(stderr,
                         "[kimi-panel-capture] stream engine failed: %s\n",
                         error.c_str());
            stream_engine.destroy();
            for (int descriptor : source_descriptors) {
                close_descriptor(descriptor);
            }
            if (expert_backend != backend && expert_backend) {
                ggml_backend_free(expert_backend);
            }
            free_kimi_k3_cache(cache);
            free_kimi_k3_weights(weights);
            ggml_backend_free(backend);
            return 1;
        }
        if (capture_all_layers) {
            std::fprintf(stderr,
                "[kimi-panel-capture] exact NVMe full-model capture enabled "
                "for routed layers %d..%d\n",
                weights.n_dense_lead, weights.n_layer - 1);
        } else {
            std::fprintf(stderr,
                "[kimi-panel-capture] exact NVMe prefix enabled through layer %d\n",
                model_layer - 1);
        }
    }

    std::vector<int> capture_layers;
    if (capture_all_layers) {
        std::error_code directory_error;
        if (!std::filesystem::create_directories(output_path, directory_error) &&
            directory_error) {
            error = "cannot create all-layer output directory " + output_path +
                ": " + directory_error.message();
        }
        for (int layer = weights.n_dense_lead;
             error.empty() && layer < weights.n_layer; ++layer) {
            capture_layers.push_back(layer);
        }
    } else {
        capture_layers.push_back(model_layer);
    }

    std::vector<CaptureOutput> outputs(capture_layers.size());
    bool ok = error.empty();
    for (size_t index = 0; ok && index < capture_layers.size(); ++index) {
        const int layer = capture_layers[index];
        const std::string path = capture_all_layers
            ? all_layer_capture_path(output_path, layer, total_token_limit)
            : output_path;
        ok = open_capture_output(
            path, model_path, layer, weights, outputs[index], error);
        if (ok && capture_all_layers) {
            outputs[index].index["capture_mode"] =
                "one-pass-all-routed-layers";
        }
    }
    if (!ok) abandon_capture_outputs(outputs);

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
        std::vector<std::vector<ggml_bf16_t>> latent(outputs.size());
        std::vector<std::vector<int32_t>> expert_ids(outputs.size());
        std::vector<std::vector<float>> router_weights(outputs.size());
        for (size_t index = 0; index < outputs.size(); ++index) {
            latent[index].reserve(
                tokens.size() *
                static_cast<size_t>(weights.n_expert_latent));
            expert_ids[index].reserve(
                tokens.size() *
                static_cast<size_t>(weights.n_expert_used));
            router_weights[index].reserve(
                tokens.size() *
                static_cast<size_t>(weights.n_expert_used));
        }

        for (size_t begin = 0; ok && begin < tokens.size();
             begin += static_cast<size_t>(chunk_tokens)) {
            const size_t end = std::min(
                tokens.size(), begin + static_cast<size_t>(chunk_tokens));
            const std::vector<int32_t> chunk(
                tokens.begin() + static_cast<ptrdiff_t>(begin),
                tokens.begin() + static_cast<ptrdiff_t>(end));
            std::vector<KimiK3MoePanelCapture> captures(
                capture_all_layers ? 0 : 1);
            KimiK3ForwardOptions forward_options;
            forward_options.read_logits = false;
            forward_options.read_argmax = capture_all_layers;
            if (capture_all_layers) {
                forward_options.panel_capture_layer_ids = &capture_layers;
                forward_options.panel_captures = &captures;
            } else {
                forward_options.stop_before_moe_layer = model_layer;
                forward_options.panel_capture = &captures[0];
            }
            KimiK3ForwardResult result;
            ok = kimi_k3_forward(
                backend, weights, cache, chunk, static_cast<int>(begin),
                forward_options, result,
                needs_streaming ? &stream_engine : nullptr);
            if (!ok) {
                error = dflash27b_last_error();
                break;
            }
            const size_t latent_values = chunk.size() *
                static_cast<size_t>(weights.n_expert_latent);
            const size_t route_values = chunk.size() *
                static_cast<size_t>(weights.n_expert_used);
            if (captures.size() != outputs.size()) {
                ok = false;
                error = "capture returned the wrong layer count";
                break;
            }
            for (size_t index = 0; index < captures.size(); ++index) {
                const KimiK3MoePanelCapture & capture = captures[index];
                if (capture.layer != capture_layers[index] ||
                    capture.base_pos != static_cast<int>(begin) ||
                    capture.n_tokens != static_cast<int>(chunk.size()) ||
                    capture.latent_dimension != weights.n_expert_latent ||
                    capture.top_k != weights.n_expert_used ||
                    capture.latent.size() != latent_values ||
                    capture.expert_ids.size() != route_values ||
                    capture.router_weights.size() != route_values) {
                    ok = false;
                    error = "capture returned an inconsistent shape at layer " +
                        std::to_string(capture_layers[index]);
                    break;
                }
                const size_t old_latent_size = latent[index].size();
                latent[index].resize(old_latent_size + latent_values);
                ggml_fp32_to_bf16_row(
                    capture.latent.data(),
                    latent[index].data() + old_latent_size,
                    static_cast<int64_t>(latent_values));
                expert_ids[index].insert(
                    expert_ids[index].end(), capture.expert_ids.begin(),
                    capture.expert_ids.end());
                router_weights[index].insert(
                    router_weights[index].end(),
                    capture.router_weights.begin(),
                    capture.router_weights.end());
            }
            maybe_release_mapped_pages(weights);
        }
        if (!ok) break;

        for (size_t index = 0; ok && index < outputs.size(); ++index) {
            ok = append_capture_record(
                outputs[index], sequence, tokens, latent[index],
                expert_ids[index], router_weights[index], error);
        }
        if (!ok) break;

        written_tokens += tokens.size();
        ++written_sequences;
        std::fprintf(stderr,
            "[kimi-panel-capture] sequence=%s split=%s tokens=%zu "
            "total=%llu/%d\n",
            sequence.id.c_str(), sequence.split.c_str(), tokens.size(),
            static_cast<unsigned long long>(written_tokens), total_token_limit);
    }

    for (CaptureOutput & output : outputs) {
        if (!ok) break;
        ok = finalize_capture_output(
            output, written_sequences, written_tokens, error);
    }

    if (ok && capture_all_layers) {
        json manifest;
        manifest["schema"] = "kimi-k3-panel-multi-layer-capture-v1";
        manifest["model_path"] = model_path;
        manifest["sequence_count"] = written_sequences;
        manifest["token_count"] = written_tokens;
        manifest["first_routed_layer"] = weights.n_dense_lead;
        manifest["last_routed_layer"] = weights.n_layer - 1;
        manifest["layer_count"] = outputs.size();
        manifest["captures"] = json::array();
        for (const CaptureOutput & output : outputs) {
            manifest["captures"].push_back({
                {"model_layer", output.model_layer},
                {"path", output.path},
            });
        }
        const std::string manifest_path =
            (std::filesystem::path(output_path) /
             "all_layers_capture_manifest.json").string();
        std::ofstream manifest_file(manifest_path);
        manifest_file << manifest.dump(2) << '\n';
        if (!manifest_file) {
            ok = false;
            error = "cannot write all-layer capture manifest " + manifest_path;
        }
    }
    if (!ok) {
        abandon_capture_outputs(outputs);
        std::fprintf(stderr, "[kimi-panel-capture] failed: %s\n",
                     error.c_str());
    } else {
        std::fprintf(stderr,
            "[kimi-panel-capture] wrote %llu sequences, %llu tokens, "
            "%zu layer capture(s) to %s\n",
            static_cast<unsigned long long>(written_sequences),
            static_cast<unsigned long long>(written_tokens),
            outputs.size(),
            output_path.c_str());
    }

    stream_engine.destroy();
    for (int descriptor : source_descriptors) close_descriptor(descriptor);
    if (expert_backend != backend && expert_backend) {
        ggml_backend_free(expert_backend);
    }
    free_kimi_k3_cache(cache);
    free_kimi_k3_weights(weights);
    ggml_backend_free(backend);
    return ok ? 0 : 1;
}
