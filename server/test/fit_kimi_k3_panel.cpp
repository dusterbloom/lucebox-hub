#include "internal.h"
#include "kimi_k3/kimi_k3_internal.h"
#include "kimi_k3/kimi_k3_panel_artifact.h"
#include "kimi_k3/kimi_k3_panel_fit.h"
#include "common/moe_hybrid_stream.h"

#include "ggml-alloc.h"
#include "ggml-cuda.h"

#include <nlohmann/json.hpp>

#include <algorithm>
#include <array>
#include <cerrno>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <functional>
#include <limits>
#include <numeric>
#include <string>
#include <vector>

#if defined(_WIN32)
#include <fcntl.h>
#include <io.h>
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

constexpr std::array<char, 8> kStatsMagic = {
    'K', '3', 'S', 'T', 'A', '0', '0', '1'};
constexpr std::array<char, 8> kAggregateMagic = {
    'K', '3', 'A', 'G', 'G', '0', '0', '1'};
constexpr std::array<char, 8> kPanelMagic = {
    'K', '3', 'F', 'I', 'T', '0', '0', '1'};
constexpr uint32_t kFitVersion = 1;
constexpr int kAggregateCheckpointExperts = 16;

struct RouteReference {
    uint32_t record = 0;
    uint32_t token = 0;
    float router_weight = 0.0f;
};

struct ExpertStatsHeader {
    std::array<char, 8> magic = kStatsMagic;
    uint32_t version = kFitVersion;
    int32_t expert = -1;
    uint32_t dimension = 0;
    uint32_t reserved = 0;
    uint64_t calibration_hits = 0;
    uint64_t validation_hits = 0;
    double unweighted_s0 = 0.0;
    double weighted_s0 = 0.0;
};

struct AggregateHeader {
    std::array<char, 8> magic = kAggregateMagic;
    uint32_t version = kFitVersion;
    int32_t completed_expert = -1;
    uint64_t validation_tokens = 0;
    uint32_t dimension = 0;
    uint32_t reserved = 0;
};

struct PanelHeader {
    std::array<char, 8> magic = kPanelMagic;
    uint32_t version = kFitVersion;
    int32_t model_layer = -1;
    uint32_t expert_count = 0;
    uint32_t dimension = 0;
    uint32_t arrays = 5;
    uint32_t reserved = 0;
};
static_assert(sizeof(PanelHeader) == 32,
              "panel fit header must remain byte-stable");

struct ExpertStatsPair {
    ExpertStatsHeader header;
    KimiK3DiagonalStats unweighted;
    KimiK3DiagonalStats weighted;
};

struct MetricValues {
    std::vector<double> cosine;
    std::vector<double> relative_l2;
};

struct PanelArrays {
    std::vector<float> codeword;
    std::vector<float> unweighted_offset;
    std::vector<float> unweighted_gain;
    std::vector<float> weighted_offset;
    std::vector<float> weighted_gain;
};

int process_id() {
#if defined(_WIN32)
    return _getpid();
#else
    return static_cast<int>(getpid());
#endif
}

void close_descriptor(int descriptor) {
#if defined(_WIN32)
    _close(descriptor);
#else
    close(descriptor);
#endif
}

bool sync_file(const std::string & path) {
#if defined(_WIN32)
    const int descriptor = _open(path.c_str(), _O_RDONLY | _O_BINARY);
    if (descriptor < 0) return false;
    const bool ok = _commit(descriptor) == 0;
#else
    const int descriptor = open(path.c_str(), O_RDONLY | O_CLOEXEC);
    if (descriptor < 0) return false;
    const bool ok = fsync(descriptor) == 0;
#endif
    close_descriptor(descriptor);
    return ok;
}

bool atomic_binary_write(
        const std::string & path,
        const std::function<bool(std::ofstream &)> & writer,
        std::string & error) {
    const std::string temporary = path + ".partial." +
        std::to_string(process_id());
    {
        std::ofstream output(temporary, std::ios::binary | std::ios::trunc);
        if (!output || !writer(output)) {
            error = "cannot write " + temporary;
            (void) std::remove(temporary.c_str());
            return false;
        }
        output.flush();
        if (!output) {
            error = "cannot flush " + temporary;
            (void) std::remove(temporary.c_str());
            return false;
        }
    }
    if (!sync_file(temporary)) {
        error = "cannot synchronize " + temporary;
        (void) std::remove(temporary.c_str());
        return false;
    }
    if (std::rename(temporary.c_str(), path.c_str()) != 0) {
        error = "cannot publish " + path + ": " + std::strerror(errno);
        (void) std::remove(temporary.c_str());
        return false;
    }
    return true;
}

template <typename T>
bool write_value(std::ofstream & output, const T & value) {
    output.write(reinterpret_cast<const char *>(&value), sizeof(value));
    return static_cast<bool>(output);
}

template <typename T>
bool write_vector(std::ofstream & output, const std::vector<T> & values) {
    output.write(reinterpret_cast<const char *>(values.data()),
                 static_cast<std::streamsize>(values.size() * sizeof(T)));
    return static_cast<bool>(output);
}

template <typename T>
bool read_value(std::ifstream & input, T & value) {
    return static_cast<bool>(input.read(
        reinterpret_cast<char *>(&value), sizeof(value)));
}

template <typename T>
bool read_vector(std::ifstream & input, std::vector<T> & values, size_t count) {
    values.resize(count);
    return count == 0 || static_cast<bool>(input.read(
        reinterpret_cast<char *>(values.data()),
        static_cast<std::streamsize>(count * sizeof(T))));
}

bool write_stats(const std::string & path,
                 const ExpertStatsPair & pair,
                 std::string & error) {
    return atomic_binary_write(path, [&](std::ofstream & output) {
        return write_value(output, pair.header) &&
               write_vector(output, pair.unweighted.sx) &&
               write_vector(output, pair.unweighted.sxx) &&
               write_vector(output, pair.unweighted.sy) &&
               write_vector(output, pair.unweighted.sxy) &&
               write_vector(output, pair.weighted.sx) &&
               write_vector(output, pair.weighted.sxx) &&
               write_vector(output, pair.weighted.sy) &&
               write_vector(output, pair.weighted.sxy);
    }, error);
}

bool read_stats(const std::string & path,
                int expected_expert,
                size_t expected_dimension,
                ExpertStatsPair & pair,
                std::string & error) {
    std::ifstream input(path, std::ios::binary);
    if (!input || !read_value(input, pair.header) ||
        pair.header.magic != kStatsMagic ||
        pair.header.version != kFitVersion ||
        pair.header.expert != expected_expert ||
        pair.header.dimension != expected_dimension) {
        error = "invalid expert fit state " + path;
        return false;
    }
    pair.unweighted.reset(expected_dimension);
    pair.weighted.reset(expected_dimension);
    pair.unweighted.s0 = pair.header.unweighted_s0;
    pair.weighted.s0 = pair.header.weighted_s0;
    pair.unweighted.observations = pair.header.calibration_hits;
    pair.weighted.observations = pair.header.calibration_hits;
    if (!read_vector(input, pair.unweighted.sx, expected_dimension) ||
        !read_vector(input, pair.unweighted.sxx, expected_dimension) ||
        !read_vector(input, pair.unweighted.sy, expected_dimension) ||
        !read_vector(input, pair.unweighted.sxy, expected_dimension) ||
        !read_vector(input, pair.weighted.sx, expected_dimension) ||
        !read_vector(input, pair.weighted.sxx, expected_dimension) ||
        !read_vector(input, pair.weighted.sy, expected_dimension) ||
        !read_vector(input, pair.weighted.sxy, expected_dimension) ||
        input.peek() != std::ifstream::traits_type::eof()) {
        error = "truncated or extended expert fit state " + path;
        return false;
    }
    return true;
}

bool write_aggregate_state(const std::string & path,
                           int completed_expert,
                           uint64_t validation_tokens,
                           uint32_t dimension,
                           const std::vector<float> & aggregate,
                           std::string & error) {
    AggregateHeader header;
    header.completed_expert = completed_expert;
    header.validation_tokens = validation_tokens;
    header.dimension = dimension;
    return atomic_binary_write(path, [&](std::ofstream & output) {
        return write_value(output, header) && write_vector(output, aggregate);
    }, error);
}

bool read_aggregate_state(const std::string & path,
                          uint64_t validation_tokens,
                          uint32_t dimension,
                          int & completed_expert,
                          std::vector<float> & aggregate,
                          std::string & error) {
    std::ifstream input(path, std::ios::binary);
    AggregateHeader header;
    const size_t values = static_cast<size_t>(validation_tokens) * dimension;
    if (!input || !read_value(input, header) ||
        header.magic != kAggregateMagic ||
        header.version != kFitVersion ||
        header.validation_tokens != validation_tokens ||
        header.dimension != dimension ||
        !read_vector(input, aggregate, values) ||
        input.peek() != std::ifstream::traits_type::eof()) {
        error = "invalid exact validation aggregate state " + path;
        return false;
    }
    completed_expert = header.completed_expert;
    return true;
}

std::string expert_state_path(const std::filesystem::path & directory,
                              int expert) {
    char name[64];
    std::snprintf(name, sizeof(name), "expert_%04d.stats", expert);
    return (directory / name).string();
}

bool parse_nonnegative_int(const char * raw, int & value) {
    if (!raw || !*raw) return false;
    char * end = nullptr;
    errno = 0;
    const long parsed = std::strtol(raw, &end, 10);
    if (errno != 0 || !end || *end != '\0' || parsed < 0 ||
        parsed > std::numeric_limits<int>::max()) return false;
    value = static_cast<int>(parsed);
    return true;
}

bool parse_positive_int(const char * raw, int & value) {
    return parse_nonnegative_int(raw, value) && value > 0;
}

json summarize_values(std::vector<double> values) {
    if (values.empty()) return nullptr;
    std::sort(values.begin(), values.end());
    const auto quantile = [&](double probability) {
        const double position = probability * (values.size() - 1);
        const size_t lower = static_cast<size_t>(std::floor(position));
        const size_t upper = static_cast<size_t>(std::ceil(position));
        const double fraction = position - lower;
        return values[lower] * (1.0 - fraction) + values[upper] * fraction;
    };
    const double mean = std::accumulate(
        values.begin(), values.end(), 0.0) / values.size();
    return {
        {"mean", mean},
        {"median", quantile(0.5)},
        {"p01", quantile(0.01)},
        {"p05", quantile(0.05)},
        {"min", values.front()},
        {"max", values.back()},
    };
}

void append_pair_metric(const float * exact,
                        const float * approximate,
                        size_t dimension,
                        MetricValues & metrics) {
    double dot = 0.0;
    double exact_squared = 0.0;
    double approximate_squared = 0.0;
    double difference_squared = 0.0;
    for (size_t coordinate = 0; coordinate < dimension; ++coordinate) {
        const double a = exact[coordinate];
        const double b = approximate[coordinate];
        dot += a * b;
        exact_squared += a * a;
        approximate_squared += b * b;
        const double difference = b - a;
        difference_squared += difference * difference;
    }
    const double denominator = std::sqrt(exact_squared * approximate_squared);
    metrics.cosine.push_back(denominator > 0.0 ? dot / denominator : 0.0);
    metrics.relative_l2.push_back(
        exact_squared > 0.0 ? std::sqrt(difference_squared / exact_squared)
                            : std::numeric_limits<double>::infinity());
}

json metric_summary(const MetricValues & metrics) {
    return {
        {"cosine", summarize_values(metrics.cosine)},
        {"relative_l2", summarize_values(metrics.relative_l2)},
    };
}

bool read_vector_tensor(ggml_tensor * tensor,
                        std::vector<float> & values,
                        std::string & error) {
    if (!tensor || tensor->ne[1] != 1 || tensor->ne[2] != 1 ||
        tensor->ne[3] != 1) {
        error = "routed normalization tensor is absent or not a vector";
        return false;
    }
    const size_t count = static_cast<size_t>(tensor->ne[0]);
    std::vector<uint8_t> raw(ggml_nbytes(tensor));
    ggml_backend_tensor_get(tensor, raw.data(), 0, raw.size());
    values.resize(count);
    if (tensor->type == GGML_TYPE_F32) {
        std::memcpy(values.data(), raw.data(), count * sizeof(float));
    } else if (tensor->type == GGML_TYPE_F16) {
        const auto * source = reinterpret_cast<const ggml_fp16_t *>(raw.data());
        ggml_fp16_to_fp32_row(source, values.data(), count);
    } else if (tensor->type == GGML_TYPE_BF16) {
        const auto * source = reinterpret_cast<const ggml_bf16_t *>(raw.data());
        ggml_bf16_to_fp32_row(source, values.data(), count);
    } else {
        error = "unsupported routed normalization tensor type";
        return false;
    }
    return true;
}

void apply_routed_normalization(const std::vector<float> & input,
                                const std::vector<float> & weight,
                                size_t tokens,
                                float epsilon,
                                std::vector<float> & output) {
    const size_t dimension = weight.size();
    output.resize(input.size());
    for (size_t token = 0; token < tokens; ++token) {
        const float * source = input.data() + token * dimension;
        float * destination = output.data() + token * dimension;
        double squared = 0.0;
        for (size_t coordinate = 0; coordinate < dimension; ++coordinate) {
            squared += static_cast<double>(source[coordinate]) *
                source[coordinate];
        }
        const double inverse = 1.0 / std::sqrt(
            squared / static_cast<double>(dimension) + epsilon);
        for (size_t coordinate = 0; coordinate < dimension; ++coordinate) {
            destination[coordinate] = static_cast<float>(
                source[coordinate] * inverse * weight[coordinate]);
        }
    }
}

bool measure_projected_pairs(ggml_backend_t backend,
                             ggml_tensor * projection,
                             const std::vector<float> & exact,
                             const std::vector<float> & approximate,
                             size_t tokens,
                             size_t input_dimension,
                             size_t batch_tokens,
                             MetricValues & metrics,
                             std::string & error) {
    if (!projection || projection->ne[0] !=
            static_cast<int64_t>(input_dimension) ||
        exact.size() != approximate.size() ||
        exact.size() != tokens * input_dimension) {
        error = "invalid routed up-projection measurement shape";
        return false;
    }
    const size_t output_dimension = static_cast<size_t>(projection->ne[1]);
    for (size_t begin = 0; begin < tokens; begin += batch_tokens) {
        const size_t count = std::min(batch_tokens, tokens - begin);
        std::vector<float> pair_input(input_dimension * count * 2);
        for (size_t token = 0; token < count; ++token) {
            std::memcpy(pair_input.data() + token * input_dimension,
                        exact.data() + (begin + token) * input_dimension,
                        input_dimension * sizeof(float));
            std::memcpy(pair_input.data() + (count + token) * input_dimension,
                        approximate.data() +
                            (begin + token) * input_dimension,
                        input_dimension * sizeof(float));
        }

        ggml_init_params parameters{};
        parameters.mem_size = 4ULL * 1024ULL * 1024ULL;
        parameters.no_alloc = true;
        ggml_context * context = ggml_init(parameters);
        if (!context) {
            error = "cannot allocate routed projection graph context";
            return false;
        }
        ggml_tensor * input = ggml_new_tensor_2d(
            context, GGML_TYPE_F32, input_dimension, count * 2);
        ggml_set_input(input);
        ggml_tensor * output = ggml_mul_mat(context, projection, input);
        ggml_set_output(output);
        ggml_cgraph * graph = ggml_new_graph_custom(context, 64, false);
        ggml_build_forward_expand(graph, output);
        ggml_gallocr_t allocator = ggml_gallocr_new(
            ggml_backend_get_default_buffer_type(backend));
        if (!allocator || !ggml_gallocr_alloc_graph(allocator, graph)) {
            if (allocator) ggml_gallocr_free(allocator);
            ggml_free(context);
            error = "cannot allocate routed projection graph";
            return false;
        }
        ggml_backend_tensor_set(input, pair_input.data(), 0,
                                pair_input.size() * sizeof(float));
        if (ggml_backend_graph_compute(backend, graph) !=
            GGML_STATUS_SUCCESS) {
            ggml_gallocr_free(allocator);
            ggml_free(context);
            error = "routed projection graph failed";
            return false;
        }
        std::vector<float> pair_output(output_dimension * count * 2);
        ggml_backend_tensor_get(output, pair_output.data(), 0,
                                pair_output.size() * sizeof(float));
        for (size_t token = 0; token < count; ++token) {
            append_pair_metric(
                pair_output.data() + token * output_dimension,
                pair_output.data() + (count + token) * output_dimension,
                output_dimension, metrics);
        }
        ggml_gallocr_free(allocator);
        ggml_free(context);
    }
    return true;
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

} // namespace

int main(int argc, char ** argv) {
    if (argc < 5) {
        std::fprintf(stderr,
            "usage: %s <first-model-shard.gguf> <capture.bin> "
            "<fit-state-directory> <output-prefix> [gpu=0] [batch=128]\n",
            argv[0]);
        return 2;
    }
    const std::string model_path = argv[1];
    const std::string capture_path = argv[2];
    const std::filesystem::path state_directory = argv[3];
    const std::string output_prefix = argv[4];
    int gpu = 0;
    int batch_tokens = 128;
    if ((argc > 5 && !parse_nonnegative_int(argv[5], gpu)) ||
        (argc > 6 && !parse_positive_int(argv[6], batch_tokens))) {
        std::fprintf(stderr, "[kimi-panel-fit] invalid numeric argument\n");
        return 2;
    }

    KimiK3PanelCaptureArtifact artifact;
    std::string error;
    if (!read_kimi_k3_panel_capture(capture_path, artifact, &error)) {
        std::fprintf(stderr, "[kimi-panel-fit] %s\n", error.c_str());
        return 1;
    }
    const size_t dimension = artifact.header.latent_dimension;
    const int model_layer = artifact.header.model_layer;
    uint64_t calibration_tokens = 0;
    uint64_t validation_tokens = 0;
    std::vector<std::vector<int64_t>> validation_index;
    validation_index.resize(artifact.records.size());
    for (size_t record_index = 0;
         record_index < artifact.records.size(); ++record_index) {
        const auto & record = artifact.records[record_index];
        validation_index[record_index].assign(record.tokens.size(), -1);
        if (record.split == 0) {
            calibration_tokens += record.tokens.size();
        } else {
            for (size_t token = 0; token < record.tokens.size(); ++token) {
                validation_index[record_index][token] =
                    static_cast<int64_t>(validation_tokens++);
            }
        }
    }
    if (calibration_tokens == 0 || validation_tokens == 0) {
        std::fprintf(stderr,
            "[kimi-panel-fit] capture needs calibration and validation tokens\n");
        return 1;
    }

    ggml_backend_t backend = ggml_backend_cuda_init(gpu);
    if (!backend) {
        std::fprintf(stderr, "[kimi-panel-fit] graphics backend init failed\n");
        return 1;
    }
    KimiK3Weights weights;
    KimiK3LoadOptions load_options;
    load_options.stream_routed_experts = true;
    load_options.stop_before_moe_layer = model_layer;
    if (!load_kimi_k3_gguf(model_path, backend, weights, load_options)) {
        std::fprintf(stderr, "[kimi-panel-fit] selective load failed: %s\n",
                     dflash27b_last_error());
        ggml_backend_free(backend);
        return 1;
    }
    const auto cleanup_weights = [&]() {
        free_kimi_k3_weights(weights);
        ggml_backend_free(backend);
    };
    if (weights.n_expert_latent != static_cast<int>(dimension) ||
        weights.n_expert_used != static_cast<int>(artifact.header.top_k) ||
        model_layer < weights.n_dense_lead || model_layer >= weights.n_layer) {
        std::fprintf(stderr, "[kimi-panel-fit] capture/model shape mismatch\n");
        cleanup_weights();
        return 1;
    }

    std::vector<std::vector<RouteReference>> routes(
        static_cast<size_t>(weights.n_expert));
    bool routes_valid = true;
    for (size_t record_index = 0;
         record_index < artifact.records.size(); ++record_index) {
        const auto & record = artifact.records[record_index];
        for (size_t token = 0; token < record.tokens.size(); ++token) {
            double route_sum = 0.0;
            for (int rank = 0; rank < weights.n_expert_used; ++rank) {
                const size_t route_index = token *
                    static_cast<size_t>(weights.n_expert_used) + rank;
                const int32_t expert = record.expert_ids[route_index];
                const float route_weight = record.router_weights[route_index];
                if (expert < 0 || expert >= weights.n_expert ||
                    !std::isfinite(route_weight) || route_weight < 0.0f) {
                    routes_valid = false;
                    break;
                }
                routes[static_cast<size_t>(expert)].push_back({
                    static_cast<uint32_t>(record_index),
                    static_cast<uint32_t>(token), route_weight});
                route_sum += route_weight;
            }
            if (!routes_valid || std::fabs(route_sum - 1.0) > 2.0e-3) {
                routes_valid = false;
                break;
            }
        }
        if (!routes_valid) break;
    }
    if (!routes_valid) {
        std::fprintf(stderr, "[kimi-panel-fit] invalid native route capture\n");
        cleanup_weights();
        return 1;
    }

    std::error_code filesystem_error;
    std::filesystem::create_directories(state_directory, filesystem_error);
    if (filesystem_error) {
        std::fprintf(stderr, "[kimi-panel-fit] cannot create fit state: %s\n",
                     filesystem_error.message().c_str());
        cleanup_weights();
        return 1;
    }
    const std::filesystem::path output_parent =
        std::filesystem::path(output_prefix).parent_path();
    if (!output_parent.empty()) {
        std::filesystem::create_directories(output_parent, filesystem_error);
        if (filesystem_error) {
            std::fprintf(stderr,
                "[kimi-panel-fit] cannot create result directory: %s\n",
                filesystem_error.message().c_str());
            cleanup_weights();
            return 1;
        }
    }

    std::vector<int> descriptors;
    std::vector<MoeNvmeSource> sources;
    if (!open_model_sources(weights.shard_paths, descriptors, sources, error)) {
        std::fprintf(stderr, "[kimi-panel-fit] %s\n", error.c_str());
        cleanup_weights();
        return 1;
    }
    const auto close_sources = [&]() {
        for (int descriptor : descriptors) close_descriptor(descriptor);
    };

    MoeStreamConfig stream_config = MoeStreamConfig::from_env();
    stream_config.device_cache_bytes = 0;
    stream_config.fused_decode = false;
    stream_config.cache_first_decode = false;
    stream_config.graph_cache_entries = 4;
    MoeHybridStreamEngine engine;
    if (!engine.init(backend, weights.max_streamed_expert_bytes,
                     stream_config, &error) ||
        !engine.bind_sources(sources, weights.streamed_layer_regions, &error)) {
        std::fprintf(stderr, "[kimi-panel-fit] stream engine failed: %s\n",
                     error.c_str());
        engine.destroy();
        close_sources();
        cleanup_weights();
        return 1;
    }

    MoeStreamExpertSpec specification;
    const KimiK3Layer & layer = weights.layers[static_cast<size_t>(model_layer)];
    specification.input_dim = weights.n_expert_latent;
    specification.intermediate_dim = weights.n_ff_exp;
    specification.output_dim = weights.n_expert_latent;
    specification.gate_type = layer.ffn_gate_exps->type;
    specification.up_type = layer.ffn_up_exps->type;
    specification.down_type = layer.ffn_down_exps->type;
    specification.gated_activation = MoeGatedActivation::Situ;
    specification.situ_beta = weights.situ_beta;
    specification.situ_linear_beta = weights.situ_linear_beta;

    const std::string aggregate_path =
        (state_directory / "validation_exact_aggregate.state").string();
    std::vector<float> exact_aggregate(
        static_cast<size_t>(validation_tokens) * dimension, 0.0f);
    int completed_expert = -1;
    if (std::filesystem::exists(aggregate_path) &&
        !read_aggregate_state(
            aggregate_path, validation_tokens,
            static_cast<uint32_t>(dimension), completed_expert,
            exact_aggregate, error)) {
        std::fprintf(stderr, "[kimi-panel-fit] %s\n", error.c_str());
        engine.destroy();
        close_sources();
        cleanup_weights();
        return 1;
    }
    if (completed_expert >= weights.n_expert) {
        std::fprintf(stderr, "[kimi-panel-fit] invalid completed expert state\n");
        engine.destroy();
        close_sources();
        cleanup_weights();
        return 1;
    }

    const int local_layer = model_layer - weights.n_dense_lead;
    for (int expert = completed_expert + 1;
         expert < weights.n_expert; ++expert) {
        ExpertStatsPair pair;
        pair.header.expert = expert;
        pair.header.dimension = static_cast<uint32_t>(dimension);
        pair.unweighted.reset(dimension);
        pair.weighted.reset(dimension);
        const auto & expert_routes = routes[static_cast<size_t>(expert)];
        for (size_t begin = 0; begin < expert_routes.size();
             begin += static_cast<size_t>(batch_tokens)) {
            const size_t count = std::min(
                static_cast<size_t>(batch_tokens), expert_routes.size() - begin);
            std::vector<float> inputs(count * dimension);
            std::vector<int32_t> selected(count, expert);
            std::vector<float> unit_weights(count, 1.0f);
            for (size_t item = 0; item < count; ++item) {
                const RouteReference & reference = expert_routes[begin + item];
                const auto & record = artifact.records[reference.record];
                const ggml_bf16_t * source = record.latent.data() +
                    static_cast<size_t>(reference.token) * dimension;
                ggml_bf16_to_fp32_row(
                    source, inputs.data() + item * dimension, dimension);
            }
            MoeStreamRouteBatch route_batch;
            route_batch.layer = local_layer;
            route_batch.n_expert = weights.n_expert;
            route_batch.top_k = 1;
            route_batch.n_tokens = static_cast<int>(count);
            route_batch.inputs = inputs.data();
            route_batch.selected_ids = selected.data();
            route_batch.selected_weights = unit_weights.data();
            std::vector<float> outputs;
            if (!eval_moe_streamed_experts(
                    engine, specification, route_batch, outputs, &error) ||
                outputs.size() != count * dimension) {
                std::fprintf(stderr,
                    "[kimi-panel-fit] expert %d evaluation failed: %s\n",
                    expert, error.c_str());
                engine.destroy();
                close_sources();
                cleanup_weights();
                return 1;
            }
            for (size_t item = 0; item < count; ++item) {
                const RouteReference & reference = expert_routes[begin + item];
                const auto & record = artifact.records[reference.record];
                const float * input = inputs.data() + item * dimension;
                const float * output = outputs.data() + item * dimension;
                if (record.split == 0) {
                    if (!pair.unweighted.observe(
                            input, output, dimension, 1.0, &error) ||
                        !pair.weighted.observe(
                            input, output, dimension,
                            static_cast<double>(reference.router_weight) *
                                reference.router_weight,
                            &error)) {
                        std::fprintf(stderr,
                            "[kimi-panel-fit] expert %d statistics failed: %s\n",
                            expert, error.c_str());
                        engine.destroy();
                        close_sources();
                        cleanup_weights();
                        return 1;
                    }
                    ++pair.header.calibration_hits;
                } else {
                    ++pair.header.validation_hits;
                    const int64_t validation_token =
                        validation_index[reference.record][reference.token];
                    if (validation_token < 0) {
                        std::fprintf(stderr,
                            "[kimi-panel-fit] validation index is inconsistent\n");
                        engine.destroy();
                        close_sources();
                        cleanup_weights();
                        return 1;
                    }
                    float * aggregate = exact_aggregate.data() +
                        static_cast<size_t>(validation_token) * dimension;
                    for (size_t coordinate = 0;
                         coordinate < dimension; ++coordinate) {
                        aggregate[coordinate] +=
                            reference.router_weight * output[coordinate];
                    }
                }
            }
        }
        pair.header.unweighted_s0 = pair.unweighted.s0;
        pair.header.weighted_s0 = pair.weighted.s0;
        if (!write_stats(
                expert_state_path(state_directory, expert), pair, error)) {
            std::fprintf(stderr, "[kimi-panel-fit] %s\n", error.c_str());
            engine.destroy();
            close_sources();
            cleanup_weights();
            return 1;
        }
        if ((expert + 1) % kAggregateCheckpointExperts == 0 ||
            expert + 1 == weights.n_expert) {
            if (!write_aggregate_state(
                    aggregate_path, expert, validation_tokens,
                    static_cast<uint32_t>(dimension), exact_aggregate, error)) {
                std::fprintf(stderr, "[kimi-panel-fit] %s\n", error.c_str());
                engine.destroy();
                close_sources();
                cleanup_weights();
                return 1;
            }
        }
        std::fprintf(stderr,
            "[kimi-panel-fit] expert=%d/%d calibration=%llu validation=%llu\n",
            expert + 1, weights.n_expert,
            static_cast<unsigned long long>(pair.header.calibration_hits),
            static_cast<unsigned long long>(pair.header.validation_hits));
    }

    const MoeNvmeStats io_stats = engine.io_stats();
    const MoeStreamComputeStats compute_stats = engine.compute_stats();
    const std::string io_backend_name = engine.io_backend_name();
    engine.destroy();
    close_sources();

    const size_t panel_values =
        static_cast<size_t>(weights.n_expert) * dimension;
    PanelArrays panels;
    panels.codeword.resize(panel_values);
    panels.unweighted_offset.resize(panel_values);
    panels.unweighted_gain.resize(panel_values);
    panels.weighted_offset.resize(panel_values);
    panels.weighted_gain.resize(panel_values);
    std::vector<uint64_t> calibration_hits(weights.n_expert);
    std::vector<uint64_t> validation_hits(weights.n_expert);
    uint64_t missing_experts = 0;
    uint64_t unweighted_degenerate = 0;
    uint64_t weighted_degenerate = 0;
    std::ofstream expert_csv(output_prefix + ".csv");
    expert_csv << "expert,calibration_hits,validation_hits,"
                  "unweighted_degenerate,weighted_degenerate\n";
    for (int expert = 0; expert < weights.n_expert; ++expert) {
        ExpertStatsPair pair;
        if (!read_stats(expert_state_path(state_directory, expert),
                        expert, dimension, pair, error)) {
            std::fprintf(stderr, "[kimi-panel-fit] %s\n", error.c_str());
            cleanup_weights();
            return 1;
        }
        calibration_hits[expert] = pair.header.calibration_hits;
        validation_hits[expert] = pair.header.validation_hits;
        const size_t offset = static_cast<size_t>(expert) * dimension;
        if (pair.header.calibration_hits == 0) {
            ++missing_experts;
            expert_csv << expert << ",0," << pair.header.validation_hits
                       << ',' << dimension << ',' << dimension << '\n';
            continue;
        }
        KimiK3DiagonalPanel unweighted_panel;
        KimiK3DiagonalPanel weighted_panel;
        if (!fit_kimi_k3_diagonal_panel(
                pair.unweighted, unweighted_panel, &error) ||
            !fit_kimi_k3_diagonal_panel(
                pair.weighted, weighted_panel, &error)) {
            std::fprintf(stderr,
                "[kimi-panel-fit] expert %d fit failed: %s\n",
                expert, error.c_str());
            cleanup_weights();
            return 1;
        }
        unweighted_degenerate += unweighted_panel.degenerate_coordinates;
        weighted_degenerate += weighted_panel.degenerate_coordinates;
        for (size_t coordinate = 0; coordinate < dimension; ++coordinate) {
            panels.codeword[offset + coordinate] = static_cast<float>(
                pair.unweighted.sy[coordinate] / pair.unweighted.s0);
            panels.unweighted_offset[offset + coordinate] =
                unweighted_panel.offset[coordinate];
            panels.unweighted_gain[offset + coordinate] =
                unweighted_panel.gain[coordinate];
            panels.weighted_offset[offset + coordinate] =
                weighted_panel.offset[coordinate];
            panels.weighted_gain[offset + coordinate] =
                weighted_panel.gain[coordinate];
        }
        expert_csv << expert << ',' << pair.header.calibration_hits << ','
                   << pair.header.validation_hits << ','
                   << unweighted_panel.degenerate_coordinates << ','
                   << weighted_panel.degenerate_coordinates << '\n';
    }
    expert_csv.close();
    if (!expert_csv) {
        std::fprintf(stderr, "[kimi-panel-fit] cannot write expert CSV\n");
        cleanup_weights();
        return 1;
    }

    PanelHeader panel_header;
    panel_header.model_layer = model_layer;
    panel_header.expert_count = weights.n_expert;
    panel_header.dimension = static_cast<uint32_t>(dimension);
    if (!atomic_binary_write(output_prefix + ".panel.f32",
            [&](std::ofstream & output) {
                return write_value(output, panel_header) &&
                       write_vector(output, panels.codeword) &&
                       write_vector(output, panels.unweighted_offset) &&
                       write_vector(output, panels.unweighted_gain) &&
                       write_vector(output, panels.weighted_offset) &&
                       write_vector(output, panels.weighted_gain);
            }, error)) {
        std::fprintf(stderr, "[kimi-panel-fit] %s\n", error.c_str());
        cleanup_weights();
        return 1;
    }

    std::vector<float> normalization_weight;
    if (!read_vector_tensor(
            layer.ffn_routed_norm, normalization_weight, error) ||
        normalization_weight.size() != dimension) {
        std::fprintf(stderr, "[kimi-panel-fit] %s\n", error.c_str());
        cleanup_weights();
        return 1;
    }

    json variants = json::object();
    auto evaluate_variant = [&](
            const char * name,
            const std::vector<float> & offsets,
            const std::vector<float> * gains) -> bool {
        std::vector<float> approximate(exact_aggregate.size(), 0.0f);
        for (size_t record_index = 0;
             record_index < artifact.records.size(); ++record_index) {
            const auto & record = artifact.records[record_index];
            if (record.split != 1) continue;
            for (size_t token = 0; token < record.tokens.size(); ++token) {
                const size_t validation_token = static_cast<size_t>(
                    validation_index[record_index][token]);
                float * destination = approximate.data() +
                    validation_token * dimension;
                std::vector<float> latent(dimension);
                ggml_bf16_to_fp32_row(
                    record.latent.data() + token * dimension,
                    latent.data(), dimension);
                for (int rank = 0; rank < weights.n_expert_used; ++rank) {
                    const size_t route_index = token *
                        static_cast<size_t>(weights.n_expert_used) + rank;
                    const int expert = record.expert_ids[route_index];
                    const float route_weight = record.router_weights[route_index];
                    const size_t panel_offset =
                        static_cast<size_t>(expert) * dimension;
                    for (size_t coordinate = 0;
                         coordinate < dimension; ++coordinate) {
                        float value = offsets[panel_offset + coordinate];
                        if (gains) {
                            value += (*gains)[panel_offset + coordinate] *
                                latent[coordinate];
                        }
                        destination[coordinate] += route_weight * value;
                    }
                }
            }
        }

        MetricValues raw_metrics;
        for (size_t token = 0; token < validation_tokens; ++token) {
            append_pair_metric(
                exact_aggregate.data() + token * dimension,
                approximate.data() + token * dimension,
                dimension, raw_metrics);
        }
        std::vector<float> exact_normalized;
        std::vector<float> approximate_normalized;
        apply_routed_normalization(
            exact_aggregate, normalization_weight, validation_tokens,
            weights.rms_eps, exact_normalized);
        apply_routed_normalization(
            approximate, normalization_weight, validation_tokens,
            weights.rms_eps, approximate_normalized);
        MetricValues normalized_metrics;
        for (size_t token = 0; token < validation_tokens; ++token) {
            append_pair_metric(
                exact_normalized.data() + token * dimension,
                approximate_normalized.data() + token * dimension,
                dimension, normalized_metrics);
        }
        MetricValues projected_metrics;
        if (!measure_projected_pairs(
                backend, layer.ffn_routed_up,
                exact_normalized, approximate_normalized,
                validation_tokens, dimension, 32,
                projected_metrics, error)) {
            return false;
        }
        variants[name] = {
            {"routed_aggregate", metric_summary(raw_metrics)},
            {"post_routed_normalization", metric_summary(normalized_metrics)},
            {"post_routed_up_projection", metric_summary(projected_metrics)},
        };
        return true;
    };
    if (!evaluate_variant("codeword", panels.codeword, nullptr) ||
        !evaluate_variant("unweighted_diagonal",
                          panels.unweighted_offset,
                          &panels.unweighted_gain) ||
        !evaluate_variant("router_weighted_diagonal",
                          panels.weighted_offset,
                          &panels.weighted_gain)) {
        std::fprintf(stderr, "[kimi-panel-fit] evaluation failed: %s\n",
                     error.c_str());
        cleanup_weights();
        return 1;
    }

    const double weighted_mean =
        variants["router_weighted_diagonal"]["routed_aggregate"]
                ["cosine"]["mean"].get<double>();
    std::string verdict;
    if (missing_experts > 0) {
        verdict = "BLOCKED";
    } else if (weighted_mean >= 0.9998) {
        verdict = "GREEN";
    } else if (weighted_mean >= 0.99) {
        verdict = "YELLOW";
    } else {
        verdict = "RED";
    }

    json result = {
        {"schema", "kimi-k3-layer-panel-fit-v1"},
        {"verdict", verdict},
        {"model_path", model_path},
        {"capture_path", capture_path},
        {"model_layer", model_layer},
        {"expert_count", weights.n_expert},
        {"top_k", weights.n_expert_used},
        {"latent_dimension", dimension},
        {"calibration_tokens", calibration_tokens},
        {"validation_tokens", validation_tokens},
        {"missing_calibration_experts", missing_experts},
        {"unweighted_degenerate_coordinates", unweighted_degenerate},
        {"weighted_degenerate_coordinates", weighted_degenerate},
        {"variants", variants},
        {"storage", {
            {"io_backend", io_backend_name},
            {"read_operations", io_stats.read_ops},
            {"payload_bytes", io_stats.payload_bytes},
            {"physical_bytes", io_stats.physical_bytes},
            {"active_io_seconds", io_stats.active_io_ns / 1.0e9},
            {"errors", io_stats.errors},
            {"timeouts", io_stats.demand_timeouts},
        }},
        {"compute", {
            {"graph_builds", compute_stats.graph_builds},
            {"graph_cache_hits", compute_stats.graph_cache_hits},
            {"graph_launches", compute_stats.graph_launches},
        }},
    };
    std::ofstream json_output(output_prefix + ".json");
    json_output << result.dump(2) << '\n';
    json_output.close();
    if (!json_output) {
        std::fprintf(stderr, "[kimi-panel-fit] cannot write result JSON\n");
        cleanup_weights();
        return 1;
    }
    std::printf("VERDICT: %s\n", verdict.c_str());
    std::printf("weighted diagonal mean held-out cosine: %.9f\n",
                weighted_mean);
    cleanup_weights();
    return 0;
}
