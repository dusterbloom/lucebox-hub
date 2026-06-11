#include "qwen35_lsa_raw_writer.h"

#include <algorithm>
#include <cstring>
#include <cstdint>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <sstream>
#include <unordered_map>
#include <utility>

namespace dflash::common {
namespace {

constexpr uint64_t FNV_OFFSET = 14695981039346656037ULL;
constexpr uint64_t FNV_PRIME = 1099511628211ULL;

std::string json_escape(const std::string & value) {
    std::string result;
    result.reserve(value.size());
    for (const unsigned char ch : value) {
        switch (ch) {
            case '"': result += "\\\""; break;
            case '\\': result += "\\\\"; break;
            case '\n': result += "\\n"; break;
            case '\r': result += "\\r"; break;
            case '\t': result += "\\t"; break;
            default:
                if (ch < 0x20) {
                    char escaped[7];
                    std::snprintf(escaped, sizeof(escaped), "\\u%04x", ch);
                    result += escaped;
                } else {
                    result.push_back(static_cast<char>(ch));
                }
        }
    }
    return result;
}

struct Stream {
    std::string name;
    FILE * file = nullptr;
    uint64_t checksum = FNV_OFFSET;
    uint64_t bytes = 0;

    Stream() = default;
    ~Stream() {
        close();
    }
    Stream(const Stream &) = delete;
    Stream & operator=(const Stream &) = delete;
    Stream(Stream && other) noexcept {
        *this = std::move(other);
    }
    Stream & operator=(Stream && other) noexcept {
        if (this != &other) {
            close();
            name = std::move(other.name);
            file = other.file;
            checksum = other.checksum;
            bytes = other.bytes;
            other.file = nullptr;
            other.checksum = FNV_OFFSET;
            other.bytes = 0;
        }
        return *this;
    }

    bool write(const void * data, size_t size) {
        if (!file || (size > 0 && std::fwrite(data, 1, size, file) != size)) {
            return false;
        }
        const auto * input = static_cast<const uint8_t *>(data);
        for (size_t i = 0; i < size; ++i) {
            checksum ^= input[i];
            checksum *= FNV_PRIME;
        }
        bytes += size;
        return true;
    }

    void close() {
        if (file) {
            std::fclose(file);
            file = nullptr;
        }
    }
};

uint16_t fp32_to_fp16_bits(float value) {
    uint32_t bits = 0;
    std::memcpy(&bits, &value, sizeof(bits));
    const uint32_t sign = (bits >> 16) & 0x8000U;
    const uint32_t exponent = (bits >> 23) & 0xffU;
    uint32_t mantissa = bits & 0x7fffffU;

    if (exponent == 0xffU) {
        return static_cast<uint16_t>(
            sign | (mantissa ? 0x7e00U : 0x7c00U));
    }

    int half_exponent = static_cast<int>(exponent) - 127 + 15;
    if (half_exponent >= 31) {
        return static_cast<uint16_t>(sign | 0x7c00U);
    }
    if (half_exponent <= 0) {
        if (half_exponent < -10) {
            return static_cast<uint16_t>(sign);
        }
        mantissa |= 0x800000U;
        const int shift = 14 - half_exponent;
        uint32_t half_mantissa = mantissa >> shift;
        const uint32_t remainder = mantissa & ((1U << shift) - 1U);
        const uint32_t halfway = 1U << (shift - 1);
        if (remainder > halfway ||
            (remainder == halfway && (half_mantissa & 1U))) {
            ++half_mantissa;
        }
        return static_cast<uint16_t>(sign | half_mantissa);
    }

    uint32_t half_mantissa = mantissa >> 13;
    const uint32_t remainder = mantissa & 0x1fffU;
    if (remainder > 0x1000U ||
        (remainder == 0x1000U && (half_mantissa & 1U))) {
        ++half_mantissa;
        if (half_mantissa == 0x400U) {
            half_mantissa = 0;
            ++half_exponent;
            if (half_exponent >= 31) {
                return static_cast<uint16_t>(sign | 0x7c00U);
            }
        }
    }
    return static_cast<uint16_t>(
        sign | (static_cast<uint32_t>(half_exponent) << 10) |
        half_mantissa);
}

uint16_t fp32_to_bf16_bits(float value) {
    uint32_t bits = 0;
    std::memcpy(&bits, &value, sizeof(bits));
    if ((bits & 0x7f800000U) == 0x7f800000U &&
        (bits & 0x007fffffU) != 0) {
        return static_cast<uint16_t>((bits >> 16) | 0x0040U);
    }
    const uint32_t rounding = 0x7fffU + ((bits >> 16) & 1U);
    return static_cast<uint16_t>((bits + rounding) >> 16);
}

bool open_stream(const std::filesystem::path & directory,
                 const std::string & name,
                 Stream & stream,
                 std::string & error) {
    stream.name = name;
    stream.file = std::fopen((directory / name).c_str(), "wb");
    if (!stream.file) {
        error = "failed to open LSA raw file: " + name;
        return false;
    }
    return true;
}

bool write_f16(Stream & stream, const std::vector<float> & values) {
    std::vector<uint16_t> converted(values.size());
    std::transform(values.begin(), values.end(), converted.begin(),
                   fp32_to_fp16_bits);
    return stream.write(converted.data(),
                        converted.size() * sizeof(uint16_t));
}

bool write_bf16(Stream & stream, const std::vector<float> & values) {
    std::vector<uint16_t> converted(values.size());
    std::transform(values.begin(), values.end(), converted.begin(),
                   fp32_to_bf16_bits);
    return stream.write(converted.data(),
                        converted.size() * sizeof(uint16_t));
}

const Qwen35LsaLayerCapture * find_layer(
    const Qwen35LsaCaptureBatch & batch, int layer) {
    const auto it = std::find_if(
        batch.layers.begin(), batch.layers.end(),
        [layer](const Qwen35LsaLayerCapture & capture) {
            return capture.layer == layer;
        });
    return it == batch.layers.end() ? nullptr : &*it;
}

}  // namespace

struct Qwen35LsaRawWriter::Impl {
    Qwen35LsaRawWriterConfig config;
    std::filesystem::path directory;
    Stream chunk_tokens;
    Stream boundary_pos;
    Stream query_hidden;
    Stream key_pre;
    std::unordered_map<int, Stream> key_post;
    std::unordered_map<int, Stream> query_post;
    std::vector<float> pending_hidden;
    int tokens = 0;
    int examples = 0;
    bool finalized = false;

    ~Impl() {
        close();
    }

    void close() {
        chunk_tokens.close();
        boundary_pos.close();
        query_hidden.close();
        key_pre.close();
        for (auto & item : key_post) item.second.close();
        for (auto & item : query_post) item.second.close();
    }

    std::vector<const Stream *> streams() const {
        std::vector<const Stream *> result = {
            &chunk_tokens, &boundary_pos, &query_hidden, &key_pre};
        for (int layer : config.oracle_layers) {
            result.push_back(&key_post.at(layer));
            result.push_back(&query_post.at(layer));
        }
        return result;
    }
};

Qwen35LsaRawWriter::Qwen35LsaRawWriter() = default;
Qwen35LsaRawWriter::~Qwen35LsaRawWriter() = default;
Qwen35LsaRawWriter::Qwen35LsaRawWriter(
    Qwen35LsaRawWriter &&) noexcept = default;
Qwen35LsaRawWriter & Qwen35LsaRawWriter::operator=(
    Qwen35LsaRawWriter &&) noexcept = default;

bool Qwen35LsaRawWriter::open(
    const Qwen35LsaRawWriterConfig & config, std::string & error) {
    impl_.reset();
    error.clear();
    if (config.output_dir.empty() || config.hidden_size <= 0 ||
        config.kv_heads <= 0 || config.query_heads <= 0 ||
        config.head_dim <= 0 || config.block_size <= 0 ||
        config.lookahead_horizon != config.block_size ||
        config.oracle_layers.empty() ||
        std::find(config.oracle_layers.begin(), config.oracle_layers.end(),
                  config.key_layer) == config.oracle_layers.end()) {
        error = "Qwen LSA raw-writer configuration is invalid";
        return false;
    }

    auto state = std::make_unique<Impl>();
    state->config = config;
    state->directory = config.output_dir;
    std::error_code fs_error;
    std::filesystem::create_directories(state->directory, fs_error);
    if (fs_error) {
        error = "failed to create LSA raw output directory";
        return false;
    }

    if (!open_stream(state->directory, "chunk_tokens.i32",
                     state->chunk_tokens, error) ||
        !open_stream(state->directory, "boundary_pos.i32",
                     state->boundary_pos, error) ||
        !open_stream(state->directory, "query_hidden.bf16",
                     state->query_hidden, error) ||
        !open_stream(state->directory, "key_pre.f16",
                     state->key_pre, error)) {
        return false;
    }
    for (int layer : config.oracle_layers) {
        Stream key;
        Stream query;
        char key_name[64];
        char query_name[64];
        std::snprintf(key_name, sizeof(key_name),
                      "layer_%02d.key_post.f16", layer);
        std::snprintf(query_name, sizeof(query_name),
                      "layer_%02d.query_post.f16", layer);
        if (!open_stream(state->directory, key_name, key, error) ||
            !open_stream(state->directory, query_name, query, error)) {
            return false;
        }
        state->key_post.emplace(layer, std::move(key));
        state->query_post.emplace(layer, std::move(query));
    }
    impl_ = std::move(state);
    return true;
}

bool Qwen35LsaRawWriter::append(
    int chunk_start,
    const Qwen35LsaCaptureBatch & batch,
    bool store_boundary_example,
    std::string & error) {
    error.clear();
    if (!impl_ || impl_->finalized) {
        error = "Qwen LSA raw writer is not open";
        return false;
    }
    const auto & config = impl_->config;
    if (chunk_start != impl_->tokens || batch.n_tokens <= 0 ||
        batch.hidden.size() !=
            static_cast<size_t>(batch.n_tokens) * config.hidden_size) {
        error = "Qwen LSA raw chunk geometry is invalid";
        return false;
    }
    if (store_boundary_example &&
        (batch.n_tokens != config.lookahead_horizon ||
         impl_->pending_hidden.empty())) {
        error = "Qwen LSA boundary example requires a full future chunk";
        return false;
    }

    const Qwen35LsaLayerCapture * key_layer =
        find_layer(batch, config.key_layer);
    if (!key_layer) {
        error = "Qwen LSA raw chunk is missing the key layer";
        return false;
    }
    const size_t key_values =
        static_cast<size_t>(batch.n_tokens) *
        config.kv_heads * config.head_dim;
    const size_t query_values =
        static_cast<size_t>(batch.n_tokens) *
        config.query_heads * config.head_dim;
    if (key_layer->k_pre_rope.size() != key_values) {
        error = "Qwen LSA raw pre-RoPE key geometry is invalid";
        return false;
    }

    for (int layer : config.oracle_layers) {
        const Qwen35LsaLayerCapture * capture = find_layer(batch, layer);
        if (!capture || capture->k_post_rope.size() != key_values ||
            capture->q_post_rope.size() != query_values) {
            error = "Qwen LSA raw Q/K geometry is invalid";
            return false;
        }
    }

    const int32_t chunk_tokens = batch.n_tokens;
    if (!impl_->chunk_tokens.write(&chunk_tokens, sizeof(chunk_tokens)) ||
        !write_f16(impl_->key_pre, key_layer->k_pre_rope)) {
        error = "failed writing Qwen LSA raw key stream";
        return false;
    }
    for (int layer : config.oracle_layers) {
        const auto * capture = find_layer(batch, layer);
        if (!write_f16(impl_->key_post.at(layer),
                       capture->k_post_rope)) {
            error = "failed writing Qwen LSA historical key stream";
            return false;
        }
    }

    if (store_boundary_example) {
        const int32_t boundary = chunk_start;
        if (!impl_->boundary_pos.write(&boundary, sizeof(boundary)) ||
            !write_bf16(impl_->query_hidden, impl_->pending_hidden)) {
            error = "failed writing Qwen LSA boundary metadata";
            return false;
        }
        for (int layer : config.oracle_layers) {
            const auto * capture = find_layer(batch, layer);
            if (!write_f16(impl_->query_post.at(layer),
                           capture->q_post_rope)) {
                error = "failed writing Qwen LSA future-query stream";
                return false;
            }
        }
        ++impl_->examples;
    }

    const size_t hidden_offset =
        static_cast<size_t>(batch.n_tokens - 1) * config.hidden_size;
    impl_->pending_hidden.assign(
        batch.hidden.begin() + hidden_offset,
        batch.hidden.begin() + hidden_offset + config.hidden_size);
    impl_->tokens += batch.n_tokens;
    return true;
}

bool Qwen35LsaRawWriter::finalize(std::string & error) {
    error.clear();
    if (!impl_ || impl_->finalized) {
        error = "Qwen LSA raw writer is not open";
        return false;
    }
    impl_->close();

    std::ofstream manifest(impl_->directory / "manifest.json",
                           std::ios::binary | std::ios::trunc);
    if (!manifest) {
        error = "failed to create Qwen LSA raw manifest";
        return false;
    }
    const auto & config = impl_->config;
    manifest << "{\n"
             << "  \"schema\": \"luce.lsa.qwen35.raw.v1\",\n"
             << "  \"endianness\": \"little\",\n"
             << "  \"checksum\": \"fnv1a64\",\n"
             << "  \"model_fingerprint\": \""
             << json_escape(config.model_fingerprint) << "\",\n"
             << "  \"tokens\": " << impl_->tokens << ",\n"
             << "  \"examples\": " << impl_->examples << ",\n"
             << "  \"hidden_size\": " << config.hidden_size << ",\n"
             << "  \"kv_heads\": " << config.kv_heads << ",\n"
             << "  \"query_heads\": " << config.query_heads << ",\n"
             << "  \"head_dim\": " << config.head_dim << ",\n"
             << "  \"block_size\": " << config.block_size << ",\n"
             << "  \"lookahead_horizon\": "
             << config.lookahead_horizon << ",\n"
             << "  \"key_layer\": " << config.key_layer << ",\n"
             << "  \"oracle_layers\": [";
    for (size_t i = 0; i < config.oracle_layers.size(); ++i) {
        if (i) manifest << ", ";
        manifest << config.oracle_layers[i];
    }
    manifest << "],\n  \"files\": {\n";
    const auto streams = impl_->streams();
    for (size_t i = 0; i < streams.size(); ++i) {
        const Stream & stream = *streams[i];
        manifest << "    \"" << stream.name << "\": {\"bytes\": "
                 << stream.bytes << ", \"fnv1a64\": \""
                 << std::hex << std::setfill('0') << std::setw(16)
                 << stream.checksum << std::dec << "\"}";
        manifest << (i + 1 == streams.size() ? "\n" : ",\n");
    }
    manifest << "  }\n}\n";
    if (!manifest) {
        error = "failed writing Qwen LSA raw manifest";
        return false;
    }
    impl_->finalized = true;
    return true;
}

bool Qwen35LsaRawWriter::is_open() const {
    return impl_ && !impl_->finalized;
}

}  // namespace dflash::common
