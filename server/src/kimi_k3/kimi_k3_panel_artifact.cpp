#include "kimi_k3_panel_artifact.h"

#include <cstring>
#include <fstream>
#include <limits>
#include <type_traits>

namespace dflash::common {
namespace {

template <typename T>
bool read_value(std::ifstream & input, T & value) {
    static_assert(std::is_trivially_copyable_v<T>);
    return static_cast<bool>(input.read(
        reinterpret_cast<char *>(&value), sizeof(value)));
}

template <typename T>
bool read_vector(std::ifstream & input, std::vector<T> & values, size_t count) {
    static_assert(std::is_trivially_copyable_v<T>);
    if (count > std::numeric_limits<size_t>::max() / sizeof(T)) return false;
    values.resize(count);
    return count == 0 || static_cast<bool>(input.read(
        reinterpret_cast<char *>(values.data()),
        static_cast<std::streamsize>(count * sizeof(T))));
}

bool checked_product(size_t a, size_t b, size_t & product) {
    if (a != 0 && b > std::numeric_limits<size_t>::max() / a) return false;
    product = a * b;
    return true;
}

} // namespace

bool read_kimi_k3_panel_capture(
        const std::string & path,
        KimiK3PanelCaptureArtifact & artifact,
        std::string * error) {
    artifact = KimiK3PanelCaptureArtifact{};
    auto fail = [&](const std::string & message) {
        if (error) *error = message;
        artifact = KimiK3PanelCaptureArtifact{};
        return false;
    };

    std::ifstream input(path, std::ios::binary);
    if (!input) return fail("cannot open panel capture " + path);
    if (!read_value(input, artifact.header)) {
        return fail("panel capture header is truncated");
    }
    const KimiK3PanelCaptureHeader & header = artifact.header;
    if (header.magic != kKimiK3PanelCaptureMagic ||
        header.version != kKimiK3PanelCaptureVersion ||
        header.model_layer < 0 || header.latent_dimension == 0 ||
        header.top_k == 0 || header.sequence_count == 0 ||
        header.token_count == 0 || header.latent_storage != 1 ||
        header.route_weight_storage != 0) {
        return fail("panel capture header is invalid or unsupported");
    }
    if (header.sequence_count > std::numeric_limits<size_t>::max()) {
        return fail("panel capture sequence count is too large");
    }
    artifact.records.reserve(static_cast<size_t>(header.sequence_count));

    uint64_t observed_tokens = 0;
    for (uint64_t sequence = 0; sequence < header.sequence_count; ++sequence) {
        uint32_t id_bytes = 0;
        uint8_t split = 2;
        std::array<uint8_t, 3> reserved{};
        uint32_t token_count = 0;
        if (!read_value(input, id_bytes) || !read_value(input, split) ||
            !read_value(input, reserved) || !read_value(input, token_count)) {
            return fail("panel capture record header is truncated");
        }
        if (id_bytes == 0 || id_bytes > (1U << 20) || split > 1 ||
            token_count == 0) {
            return fail("panel capture record header is invalid");
        }

        KimiK3PanelCaptureRecord record;
        record.split = split;
        record.id.resize(id_bytes);
        if (!input.read(record.id.data(), id_bytes)) {
            return fail("panel capture sequence identifier is truncated");
        }
        size_t latent_values = 0;
        size_t route_values = 0;
        if (!checked_product(token_count, header.latent_dimension,
                             latent_values) ||
            !checked_product(token_count, header.top_k, route_values) ||
            !read_vector(input, record.tokens, token_count) ||
            !read_vector(input, record.latent, latent_values) ||
            !read_vector(input, record.expert_ids, route_values) ||
            !read_vector(input, record.router_weights, route_values)) {
            return fail("panel capture record payload is truncated or too large");
        }
        observed_tokens += token_count;
        if (observed_tokens > header.token_count) {
            return fail("panel capture record tokens exceed the header count");
        }
        artifact.records.push_back(std::move(record));
    }
    if (observed_tokens != header.token_count) {
        return fail("panel capture token count does not match its records");
    }
    if (input.peek() != std::ifstream::traits_type::eof()) {
        return fail("panel capture has unexpected trailing bytes");
    }
    return true;
}

} // namespace dflash::common
