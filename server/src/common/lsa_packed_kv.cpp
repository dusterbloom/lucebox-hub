#include "lsa_packed_kv.h"

#include "attn_masks.h"

#include <algorithm>
#include <cstring>
#include <limits>
#include <unordered_map>
#include <unordered_set>

namespace dflash::common {
namespace {

bool checked_mul(size_t lhs, size_t rhs, size_t & out) {
    if (lhs != 0 && rhs > std::numeric_limits<size_t>::max() / lhs) {
        return false;
    }
    out = lhs * rhs;
    return true;
}

bool validate_catalog(const std::vector<LsaChunk> & catalog,
                      std::unordered_map<int, const LsaChunk *> & by_id,
                      std::string & error) {
    int previous_end = 0;
    for (size_t i = 0; i < catalog.size(); ++i) {
        const LsaChunk & chunk = catalog[i];
        if (chunk.id < 0 || chunk.token_begin < 0 ||
            chunk.token_end <= chunk.token_begin) {
            error = "packed KV catalog contains an invalid chunk";
            return false;
        }
        if (i > 0 && chunk.token_begin < previous_end) {
            error = "packed KV catalog is not in non-overlapping token order";
            return false;
        }
        if (!by_id.emplace(chunk.id, &chunk).second) {
            error = "packed KV catalog contains a duplicate chunk id";
            return false;
        }
        previous_end = chunk.token_end;
    }
    return true;
}

}  // namespace

bool build_lsa_packed_plan(const std::vector<LsaChunk> & catalog,
                           const std::vector<int> & selected_chunk_ids,
                           int committed_tokens,
                           const LsaPackedConfig & config,
                           LsaPackedPlan & out,
                           std::string & error) {
    out = {};
    error.clear();
    if (committed_tokens < 0) {
        error = "committed token count must be non-negative";
        return false;
    }
    if (config.token_capacity <= 0 || config.sink_tokens < 0 ||
        config.recent_tokens < 0) {
        error = "packed KV capacities must be positive or zero as appropriate";
        return false;
    }

    std::unordered_map<int, const LsaChunk *> by_id;
    if (!validate_catalog(catalog, by_id, error)) return false;

    std::vector<uint8_t> selected(static_cast<size_t>(committed_tokens), 0);
    const auto mark_range = [&](int begin, int end) {
        begin = std::max(0, std::min(begin, committed_tokens));
        end = std::max(begin, std::min(end, committed_tokens));
        std::fill(selected.begin() + begin, selected.begin() + end, uint8_t{1});
    };

    mark_range(0, config.sink_tokens);
    mark_range(committed_tokens - config.recent_tokens, committed_tokens);

    std::unordered_set<int> seen;
    for (int id : selected_chunk_ids) {
        if (!seen.insert(id).second) continue;
        const auto found = by_id.find(id);
        if (found == by_id.end()) {
            error = "selected chunk id is absent from the packed KV catalog";
            return false;
        }
        mark_range(found->second->token_begin, found->second->token_end);
    }

    out.committed_tokens = committed_tokens;
    out.token_capacity = config.token_capacity;
    out.source_positions.reserve(
        std::min(committed_tokens, config.token_capacity));
    for (int position = 0; position < committed_tokens; ++position) {
        if (selected[static_cast<size_t>(position)]) {
            out.source_positions.push_back(position);
        }
    }
    if (out.active_tokens() > config.token_capacity) {
        error = "packed KV selection exceeds fixed token capacity";
        out = {};
        return false;
    }
    return true;
}

bool build_lsa_packed_causal_mask(const LsaPackedPlan & plan,
                                  const std::vector<int> & query_positions,
                                  int kq_stride_pad,
                                  std::vector<uint16_t> & out,
                                  std::string & error,
                                  int kv_pad_override) {
    out.clear();
    error.clear();
    if (plan.token_capacity <= 0 || plan.committed_tokens < 0 ||
        plan.active_tokens() > plan.token_capacity) {
        error = "packed KV plan is invalid";
        return false;
    }
    if (kq_stride_pad <= 0 || kv_pad_override < 0) {
        error = "packed KV mask alignment is invalid";
        return false;
    }
    if (!std::is_sorted(plan.source_positions.begin(),
                        plan.source_positions.end()) ||
        std::adjacent_find(plan.source_positions.begin(),
                           plan.source_positions.end()) !=
            plan.source_positions.end()) {
        error = "packed KV source positions must be sorted and unique";
        return false;
    }
    if (std::any_of(plan.source_positions.begin(), plan.source_positions.end(),
                    [&](int position) {
                        return position < 0 ||
                               position >= plan.committed_tokens;
                    })) {
        error = "packed KV source position is outside committed history";
        return false;
    }
    if (std::any_of(query_positions.begin(), query_positions.end(),
                    [](int position) { return position < 0; })) {
        error = "query positions must be non-negative";
        return false;
    }

    const int kv_pad = kv_pad_override > 0
                           ? kv_pad_override
                           : align_up(plan.token_capacity, kq_stride_pad);
    if (kv_pad < plan.token_capacity) {
        error = "packed KV mask row is smaller than token capacity";
        return false;
    }
    const int q_pad =
        align_up(static_cast<int>(query_positions.size()), KQ_MASK_PAD);
    size_t elements = 0;
    if (!checked_mul(static_cast<size_t>(kv_pad),
                     static_cast<size_t>(q_pad), elements)) {
        error = "packed KV mask size overflows";
        return false;
    }
    out.assign(elements, F16_NEG_INF);
    for (size_t query = 0; query < query_positions.size(); ++query) {
        for (size_t key = 0; key < plan.source_positions.size(); ++key) {
            if (plan.source_positions[key] <= query_positions[query]) {
                out[query * static_cast<size_t>(kv_pad) + key] = F16_ZERO;
            }
        }
    }
    return true;
}

bool gather_lsa_token_axis(const void * source,
                           size_t source_bytes,
                           int head_dim,
                           int source_tokens,
                           int heads,
                           size_t element_size,
                           const LsaPackedPlan & plan,
                           std::vector<uint8_t> & packed,
                           std::string & error) {
    packed.clear();
    error.clear();
    if (!source || head_dim <= 0 || source_tokens < 0 || heads <= 0 ||
        element_size == 0) {
        error = "packed KV source geometry is invalid";
        return false;
    }
    if (plan.token_capacity <= 0 ||
        plan.active_tokens() > plan.token_capacity) {
        error = "packed KV plan is invalid";
        return false;
    }
    if (std::any_of(plan.source_positions.begin(), plan.source_positions.end(),
                    [&](int position) {
                        return position < 0 || position >= source_tokens;
                    })) {
        error = "packed KV source position is outside the source tensor";
        return false;
    }

    size_t row_bytes = 0;
    size_t source_plane_bytes = 0;
    size_t required_source_bytes = 0;
    size_t packed_plane_bytes = 0;
    size_t packed_bytes = 0;
    if (!checked_mul(static_cast<size_t>(head_dim), element_size, row_bytes) ||
        !checked_mul(row_bytes, static_cast<size_t>(source_tokens),
                     source_plane_bytes) ||
        !checked_mul(source_plane_bytes, static_cast<size_t>(heads),
                     required_source_bytes) ||
        !checked_mul(row_bytes, static_cast<size_t>(plan.token_capacity),
                     packed_plane_bytes) ||
        !checked_mul(packed_plane_bytes, static_cast<size_t>(heads),
                     packed_bytes)) {
        error = "packed KV tensor size overflows";
        return false;
    }
    if (source_bytes < required_source_bytes) {
        error = "packed KV source buffer is smaller than its geometry";
        return false;
    }

    packed.assign(packed_bytes, uint8_t{0});
    const auto * source_bytes_ptr =
        static_cast<const uint8_t *>(source);
    for (int head = 0; head < heads; ++head) {
        const size_t source_head =
            static_cast<size_t>(head) * source_plane_bytes;
        const size_t packed_head =
            static_cast<size_t>(head) * packed_plane_bytes;
        for (size_t destination = 0;
             destination < plan.source_positions.size(); ++destination) {
            const size_t source_offset =
                source_head +
                static_cast<size_t>(plan.source_positions[destination]) *
                    row_bytes;
            const size_t destination_offset =
                packed_head + destination * row_bytes;
            std::memcpy(packed.data() + destination_offset,
                        source_bytes_ptr + source_offset, row_bytes);
        }
    }
    return true;
}

}  // namespace dflash::common
