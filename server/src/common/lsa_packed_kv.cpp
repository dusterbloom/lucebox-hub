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

bool build_lsa_packed_step_plan(const LsaPackedPlan & history,
                                const std::vector<int> & query_positions,
                                int kv_heads,
                                int kq_stride_pad,
                                LsaPackedStepPlan & step,
                                std::vector<uint16_t> & mask,
                                std::string & error,
                                int kv_pad_override) {
    step = {};
    mask.clear();
    error.clear();
    if (history.committed_tokens < 0 || history.token_capacity <= 0 ||
        history.active_tokens() > history.token_capacity ||
        query_positions.empty() || kv_heads <= 0) {
        error = "packed KV step geometry is invalid";
        return false;
    }
    if (!std::is_sorted(history.source_positions.begin(),
                        history.source_positions.end()) ||
        std::adjacent_find(history.source_positions.begin(),
                           history.source_positions.end()) !=
            history.source_positions.end() ||
        std::any_of(history.source_positions.begin(),
                    history.source_positions.end(),
                    [&](int position) {
                        return position < 0 ||
                               position >= history.committed_tokens;
                    })) {
        error =
            "packed KV history positions must be sorted, unique, and committed";
        return false;
    }
    if (!std::is_sorted(query_positions.begin(), query_positions.end()) ||
        std::adjacent_find(query_positions.begin(), query_positions.end()) !=
            query_positions.end() ||
        query_positions.front() < history.committed_tokens ||
        query_positions.back() == std::numeric_limits<int>::max()) {
        error = "packed KV query positions must be sorted new positions";
        return false;
    }
    const size_t remaining_capacity =
        static_cast<size_t>(history.token_capacity - history.active_tokens());
    if (query_positions.size() > remaining_capacity) {
        error = "packed KV step exceeds fixed token capacity";
        return false;
    }

    step.token_capacity = history.token_capacity;
    step.historical_tokens = history.active_tokens();
    step.key_positions = history.source_positions;
    step.key_positions.insert(step.key_positions.end(),
                              query_positions.begin(),
                              query_positions.end());
    step.write_rows.reserve(
        query_positions.size() * static_cast<size_t>(kv_heads));
    for (int head = 0; head < kv_heads; ++head) {
        for (size_t token = 0; token < query_positions.size(); ++token) {
            step.write_rows.push_back(
                static_cast<int64_t>(history.active_tokens() + token));
        }
    }

    LsaPackedPlan visible;
    visible.committed_tokens =
        std::max(history.committed_tokens, query_positions.back() + 1);
    visible.token_capacity = history.token_capacity;
    visible.source_positions = step.key_positions;
    return build_lsa_packed_causal_mask(
        visible, query_positions, kq_stride_pad, mask, error,
        kv_pad_override);
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
    if (head_dim <= 0 || source_tokens < 0 || heads <= 0 ||
        element_size == 0) {
        packed.clear();
        error = "packed KV source geometry is invalid";
        return false;
    }
    size_t row_bytes = 0;
    size_t head_stride = 0;
    if (!checked_mul(static_cast<size_t>(head_dim), element_size, row_bytes) ||
        !checked_mul(row_bytes, static_cast<size_t>(source_tokens),
                     head_stride)) {
        packed.clear();
        error = "packed KV tensor size overflows";
        return false;
    }
    return gather_lsa_token_rows(
        source, source_bytes,
        {source_tokens, heads, row_bytes, row_bytes, head_stride},
        plan, packed, error);
}

bool gather_lsa_token_rows(const void * source,
                           size_t source_bytes,
                           const LsaTokenAxisLayout & layout,
                           const LsaPackedPlan & plan,
                           std::vector<uint8_t> & packed,
                           std::string & error) {
    packed.clear();
    error.clear();
    if (!source || layout.source_tokens <= 0 || layout.heads <= 0 ||
        layout.row_bytes == 0 ||
        layout.token_stride_bytes < layout.row_bytes ||
        layout.head_stride_bytes < layout.token_stride_bytes) {
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
                        return position < 0 ||
                               position >= layout.source_tokens;
                    })) {
        error = "packed KV source position is outside the source tensor";
        return false;
    }

    size_t last_head_offset = 0;
    size_t last_token_offset = 0;
    size_t packed_plane_bytes = 0;
    size_t packed_bytes = 0;
    if (!checked_mul(layout.head_stride_bytes,
                     static_cast<size_t>(layout.heads - 1),
                     last_head_offset) ||
        !checked_mul(layout.token_stride_bytes,
                     static_cast<size_t>(
                         std::max(0, layout.source_tokens - 1)),
                     last_token_offset) ||
        last_head_offset >
            std::numeric_limits<size_t>::max() - last_token_offset ||
        last_head_offset + last_token_offset >
            std::numeric_limits<size_t>::max() - layout.row_bytes ||
        !checked_mul(layout.row_bytes,
                     static_cast<size_t>(plan.token_capacity),
                     packed_plane_bytes) ||
        !checked_mul(packed_plane_bytes,
                     static_cast<size_t>(layout.heads),
                     packed_bytes)) {
        error = "packed KV tensor size overflows";
        return false;
    }
    const size_t required_source_bytes =
        last_head_offset + last_token_offset + layout.row_bytes;
    if (source_bytes < required_source_bytes) {
        error = "packed KV source buffer is smaller than its geometry";
        return false;
    }

    packed.assign(packed_bytes, uint8_t{0});
    const auto * source_bytes_ptr =
        static_cast<const uint8_t *>(source);
    for (int head = 0; head < layout.heads; ++head) {
        const size_t source_head =
            static_cast<size_t>(head) * layout.head_stride_bytes;
        const size_t packed_head =
            static_cast<size_t>(head) * packed_plane_bytes;
        for (size_t destination = 0;
             destination < plan.source_positions.size(); ++destination) {
            const size_t source_offset =
                source_head +
                static_cast<size_t>(plan.source_positions[destination]) *
                    layout.token_stride_bytes;
            const size_t destination_offset =
                packed_head + destination * layout.row_bytes;
            std::memcpy(packed.data() + destination_offset,
                        source_bytes_ptr + source_offset,
                        layout.row_bytes);
        }
    }
    return true;
}

}  // namespace dflash::common
