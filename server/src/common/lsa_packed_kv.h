#pragma once

#include "lsa_runtime.h"

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace dflash::common {

struct LsaPackedConfig {
    int token_capacity = 0;
    int sink_tokens = 64;
    int recent_tokens = 8192;
};

struct LsaPackedPlan {
    int committed_tokens = 0;
    int token_capacity = 0;
    std::vector<int> source_positions;

    int active_tokens() const {
        return static_cast<int>(source_positions.size());
    }
};

// Build the active historical-token map. Selected chunks are combined with
// forced sink/recent ranges, deduplicated, and sorted by original position.
bool build_lsa_packed_plan(const std::vector<LsaChunk> & catalog,
                           const std::vector<int> & selected_chunk_ids,
                           int committed_tokens,
                           const LsaPackedConfig & config,
                           LsaPackedPlan & out,
                           std::string & error);

// Build an F16 mask for packed keys. A key is visible when its original token
// position is not later than the query's original position. Unused fixed-
// capacity columns and padded query rows remain -inf.
bool build_lsa_packed_causal_mask(const LsaPackedPlan & plan,
                                  const std::vector<int> & query_positions,
                                  int kq_stride_pad,
                                  std::vector<uint16_t> & out,
                                  std::string & error,
                                  int kv_pad_override = 0);

// Gather a tensor whose logical layout is [head_dim, source_tokens, heads]
// into [head_dim, token_capacity, heads]. Unused packed rows are zeroed.
bool gather_lsa_token_axis(const void * source,
                           size_t source_bytes,
                           int head_dim,
                           int source_tokens,
                           int heads,
                           size_t element_size,
                           const LsaPackedPlan & plan,
                           std::vector<uint8_t> & packed,
                           std::string & error);

}  // namespace dflash::common
