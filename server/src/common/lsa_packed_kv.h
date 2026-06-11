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

struct LsaTokenAxisLayout {
    int source_tokens = 0;
    int heads = 0;
    size_t row_bytes = 0;
    size_t token_stride_bytes = 0;
    size_t head_stride_bytes = 0;
};

struct LsaPackedStepPlan {
    int token_capacity = 0;
    int historical_tokens = 0;
    std::vector<int> key_positions;
    // GGML [n_tokens, n_head_kv] i64 layout: token axis is contiguous.
    std::vector<int64_t> write_rows;
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

bool build_lsa_packed_step_plan(const LsaPackedPlan & history,
                                const std::vector<int> & query_positions,
                                int kv_heads,
                                int kq_stride_pad,
                                LsaPackedStepPlan & step,
                                std::vector<uint16_t> & mask,
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

// Stride-aware form used by GGML quantized tensors. Each token row may be a
// packed quant block, and source head planes may include backend padding.
bool gather_lsa_token_rows(const void * source,
                           size_t source_bytes,
                           const LsaTokenAxisLayout & layout,
                           const LsaPackedPlan & plan,
                           std::vector<uint8_t> & packed,
                           std::string & error);

}  // namespace dflash::common
