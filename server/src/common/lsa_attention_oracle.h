#pragma once

#include "lsa_packed_kv.h"

#include <string>
#include <vector>

namespace dflash::common {

// Host reference for one grouped-query attention token. Query layout is
// [q_heads, head_dim]. Packed K/V layout is
// [head_dim, token_capacity, kv_heads], with head_dim contiguous.
bool lsa_reference_packed_attention(const std::vector<float> & query,
                                    const std::vector<float> & packed_key,
                                    const std::vector<float> & packed_value,
                                    int q_heads,
                                    int kv_heads,
                                    int head_dim,
                                    const LsaPackedPlan & plan,
                                    int query_position,
                                    std::vector<float> & output,
                                    std::string & error);

// Dense reference over all source tokens in original order. K/V use the same
// [head_dim, source_tokens, kv_heads] physical layout as TargetCache.
bool lsa_reference_dense_attention(const std::vector<float> & query,
                                   const std::vector<float> & key,
                                   const std::vector<float> & value,
                                   int q_heads,
                                   int kv_heads,
                                   int head_dim,
                                   int source_tokens,
                                   int query_position,
                                   std::vector<float> & output,
                           std::string & error);

}  // namespace dflash::common
