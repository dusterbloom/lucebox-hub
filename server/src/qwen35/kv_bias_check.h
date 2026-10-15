// KVFlash AM compaction: two small pure helpers around the per-layer kv_bias.
#pragma once

#include "ggml.h"

#include <vector>

namespace luce::common {

// Whether the target cache allocates the persistent kv_bias tensors: whenever a
// KVFlash pool backs a single-sequence dense cache, including a pool equal to
// max_ctx (the pool never exceeds it). Without the tensors a compacted chunk
// would silently lose its beta.
inline bool target_cache_wants_kv_bias(bool kvflash_bias, int ctx_alloc, int max_ctx,
                                       bool paged_attention, bool multi_slot,
                                       bool bailingmoe3) {
    return kvflash_bias && ctx_alloc > 0 && ctx_alloc <= max_ctx &&
           !paged_attention && !multi_slot && !bailingmoe3;
}

// Count the flash-attention nodes of `gf` whose key bias is wrong: each node
// whose K is (a view of) attn_k[i] must carry kv_bias[i] when expect_on and no
// bias otherwise. `n_fa` receives the number of nodes bound to a cache layer, so
// a caller can also require one per attention layer.
inline int kv_bias_attachment_errors(const ggml_cgraph * gf,
                                     const std::vector<ggml_tensor *> & attn_k,
                                     const std::vector<ggml_tensor *> & kv_bias,
                                     bool expect_on, int * n_fa = nullptr) {
    int bad = 0, seen = 0;
    for (int n = 0; n < ggml_graph_n_nodes(const_cast<ggml_cgraph *>(gf)); n++) {
        const ggml_tensor * node = ggml_graph_node(const_cast<ggml_cgraph *>(gf), n);
        if (node->op != GGML_OP_FLASH_ATTN_EXT) continue;
        const ggml_tensor * k = node->src[1]->view_src ? node->src[1]->view_src : node->src[1];
        for (size_t i = 0; i < attn_k.size(); i++) {
            if (attn_k[i] != k) continue;
            seen++;
            const ggml_tensor * want = expect_on && i < kv_bias.size() ? kv_bias[i] : nullptr;
            if (ggml_flash_attn_ext_get_kv_bias(node) != want) bad++;
            break;
        }
    }
    if (n_fa) *n_fa = seen;
    return bad;
}

} // namespace luce::common
