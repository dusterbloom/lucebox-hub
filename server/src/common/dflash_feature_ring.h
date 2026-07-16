// dflash_feature_ring.h — DFlash draft feature ring buffer (target-agnostic).
//
// Hosts the ring buffer that mirrors target hidden-state captures on the
// draft GPU, plus the helpers that move data across the storage/input dtype
// boundary:
//   - target activation tensor → ring slot
//   - ring range → contiguous draft input tensor
//   - target BF16 feature cache tensor → ring (with dtype conversion,
//     possibly across devices)
//
// Lives in common/ so any DFlash target architecture (qwen35, gemma4,
// laguna, ...) can reuse it without depending on architecture-specific
// weight or cache structs.

#pragma once

#include "ggml.h"
#include "ggml-backend.h"

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <vector>

namespace dflash::common {

struct DraftFeatureMirror {
    ggml_context * ctx = nullptr;
    ggml_backend_buffer_t buf = nullptr;
    ggml_tensor * target_feat = nullptr; // [n_target_layers*hidden_size, cap]
    void * staging = nullptr;
    size_t staging_bytes = 0;
    int device = 0;
    int target_device = 0;
    int cap = 0;
    int n_target_layers = 0;
    int hidden_size = 0;
    ggml_type storage_type = GGML_TYPE_F32;
};

inline bool draft_feature_storage_override_supported(const char * value) {
    return !value || !value[0] ||
           std::strcmp(value, "f32") == 0 || std::strcmp(value, "F32") == 0 ||
           std::strcmp(value, "f16") == 0 || std::strcmp(value, "F16") == 0 ||
           std::strcmp(value, "bf16") == 0 || std::strcmp(value, "BF16") == 0 ||
           std::strcmp(value, "q8_0") == 0 || std::strcmp(value, "Q8_0") == 0 ||
           std::strcmp(value, "q8") == 0 || std::strcmp(value, "Q8") == 0;
}

inline ggml_type draft_feature_storage_type(
        const char * value,
        ggml_type preferred_storage_type) {
    if (!value || !value[0]) return preferred_storage_type;
    if (std::strcmp(value, "f16") == 0 || std::strcmp(value, "F16") == 0) {
        return GGML_TYPE_F16;
    }
    if (std::strcmp(value, "bf16") == 0 || std::strcmp(value, "BF16") == 0) {
        return GGML_TYPE_BF16;
    }
    if (std::strcmp(value, "q8_0") == 0 || std::strcmp(value, "Q8_0") == 0 ||
        std::strcmp(value, "q8") == 0 || std::strcmp(value, "Q8") == 0) {
        return GGML_TYPE_Q8_0;
    }
    return GGML_TYPE_F32;
}

void draft_feature_mirror_free(DraftFeatureMirror & mirror);

bool draft_feature_mirror_init(DraftFeatureMirror & mirror,
                               ggml_backend_t backend,
                               int device,
                               int target_device,
                               int cap,
                               int n_target_layers,
                               int hidden_size,
                               ggml_type preferred_storage_type = GGML_TYPE_F32);

// Check whether the mirror ring buffer can provide a contiguous view of
// ctx_len slots ending at committed. Returns true and writes slot0 (the
// starting slot in the ring buffer) on success.
bool draft_feature_mirror_can_view(const DraftFeatureMirror & mirror,
                                   int committed,
                                   int ctx_len,
                                   int & slot0);

// Copy n_tokens starting at start_pos from a target-side BF16 feature ring
// (`src_target_feat` / `src_cap`) into the draft-side mirror, converting only
// when its configured storage type differs.
bool draft_feature_mirror_sync_range(const ggml_tensor * src_target_feat,
                                     int src_cap,
                                     DraftFeatureMirror & mirror,
                                     int start_pos,
                                     int n_tokens);

// Convenience: sync the last `committed` tokens (or mirror.cap, whichever is smaller).
bool draft_feature_mirror_sync_tail(const ggml_tensor * src_target_feat,
                                    int src_cap,
                                    DraftFeatureMirror & mirror,
                                    int committed);

// ── Ring ↔ tensor copy helpers (target-agnostic) ────────────────────

// Copy one capture slice from a target layer's activation output into the
// DraftFeatureMirror ring buffer. src_device is the GPU device of act_out.
bool copy_capture_slice_to_draft_ring(
    DraftFeatureMirror & feature_ring,
    int capture_idx,
    const ggml_tensor * act_out,
    int src_device,
    int chunk_start,
    int start_pos,
    int n_tokens);

// Copy one host-side F32 capture slice into the DraftFeatureMirror ring buffer.
bool copy_host_capture_slice_to_draft_ring(
    DraftFeatureMirror & feature_ring,
    int capture_idx,
    int start_pos,
    int n_tokens,
    const float * host,
    size_t host_elems);

// Copy n_tokens rows from the DraftFeatureMirror ring buffer into a
// destination tensor (typically the draft graph's target_hidden_cat input).
bool copy_feature_ring_range_to_tensor(
    const DraftFeatureMirror & feature_ring,
    ggml_tensor * dst,
    int start_pos,
    int n_tokens);

bool copy_feature_ring_range_to_host_f32(
    const DraftFeatureMirror & feature_ring,
    int start_pos,
    int n_tokens,
    std::vector<float> & out);

bool copy_host_f32_to_feature_ring_range(
    DraftFeatureMirror & feature_ring,
    int start_pos,
    int n_tokens,
    const std::vector<float> & src);

}  // namespace dflash::common
