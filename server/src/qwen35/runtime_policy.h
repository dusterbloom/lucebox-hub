// Pure runtime-policy helpers for the Qwen3.5 target.
//
// Keeping these decisions independent of graph construction makes the
// correctness contract testable without a GPU or model weights.

#pragma once

#include <cstring>

namespace dflash::common {

struct Qwen35TargetStepPolicy {
    bool use_kv_write_rows = false;
    bool force_validity_mask = false;
};

// A step-invariant set_rows append keeps the CUDA graph topology stable, but
// it also pads the flash-attention K/V span. Every such non-paged step must
// carry a mask so padded cache rows cannot enter the softmax denominator.
inline Qwen35TargetStepPolicy qwen35_target_step_policy(
        bool no_kvpad,
        bool kvflash_mask,
        int  fa_window,
        int  n_tokens,
        bool capture,
        bool dynamic_rows) {
    Qwen35TargetStepPolicy out;
    if (no_kvpad || fa_window != 0) return out;

    out.use_kv_write_rows =
        kvflash_mask || dynamic_rows || (n_tokens == 1 && !capture);
    out.force_validity_mask = out.use_kv_write_rows && !kvflash_mask;
    return out;
}

enum class Qwen35RollbackStorage {
    f32,
    f16,
};

// F32 is the correctness default. F16 remains an explicit experimental lane
// for measuring the memory/bandwidth tradeoff, not a lossless rollback mode.
inline Qwen35RollbackStorage qwen35_rollback_storage_from_string(
        const char * value) {
    if (value && std::strcmp(value, "f16") == 0) {
        return Qwen35RollbackStorage::f16;
    }
    return Qwen35RollbackStorage::f32;
}

}  // namespace dflash::common
