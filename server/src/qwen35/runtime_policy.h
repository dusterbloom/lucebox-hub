// Pure runtime policy for the Qwen3.5 target.

#pragma once

#include <cstring>

namespace dflash::common {

struct Qwen35TargetStepPolicy {
    bool use_kv_write_rows = false;
    bool force_validity_mask = false;
};

constexpr Qwen35TargetStepPolicy qwen35_target_step_policy(
        bool no_kvpad,
        bool kvflash_mask,
        int fa_window,
        int n_tokens,
        bool capture,
        bool dynamic_rows) noexcept {
    if (no_kvpad || fa_window != 0) return {};

    const bool use_rows =
        kvflash_mask || dynamic_rows || (n_tokens == 1 && !capture);
    return {use_rows, use_rows && !kvflash_mask};
}

enum class Qwen35RollbackStorage {
    F32,
    F16,
};

inline Qwen35RollbackStorage qwen35_rollback_storage_from_string(
        const char * value) noexcept {
    return value && std::strcmp(value, "f16") == 0
        ? Qwen35RollbackStorage::F16
        : Qwen35RollbackStorage::F32;
}

}  // namespace dflash::common
