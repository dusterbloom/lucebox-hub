#include "qwen35/runtime_policy.h"

using dflash::common::Qwen35RollbackStorage;
using dflash::common::qwen35_rollback_storage_from_string;
using dflash::common::qwen35_target_step_policy;

int main() {
    const auto ar = qwen35_target_step_policy(
        false, false, 0, 1, false, false);
    if (!ar.use_kv_write_rows || !ar.force_validity_mask) return 1;

    const auto exact_ar = qwen35_target_step_policy(
        true, false, 0, 1, false, false);
    if (exact_ar.use_kv_write_rows || exact_ar.force_validity_mask) return 2;

    const auto verify = qwen35_target_step_policy(
        false, false, 0, 5, true, true);
    if (!verify.use_kv_write_rows || !verify.force_validity_mask) return 3;

    const auto windowed = qwen35_target_step_policy(
        false, false, 4096, 1, false, false);
    if (windowed.use_kv_write_rows || windowed.force_validity_mask) return 4;

    if (qwen35_rollback_storage_from_string(nullptr) != Qwen35RollbackStorage::F32) return 5;
    if (qwen35_rollback_storage_from_string("") != Qwen35RollbackStorage::F32) return 6;
    if (qwen35_rollback_storage_from_string("f32") != Qwen35RollbackStorage::F32) return 7;
    if (qwen35_rollback_storage_from_string("f16") != Qwen35RollbackStorage::F16) return 8;
    if (qwen35_rollback_storage_from_string("invalid") != Qwen35RollbackStorage::F32) return 9;
    return 0;
}
