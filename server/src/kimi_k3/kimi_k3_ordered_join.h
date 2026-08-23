#pragma once

#include <cstdint>

namespace dflash::common {

// Apply the frozen single-owner accumulation schedule to device-resident F32
// operands. The launch completes before returning. Arithmetic is
// explicitly rounded multiply followed by rounded add,
// matching the qualified Lucebox4 host SSE sequence rather than contracting
// to FMA.
bool kimi_k3_ordered_join_launch(
    const float * device_rows,
    int row_count,
    int width,
    const float * device_resident_means,
    int resident_mean_count,
    const int32_t * device_row_indices,
    const float * device_weights,
    int operation_count,
    int calibrated_operations,
    float * device_output,
    const char ** failure_reason = nullptr);

// Apply several calibrated-only schedules in one launch. Each schedule owns
// one fixed-stride descriptor row; operation_counts selects its live prefix.
// The arithmetic within every schedule is identical to
// kimi_k3_ordered_join_launch's calibrated loop. This interface deliberately
// has no fallback subtotal: callers with exact fallbacks retain the scalar
// path above.
bool kimi_k3_ordered_join_calibrated_batch_launch(
    const float * device_rows,
    int row_count,
    int width,
    const float * device_resident_means,
    int resident_mean_count,
    const int32_t * device_row_indices,
    const float * device_weights,
    int operation_stride,
    const int32_t * device_operation_counts,
    int batch_count,
    float * device_outputs,
    const char ** failure_reason = nullptr);

} // namespace dflash::common
