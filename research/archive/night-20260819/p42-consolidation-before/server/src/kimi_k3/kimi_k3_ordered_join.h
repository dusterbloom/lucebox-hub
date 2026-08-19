#pragma once

#include <cstdint>

namespace dflash::common {

// Apply the frozen single-owner accumulation schedule to row-major F32
// operands. Operations [0, calibrated_ops) accumulate directly into the
// destination. Remaining operations accumulate from +0 into the exact
// fallback subtotal, which is added to the destination once.
bool kimi_k3_ordered_join_reference(
    const float * rows,
    int row_count,
    int width,
    const int32_t * row_indices,
    const float * weights,
    int operation_count,
    int calibrated_operations,
    float * output,
    const char ** failure_reason = nullptr);

// Device equivalent of kimi_k3_ordered_join_reference. All operands are
// device-resident and the launch completes before returning. Arithmetic is
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

} // namespace dflash::common
