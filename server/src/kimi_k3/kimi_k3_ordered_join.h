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

} // namespace dflash::common
