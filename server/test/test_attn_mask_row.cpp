#include "common/attn_masks.h"

using namespace dflash::common;

int main() {
    std::vector<uint16_t> row;
    build_causal_mask_row(row, 8, 3, 2);
    if (row.size() != 8) return 1;
    if (row[0] != F16_ZERO || row[1] != F16_ZERO || row[2] != F16_ZERO) return 2;
    for (std::size_t i = 3; i < row.size(); ++i) {
        if (row[i] != F16_NEG_INF) return 3;
    }

    build_causal_mask_row(row, 8, 4, 6, 3);
    if (row[0] != F16_ZERO || row[1] != F16_ZERO || row[2] != F16_ZERO ||
        row[3] != F16_ZERO) return 4;
    for (std::size_t i = 4; i < row.size(); ++i) {
        if (row[i] != F16_NEG_INF) return 5;
    }
    return 0;
}
