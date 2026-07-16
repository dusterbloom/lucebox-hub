#include "common/dflash_draft_kv_policy.h"
#include "common/dflash_feature_ring.h"

using dflash::common::draft_feature_storage_type;
using dflash::common::draft_kv_append_capacity;

int main() {
    if (draft_kv_append_capacity(4, true) != 5) return 1;
    if (draft_kv_append_capacity(16, false) != 34) return 2;
    if (draft_kv_append_capacity(0, true) != 1) return 3;

    if (draft_feature_storage_type(nullptr, GGML_TYPE_BF16) != GGML_TYPE_BF16) return 4;
    if (draft_feature_storage_type("", GGML_TYPE_F16) != GGML_TYPE_F16) return 5;
    if (draft_feature_storage_type("f32", GGML_TYPE_BF16) != GGML_TYPE_F32) return 6;
    if (draft_feature_storage_type("BF16", GGML_TYPE_F32) != GGML_TYPE_BF16) return 7;
    if (draft_feature_storage_type("q8", GGML_TYPE_F32) != GGML_TYPE_Q8_0) return 8;
    if (draft_feature_storage_type("invalid", GGML_TYPE_BF16) != GGML_TYPE_F32) return 9;
    return 0;
}
