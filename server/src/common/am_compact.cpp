// RED STUB: replaced by the implementation in the green commit.
#include "common/am_compact.h"

namespace luce::am {

AutoConfig auto_config_from_strings(const char *, const char *, const char *, std::string *) { return {}; }
bool auto_active(const AutoConfig &, bool, bool) { return false; }
bool should_compact(const AutoConfig &, const SealInfo &) { return false; }
int64_t auto_reserve_bytes(int64_t, int, int) { return 0; }
bool compact_block(ggml_backend_t, luce_am::GramAccumulator &, ggml_tensor *, ggml_tensor *, ggml_tensor *, int,
                   const luce_am::NnlsParams &, BlockFit &, std::string * err) {
    if (err) *err = "stub";
    return false;
}
bool compact_block_host(ggml_backend_t, luce_am::GramAccumulator &, int, int, int, int, int, int, const ggml_fp16_t *,
                        const float *, const float *, const luce_am::NnlsParams &, BlockFit &, std::string * err) {
    if (err) *err = "stub";
    return false;
}
void sort_sel_ascending(std::vector<int32_t> &, int, int) {}
void gather_rows(const uint8_t *, size_t, const int32_t *, int, uint8_t *) {}

}  // namespace luce::am
