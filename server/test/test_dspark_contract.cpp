#include "draft/draft_contract.h"

#include <cmath>
#include <cstdio>
#include <vector>

using namespace dflash::common;

namespace {

bool check(bool condition, const char * expression, int line) {
    if (condition) return true;
    std::fprintf(stderr, "check failed at line %d: %s\n", line, expression);
    return false;
}

}  // namespace

#define CHECK(expr) do { if (!check((expr), #expr, __LINE__)) return 1; } while (false)

int main() {
    const DraftProposalShape legacy{16, DraftProposalLayout::seed_then_proposals};
    CHECK(legacy.first_proposal_row() == 1);
    CHECK(legacy.proposal_count() == 15);
    CHECK(legacy.verify_width() == 16);

    const DraftProposalShape native{4, DraftProposalLayout::proposals_only};
    CHECK(native.first_proposal_row() == 0);
    CHECK(native.proposal_count() == 4);
    CHECK(native.verify_width() == 5);
    CHECK(draft_max_verify_width(native, false, 22) == 5);
    CHECK(draft_max_verify_width(legacy, true, 22) == 23);
    CHECK(draft_can_keep_full_verify(native, true, 5, 5));
    CHECK(!draft_can_keep_full_verify(native, false, 5, 5));
    CHECK(!draft_can_keep_full_verify(native, true, 4, 5));
    CHECK(!draft_can_keep_full_verify(native, true, 5, 4));
    CHECK(!draft_can_keep_full_verify(legacy, true, 16, 16));
    CHECK(draft_requires_ssm_intermediate_capture(native, true));
    CHECK(!draft_requires_ssm_intermediate_capture(native, false));
    CHECK(draft_requires_ssm_intermediate_capture(legacy, true));
    CHECK(draft_requires_ssm_intermediate_capture(legacy, false));
    CHECK(draft_can_rollback_partial_verify(native, true, 1));
    CHECK(draft_can_rollback_partial_verify(native, true, 4));
    CHECK(!draft_can_rollback_partial_verify(native, true, 5));
    CHECK(!draft_can_rollback_partial_verify(native, false, 4));
    CHECK(!draft_can_rollback_partial_verify(legacy, true, 15));
    CHECK(!draft_requires_preverify_snapshot(native, true));
    CHECK(draft_requires_preverify_snapshot(native, false));
    CHECK(draft_requires_preverify_snapshot(legacy, true));
    CHECK(draft_requires_preverify_snapshot(legacy, false));
    CHECK(draft_kv_append_width(native) == 5);
    CHECK(draft_kv_append_width(legacy) == 34);

    std::vector<float> feat;
    CHECK(make_dspark_log_snr_features(4, -9.0f, 9.0f, feat));
    CHECK(feat.size() == 4u * 128u);
    // t=0 => sin=0 and cos=1 for every frequency on masked rows.
    for (int pos = 1; pos < 4; ++pos) {
        for (int i = 0; i < 64; ++i) {
            CHECK(feat[(size_t)pos * 128u + (size_t)i] == 0.0f);
            CHECK(feat[(size_t)pos * 128u + 64u + (size_t)i] == 1.0f);
        }
    }
    // The anchor is t=1000 and therefore differs from the masked rows.
    CHECK(std::fabs(feat[0]) > 0.1f);
    CHECK(std::fabs(feat[64] - 1.0f) > 0.1f);

    CHECK(!make_dspark_log_snr_features(0, -9.0f, 9.0f, feat));
    CHECK(!make_dspark_log_snr_features(4, 9.0f, -9.0f, feat));
    return 0;
}
