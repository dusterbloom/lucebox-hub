#include "draft/draft_contract.h"
#include "common/attn_masks.h"
#include "qwen35/runtime_policy.h"

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
    {
        const auto ar = qwen35_target_step_policy(
            /*no_kvpad=*/false, /*kvflash_mask=*/false,
            /*fa_window=*/0, /*n_tokens=*/1,
            /*capture=*/false, /*dynamic_rows=*/false);
        CHECK(ar.use_kv_write_rows);
        CHECK(ar.force_validity_mask);

        const auto exact_ar = qwen35_target_step_policy(
            /*no_kvpad=*/true, /*kvflash_mask=*/false,
            /*fa_window=*/0, /*n_tokens=*/1,
            /*capture=*/false, /*dynamic_rows=*/false);
        CHECK(!exact_ar.use_kv_write_rows);
        CHECK(!exact_ar.force_validity_mask);

        const auto verify = qwen35_target_step_policy(
            /*no_kvpad=*/false, /*kvflash_mask=*/false,
            /*fa_window=*/0, /*n_tokens=*/5,
            /*capture=*/true, /*dynamic_rows=*/true);
        CHECK(verify.use_kv_write_rows);
        CHECK(verify.force_validity_mask);

        const auto windowed = qwen35_target_step_policy(
            /*no_kvpad=*/false, /*kvflash_mask=*/false,
            /*fa_window=*/4096, /*n_tokens=*/1,
            /*capture=*/false, /*dynamic_rows=*/false);
        CHECK(!windowed.use_kv_write_rows);
        CHECK(!windowed.force_validity_mask);
    }

    {
        std::vector<uint16_t> row;
        build_causal_mask_row(row, /*kv_pad=*/8, /*kv_len=*/3,
                              /*query_pos=*/2, /*win_start=*/0);
        CHECK(row.size() == 8);
        CHECK(row[0] == F16_ZERO);
        CHECK(row[1] == F16_ZERO);
        CHECK(row[2] == F16_ZERO);
        for (size_t i = 3; i < row.size(); ++i) {
            CHECK(row[i] == F16_NEG_INF);
        }
    }

    {
        CHECK(qwen35_rollback_storage_from_string(nullptr) ==
              Qwen35RollbackStorage::f32);
        CHECK(qwen35_rollback_storage_from_string("") ==
              Qwen35RollbackStorage::f32);
        CHECK(qwen35_rollback_storage_from_string("f32") ==
              Qwen35RollbackStorage::f32);
        CHECK(qwen35_rollback_storage_from_string("f16") ==
              Qwen35RollbackStorage::f16);
        CHECK(qwen35_rollback_storage_from_string("invalid") ==
              Qwen35RollbackStorage::f32);
    }

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
