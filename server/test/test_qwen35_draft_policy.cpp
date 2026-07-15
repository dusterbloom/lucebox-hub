#include "qwen35/draft_policy.h"

using dflash::common::DraftProposalLayout;
using dflash::common::DraftProposalShape;
using dflash::common::qwen35_can_keep_full_dspark_verify;
using dflash::common::qwen35_can_rollback_dspark_verify;
using dflash::common::qwen35_dspark_retained_rows;
using dflash::common::qwen35_max_verify_width;
using dflash::common::qwen35_requires_preverify_snapshot;
using dflash::common::qwen35_requires_ssm_intermediate_capture;

constexpr DraftProposalShape kLegacy{16, DraftProposalLayout::SeedThenProposals};
constexpr DraftProposalShape kBonsai{4, DraftProposalLayout::ProposalsOnly};

static_assert(qwen35_max_verify_width(kLegacy, false, 0) == 16);
static_assert(qwen35_max_verify_width(kBonsai, false, 0) == 5);
static_assert(qwen35_max_verify_width(kBonsai, true, 31) == 32);
static_assert(qwen35_max_verify_width(kLegacy, true, 7) == 16);

static_assert(qwen35_can_keep_full_dspark_verify(kBonsai, true, 5, 5));
static_assert(!qwen35_can_keep_full_dspark_verify(kLegacy, true, 16, 16));
static_assert(!qwen35_can_keep_full_dspark_verify(kBonsai, false, 5, 5));
static_assert(!qwen35_can_keep_full_dspark_verify(kBonsai, true, 4, 5));
static_assert(!qwen35_can_keep_full_dspark_verify(kBonsai, true, 5, 4));

static_assert(qwen35_requires_ssm_intermediate_capture(kLegacy, false));
static_assert(qwen35_requires_ssm_intermediate_capture(kBonsai, true));
static_assert(!qwen35_requires_ssm_intermediate_capture(kBonsai, false));

static_assert(qwen35_can_rollback_dspark_verify(kBonsai, true, 1));
static_assert(qwen35_can_rollback_dspark_verify(kBonsai, true, 4));
static_assert(!qwen35_can_rollback_dspark_verify(kBonsai, true, 5));
static_assert(!qwen35_can_rollback_dspark_verify(kBonsai, false, 4));
static_assert(!qwen35_can_rollback_dspark_verify(kLegacy, true, 15));

static_assert(!qwen35_requires_preverify_snapshot(kBonsai, true));
static_assert(qwen35_requires_preverify_snapshot(kBonsai, false));
static_assert(qwen35_requires_preverify_snapshot(kLegacy, true));

constexpr bool native_tail_policy_is_complete() {
    for (int accepted = 1; accepted <= kBonsai.verify_width(); ++accepted) {
        for (int budget = 1; budget <= kBonsai.verify_width(); ++budget) {
            const int retained = qwen35_dspark_retained_rows(accepted, budget);
            const bool keep = qwen35_can_keep_full_dspark_verify(
                kBonsai, true, accepted, budget);
            const bool rollback = qwen35_can_rollback_dspark_verify(
                kBonsai, true, retained);
            if (keep == rollback) return false;
        }
    }
    return true;
}

static_assert(qwen35_dspark_retained_rows(5, 2) == 2);
static_assert(qwen35_dspark_retained_rows(2, 5) == 2);
static_assert(qwen35_dspark_retained_rows(0, 5) == 0);
static_assert(qwen35_dspark_retained_rows(5, 0) == 0);
static_assert(native_tail_policy_is_complete());

int main() {
    return 0;
}
