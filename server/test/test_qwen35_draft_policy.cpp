#include "qwen35/draft_policy.h"

using dflash::common::DraftProposalLayout;
using dflash::common::DraftProposalShape;
using dflash::common::qwen35_can_keep_full_dspark_verify;
using dflash::common::qwen35_max_verify_width;

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

int main() {
    return 0;
}
