// Qwen3.5-specific scheduling decisions for draft proposal rows.

#pragma once

#include "draft/draft_contract.h"

namespace dflash::common {

constexpr int qwen35_max_verify_width(
        DraftProposalShape shape,
        bool tree_mode,
        int tree_budget) noexcept {
    const int chain_width = shape.verify_width();
    const int tree_width = tree_budget + 1;
    return tree_mode && tree_width > chain_width ? tree_width : chain_width;
}

// A fully accepted proposal-only batch already leaves Qwen3.5's target state
// at the committed position. Keep that state only when fast rollback is
// enabled, the entire block fits the remaining generation budget, and the
// artifact explicitly uses the proposals-only layout.
constexpr bool qwen35_can_keep_full_dspark_verify(
        DraftProposalShape shape,
        bool fast_rollback_enabled,
        int accepted,
        int remaining_budget) noexcept {
    return shape.layout == DraftProposalLayout::ProposalsOnly &&
           fast_rollback_enabled &&
           accepted == shape.verify_width() &&
           remaining_budget >= accepted;
}

constexpr bool qwen35_requires_ssm_intermediate_capture(
        DraftProposalShape shape,
        bool fast_rollback_enabled) noexcept {
    return shape.layout != DraftProposalLayout::ProposalsOnly ||
           fast_rollback_enabled;
}

constexpr int qwen35_dspark_retained_rows(
        int accepted,
        int remaining_budget) noexcept {
    if (accepted <= 0 || remaining_budget <= 0) return 0;
    return accepted < remaining_budget ? accepted : remaining_budget;
}

constexpr bool qwen35_can_rollback_dspark_verify(
        DraftProposalShape shape,
        bool fast_rollback_enabled,
        int retained) noexcept {
    return shape.layout == DraftProposalLayout::ProposalsOnly &&
           fast_rollback_enabled &&
           retained > 0 &&
           retained < shape.verify_width();
}

constexpr bool qwen35_requires_preverify_snapshot(
        DraftProposalShape shape,
        bool fast_rollback_enabled) noexcept {
    return shape.layout != DraftProposalLayout::ProposalsOnly ||
           !fast_rollback_enabled;
}

}  // namespace dflash::common
