#pragma once

#include <cmath>
#include <algorithm>
#include <cstddef>
#include <vector>

namespace dflash::common {

enum class DraftProposalLayout {
    seed_then_proposals,
    proposals_only,
};

struct DraftProposalShape {
    int draft_width = 0;
    DraftProposalLayout layout = DraftProposalLayout::seed_then_proposals;

    int first_proposal_row() const {
        return layout == DraftProposalLayout::proposals_only ? 0 : 1;
    }

    int proposal_count() const {
        return draft_width > first_proposal_row()
            ? draft_width - first_proposal_row()
            : 0;
    }

    int verify_width() const { return 1 + proposal_count(); }
};

inline int draft_max_verify_width(
        const DraftProposalShape & shape,
        bool tree_mode,
        int tree_budget) {
    return tree_mode
        ? std::max(shape.verify_width(), tree_budget + 1)
        : shape.verify_width();
}

// Native DSpark verifies [anchor, proposal...] in one target batch. When every
// proposal is accepted, that batch already contains the committed target state
// and can be retained without a restore/replay pass. Legacy drafters keep their
// existing rollback policy because their first row is the seed itself.
inline bool draft_can_keep_full_verify(
        const DraftProposalShape & shape,
        bool fast_rollback_enabled,
        int accepted,
        int remaining_budget) {
    return shape.layout == DraftProposalLayout::proposals_only &&
           fast_rollback_enabled && accepted == shape.verify_width() &&
           remaining_budget >= accepted;
}

// Native DSpark can restore a partially accepted recurrent state from the
// target's per-token intermediates, just like the legacy speculative path.
// Capturing those rows is only useful when fast rollback is enabled; legacy
// drafters retain their historical capture policy.
inline bool draft_requires_ssm_intermediate_capture(
        const DraftProposalShape & shape,
        bool fast_rollback_enabled) {
    return shape.layout != DraftProposalLayout::proposals_only ||
           fast_rollback_enabled;
}

inline bool draft_can_rollback_partial_verify(
        const DraftProposalShape & shape,
        bool fast_rollback_enabled,
        int accepted) {
    return shape.layout == DraftProposalLayout::proposals_only &&
           fast_rollback_enabled && accepted > 0 &&
           accepted < shape.verify_width();
}

// Native DSpark always commits its anchor, so captured recurrent
// intermediates cover every partial-accept path; a fully accepted block keeps
// the verify state directly. A pre-verify snapshot is only needed when that
// fast rollback capability is unavailable. Legacy proposal layouts retain
// their replay fallback and therefore still require a snapshot.
inline bool draft_requires_preverify_snapshot(
        const DraftProposalShape & shape,
        bool fast_rollback_enabled) {
    return shape.layout != DraftProposalLayout::proposals_only ||
           !fast_rollback_enabled;
}

// The cached drafter folds newly committed target features into its KV cache.
// Legacy DFlash can advance by up to two draft blocks plus a bonus; native
// DSpark advances by at most its target verify width (anchor + proposals).
inline int draft_kv_append_width(const DraftProposalShape & shape) {
    return shape.layout == DraftProposalLayout::proposals_only
        ? shape.verify_width()
        : 2 * shape.draft_width + 2;
}

// Build the deterministic 128-wide GIDD LogSnrEmbed input used by DSpark.
// Row zero of each block is the clean anchor (t=1000); all remaining rows
// are masked/noisy (t=0). The returned storage is row-major by draft row.
inline bool make_dspark_log_snr_features(
        int block_size,
        float min_log_snr,
        float max_log_snr,
        std::vector<float> & out) {
    constexpr int n_freq = 128;
    constexpr int half = n_freq / 2;
    if (block_size <= 0 || !std::isfinite(min_log_snr) ||
        !std::isfinite(max_log_snr) || max_log_snr <= min_log_snr) {
        out.clear();
        return false;
    }

    out.resize((size_t)n_freq * (size_t)block_size);
    for (int pos = 0; pos < block_size; ++pos) {
        const float log_snr = pos == 0 ? max_log_snr : min_log_snr;
        const float t = (log_snr - min_log_snr) /
                        (max_log_snr - min_log_snr) * 1000.0f;
        for (int i = 0; i < half; ++i) {
            const float freq = std::exp(-std::log(10000.0f) * (float)i / (float)half);
            const float angle = t * freq;
            out[(size_t)pos * n_freq + (size_t)i] = std::sin(angle);
            out[(size_t)pos * n_freq + (size_t)half + (size_t)i] = std::cos(angle);
        }
    }
    return true;
}

}  // namespace dflash::common
