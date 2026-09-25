// /v1/systemone candidate scoring.
//
// The endpoint renders a prompt that ends where the answer belongs, then asks
// the backend for a distribution over the candidate label tokens. Causal
// backends answer from one post-prefill position; the diffusion structured
// read returns a canvas (it can emit a thinking block before the answer, and
// the answer label is the *bare* token, e.g. "Paris", not the leading-space
// " Paris").
//
// Contract (see server/docs/systemone.md):
//   - A label resolves to its single-token surface forms, " <label>" and
//     "<label>". A label with no single-token form is UNRESOLVABLE — callers
//     must reject the request or abstain, never silently collapse it.
//   - The returned probabilities are a proper distribution over the labels
//     (they sum to 1) whenever candidate mass exists; `candidate_mass` records
//     how much of the model's probability sits on the candidate tokens at all,
//     so "0.999 between two near-zero options" is visible.
//   - Callers must pass disjoint candidate sets (unique aliases); the scorer
//     de-duplicates ids across labels but a shared id still loses mass.
//
// Header-only and ggml-free so it is unit-testable on CPU.

#pragma once

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <functional>
#include <limits>
#include <string>
#include <unordered_set>
#include <vector>

namespace luce::common {

using SystemoneEncodeFn =
    std::function<std::vector<int32_t>(const std::string &)>;

// Single-token surface forms of `label`: " <label>" (causal / mid-sentence)
// and "<label>" (diffusion's bare form), de-duplicated. Empty if neither is a
// single token — the label is unresolvable.
inline std::vector<int32_t> systemone_label_token_ids(
        const SystemoneEncodeFn & encode, const std::string & label) {
    std::vector<int32_t> ids;
    auto add = [&](const std::string & s) {
        const std::vector<int32_t> t = encode(s);
        if (t.size() == 1 && t[0] >= 0 &&
            std::find(ids.begin(), ids.end(), t[0]) == ids.end()) {
            ids.push_back(t[0]);
        }
    };
    add(" " + label);
    add(label);
    return ids;
}

// Per-label distribution plus the model's total probability on the candidate
// tokens (the complement of `candidate_mass` is mass on non-candidate tokens —
// i.e. the model wanted something outside the offered options).
struct SystemoneScore {
    std::vector<float> label_probs;   // sums to 1 when candidate_mass > 0
    float              candidate_mass = 0.0f;
};

// Softmax over the union of unique candidate ids, collapsed per label by
// summing its forms. Returns a normalized distribution (or all-zeros when no
// candidate token resolves).
inline SystemoneScore systemone_score_row(
        const float * row, int vocab,
        const std::vector<std::vector<int32_t>> & label_ids) {
    SystemoneScore out;
    out.label_probs.assign(label_ids.size(), 0.0f);

    std::vector<int32_t> uniq;
    std::unordered_set<int32_t> seen;
    for (const auto & ids : label_ids) {
        for (int32_t id : ids) {
            if (id < 0 || id >= vocab) continue;
            if (seen.insert(id).second) uniq.push_back(id);
        }
    }
    if (uniq.empty()) return out;

    float max_logit = -std::numeric_limits<float>::infinity();
    for (int32_t id : uniq) max_logit = (std::max)(max_logit, row[id]);
    if (!std::isfinite(max_logit)) return out;

    double denom = 0.0;
    for (int32_t id : uniq) {
        denom += std::exp((double) (row[id] - max_logit));
    }
    if (denom <= 0.0) return out;

    for (size_t i = 0; i < label_ids.size(); ++i) {
        float p = 0.0f;
        for (int32_t id : label_ids[i]) {
            if (id < 0 || id >= vocab) continue;
            p += (float) (std::exp((double) (row[id] - max_logit)) / denom);
        }
        out.label_probs[i] = p;
        out.candidate_mass += p;
    }

    // Normalize to a proper distribution over the labels.
    if (out.candidate_mass > 0.0f) {
        for (float & p : out.label_probs) p /= out.candidate_mass;
    }
    return out;
}

// 1 - H(p)/ln K over a normalized distribution: 1 when certain, 0 when
// uniform. NOTE: this is concentration over the options the caller supplied,
// NOT a probability that the answer is correct, and it is only meaningful
// together with `candidate_mass`. Do not gate on it until calibrated.
inline float systemone_confidence(const std::vector<float> & probs) {
    const size_t K = probs.size();
    if (K <= 1) return 1.0f;
    double H = 0.0;
    for (float p : probs) {
        if (p > 0.0f) H -= (double) p * std::log((double) p);
    }
    return (float) (1.0 - H / std::log((double) K));
}

// Index of the first canvas slot whose full-vocabulary argmax is any candidate
// id (any surface form of any label); -1 if none. Used only for the diffusion
// canvas (slot_count > 1); causal backends score their single slot directly.
inline int systemone_pick_answer_slot(
        const std::vector<float> & logits, int slot_count, int vocab,
        const std::vector<std::vector<int32_t>> & label_ids) {
    if (vocab <= 0 || slot_count <= 0 ||
        (int) logits.size() < slot_count * vocab) {
        return -1;
    }
    for (int s = 0; s < slot_count; ++s) {
        const float * row = logits.data() + (size_t) s * vocab;
        int best = 0;
        for (int v = 1; v < vocab; ++v) {
            if (row[v] > row[best]) best = v;
        }
        for (const auto & ids : label_ids) {
            for (int32_t id : ids) {
                if (id == best) return s;
            }
        }
    }
    return -1;
}

}  // namespace luce::common
