// /v1/systemone candidate scoring.
//
// Causal backends answer from a single post-prefill position, so the
// distribution is one vocab-wide row. The diffusion structured read returns a
// canvas: it can emit a channel/formatting marker at slot 0 (e.g. <|channel>)
// and place the answer label a slot or two later. These helpers score the
// labeled candidates at the slot the answer actually lands on.
//
// Header-only and ggml-free so it is unit-testable on CPU.

#pragma once

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <vector>

namespace luce::common {

// Softmax restricted to `candidates` (token ids into a vocab-wide `row`).
// Unresolvable candidates (id < 0 or >= vocab) get probability 0.
inline std::vector<float> systemone_softmax_restricted_row(
        const float * row, int vocab,
        const std::vector<int32_t> & candidates) {
    std::vector<float> probs(candidates.size(), 0.0f);
    float max_logit = -std::numeric_limits<float>::infinity();
    for (int32_t id : candidates) {
        if (id < 0 || id >= vocab) continue;
        max_logit = (std::max)(max_logit, row[id]);
    }
    if (!std::isfinite(max_logit)) return probs;  // no candidate tokenized

    std::vector<double> exps(candidates.size(), 0.0);
    double sum = 0.0;
    for (size_t i = 0; i < candidates.size(); ++i) {
        const int32_t id = candidates[i];
        if (id < 0 || id >= vocab) continue;
        const double e = std::exp((double) (row[id] - max_logit));
        exps[i] = e;
        sum += e;
    }
    if (sum <= 0.0) return probs;
    for (size_t i = 0; i < candidates.size(); ++i) {
        probs[i] = (float) (exps[i] / sum);
    }
    return probs;
}

// Index of the first canvas slot whose full-vocabulary argmax is a candidate
// token; -1 if none. This is where a labeled answer actually lands, past any
// leading channel/formatting token the canvas emits at slot 0.
inline int systemone_pick_answer_slot(const std::vector<float> & logits,
                                      int slot_count, int vocab,
                                      const std::vector<int32_t> & candidates) {
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
        for (int32_t id : candidates) {
            if (id == best) return s;
        }
    }
    return -1;
}

}  // namespace luce::common
