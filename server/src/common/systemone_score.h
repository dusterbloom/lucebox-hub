// /v1/systemone candidate scoring.
//
// Causal backends answer from a single post-prefill position, so the
// distribution is one vocab-wide row. The diffusion structured read returns a
// canvas: it can emit a thinking block (e.g. <|channel>thought\n<channel|>)
// before the answer, and it emits the answer label as the BARE token ("Paris",
// id 50429) rather than the leading-space token (" Paris", id 9079) that causal
// BPE answers use. Scoring a single surface form therefore ranks the wrong
// token, and slot 0 is a channel marker rather than the answer.
//
// These helpers resolve each label to all single-token surface forms and score
// it at the slot the answer actually lands on.
//
// Header-only and ggml-free so it is unit-testable on CPU.

#pragma once

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <functional>
#include <limits>
#include <string>
#include <vector>

namespace luce::common {

using SystemoneEncodeFn =
    std::function<std::vector<int32_t>(const std::string &)>;

// Candidate token ids for `label`: the single-token encodings of both
// " <label>" (causal / mid-sentence form) and "<label>" (diffusion's bare
// form), deduplicated. Labels that are not single tokens contribute whatever
// single-token forms exist (possibly none).
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

// Softmax restricted to the union of all candidate ids, collapsed per label:
// label i gets the max probability over its own ids. Labels with no resolvable
// id get probability 0.
inline std::vector<float> systemone_label_probs_row(
        const float * row, int vocab,
        const std::vector<std::vector<int32_t>> & label_ids) {
    std::vector<float> out(label_ids.size(), 0.0f);

    int32_t max_id = -1;
    float   max_logit = -std::numeric_limits<float>::infinity();
    for (const auto & ids : label_ids) {
        for (int32_t id : ids) {
            if (id < 0 || id >= vocab) continue;
            if (row[id] > max_logit) { max_logit = row[id]; max_id = id; }
        }
    }
    if (max_id < 0 || !std::isfinite(max_logit)) return out;

    double sum = 0.0;
    std::vector<std::pair<const std::vector<int32_t> *, std::vector<double>>> exps;
    exps.reserve(label_ids.size());
    for (const auto & ids : label_ids) {
        std::vector<double> e(ids.size(), 0.0);
        for (size_t k = 0; k < ids.size(); ++k) {
            const int32_t id = ids[k];
            if (id < 0 || id >= vocab) continue;
            e[k] = std::exp((double) (row[id] - max_logit));
            sum += e[k];
        }
        exps.emplace_back(&ids, std::move(e));
    }
    if (sum <= 0.0) return out;

    for (size_t i = 0; i < exps.size(); ++i) {
        float best = 0.0f;
        const auto & e = exps[i].second;
        for (double p : e) best = (std::max)(best, (float) (p / sum));
        out[i] = best;
    }
    return out;
}

// Index of the first canvas slot whose full-vocabulary argmax is any candidate
// id (any surface form of any label); -1 if none. This skips a leading
// channel/thinking token and finds where the answer actually lands.
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
