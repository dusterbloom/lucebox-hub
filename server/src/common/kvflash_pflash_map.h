// KVFlash recall mode 5: PFlash segment densities as chunk scores (pure, no ggml, no GPU).

#pragma once

#include "pflash_types.h"

#include <algorithm>
#include <vector>

namespace luce::common {

// The chunk score of each chunk_tokens-sized chunk of an n_tokens prompt: the largest density of the
// segments overlapping it, 0 where none does (pflash_unit_round.py R1). kvflash_recall_pick then sums
// the chunk scores of each unit and keeps the tail, window and pins out of the candidates.
inline std::vector<float> kvflash_pflash_chunk_scores(const std::vector<PFlashTokenSpan> & segments,
                                                      const std::vector<double> & densities,
                                                      int n_tokens, int chunk_tokens) {
    if (n_tokens <= 0 || chunk_tokens <= 0) return {};
    std::vector<float> out((size_t)((n_tokens + chunk_tokens - 1) / chunk_tokens), 0.0f);
    for (size_t i = 0; i < segments.size() && i < densities.size(); i++) {
        const int b = std::max(0, segments[i].begin), e = std::min(n_tokens, segments[i].end);
        if (e <= b) continue;
        for (int c = b / chunk_tokens; c <= (e - 1) / chunk_tokens; c++)
            out[(size_t)c] = std::max(out[(size_t)c], (float)densities[i]);
    }
    return out;
}

}  // namespace luce::common
