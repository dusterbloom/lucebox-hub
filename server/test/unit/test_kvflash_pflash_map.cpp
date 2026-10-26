// Unit test for the mode-5 segment-to-chunk mapping (no GPU); mirrors pflash_unit_round.py R1.

#include "CppUnitTestFramework.hpp"

#include "kvflash_pflash_map.h"
#include "kvflash_prefill_reselect.h"

#include <vector>

using namespace luce::common;

namespace {
struct KvflashPflashMapFixture {};
}

TEST_CASE(KvflashPflashMapFixture, a_chunk_takes_the_max_density_of_the_segments_overlapping_it) {
    // chunks of 64: [0,64) [64,128) [128,192) [192,256)
    const auto s = kvflash_pflash_chunk_scores({{10, 70}, {70, 100}, {200, 210}}, {2.0, 5.0, 3.0}, 256, 64);
    CHECK((s == std::vector<float>{2.0f, 5.0f, 0.0f, 3.0f}));
}

TEST_CASE(KvflashPflashMapFixture, a_segment_spanning_chunks_scores_each_of_them_and_gaps_stay_zero) {
    // [100, 300) touches chunks 1..4 (299 / 64 = 4); chunks 0 and 5 are gaps
    const auto s = kvflash_pflash_chunk_scores({{100, 300}}, {4.0}, 384, 64);
    CHECK((s == std::vector<float>{0.0f, 4.0f, 4.0f, 4.0f, 4.0f, 0.0f}));
}

TEST_CASE(KvflashPflashMapFixture, the_end_is_exclusive_and_the_partial_last_chunk_counts) {
    const auto s = kvflash_pflash_chunk_scores({{0, 64}, {128, 129}}, {1.0, 2.0}, 130, 64);
    CHECK((s == std::vector<float>{1.0f, 0.0f, 2.0f}));
}

TEST_CASE(KvflashPflashMapFixture, spans_past_the_prompt_are_clipped_and_empty_or_mismatched_input_scores_zero) {
    CHECK((kvflash_pflash_chunk_scores({{100, 900}}, {1.0}, 192, 64) == std::vector<float>{0.0f, 1.0f, 1.0f}));
    CHECK((kvflash_pflash_chunk_scores({{70, 70}, {-5, 3}}, {9.0, 1.0}, 128, 64) == std::vector<float>{1.0f, 0.0f}));
    CHECK((kvflash_pflash_chunk_scores({{0, 64}}, {}, 128, 64) == std::vector<float>{0.0f, 0.0f}));
    CHECK(kvflash_pflash_chunk_scores({{0, 64}}, {1.0}, 0, 64).empty());
    CHECK(kvflash_pflash_chunk_scores({{0, 64}}, {1.0}, 64, 0).empty());
}

TEST_CASE(KvflashPflashMapFixture, the_pick_keeps_pinned_chunks_and_the_tail_out_whatever_their_density) {
    // 40 chunks, pins end at chunk 18 (first_chunk), window starts at chunk 30; units of 2 chunks.
    // Hot segments sit in the pinned preamble (chunks 2-3), the window (chunks 33-34) and the candidates.
    const int ct = 64;
    const auto s = kvflash_pflash_chunk_scores(
        {{2 * ct, 4 * ct}, {33 * ct, 35 * ct}, {20 * ct, 21 * ct}, {26 * ct, 27 * ct + 1}}, {9.0, 9.0, 3.0, 2.0}, 40 * ct, ct);
    const auto picks = kvflash_recall_pick(s, 30, 2, 2, 18);
    // unit 10 = chunks 20-21 (3.0), unit 13 = chunks 26-27 (2.0 + 2.0 = 4.0): unit 13 first
    CHECK((picks == std::vector<int>{13, 10}));
}

TEST_CASE(KvflashPflashMapFixture, a_unit_scores_the_sum_of_its_chunk_maxima_as_in_the_offline_rounding) {
    const int ct = 64;
    // two thin segments in one unit (chunks 20, 21) outrank one dense segment in a chunk of another unit
    const auto s = kvflash_pflash_chunk_scores({{20 * ct, 20 * ct + 8}, {21 * ct, 21 * ct + 8}, {24 * ct, 24 * ct + 8}},
                                               {2.0, 2.0, 3.0}, 40 * ct, ct);
    CHECK((kvflash_recall_pick(s, 30, 1, 2, 18) == std::vector<int>{10}));
}
