// Unit test for the LUCE_KVFLASH_PREFILL_RESELECT pure helpers (no GPU).

#include "CppUnitTestFramework.hpp"

#include "kvflash_prefill_reselect.h"

#include <vector>

using namespace luce::common;

namespace {
struct KvflashPrefillReselectFixture {};
}

TEST_CASE(KvflashPrefillReselectFixture, empty_cuts_gives_end_minus_1) {
    CHECK(kvflash_prefill_reselect_point({}, 0, 1000) == 999);
}
TEST_CASE(KvflashPrefillReselectFixture, last_cut_inside_wins) {
    CHECK(kvflash_prefill_reselect_point({100, 900}, 0, 1000) == 900);
}
TEST_CASE(KvflashPrefillReselectFixture, cut_at_end_falls_back) {
    CHECK(kvflash_prefill_reselect_point({100, 1000}, 0, 1000) == 999);
}
TEST_CASE(KvflashPrefillReselectFixture, cut_at_or_before_offset_falls_back) {
    CHECK(kvflash_prefill_reselect_point({500}, 500, 1000) == 999);
    CHECK(kvflash_prefill_reselect_point({400}, 500, 1000) == 999);
}
TEST_CASE(KvflashPrefillReselectFixture, one_token_prefill_is_disabled) {
    CHECK(kvflash_prefill_reselect_point({}, 999, 1000) == -1);
    CHECK(kvflash_prefill_reselect_point({}, 0, 1) == -1);
}
TEST_CASE(KvflashPrefillReselectFixture, cuts_with_inserts_sorted) {
    CHECK((kvflash_cuts_with({100, 900}, 500) == std::vector<int>{100, 500, 900}));
    CHECK((kvflash_cuts_with({}, 7) == std::vector<int>{7}));
    CHECK((kvflash_cuts_with({100, 900}, 900) == std::vector<int>{100, 900}));
    CHECK((kvflash_cuts_with({100, 900}, -1) == std::vector<int>{100, 900}));
}
