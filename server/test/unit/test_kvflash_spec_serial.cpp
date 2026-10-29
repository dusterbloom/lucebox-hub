#include "CppUnitTestFramework.hpp"
#include "../../src/common/kvflash_pager.h"
#include "../../src/qwen35/prefill_helpers.h"

namespace {
struct KvflashSpecSerialFixture {};
}

using namespace luce::common;

// F1: DDTree commits every accepted token before it checks termination and
// never settles, so on EOS/cancel/budget end it finishes one position past AR.
// DDTree is opt-in and no launcher uses it: --ddtree with KVFlash fails closed.
TEST_CASE(KvflashSpecSerialFixture, ddtree_with_kvflash_is_refused) {
    CHECK(!kvflash_ddtree_refusal(true, "4096", nullptr).empty());      // --kvflash 4096
    CHECK(!kvflash_ddtree_refusal(true, "auto", nullptr).empty());
    CHECK(!kvflash_ddtree_refusal(true, "", "8192").empty());           // LUCE_KVFLASH env
    CHECK(!kvflash_ddtree_refusal(true, "", "auto").empty());
}

TEST_CASE(KvflashSpecSerialFixture, ddtree_alone_or_kvflash_alone_is_accepted) {
    CHECK(kvflash_ddtree_refusal(true, "", nullptr).empty());
    CHECK(kvflash_ddtree_refusal(true, "", "0").empty());               // pool 0 = off
    CHECK(kvflash_ddtree_refusal(false, "4096", nullptr).empty());
    CHECK(kvflash_ddtree_refusal(false, "", "8192").empty());
}

TEST_CASE(KvflashSpecSerialFixture, ddtree_refusal_is_one_line_naming_both_flags) {
    const std::string m = kvflash_ddtree_refusal(true, "4096", nullptr);
    CHECK(m.find("--ddtree") != std::string::npos);
    CHECK(m.find("--kvflash") != std::string::npos);
    CHECK(m.find('\n') == std::string::npos);
}
