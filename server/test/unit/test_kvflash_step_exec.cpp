#include "CppUnitTestFramework.hpp"
#include "../../src/common/kvflash_step_exec.h"

using namespace luce::common;

namespace {
struct KvflashStepExecFixture {};

struct Rec {
    std::vector<KvFlashOpKind> seen;
    std::vector<std::string> calls;
    int fail_at = -1;                      // op index that fails
    bool run(const KvFlashOp & op) {
        const int i = (int)seen.size();
        seen.push_back(op.kind);
        calls.push_back("run");
        return i != fail_at;
    }
    void end_step() { calls.push_back("end"); }
    void abort_step() { calls.push_back("abort"); }
};

std::vector<KvFlashOp> real_plan() {
    KvFlashStepSlot a; a.slot = 0; a.pin_chunks = {1}; a.page_out = {0}; a.alloc_tokens = 64; a.free_blocks = 0; a.alloc_new_chunks = 1;
    KvFlashStepSlot b; b.slot = 1; b.pin_chunks = {2}; b.page_in = {3}; b.free_blocks = 1; b.wants_score = true;
    return kvflash_plan_step({a, b});
}
}  // namespace

TEST_CASE(KvflashStepExecFixture, a_good_plan_runs_in_plan_order_with_one_fence_then_end_step) {
    const auto plan = real_plan();
    Rec be;
    const auto r = kvflash_execute_step(plan, be);
    CHECK(r.status == KvFlashExec::Ok);
    REQUIRE(be.seen.size() + 1 == plan.size());
    for (size_t i = 0; i + 1 < plan.size(); i++) CHECK(be.seen[i] == plan[i].kind);
    int fences = 0;
    for (auto k : be.seen) fences += k == KvFlashOpKind::Fence;
    CHECK(fences == 1);
    CHECK(be.calls.back() == "end");
    for (const auto & c : be.calls) CHECK(c != "abort");
}

TEST_CASE(KvflashStepExecFixture, the_first_failure_aborts_and_nothing_runs_after_it) {
    const auto plan = real_plan();
    for (int f = 0; f + 1 < (int)plan.size(); f++) {
        Rec be; be.fail_at = f;
        const auto r = kvflash_execute_step(plan, be);
        CHECK(r.status == KvFlashExec::Failed);
        CHECK(r.failed_at == (size_t)f);
        CHECK((int)be.seen.size() == f + 1);              // stopped at the failing op
        CHECK(be.calls.back() == "abort");
        int aborts = 0, ends = 0;
        for (const auto & c : be.calls) { aborts += c == "abort"; ends += c == "end"; }
        CHECK(aborts == 1);
        CHECK(ends == 0);                                  // EndStep never follows an abort
    }
}

TEST_CASE(KvflashStepExecFixture, a_malformed_plan_is_refused_before_any_backend_call) {
    const std::vector<std::vector<KvFlashOp>> bad = {
        {},                                                                         // empty
        {{KvFlashOpKind::Pin, 0, 0}},                                               // no EndStep
        {{KvFlashOpKind::EndStep, -1, 0}, {KvFlashOpKind::Pin, 0, 0}},              // EndStep not last
        {{KvFlashOpKind::Fence, -1, 0}, {KvFlashOpKind::Fence, -1, 0}, {KvFlashOpKind::EndStep, -1, 0}},   // two fences
        {{KvFlashOpKind::Score, 0, 0}, {KvFlashOpKind::Fence, -1, 0}, {KvFlashOpKind::EndStep, -1, 0}},    // score before fence
        {{KvFlashOpKind::Score, 0, 0}, {KvFlashOpKind::Score, 1, 0}, {KvFlashOpKind::EndStep, -1, 0}},     // two scores
        {{KvFlashOpKind::EndStep, -1, 0}, {KvFlashOpKind::EndStep, -1, 0}},         // two EndSteps
    };
    for (const auto & p : bad) {
        Rec be;
        CHECK(kvflash_execute_step(p, be).status == KvFlashExec::InvalidPlan);
        CHECK(be.calls.empty());
    }
}

TEST_CASE(KvflashStepExecFixture, an_idle_step_is_just_end_step) {
    Rec be;
    CHECK(kvflash_execute_step(kvflash_plan_step({}), be).status == KvFlashExec::Ok);
    CHECK(be.seen.empty());
    CHECK(be.calls.size() == 1);
}
