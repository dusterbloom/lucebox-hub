// Runs one planned KVFlash step (kvflash_plan_step) against a backend.
// Pure control flow: the backend does the device work. The
// executor owns the invariants the plan only describes: plan shape checked
// before anything runs, one Fence, EndStep on success, abort_step (D3) on the
// first failure, nothing after a failure.

#pragma once

#include "kvflash_slot_plan.h"

namespace luce::common {

enum class KvFlashExec { Ok, InvalidPlan, Failed };

struct KvFlashExecResult {
    KvFlashExec status = KvFlashExec::Ok;
    size_t failed_at = 0;                         // index of the op that failed
};

// A plan is valid when it ends in exactly one EndStep and has at most one Fence.
inline bool kvflash_plan_valid(const std::vector<KvFlashOp> & ops) {
    if (ops.empty() || ops.back().kind != KvFlashOpKind::EndStep) return false;
    int fences = 0, ends = 0;
    for (const auto & o : ops) {
        if (o.kind == KvFlashOpKind::Fence) fences++;
        if (o.kind == KvFlashOpKind::EndStep) ends++;
    }
    return ends == 1 && fences <= 1;
}

// ponytail: Backend is any type with bool run(const KvFlashOp &) (Pin..Fence; false = failed), void end_step() (pins
// expire) and void abort_step() (unpin, settle in-flight copies, release; idempotent). The one production backend is
// KvFlashStepDriver; the test fake is the second. Upgrade to a virtual base only when a second runtime-selected backend exists.
template <class Backend>
inline KvFlashExecResult kvflash_execute_step(const std::vector<KvFlashOp> & ops, Backend & be) {
    KvFlashExecResult r;
    if (!kvflash_plan_valid(ops)) { r.status = KvFlashExec::InvalidPlan; return r; }
    for (size_t i = 0; i + 1 < ops.size(); i++) {      // the last op is EndStep
        if (!be.run(ops[i])) {
            be.abort_step();
            r.status = KvFlashExec::Failed;
            r.failed_at = i;
            return r;
        }
    }
    be.end_step();
    return r;
}

}  // namespace luce::common
