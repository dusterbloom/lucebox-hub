// Runs one planned KVFlash step (kvflash_plan_step) against a backend.
// Pure control flow: the backend does the device work (C1.3-03). The
// executor owns the invariants the plan only describes: plan shape checked
// before anything runs, one Fence, EndStep on success, abort_step (D3) on the
// first failure, nothing after a failure.

#pragma once

#include "kvflash_slot_pager.h"

namespace luce::common {

class KvFlashStepBackend {
public:
    virtual ~KvFlashStepBackend() = default;
    virtual bool run(const KvFlashOp & op) = 0;   // Pin..Score; false = failed
    virtual void end_step() = 0;                  // pins expire
    virtual void abort_step() = 0;                // unpin, settle in-flight copies, release; idempotent
};

enum class KvFlashExec { Ok, InvalidPlan, Failed };

struct KvFlashExecResult {
    KvFlashExec status = KvFlashExec::Ok;
    size_t failed_at = 0;                         // index of the op that failed
};

// A plan is valid when it ends in exactly one EndStep, has at most one Fence
// and one Score, and every Score comes after the Fence when there is one.
inline bool kvflash_plan_valid(const std::vector<KvFlashOp> & ops) {
    if (ops.empty() || ops.back().kind != KvFlashOpKind::EndStep) return false;
    int fences = 0, scores = 0, ends = 0;
    bool score_before_fence = false;
    for (const auto & o : ops) {
        if (o.kind == KvFlashOpKind::Fence) fences++;
        if (o.kind == KvFlashOpKind::Score) { scores++; if (fences == 0) score_before_fence = true; }
        if (o.kind == KvFlashOpKind::EndStep) ends++;
    }
    const bool has_fence = fences > 0;
    return ends == 1 && fences <= 1 && scores <= 1 && !(has_fence && score_before_fence);
}

inline KvFlashExecResult kvflash_execute_step(const std::vector<KvFlashOp> & ops, KvFlashStepBackend & be) {
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
