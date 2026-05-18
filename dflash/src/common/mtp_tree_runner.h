// mtp_tree_runner.h — Tree-MTP γ-loop with B>=2 sibling paths per depth.
//
// Counterpart to MtpChainRunner: instead of one autoregressive draft chain,
// the tree runner fans out B sibling paths per depth, evaluates them in
// parallel against an arena-routed head_kv (via INativeMtp::step_batch_arena),
// builds a DDTree over the per-sibling logprobs, verifies the flat tree on
// the target, and accepts the longest matching path.
//
// B=1 path is preserved for byte-identical fallback: the runner delegates to
// MtpChainRunner.  This is the T6 regression gate — env DFLASH27B_MTP_TREE_B=1
// (default) must produce decode output identical to the chain runner.
//
// The arena slot allocation contract is:
//   slot(path_id, depth) = path_id * gamma_max + depth
// where path_id in [0, B_max) and depth in [0, gamma_max).  The module
// implementing INativeMtp owns the arena tensor and the slot routing inside
// its step graph; the runner only supplies (path_id, depth) per step.

#pragma once

#include "model_backend.h"
#include "mtp_interface.h"
#include "sampler.h"

namespace dflash27b {

struct DFlashTarget;

namespace mtp {

struct MtpTreeStats {
    int total_iters    = 0;
    int total_proposed = 0;
    int total_accepted = 0;
    int total_emitted  = 0;
    int eos_hits       = 0;
};

class MtpTreeRunner {
public:
    MtpTreeRunner(IMtpModule & mtp,
                  DFlashTarget & target,
                  const SamplerCfg & sampler,
                  int B,
                  int K);

    // Drive Tree-MTP decoding.  When B<=1 the runner is a thin shim over
    // MtpChainRunner so the chain regression gate stays byte-identical.
    //
    // last_prefill_token and committed_pos must satisfy the same invariants
    // as MtpChainRunner::run.
    GenerateResult run(const GenerateRequest & req,
                       const DaemonIO & io,
                       int32_t last_prefill_token,
                       int committed_pos,
                       int gamma);

    const MtpTreeStats & stats() const { return stats_; }

private:
    IMtpModule       & mtp_;
    DFlashTarget     & target_;
    SamplerCfg         sampler_cfg_;
    MtpTreeStats       stats_;
    int                B_;     // sibling paths per depth
    int                K_;     // top-K per node from each sibling
};

// Read DFLASH27B_MTP_TREE_B from env; clamp to [1, hard_max].  Returns 1
// (chain mode) when unset or non-numeric.  Exposed so callers (orchestrator,
// tests) all see the same parsing rule.
int env_tree_b(int hard_max = 8);

}  // namespace mtp
}  // namespace dflash27b
