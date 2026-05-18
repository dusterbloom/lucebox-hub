// mtp_shape_selector.h — entropy-adaptive speculation shape.
//
// Pure logits-in / shape-out: given the MTP head's top-K logprobs for the
// FIRST draft position, decide chain / tree / AR-only for this iter.
// No module pointer, no global state, no dependency on INativeMtp.

#pragma once

namespace dflash27b {
namespace mtp {

enum class SpecShapeKind { Chain, Tree, ArOnly };

struct SpecShape {
    SpecShapeKind kind         = SpecShapeKind::Chain;
    int           gamma        = 3;    // Chain/Tree: γ depth. ArOnly: 1.
    int           branches     = 1;    // Tree only: B siblings.
    int           topk         = 1;    // Tree only: K per node.
    float         entropy_bits = 0.0f; // Shannon entropy (bits) that drove the decision.
};

struct SpecShapeConfig {
    float h_low      = 10.0f;  // default: Tree never fires; Qwen3.6 sweep shows no winning threshold at B=2 K=2
    float h_high     = 100.0f;
    int   chain_gamma = 3;
    int   tree_B      = 2;
    int   tree_K      = 2;
    int   tree_gamma  = 2;

    // Read DFLASH27B_MTP_H_LOW / _H_HIGH; clamp to sane bounds and ensure
    // h_low ≤ h_high. Other fields keep their defaults.
    static SpecShapeConfig from_env();
};

// `topk_logprobs` is K values sorted DESCENDING (rank 0 == argmax). The
// selector applies a stable softmax + Shannon entropy (bits) internally —
// values may be raw logits or already-log-softmax, entropy is invariant
// to a global shift. Returns ShapeArOnly defensively for null / k ≤ 0.
SpecShape select_spec_shape(const float * topk_logprobs, int k,
                            const SpecShapeConfig & cfg);

}  // namespace mtp
}  // namespace dflash27b
