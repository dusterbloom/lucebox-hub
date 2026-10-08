// Timing-only knockout switches for the hc_* hyper-connection kernel family and the
// quantize_q8_1 producer, used to measure an UPPER BOUND on the launch/compute cost they
// contribute to exact-stack decode. All default OFF (zero behavior change unless set).
//
//   LUCE_KO_HC=1       -- skip the hc_* launches entirely (dst buffers left stale/garbage;
//                          timing bound only, do not use for correctness).
//   LUCE_KO_EMPTY_HC=1 -- replace each hc_* launch with a <<<1,32>>> no-op kernel, isolating
//                          launch+dispatch-gap overhead from the hc kernel's own compute.
//   LUCE_KO_EMPTY_Q8=1 -- same no-op substitution for the plain quantize_q8_1 producer.
//
// Not for production use; gated off by default and read once via a cached static.
#pragma once
#include <cstdlib>

static __global__ void luce_ko_noop_kernel() {}

static inline bool luce_ko_flag(const char * name) {
    static_assert(true, "");
    const char * v = getenv(name);
    return v && v[0] == '1';
}

static inline bool luce_ko_hc() {
    static const bool v = luce_ko_flag("LUCE_KO_HC");
    return v;
}

static inline bool luce_ko_empty_hc() {
    static const bool v = luce_ko_flag("LUCE_KO_EMPTY_HC");
    return v;
}

static inline bool luce_ko_empty_q8() {
    static const bool v = luce_ko_flag("LUCE_KO_EMPTY_Q8");
    return v;
}

// LUCE_QWEN_HC_CN_FAST=1 -- opt-in to the gamma-prefetch variant of hc_combine_norm_f32_b256
// (hc_combine_norm_f32_b256_fast). Same arithmetic, same order, same single launch -- only the
// timing of the gamma global load changes (moved earlier, hidden behind the reduction's
// sync latency instead of issued after it). Default off: byte-identical to today's kernel.
static inline bool luce_hc_cn_fast() {
    static const bool v = luce_ko_flag("LUCE_QWEN_HC_CN_FAST");
    return v;
}
