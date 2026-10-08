// Pipelined T=1 greedy autoregressive decode (LUCE_QWEN_PIPELINE=1).
//
// The serial decode loop waits for token n, reads its argmax, builds token n+1's
// inputs on the host and only then launches n+1: the GPU idles for the whole host
// turnaround. This session keeps the greedy pick on the device (argmax -> tok_in ->
// get_rows of a device copy of token_embd), splits the stable T=1 graph into two
// views at the first PLE layer, and enqueues the pre-PLE half of step n+1 before
// blocking on step n. The only host work that depends on token n -- the PLE n-gram
// rows, which live on disk -- runs under the GPU's pre-PLE layers.
//
// Scope: stable T=1 graphs only (dense or stable QSA), greedy, no MTP/verify, no
// hidden export, no shared overlap. Everything else keeps the serial path. Ported
// from branch qwen4exp-pipelined-decode (32f05889) onto the exact 38.41 tree.

#pragma once

#include "qwen4exp_internal.h"
#include "qwen4exp_cache.h"

#include "ggml-backend.h"

#include <cstdint>
#include <vector>

namespace luce::common {

struct Qwen4ExpPipeline {
    ggml_backend_t backend = nullptr;
    const Qwen4ExpWeights * w = nullptr;
    Qwen4ExpCache * cache = nullptr;

    // Device: token_embd copy, the fed token, and pre-PLE linear-layer state
    // snapshots (taken at the head of the lookahead half-step, restored by end()).
    ggml_context * ctx = nullptr;
    ggml_backend_buffer_t buf = nullptr;
    ggml_tensor * tok_embd = nullptr;
    ggml_tensor * tok_in = nullptr;
    int ple_layer = -1;                    // graph split point; n_layer when the model has no PLE
    std::vector<int> snap_layers;          // linear layers before ple_layer
    std::vector<ggml_tensor *> snap_ssm, snap_conv;

    // Pinned host staging ring. Slot i % slots carries step i's PLE rows and pick
    // plus step i+1's position-only inputs and forced token.
    ggml_backend_buffer_t host_buf = nullptr;
    char * host = nullptr;
    size_t slot_bytes = 0;
    size_t off_tok = 0, off_pick = 0, off_pos = 0, off_kv_row = 0, off_params = 0, off_vis = 0, off_ple = 0, off_mask = 0;
    int slots = 0;
    std::vector<ggml_backend_event_t> events;

    // Session.
    bool active = false;
    bool rebuild_ok = false;               // forward_impl may rebuild (the stream is drained)
    bool needs_rebuild = false;            // set by forward_impl when the stable graph does not fit
    int pos0 = 0;                          // position of step 0
    int enqueued_a = -1;                   // last step whose pre-PLE half is enqueued
    int enqueued_b = -1;                   // last step whose post-PLE half is enqueued
    int synced = 0;                        // picks[0, synced) are on the host and committed
    std::vector<int32_t> inputs;           // x_i: input token of step i (inputs[0] = begin token)
    std::vector<int32_t> picks;            // argmax of step i, valid below `synced`
    std::vector<int32_t> ple_prev;         // rolling n-gram window ahead of the committed cache
    std::vector<int32_t> ple_prev_base;    // cache.ple_prev at begin, for exact commit
};

// Allocates the session resources once per cache: the token_embd upload (~0.7 GB
// for a 248k x 2560 Q8_0 table), staging ring, events. Null when unsupported.
Qwen4ExpPipeline * qwen4exp_pipeline_create(ggml_backend_t backend, const Qwen4ExpWeights & w,
                                            Qwen4ExpCache & cache, int slots = 8);
void qwen4exp_pipeline_destroy(Qwen4ExpPipeline * pipe);

// Start at cache.cur_pos with input token x0: (re)builds the split stable graph
// and enqueues step 0's pre-PLE half.
bool qwen4exp_pipeline_begin(Qwen4ExpPipeline & pipe, int32_t x0);

// Complete step i (uploads its PLE rows, enqueues the post-PLE half and the pick
// readback) and speculatively enqueue step i+1's pre-PLE half. `next` is step i+1's
// input when known ahead (forced ids); null feeds step i's greedy pick on the
// device -- the caller must then qwen4exp_pipeline_wait(i) before the next step.
bool qwen4exp_pipeline_step(Qwen4ExpPipeline & pipe, const int32_t * next);

// Pick of step i; blocks until it is computed and commits the cache through step i.
bool qwen4exp_pipeline_wait(Qwen4ExpPipeline & pipe, int i, int32_t & pick);

// Drains the stream and rolls back the speculative half-step past the last
// completed step. The cache then reads as after the last waited step.
bool qwen4exp_pipeline_end(Qwen4ExpPipeline & pipe);

}  // namespace luce::common
