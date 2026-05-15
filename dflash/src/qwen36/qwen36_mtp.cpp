// qwen36_mtp.cpp — see qwen36_mtp.h for contract.
//
// PR 2 skeleton: scaffolds Qwen36MtpModule against the foundation
// interface (INativeMtp). The forward pass (step_batch's real impl) is
// deferred to PR 2b. Today step_batch returns a clearly-marked error so
// callers can detect "Qwen3.6 MTP not yet implemented" without
// segfaulting, and the contract test exercises the lifecycle (attach,
// reset_chain, shutdown, hidden_size, num_heads, max_gamma).

#include "qwen36_mtp.h"

#include "common/dflash_target.h"

#include "ggml.h"

#include <cstdio>
#include <memory>
#include <string>
#include <vector>

namespace dflash27b::mtp {

struct Qwen36MtpModule::State {
    Qwen36MtpWeights  weights;
    DFlashTarget *    target = nullptr;
    bool              loaded = false;
    bool              attached = false;
    // PR 2b: persistent ggml_context + cgraph + gallocr for the MTP
    // forward will live here. Today the state only carries metadata so
    // the module compiles into dflash27b without dragging the qwen35
    // backbone graph machinery yet.
};

Qwen36MtpModule::Qwen36MtpModule() : state_(std::make_unique<State>()) {}
Qwen36MtpModule::~Qwen36MtpModule() = default;

bool Qwen36MtpModule::init(const std::string & gguf_path,
                           DFlashTarget * target,
                           std::string & out_error) {
    if (!target) {
        out_error = "Qwen36MtpModule::init: target is null";
        return false;
    }
    // PR 2b will use a backend-allocated ctx co-owned with the qwen35
    // backbone. For PR 2 we open a transient ctx for tensor discovery.
    struct ggml_init_params ip{};
    ip.mem_size   = 32 * 1024 * 1024;
    ip.mem_buffer = nullptr;
    ip.no_alloc   = true;
    ggml_context * ctx = ggml_init(ip);
    if (!ctx) {
        out_error = "Qwen36MtpModule::init: ggml_init failed";
        return false;
    }

    // We don't know n_vocab here without the backbone; pass 0 to skip
    // the cross-check. PR 2b will plumb backbone n_vocab through.
    const bool ok = load_qwen36_mtp_weights(
        gguf_path, ctx,
        /*expected_n_embd=*/target->hidden_size(),
        /*expected_n_vocab=*/0,
        state_->weights, out_error);
    ggml_free(ctx);

    if (!ok) return false;
    state_->loaded = true;
    return attach(target);
}

int  Qwen36MtpModule::max_gamma()   const { return state_->weights.n_heads; }
int  Qwen36MtpModule::hidden_size() const { return state_->weights.n_embd; }
int  Qwen36MtpModule::num_heads()   const { return state_->weights.n_heads; }

bool Qwen36MtpModule::attach(DFlashTarget * target) {
    if (!target) return false;
    if (state_->loaded && target->hidden_size() != state_->weights.n_embd) {
        std::fprintf(stderr,
            "[qwen36_mtp] hidden_size mismatch (target=%d, mtp=%d)\n",
            target->hidden_size(), state_->weights.n_embd);
        return false;
    }
    state_->target   = target;
    state_->attached = true;
    return true;
}

void Qwen36MtpModule::reset_chain() {
    // No persistent per-chain state in the PR 2 skeleton.
}

void Qwen36MtpModule::shutdown() {
    state_->target   = nullptr;
    state_->attached = false;
    state_->loaded   = false;
    state_->weights  = {};
}

bool Qwen36MtpModule::step_batch(int32_t /*current_token*/,
                                 int /*base_pos*/,
                                 std::vector<StepOutput> & out) {
    // PR 2b: implement the NextN forward.
    //
    //   for h in 0..n_heads:
    //     e = embed(cur) via head.embed_tokens (if non-null) else
    //         backbone tok_embd
    //     h_in = norm(last_hidden, head.hnorm)
    //     e_in = norm(e, head.enorm)
    //     x = head.eh_proj @ concat(e_in, h_in)
    //     x = transformer_block(layer_idx, x, base_pos)    // attn + ffn
    //     x = norm(x, head.shared_head_norm or backbone out_norm)
    //     logits = (head.shared_head_head or backbone output) @ x
    //     draft_h = argmax(logits)
    //     cur = draft_h     // for next head's input
    //
    // Requires:
    //   - access to the same KV cache the backbone uses (Qwen35Backend)
    //   - persistent StepGraph (or one per head) with shared gallocr
    //   - host-side flow through DFlashTarget::embed_tokens for the
    //     initial embedding and project_hidden_to_tokens for the final
    //     logits when shared_head_head is absent
    //
    // The empty `out` is the explicit signal to MtpChainRunner that no
    // drafts were produced; the runner will fall back to a γ=0 iteration
    // (target-only single-token forward) so the loop does not stall.
    out.clear();
    std::fprintf(stderr,
        "[qwen36_mtp] step_batch: forward not implemented in PR 2 — "
        "see PR 2b for the NextN forward.\n");
    return false;
}

const Qwen36MtpWeights & Qwen36MtpModule::weights() const {
    return state_->weights;
}

}  // namespace dflash27b::mtp
