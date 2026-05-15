// qwen36_mtp.h — Native-heads MTP module for unsloth Qwen3.6 GGUFs.
//
// Qwen3.6's GGUF architecture is `qwen35` (same backbone). The MTP heads are
// stored as additional per-layer tensors on the last `qwen35.nextn_predict_layers`
// blocks of the GGUF, following DeepSeek-V3 / NextN conventions:
//
//   blk.{bid}.nextn.eh_proj          : [2*n_embd, n_embd]   — concat[embed;hidden] -> embed
//   blk.{bid}.nextn.enorm            : [n_embd]              — embed-side norm
//   blk.{bid}.nextn.hnorm            : [n_embd]              — hidden-side norm
//   blk.{bid}.nextn.embed_tokens     : [n_embd, n_vocab]     — optional; shared with backbone if absent
//   blk.{bid}.nextn.shared_head_head : [n_embd, n_vocab]     — optional; shared with backbone if absent
//   blk.{bid}.nextn.shared_head_norm : [n_embd]              — optional; shared with backbone if absent
//
// Each MTP head also carries its own transformer block tensors (attn_q/k/v/o,
// ffn_*, ssm_*) at the same layer index — these were already loaded for the
// backbone forward (qwen35 path). The MTP forward reuses them via the
// supplied Qwen35Backend / DFlashTarget.
//
// Per unsloth: recommended γ ≤ 2 (accept rate drops from ~83% at γ=1 to
// ~50% at γ=4). `max_gamma()` honors `nextn_predict_layers` from GGUF.
//
// This module implements INativeMtp from mtp_interface.h: `step_batch` emits
// up to num_heads() draft tokens per call.

#pragma once

#include "common/mtp_interface.h"

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

struct ggml_tensor;
struct ggml_context;

namespace dflash27b {

struct DFlashTarget;

namespace mtp {

// One Qwen3.6 MTP head's weights. There are `n_heads` such entries; head i
// corresponds to GGUF block index `(n_layer - n_heads + i)`.
struct Qwen36MtpHeadWeights {
    int layer_idx     = -1;            // absolute GGUF block index
    ggml_tensor * eh_proj          = nullptr;   // required
    ggml_tensor * enorm            = nullptr;   // required
    ggml_tensor * hnorm            = nullptr;   // required
    ggml_tensor * embed_tokens     = nullptr;   // optional (nullable → use backbone)
    ggml_tensor * shared_head_head = nullptr;   // optional (nullable → use backbone)
    ggml_tensor * shared_head_norm = nullptr;   // optional (nullable → use backbone)
    // The per-head transformer block tensors live on Qwen35Backend at the
    // same `layer_idx`. The module reaches them through the bound target
    // rather than duplicating pointers here.
};

struct Qwen36MtpWeights {
    int                                   n_embd            = 0;
    int                                   n_vocab           = 0;
    int                                   n_heads           = 0;   // == nextn_predict_layers
    int                                   n_backbone_layers = 0;   // total backbone n_layer
    std::vector<Qwen36MtpHeadWeights>     heads;                   // size == n_heads
    // GGUF metadata copies for cross-checks
    std::string                           backbone_arch;           // e.g. "qwen35"
    std::string                           base_model_name;         // e.g. "Qwen3.6-27B"
};

// Load Qwen3.6 MTP weights from a GGUF file. Returns false if the file does
// not contain NextN tensors (i.e. it is a non-MTP GGUF). The `ctx` parameter
// owns the loaded tensors' lifetime; pass the same ctx used for backbone.
//
// `expected_n_embd` / `expected_n_vocab` are sanity-checked against GGUF
// metadata; pass values from the bound target.
bool load_qwen36_mtp_weights(const std::string & gguf_path,
                             ggml_context * ctx,
                             int expected_n_embd,
                             int expected_n_vocab,
                             Qwen36MtpWeights & out_weights,
                             std::string & out_error);

// Concrete INativeMtp impl for Qwen3.6 (unsloth -MTP-GGUF). Wraps the
// loaded weights + a bound DFlashTarget (typically Qwen35Backend's
// dflash_target()) which provides backbone embedding, KV, LM head.
class Qwen36MtpModule : public INativeMtp {
public:
    Qwen36MtpModule();
    ~Qwen36MtpModule() override;

    // One-shot construction: load weights then attach in a single call.
    // The returned module is ready for step_batch() once attach() succeeds.
    bool init(const std::string & gguf_path,
              DFlashTarget * target,
              std::string & out_error);

    // ── IMtpModule ──────────────────────────────────────────────────
    int  max_gamma()    const override;
    int  hidden_size()  const override;
    bool attach(DFlashTarget * target) override;
    void reset_chain() override;
    void shutdown() override;

    // ── INativeMtp ─────────────────────────────────────────────────
    int  num_heads() const override;
    bool step_batch(int32_t current_token,
                    int base_pos,
                    std::vector<StepOutput> & out) override;

    // Read-only access for tests / introspection.
    const Qwen36MtpWeights & weights() const;

private:
    struct State;
    std::unique_ptr<State> state_;
};

}  // namespace mtp
}  // namespace dflash27b
