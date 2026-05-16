# Qwen3.6 MTP — Shape B Redesign

## Context

Phase B of the Qwen3.6 MTP work (`af2f634`, `e845b32`) introduced `IBackboneBlock` and
`Qwen35BackboneBlock` on the assumption that each NextN head reuses the backbone's per-layer
transformer-block forward at the head's `layer_idx`. Tensor inspection of the GGUF on disk
proves that assumption wrong: each head owns its own full transformer-block weights at blocks
beyond the backbone range, not shared with the backbone path at all. This document captures
the correct architecture (Shape B, head-owned weights) and locks the PR sequence for the
redesign before any code is reverted or rewritten.

Source material: `thoughts/shared/handoffs/general/2026-05-15_22-46-33_qwen36-mtp-phase-b-redesign-needed.md`.
Outer plan: `/home/peppi/.claude/plans/dynamic-swinging-adleman.md` (3-PR structure unchanged;
only the per-head Qwen3.6 shape changes). Do not modify that plan file directly; cross-link
from the relevant PR commits.

The foundation layer — `IMtpModule`, `INativeMtp`, `MtpChainRunner`
(`dflash/src/common/mtp_interface.h`, `mtp_chain_runner.{h,cpp}`) — is correct and stays.
Only the per-head Qwen3.6 weight layout and forward path are redesigned.

**Verification status**: all GGUF-dependent numbers in this doc are sourced from `gguf-dump` on the actual file (`/home/peppi/models/qwen3.6-27b-mtp/Qwen3.6-27B-UD-Q2_K_XL.gguf`). The "Verified GGUF Constants" section below is the contract; if a future GGUF disagrees, this doc needs to be re-derived from that file before any code change.

---

## Architectural Reality (Shape B)

Per DeepSeek-V3 §2.2 Eq 21-23 (https://arxiv.org/html/2412.19437v1):

```
h'_i^(k) = M_k · [ RMSNorm(h_i^(k-1)) ; RMSNorm(Emb(t_{i+k})) ]
h_i^(k)  = TRMBlock_k( h'_i^(k) )
p_i^(k)  = OutHead( h_i^(k) )
```

Where:
- `k` indexes heads from 1 to `n_heads` (γ ceiling).
- `h_i^(0)` is the backbone's final post-norm hidden after the last committed token.
- `h_i^(k-1)` for `k > 1` is the previous head's output hidden (un-normed, same dim as backbone).
- `M_k` is `eh_proj.weight` at the head's GGUF block — a `[n_embd, 2*n_embd]` projection.
- `TRMBlock_k` is the head's **own** attn+ffn block, loaded from the same GGUF block as `M_k`.
- `OutHead` = `shared_head_norm` followed by the **shared** LM head (`output.weight` /
  `output_norm.weight` from the backbone). Drafts project through the same final head as
  committed tokens; no per-head LM head unless `nextn.shared_head_head` is present.
- Each head has its **own KV cache**, sized `gamma_max × head_n_kv × head_dim_kv`. Using the
  backbone's KV cache breaks correctness (per llama.cpp PR #22673 design note:
  https://github.com/ggml-org/llama.cpp/pull/22673).

### Tensor dump — why this is Shape B and not Shape A

`strings -n 12 /home/peppi/models/qwen3.6-27b-mtp/Qwen3.6-27B-UD-Q2_K_XL.gguf | grep -E '^blk\.(63|64)\.' | sort -u` shows:

```
blk.63.attn_k.weight          blk.63.attn_norm.weight
blk.63.attn_output.weight     blk.63.attn_q.weight
blk.63.attn_q_norm.weight     blk.63.attn_k_norm.weight
blk.63.attn_v.weight          blk.63.ffn_down.weight
blk.63.ffn_gate.weight        blk.63.ffn_up.weight
blk.63.post_attention_norm.weight

blk.64.attn_k.weight          blk.64.attn_norm.weight
blk.64.attn_output.weight     blk.64.attn_q.weight
blk.64.attn_q_norm.weight     blk.64.attn_k_norm.weight
blk.64.attn_v.weight          blk.64.ffn_down.weight
blk.64.ffn_gate.weight        blk.64.ffn_up.weight
blk.64.post_attention_norm.weight
blk.64.nextn.eh_proj.weight   blk.64.nextn.enorm.weight
blk.64.nextn.hnorm.weight     blk.64.nextn.shared_head_norm.weight
```

Findings:
- `blk.63`: full transformer block, **no** `nextn.*` tensors → last **backbone** layer
  (backbone has 63 layers: blocks 0–62 are SSM/hybrid, block 63 is the final attention layer).
  `nextn_predict_layers` from GGUF metadata is `1`; divisibility fix in
  `gguf_target_loader.cpp:282-287` correctly computes `n_layer = block_count - 1 = 64 - 1 = 63`.
- `blk.64`: full transformer block PLUS `nextn.*` tensors → **1 NextN head**.

This GGUF therefore has `n_heads = 1` and `max_gamma = 1`. The earlier handoff's
`max_gamma = 2` was wrong; the root cause was that `materialize_mtp_tensors` did not load
the head's transformer-block weights, leaving tensor pointers null and making the loader
find only 0 materialized heads. See "Open questions" for the unsloth γ=2 discrepancy.

### llama.cpp PR #22673 design notes (am17an)

- MTP heads run as a **post-ubatch hook**, not inline in the main decode graph.
- Each head has its own context and KV cache (memory overhead <10% for 1–2 heads).
- Draft acceptance rate ~75% at 3 draft tokens in their multi-head GGUFs → >2× speedup.
- `n_parallel = 1` is a known limitation of their implementation.
- Integration point: `--spec-type draft-mtp` in their CLI; equivalent in dflash is the
  `MtpChainRunner` + `DFlashTarget::verify_batch` path already in place.

---

## Verified GGUF Constants (Qwen3.6-27B-UD-Q2_K_XL.gguf)

All values dumped from the actual GGUF via `gguf-dump`. No assumptions, no inference.

| Key | Value |
|---|---|
| `general.architecture` | `qwen35` (namespace is qwen35, despite the "Qwen3.6-27B" name) |
| `qwen35.block_count` | 65 |
| `qwen35.nextn_predict_layers` | **1** |
| `qwen35.embedding_length` | 5120 |
| `qwen35.attention.head_count` | 24 |
| `qwen35.attention.head_count_kv` | 4 (GQA, 6:1) |
| `qwen35.attention.key_length` | 256 |
| `qwen35.attention.value_length` | 256 |
| `qwen35.full_attention_interval` | 4 (hybrid; every 4th block is full attention, rest are SSM) |
| `qwen35.feed_forward_length` | 17408 |
| `qwen35.rope.dimension_count` | 64 |
| `qwen35.rope.freq_base` | 1e7 |
| `GGUF.tensor_count` | 866 |

`max_gamma = 1` for this GGUF, full stop. The unsloth γ=2 doc claim refers to a different variant (likely 35B-A3B MoE); the 27B Dense ships with exactly one NextN head.

The backbone is **hybrid SSM/Mamba + attention**:
- blocks 0..63 are backbone. Of those, blocks 3, 7, 11, ..., 63 (i.e. `i % 4 == 3`) are full-attention blocks with split `attn_q/k/v/output`; the rest are SSM blocks with `attn_qkv` (fused), `ssm_*` tensors. This is what `qwen35.full_attention_interval = 4` means.
- block 64 is the NextN head. Its transformer block is **attention-style**, not SSM (verified: it has `attn_q/k/v/output`, not `attn_qkv` or `ssm_*`).

This matters for the redesign only at the boundary: `h_prev_0` comes from the backbone's final post-norm hidden state, which `DFlashTarget` already exposes. The head's forward uses its own attention block — no SSM code path needed inside the head.

### Verified blk.64 (NextN head) tensor inventory

```
blk.64.attn_norm.weight                  F32      [5120]
blk.64.attn_q.weight                     Q2_K     [5120, 12288]   ← head's own attention
blk.64.attn_q_norm.weight                F32      [256]           ← QK-norm
blk.64.attn_k.weight                     Q2_K     [5120, 1024]
blk.64.attn_k_norm.weight                F32      [256]
blk.64.attn_v.weight                     Q3_K     [5120, 1024]
blk.64.attn_output.weight                Q3_K     [6144, 5120]
blk.64.post_attention_norm.weight        F32      [5120]
blk.64.ffn_gate.weight                   Q2_K     [5120, 17408]   ← head's own FFN
blk.64.ffn_up.weight                     Q2_K     [5120, 17408]
blk.64.ffn_down.weight                   Q3_K     [17408, 5120]
blk.64.nextn.eh_proj.weight              Q8_0     [10240, 5120]   ← M_k: 2d → d
blk.64.nextn.enorm.weight                F32      [5120]
blk.64.nextn.hnorm.weight                F32      [5120]
blk.64.nextn.shared_head_norm.weight     F32      [5120]
```

Absent in the GGUF (confirmed shared with backbone):
- `blk.64.nextn.embed_tokens` — none. Head uses backbone's `token_embd.weight`.
- `blk.64.nextn.shared_head_head` — none. Head projects through backbone's `output.weight`.

Per-head KV cache sizing: `head_count_kv × key_length × γ_max × bytes` = `4 × 256 × 1 × 2 (fp16)` = **2 KB** per position. With max chain length n_tokens ≈ 256, KV cost is ~512 KB. Negligible.

---

## What We Keep from the Prior PR Stack

- `dflash/src/common/mtp_interface.h` (PR 1) — keep, no change.
- `dflash/src/common/mtp_chain_runner.{h,cpp}` (PR 1) — keep. The `recommit` failure at
  `mtp_chain_runner.cpp:198` flagged in the handoff is likely a cascade from `max_gamma=1`
  causing the propose step to return 0 drafts; re-test after PR 2d-bis. If still failing,
  file as a separate task — it is independent of the Shape B change.
- `dflash/src/common/gguf_mmap.h`, `dflash/src/common/gguf_metadata.h` — keep (loader
  helpers, extracted in `cc8a88b`).
- `dflash/src/qwen35/gguf_target_loader.cpp:282-287` — keep. Divisibility fix
  (`n_layer = block_count - nextn_predict_layers`) is uncommitted but correct; include in
  PR 2c-bis commit.
- `dflash/src/qwen36/qwen36_mtp.{h,cpp}` skeleton — keep the class layout
  (`Qwen36MtpModule`, `State`, Phase A CPU forward, `materialize_mtp_tensors` helper,
  `attach_weights_for_test`); redesign the internals per the sections below.
- `dflash/src/qwen36/qwen36_mtp_loader.cpp` NextN tensor binding — keep, extend (PR 2c-bis).
- `dflash/src/qwen35/qwen35_backend.{h,cpp}` `tensor_context()` getter (codex WIP,
  uncommitted) — keep; required so `test_dflash.cpp` can pass the backbone's ggml_context
  to the MTP loader without a second `gguf_init_from_file` call.
- `dflash/test/test_dflash.cpp` `run_qwen36_mtp_harness` (lines 646–792, uncommitted) —
  keep; AR path already works at ~37 tok/s.
- `dflash/test/test_qwen36_mtp_step_unit.cpp` 8 T1 cases — keep, all green.
- `dflash/test/test_qwen36_mtp_e2e.sh` (uncommitted) — keep; MTP cell will populate once
  PR 2d-bis lands.
- `materialize_mtp_tensors` helper in `qwen36_mtp.cpp` — keep the structure, fix its tensor
  scope (PR 2c-bis).

---

## What We Revert / Replace

- **Revert `e845b32`** — `dflash/src/qwen35/qwen35_backbone_block.cpp` (+184 LoC):
  implements `run_block` by calling into the backbone's per-layer graph builder, which is
  the wrong forward for Shape B. Entire file goes away.
- **Revert `af2f634`** — removes:
  - `dflash/src/common/backbone_block.h` (wrong interface premise: heads share backbone weights)
  - `dflash/src/qwen35/qwen35_backbone_block.{h,cpp}` (the wrong adapter)
  - `dflash/src/qwen35/qwen35_backend.{h,cpp}` `backbone_block()` getter and the
    `Qwen35BackboneBlock` member it returns
  - `dflash/src/qwen36/qwen36_mtp.{h,cpp}` `attach_backbone_block` member and
    `state_->backbone_block` field
  - `dflash/test/test_qwen36_mtp_step_unit.cpp` cases 7 and 8 that exercise
    `attach_backbone_block` (cases 1–6 stay)

Recommended approach: two clean `git revert` commits (`git revert e845b32` then
`git revert af2f634`) rather than editing the wrong files in PR 2c-bis. Reverting produces
a reviewable single-purpose diff that clearly says "this design was wrong". An alternative
is keeping the file shells and surgically removing the wrong members in PR 2c-bis; prefer
the clean revert path unless cherry-pick conflicts make it untenable.

---

## PR Sequence (Shape B)

### PR 2c-bis — Head-owned weights, loader, KV alloc

**Goal:** Expand `Qwen36MtpHeadWeights` with the head's full transformer-block tensors,
wire the loader to bind them, and fix `materialize_mtp_tensors` so it actually loads them.
After this PR: `cmake --build` green, T1 suite green, head tensor pointers non-null on
real GGUF.

**`dflash/src/qwen36/qwen36_mtp.h`**

Expand `Qwen36MtpHeadWeights`:
```cpp
struct Qwen36MtpHeadWeights {
    int layer_idx = -1;
    // NextN-specific tensors (required)
    ggml_tensor * eh_proj          = nullptr;
    ggml_tensor * enorm            = nullptr;
    ggml_tensor * hnorm            = nullptr;
    ggml_tensor * shared_head_norm = nullptr;   // optional → backbone fallback
    // Head-owned transformer block (required — not shared with backbone)
    ggml_tensor * attn_norm           = nullptr;
    ggml_tensor * attn_q             = nullptr;
    ggml_tensor * attn_k             = nullptr;
    ggml_tensor * attn_v             = nullptr;
    ggml_tensor * attn_output        = nullptr;
    ggml_tensor * attn_q_norm        = nullptr;
    ggml_tensor * attn_k_norm        = nullptr;
    ggml_tensor * post_attention_norm = nullptr;
    ggml_tensor * ffn_gate           = nullptr;
    ggml_tensor * ffn_up             = nullptr;
    ggml_tensor * ffn_down           = nullptr;
};
```

Replace `attach_backbone_block(IBackboneBlock *)` with a method to receive the backbone's
final hidden state:
```cpp
void set_initial_hidden(const float * h_prev, int dim);
```
Rationale: the backbone hands off `h_prev_0` once per chain step (after its own last-layer
forward); a direct setter on `State::last_hidden` is simpler and avoids a virtual callback
layer. The alternative (callback registration) adds indirection without value since the
caller owns both the backbone and this module.

**`dflash/src/qwen36/qwen36_mtp_loader.cpp`**

In the per-head binding loop, add binds for the 11 new tensors using the same `bind(name,
required)` lambda. Required tensors: `attn_norm`, `attn_q`, `attn_k`, `attn_v`,
`attn_output`, `attn_q_norm`, `attn_k_norm`, `post_attention_norm`, `ffn_gate`, `ffn_up`,
`ffn_down`. These live at `blk.{layer_idx}.{tensor_name}.weight` — no `nextn.` prefix.

**`dflash/src/qwen36/qwen36_mtp.cpp` — `materialize_mtp_tensors`**

Add all 15 per-head tensor pointers (4 NextN + 11 transformer-block) to the `tensors`
vector that feeds the buffer allocation + mmap-read loop. Currently only `eh_proj`,
`enorm`, `hnorm`, `shared_head_norm` are included; the transformer-block weights remain
nullptr after `init()`, causing a crash on any head forward. This single omission explains
`max_gamma = 1` (no head can be found valid) and the `recommit` cascade.

**KV cache allocation**

Add a per-head CUDA-side buffer to `State`:
```cpp
struct HeadKvBuffer {
    ggml_backend_buffer_t buf = nullptr;
    ggml_tensor * k_cache = nullptr;
    ggml_tensor * v_cache = nullptr;
};
std::vector<HeadKvBuffer> head_kv;
```
Sized `gamma_max × head_n_kv × head_dim_kv × sizeof(ggml_fp16_t)` per head (or q8_0 to
match the backbone's KV type). Allocate in `init()`, free in `shutdown()`. Head-owned KV
is not shared with backbone KV; writes at position 0 (the single draft slot for a 1-head
chain).

**Tests (T1 only this PR)**

Extend `dflash/test/test_qwen36_mtp_step_unit.cpp` with 2 cases:
1. `materialize_mtp_tensors_head_block_ptrs_nonnull`: constructs a fake `Qwen36MtpWeights`
   with all 15 tensor pointers set to non-null stubs; calls `materialize_mtp_tensors` on
   a synthetic 1-head GGUF fragment; asserts all pointers remain non-null after the call.
2. `set_initial_hidden_propagates`: calls `set_initial_hidden` with a known float array,
   then inspects `state_->last_hidden` to assert the values were copied.

No GGUF I/O test in this PR; real-GGUF validation is PR 2d-bis.

---

### PR 2d-bis — Head transformer-block forward

**Goal:** Build the per-head forward graph in `Qwen36MtpModule::step_batch` using the
head's own weights. After this PR: `test_dflash --mtp-gguf ...` runs without crash and
emits a non-zero MTP tok/s.

**`dflash/src/qwen36/qwen36_mtp.cpp` — `step_batch`**

Replace the current Phase A + Phase B branch with the Shape B forward. For each head `k`:

1. Read `h_prev` from `state_->last_hidden` (set via `set_initial_hidden` for `k=0`,
   or written by the previous head for `k>0`).
2. Embed `cur_or_drafted_tok` via `state_->target->embed_tokens(...)` (shared backbone
   embedding; `embed_tokens.weight` is the backbone's token table).
3. `e_in = RMSNorm(embed_buf, head.enorm)` and `h_in = RMSNorm(h_prev, head.hnorm)`.
4. Concatenate `[h_in; e_in]` (hidden first, embed second — matches `eh_proj` shape
   `[n_embd, 2*n_embd]` where rows = output dim, cols = `n_embd(hidden) + n_embd(embed)`).
5. `x = eh_proj @ concat` → `[n_embd]`.
6. TRMBlock forward on `x` using head's own tensors:
   - `cur = RMSNorm(x, head.attn_norm)`
   - `q = head.attn_q @ cur`; `k = head.attn_k @ cur`; `v = head.attn_v @ cur`
   - Apply `head.attn_q_norm` and `head.attn_k_norm` (per-head RMSNorm, Qwen3 style).
   - RoPE at draft position `base_pos + k` (use the same RoPE helper as backbone).
   - Attention against head's **own** KV cache (`head_kv[k]`), writing at position 0
     for this chain step.
   - `attn_out = head.attn_output @ attn_result`; residual `x = x + attn_out`.
   - `cur = RMSNorm(x, head.post_attention_norm)`
   - SwiGLU: `ffn = head.ffn_down @ (silu(head.ffn_gate @ cur) * (head.ffn_up @ cur))`
   - Residual: `x = x + ffn`.
7. `x_normed = RMSNorm(x, head.shared_head_norm)`.
8. Draft logits via `state_->target->project_hidden_to_tokens(x_normed, 1, ...)`.
   Argmax → `draft_token`.
9. `state_->last_hidden = x` (un-normed, for next head's `h_in`).

Build using `ggml_*` operators consistent with the existing `qwen35_layer_forward`
patterns in `qwen35/`. Do NOT call into `qwen35_backend.cpp` or any backbone graph
builder — that is what made Phase B wrong. The head's forward is self-contained.

KV usage: each head writes to its own `head_kv[k]` at offset 0. For a 1-head chain
(`max_gamma=1`), only `head_kv[0]` is used; it holds 1 KV slot.

**Tests (T2/T3)**

Real-GGUF validation lives at T2 (`smoke_qwen36_mtp_load.cpp`, skipped when
`QWEN36_MTP_GGUF` unset) and T3 (`test_qwen36_mtp_e2e.sh`). There is no meaningful T1
for the transformer-block forward without a real GGUF: the eh_proj + attn + ffn weights
are too large to stub synthetically. The 8 existing T1 cases in
`test_qwen36_mtp_step_unit.cpp` remain as-is (they cover Phase A math and `set_initial_hidden`).

---

### PR 2e-final — Wire to test_dflash, bench

**`dflash/test/test_dflash.cpp:718`**

Replace the commented-out `attach_backbone_block(backend.backbone_block())` line with:
```cpp
// After prefill, feed backbone's final hidden to the MTP module.
// Backbone exposes this via DFlashTarget::last_hidden() (to be added in PR 2c-bis
// or wired through a post-prefill callback from Qwen35Backend).
mtp_module->set_initial_hidden(target->last_hidden(), target->hidden_size());
```
The exact wiring depends on whether `DFlashTarget` grows a `last_hidden()` accessor or
the harness calls a new `Qwen35Backend` method. Settle this in PR 2c-bis; the comment
here is the stake.

**Bench**

Run `bash dflash/test/test_qwen36_mtp_e2e.sh` per `BENCH.md` protocol (5 prompts × 3
runs × 2 cells, paired A/B, JSON to `dflash/bench/results/`):
- AR baseline cell: target ~37 tok/s (±5%). Must be within this range; if not,
  investigate thermal / power state before attributing to code.
- MTP cell (γ=1 with 1 head): llama.cpp PR #22673 reports >2× at 3 draft tokens on a
  multi-head GGUF. With 1 head, expected gain is more modest — a single draft token
  accepted ~75% of the time gives roughly 1.5× in theory. Accept any result > 0.5× AR
  as "non-degenerate"; the bench number goes into the placeholder table in
  `dynamic-swinging-adleman.md`.

Fill the placeholder table in `/home/peppi/.claude/plans/dynamic-swinging-adleman.md` only
after the bench script produces a complete JSON. Do not hand-edit the table.

---

## Open Questions

1. **27B Dense ships with `nextn_predict_layers=1`** — confirmed via `gguf-dump`. The unsloth γ=2 recommendation in their docs must refer to the 35B-A3B MoE variant or a different quant we don't have on disk. **No action needed**, but bench targets are pinned to γ=1 for this GGUF. If a multi-head 27B variant appears later, scale up `max_gamma` accordingly.

2. **35B-A3B MoE GGUF not on disk**: the MoE variant is not present
   (`dflash/test/test_qwen36_mtp_e2e.sh` skips the MoE cells cleanly). Decide whether to
   download (~12 GB) for this milestone or gate MoE support on PR 3. Per current priority
   ("beat baseline on 3090 first"), recommendation is to skip for now.

3. **Per-head KV cache sizing on RTX 3090** — pinned: `n_heads=1`, `γ_max=1`, `head_count_kv=4`, `key_length=value_length=256`. fp16 KV = 4 KB per position; over a 256-token chain that's 1 MB. Negligible vs 12 GB model. Verify the CUDA layout matches what `DFlashTarget::verify_batch` expects when chaining; chain runner's `snapshot_kv`/`restore_kv` must NOT touch the head's KV (head KV is implicitly reset at `reset_chain()`).

4. **`recommit` failure at `mtp_chain_runner.cpp:198`**: the handoff flags this as failing
   after the divisibility fix. Most likely a cascade from `max_gamma=1` → propose returns
   0 drafts → the commit sequence is `[cur_tok, bonus]` but `all_argmax` has only 1 element
   (from a 1-token `verify_batch`), producing an out-of-bounds read at `all_argmax[accept_n]`
   when `accept_n=0`. Re-test after PR 2d-bis with a correctly materialized 1-head module;
   if the failure persists, audit the `commit_seq` construction in the `accept_n < g_actual`
   branch. File as a separate task if still present after PR 2d-bis.

---

## Verification

### PR 2c-bis

```bash
cmake --build dflash/build -j$(nproc)
# T1 suite — all must pass, zero model files required:
for t in test_gguf_metadata_unit test_mtp_interface_contract \
          test_mtp_chain_runner test_qwen36_mtp_basic \
          test_qwen36_mtp_step_unit; do
    ./dflash/build/$t && echo "$t PASS" || echo "$t FAIL"
done
```

All 5 binaries must exit 0 before PR is mergeable. In particular:
- `test_qwen36_mtp_step_unit` must include the 2 new cases for
  `materialize_mtp_tensors` tensor-ptr assertions and `set_initial_hidden` propagation.
- `test_mtp_chain_runner` exercises `propose_drafts_` for `NativeHeads`; no change expected.

### PR 2d-bis

Above T1 suite plus:
```bash
QWEN36_MTP_GGUF=/home/peppi/models/qwen3.6-27b-mtp/Qwen3.6-27B-UD-Q2_K_XL.gguf \
  ./dflash/build/test_dflash --mode qwen36-mtp \
    --target $QWEN36_MTP_GGUF --mtp-gguf $QWEN36_MTP_GGUF \
    --prompt-bin dflash/test/prompts/he_001.bin \
    --n-gen 64 --gamma 1 --gpu 0
```
Expected: no crash, `RESULT tok_s=<nonzero>` line on stdout, MTP tok/s > 0. Optional T2
smoke: assert the first 5 draft tokens are non-EOS, non-pad (token IDs > 3).

### PR 2e-final

```bash
bash dflash/test/test_qwen36_mtp_e2e.sh
```
Expected output: JSON written to `dflash/bench/results/<date>-qwen36-mtp.json` with:
- AR cell: `tok_s_median` in `[35.2, 38.8]` (37 tok/s ±5%).
- MTP cell: `tok_s_median` > 0 (non-degenerate).
- Both cells populated in the same JSON (not two separate files).

Per `BENCH.md`: 5 prompts × 3 runs per cell, interleaved, median reported, GPU fingerprint
captured. The MTP cell is "Dense ↔ Dense" within the same family — no cross-family
comparison (the 35B-A3B MoE cell may be absent if the GGUF is not on disk; that is
acceptable).

---

*Last updated: 2026-05-15. Do not edit bench numbers by hand; fill from JSON output only.*
