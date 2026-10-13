# DFlash 2 (Qwen3.8-27B-DFlash2) port scope — qwen35 path

Status: scoping only, nothing implemented. Worktree `kvflash-am` @ 3f7ddfbf, 2026-10-13.

**Update (branch `dflash2/qwen38-upstream-gguf`): M1, M2 and M3 are implemented.** M1: pure mapping in
`server/src/common/dflash2_upstream_schema.h` (unprefixed keys, target_layers - 1, tokenizer mask id, tensor aliases).
M2: `server/src/common/draft_attn_mask.h` per-query non-causal SWA mask, gated on `attention.causal=false`; the legacy
one-shot graph refuses it. M3 evidence: `DFLASH2_M3_3090.md` (AL 6.10 greedy, 2.9x decode; byte-identity vs AR not met,
FP batch-verify near-ties suspected). Skipped: section 3.C, converter SWA change, M1b tensor byte-compare (no HF
safetensors locally). The text below is the original scoping note.
Inputs read: our server tree, `Qwen3.8-27B-DFlash2-Q8_0.gguf` (sha256 c18e800d…, matches expected.sha256),
`dflash-draft-3.6-q8_0.gguf`, z-lab/dflash `dflash/model.py`, llama.cpp PR #27342 head `2f3923b`
(`src/models/dflash.cpp`, `common/speculative.cpp`, `conversion/qwen.py`). The references were shallow-cloned to
`/mnt/kimi-k3/repo/dflash2-ref/` and read only. Nothing was built or executed.

## 0. Headline: the algorithm is already in-tree; the upstream GGUF is not loadable

DFlash 2 already runs in the server: dynamic convs, the selector, greedy chain decode, batched lanes and DDTree scoring
(commits f686c447, 24be6f51, f66a5b77, 20670a31; README:721 documents Qwen3.8-27B + "Q8_0 DFlash2" on R9700).
That path loads GGUFs made by **our** converter (`server/scripts/convert_dflash_to_gguf.py`, arch `qwen35-dflash-draft`,
keys `*.dflash.dflash2.*`, tensors `dflash.selector.*`, `blk.N.attn_conv.base`).

The new file comes from llama.cpp's converter (arch `dflash`), and our loader rejects it at the arch check. If it got
past that check it would be **silently wrong in three places**: capture layers, causal masking, and the attention window.
The work is a schema-compat layer plus a semantics fix in the masks. It is not a new forward pass.

## 1. Forward-pass deltas vs DFlash 1 (and our current implementation)

| Item | DFlash 1 (3.6, 58 tensors) | DFlash 2 reference (z-lab model.py / PR dflash.cpp) | Ours today |
|---|---|---|---|
| Block | 16 | 8 = anchor + 7 MASK (`tokenizer.ggml.mask_token_id`=248070) | OK (`dw_.block_size`) |
| Feature fusion | fc(5*5120→5120) + hidden_norm | same; `fc.weight` + `enc.output_norm.weight` | OK, names differ |
| Target layers | default `1+k*step` = [1,16,31,46,61] | GGUF `dflash.target_layers`=[6,20,34,48,62] = **HF ids + 1** (PR `conversion/qwen.py:709` writes `i+1`; it captures layer *input*). HF ids = [5,19,33,47,61], i.e. output of layer id (z-lab `extract_context_feature` uses `hidden_states[id+1]`) | We capture the *output* of layer `il==id` (`qwen35_target_graph.cpp:2593-2597`), so the loader must store **gguf-1** |
| Two-tap dynamic conv | none | Per layer, twice: around attention (prepare on `attn_norm(h)` before Q/K/V, finish on the attention output before the residual add) and around the MLP (prepare on `ffn_norm(h)`, finish on the MLP output before the residual add). `dyn = proj @ x_in` (one GEMV, [2*K*groups]), used by both prepare and finish. `out[t] = Σ_tap (base[side][tap][c] + dyn[t][side][tap][c/16]) * x[t-tap]`, K=2, groups=320, **shift is within the 8-token block, zero-pad at slot 0**. Applies to noise/block tokens only, never to the context K/V | Matches exactly: `draft_graph.cpp:89-172` (index `(s*K+k)*groups`, base offset `(s*K+k)*nb[1]`), fused `ggml_dflash_dyn_conv`. Base `[5120,2,2]` and proj `[5120,1280]` shapes are identical |
| Candidate selector | none | logits = target lm_head(out_norm(h)); per slot p≥1: top-16 candidates, `score[j] = logit[j] + <pred_cb[prev] ⊙ (hproj @ h_p), succ_cb[cand_j]>`, prev = argmax (greedy), with the anchor as the predecessor at slot 1. Rank 256, codebooks [256, 248320] | Matches: `dflash2_head.cpp:246` `dflash2_select_chain` (log-probs, not raw logits; the per-position shift does not change the argmax). The PR uses the same greedy walk (`speculative.cpp:1252-1274`) |
| Attention | per-layer SWA pattern (3.6: 4 SWA, causal) + full non-causal | `attention.causal=False`, SWA 2048 on **all 5 layers**: **non-causal + sliding**. Visible iff `q_pos-k_pos < 2048` (and `k_pos-q_pos < 2048`) per query (`model.py:157-171`; llama STANDARD SWA with causal_attn=false) | **Mismatch.** We treat `is_swa` as causal everywhere: `draft_graph.cpp:300-302`, `draft_graph.cpp:614`, `dflash_draft_kv.cpp:300-337`. With the PR GGUF, `attention.sliding_window(_pattern)` loads as-is (`draft_gguf_loader.cpp:622-650`), so all 5 layers would become causal. Our converter instead drops SWA when `is_causal=false` (`convert_dflash_to_gguf.py:149-156`), so our current DFlash2 attends a uniform ≤4096-row window (`qwen35_backend.cpp:3541-3542`, `draft_ctx_max=4096`, `qwen35_backend.h:83`) where the reference uses a 2048 per-query window. They agree below 2048 tokens of context and differ above |
| RoPE | NEOX, θ 1e6 | `rope.dimension_sections=[64,0,0,0]` (degenerate M-RoPE, PR feeds pos [p,p,p,0]), θ 1e7 | Equivalent to our NEOX on all 128 dims (`draft_graph.cpp:47-54`). θ read from `dflash.rope.freq_base`. No change |
| Logit transforms | n/a | optional logit_scale / softcap before the selector | Absent in this GGUF. Add a guard that rejects the file if `dflash.logit_scale` or `final_logit_softcapping` is present |

## 2. Verification

It is still a **single linear chain**: the selector emits one path of ≤7 tokens and the target verifies it in one batch,
as in DFlash 1. Greedy output is lossless, byte-identical to AR (README:721 already reports identical completions across
widths 8/10/12/16). The selector only changes which tokens get proposed. DDTree and `--draft-block-size` overrides keep working.

Gap: our selector path is gated `!sampled_verify` (`qwen35_backend.cpp:3655`), so sampled requests fall back to plain
argmax DFlash projection. The PR's published AL (Q8_0 5.08 PR / 5.13 card, GSM8K-8, temp 1.0) uses the greedy selector
draft plus sampled target verification. For AL parity under sampling, allow the selector chain when `sampled_verify`
is on. That stays lossless as long as our sampled verify treats the draft tokens as deterministic proposals. This must be
confirmed before enabling it (see risks).

## 3. Change list (file:line, current tree)

**A. Loader schema compat (`server/src/draft/draft_gguf_loader.cpp`)** — extract it as a pure function
`resolve_draft_gguf_schema(gguf kv/tensor-name lookup) -> DraftSchema` per pure-functions rule, unit-testable without a file.
- :138-142 accept arch `dflash`. Also check `general.tags` contains `dflash2` or that selector tensors are present, and
  reject DSV4 DSpark `dflash` files (`hyper_connection_count` present) with a clear error.
- :149-158 the `read_u32` prefix: for arch `dflash` the keys are `dflash.block_size`, not `dflash.dflash.block_size`. Add a
  second lookup for un-nested keys: :169 `block_size`, :512/:526 `conv_kernel_size`/`conv_group_size`,
  :571/:580 `selector_rank`/`selector_top_k`.
- :186-192 read `dflash.target_layers` (INT32 array) as well and **subtract 1** for the `dflash` arch. Validate
  `0 <= id-1 < target.n_layer` and keep `n_target_layers == fc.ne0/n_embd` (:725-747).
- :217-236 also read `tokenizer.ggml.mask_token_id`.
- :309-310 alias `fc.weight` and `enc.output_norm.weight`. :355-359 alias `blk.N.attn_conv_base`,
  `attn_conv_proj.weight`, `ffn_conv_base`, `ffn_conv_proj.weight`. :568-570 alias `selector_hidden.weight` (hproj),
  `selector_predecessor.weight` (pred_cb), `selector_successor.weight` (succ_cb). The shapes already pass
  `check_shape_*` and `validate_dflash2_selector_layout` (Q8_0 codebooks: the GET_ROWS path must accept Q8_0; check that
  `dflash2_head.cpp` gather and `dflash2_batch.cpp` do not assume F16/F32).
- :622-650 read `<A>.attention.causal` (default true, which keeps today's semantics for 3.6). Store it per layer.
- `server/src/qwen35/qwen35_backend.cpp:238-258` `apply_drafter_capture_layer_ids` needs no change once the loader
  normalises the ids. It already prints the change; assert that the log shows `5 19 33 47 61`.

**B. Non-causal SWA (semantics). Gate on the GGUF causal flag so 3.6 DFlash1 behaviour stays bit-identical.**
- `server/src/internal.h:322-337` add `bool is_causal` to `DraftLayer` (decoupled from `is_swa`). `:401-402`: unchanged.
- `server/src/common/dflash_draft_kv.cpp:300-337` mask build. For a non-causal SWA layer, the noise columns are fully
  visible (as in `mask_full`) and the ctx lower bound is **per query row** `lo_q = committed + q - W + 1` (today one row
  template is memcpy'd to every q). Default path on qwen35 (`LUCE_DRAFT_KV` on, `qwen35_backend.cpp:3531-3567`).
- `server/src/draft/draft_graph.cpp:203-207,237-241,300-302` legacy stateless graph: uniform view
  `ctx_len - swa_window` plus a causal mask. Either build a per-query non-causal SWA mask, or refuse DFlash2 on this path
  (it is only used when draft-kv is off or init fails). Refusing is cheaper.
- `server/src/draft/draft_graph.cpp:527-614` multilane / `dflash2_batch.cpp` (concurrent engine
  `qwen35_seq_engine.cpp:454`): same mask rule via `in.mask_swa`.
- `server/scripts/convert_dflash_to_gguf.py:149-156` stop dropping SWA when `is_causal=false`. Emit the pattern plus
  `attention.causal=false` so our own GGUFs match the reference beyond 2048 tokens.
  **This is a behaviour change for existing DFlash2 users above 2K context. Ship it behind the GGUF flag and A/B the AL.**

**C. Optional (not in the first milestone)**
- Selector under sampled verify: `qwen35_backend.cpp:3655`, relaxing `!sampled_verify`.
- Remote draft IPC (`server/src/common/dflash_draft_ipc.h:51-54`, `propose` already returns hidden): the selector needs
  `dw_.selector` weights on the target side, so load only the selector tensors locally. Today it is gated `!use_remote_draft`.
- Other backends that call `load_draft_gguf` (qwen35moe, layer-split daemon) inherit A for free. B touches only the
  shared mask builders.

**Reuse from deepseek4:** nothing material. `deepseek4_candidates.cpp` is the V4.1 attention-indexer block
pre-selection (`deepseek4_candidate_blocks`), not a token-path selector. `deepseek4_dspark.*` is the DSV4 DSpark drafter
(MLA/MoE stages, Markov/confidence heads, own rollback). The DFlash2-relevant reuse is already in
`server/src/common/dflash2_head.cpp` / `dflash2_batch.cpp` / `dflash2_selector_validation.h`.

## 4. KVFlash session restore

The drafter keeps no new state:
- Conv: within-block shift, zero-padded at slot 0 (PR `dflash.cpp:435-447`, z-lab `F.pad(blocks[:, :-offset])`).
  No conv state crosses steps.
- Selector: stateless, works per block.
- Context: the only persistent drafter input is the target feature ring (`cache_.target_feat`, 5×5120 per row). It is
  already captured by the state ladder when a drafter is configured: `state_ladder_capture`,
  `qwen35_backend.cpp:1024-1027` (`skip_target_feat=!cfg_.draft_path`). It is re-synced into `feature_mirror_` on live
  restore (`:1115-1128`) and on restore_and_generate (`:1994-2015`).
- The draft K/V ring (`draft_kv_`) is reset at the start of every `do_spec_decode` (`:3178`) and bulk-refilled from the
  mirror, so it never needs checkpointing.
- Caveats:
  1. Fixing capture ids changes the *contents* of `target_feat`. Any snapshot taken with the old ids must not be reused
     after the change. The ladder is in-memory, so confirm that `state_ladder_clear` runs on draft load/unpark (:893-986).
  2. The window needs ≥2048 ring rows. `target_feat_cap` (`:1376`) and `kv_cap=min(ring_cap, 4096)` already cover it.
  3. Thin snapshots carry no `target_feat`; the sync-failure branch falls back to a cold prefill (`:1126-1128`), which is
     correct.

## 5. Estimates

| Piece | Effort |
|---|---|
| A. schema resolver (pure fn) + aliases + target_layers −1 + unit tests on synthetic kv maps | 0.75 d |
| B. per-layer `is_causal`, per-query non-causal SWA masks in draft_kv + multilane, legacy-path refusal, CPU mask unit test | 1.25 d |
| M1 parity harness on 3090 (below) | 1 d |
| Converter SWA/causal fix + long-context AL A/B (>2K) on our GGUF | 0.5 d |
| C1 selector under sampled verify (incl. lossless proof test) | 1 d |
| C2 remote-draft selector | 1 d |
| **Core (A+B+M1)** | **~3 d**; with converter + C1 ~4.5 d; everything ~5.5 d |

## 6. First milestone (testable on the 3090)

PR #27342 publishes only acceptance/TPS (Q8_0: AL 5.08, 18.43 tok/s, 1.77×, block 8, n_max 7). It ships no per-step
tensors, and its code may not be executed here, so the single-step oracle has to be ours:

M1a — load: `test_dflash` (or a new `test_dflash2_gguf_load`) loads the upstream Q8_0 with the Qwen3.8-27B
UD-Q4_K_M target (`/home/peppi/models/qwen3.8-27b/`, 16.4 GB + 2.06 GB fits in 24 GB at small ctx). Assert: arch
`dflash`, block 8, conv k2/g16, selector r256/k16, capture ids `[5,19,33,47,61]`, mask 248070, all 5 layers
`is_swa && !is_causal`, W=2048.

M1b — one draft step, bit-exact vs our trusted path: convert the HF `incoai/Qwen3.8-27B-DFlash2` safetensors with our
own `convert_dflash_to_gguf.py` + `quantize_draft_q8.py` (both run Q8_0 through gguf-py, so the bytes should be
identical). (i) Compare all 81 tensors' bytes against the upstream GGUF. (ii) On a fixed prompt with ctx < 2048 (where
window semantics agree), run one draft step with each GGUF. Hidden [8×5120] must be bit-equal, the top-16 ids per slot
equal, and the selected 7-token chain equal.

M1c — e2e gate: greedy, the PR prompt set (first 8 GSM8K test items, max 2048 new, n_max 7). Output must be
byte-identical to AR. Report AL plus prefill/decode/wall for both arms (bench-parity rule: same target md5, same
max-ctx, prefix-cache symmetric). Sanity band: AL near 5. The PR's 5.08 was measured with sampling, so it is not a
strict greedy target; a strict comparison needs C1 at temp 1.0 and a median over repeats.

## 7. Top risks

1. **Off-by-one capture layers.** The upstream GGUF stores HF id+1 (layer-input convention); we capture layer outputs.
   Getting this wrong raises no error and just collapses acceptance. Mitigation: the M1a assert, plus the M1b bit-match
   against our converter, which uses HF ids directly.
2. **SWA ≡ causal conflation.** Loading the PR GGUF without B makes all 5 layers causal. That is silent and lowers AL at
   every context length. B must land with A, never A alone.
3. **Window semantics at >2K context.** The reference uses a 2048 per-query window; our current DFlash2 uses ≤4096 uniform.
   The fix changes behaviour for existing users of our converter GGUFs. Gate it on `attention.causal` and A/B the AL
   before flipping the converter (preserve-working-behaviour).
4. **No runnable reference.** Parity rests on byte-identical Q8_0 from our converter. If gguf-py versions differ, the bytes
   may not match; then fall back to comparing dequantised tensors within a Q8_0 tolerance and matching top-k/chain.
5. **Sampled AL parity.** The selector is disabled under sampled verify, so temp>0 AL will trail the PR's numbers until C1 lands.
6. **Q8_0 codebooks.** Selector GET_ROWS on Q8_0 [256, 248320] tensors in `dflash2_head.cpp` and `dflash2_batch.cpp` has
   not been exercised if earlier runs used F16 codebooks. Check that backend support exists and that it is not slow.
