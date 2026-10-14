# AM compaction, setup F, on latest main

Date 2026-10-14. Base: `origin/main` = fbcefbc4. On top of our merge-base
5f402c61 it adds #839 (disk sessions), #842 (reply cap) and #841 (structured
output). Branch `worktree-kvflash-am` @ 5429173d. Line numbers below are from
`origin/main:<path>` unless marked "branch".

Setup F (R1b, `evidence/am/R1b.md`, `PAPER_MAP.md` row F) is:
- context-prefill queries;
- HighestAttnKeys with the rms score;
- beta by NNLS, 2 projected-gradient steps, w in [e^-3, e^3];
- values by least squares;
- per-head budgets from `evidence/am/R1b/r1c/optimized_agnostic.json`, with
  `max_ratio_per_head` 0.75;
- the `_on-policy` flag set.

Only the 16 full-attention layers are compacted. DeltaNet state is untouched.

The engine must reproduce these reference numbers. They are counted from the
committed jsonl rows (Qwen3.8-27B BF16 on Modal):

| split | target | orig | F | D = orig - F, points [95% CI] |
|---|---|---|---|---|
| selection (357 q, 20 articles) | 0.10 | 320 | 309 | 3.1 [0.6, 5.7] |
| selection | 0.05 | 320 | 300 | 5.6 [2.5, 8.5] |
| held-out (537 q, 30 articles) | 0.10 | 460 | 439 | 3.9 [2.0, 5.9] |
| held-out | 0.05 | 460 | 431 | 5.4 [3.1, 7.9] |

## 0. Findings that shape the plan

1. **On-policy is a no-op for F, so F needs no second pass.** I read
   `compaction/compaction_methods/per_layer_head_on_policy.py` in the repo
   without executing it.
   - Per-layer re-extraction is built only for query ranges that `self_study`
     produced (L328-355). L357: "context_prefill uses original queries".
   - With no extractor, layer l uses `original_queries[layer_idx]`
     (L385-387).
   - So F's queries are exactly the queries of the one article prefill: every
     layer, uncompacted cache. The engine captures them during the prefill it
     already runs.
   - Contrast: `repeat` is implemented as a `self_study` spec, so O1, O2 and C
     would need a re-prefill of the repeat suffix for each compacted layer l,
     through layers <= l over the partly compacted cache. That is 16 partial
     passes, out of scope here.
2. **The fit is non-causal and span-local.**
   - HighestAttnKeys scores `queries @ K.T` over the span's keys only, with no
     causal mask (`highest_attention_keys.py` L150-205).
   - The beta target is the per-query row sum of exp-scores over the span
     (L211-224). The values fit the attention output over the span.
   - All queries are used. An article of ~6.6K tokens x 6 query heads per KV
     head gives ~40K vectors, under the 100,000 cap of `context-prefill.py`.
3. **Per-head budgets cannot share one chunk table.**
   - Proportions are per (layer, KV head). Per-head keys = `int(p * S * H)`
     with H = 64 (`per_layer_head_on_policy.py` L243-260).
   - Computed from the budget file: padding every head to the global max head
     costs **5.54x** the ideal slots. Padding each layer to its own max head
     costs **1.66x** (the "1.6x").
   - The E3 mapping "segment S = 64/r tokens -> one 64-slot block" holds
     exactly only for uniform budgets. That is U-AM, which loses 9.8 points at
     0.10 against F's 3.1.
   - With F's budgets, one segment needs k_L = ceil(max_h t_{L,h} / 64) blocks
     in layer L. At r = 0.10, k_L runs from 1 to 6: the max p = 0.0866 at
     L43H0 gives 354 of 640 slots.
4. **beta must not be a per-head FA mask.**
   - ggml broadcasts mask planes by `iq2 % mask->ne[2]`
     (`ggml-cpu/ops.cpp:8405`, `:8686`), while GQA maps a query head to its KV
     head by division.
   - A per-head mask is n_q x n_kv x heads F16: 786 MB for a 512-token ubatch
     over a 32K pool at 24 heads.
   - beta does not depend on the query, so it goes in as a separate
     per-KV-head key bias `[n_kv, n_head_kv]`: 256 KB per layer at 32K. It is
     wired like `sinks` (`ggml.c:5926`, `fattn-common.cuh:1022`).
   - The `mask->ne[2] != 1` gate (`ggml-cuda/fattn.cu:4557`, the PLAN's
     ":4433") stays.
5. **Keys and queries are already in one basis.**
   - `build_full_attn_block` exposes post-RoPE, post-rotation Q as `q_fa_out`
     (`qwen35/qwen35_target_graph.cpp:1141`, set at :1380/:1394/:1405/:1459).
   - Cache K rows are post-RoPE and rotated the same way.
   - C1 is a copy of the original K rows, with RoPE baked in and no
     requantization.
   - V rotation (TQ3) is orthogonal, so the least-squares fit in the rotated
     basis equals the rotated fit.
   - C2 is F32 and is quantized to the pool type (q8_0) when written. R2
     measures that error.

## 1. What lands first (from our branch, PR-sized, in order)

Every PR is rebased onto origin/main. PR-B and PR-D resolve the three merge
conflicts (section 6).

| PR | Commits (branch) | Content | Why AM needs it |
|---|---|---|---|
| A. pager library | b64d1630, 5cd2a923, 595b40b9, febb1885 | In `common/kvflash_pager.h`: serialize/deserialize, pins, score ledger, `truncate(P)`, refill after truncate, host block memset. Tests: `test_kvflash_pager_serde.cpp`, `test_kvflash_pager_truncate.cpp`. Header and tests only. Supersedes #466 | Compacted segments are pager chunks, so truncate and serde must know about them |
| B. pooled snapshots | b03f6ea7, 6b9f8b59, plus the #839 adaptation | Pager blob in `PrefixSnapshot`, `SnapshotRef::pooled`, pooled snapshot at a chunk boundary, suffix-only restore (opt-in). Adds: `take_prefix_snapshot` (`qwen35_backend.cpp:1167`) frees a snapshot with `is_pooled` set and returns `{}`, so #839 eviction never writes a file without KV | The compacted state is saved through this path |
| C. 512 grid | 03ea42af | `qwen35/prefill_helpers.h`: absolute U = 512 prefill grid, remainders without NaN | Segment boundaries sit on 64-token chunks inside an aligned grid |
| D. state ladder + live restore | 7154dfb1, 6177bdaf, 1e853bfd, 08594439, 677f2b4d, 085f6f4b | S7 DeltaNet ladder; S8a live restore that publishes the cut (`http_server.cpp` log, `model_backend.h` retry); spec-decode history; pool crossing | Every question or turn after a compaction restores from the live session |
| E. S9/S9b | 786d0185, ae27df75, 8975fb7d, 0697848b | Generation-end checkpoint, turn rung, turn cuts on by default with live restore | Multi-turn sessions keep the compacted prefix across turns |

Two independent PRs are not on the AM path:
- d2da55ed think-marker replay (fork branch `server/qwen-think-marker-replay`);
- 1caadab8 DFlash 2 upstream GGUF loader (`dflash2/qwen38-upstream-gguf`).

## 2. Engine design for F

- **Pool regions.**
  - Each full-attention layer's KV tensor becomes
    `[head_dim, A + B_L, n_head_kv]`.
  - Region A is today's KVFlash pool: shared chunk table, shared slot mask,
    unchanged.
  - Region B_L holds layer L's compacted blocks and is allocated per layer.
  - B_L capacity at init is `ceil(r * H * max_h p_{L,h} * (max_ctx - A) / 64)
    * 64`. Per-layer memory follows that layer's budget, not the global max:
    1.66x of ideal instead of 5.54x.
- **Mask contract.**
  - The shared F16 mask covers region A only (`mask->ne[0] = A`). The kernel
    treats columns `>= mask->ne[0]` as mask 0.
  - The per-layer key bias `kv_bias[L]` is `[A + B_L, n_head_kv]` F32. It
    holds 0 on region A, beta on live compacted slots, and -inf on a head's
    padding slots and on free B slots.
  - The mask stays shared across layers and contiguous (`ggml.c:5735` asserts
    contiguity), and no mask is n_head-sized.
- **Segment map.**
  - The pager's chunk table gains compacted entries:
    `{lo, hi, per-layer block list, per-(layer, head) live count}`.
  - Logical positions inside a compacted segment never get slots;
    `slot_for(pos)` is invalid for `lo <= pos < hi`.
  - Every compacted key precedes every later query, so causality holds without
    extra work.
- **When to compact.**
  - Each segment is fitted once from its originals, eagerly, when its last
    token is prefilled.
  - Query memory per segment at r = 0.10: 640 x 24 heads x 256 x 2 B = 7.9 MB
    per layer, 126 MB for 16 layers.
  - The result is published when the segment leaves region A.
  - The recent tail, the pins (system/tools head) and the current turn stay
    uncompacted.
- **Restore rule.**
  - `truncate(C)` with `lo < C < hi` of a compacted segment drops that segment
    and replays from `lo`. That needs a DeltaNet rung `<= lo`.
  - One extra rung, the **compaction frontier rung**, sits at the end of the
    last compacted segment. It costs 150 MiB and is replaced as the frontier
    moves.
  - Turn continuation always cuts above the frontier rung.

## 3. Milestones

Effort is engineering days for one person, tests included.

### M1. Budget loader (0.5 d)
- Files:
  - new `server/src/common/am_budget.h` (pure);
  - flags next to `--kvflash` in `server/src/server/server_main.cpp:286`:
    `--kvflash-am-budgets <json>` and `--kvflash-am-ratio <r>`.
- Logic:
  - parse `"L<l>H<h>": p`;
  - map l to the full-attention index
    `fa = (l + 1) / full_attention_interval - 1`, asserting `l % 4 == 3` for
    Qwen3.8 (keys L3..L63);
  - port the `apply_max_ratio_cap` water-filling (`compaction_methods/base.py`
    L208-302, per `PAPER_MAP.md`);
  - budgets are `int(p * S_span * 64)`, and a missing head gets `S_span`.
- Test first:
  - on the real file at r = 0.10 with a 6626-token span, the per-head counts
    equal the repo formula: S = 662, total = 42368 minus truncation, L43H0 =
    int(0.086625 * 42368) = 3670;
  - the cap does not bind (max 0.0866 < 0.117);
  - a synthetic file with one p = 0.5 binds, and the redistribution keeps the
    sum.
- Gate: none (CPU only).

### M2. Key bias in flash attention, CUDA + HIP (6 d)
- Files:
  - `ggml/include/ggml.h:2501`: add `ggml_flash_attn_ext_add_kv_bias`;
  - `ggml/src/ggml.c:5718/5926`: a src slot after sinks;
  - CPU reference `ggml-cpu/ops.cpp:8405/8686`: add `kv_bias[ik2][ic]` after
    scale and softcap;
  - CUDA and HIP share the `ggml-cuda` kernel sources:
    `fattn-common.cuh:27/1022/1246` (launch param), `fattn-vec.cuh:26`,
    `fattn-tile.cuh:763`, `fattn-mma-f16.cuh:1016`, `fattn-wmma-f16.cu:31/97`,
    `fattn-chunked.cu` (TQ3 path);
  - the mask-shorter-than-KV contract goes in the same kernels, including the
    `KV_max` trailing-tile skip, which must not cut off region B;
  - dispatch in `fattn.cu` around :4557.
- Test first: `test-backend-ops` FLASH_ATTN_EXT cases with kv_bias:
  - random beta in [-3, 3] and -inf padding;
  - n_head 24 / n_head_kv 4, D 256;
  - KV F16 / Q8_0 / TQ3_0;
  - n_q 1 / 16 / 512;
  - mask ne0 < n_kv.

  Pass: max abs error against the CPU reference within the existing FA
  tolerances, and a zero bias bit-identical to no bias.
- GPU gate:
  - the 3090 (sm_86, mma/vec) and the R9700 (gfx1201, tile/wmma/vec) both pass
    the same cases;
  - 3090, Qwen3.8-27B UD-Q4_K_M, `--kvflash 8192`, temp 0, 30 decode runs at
    20K context: decode median with a zero bias attached is within noise of no
    bias, under the `bench-engine-parity` rules.

### M3. Query capture during prefill (2 d)
- Files:
  - `qwen35/qwen35_target_graph.cpp:2433`, beside `want_q_cap`: route every
    full-attention layer's `q_fa_out` into a per-ubatch capture tensor
    `[head_dim, n_tokens, n_head]` F16;
  - `internal.h:888/978`: `QwenGraphInputs::am_q_capture`;
  - `internal.h:452/570`: a TargetCache staging buffer;
  - the backend prefill loop in `qwen35_backend.cpp` around :2222: copy device
    to host per ubatch, only for tokens in a span marked for compaction;
  - decode steps append the `q_cap` row (`:288`, `:2522`) to the open
    segment's reservoir.
- Test first: a unit test with a 2-layer synthetic graph on CPU:
  - the captured Q equals Q recomputed from the layer input with norm, RoPE and
    rotation, for a ubatch that does not start at zero;
  - with capture off, the graph is identical (same node count).
- GPU gate, 3090, one 6.6K-token QuALITY article:
  - layer L3's captured Q matches a Modal dump of the same tokens' post-RoPE Q
    to the Q4-vs-BF16 noise floor, measured on uncompacted attention outputs.
    Record the cosine; the pass threshold is the R2 rule;
  - prefill wall grows by at most 3% with capture on.

### M4. HighestAttnKeys scoring on GPU (2 d)
- Files: new `server/src/common/am_fit.{h,cpp}`, which builds ggml graphs on
  the target backend, so CUDA and HIP come for free:
  - chunked over queries, 8K rows at a time;
  - `scores = Q K^T / sqrt(d)` in fp32, then row max, exp, row sum and
    normalized weights;
  - accumulate `sum_q w^2` per key, giving the rms score;
  - `ggml_argsort_top_k` per head, with `t_h` from M1.
- Test first: a CPU-backend test against a reference written in C++ (no
  NumPy), on synthetic K/Q with T = 700, n = 4200, t = 64:
  - selected indices are identical;
  - exact ties are resolved like `torch.topk`. Document the rule; the R3 bar is
    99%, not 100%, because of ties.
- GPU gate: R3 fixtures, a Modal dump of K, Q and the repo's selected indices
  for 2 articles, all 64 heads at r = 0.10. At least 99% of selected keys are
  identical per head.

### M5. beta NNLS and value least squares (3 d)
- Files: `common/am_fit.{h,cpp}`.
- The GPU only accumulates Gram matrices, chunked over queries:
  - `M^T M` and `M^T y`, where M = exp-scores at the selected columns and
    y = row sums;
  - `A^T A` and `A^T Y`, where A = softmax over the selected keys with beta and
    Y = the span's attention output.
- The host solves the t x t systems in fp64. t is at most ~3.7K in the global
  gate fit and at most 354 per 640-token segment:
  - least-squares init;
  - 2 projected-gradient steps with step 1/L (L by power iteration on
    `M^T M`), clamped to [e^-3, e^3];
  - `beta = log(w)` (`algorithms/base.py` `_nnls_pg` L472);
  - then `C2 = (A^T A)^-1 A^T Y`, with the repo's fallback ridge 1e-6 on
    failure.
- Test first:
  - noise-free synthetic problems with known w and known C2 are recovered to
    1e-6;
  - an ill-conditioned M engages the fallback path and stays finite.
- GPU gate:
  - R3 fixtures: beta and C2 within 1e-3 relative of the repo's values (the
    `R3` row of PLAN.md);
  - if the Gram route misses 1e-3, switch the host solve to Householder QR on
    the n x t matrix (56 MB per head) and re-measure;
  - record the fit wall on the 3090 for one 6.6K-token article (64 heads).
    Modal F takes 6-7 s per article.

### M6. Region B, segment map, per-layer ragged extents (5 d)
- Files:
  - KV allocation in `qwen35/qwen35_target_graph.cpp:197`: pool tensors of
    `A + B_L` per layer;
  - the pager `common/kvflash_pager.h` (main has 770 lines; this goes on top of
    PR-A): `KvFlashConfig` :72, `attach` :101, `slot_for` :240,
    `fill_slot_pos` :363, `fill_slot_mask` :384 (region A only), and
    `copy_chunk` :466 (per-(tensor, head) segments, extended to B blocks),
    plus PR-A's truncate/serialize;
  - a new `kv_bias` upload beside `kvflash_upload_mask`
    (`qwen35_backend.cpp:2467`), epoch-cached the same way;
  - the graph: the `build_full_attn_block` view (:1437-1461) spans `A + B_L`
    and passes kv_bias. The verify path keeps the #652 rule (view = whole
    tensor).
- Test first (pager, CPU):
  - insert a compacted segment with per-(layer, head) counts, and
    `fill_kv_bias` puts beta, 0 and -inf in the right columns;
  - truncate inside the segment drops it and reports `lo`;
  - serialize then deserialize keeps the segments and the B blocks;
  - with B empty, the region A mask is byte-identical to today's.
- GPU gate, 3090, a 20K-token session without compaction, with B_L allocated
  but empty:
  - outputs are identical to main (greedy tokens, 5 prompts);
  - decode median is within noise.

### M7. F gate, global span (3 d)
- Files:
  - a request extension `am_compact: {span: [a0, a1], ratio}`, parsed in
    `server/http_server.cpp` into `GenerateRequest`;
  - the backend, after prefilling the span: fit (M4/M5) with the span's
    captured queries, write C1/C2 into B blocks, publish the segment and take
    the frontier rung;
  - the harness `optimizations/kvflash/am_gate/run_quality.py`. Token ids per
    article and question come from the Modal driver (`format_context` /
    `format_question`, non-thinking) and are sent as token arrays. Greedy
    decoding, 256 new tokens, the same parser as `r1b_core.py`.
- Test first: a model-free server test:
  - `am_compact` with an out-of-range or non-chunk-aligned span returns 400;
  - a request without it runs the old path (same graph).
- GPU gate: 3090, Qwen3.8-27B UD-Q4_K_M, q8_0 KV, `--kvflash` pool >= 8192,
  selection split (articles 0-19, 357 q):
  1. Precondition: engine orig within 320 +- 11. Otherwise the Q4 weights are
     the confound and the gate is void: fix the protocol before reading F.
  2. r = 0.10: engine D = orig_engine - F_engine <= 5.7 points, and its 95%
     cluster-bootstrap CI (10,000 resamples, seed 0) overlaps [0.6, 5.7].
     Modal F = 309.
  3. r = 0.05: D <= 8.5, and the CI overlaps [2.5, 8.5]. Modal F = 300.
  4. No non-finite logits: `nan_layers` empty in every row.
  5. Then, once, held-out (537 q): D(0.10) overlaps [2.0, 5.9] (Modal F 439),
     and D(0.05) overlaps [3.1, 7.9] (Modal F 431).

  Report per-question agreement of F_engine vs F_Modal beside the agreement of
  orig_engine vs orig_Modal.

### M8. Segment-local F and automatic compaction (4 d)
- Files:
  - the backend prefill/decode loop: close a segment every S = 64/r tokens
    (640 at 0.10);
  - the pager policy: segments leave region A oldest first; pins and the tail
    are exempt;
  - the frontier rung in the S7 ladder.
- Prerequisite on Modal: R1(e) (segment-local vs global, same total budget)
  has not run yet. Run it first with the F configuration; it sets the pass bar
  here.
- Test first: a pager policy unit test covering which segments compact for
  synthetic histories, that pins never compact, and the frontier rung
  position.
- GPU gate:
  - QuALITY selection split at r = 0.10 with segment-local fitting: D within
    the R1(e) CI from Modal;
  - 120-turn trace: 120/120 turns, 0 degenerate, decode median flat across
    turns, and prefill per turn < U + new tokens (the S8 bar).

### M9. Disk persistence of compacted state (2 d)
- Files:
  - the snapshot ctx gains tensors `snap_kvflash_blob` (I8, the pager blob,
    replacing the `std::vector`), `snap_am_k_<L>`, `snap_am_v_<L>` and
    `snap_am_bias_<L>`;
  - `take_prefix_snapshot` (`qwen35_backend.cpp:1167`) passes pooled snapshots
    through again;
  - `snapshot_adopt` (:1275) rebinds the new names;
  - the disk identity salt (`disk_prefix_cache.cpp:701`) includes the budget
    file hash, r, the method and the fitter version.
- Test first (model-free):
  - detach a pooled and compacted snapshot, write, read and adopt it; the
    restored pager, segment map and kv_bias equal the originals;
  - a file written under another budget file is a miss.
- GPU gate, 3090: a compacted session gets SIGTERM, the server restarts, and
  the next turn restores from disk with greedy output identical to the
  uninterrupted run.

Total: about 28 engineering days after the prerequisite stack, plus the R1(e)
Modal run before M8.

Order: M1 -> M2 -> M3 -> M4 -> M5 -> M6 -> M7 (gate) -> R1(e) -> M8 -> M9.
M1 and M3 can run alongside M2.

## 4. Multiple rounds of compaction

**What the paper does** (App F.3 / Table 9, online compaction). In the repo
(`evaluation/run_reasoning_evaluation.py` L1-20, L490-568):
- Each time the physical cache reaches `max_seq_len`, the **whole current cache
  except the last 20 protected tokens** is compacted to `target_size` of its
  length. That includes entries compacted in earlier rounds.
- The non-on-policy wrapper used there passes the previous beta in as
  `attention_bias` (`per_layer_head.py` L101, L525). So compacted blocks are
  re-compacted with their bias: a summary of a summary.
- Our R0 run reproduced Table 9 on Qwen3-4B AIME at 0.5 per round: c <= 6 at a
  2048 cache scores 12.0/30, the same as the 8192 baseline
  (`R0_aime_table9.md`).
- The on-policy wrapper that F uses (`per_layer_head_on_policy.py` L470) calls
  the algorithm without `attention_bias`. Re-compacting under it would silently
  drop the earlier beta. That is a repo limitation the engine must not copy.

**What the engine does:**
1. It compacts each segment **once**, from its originals: S tokens go to the
   segment's k_L blocks per layer at ratio r. Compacted blocks are not
   re-compacted while the region B budget holds.
2. When region B goes over budget, it merges two adjacent segments by
   **refitting from their host originals**: 2S tokens into the same block
   count, effective ratio r/2 (PLAN E4).
   - This is a fresh fit, so errors do not compound.
   - The refit needs the segments' reference queries again. The plan keeps a
     per-segment query reservoir at a fixed subsample; the repo itself
     subsamples at its vector cap.
   - An R1(e)-style Modal check sets the subsample rate before E4.
3. Only when the originals are gone (host and disk full) does it fall back to
   the paper's behaviour: bias-aware re-compaction of compacted blocks, with
   beta as `attention_bias` in the scoring and in both fits.
   - Evidence for that mode exists only for the 4B at 0.5 per round, at most 6
     rounds. The 27B at 0.1 per round is untested.
   - Count the rounds per segment and log them.

## 5. Risks

- **#839 disk persistence and compacted blocks.**
  - #839 writes only the ggml tensors in the snapshot ctx (`save_async`,
    `disk_prefix_cache.cpp:1131`). `layout_of` (:683) fingerprints tensor
    names, types and shapes, with ne[1] normalized.
  - Today our pager blob is a `std::vector` outside the ctx. A pooled or
    compacted snapshot evicted through `snapshot_detach`
    (`http_server.cpp:4557`, `scheduler.cpp:1219-1253`) would be written
    without its KV.
  - Mitigation in PR-B: `take_prefix_snapshot` refuses pooled snapshots, and
    `learn_layout` (:736) never learns the layout from a pooled ctx.
  - M9 then turns the blob, region B and kv_bias into ctx tensors.
  - The disk key is the raw token prefix. It stays valid only because
    compaction is a deterministic function of (tokens, model, budgets, r,
    fitter version), so all of these go into the identity salt (:701).
  - #839 gives PFlash-compressed prompts no disk keys because their token
    stream differs. AM keeps the raw stream, so AM states keep their keys.
  - A deeper disk hit is adopted into staging slot 63 and then loses to a
    deeper live-session cut in `kvflash_pick_resume` (branch
    `qwen35_backend.cpp` ~1925). That wastes an SSD read but gives a correct
    result.
- **Kernel bias on both backends.**
  - CUDA (3090) dispatches mma/vec. HIP (R9700 gfx1201) dispatches
    tile/wmma/vec and the chunked TQ3 path.
  - Every variant needs the bias add and the short-mask rule. That includes
    the `KV_max` trailing-tile skip computed from the mask, which would
    otherwise cut off region B.
  - #792 found tile-kernel accuracy problems on RDNA4 at D = 256 with
    prefill-shaped masks. The kv_bias tests must cover that shape on gfx1201
    against the CPU reference.
  - MoE pipelined decode (`qwen35moe_pipelined_decode.cpp`) reads the pool
    without a mask. Refuse AM there until it takes kv_bias.
- **Ragged pager vs DFlash verify views (#652).**
  - #652 made slot-mapped verify attention span the whole pool, because the
    mask lives in slot space (`qwen35_target_graph.cpp:1437-1450`).
  - With region B, the view is the layer's whole `A + B_L` tensor, and by
    contract the mask is A-sized.
  - Different layers then have different K lengths under one shared mask. That
    is legal only with the new short-mask rule.
  - The drafter's own KV window is uncompacted and unaffected.
  - Verify must use the same per-layer kv_bias tensor as decode, or the verify
    argmax differs from decode.
- **Per-head ragged beyond 1.66x.**
  - The `[head_dim, slots, n_head_kv]` layout forces within-layer padding to
    the largest head.
  - Reaching 1.0x needs per-KV-head pool tensors and 4 FA calls per layer:
    +48 graph nodes per token, about 90 us at ~1.8 us per node on our
    hardware.
  - Measured cost of the 1.66x layout at 131K, r = 0.10, q8_0: ~720 MB against
    ~433 MB ideal; uncompacted is ~4.6 GB.
  - Deferred until a context target needs those ~290 MB.
- **Quantization.** The Modal F rows used BF16 weights and BF16 KV. The engine
  uses Q4_K_M weights, q8_0 KV and C2 requantized to q8_0. The M7 precondition
  (orig within +-11) and R2 (no amplification) separate engine error from
  quantization error.
- **Segment-local vs global.** R1(e) has not run. The M7 gate fits the whole
  article as one span to match Modal. M8's segment-local fit is a separate
  claim with its own Modal reference.
- **Truncate inside a compacted segment.** An edit below the frontier drops
  every compacted segment above the edit and re-prefills from the last rung
  `<= lo`. The result is correct but slow. With S9b turn rungs above the
  frontier, normal turns never hit this.

## 6. Merge of origin/main into the branch (strategy, not applied)

- `server/src/server/http_server.h` (main :299-318 vs branch :292-298).
  - #841 added the structured-output declarations: `request_output_format`,
    `output_format_schema`, `tool_choice_requires_call` and
    `output_constraint_grammar`. We added `think_marker_text`. Keep both.
  - After the merge, check with `test_server_unit` that #841's constraint,
    which waits for `</think>`, still sees the think tag now that `</think>`
    is emitted without the trailing newline.
- `server/src/server/disk_prefix_cache.cpp`, two hunks.
  1. `learn_layout` (:736): take main's body (`learn_layout_from(ref.ctx)`)
     and put `if (ref.pooled) return;` in front of it. Drop our copy of the
     old body.
  2. `save_async` (:1131) has no slot parameter on main, so our
     `snapshot_ref(slot)` guard cannot stay there.
     - Move the guard to `save(slot, ...)` (:1219).
     - Add the backend-side refusal in `take_prefix_snapshot`
       (`qwen35_backend.cpp:1167`), because eviction calls `snapshot_detach`
       and bypasses `save`.
     - Add a test that a pooled detach returns empty and writes no file.
- `server/src/qwen35/qwen35_backend.cpp` (restore, main :1895).
  - Main now fails the request (`InvalidSnapshotSlot`, `cur_pos = 0`) when
    `restore_target_cache` fails, because disk-adopted snapshots can be
    rejected.
  - We replaced that block with the restore plan: full re-prefill, consume, or
    dense + adopt. Keep our plan structure.
  - Wrap the dense `restore_target_cache` in main's failure handling, before
    `state_ladder_clear` and the adopt sync.
  - Set `cur_pos = 0` on a consume failure too.
