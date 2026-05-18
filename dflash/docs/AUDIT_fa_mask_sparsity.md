mtp: tree-mode FA mask sparsity audit

Audit date: 2026-05-18
Branch: feat/mtp-tree-arena
Commit range: 843f266 (Phase 2 arena) .. HEAD

## Question

In tree mode (DFLASH27B_MTP_TREE_B=2 K=2, gamma=2), does sibling p=0
attending the arena window accidentally read K/V slots that belong to sibling
p=1? If yes, is it a correctness bug (cross-sibling leakage) or only a
performance problem (wasted FA compute)?

## Answer: already correct — sparse-per-path by index routing

Sparsity is enforced at the `get_rows` index level, not via the mask tensor.
No code change required.

## Mechanism (file:line references)

### Arena layout

`ensure_arena_` (qwen36_mtp.cpp:2128) allocates `slots = B_max * gamma_max`
rows. For B=2 gamma=2 this is 4 rows. Path ownership is implicit:
  - path 0: arena rows 0, 1
  - path 1: arena rows 2, 3 (== gamma_max, gamma_max+1)

### Write routing (per-step)

`step_batch_gpu_arena_` (qwen36_mtp.cpp:2803):

    arena_slot = path_id * gamma_max + depth

path_id=0 depth=0 -> slot 0; path_id=1 depth=0 -> slot 2. No aliasing.

### Read routing (per-step)

`push_kv_slot_inputs_` (qwen36_mtp.cpp:1996-2001):

    if (abs_pos >= arena_base_pos) {
        slot = arena_path_base + (abs_pos - arena_base_pos);
        row[i] = n_ctx + slot;   // arena region index
    } else {
        row[i] = abs_pos;        // chain warm prefix
    }

`arena_path_base = path_id * gamma_max` (pushed at qwen36_mtp.cpp:2871).

For path_id=0: arena indices = n_ctx+0, n_ctx+1.
For path_id=1: arena indices = n_ctx+2, n_ctx+3.

The composed read tensor is `concat(head_k_cache, arena_after)` along dim 1
(qwen36_mtp_graph.cpp:355-356). `ggml_get_rows` at qwen36_mtp_graph.cpp:358
materialises only the slots listed in `inp_kv_idxs_read`. Sibling slots never
appear in the index array of the other path and are therefore never gathered.

### Mask tensor role

`inp_kv_mask` (shape [fa_max, 1], qwen36_mtp_graph.cpp:228) only masks
inactive trailing rows (indices >= fa_kv_n) to -INF (qwen36_mtp.cpp:2013-2016).
All live rows have mask value 0. The mask plays no role in per-path
isolation — that is fully handled by the index routing above.

## Conclusion

The existing implementation is equivalent to a per-path sparse mask: each
path attends only its own slots because only its own slots are ever loaded by
`get_rows`. There is no cross-sibling K/V leakage (not case a) and no wasted
FA compute over sibling slots (not case b). The FA window seen by each path
is exactly [full_prefix x gamma] as intended.
