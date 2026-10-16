# Codex root cause of the mode-4 drift, 2026-10-16 ~13:19 CEST, relayed by the codex-rescue agent

Codex thread 01a1445f-8623-7951-b4e7-823cf81eea32, three runs. Run 1 (original brief) is saved in full at
/home/peppi/.claude/projects/-home-peppi-Dev-lucebox--claude-worktrees-kvflash-am/1e9aa6e6-56b6-44bd-940f-2f749c364caf/tool-results/b623l646q.txt
and is superseded by the two follow-ups below. Static review only: no edits, builds, model runs or GPU work.
Line numbers are on 2e83d216 / 4eb3ba60 (worktree agent-a9111fc724dda7b31) unless they point into evidence/.

## Summary

- Confirmed: truncate() keeps history-dependent physical placement. It pushes freed physical blocks onto
  free_blocks_ in descending logical order; refill_after_truncate_() sorts chunk ids but slot_for() pops the
  history-dependent stack; surviving residents keep their old blocks. kvflash_pager.h:217-262, 350-369.
  Red test 4eb3ba60 fails on a plain LRU overflow then truncate (test_kvflash_pager_serde.cpp:662-663) and after a
  reselect (:646-647).
- Confirmed: pool attention reads physical K/V views with slot masks; CUDA reduces over physical addresses, so a
  permutation changes floating-point grouping. qwen35_target_graph.cpp:1446-1471, fattn-vec.cuh:244-296.
- Hypothesis (answer level): this rounding change explains the 21 drifting skipped probes and OFF's history
  dependence. Not proven; proof is a placement-only fix that removes OFF's history dependence and every
  skipped-probe difference, or first-divergence logits with argmax margins.
- Plain mode 0 drifts too, with paging and live restore: reduced-history OFF differs from full-history OFF on four
  probes, two of them short p12 probes (evidence/agentic/rlru4/drift/cmp_drift.out:3-14).
- Refuted: the stale device-to-host CPU-read lead. The exact scorer drains paging before taking host pointers
  (qwen35_backend.cpp:2963-2999, fix 43c407ed is an ancestor of 2e83d216); serialize() drains before its memcpy
  (kvflash_pager.h:602-632); ladder snapshots use skip_kv=true (backend:1034-1042); truncate drains before
  touching backing storage (pager:215-237). Pinned-buffer lifetime is fenced (pager:904-920).
- The state ladder propagates the difference (rungs recomputed over the current layout after restoring C=5632)
  but is not an independent cause.
- Open, lower confidence, possibly a correctness bug rather than rounding: GPU handoff race. Page-out queues the
  device-to-host copy and the zeroing on the pager stream and releases the block at once; the decode step calls
  slot_for() and computes on a nonblocking stream with no explicit wait on the pager stream. A reused slot could
  be written before the queued copy or zeroing runs. pager:389-393, backend:3237-3265, common.cuh:1634-1643.
  Missing: a stream timeline or a corrupted-byte demonstration.
- Other reduction-order inputs checked: window and sink (fa_window=0 on these paths; sinks protect residency
  only, backend:2503, 3224, pager:738-742); mask epoch (regeneration only, backend:2912-2919); KV splits,
  parallel_blocks and stream-K depend on shapes and SM count, with no evidence of history dependence at matching
  shapes (fattn-common.cuh:1143-1211, 991-1002).

## Fix direction (Codex)

When the retained prefix fits, after truncate:
1. Fence paging and stage the authoritative retained KV bytes (resident host copies may be stale, pager:864-866).
2. Move every retained chunk, including surviving residents, to block_of(c) = c, with overwrite-safe staging or a
   cycle-safe permutation.
3. Rebuild the free stack so later appends get N, N+1, ...
4. Keep partial-chunk zeroing, clipped pins, scores and policy state; advance the mask epoch.
Apply it before primer replay. Sorting the free list alone is not enough.

Minimal CPU reproduction (eight-block fixture): append chunks 0-5, append 6-13, truncate to six chunks; the
retained map becomes [0,7,1,2,3,4] against a fresh [0,1,2,3,4,5]. Answer-level reproduction: the 13 requests of
evidence/agentic/rlru4/drift/ids.json; OFF and ON must agree on the skipped probes (two fail today).
