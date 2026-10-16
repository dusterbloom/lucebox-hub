# Engine spec: mode 4 scores on a recency pool (`LUCE_KVFLASH_SCORE_POOL_RECENT=1`)

Written 2026-10-16 during the rlru4k ON10 arm. Branch `kvflash/2.0r-prefill-reselect` (implementer worktree). Do not compile until the rlru4k run has ended.

## Why

ON6 loses within-16K probes. Of 27 within-16K probes whose source fell out of ON6's window and was not recalled, the OFF-state query (`../rlru/dump`, `mass_end`, same block grid and sum) holds the source in its top 6 for 22. The engine never hits a within-16K source that the dump misses (`missdump.py` on `on6/`).

Cause: `KvFlashPager::truncate(C)` with `keep > n_blocks_` calls `refill_after_truncate_()`. That keeps every current resident and only fills freed blocks. The fill is ranked by `score_hook`, which still reads the previous request's ledger, because live restore clears `kvflash_scores_` after the truncate. So after a mode-4 request, its k recalled blocks stay resident through the next request's first (scoring) pass. They take k × 640 tokens that OFF spends on the newest history, so the scoring query is formed without the 12–16K-back region.

## Change

1. New env `LUCE_KVFLASH_SCORE_POOL_RECENT` (int, default 0). With 0, behaviour, logs and picks are byte-identical to today.
2. With 1, mode 4 only. When the request will run mode 4 (`rw.run` still true after the `kvflash_pool_fits` check), and before the first batch of the first pass:
   - reselect the pool to the recency set: `score_hook = [](int c) { return (float)c; }`, then `reselect(0)`. Resident = sink plus the newest chunks that fit;
   - then set `score_hook` to empty, so that evictions during the first pass are plain LRU, as in OFF;
   - log `[kvflash] prefill reselect mode=4 recent-pool @<kv_offset>: <events> page events; <ins> in (<MiB> MiB H2D)`;
   - place it right after the `mode=4: R4=` log line in the setup block (qwen35_backend.cpp ~2368). That is after every restore path (live restore `truncate(C)`, pooled-blob restore, cold reset), so it covers each of them.
   - Why plain LRU is OFF's pool: OFF (`QK16KND`) never reselects (its server log has no `reselect` line, because the adaptive tau exceeds the ~20-token probe answers) and never installs a hook.
3. Nothing else changes. `kvflash_prefill_apply` installs its own hook at the rewind, as today.

## Amendments (kvflash-am-a0 review, 2026-10-16 17:36)

A1 (taken). Apply the recency step on every pooled request while the flag is on, before the `rw.run` decision, not only inside the mode-4 branch.
- Invariant: the pool is OFF's plain-LRU set unless this request's mode 4 is installing picks.
- In the rlru4k arms this changes no request. `kvflash_mode4_plan` always sets `run` for a non-empty prompt, so the only skip is "fits", and a fitting prefix is fully resident after `truncate` (`relayout_identity_`). There `reselect(0)` wants every chunk and makes 0 page events.
- Log the recency line on every request while the flag is on, including 0 events.

A2 (not taken). Clearing the stale ledger before `truncate(C)` in live restore is subsumed: `reselect(0)` with the recency hook overrides whatever the refill brought in, and it is the smaller diff.

A3 (structural check, needed by PREREG_v2). New env `LUCE_KVFLASH_LOG_RESIDENT=1`, independent of the flag, so OFF can log too.
- On every pooled request, log the resident chunk set as sorted ranges: `[kvflash] resident @<kv_offset>: n=<N> chunks=<a-b,c,...>`.
- Log point: after every restore path and after the recency step (if any), immediately before the first prefill batch.
- It must not change compute.
- PREREG_v2 asserts that ON's line equals OFF's on every request, with 0 mismatches.

Tests:
- Pager: with the flag off, the stale pick is still resident after `truncate(C)`. With `reselect_recent()`, the set equals that of a fresh no-hook pager truncated at C.
- Add: `reselect_recent()` on a pool whose prefix fits is a no-op (0 events, same set).
- A pure helper that formats the ranges, with one case.

## Test (red first)

In `test_kvflash_pager*.cpp`, or a pure helper if the hook wiring is in the backend:
- Feed a pager more chunks than it holds.
- Reselect with a hook that pins an old block and a short tail (a mode-4 pick).
- Append a few chunks, then `truncate(C)` with C not fitting.
- Apply the recency step.
- The resident set must equal that of a fresh pager fed the same chunks with no hook (plain LRU), truncated at C.
- Also assert that with the flag off, the resident set after `truncate(C)` is unchanged from today (the stale picks are still resident). That pins the bug the flag fixes.

## Build and checks

- Merge `kvflash/2.0r-fast-scorer` (002cb9b9, 43735d13) in the same build. Build to a new dir, `/mnt/kimi-k3/build/prs8-cuda`. prs7 stays untouched.
- `ctest -R "Kvflash|kvflash"` fully green.
- Report the commit, md5 and build time.
