# Ship Plan Handoff — feature/mtp-foundation-v2

**Date:** 2026-05-16
**Branch:** `feature/mtp-foundation-v2` (25 commits ahead of `origin/main`, local-only — `git ls-remote origin feature/mtp-foundation-v2` returns empty)
**Decision:** Option B — split into 4 PRs, all ≤3,000 LOC added
**Source review:** `/tmp/codex_ship_strategy.md` (general-purpose agent, codex-rescue substitute)
**Companion memo:** `thoughts/2026-05-16-mtp-optimization-day.md`

## Why split (Option B)

- Repo norm: median merged PR ~3.5 files / +100 LOC. Largest historical (#175) +8,759 LOC.
- This branch is **+10,184 LOC**, 43 files — bigger than any PR in repo history.
- Option A reviewer load: ~3.3 hrs single-sitting; ~certain to earn "please split this" as the first comment.
- Option B: ~25-55 min per PR, parallelizable across reviewers, individually revertable.

## The 4 PRs

| # | Title | LOC | Files | Review | Depends on |
|---|---|---|---|---|---|
| 1 | MTP infrastructure foundation (GGUF, interfaces, chain runner, docs) | ~2,400 | ~17 | ~50 min | main |
| 2 | Qwen3.6 NextN MTP head | ~2,650 | ~12 | ~55 min | PR1 |
| 3 | Verify-side optimizations (chain draft, step cache, hidden capture) | ~1,700 | ~9 | ~35 min | PR2 |
| 4 | Tree-verify + CUDA regression test + bench harness | ~2,800 | ~14 | ~60 min | PR3 |
| **Total** | | **~9,550** | **~52** | **~3.3 hr distributed** | |

Squashing eliminates ~635 LOC of intra-branch churn vs the raw branch total.

## Hard prerequisite: split `96215ae`

`96215ae` ("Qwen3.6 NextN head with KV warmup and GPU forward") is itself **+6,025 LOC / 33 files** — 2× the per-PR cap. It must be split during pre-PR rebase before grouping into PR1+PR2.

Procedure:
```bash
git rebase -i origin/main
# mark 96215ae as `edit`
# at the edit stop:
git reset HEAD^
# stage in 3 groups:
# (a) common/ + docs + tests for PR1
git add dflash/src/common/{gguf_metadata.h,gguf_mmap.h,model_backend.h,mtp_interface.h} \
        dflash/src/common/{mtp_chain_runner.{h,cpp},step_graph.h,dflash_target.h} \
        dflash/src/internal.h dflash/docs/ dflash/CMakeLists.txt \
        dflash/test/test_gguf_metadata_unit.cpp \
        dflash/test/test_mtp_interface_contract.cpp \
        dflash/test/test_mtp_chain_runner.cpp
git commit -m "mtp: chain runner + interface + GGUF mmap reader (common/)"
# (b) qwen36/ MTP head sources + tests for PR2
git add dflash/src/qwen36/ dflash/src/qwen35/ dflash/test/test_qwen36_*
git commit -m "mtp: Qwen3.6 NextN head — qwen36/ source + tests"
git rebase --continue
```

## Squash plan (`git rebase -i --autosquash` after the foundation split)

| Bug-fix commit | Squash into | Why | After squash reads as |
|---|---|---|---|
| `ea9ddbf` | `96215ae`-pt2 (NextN head) | recommit base_pos off-by-one + always-on hidden capture + KV leak were never the intended design | "Qwen3.6 NextN head with KV warmup, correct base_pos, opt-in hidden capture" |
| `8a626fe` (momus #1-3) | `82873d7` (fast-path) | The fast-path code path WAS the source of the off-by-one + capture-ownership + sync stream bugs | "skip recommit verify_batch via restore_kv_at_chain fast path (correct base_pos, owning capture, async stream)" |
| `b5f46e6` (momus #5) | `94d8583` (Phase B+) | `94d8583` shipped an O(n_heads × n_ctx) static-view cache; `b5f46e6` is the shape-only rewrite that was always the right design | "Phase B+ — shape-only step graph cache (ggml_set_rows), fused LM-head, F16 head KV, fa_window" |
| `1edf6c8` (momus #4) | `96215ae`-pt2 (or keep visible — reviewer's choice) | n_heads=1 contract + pre-norm fallback warning was always required for correctness | folded into PR2's MTP head commit, OR kept as 1-commit follow-up titled "harden n_heads=1 contract" |
| `2036dfc` + `0d2e770` | `ce62ad8` (flag-guard) | Always-on profiling was never the right form — flag-guards are the correct first version | "instrument step_chain + verify_batch per-iter timing, gated on DFLASH_*_PROFILE CMake options" |

## Kept separate (defended)

- `586a867` (comment essay trim) — reviewer-visible prose cleanup, not a code fix. Squashing into source commits hides that style choices were made.
- `6ce4167` (HIP→CUDA alias delete) — orthogonal dead-code removal. One-line PR-worthy on its own; keeping separate makes that review trivial.
- `4c88375` (CUDA regression test) — *new* test, not a fix. Written explicitly to gate the cast-removal decision.

## Per-PR detail

### PR 1 — MTP infrastructure foundation

- **Files**: `dflash/src/common/` (gguf_metadata, gguf_mmap, model_backend, mtp_interface, mtp_chain_runner.{h,cpp}, step_graph, dflash_target.h), `dflash/src/internal.h`, `dflash/docs/*`, `dflash/CMakeLists.txt`, `dflash/test/{test_gguf_metadata_unit, test_mtp_interface_contract, test_mtp_chain_runner}.cpp`
- **Stand-alone test gate**: `test_mtp_interface_contract`, `test_mtp_chain_runner`, `test_gguf_metadata_unit` — all CPU-only.
- **PR body must include**: explanation for "why interface ships before implementation" (because the interface + CPU-stub tests are the contract PR2 must satisfy).
- **Add CMake gate for GPU tests**: `option(DFLASH_GPU_TESTS "Build GPU-required test targets" ON)` so CI without GPU can opt out.

### PR 2 — Qwen3.6 NextN MTP head

- **Commits**: `96215ae`-pt2 (qwen36/ extraction) + squashes (`ea9ddbf`, `c14d3b1`, `85ee2d4`, `8bae223`, `428abdf`, `0f3b96d`, optionally `1edf6c8`)
- **Files**: `dflash/src/qwen36/{qwen36_mtp.cpp,qwen36_mtp.h,qwen36_mtp_graph.{cpp,h},qwen36_mtp_loader.cpp}`, `dflash/src/qwen35/` hookup hunks, `dflash/test/{test_qwen36_mtp_basic,test_qwen36_mtp_step_unit,test_mtp_topk}.cpp`, `dflash/test/test_qwen36_mtp_e2e.sh`
- **Cap-tight at ~2,650** — if pushing over, peel `test_qwen36_mtp_e2e.sh` (320 LOC) into PR4.
- **Stand-alone test gate**: GPU-required tests (gated behind `DFLASH_GPU_TESTS`).
- **PR body must address**: `qwen36_mtp.cpp` is the largest single cpp in the repo at ~2,092 LOC; flag as "follow-up split" rather than blocking precondition.

### PR 3 — Verify-side optimizations

- **Commits**: `b9f604e` (Phase A chain), `94d8583`+`b5f46e6` squashed (Phase B+ shape-only cache), `2005425` (LAST_ROW_ONLY hidden), `82873d7`+`8a626fe` squashed (fast-path with bugs fixed), `f65532a` (max_gamma ceiling)
- **Stand-alone test gate**: `test_mtp_chain_strict_spec` (byte-identity vs greedy AR), `test_mtp_step_graph_cache`.
- **PR body must include**: Q2 D=3 = 1.67× median + D=5/D=7 negative result (47/41 tok/s — proves chain-depth was empirically explored).
- **IF Q4 like-vs-like bench lands first AND wins**: include the apples-to-apples MTP-Q4 vs DFlash-Q4 comparison here. **IF NOT**: framing is "infrastructure with Q2 strict-spec coverage; perf characterization in follow-up".

### PR 4 — Tree-verify + CUDA regression test + bench harness

- **Commits**: `f8b8fa6` (verify_tree stub), `d3cc9f0` (verify_tree real), `539d85d` (DeltaNet rollback), `8ab7ebe` (`bench_agent_mtp.py`), `4c88375` (CUDA cast regression test), `ce62ad8`+`2036dfc`+`0d2e770` squashed (flag-guarded profiling), `6ce4167` (HIP alias delete), `586a867` (comment trim)
- **Largest PR** at ~2,800. If over cap, peel `586a867` to a trivial PR5.
- **Stand-alone test gate**: `test_mtp_tree_strict_spec` (CPU strict), `test_mtp_step_graph_cuda{,_nocast}` (GPU, gated).
- **Risk to call out**: tree-verify path has **no GPU experiment proving ≥1.0× speedup** — the combined-shot that would have validated it regressed to 0.7× and was reverted. PR body must defend as "infrastructure for the K-way autoregressive draft research project, validated by strict-spec on CPU stub only."

## Sequencing

| Step | Status | Gate |
|---|---|---|
| 1. Q4 loader fix lands | in flight (debug-agent) | unlocks PR3 strong-headline framing |
| 2. Q4 bench MTP D=3 (n=5) | queued, ~10 min | confirms apples-to-apples vs DFlash Q4 |
| 3. Add `DFLASH_GPU_TESTS` CMake gate | not started, ~5 min | required for CI compatibility |
| 4. Split `96215ae` (rebase -i + reset HEAD^) | not started | ~30 min |
| 5. Squash bug-fix commits via `--autosquash` | not started | ~30 min |
| 6. Force-push branch | not started | safe — branch is local-only |
| 7. Open 4 PRs in order, each gated on its predecessor's review | not started | maintainer cadence |

## Risks to ship-readiness

- **Q4 loader bug** (active investigation): if the perf claim cannot be made apples-to-apples vs DFlash Q4, PR3 ships as pure infrastructure rather than "+X% vs DFlash."
- **Tree-verify is unproven at runtime**: only CPU strict-spec coverage. Acceptable per the PR4 defense above, but reviewers WILL ask.
- **GPU-required tests need CMake gate**: 4 of 8 new tests need sm_86+. CI has CUDA toolkit (compile) but no GPU runner. Without the gate, CI breaks.
- **`qwen36_mtp.cpp` at ~2,092 LOC** — flagged for follow-up split in memo. Reviewers may demand precondition split.
- **`gguf_mmap.h` (214 LOC)** likely duplicates upstream `ggml_mmap` — PR1 body must defend the local copy.
- **Reputation context**: per `external-repo-contribution.md`, the user's standing rule is "we want to contribute and first we need to always know the standards." Cite `CONTRIBUTING.md` in each PR body if present.

## Open questions answered

| # | Question | Answer | Source |
|---|---|---|---|
| Q1 | Q4 like-vs-like bench done? | **PENDING** | debug-agent in flight at handoff time |
| Q2 | GPU runner in CI? | **NO** — ubuntu-latest + cuda-toolkit compile only | `.github/workflows/ci.yml` line 1 + grep |
| Q3 | Branch pushed to origin? | **NO — local only** | `git ls-remote origin feature/mtp-foundation-v2` empty |

## Final commit graph (target, post-rebase)

```
* (PR4 tip) mtp: trim comment essays per codex review                        [586a867]
* dflash: drop unused HIP→CUDA aliases in device_runtime.h                   [6ce4167]
* mtp: flag-guard profiling instrumentation (DFLASH_*_PROFILE)               [ce62ad8 ⊕ 2036dfc ⊕ 0d2e770]
* test(mtp): bug #5 cast proof-of-teeth on CUDA                              [4c88375]
* bench: wire experiment-C (MTP top-K -> DDTree) into qwen36-mtp harness     [8ab7ebe]
* mtp: Stage 3 — DeltaNet rollback for tree-verify multi-iter                [539d85d]
* dflash: implement verify_tree on Qwen35DFlashTarget                        [d3cc9f0]
* dflash: add verify_tree virtual on DFlashTarget (stub)                     [f8b8fa6]
*---- PR4 base
* (PR3 tip) mtp: max_gamma() returns chain-depth ceiling (8) post-init       [f65532a]
* mtp: skip recommit verify_batch via fast path (correct base_pos)           [82873d7 ⊕ 8a626fe]
* mtp: switch verify_batch hidden capture to LAST_ROW_ONLY after prefill     [2005425]
* mtp: Phase B+ — shape-only step graph cache, fused LM-head, F16 head KV    [94d8583 ⊕ b5f46e6]
* mtp: Phase A — autoregressive chain draft via INativeMtp::step_chain       [b9f604e]
*---- PR3 base
* (PR2 tip) mtp: harden n_heads=1 contract (or folded silently into PR2)     [1edf6c8]
* mtp: skip h_pre device->host download on final chain iter                  [0f3b96d]
* mtp: feed pre-shared_head_norm hidden as h_prev on GPU chain path          [428abdf]
* mtp: seed chain h_prev_0 from target pre-output-norm hidden (#22673)       [8bae223]
* mtp: enable top-K logprobs on Qwen3.6-27B (no shared_head_head)            [85ee2d4]
* mtp: surface top-K logprobs from Qwen3.6 NextN head (opt-in)               [c14d3b1]
* mtp: Qwen3.6 NextN head — qwen36/ source + tests (correct base_pos)        [96215ae-pt2 ⊕ ea9ddbf]
*---- PR2 base
* (PR1 tip) docs: ARCH_ONBOARDING, BENCH, TEST_TIERS, qwen36_mtp_redesign
* test: MTP interface contract + chain runner + GGUF metadata unit
* mtp: chain runner + interface + GGUF mmap reader (common/ only)            [96215ae-pt1]
*---- PR1 base
* (origin/main)
```

`⊕` = squashed commits.

## Authoritative source files for any reviewer joining cold

- This handoff: `thoughts/2026-05-16-ship-plan-handoff.md`
- Today's work memo (audited): `thoughts/2026-05-16-mtp-optimization-day.md`
- Raw codex review (more detail): `/tmp/codex_ship_strategy.md`
- Evidence audit (number-by-number verification): `/tmp/memo_evidence_check.md`
- Momus BS review (framing critique): (returned in chat at 22:50, not file-persisted)
