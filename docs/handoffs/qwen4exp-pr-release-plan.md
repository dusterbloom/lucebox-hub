# qwen4exp PR & release plan (2026-09-19)

Planning the upstream PR for the hand-written Qwen3.8-Flash-Next backend
(`server/src/qwen4exp/` + the ggml-cuda kernels it needs). Branch
`feat/qwen4exp-strix-halo`, currently 75 commits ahead of `origin/main`
(`e0048e01`).

## 1. The upstream bar (CONTRIBUTING.md + merged PRs)

`CONTRIBUTING.md` is explicit:
- **One concern per PR.** Kernel/algorithm changes, docs, build config in
  separate commits or PRs.
- **Benchmark before and after**, same hardware/power/warmup, with methodology.
- **Conventional commits** (`feat|fix|refactor|perf|docs|test|bench|chore|ci(scope):`).
- Run the existing correctness check; `make lint` (ruff check + format).

Recent merged PRs set the size/style bar:

| PR | title | adds/deletes | files | commits | tests touched |
|---|---|---|---|---|---|
| 738 | WMMA paged-attention kernel (RDNA4) | +1776/-16 | 13 | 11 | 4 test/bench files + harness |
| 723 | suspend decode to RAM | +2034/-64 | 22 | 2 | 4 test files |
| 720 | separate model placement/load balance | +2064/-340 | 28 | 2 | 6 test files |
| 687 | kernel qualification (GDN/MoE/DS4) | +1691/-15 | 9 | 2 | **all tests** |

Takeaway: each PR is single-purpose, 1.7–2.1k additions, **2–11 commits**, and
ships a *substantial* test suite with the change.

## 2. Our branch vs that bar

- **+9220/-79, 70 files, 75 commits.** Grouped:
  `server/deps/llama.cpp/ggml` **+5171 (44 files)**, `server/src/qwen4exp`
  +2703 (8 files), other `server/src` +25, `server/test` **+388 (3 files)**,
  `harness` +487/-287 (4 files), `docs` +807 (3 files).
- Commit types: 44 `qwen4exp:`, 14 `docs:`, 7 `ggml-cuda:`, 2 `server:`,
  2 `harness:`, 2 `ggml:`, 1 `test:`, 1 `WIP`, 1 merge. **Not conventional**
  (`qwen4exp:` is a scope, not a type), and the 14 `docs:` are internal handoff
  churn (including reverted experiments).
- **Tests are ~5% of the change** (388 lines / ~8000), vs upstream ~20–100%.
- **Comments are ~12% of added lines**; several files carry long rationale.
- `docs/handoffs/qwen4exp-*.md` (807 lines) are **internal** (reference `/tmp`,
  local paths, session/review notes); only `pr651-*.md` is on main. They should
  not go upstream as-is.

## 3. Answer: squash — yes, and split

75 commits across one PR violates "one concern per PR" and is unreviewable.
Squash to a small, logical series. Two options:

**A. Stacked PRs (preferred, matches the repo norm)**
1. `feat(qwen4exp): hand-written Qwen3.8-Flash-Next backend` — `server/src/qwen4exp/*`,
   `backend_factory`/`model_capabilities` wiring, `ENVIRONMENT.md`,
   `smoke_qwen4exp_forward`, one `docs/qwen4exp.md`.
2. `perf(ggml-cuda): QSA sparse attention + packed get_rows` — `qsa.cu|.cuh`,
   `qsa-decode*`, `fattn*`, `ds4-indexer`, `getrows.cu`.
   (If the kernels cannot land without the consumer, land 1 and 2 as **one PR
   with two clearly separated commits**.)
3. `perf(ggml-cuda): bf16 HC stream, cuBLAS/MMB route, Q5_K shadow` — `mmb.cu`,
   `mmb-quant.cuh`, `ggml-cuda.cu`, `hc-cn.cu`, `convert.cu`, `dequantize.cuh`.
4. `test(qwen4exp): kernel differentials + correctness gates` — `server/test/*`.
5. `harness(recall): planted-fact suite + GSM/Math scorer fix` — `harness/*`.
6. `docs(qwen4exp): design + benchmark methodology` — distilled from the
   handoffs; **drop the internal `docs/handoffs/qwen4exp-*.md`** from the PR.

**B. One PR, clean commit series** — same 6 groups as commits, conventional
messages, on top of a rebase of `origin/main`.

Squash procedure (history rewrite → force-push; only on explicit go-ahead):
```
git fetch origin main
git rebase --onto origin/main $(git merge-base origin/main HEAD) feat/qwen4exp-strix-halo
# or, to re-group:
git reset --soft origin/main && git commit ...   # per logical group
```

## 4. Tests to add (the real gap)

Differentials/regressions under `server/test/`, wired into CTest:
1. `convert` vectorized bf16<->f32 == scalar (bit-exact) — trivial, high value.
2. `k_get_rows_scalar` (ne00==1) == generic `k_get_rows` (bit-exact).
3. Q5_K bf16 shadow GEMM == Q5_K mmb (extend `test_mmb_cublas_diff`).
4. HC16 mark: fused bf16 HC down/up == unfused f32 (extend `test_ds4_mmid_regression` style).
5. **Indexer-K cache: chunked-prefill QSA == single-chunk QSA** (logits distance + planted).
6. QSA selected attention == dense attention on random masks (kernel-level).
7. Mask elision: FA counter asserts `qsa>` and that the dense mask is absent.
8. Harness scorer unit tests (GSM/Math extraction, `\frac`, intervals) — CPU, CI-safe.
9. Extend `smoke_qwen4exp_forward` to a golden-logits/planted check.

CI: mirror PR 694 ("enable model-backed ROCm tests") for a qwen4exp job on a
gfx1151 runner; the CPU-only scorer tests can run on hosted runners.

## 5. Comment & docs cleanup

- Target ≤5% comment lines. Keep one-line "why"; move multi-paragraph rationale
  and `journey step N` / ROCm-10 / GLM-review / `/tmp` references out of code.
- Hot spots: `qwen4exp_graph.cpp` (1048 lines, heavy block comments),
  `mmb.cu` (1099), `ggml-cuda.cu` route comments, `qwen4exp_loader.cpp` (825).
- Replace the internal handoffs with one `docs/qwen4exp.md` (architecture,
  measured numbers, methodology) for the PR.

## 6. Release

- Rebase onto `origin/main` (`e0048e01`); drop the `Merge` commit.
- `make lint` clean (done for our harness files: `e6ff6611`; note the format
  pass on `client_test_runner.py`).
- Build + run the correctness gates (`ninja test_server_unit test_kvflash
  bench_paged_attention`; then the CTest subset) — Tier 2 evidence.
- PR description: before/after table (min-of-8 prefill 16,366: 924 → 988 t/s;
  64K 912 → 926; HE 10/10, GSM 10/10, Math 9/10, recall 2/2) + methodology
  (min-of-N prefill-only; planted gate; harness suites).
- README "Supported Models" row + `docs/specs/model-cards.md` + `/props`
  capability entry; ENVIRONMENT.md var list.
- Conventional-commit rewrite of the 44 `qwen4exp:` commits
  (`feat`/`perf`/`test`/`docs` by what they actually do), 14 `docs:` folded to one.

## 7. Immediate next steps

1. Rebase onto `origin/main` and re-group (after explicit approval to rewrite
   the fork branch).
2. Add tests 1–5, 8 (the cheapest, highest-signal ones).
3. Comment trim on the four hot files.
4. Write `docs/qwen4exp.md`; exclude the internal handoffs.
5. `make lint`, build the gates, open PR(s) with the before/after table.
