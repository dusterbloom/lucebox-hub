# K3 upstream PR stack progression

This document tracks how the Kimi K3 integration is split into reviewable,
dependency-ordered pull requests. It is the source of truth for branch scope,
not a claim that the complete K3 integration is smaller than it is.

## Frozen audit baseline

The initial scope audit was performed on 2026-08-21 against these immutable
commits:

- upstream `Luce-Org/lucebox:main`: `21506614cf99be7fc339fdacb6ae05c17fa22503`
- upstream tree: `bbe79803d80d9e04da6815c6cb477e90f83769aa`
- archive head `dusterbloom:agent/k3-gpu1-device-chain-orderly`:
  `9b445796e90966442f2b06a4d69f00ed7d220397`
- archive tree: `bc07ac2a35eb4b7881376ac157ac9f3fc29a84a0`
- merge base: `21506614cf99be7fc339fdacb6ae05c17fa22503`
- full comparison: 186 commits, 474 files, 123,872 insertions and 532
  deletions
- recent production delta, `582fed0..9b44579`: 15 files, 2,579
  insertions and 16 deletions
- recent P56-P63 delivery window, `3b0e6bd..9b44579`: 52 files,
  9,018 insertions and 71 deletions
- full binary patch SHA-256:
  `7b5b1063b0456d4194ffbbe3952063a7de8b958c0792e9d336b06a6c31bc0740`
- full numstat SHA-256:
  `8e36872e6023270bf0c30cf179e691389d5e5e73778f86b2bf3d52ee6ecf8450`

The full comparison is large because upstream `main` has no K3 runtime. It
contains the complete K3 and NVMe/MoE integration lineage, not merely the
recent production-parity work. Committed experiment output is a secondary
source of volume: `results/` accounts for 172 files and 52,844 inserted text
lines, plus one NPZ artifact.

## Non-destructive split policy

1. Keep `agent/k3-gpu1-device-chain-orderly` intact as the provenance and
   integration archive. Never force-push or destructively rebase it.
2. Create new stacked delivery branches from the current upstream `main`.
   Materialize reviewed paths from an immutable archive commit; do not assume
   that cherry-picking the recent tip onto `main` is sufficient.
3. Merge the stack from the bottom upward. Every PR names its exact parent
   branch and immutable source commit.
4. Keep generated benchmark results and binary artifacts out of code PRs.
   Publish them separately and retain a small manifest containing identities,
   hashes, schemas, and the producing source commit.
5. Target fewer than 10,000 changed lines per review unit where dependencies
   permit it. This is a reviewability target, not a promise that the complete
   K3 implementation is below 10,000 lines.
6. Every stack update must refresh the coverage and exclusion checks below.

Current overall status: `AUDITED_UNSTACKED`.

The foundation stack remains uncut. Focused follow-up branches may be frozen
and reviewed in the meantime, but they are not upstream-ready until replayed
onto their owning stack layer.

## Audited path inventory

These non-overlapping groups account for all 474 paths in the frozen archive
comparison. They are an ownership census, not necessarily the final PR
boundaries: groups above the review budget must be subdivided without changing
their aggregate path coverage.

| Inventory group | Owned paths | Files | + / - | Patch SHA-256 |
|---|---|---:|---:|---|
| I-01 ggml prerequisites | `server/deps/llama.cpp/**` | 8 | +196/-8 | `ff7c2feec0af70c2fe5d01e233a37c08999a14d07c8ea2a683780056c1385628` |
| I-02 generic runtime | common MoE plus required common/model runtime paths | 40 | +7,673/-512 | `92572f00ecb58ba04e21dd81a01aa2ae47ff003b3958f398e7041b3fd4eb3bff` |
| I-03 K3 runtime | `server/src/kimi_k3/**` plus its K3 CMake ownership | 18 | +17,933/-1 | `03993217ae2547d96a746fadd633f6cb19ddd4673ec8ab17f8fc64b45c876049` |
| I-04 product surface | server HTTP surface, model card, CI and variables | 9 | +115/-4 | `e8f6ca1ae717c2d3f713d675c8cc99437934bf4bcc34a4dad0c06bd48dd75c1e` |
| I-05 native tests | `server/test`, `server/tests`, `server/scripts`, `server/docs` | 57 | +12,074/-7 | `08032a96132de6c20d21681268f274190847607eb1d0270a2949e6e58cc5be54` |
| I-06 offline tools | top-level `scripts/**` | 99 | +23,177/-0 | `43d04f2b8e32fa44ecff5b4a288af4384b2d4a9eaff593a5e85fd7435db2a9a6` |
| I-07 design docs | `docs/**` and `research/**` | 71 | +9,860/-0 | `e4807e6e87d7cd2c3de1429fb7b6190d313ad214b35b66dfb7a52b7999125057` |
| I-08 evidence | `results/**` | 172 | +52,844/-0 plus 2,899,576-byte NPZ | `c773d3c65280765a83e5cf60534dde70b9b99ba6ae2619c252c5c74aab89a9aa` |

I-08 is evidence, not production source, and is excluded from the code merge
stack unless maintainers explicitly approve a separate evidence-pack review.
I-03, I-05, and I-06 exceed the target review size and therefore require the
smaller delivery layers below.

## Proposed stack

Exact path manifests and line budgets are frozen before creating each branch.
The sizes below are audit estimates and must be replaced with measured values
in the progress ledger.

| ID | Proposed branch | Responsibility | Depends on | Initial budget | Status |
|---|---|---|---|---:|---|
| K3-00 | `stack/k3-ggml-prereqs` | Eight focused in-tree ggml changes required by K3/MoE execution | upstream `main` | +196/-8 | planned |
| K3-01 | `stack/k3-moe-storage-stream` | Common MoE storage, NVMe scheduler, hybrid stream interfaces and focused tests | K3-00 | about +6,100/-350 | planned |
| K3-02 | `stack/k3-model-core` | K3 tensor model, loader, core backend declarations, and panel artifact representation | K3-01 | less than 10,000 target | planned |
| K3-03 | `stack/k3-graph-kernels` | K3 graph construction, ordered join, sparse scatter, and graph-focused tests | K3-02 | less than 10,000 target | planned |
| K3-04 | `stack/k3-progressive-provider` | Progressive provider and its bounded storage/cache behavior | K3-03 | less than 10,000 target | planned |
| K3-05 | `stack/k3-runtime-glue` | Runtime backend orchestration, factory/CMake wiring, feature gates, model card, smoke and integration tests | K3-04 | measured before cut | planned |
| K3-06 | `stack/k3-production-parity` | Production HTTP token trace, parity/telemetry qualification and frozen P30 GPU-residency simulator | K3-05 | +2,579/-16 at audit | planned |
| K3-07 | `stack/k3-operator-tooling` | Reusable deployment and analysis tools that are required after runtime merge | K3-06 | split again if above 10,000 | planned |
| K3-R | separate research archive | Historical experiments, broad research docs, raw JSON/CSV/NPZ results | none; not merge-blocking | excluded from code stack | planned |

The K3 runtime and its required common infrastructure cannot truthfully be a
single sub-10,000-line PR against upstream `main`: `server/src/kimi_k3/` alone
adds 17,598 lines at the audit baseline. The stack makes individual review
units smaller while preserving the complete dependency chain.

## Progress ledger

Update one row only after its branch, base, path manifest, diff hash, and tests
are frozen.

| ID | Base SHA | Head SHA | Files | + / - | Diff SHA-256 | Tests | Review | PR |
|---|---|---|---:|---:|---|---|---|---|
| K3-00 | pending | pending | pending | pending | pending | pending | pending | pending |
| K3-01 | pending | pending | pending | pending | pending | pending | pending | pending |
| K3-02 | pending | pending | pending | pending | pending | pending | pending | pending |
| K3-03 | pending | pending | pending | pending | pending | pending | pending | pending |
| K3-04 | pending | pending | pending | pending | pending | pending | pending | pending |
| K3-05 | pending | pending | pending | pending | pending | pending | pending | pending |
| K3-06 | pending | pending | pending | pending | pending | pending | pending | pending |
| K3-07 | pending | pending | pending | pending | pending | pending | pending | pending |

## Required checks for every stack layer

Record the exact output or artifact hash in the ledger or linked PR.

```bash
git merge-base <base> <head>
git rev-list --left-right --count <base>...<head>
git diff --shortstat <base>...<head>
git diff --numstat <base>...<head>
git diff --check <base>...<head>
git diff --binary <base>...<head> | sha256sum
```

Additionally:

- list every changed path and assign it to exactly one stack layer or the
  explicit research exclusion manifest;
- fail the code stack if `results/` or binary benchmark output appears;
- verify all modified upstream files, not just newly added K3 files;
- configure and build each layer on its declared base, with CUDA/HIP build
  parallelism capped at `-j4` on the development hosts;
- run focused unit tests at the layer where a source is first compiled;
- run complete K3 model-free gates after K3-05 and again after K3-06;
- require independent review before changing a row to `ready`;
- refresh from upstream by an explicit merge/rebase commit on a temporary
  integration branch, then re-freeze hashes before opening or updating a PR.

## Coverage and omission proof

Before the top code PR is declared complete:

1. Produce a sorted path manifest for the archive comparison.
2. Produce a sorted union of every K3-00 through K3-07 path manifest.
3. Produce the explicit K3-R exclusion manifest.
4. Require the union of delivery and exclusion manifests to equal the archive
   path manifest exactly, with no duplicates.
5. Compare the stacked code tree with the archive tree while excluding only
   K3-R paths. Any remaining diff is an unassigned or accidentally omitted
   change.
6. Confirm that all files modifying existing upstream behavior—especially
   common runtime, factory, CMake, CI, and ggml paths—have an owning PR and
   focused tests.

## Current work not yet assigned

New work is assigned only after it has passed its own correctness review.
Validated follow-ups are tracked here without pretending that their archive
base is an upstream-review base:

| Follow-up | Parent | Head | Files | + / - | Patch SHA-256 | Validation | Intended owner |
|---|---|---|---:|---:|---|---|---|
| K3 response framing | `8a319bdf` | `54023bdd` | 12 | +532/-70 | `227e6db024263eaf44c17a372c5451f82374da3bc0ef132dc57f50988261413d` | Lucebox4 HIP `-j4`; K3 9/9; server unit 397/397; Math10 HTTP/trace 10/10 | K3-05 |
| Math answer equivalence | `54023bdd` | `ace48041` | 2 | +131/-2 | `bc77c85756c37ebe55fda94d2e6cc2ff48c35f6a57ed8124bd0793e562d5bdda` | focused tests, pycompile, diff check; immutable Math10 rescore 7/10 versus raw 4/10 | K3-07 |
| K3 reasoning policy | `54023bdd` | `bdb2ca83` | 7 | +597/-97 | `d6f4d039fee18c88236841f85124113aaa10595268ec415704d86285f7402dd0` | independent GO after two NO-GO review cycles; capped build and focused policy/props/Qwen/P55 gates | K3-05 |

These heads are published respectively as
`agent/k3-output-framer`, `agent/k3-math-scorer`, and
`agent/k3-reasoning-policy`. They remain follow-ups to the integration archive,
not substitutes for K3-00 through K3-07. Before opening upstream PRs, recreate
each change on the exact owning stack parent, recompute its diff hash, and rerun
the layer gates.

The HumanEval score-only sandbox, D0 width-four exactness probe, and any future
production optimization remain outside the frozen baseline until their source,
tests, build, and evidence are complete. A new audited archive head must not be
hidden inside an unrelated layer.
