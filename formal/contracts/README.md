# Approved advisory formal contracts

`registry.toml` is the source of truth for deterministic selection of the
approved minimal formal boundaries. It coexists with `../manifest.toml`, which
preserves compatibility with the verifier's legacy comparison mode.

Each target records the exact production symbol and signature, approved
template, execution bounds, mutable implementation paths, immutable contract
paths, and paired native regression. Templates use only literal `{{ID}}`,
`{{SYMBOL}}`, `{{SIGNATURE}}`, and optional declared variables. The planner
substitutes those tokens deterministically; it does not generate contracts.

The current targets have `policy = "advisory"`. The workflow may report a
failure, but this change neither makes its status required nor alters branch
protection. Promotion to a failing or required check is a separate per-target
contract-review PR and repository-administrator decision.

## Trust boundary

For a pull request, the companion planner reads this registry and every
selected contract blob from the exact trusted base revision. It records the
immutable inputs in the plan, then evaluates them against the exact proposed
head. A PR therefore cannot change the policy or template used to judge
itself. The workflow separately records the base, head, and repositories in
the result artifact.

The separate nightly soak evaluates the accepted revision with extended
bounds and replays mutation sensitivity. Its outcomes and artifacts remain
advisory.

## Registered boundaries

The registry contains three complementary prefix-cache capsules, one advisory
spec-commit capsule, and one advisory KVFlash capsule:

- `prefix-cache-inline` maps to the prepare, confirm, and exact-lookup
  harness.
- `prefix-cache-abort-hole` checks the bounded scalar free-slot selector and
  pairs it with the base-approved native regression for the production
  prepare/abort/prepare integration point. Its historical call-site mutation
  is an explicit immutable patch under `mutations/`.
- `prefix-cache-full-lifecycle` is advisory while its full snapshot lifecycle
  proof and mutation sensitivity are evaluated.
- `spec-commit-exactness` checks the shared speculative acceptance, optional
  bonus, commit budget, and safe token-selection boundary at widths four and
  eight.
- `kvflash-residency-map` checks the production CPU ownership map at four
  blocks for PR planning and five blocks for nightly planning. Its optional
  `nightly_timeout_seconds` is 240 seconds; targets that omit that field
  inherit their PR timeout.

All five templates call production code rather than duplicate it. The full
lifecycle production state extraction and native regression are included in
this cumulative change. The full lifecycle wrapper's quoted harness-body
include is also an immutable contract path and trigger, so protected template
inputs cover that transitive formal source. The spec-commit wrapper's shared
harness body receives the same protection, as does the KVFlash wrapper body.

`[[critical_paths]]` also describes narrow state-machine areas that deserve
review when changed without a matching target. An unmatched critical path is
reported as an advisory coverage gap, never as verified. `watch_paths` and
`include_roots` are routing hints only; they do not expand formal coverage.

## Local validation

```bash
python3 scripts/formal_plan.py validate
python3 scripts/formal_plan.py plan \
  --changed-path server/src/server/prefix_cache_state.h
python3 scripts/formal_plan.py plan \
  --changed-path server/src/qwen35/qwen35_backend.cpp
python3 scripts/formal_plan.py plan \
  --changed-path server/src/common/kvflash_pager.h
python3 -m unittest formal/contracts/tests/test_formal_plan.py -v
```

The `emit` command renders selected templates into an output directory and
records their hashes. It does not invoke ESBMC or modify the manifest lane.
