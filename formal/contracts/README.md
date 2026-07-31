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

## Registered boundaries

The registry contains two complementary prefix-cache capsules:

- `prefix-cache-inline` maps to the prepare, confirm, and exact-lookup
  harness.
- `prefix-cache-abort-hole` checks the bounded scalar free-slot selector and
  pairs it with the base-approved native regression for the production
  prepare/abort/prepare integration point. Its historical call-site mutation
  is an explicit immutable patch under `mutations/`.

Both templates call production code rather than duplicate it. The component
depends on the production state extraction and regression from the preceding
prefix-cache correctness PR.

`[[critical_paths]]` also describes narrow state-machine areas that deserve
review when changed without a matching target. An unmatched critical path is
reported as an advisory coverage gap, never as verified. `watch_paths` and
`include_roots` are routing hints only; they do not expand formal coverage.

## Local validation

```bash
python3 scripts/formal_plan.py validate
python3 scripts/formal_plan.py plan \
  --changed-path server/src/server/prefix_cache_state.h
python3 -m unittest formal/contracts/tests/test_formal_plan.py -v
```

The `emit` command renders selected templates into an output directory and
records their hashes. It does not invoke ESBMC or modify the manifest lane.
