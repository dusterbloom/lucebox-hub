# Advisory formal-verification pilot

This directory defines three deterministic proof capsules and the planning tools
that select them. `Formal Verification (advisory)` runs the same capsules for
proposed changes, while `Formal Verification Nightly (advisory)` exercises
their extended bounds and mutation sensitivity on accepted revisions. Neither
workflow is required: this pilot adds no branch-protection rule, credentials,
AI integration, or repair execution.

The capsules run against the production prefix-cache transition boundaries,
including the full lifecycle extraction added in this cumulative change.

## Current capsules

`prefix-cache-inline` checks the production
`InlinePrefixCacheState` prepare, confirm, and exact-lookup path for symbolic
cache capacity, branch, and prefix depth. Its checked properties and exclusions
are documented in
[`prefix_cache/PROPERTIES.md`](prefix_cache/PROPERTIES.md).

`prefix-cache-abort-hole` checks the scalar free-slot selector. The complete
local plan also compiles and runs the paired native regression against the
selected source revision, covering the production
prepare/confirm/prepare/abort/prepare sequence. The formal and native
guarantees are separated in
[`prefix_cache/ABORT_HOLE_PROPERTIES.md`](prefix_cache/ABORT_HOLE_PROPERTIES.md).

`prefix-cache-full-lifecycle` is an advisory capsule for the full snapshot
key/slot/boundary/victim lifecycle. Its fresh formal trace and wider exact-head
native regression are documented in
[`prefix_cache/FULL_LIFECYCLE_PROPERTIES.md`](prefix_cache/FULL_LIFECYCLE_PROPERTIES.md).
The production extraction and native test are included in this cumulative
change.

Only behavior named in those property documents is claimed as checked.

## Policy files

[`manifest.toml`](manifest.toml) is the compatibility description consumed by
the verifier's legacy local mode.

[`contracts/registry.toml`](contracts/registry.toml) records the same three
capsules as deterministic templates, along with their source triggers, bounds,
mutable implementation paths, immutable contract paths, and critical-path
routing metadata. [`contracts/README.md`](contracts/README.md) describes the
registry and planner in detail.

All three targets use `policy = "advisory"` during the proving period. The
workflow records failures and preserves their evidence, but the policy does
not make the check a merge requirement. Promoting an individual target to a
failing or required check is a separate contract-review PR and
repository-administrator decision.

## CI trust boundary

For pull requests, the workflow itself runs from the target branch. It reads
the registry and templates from the exact trusted base commit, checks out the
exact proposed head without persisting credentials, and refuses a SHA
mismatch. The only container image it executes is the verifier image whose
digest is pinned both in the workflow and in base policy.

The repository is mounted read-only. Only plan and result directories are
writable, and verifier containers run without network access, Linux
capabilities, or a writable root. Plans, results, and source/base SHA metadata
are retained as workflow artifacts. The legacy manifest run remains an
advisory comparison during migration.

The nightly workflow applies the same verifier-only boundary to the exact
scheduled or manually dispatched revision. It records generated, legacy, and
mutation outcomes without turning a soak failure into merge policy.

## Local validation

Registry and planner validation do not require Docker:

```bash
python3 scripts/formal_plan.py validate
python3 scripts/formal_plan.py plan \
  --changed-path server/src/server/prefix_cache_state.h
python3 -m unittest formal/contracts/tests/test_formal_plan.py -v
```

The full local verifier uses the immutable verifier image declared in the
registry. It requires Docker and a committed checkout:

```bash
./scripts/formal.sh --all
./scripts/formal.sh --nightly
./scripts/formal.sh --all --legacy
./scripts/formal_mutation_test.sh
```

`--base-sha REVISION` is available for local comparison only after that
revision contains the registry; it is not a bootstrap command for this first
registry change.

Results are written to `.formal-results/`. Set
`LUCEBOX_FORMAL_IMAGE` only when deliberately testing a different companion
image.

Local verifier containers use the same networkless, capability-free,
read-only-workspace boundary as CI. Immutable image digests make the runs
reproducible.

## Adding a contract

1. Extract a dependency-light production boundary; do not verify a duplicate
   implementation.
2. Write a deterministic template and a property document that separates
   checked properties from exclusions.
3. Add a deterministic native regression for integration behavior that is not
   captured by the scalar contract.
4. Declare exact symbols, triggers, PR/nightly bounds, mutable paths, and
   contract paths in the registry.
5. Keep the compatibility manifest and registry execution settings aligned.
6. Run both bounds and demonstrate mutation sensitivity using an external
   patch under `contracts/mutations/`.
7. Keep new targets advisory until their bounds, failure behavior, and
   artifacts have been reviewed.

Promotion from advisory reporting to a failing or required check is performed
per target in a separate reviewed change; this pilot does not alter repository
settings.
