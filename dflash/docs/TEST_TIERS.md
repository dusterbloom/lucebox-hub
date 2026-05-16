# Test Tiers and Trustworthiness Contract

## Purpose

This document defines test trustworthiness tiers for the dflash codebase. An audit of dflash/test/ revealed 3 T1 tests, 13 T2 tests, 8 T3 tests, and 1 stale test out of 25 total. Most tests are model-gated or GPU-gated. Critical paths—verify_batch, snapshot_kv, restore_kv, and the pFlash TQ3_0 decode-only gate—have no T1 coverage. This document establishes tier definitions, names the gaps, and sets the rule for all new tests going forward.

## The Four Tiers

### T1 — Unit Tests (Must stay green on every commit; CI fail-loud)

- No GGUF or model file required; uses scripted fakes and stubs
- No GPU required
- Uses `CHECK()` macro exclusively (always fires, even with `-DNDEBUG`); never bare `assert()` (silently stripped in Release builds)
- Runs in seconds (<5s total for all T1)
- Self-contained, deterministic, repeatable
- Examples: `test_kv_quant.cpp`, `test_mtp_interface_contract.cpp`, `test_mtp_chain_runner.cpp`

### T2 — Smoke Tests (Model-gated; run when file present, skip otherwise)

- Reads a path from environment variable (e.g., `QWEN36_MTP_GGUF`) or CLI argument
- Prints a clear skip message when the file is absent: `[test_name] env unset; skipping`
- If env var is set but test fails, that is a real failure, not a skip
- May load a small GGUF to verify interface contract but does not run performance-intensive operations
- Examples: `smoke_load_target.cpp`, `smoke_qwen36_load.cpp`, model-dependent inference stubs

### T3 — GPU and Performance Tests (Bench branch only; manual trigger)

- Requires CUDA GPU and a real quantized model file
- Runs in minutes or longer
- Output goes to structured format (JSON) under `dflash/bench/results/` for comparison and archival
- Triggered manually via `ctest -L bench` or through dedicated CI job, not on every commit
- May measure throughput, latency, memory, or correctness under scale
- Examples: `test_dflash.cpp`, `bench_laguna_ttft.cpp`, `test_qwen36_mtp_e2e.sh`

### T4 — Stale / Experimental (Quarantine; do not run)

- Tests that no longer compile, test removed code, or are incomplete exploration spikes
- Must be moved to `dflash/test/quarantine/` and excluded from default CMake target
- Reinstate or delete within one PR cycle; no in-place rot
- Examples from audit: `spike_thin_copy.cpp`

## The CHECK() Macro

Every T1 test must use the `CHECK()` macro. Definition:

```cpp
#define CHECK(cond) do {                                                     \
    if (!(cond)) {                                                           \
        std::fprintf(stderr, "%s:%d CHECK(%s) FAILED\n",                     \
                     __FILE__, __LINE__, #cond);                             \
        std::exit(1);                                                        \
    }                                                                        \
} while (0)
```

**Why:** Bare `assert()` is compiled away when `-DNDEBUG` is set (Release builds). A test with `assert(x == y)` compiled in Release will never fire that assertion; it will silently pass. We discovered this when a test reported `iters=3 accepted=3` but its `assert(iters == 2)` had been compiled out. All T1 tests are "always on" only if they use `CHECK()`.

## Migration Rule for Existing Tests

- **New tests must be T1 unless they require a model file or GPU.** If model or GPU is required, they are T2 or T3.
- **Existing T2/T3 tests with bare `assert()` get migrated to `CHECK()` on next touch.** Do not leave them as-is.
- **T4 tests get quarantined or deleted immediately.** No in-place rot.

## Current Coverage Gaps

### 1. verify_batch / snapshot_kv / restore_kv

**Status:** No T1 coverage. These functions are the heart of every spec-decode path and are only exercised end-to-end through `test_dflash.cpp` (T3, GPU-gated). A bug in KV state management would not be caught until a full GPU run.

**Action:** Add `test_kv_target_contract.cpp` (T1) with a scripted DFlashTarget fake that exercises verify_batch, snapshot_kv, and restore_kv using synthetic KV tensors and deterministic batch data.

### 2. pFlash TQ3_0 decode-only gate (commit 57c46ca)

**Status:** Only exercised end-to-end in bench. The decision to gate pFlash to decode-only when KV quantization is TQ3_0 is not isolated, tested, or code-reviewed at the unit level.

**Action:** Extract the gate decision into a pure function (following `rules/pure-functions-testable.md`) and add a T1 truth-table test covering all quant combinations and prefill/decode modes.

### 3. Cross-tokenizer drafter-to-target ID mapping

**Status:** Not plumbed; `bench_laguna_pflash.cpp` comments it out. Spec-decode requires mapping token IDs between drafter and target tokenizers, but the mapping logic is undefined and unimplemented.

**Action:** Define the mapping interface and add T1 coverage before bench is enabled. Coordinate with cross-tokenizer spec-decode design.

## Naming Convention for Tests

- `test_<unit>.cpp` for T1 unit tests
- `smoke_<arch>_<thing>.cpp` for T2 smoke tests (e.g., `smoke_qwen36_load.cpp`)
- `test_<arch>_<thing>_e2e.{cpp,sh}` for T3 end-to-end tests
- `bench_<arch>_<metric>.cpp` for T3 dedicated benchmark/perf tests

## When to Update This Document

- A new tier definition is needed (e.g., adding T1.5 for slow-but-no-model tests that take >5s)
- A coverage gap closes (remove from the gaps list)
- Audit re-runs and tier counts shift meaningfully
- Test naming convention changes

---

Last reviewed: 2026-05-15 — based on inventory audit of dflash/test/ (25 files: 3 T1, 13 T2, 8 T3, 1 T4).
