# ABI-stale test binaries — quarantined 2026-05-27

The following test binaries in `<repo-root>/dflash/build/` have been
renamed to `*.abi-stale-do-not-run`. Their `.o` files predate the `DrafterContext`
struct change introduced during the dflash→server rename, so they are linked against
an incompatible `dflash_common.a`. Running them would produce silent wrong results.

Quarantined binaries:
- `test_dflash.abi-stale-do-not-run`
- `bench_laguna_pflash.abi-stale-do-not-run`
- `smoke_qwen3_forward.abi-stale-do-not-run`
- `pflash_daemon.abi-stale-do-not-run`

To restore: run a full CMake clean rebuild.

```bash
cmake -S <repo-root>/server -B <repo-root>/server/build \
  -DGGML_CUDA=ON
cmake --build <repo-root>/server/build -j$(nproc)
```

Note: full CUDA rebuild may OOM on WSL2 with limited RAM. Use `-j2` or `-j1` if needed.

Source: Momus bench-infra audit 2026-05-27, finding #1. See `project_bench_infra_audit_findings.md`.
