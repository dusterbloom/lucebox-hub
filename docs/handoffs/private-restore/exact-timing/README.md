# Exact 38.41 reconstruction — first timing (2026-10-08)

Source: 1114/1115 sha256 match to `shared-logit-combine-exact-v4-model-build.json` (only `qwen4exp_graph.cpp.orig` differs; not compiled).
Box tree: `~/qwen4exp-private-exact/{src,build1151}`, cmake Release, `GGML_HIP_GRAPHS=ON`, gfx1151. Driver built by `build_driver.sh`
from the original `driver_shared_epilogue.cpp` with the original compile/link flags. Run with `run.sh` (original gate env, `prompt.ids 256 8`,
`MEASURE_FOLLOW=follow.ids`, inputs sha-match the gate).

Startup parity: router shadows 48/48 (120 MiB), no managed-memory line, kv f16 192 MiB @ ctx 8192.

| run | mode=1 median ms/token | mode=0 median |
|---|---:|---:|
| original 2026-10-06 (native.log) | 38.41 | 38.47 |
| exact rebuild, 2026-10-08 rep1 | 40.05 | 40.11 |
| control wire3b, recorded earlier | 42.53 | |
| control wire3b, same session today | 43.88 | |

The rebuild is 4.3% slower than the original; the unchanged wire3b control is 3.2% slower than its own record in the same session,
so most of the gap is box state, not source. The residual (~1%) may come from the build (the original linked older objects/libs from
`qwen4exp-conc/build1151`). Box today: `power_dpm_force_performance_level=auto`, `platform_profile=balanced`.
