# P38 — Kimi K3 sparse physical path on HIP

## Verdict

**MEASURED BUILD GO; RUNTIME PARITY AND PERFORMANCE OPEN.**

The P20/P23/P25/P26/P27 sparse physical expert path now builds for HIP. The
change compiles the existing Kimi slab-scatter kernel as HIP and enables its
existing pinned compact staging and persistent sparse workspace on HIP. It
does not change routing, calibration, slab selection, mean tails, native
3,072-neuron expert arithmetic, or accumulation order.

## Reproduction

- Source branch: `experiment/kimi-k3-p20-sparse-physical`
- Base commit: `03b2e0fd5d529f7f96f236df99e811556e870e8d`
- Host: `lucebox4`
- CPU: AMD Ryzen AI Max+ 395, 16 cores / 32 threads
- Accelerator: AMD `gfx1201`, 34,208,743,424 bytes visible VRAM
- ROCm: 7.2.2
- Build directory: `server/build-k3-hip-gfx1201`
- Build cap: `-j4`

```sh
cmake -S server -B server/build-k3-hip-gfx1201 -G Ninja \
  -DCMAKE_BUILD_TYPE=Release \
  -DDFLASH27B_GPU_BACKEND=hip \
  -DDFLASH27B_HIP_ARCHITECTURES=gfx1201 \
  -DDFLASH27B_FA_ALL_QUANTS=OFF

cmake --build server/build-k3-hip-gfx1201 -j4 \
  --target test_kimi_k3_progressive_provider smoke_kimi_k3_forward

./server/build-k3-hip-gfx1201/test_kimi_k3_progressive_provider
```

## Results

| Gate | Result |
|---|---|
| Baseline HIP K3 target build | PASS |
| HIP sparse-scatter compilation/link | PASS |
| Model-free calibrated-provider regression | PASS, exit 0 |
| CUDA rebuild after shared-source change | PASS |
| CUDA model-free provider regression | PASS, exit 0 |
| Full192 HIP versus native byte identity | OPEN — model package not present |
| Calibrated P37 HIP versus frozen reference | OPEN — model package not present |
| HIP throughput | OPEN |

No model inference was launched during this gate because an existing protected
accelerator workload was active. No files or processes belonging to that
workload were modified.

## Deployment blocker discovered

The current calibrated provider reads selected prefixes from the slab bank but
still serves low-coverage exact fallback routes from the original GGUF expert
bank. The Lucebox has enough free storage for the projected core + slab + aux
package, but not for both that package and a duplicate full teacher bank.
Before transfer, exact fallbacks must be recomposed from all twelve lossless
sidecar slabs through the unchanged native full-width expert graph. A strict
sidecar-authoritative mode must fail closed if any layer or provenance check
is invalid.
