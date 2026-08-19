# P39 — Lucebox4 GPU1 core pilot

## Verdict

**MEASURED SHORT-PROMPT PERFORMANCE GO; BEHAVIORAL PARITY GO; BROAD QUALITY
AND STEADY DECODE OPEN.**

The 96 GiB `gfx1151` can hold Kimi's 37.56 GiB non-routed core and execute the
calibrated sparse provider. A build containing both Lucebox4 architectures is
required: the frozen P38 `gfx1201`-only binary exits 139 when asked to execute
on GPU1, while the otherwise identical dual-architecture binary completes.

This pilot changes placement and build architecture, not model weights,
calibration, route policy, selected slabs or fallback semantics.

## Build

```sh
cmake -S server -B server/build-k3-hip-dual -G Ninja \
  -DCMAKE_BUILD_TYPE=Release \
  -DDFLASH27B_GPU_BACKEND=hip \
  -DDFLASH27B_HIP_ARCHITECTURES='gfx1201;gfx1151' \
  -DDFLASH27B_FA_ALL_QUANTS=OFF

cmake --build server/build-k3-hip-dual -j4 \
  --target test_kimi_k3_progressive_provider \
           smoke_kimi_k3_forward run_kimi_k3_h16_suite
```

The model-free provider test passes. The runner SHA-256 is
`6c4f3e7af690c52d9ad5630ee00f56f238263d172c284844fbdcd3570e25e681`.

## Results

The frozen 34-token fact fixture emits the same nine token IDs and Tokyo text
as P38. Canonical decode uses eight causal transitions.

| Run | Prefill | Decode | True AR rate |
|---|---:|---:|---:|
| first | 29.798073125 s | 6.909151009 s | 1.157884665/s |
| repeat | 30.227512344 s | 6.991637407 s | 1.144224097/s |
| combined | — | 13.900788416 s | 1.151013851/s |
| 8 GiB host cache | 27.935616430 s | 5.743965169 s | 1.392766106/s |

The combined rate is 1.896x the two-run P38 CPU-core aggregate. This remains a
short-prompt pilot, not a broad or steady throughput claim.

Across 42 model positions, the first/repeat provider counters are:

| Stage | First | Repeat |
|---|---:|---:|
| direct I/O | 12.514005991 s | 12.883083138 s |
| compact pack | 0.144406254 s | 0.140330465 s |
| compact scatter | 5.277163144 s | 6.225860996 s |
| expert graph | 1.895863480 s | 1.894210717 s |
| expert readback | 0.326412492 s | 0.328232916 s |

Compared with P38 on `gfx1201`, compact scatter is 5.74x faster while I/O is
slightly slower. Core placement and scatter therefore produce the observed
end-to-end gain; storage remains the largest provider stage.

The repeat logits SHA-256 is
`cce1bd031e90eb13928ffddfb7e9329d75d55419a8f73b6479a920fe6c561a69`.
Against P38 HIP CPU-core logits, all generation tokens agree, full-logit byte
identity fails, maximum absolute difference is 3.80177855, mean KL is
0.03901036, and top-1 agrees on 38/42 rows. Placement changes the arithmetic
backend, so broad quality—not cross-placement hashing—is the promotion gate.

### P30 host-cache control

An otherwise identical 8 GiB P30 run is 1.210x faster than the two-run GPU1
aggregate. It records 90,539 hits and 63,288 misses, cuts direct physical bytes
from 56,288,280,576 to 27,260,755,968, and reduces direct-I/O time 23.3%.
Compact scatter remains 5.840529321 s, so a larger isolated host cache cannot
reach the target by itself. The useful next design is one cache plan whose
device-resident hits bypass host copies, zeroing and scatter while P30 remains
the finer-grained lower tier.

The cached logits have the same
`cce1bd031e90eb13928ffddfb7e9329d75d55419a8f73b6479a920fe6c561a69`
SHA-256 as the uncached GPU1 repeat. This proves same-placement cache
transparency for the frozen fact gate.

### AMD telemetry preflight

The successor telemetry wrapper now auto-detects `rocm-smi` and adds VRAM plus
GTT usage for the APU. A five-sample GPU1 idle smoke records the ROCm backend,
165.50 MiB peak combined memory, 11.037 W peak package power and 11.96 J sampled
energy while `/bin/sleep 1` runs. This validates collection only; P39 model
power, memory and energy remain open until the measured command is rerun under
the wrapper.

## Next gates

1. Run the frozen 12-prompt quality/stage contract before calling GPU1 the
   production default.
2. Overlay common device-resident variants on the host cache and measure the
   scatter/H2D reduction. The P40 trace gate shows an 8 GiB expanded LRU cannot
   replace P30's byte coverage.
3. Make calibrated sparse delivery consume stable dual-owner planning; it
   currently bypasses `MoeStreamDualOwnerExecutor`.
4. Remove the routed-output host bounce only after device-join parity passes.

Machine-readable evidence is in
`results/k3_p39_lucebox4_gpu1_core_pilot.json`; the cache-capacity simulation
is in `results/k3_p40_device_variant_cache_sim.json`.
