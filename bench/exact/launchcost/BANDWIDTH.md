# Achievable GPU read bandwidth, lucebox4 (gfx1151 Strix Halo APU)

`rocm_bandwidth_test` / `rocm-bandwidth-test` is not installed on lucebox4 (checked `which` and
`find /opt/rocm -iname '*bandwidth*'` — nothing). Measured instead with a purpose-built HIP
kernel: `bandwidth_test.hip`, grid-stride 128-bit (`float4`) reads over a 2 GB device buffer,
summed into one atomic per thread-0 (compiler can't elide the loads since the sum feeds a
side-effecting atomic). 4096 blocks x 256 threads, grid-stride loop covers the full buffer.
3 warmup launches (untimed), then 20 timed launches, each individually wall-clocked with
`hipStreamSynchronize` around it; median/mean/best computed from the 20.

Build: `hipcc -O3 --offload-arch=gfx1151 -o bandwidth_test bandwidth_test.hip`
Run: `HIP_VISIBLE_DEVICES=1 TMPDIR=/home/duster/qwen4exp-f16/scratch ~/qwen4exp-f16/gpu_exec.sh ~/qwen4exp-launchcost/bandwidth_test 20`

## Result

```
GPU_EXEC_START avail_kb=63091332 gtt=18636800
bytes=2147483648 (2.147 GB) runs=20  us: median=9757.4 mean=9993.0 best=9702.3  GB/s: median=220.09 mean=214.90 best=221.34
GPU_EXEC_DONE rc=0
```

| metric | value |
|---|---|
| median GB/s | **220.09** |
| mean GB/s | 214.90 |
| best GB/s | 221.34 |

## Interpretation

Measured median **220 GB/s** read bandwidth, in line with the public Strix Halo range (~212-215
GB/s) the coordinator flagged, and well below the **242 GB/s** figure this project's earlier
bandwidth-floor estimates (e.g. `HC-AUDIT.md`) assumed. Using 242 GB/s as the denominator
understates how memory-bound the hc_* kernels actually are: any "measured/floor ratio" computed
against 242 GB/s should be rescaled by `242/220 ≈ 1.10x` — the real bandwidth floor for those
kernels is about 10% higher (in ms, i.e. worse) than previously stated, meaning the hc_* kernels
are closer to bandwidth-bound than `HC-AUDIT.md`'s ratios implied, with correspondingly less
"free" headroom to recover purely from a bandwidth argument. This does not change the per-dispatch
launch-overhead investigation (that's a floor-independent measurement), but it does mean prior
"X.Yx above floor" ratios in `HC-AUDIT.md`/`KNOCKOUT-HC.md` are mild overstatements and should be
corrected by ~10% if anyone treats them as a hard ceiling.
