# P38 — Kimi K3 sparse physical path on HIP

## Verdict

**MEASURED CUDA PACKAGING/PARITY GO; HIP RUNTIME PARITY AND PERFORMANCE OPEN.**

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
| Sidecar-authoritative exact fallback, full GGUF | PASS, frozen fact prompt byte-identical |
| Sparse-core GGUF copied-range verification | PASS, 14/14 shards |
| Sparse-core plus sidecar, CUDA end-to-end | PASS, 6,881,292/6,881,292 logits byte-identical |
| Full192 HIP versus native byte identity | OPEN — model package not present |
| Calibrated P37 HIP versus frozen reference | OPEN — model package not present |
| HIP throughput | OPEN |

No model inference was launched during this gate because an existing protected
accelerator workload was active. No files or processes belonging to that
workload were modified.

## Sidecar-authoritative exact fallback

`DFLASH_KIMI_SIDECAR_AUTHORITATIVE=1` makes the calibrated provider the sole
authoritative source for routed expert weights. It requires all 92 calibrated
layers at startup and fails closed on an invalid layer, geometry mismatch, or
unreadable artifact. A low-coverage exact fallback reads all twelve natural
slabs for that expert and invokes the unchanged native full-width 3,072-neuron
expert graph. Default behavior is unchanged when the variable is absent.

The first frozen CUDA control exercised exact fallback routes at layers 2, 12,
and 92. Its output tokens, text, and 6,881,292 final-logit floats were
byte-identical to the archived P37 reference (SHA-256
`82dfa599691d1e89215f5a9603a0134380ae7467945e10bded30f243adde5085`).

## Self-contained sparse core

`scripts/pack_kimi_k3_slim_core.py` copies GGUF metadata and every non-routed
tensor while leaving the 276 routed gate/up/down payload ranges as filesystem
holes. The file offsets and logical sizes therefore remain compatible with the
existing loader, while allocated storage contains only the core. The packer
refuses an existing destination and verifies SHA-256 over every copied byte
range.

Measured P37 package geometry:

| Item | Bytes | GiB |
|---|---:|---:|
| Sparse core, physically allocated | 40,341,995,520 | 37.57 |
| Routed payload represented by holes | 545,349,697,536 | 507.90 |
| Natural slab bank | 545,352,335,360 | 507.90 |
| Calibrated auxiliary bank | about 14.19 billion | about 13.21 |
| Deployable core + slabs + auxiliary data | about 599.9 billion | about 558.7 |

The sparse-core fact-prompt run loaded the normal 14-shard model, served routed
weights and exact fallbacks from sidecars, and reproduced the archived full
model byte-for-byte: identical nine generated token IDs and identical logits
SHA-256. This validates the packaging mechanism on CUDA. Sparse holes must be
preserved during transfer (`tar --sparse` or an independently verified sparse
copy); plain copies that materialize holes can consume the full apparent size.

## Remaining gate

Transfer the self-contained package to Lucebox, verify the copied ranges and
sparse allocation there, then run the same frozen fact prompt through HIP.
Only after byte identity passes should throughput and cache measurements begin.
