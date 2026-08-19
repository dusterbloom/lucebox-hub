# P38 — Kimi K3 sparse physical path on HIP

## Verdict

**MEASURED HIP EXECUTION AND SAME-BACKEND REPEATABILITY GO; CUDA/HIP BYTE
PARITY NO-GO; SHORT-PROMPT PERFORMANCE MEASURED.**

The P20/P23/P25/P26/P27 sparse physical expert path now builds for HIP. The
change compiles the existing Kimi slab-scatter kernel as HIP and enables its
existing pinned compact staging and persistent sparse workspace on HIP. It
does not change routing, calibration, slab selection, mean tails, native
3,072-neuron expert arithmetic, or accumulation order.

## Reproduction

- Source branch: `experiment/kimi-k3-p20-sparse-physical`
- Source commit: `3f373b072e70791ef235df5002b0a9b130f470f1`
- Host: `lucebox4`
- CPU: AMD Ryzen AI Max+ 395, 16 cores / 32 threads
- Accelerator: AMD `gfx1201`, 34,208,743,424 bytes visible VRAM
- ROCm: 7.2.2
- Build directory: `server/build-k3-hip-gfx1201`
- Runner SHA-256:
  `112a79277c78db7a1fe62800378d6dd23a6cd8b862532070354ba682cb034e24`
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
| Sparse-core GGUF copied-range verification | PASS at transfer, 14/14 shards; durable remote rehash artifact still required |
| Sparse-core plus sidecar, CUDA end-to-end | PASS, 6,881,280/6,881,280 payload logits byte-identical |
| Full192 HIP versus native byte identity | OPEN — model package not present |
| Calibrated sparse HIP execution | PASS, two complete frozen fact runs |
| HIP repeatability | PASS, logits byte-identical across 2/2 runs |
| HIP versus frozen CUDA byte identity | NO-GO under the deployed native CPU/GPU stacks |
| HIP behavioral parity | PASS, identical 9 emitted tokens and text |
| HIP short-prompt throughput | MEASURED, 0.6025 and 0.6119 true AR transitions/s |

The machine-readable result contract is
`results/k3_p38_lucebox4_hip_baseline.json`.

## Lucebox4 HIP baseline

Both runs use the same 34-token frozen fact prompt and emit nine tokens. The
first emitted token is selected from the final prefill logit, so the decode
window contains eight causal target transitions, not nine. Throughput claims
must report `8 / decode_seconds`; `9 / decode_seconds` is retained only as a
legacy emitted-token rate.

The deployed configuration keeps the main core on the 12-thread CPU, offloads
latent/shared MoE preparation plus the frozen 26-layer complete-preparation
set to `hip:0`, uses a 2 GiB device cache, and enables the P20/P23/P25/P26/P27
direct-pinned compact delivery ladder. All 92 calibrated layers load and exact
fallback is sidecar-authoritative. The complete list is embedded in the result
contract because the suite manifest does not yet serialize these variables.

| Run | Prefill | Decode | True AR rate | Legacy emitted rate |
|---|---:|---:|---:|---:|
| first | 56.519125673 s | 13.278583947 s | 0.602473881/s | 0.677783116/s |
| repeat | 55.844740570 s | 13.075050721 s | 0.611852311/s | 0.688333850/s |

This one short prompt proves deployment and repeatability. It is not a steady
64/128-token throughput claim or a broad quality result.

The two HIP traces have identical SHA-256
`31c42a5a297db20330defc1200f50777e1ef916d201057a79d9fcbb0e5b0b049`.
Against the frozen CUDA trace
`82dfa599691d1e89215f5a9603a0134380ae7467945e10bded30f243adde5085`,
all nine generation decisions and the emitted text agree, while full-logit
byte identity fails. Across 42 rows by 163,840 vocabulary entries, maximum
absolute logit error is 3.55032325, mean KL is 0.04134504, and top-1 agrees on
39/42 rows. The comparison also changes the native CPU ISA and therefore does
not attribute the numerical delta solely to HIP.

The logits trace contains 6,881,280 float payload values plus a 48-byte header.
The previous count of 6,881,292 incorrectly divided the complete file size by
four and treated the header as twelve logits.

## HIP stage account

The cumulative provider counters cover all 42 model positions in each run:

| Stage | First | Repeat |
|---|---:|---:|
| explicit provider reads | 66,937,405,440 B | 66,937,405,440 B |
| direct physical bytes | 56,431,411,200 B | 56,431,411,200 B |
| direct I/O | 11.849058409 s | 11.567150491 s |
| compact pack | 0.144155599 s | 0.150275254 s |
| compact upload/scatter | 33.029860900 s | 33.021306103 s |
| expert graph | 1.496866044 s | 1.486925587 s |
| expert readback | 1.038443171 s | 1.063103293 s |

HIP scatter/upload is therefore the largest measured provider stage and the
first performance target. These timers are cumulative, not mutually exclusive
end-to-end wall-time buckets. The identical traffic traces have SHA-256
`ebc46f90d156d8e6d16c14148bcb34cb3395daf880a3b7c606e29036a5c27e93`,
366,422 request rows, and 1.484293256 logical GiB per model position. No general
stage profile or AMD power/VRAM telemetry was enabled, so these runs cannot
support a full-stage or energy claim.

## Successor promotion contract

Performance work starts from commit `3f373b0` and must preserve this fact gate,
then pass the frozen 12-prompt suite SHA-256
`6d1c0583df52738820559bef66f6a96839bcde44c0bae7bdc4bb7bbe7332d4cc`
with thinking disabled, `n_gen=24`, resume disabled, and the same calibrated
policy. `DFLASH_KIMI_STAGE_PROFILE=1` is mandatory.

The canonical decode rate is:

```text
sum(max(0, emitted_tokens - 1)) / sum(decode_seconds)
```

Aggregate stage totals and median per-transition diagnostics are reported
separately. The quality gate is 12/12 nondegenerate prompt success, including
the exact LIME-742 and QUARTZ-918 checks. Token identity, KL and top-1 agreement
remain reported numerical deltas; they are not silently promoted to pass/fail
thresholds after a run.

The 10/s target means true causal transitions, without the prefill-selected
token and before speculative accepted-token multiplication. Each milestone
must report raw AR, emitted-token rate and speculative acceptance separately.

Production LOC is also a gate. The baseline active Kimi runtime/build delta is
9,088 non-test code lines and the target is at most 6,500. Every milestone
reports added, deleted, churn and net code separately for implementation,
tests, and docs/evidence. Tests, comments, generated files, dependencies,
build products and binary artifacts do not count toward the production-code
target. Moving code between namespaces is not a reduction.

The successor telemetry wrapper auto-detects `nvidia-smi` or `rocm-smi`; on an
AMD APU it records VRAM plus GTT usage, utilization, package power and sampled
energy. The frozen P38 runs predate that change, so their GPU power, memory and
energy fields remain unavailable; null fields must not be interpreted as zero.

## Sidecar-authoritative exact fallback

`DFLASH_KIMI_SIDECAR_AUTHORITATIVE=1` makes the calibrated provider the sole
authoritative source for routed expert weights. It requires all 92 calibrated
layers at startup and fails closed on an invalid layer, geometry mismatch, or
unreadable artifact. A low-coverage exact fallback reads all twelve natural
slabs for that expert and invokes the unchanged native full-width 3,072-neuron
expert graph. Default behavior is unchanged when the variable is absent.

The first frozen CUDA control exercised exact fallback routes at layers 2, 12,
and 92. Its output tokens, text, and 6,881,280 final-logit payload floats were
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

## Remaining gates

1. Persist an independently reproducible copied-range rehash artifact on
   Lucebox4 instead of relying only on the transfer-time 14/14 result.
2. Run a 64- or 128-token steady decode and the broad quality suite.
3. Separate CPU-ISA and GPU-backend numerical effects with a common-ISA trace.
4. Reduce or bypass compact scatter, then rerun the same byte, stage and quality
   gates before claiming a speedup.

## Lucebox4 capacity-safe materialization

The integrated upstream-main revision is `43898ea859ad2a3cc29cad4d3dd0a4d2614aa09b`.
It builds on Lucebox4 with ROCm 7.2.2 / `gfx1201` at `-j4`. The model-free Kimi
provider, MoE stream, scheduler, package, and feature gates pass. The broad
DeepSeek unit test has an upstream-baseline ROCm failure in the dual-device HC
scratch test: untouched upstream `7bea91924969f697a3b28e8c19ce67b89a255f46`
segfaults at the same call. This is registered as an upstream hardware/test
issue, not hidden as a Kimi merge pass.

Lucebox4 has 650,358,591,488 free bytes. The final P38 package consumes
599,884,431,360 allocated bytes, so directly staging the 594 GB public teacher
and the 545 GB sidecar bank at once is impossible. The local SSH uplink also
measured only about 1.5--2.9 MB/s, while a bounded Lucebox4 Hugging Face range
read measured 26,553,223 B/s.

`scripts/materialize_kimi_k3_slab_bank.py` implements the bounded path:

1. download one public Unsloth expert-containing shard with resumable `curl`;
2. pack every newly available natural-order layer sidecar;
3. compare its complete SHA-256 and byte count with the registered local
   reference manifest;
4. only then hole-punch those routed tensor ranges;
5. remove a temporary public shard after all layers that depend on it have
   verified sidecars.

The script is marker-bound to a dedicated deployment root, refuses foreign
nonempty roots, retains receipts, enforces a 32 GiB free-space reserve, and is
safe to resume. It never reads, edits, or removes DeepSeek model data. The
40.34 GB P32 sparse core and 14.19 GB calibrated auxiliary bank transfer
separately with sparse-preserving `rsync`.

```sh
PYTHONPATH=server/deps/llama.cpp/gguf-py \
python3 scripts/materialize_kimi_k3_slab_bank.py \
  /home/duster/kimi-k3-deploy/streamed-bank \
  /home/duster/kimi-k3-deploy/reference-manifests \
  --download-base-url \
  https://huggingface.co/unsloth/Kimi-K3-GGUF/resolve/main/UD-IQ1_S
```

The downloaded public source shards are temporary and intentionally retired;
the transferred P32 sparse core remains the authoritative non-routed model.
