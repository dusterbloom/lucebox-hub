# P42a — single-owner ordered device-output join

## Verdict

**HIP BYTE-EXACT GO; INTEGRATED THROUGHPUT NO-GO; BROAD RUN, DEFAULT
ENABLEMENT AND LEGACY DELETION STOPPED.**

P42a removes all 17,917 per-expert P41 host readbacks and reproduces the
frozen calibrated accumulation on GPU1. The model-free join passes on both
Lucebox4 architectures, the generated token sequence and text are unchanged,
and the complete 42-row logits payload is byte-identical to P41 with SHA-256
`cce1bd031e90eb13928ffddfb7e9329d75d55419a8f73b6479a920fe6c561a69`.

The correctness-first implementation is slower. Eight true decode
transitions take 6.874817375 seconds, or **1.163667 transitions/s**, versus
1.275600/s for the P41 compact fact control. This is an 8.77% rate regression
and a 9.62% decode-time increase. The preregistered fact gate therefore stops
before a broad run. P30, P40, P41 and the old host-result path remain intact
and P42 stays disabled by default.

## Frozen accumulation contract

P42 is accepted only for calibrated P41 execution, one token, authoritative
sidecars, all routed layers, and one identical GPU1 provider/destination
backend. Unsupported selections fail at startup or at the device-output API.

The destination kernel assigns one GPU thread to each of 3,584 output
dimensions. Every thread walks the same serial contribution list:

1. calibrated routes are stable-sorted by selected expert ID;
2. each route adds omitted calibration-rank means from rank 0 through 11;
3. the selected compact expert output is added after its omitted means;
4. mean-only routes contribute one native mean;
5. sidecar-authoritative fallbacks are stable-sorted by expert ID and reduced
   into a separate subtotal from positive zero;
6. that fallback subtotal is added to the calibrated destination exactly once.

The kernel uses separately rounded FP32 multiplication and addition. The HIP
source is compiled with `-ffp-contract=off`, CUDA with `--fmad=false`.
Disassembly shows separate `v_mul_f32_e32` and `v_add_f32_e32`, no
`v_fma_f32`/`v_fmac_f32`, no scratch or flat spill operations, and a zero-byte
private segment. The kernel uses 11 VGPRs on both architectures (28 SGPRs on
gfx1151 and 30 on gfx1201).

P41 graph outputs alias reusable compact entries. P42 therefore copies every
device output immediately into a layer-owned arena. The current P42a slice
synchronizes the default stream after each such copy and after the final copy
into the graph-owned join input. Pending outputs are discarded with a
best-effort synchronization on every post-evaluation failure. This is a safe
single-owner boundary, not the final overlap design.

## Model-free qualification

The final HIP binaries and outcomes are:

| Gate | SHA-256 | gfx1201 | gfx1151 |
|---|---|---:|---:|
| Ordered join | `4060dae1537f8e5e29ca4c59214c5840109552a0e924d756f984e6d6896abe5c` | PASS, 0.26 s | PASS, 0.11 s |
| Progressive provider | `145f66a212557b8cbda5a55a20e4bc0e75e85f4adbd543875a556a437274ad30` | PASS, 0.03 s | same host gate |
| MoE stream compute | `181ee39ad5ab129120244464f99cecd7134c9ca75a1bffcf5b334af1e499ab4b` | 2/2, 0.25 s | 2/2, 0.25 s |

The ordered test uses always-on Release checks and covers a mean-only route,
mixed calibrated means/selected output/fallback subtotal, duplicate rows, an
FMA-separating non-unit triple, signed zero, a subnormal, quiet NaN, the full
208-row/operation arena ceiling, and descriptor/output reuse. The adversarial
triple caught an initially contracted HIP implementation before qualification;
the final source-local no-contraction flags pass on both GPUs without the
private-memory spills caused by volatile materialization.

The final runner SHA-256 is
`8759b8c20e8a9176eb48e4731b643afbbd5f25fbbaec3046493a1ec936870ad6`.
The ordered-join HIP object is
`244303ddc272b0512549a2838829703789a9c945a167dac5a96b64179afdf586`.

## Integrated fact gate

The P42 run changes only the ordered-device-join opt-in relative to the P41
policy: GPU1 core/provider, calibrated 96-slab layer table, P27 direct pinned
compact delivery, 8-GiB P30 host cache, P41 compact execution, authoritative
sidecars and P40 disabled.

| Result | P41 control | P42a ordered join | Change |
|---|---:|---:|---:|
| True AR rate (8 transitions) | 1.275600/s | **1.163667/s** | **−8.77%** |
| Decode time | 6.271559 s | 6.874817 s | +9.62% |
| Prefill time | 29.940473 s | 30.016452 s | +0.25% |
| Mean decode expert stage | 469.140 ms | 541.406 ms | +15.40% |
| Mean decode join stage | 25.069 ms | 26.211 ms | +4.56% |
| Expert readback | 0.347291 s | **0 s** | removed |
| Expert graph | 1.770864 s | 2.584766 s | +45.96% |

The P42 stage counters explain the result:

| Counter | Value |
|---|---:|
| Mean rows uploaded | 733,092 |
| Mean H2D | 10,509,606,912 bytes |
| Expert D2D copies | 17,917 / 256,858,112 bytes |
| Ordered join launches | 3,864 |
| Final output D2D copies | 3,864 / 55,394,304 bytes |

The route count and compact executor behavior remain stable: P41 completes
17,917/17,917 layouts, uploads, gate, up, SiTU and sparse-down stages with
zero fallback and zero invalid events. `traffic.tsv` is byte-identical to P41
(`e2eb5f...`). The apparent wall/energy improvement is not a decode speed
claim: this is one short ordered run and storage/cache effects are
order-sensitive. The eight causal transition times reject P42a as enabled.

## Code and deletion boundary

Against the captured pre-P42 Lucebox4 source, raw numstat is 1,176 additions
and 42 deletions: production 829/40, tests 310/1 and CMake 37/1. Tokei's
comment/blank-excluding delta is 741 production code lines, 282 test code
lines and 36 CMake code lines. Comments add 22 lines and blanks add 53 across
the complete slice.

No legacy deletion is earned. The expanded evaluator, P30/P40 controls, P41
host readback wrapper and host accumulation path remain the explicit fallback
and A/B boundary. Their deletion becomes eligible only after an exact device
path wins the short and broad gates.

## Next boundary

P42a proves the ordered device-output seam but also identifies the two costs
that must be removed before heterogeneous ownership:

1. keep or cache the 12×3,584 mean cards on GPU and upload only genuinely
   missing/resident rows instead of moving 10.51 GB per short run;
2. replace per-expert default-stream synchronization with backend-stream
   events and an ordered consumer dependency, preserving the immediate copy
   before compact-entry reuse;
3. batch descriptors/copies where possible while preserving the frozen
   contribution list;
4. repeat the single-owner fact gate; only an exact speed win may advance to
   PR600-style R9700/Strix ownership.

Machine-readable evidence is in
`results/k3_p42a_ordered_device_join_runtime.json`.
