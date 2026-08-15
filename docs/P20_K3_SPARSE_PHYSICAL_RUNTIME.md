# P20 Kimi K3 sparse-physical, full-width runtime

**VERDICT: EXPERT DELIVERY GO; WHOLE-SYSTEM DECODE GATE NOT YET MET.**

P20 freezes the all-92 calibrated96 policy and changes only how authoritative
expert bytes reach the unchanged native 3,072-neuron expert arithmetic.  The
sparse path reads compact selected slab records with aligned direct I/O, copies
only those records into their natural positions in a full-width CUDA scratch
view, represents omitted blocks as verified numeric zero, and invokes the same
native gate/up/SiTU/full-down graph.  Exact fallbacks remain on the native MoE
scheduler.  Expert outputs are reduced in deterministic expert-ID order and the
frozen omitted-slab mean tail is added at the same point as the reference.

## A. Reproduction lock

- branch: `experiment/kimi-k3-p20-sparse-physical`
- base commit: `3f52ad4a2e34f6513a108bbd26f80c625da5eb5b`
- implementation commit: `07aa813`
- checkpoint: 14-shard `Kimi-K3-UD-IQ1_S`
- first checkpoint shard SHA-256:
  `5022014e7c49d8844e9f1bc7d9fb824c0d640214540aa845690518d800286083`
- natural-sidecar manifest SHA-256:
  `192efad90c790b8a8230e71bca56c260eee3baf6846108d71c9ec598f6762b7c`
- calibrated runtime manifest SHA-256:
  `ed321b400b99234522583d7ea279cca8ba2b053257daa8dd713137beb7546bc1`
- official-template comparison SHA-256:
  `8c7199e5689a74a7bc973a58a1ffa490653bbf7e851fea995c2003bd18a83b04`
- rendered prompt SHA-256:
  `0b3a1eacd64dc1e40dfec292a1c929d6e2a42a27da33209288e2cfbcdba5ac6a`
- build: CUDA 12.6, capped at `-j4`
- machine: WSL2 5.15, RTX 3090 24 GiB, 27 GiB WSL RAM
- storage: WD SN850X, `/dev/sdd1`, ext4

The locked native and calibrated96 official-template arms generated identical
tokens.  The answer was wrong in both arms; this is a parity result, not a task
quality claim.

## B. Physical data flow

```text
frozen calibrated prefix plan
  -> one layer-wide batch of selected sidecar slab records
  -> aligned O_DIRECT pread into compact host buffers
  -> selected bytes only copied/scattered to native CUDA offsets
  -> verified-zero omitted quant blocks in full-width CUDA scratch
  -> unchanged native 3,072-wide expert graph
  -> deterministic expert-ID reduction
  -> unchanged calibrated mean tail
```

The first implementation created 16 read threads at every layer.  A persistent
16-worker pool reduced measured selected-sidecar time from 1.0223 s to 0.8760 s
over the same one-token trace without changing one logit byte.

## C. Semantic parity

| Gate | Result | Evidence |
| --- | --- | --- |
| raw zero IQ1_S and IQ2_XXS blocks | exact numeric zero | native dequantization unit test |
| full192, all 92 routed layers | byte-identical logits, KL 0 | one-token native vs scratch192 |
| calibrated96 reference implementation vs sparse scratch | byte-identical logits, KL 0 | one-token all-92 control |
| calibrated96 real decode transition | two rows byte-identical, KL 0 | two-token reference vs scratch |
| generated tokens | `11 374` in both arms | prompt `Hi`, text `Hi, I` |

The earlier split-down arithmetic path is not used.  Full-width native down
reduction is preserved.  This proves that sparse delivery preserves the frozen
calibrated96 implementation; it does not prove calibrated96 broadly equals the
native model.

## D. Measured physical and transfer accounting

Two on-policy rows through all 92 routed layers:

| Quantity | Total | Per row | Ratio to logical policy |
| --- | ---: | ---: | ---: |
| logical selected + fallback | 10,710,540,288 B | 4.9875 GiB | 1.0000 |
| selected direct physical | 9,792,651,264 B | 4.5601 GiB | 0.9143 |
| native fallback physical | 934,155,387 B | 0.4350 GiB | 0.0872 |
| total expert physical | 10,726,806,651 B | 4.9951 GiB | **1.0015** |
| selected authoritative H2D | 9,738,387,456 B | 4.5348 GiB | 0.9092 |
| native fallback H2D upper bound | 933,081,645 B | 0.4345 GiB | 0.0871 |
| metadata H2D | 48,984,064 B | 0.0228 GiB | 0.0046 |
| total H2D upper bound | 10,720,453,165 B | 4.9921 GiB | **1.0009** |

Selected sidecar physical/logical alone is 1.0056.  Both the physical-I/O and
H2D gates of 1.10 pass.  No full-width host reconstruction or full-width host
DMA occurs for a partial calibrated expert.

## E. Backend measurements

| backend | physical/logical | storage GiB/s | H2D/logical | elapsed / decode | status |
| --- | ---: | ---: | ---: | ---: | --- |
| reference buffered/full H2D | process counter 9.93x | confounded | 1.45x selected path | 29.056 s decode transition | baseline |
| scratch + per-layer O_DIRECT threads | 1.0015 expert total | 4.46 selected | 1.0009 | 30.02 s one row | superseded |
| scratch + persistent layer-batch O_DIRECT | **1.0015** | **5.34 selected / 5.26 combined** | **1.0009** | **26.976 s decode transition** | best |
| CUDA VMM on RTX 3090/WSL | not run hot | n/a | n/a | n/a | one-map-per-slab killed by 2 MiB granularity |
| cuFile direct on RTX 3090/WSL | unavailable | n/a | n/a | n/a | compatibility mode only |

The two-token control reduced the latency of the single measured decode
transition by 7.2% (29.056 s to 26.976 s) and total elapsed by 5.6%, while
remaining byte-exact.
This is not a steady-state benchmark; a longer trace remains open.

## F. Roofline and binding bottleneck

- logical expert traffic: 4.9875 GiB/row
- measured combined expert storage bandwidth: 5.2573 GiB/s
- expert-only storage roofline: 1.0541 token/s
- actual measured transition rate: 0.0371 token/s
- achieved / expert-only roofline: 3.52%
- measured whole-process storage: about 49.52 GiB/row

The smoke runner prints 0.07 token/s because it divides both emitted tokens by
the timer for the single autoregressive transition after prefill.  That display
is not used as the measured transition rate or in the roofline comparison.

The expert path is no longer the binding storage path in this WSL
configuration.  The local run maps 45.34 GiB of non-routed tensor payload while
WSL exposes only 27 GiB RAM; broader deployment accounting is approximately
57.94 GiB and must be used for capacity planning.  The two-row run incurred
95.14 GB of inferred mapped-core or otherwise untracked reads.  Its near-one
mapped-payload pass per row strongly supports refaulting, but causality remains
qualified because the residual is inferred by subtraction.  More
provider-side concurrency cannot make the 70% expert-roofline gate meaningful
until the core is resident or placed differently.

### Page-retention control on the same machine

The registered cold diagnostic set `DFLASH_KIMI_MMAP_DROP_PAGES=1`, which
advises the kernel to discard every mapped shard after each forward.  A paired
two-row control disabled that advice without changing any model, provider,
cache, or expert-delivery setting.  Logits remained byte-identical (SHA-256
`7be0aa8a…`), but retaining pages did not improve the 27-GiB configuration:

| measurement | drop pages | retain pages | change |
| --- | ---: | ---: | ---: |
| prefill row | 26.842 s | 26.851 s | +0.03% |
| one decode transition | 26.976 s | 28.256 s | +4.74% |
| total elapsed | 56.168 s | 57.222 s | +1.88% |
| block-device reads | 106.343 GB | 106.468 GB | +0.12% |

This is a **MEASURED NO-GAIN** for ordinary page retention under the current
memory ceiling.  It is consistent with a cyclic scan whose 45.34-GiB mapped
working set exceeds available RAM: retaining the tail of one pass does not
create useful reuse when the next pass begins at the start.  It strengthens
the case for either more memory, explicit partial residency, or an asynchronous
core-streaming path; it does not show that page retention is useless once the
core fits.

## G. VMM, GDS, and cache decisions

CUDA VMM is supported on the measured RTX 3090/WSL system, but its minimum
mapping granularity is 2 MiB.  P20 slab
records are 537,600--652,288 bytes, producing 221%--290% padding, so VMM fails
the preregistered 10% gate for one mapping per slab.  This does not reject a
future grouped/superpage layout on another platform.

cuFile reports compatibility mode and no usable WSL PCI topology; true direct
GPU storage is unavailable on this measured system.  The honest backend here
is O_DIRECT host staging plus selected-only H2D; this is not an AMD-platform
finding.

On the two-row trace, even a perfect unlimited selected-slab cache can remove at
most 631,314,432 repeated bytes, 6.48% of selected traffic.  That upper bound is
below the 30% cache proceed gate, so no cache runtime was implemented.  A
longer trace can revisit this result, but not override the current measurement.

## H. Verdict

- **MEASURED:** full192 equals native, and sparse calibrated96 equals its
  calibrated96 reference implementation.
- **MEASURED:** expert physical/logical 1.0015 and H2D/logical 1.0009 pass.
- **MEASURED:** storage throughput gate passes at 5.26 GiB/s combined.
- **MEASURED:** one decode transition latency improves 7.2% but reaches only
  3.52% of the expert-only roofline.
- **FAILED:** one-map-per-slab CUDA VMM and true cuFile direct storage on the
  measured RTX 3090/WSL platform.
- **OPEN:** steady-state easy/hard generation and the frozen 12-prompt suite.
- **NEXT:** expose enough WSL RAM to keep the 45.34-GiB core resident, then rerun
  the identical two-token trace before optimizing graph reuse or overlap.
