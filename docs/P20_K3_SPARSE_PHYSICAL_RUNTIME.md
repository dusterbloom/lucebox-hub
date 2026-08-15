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
| calibrated96 reference vs scratch | byte-identical logits, KL 0 | one-token all-92 control |
| calibrated96 real decode transition | two rows byte-identical, KL 0 | two-token reference vs scratch |
| generated tokens | `11 374` in both arms | prompt `Hi`, text `Hi, I` |

The earlier split-down arithmetic path is not used.  Full-width native down
reduction is preserved.

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
| CUDA VMM | not run hot | n/a | n/a | n/a | killed by 2 MiB granularity |
| cuFile direct | unavailable | n/a | n/a | n/a | WSL compatibility mode only |

The two-token control improved the single measured decode transition by 7.2%
(29.056 s to 26.976 s) and total elapsed by 5.6%, while remaining byte-exact.
This is not a steady-state benchmark; a longer trace remains open.

## F. Roofline and binding bottleneck

- logical expert traffic: 4.9875 GiB/row
- measured combined expert storage bandwidth: 5.2573 GiB/s
- expert-only storage roofline: 1.0541 token/s
- runner-reported short decode rate: 0.07 token/s
- achieved / expert-only roofline: 7.0%
- measured whole-process storage: about 49.52 GiB/row

The expert path is no longer the binding storage path in this WSL
configuration.  K3's non-routed core is mapped at 45.34 GiB while WSL exposes
only 27 GiB RAM.  The two-row run incurred 95.14 GB of inferred mapped-core or
otherwise untracked reads.  The core is therefore refaulted on every pass and
dominates the roughly 27-second step.  More provider-side concurrency cannot
make the 70% expert-roofline gate meaningful until the core is resident or
placed differently.

## G. VMM, GDS, and cache decisions

CUDA VMM is supported, but its minimum mapping granularity is 2 MiB.  P20 slab
records are 537,600--652,288 bytes, producing 221%--290% padding, so VMM fails
the preregistered 10% gate.

cuFile reports compatibility mode and no usable WSL PCI topology; true direct
GPU storage is unavailable.  The honest backend is O_DIRECT host staging plus
selected-only H2D.

On the two-row trace, even a perfect unlimited selected-slab cache can remove at
most 631,314,432 repeated bytes, 6.48% of selected traffic.  That upper bound is
below the 30% cache proceed gate, so no cache runtime was implemented.  A
longer trace can revisit this result, but not override the current measurement.

## H. Verdict

- **MEASURED:** full192 and calibrated96 semantic parity pass.
- **MEASURED:** expert physical/logical 1.0015 and H2D/logical 1.0009 pass.
- **MEASURED:** storage throughput gate passes at 5.26 GiB/s combined.
- **MEASURED:** one decode transition improves 7.2% but reaches only 7% of the
  expert-only roofline.
- **FAILED:** CUDA VMM and true cuFile direct storage on this platform.
- **OPEN:** steady-state easy/hard generation and the frozen 12-prompt suite.
- **NEXT:** expose enough WSL RAM to keep the 45.34-GiB core resident, then rerun
  the identical two-token trace before optimizing graph reuse or overlap.
