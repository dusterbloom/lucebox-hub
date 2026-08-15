# P20 Kimi K3 I/O amplification audit

**STATUS: COMPLETE FOR THE TWO-ROW PRODUCTION TRACE.**

The frozen all-layer calibrated96 behavioral smoke and policy lock completed
before P20.  Broad quality remains open.  This audit changes data movement
only.

## What the old 8.69x number meant

The original smoke reported 81.32 GiB logical provider traffic and 707.04 GiB
process disk reads: 8.69x.  The later official-template run reported
375,926,734,848 logical bytes and 3,140,533,301,248 process disk-read bytes:
8.35x.  Those process counters include page faults from the 45.34-GiB mapped
non-routed core.  They were never a sidecar-only physical-I/O measurement.

The first provider range trace showed:

- selected sidecar logical: 4,869,193,728 B;
- selected sidecar explicit: 4,869,193,728 B (1.0000x);
- exact fallback logical: 399,028,224 B;
- auxiliary mean reads: 242,737,152 B;
- whole-process disk reads: 54,142,902,272 B;
- inferred mapped-core/untracked residual: about 49.03 GB.

Thus the selected sidecar was already logically honest.  The large number came
from the core not fitting in WSL RAM, plus the reference provider's full-width
host-to-device reconstruction.

## Reference data flow and amplification boundaries

| Boundary | Reference behavior | Required? | P20 disposition |
| --- | --- | --- | --- |
| prefix plan -> sidecar | selected slab records | yes | unchanged |
| sidecar -> host | blocking buffered component reads | payload yes; schedule no | layer-batched aligned O_DIRECT |
| auxiliary -> host | repeated FP32 mean reads | values yes | unchanged, separately counted |
| host expert | reconstruct and zero full gate/up/down | no | removed |
| host -> GPU | copy complete reconstructed expert | no | selected authoritative bytes only |
| GPU scratch | full native quant tensors | arithmetic contract | zero + sparse selected fill |
| native graph | 3,072-wide gate/up/SiTU/down | yes | unchanged |
| fallback | native expert scheduler | yes | unchanged |
| reduction/tail | stable expert-ID order + mean tail | yes | unchanged |
| mapped core | page-faulted 45.34-GiB non-routed core | platform limitation | OPEN |

## Two-row attribution table

| Category | Bytes | Evidence | Status |
| --- | ---: | --- | --- |
| selected slab logical | 9,738,387,456 | provider policy trace | MEASURED |
| selected aligned O_DIRECT physical | 9,792,651,264 | provider direct counter | MEASURED |
| exact fallback logical | 972,152,832 | policy trace | MEASURED |
| native fallback physical | 934,155,387 | native scheduler | MEASURED |
| auxiliary mean reads | 481,345,536 | provider trace | MEASURED |
| duplicate selected ranges | 631,314,432 | exact `(file,offset,length)` trace grouping | MEASURED |
| whole-process disk reads | 106,348,482,560 | `/proc/self/io` | MEASURED |
| mapped-core/untracked residual | 95,140,330,373 | bounded subtraction | INFERRED |
| selected authoritative H2D | 9,738,387,456 | runtime CUDA counter | MEASURED |
| metadata H2D | 48,984,064 | runtime CUDA counter | MEASURED |
| reference full reconstructed H2D | 15,536,209,920 | matched two-row reference counter | MEASURED |

Named categories bound 100% of process reads, but the mapped-core row is an
inference by subtraction.  It is corroborated by 852,949 major faults and the
fact that 45.34 GiB of mapped core cannot remain resident in 27 GiB WSL RAM.

## Physical conclusion

For the actual expert path, including exact fallbacks:

```text
physical expert bytes / logical requested expert bytes = 1.0015
```

For the whole process:

```text
process disk bytes / logical requested expert bytes = 9.9293
```

The first ratio is the P20 storage result.  The second is the machine-level
placement result.  Conflating them would incorrectly blame progressive sidecar
I/O for the non-routed core's refault traffic.
