# P56X — Lucebox4 XG7000 storage roofline

## Verdict

**THE RAW XG7000 CEILING IS 5.580 GiB/s, BUT THE COMPLETE UNMERGED P56
TRACE SUSTAINS ONLY 1.904 GiB/s. STORAGE HAS NO INDEPENDENT MARGIN FOR
10× COLD PREFILL AT THE CURRENT PHYSICAL DENSITY.**

This is a read-only Lucebox4 measurement. It runs no model or build, changes no
provider behavior, does not drop caches, and writes no data to the measured
sidecars. Every arm uses `O_DIRECT` and page-aligned `preadv()`. The trace lane
allocates one persistent aligned buffer per worker and preserves every frozen
P56 request plus the causal barrier between prompt-position/model-layer
groups.

The measured device is the only NVMe in Lucebox4: an XG7000-2TB 2280 on a
PCIe 16.0-GT/s ×4 link. Root, model, results and enabled swap share this device.
The scheduler is `none` and the filesystem is ext4 over LVM.

## Controls

All measured arms have:

- process physical-read bytes exactly equal to submitted bytes;
- zero swap-in and swap-out;
- no read errors or short reads;
- at most 0.000271% unattributed device reads;
- a maximum observed NVMe temperature of 53.85 °C;
- no concurrent model, build or benchmark process.

`O_DIRECT` bypasses the page cache, so no destructive or system-wide cache
drop was performed. The full QD-four controls differ by 10.55%. That is too
wide for marginal comparisons, but it does not affect the order-of-magnitude
storage decision.

## Raw sequential ceiling

One complete 7,013,429,248-byte layer-71 sidecar was read three times at each
block size. Order was reversed on the middle pass.

| block | pass 1 | pass 2 | pass 3 | decision |
|---:|---:|---:|---:|---|
| 1 MiB | 2.809 GiB/s | 0.883 GiB/s | 0.657 GiB/s | unstable |
| 4 MiB | 4.924 GiB/s | 3.095 GiB/s | 5.024 GiB/s | unstable |
| 8 MiB | 5.557 GiB/s | 5.587 GiB/s | 5.580 GiB/s | **registered** |

The eight-MiB arm has only 0.53% full spread. Its median,
**5.580138451 GiB/s**, is the practical raw sequential ceiling for this drive
and link.

## Persistent-buffer exact trace replay

The short 2,000-group prefix suggested a 3.38–3.51-GiB/s QD-four-to-sixteen
plateau. The complete trace falsified that extrapolation. The registered
result therefore uses all 41,458 causal groups, 524,863 original unmerged
requests and 291,245,891,584 physical bytes per arm.

| QD | seconds | GiB/s | p50 | p95 | p99 |
|---:|---:|---:|---:|---:|---:|
| 4 | 135.308 | **2.004647** | 0.558 ms | 2.344 ms | 5.057 ms |
| 8 | 147.756 | 1.835752 | 1.004 ms | 4.684 ms | 10.451 ms |
| 16 | 144.334 | 1.879279 | 1.480 ms | 6.891 ms | 14.559 ms |
| 4 repeat | 150.381 | **1.803709** | 0.602 ms | 2.779 ms | 5.784 ms |

The QD-four repeat mean is **1.904177512 GiB/s**, only 34.12% of the raw
sequential ceiling. Persistent buffers therefore do not overturn P56R:
request-buffer allocation was not the reason the full fragmented trace stayed
near 2 GiB/s, and simple larger-span coalescing remains rejected.

## Cold-prefill storage law

P56 measures:

\[
291{,}245{,}891{,}584 / 552 = 0.491383829\ \text{GiB/position}.
\]

At the registered 16.28035098 positions/s target, unchanged density requires
approximately 8.00016 GiB/s.

| service assumption | storage ceiling | gap to 16.28/s | density required |
|---|---:|---:|---:|
| Raw 8-MiB sequential | 11.356 positions/s | 1.434× | ≤0.342753 GiB/position |
| Exact trace-shaped mean | 3.875 positions/s | 4.201× | ≤0.116962 GiB/position |

Even an ideal transformation of every miss into raw sequential traffic needs
about a **30.25% physical-byte reduction**. If the present trace-shaped service
law remains, the reduction would have to be **76.20%**. The latter is not a
claim that caching alone must remove that fraction: P59 can also improve the
shape and overlap of the remaining reads. It is the measured boundary that any
combined P59/R9700/two-drive design must beat.

## Decision

The XG7000 roofline is now measured rather than borrowed from the older
SN850X. Do not promote P59 on the assumption that request submission or
persistent host buffers recover raw bandwidth by themselves. The next storage
discriminator should replay a held-out R9700 resident-slab miss plan through
the grouped P59 service boundary and report both residual bytes and achieved
trace service. A hypothetical second NVMe remains simulation-only because
Lucebox4 currently contains one NVMe.

Machine-readable results are in
`results/k3_p56x_xg7000_roofline.json`. Raw remote artifacts are retained at
`/home/duster/kimi-k3-deploy/p56x-xg7000-roofline-20260820`.
