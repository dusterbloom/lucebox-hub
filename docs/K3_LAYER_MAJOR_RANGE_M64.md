# K3 M64 layer-major range stage

## Verdict

**EXACT, STORAGE-FAST, END-TO-END NO-GO.** The temporary runtime stage is
removed. Do not tune or rerun this M64 shape.

The candidate planned all 64 routed rows once, unioned identical sidecar slab
records, merged adjacent/overlapping aligned ranges, read the misses into one
bounded layer arena through the existing 16-worker O_DIRECT pool, and then fed
the current exact one-row P41 evaluator and canonical host join. It changed no
route, weight, quantization, expert arithmetic, recurrent order, or join order.

## Qualified result

| M64 verify arm | control | layer arena | change | registered gate |
|---|---:|---:|---:|---:|
| verify + exact commit | 71.701817 s | 71.235395 s | **-0.466422 s (-0.65%)** | at least -5.0 s |
| matched direct service | 34.470752 s | 32.071139 s | **-2.399612 s (-6.96%)** | at least -15% |
| physical direct bytes | 106,598,383,616 | 105,142,243,328 | -1,456,140,288 (-1.37%) | no increase |
| expert stage | 49.386080 s | 48.907434 s | -0.478646 s | diagnostic |

Both arms used binary
`9e7bd86c394de92bb8276d99285412efd5986bce5568bbc0326b6e9fd6fe40a4`
on physical GPU1 (`gfx1151`) under the performance platform profile. Both were
bit-identical at full logits and argmax, identical at recurrent/MLA aggregate
and per-layer hashes, and identical in logical provider traffic. The candidate
covered 92/92 routed layers, had no escaped direct reads, zero compact fallback
or invalid jobs, and both qualified arms had zero swap-in/out deltas.

## What the discriminator actually taught us

The physical range reader worked. Its 26,134 submitted extents read
105,142,243,328 bytes in 19.453761 seconds: **5.033543 GiB/s**, close to the
retained sequential ceiling. The failure is above the SSD:

- 601,668 requested records collapsed to 192,814 unique records, but the P30
  control already served repeated records from its 16-GiB host cache;
- the candidate therefore removed only 1.37% of actual physical bytes at M64;
- converting the arena back into route-local P41 compact wires cost
  **12.617378 seconds**; and
- only 0.466422 seconds survived at the full exact boundary.

This is the clean answer to the earlier mystery: range ordering can drive the
XG7000 near 5 GiB/s, but M64 is too narrow to change physical density after
P30, and a read-once arena is not useful if every route is copied back out of
it before the unchanged evaluator can run.

The much wider M2048 storage-only result remains valid and separate. It does
not turn this M64 result into a live-prefill claim. Any future wide integration
must first earn both of these properties:

1. a macro wide enough that unioning changes *physical* GiB/position, not just
   logical request count; and
2. direct consumption of arena records (or an equivalent bounded view) without
   recreating route-local payload copies.

## Evidence and invalid-arm repair

The first attempted control was exact but invalid because 149 MiB of the
tmpfs-backed retained ROCm overlay was swapped out. The repair copied that
overlay to file-backed `/tmp`; source and destination trees had the same SHA-256
`b779a5c1464fb4f09a89a0abd75896c9f2e2c401efe6037edc242620372f8dbe`.
No library bytes or performance gates changed. The qualified control and
candidate then had zero swap-in and swap-out deltas. There was no rerun after
the valid miss.

Exact launch closures, pre/post telemetry, raw per-position timing logs and
full machine-readable oracle results are preserved under
`results/k3_layer_major_range_m64_*`. The concise result ledger is
`results/k3_layer_major_range_m64.json`; the frozen decision rule is
`results/k3_layer_major_range_m64_prereg.json`.
