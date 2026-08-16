# P25 — Compact selected-slab upload

VERDICT: GO

## Question

Can the P24 persistent full-width expert workspace preserve the frozen
calibrated96 arithmetic while replacing thousands of small synchronous slab
transfers with one compact upload and a deterministic device scatter per routed
expert?

## Change

The opt-in path packs the selected gate, up, and down slab records for one
expert behind a 32-byte natural-slab index header. It then:

1. clears the persistent full-width device tensors;
2. performs one compact host-to-device upload;
3. scatters the records into their native full-width tensor positions on the
   device;
4. executes the unchanged full-width expert graph;
5. preserves the existing expert-ID accumulation and calibrated mean-tail
   semantics.

It is enabled with:

```text
DFLASH_KIMI_P23_PERSISTENT_SCRATCH=1
DFLASH_KIMI_P25_COMPACT_UPLOAD=1
```

The default path remains unchanged. The compact mode currently requires the
persistent scratch layout, direct-pread provider, and CUDA. Its device scatter
has not yet been ported or measured on HIP.

## Reproduction

Base commit: `075d7e246eb8ae3b0973cd2ad063253e841a6e9a`

```bash
KIMI_P23_PERSISTENT_SCRATCH=1 KIMI_P25_COMPACT_UPLOAD=1 \
  KIMI_P23_STAGE_PROFILE=1 \
  scripts/run_kimi_p23_core_family_smoke.sh latent,shared \
  /mnt/kimi-k3/results/kimi-p25-compact-upload-eight-row-20260816 8

KIMI_P23_PERSISTENT_SCRATCH=1 KIMI_P25_COMPACT_UPLOAD=0 \
  KIMI_P23_STAGE_PROFILE=1 \
  scripts/run_kimi_p23_core_family_smoke.sh latent,shared \
  /mnt/kimi-k3/results/kimi-p25-compact-control-eight-row-20260816 8
```

The compact run was followed immediately by the compact-off control.

## Semantic gate — MEASURED PASS

Both arms produced byte-identical eight-row logits:

```text
SHA-256 8daa924c13dd94489541f5d259eb2b72873b9cd49a074aee348aecd5dae90ca7
output IDs 11 374 4936 261 814 2742 316 374
```

The preceding two-row compact and compact-off pair was also byte-identical to
the established P24 reference:

```text
SHA-256 f4f694d31d6d00c3b0d941b66440f9c271be60e560f8c0e87ea25d230e3848af
```

## Adjacent eight-row comparison — MEASURED

| metric | compact-off | compact | change |
|---|---:|---:|---:|
| decode total, 7 transitions | 58.701 s | **31.284 s** | **-46.7%** |
| seconds/transition | 8.386 | **4.469** | **-46.7%** |
| true transitions/s | 0.1192 | **0.2238** | **+87.6%** |
| routed preparation | 18.242 s | **8.751 s** | **-52.0%** |
| expert-provider stage | 39.206 s | **21.382 s** | **-45.5%** |
| prefill | 11.606 s | **5.309 s** | cache-sensitive |
| peak VRAM | 15,989 MiB | 16,035 MiB | +46 MiB |
| process energy | 8.788 kJ | 5.463 kJ | -37.8% |

The compact run handled 70,596 selected slab records across 8,178 routed expert
evaluations. The prior path issues roughly three transfer operations per slab;
the new path issues one authoritative transfer per expert, reducing that
submission count by approximately 25.9x. Authoritative bytes and calibrated96
model semantics are unchanged.

Direct-I/O time was 6.348 seconds in the compact arm and 6.583 seconds in the
control. The latency gain therefore comes primarily from upload submission and
device-layout overhead rather than reduced storage traffic.

## 32-row steady-state check — MEASURED

The compact path completed 32 rows / 31 real decode transitions in 146.182
seconds:

| window | seconds/transition | transitions/s |
|---|---:|---:|
| all 31 decode transitions | 4.716 | 0.2121 |
| final 16 transitions | 4.831 | 0.2070 |
| final 8 transitions | 4.783 | 0.2091 |

Peak VRAM remained 16,035 MiB. The first eight logit payload rows were
byte-identical to the established eight-row run; only the 48-byte trace header
differs because it records the requested row count (32 versus 8).

The 32-row trace requested 198,755,192,832 logical provider bytes, including
exact fallbacks. Selected direct reads consumed 24.860 seconds; the separate
exact-fallback stream reported 33.598 GiB at 4.640 GiB/s. Storage is therefore
no longer the only or dominant serialized cost at this policy. The final 16
rows averaged 3.269 seconds in the expert stage and 1.397 seconds in routed
preparation.

## Interpretation

This is a measured systems improvement with zero model-quality trade. It
removes a large serialization cost that had been hidden behind the logical
selected-byte accounting. It does not reduce authoritative expert bytes, the
native 3,072-neuron arithmetic width, or per-expert graph launches.

The independently tested attempt to place all experts from a layer into one
larger graph remained byte-identical but was about 7% slower and consumed about
1.29 GiB more VRAM, so it is not part of this change.

Next: eliminate the extra pageable host repack by reading/coalescing selected
records into reusable pinned compact staging, instrument packing/upload/scatter
with CUDA events, then overlap expert N+1 delivery with expert N native
execution while retaining deterministic final accumulation.
