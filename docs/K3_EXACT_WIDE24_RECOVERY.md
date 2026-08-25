# K3 exact uniform-budget wide-prefill recovery

## Result

The prior budget-24 wide-prefill rejection was confounded: the retained run
used budget 24 for scalar rows but budget 96 for macro rows. A minimal,
default-off provider change now admits uniform all-24 or all-96 wide policies
and rejects mixed tables.

On physical GPU1 `gfx1151`, the matched M64 reversal was:

| Arm | Prefill | Wall | Raw logits |
| --- | ---: | ---: | --- |
| exact scalar, chunk 1 | 1.820531 pos/s | 35.155 s | `aacdaac4...` |
| exact wide24, chunk 64 | **4.331290 pos/s** | **14.776 s** | `aacdaac4...` |

This is a **2.3791x** prefill gain with byte-identical raw logits. The first
unlogged signal was 1.992792 versus 4.225418 positions/s (`2.1204x`), so the
direction survived reversal.

The M64 wide stage decomposed as:

```text
causal core       4.286794 s
expert service    9.495063 s
join/output       0.958824 s
other             0.011381 s
----------------------------
stage            14.752063 s
```

Direct I/O was 7.079635 seconds inside expert service, not additive. Logical
provider traffic was identical at 92,518,293,504 bytes. Physical traffic fell
from 39,999,864,832 to 37,335,130,112 bytes and the wide arm served it at
4.911 GiB/s.

## Continued-state gate

M64 followed by live scalar decode produced the same two output IDs
(`114820 9196`, text ` causal core`) and byte-identical post-decode logits:

`09e346c4e9af1859ce4cb52b6d89e0f91564c85ab75bcd641e4935143b01f6a8`

The wide arm prefilling at 4.361404 positions/s matched the scalar arm at
1.824532 positions/s. The single decode transition is too short for a
throughput claim.

After narrowing the guard to reject mixed 24/96 tables, the final build passed
a fresh matched M8 scalar/wide raw-logit comparison with hash
`516a50ad88abd6ec6a7de7988d7ce39131c2e579e748abef73202f9f1c9e15e9`.
The existing calibrated-provider unit test also passed with physical GPU1
isolated through `HIP_VISIBLE_DEVICES=1`.

## Scope and next gate

This is an **ENGINEERING GO, EXPLICIT/DEFAULT OFF**. Production scalar defaults
are unchanged. The change only affects an explicitly selected exact-multirow
path; unsupported widths, non-authoritative sidecars, traces, mixed budgets,
and other budgets fail closed.

M64 is a correctness gate, not the 20-pos/s production width. Its 34.771 GiB
physical payload has a 6.231-second floor at the measured 5.580-GiB/s raw
ceiling, already above the 3.2-second 20-pos/s budget. M1024 is the next valid
width. At the current M1024 ledger, reaching 20 requires reductions in both
the 77.221-second causal core and the 43.682-second expert graph; further SSD
queue tuning is not the primary lever.

The 53-token P55 continuation gate was also byte-identical, but the wide phase
left a cache population that slowed subsequent scalar decode from 2.355260 to
1.965083 AR/s. Wide prefill must therefore hand off by invalidating or
partitioning only phase-local cache state before it can become a server
default.

Machine-readable evidence is in
`results/k3_exact_wide24_recovery.json`. Raw logs and logits are retained on
Lucebox4 under
`/home/duster/kimi-k3-deploy/k3-wide24-recovery-20260825/`.
