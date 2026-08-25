# K3 X8 Lane B: perfect-token production verifier oracle

## Verdict

**NO-GO.** A perfect q=4 draft is slower than exact autoregressive decode on
the frozen X7 production Ponytail target stack. The preregistered `1.15x`
promotion gate was missed by a wide margin in both candidate arms. A bounded
follow-up made logical verification width first-class and excluded inactive
rows from routes, union payload, expert work, joins, and output. That removed
the padding traffic bug but still lost end to end. Do not integrate the current
DSpark under this target closure.

| Arm | Mode | Decode | Useful tok/s | Ratio to paired AR |
| --- | --- | ---: | ---: | ---: |
| A1 | exact AR | 31.890 s | 1.097516 | control |
| B1 | perfect q=4 | 45.691 s | 0.766020 | 0.697958x |
| B2 | perfect q=4 | 43.072 s | 0.812597 | 0.823684x |
| A2 | exact AR | 35.478 s | 0.986540 | control |

The ratio of the two arm means is `0.757473x`. Control throughput drifted
`-10.11%` and oracle throughput `+6.08%`, but even the favorable B2/A2 pairing
remained a decisive loss.

## What was tested

The retained P55 code-function fixture produced 36 natural target tokens and
terminated at EOS. The oracle received those archived target-correct IDs; no
draft was run or trained. In verifier terminology q=4 means one target seed
plus four future proposals, so each full target step had five valid causal
rows. The only qualified production outer shape is width8, so three padded
rows were masked after the valid prefix on each of seven steps:

```text
q=4; valid width=5; physical width=8
7 target steps; 35 committed state rows; 21 padded rows
36 output tokens including the final unprocessed target prediction
```

The measured binary was shared by all four performance arms. The target stack
was P40 layer-epoch cache, P46 routed preparation, exact macro union, direct
async upload, and Router8/Core8/MLA8/Tail8 on the single-owner gfx1151 path.
The platform profile was `performance`; both AMDGPU devices reported DPM
`high`.

## Exactness

All four arms emitted the same 36 IDs. Boundary logits were byte-identical at
all seven committed boundaries (`fe194e2e...90b9e1`), and terminal logits were
byte-identical (`b41609a3...ecf22`).

Before timing, the short fact-capital sentinel compared AR with the oracle at
both commit boundaries. Full convolution and SSM state plus the live MLA
prefix were byte-identical (`76cf8bea...9c8622`); boundary logits were also
byte-identical (`7d4de3f1...ced99`). State capture was omitted from all timed
arms.

`oracle-fallbacks=0`, P40 fallbacks were zero, and P41 fallbacks were zero.
The calibrated96 policy separately recorded exact source routes outside its
96-slab calibrated selection; those existing exact routes are not oracle or
execution fallbacks.

## Why it lost

| Candidate | Target | Core | Experts | Join | Output |
| --- | ---: | ---: | ---: | ---: | ---: |
| B1 | 43.115 s | 4.021 s | 38.062 s | 0.779 s | 0.247 s |
| B2 | 40.496 s | 4.026 s | 35.439 s | 0.776 s | 0.249 s |

The target still executes eight physical rows for every five useful causal
rows. Across the full matched prompt and decode, physical traffic rose from
about `231.34 GB` in AR to `302.21 GB` in the oracle (`1.306x`), while logical
provider traffic rose from `433.91 GB` to `537.39 GB` (`1.238x`). The perfect
acceptance did amortize target launches and joins, but expert weight service
remained 87--88% of target wall and dominated the saved work.

B1 and B2 had identical physical and logical byte counts. A1 and A2 had the
same logical traffic-plan hash; their physical reads differed by nine cache
records (`0.002153%`), far too little to affect the verdict. Direct-I/O service
varied materially with machine state, which is why the reversed sequence was
required; neither candidate approached parity under either storage interval.

## Closure

The initial result closed training against the physical-width8 implementation
but left one precise seam open: inactive rows could be removed before expert
service. The bounded follow-up below tested and closed that seam.

Machine-readable closure, exact byte counters, hashes, and retained artifact
paths for the initial run are in `results/k3_x8_lane_b_q4_ab.json`.

## Active-row follow-up

The follow-up kept Core8/MLA8/Tail8 as the qualified physical graph shape but
passed `logical_rows=5` to the routed provider. Only the valid causal prefix
could generate routes, union records, expert jobs, joins, or output. The
implementation was 29 net production-path lines and introduced no scheduler,
queue, cache, or alternate execution path.

| Arm | Mode | Decode | Useful tok/s | Ratio to paired AR |
| --- | --- | ---: | ---: | ---: |
| A1 | exact AR | 33.588 s | 1.042037 | control |
| B1 | active-row perfect q=4 | 35.566 s | 0.984086 | 0.944386x |
| B2 | active-row perfect q=4 | 38.723 s | 0.903849 | 0.972247x |
| A2 | exact AR | 37.649 s | 0.929650 | control |

The ratio of throughput means was `0.957523x`; the ratio derived from mean
wall time was `0.958917x`. Thus masking recovered most of the original loss
but remained about 4% slower than AR and far below the `1.15x` gate.

The discriminator did exactly what it was intended to do:

* all four traffic TSVs are byte-identical (`6c372b86...1c6`);
* the oracle serviced 35 valid rows, not 56 physical rows;
* B1 expert wall fell from the unmasked `38.062 s` to `28.165 s`;
* sparse authoritative H2D fell from AR's `433.915 GB` to `374.259 GB`;
* boundary logits, final logits, the short-sentinel recurrent state, and all
  output IDs remained byte-identical.

The remaining loss is not inactive-row expert work. Target-only work was
slightly faster than paired AR (`32.987` versus `33.588 s` in pair 1 and
`36.148` versus `37.649 s` in pair 2), but verifier snapshot/commit/driver work
added about `2.58 s` per run. Even deleting all of that overhead would yield
only about `1.02--1.04x`, not `1.15x`. In addition, the wide macro/cache service
read `15.81%` more physical bytes than AR despite the identical logical record
plan (`237.77` versus `205.31 GB`).

The active-row code is therefore removed after preserving this evidence. The
NO-GO closes perfect-q4/current-DSpark integration on the current X7 target
stack. It does not close a fundamentally cheaper target verifier, but q7 or
another width-only sweep is not earned by these bounds.

Machine-readable evidence is in
`results/k3_x8_lane_b_q4_active_rows_ab.json`.
