# K3 X8 Lane B: perfect-token production verifier oracle

## Verdict

**NO-GO.** A perfect q=4 draft is slower than exact autoregressive decode on
the frozen X7 production Ponytail target stack. The preregistered `1.15x`
promotion gate was missed by a wide margin in both candidate arms. Do not
integrate the current DSpark under this target closure.

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

This result closes training or integrating a draft against the current
physical-width8 X7 production verifier: even a perfect draft cannot pass the
economic gate. It does not prove speculation is universally impossible. A
future verifier would need to remove the width8 padding/traffic tax or make
multirow expert service materially cheaper before draft quality becomes the
binding term.

Machine-readable closure, exact byte counters, hashes, and retained artifact
paths are in `results/k3_x8_lane_b_q4_ab.json`.
