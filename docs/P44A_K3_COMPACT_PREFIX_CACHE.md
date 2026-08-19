# P44a — compact-prefix GPU slab cache

## Verdict

**BYTE-EXACT AND RESOURCE GO; PERFORMANCE/RETENTION NO-GO; IMPLEMENTATION
REMOVED.**

P44a kept P41 component-major compact slab images resident on GPU1 under a
bounded 8-GiB common-prefix cache. It removes some repeated sidecar reads,
component-major packing and payload H2D without changing logits or traffic.
That resource result does not make it a throughput result. The final
preregistered A1/B/A2 fact is invalid for a comparative claim because the two
off controls differ by 4.06515%, above the 1.5% limit; independently, B reaches
only 1.432308353 true AR/s, below the 1.480000/s absolute floor. B's direct-I/O
counter is 11.1% better than the off bracket, so the environmental skew favors
B rather than explaining its miss.

No further repeat is allowed under the one-retry cap. The feature is removed,
not retained behind a default-off switch. The reusable conclusion is narrower:
residency can cut delivery work, but launch, compact graph execution and the
routed-preparation floor dominate. The next credible systems boundary is
batching or a persistent execution boundary, not another version of this cache.

## Frozen trace design and corrected ceiling

The frozen 17,917-event P41/P42c route trace initially predicted an 8-GiB
common fixed-slot cache would yield 33.37% full event hits and 43.21% requested
slab coverage. That accounting was incomplete: a cache lease can expose a
resident superset to the compact executor. Replaying completed available events
as `executed_depth = max(requested_depth, resident_depth_before_lease)` gives
119,417 executed versus 101,532 requested slab depths at 8 GiB: **+17.6% graph
work**. With P41 graph time scaled by this work and only proportional compact
packing credit, the fact-only ceiling is about **1.49 true AR/s**, not a path to
10/s.

This is a capacity/ceiling analysis, not a runtime forecast: it does not credit
unproven I/O overlap or extend the short fact to a broad workload.

## Bounded functional retry

The first B attempt, rooted at
`/home/duster/kimi-k3-deploy/p44a-current-aba-20260819/b-on-8g`, fails after 58
completed events at layer 11. The cache was sized for the 6,451,200-byte layout
but encountered the 7,827,456-byte layout. It fails closed rather than using an
undersized slot. One and only one bounded retry pre-sizes the pool for the
maximum observed 7,827,712-byte aligned compact slot. The final A1/B/A2 triplet
then has 17,917/17,917 completed events, zero aborts and zero non-prefix uses.

## Final adjacent fact

The final roots are
`/home/duster/kimi-k3-deploy/p44a-final-aa48-aba-20260819/{a1-off,b-on-8192,a2-off}`.
All arms use runner
`aa48c84576ab825d9d7317b4149f4422f25d4aba0d0bb3eff62dd3b15c2941ba` and match
the qualified logits and traffic SHA-256 values.

| Arm | Decode | True AR | Total | Routed prep | Experts | Direct I/O |
|---|---:|---:|---:|---:|---:|---:|
| A1 off | 6.324445294 s | 1.264933070/s | 790.318 ms | 272.461 ms | 483.332 ms | 12.118 s |
| B on, 8 GiB | 5.585389477 s | **1.432308353/s** | 697.940 ms | 271.274 ms | 390.789 ms | 11.239 s |
| A2 off | 6.072468479 s | 1.317421412/s | 758.827 ms | 274.028 ms | 449.999 ms | 13.168 s |

The off bracket is 6.198456887 seconds and 1.290643808/s. B appears 10.9763%
above that rate, but this comparison is formally invalid: control spread is
4.06515%, versus the 1.5% preregistration maximum. More importantly, B misses
the absolute 1.480000/s floor by 0.047691647/s (3.22%). The next gate, broad
and steady-state qualification, is therefore stopped.

## Cache and resource counters

The final B pool allocates 8,587,000,064 bytes, with 8,571,344,640 usable bytes,
and has 1,097 fixed 7,827,712-byte slots. Its request/caching counters are:

| Counter | Value |
|---|---:|
| Requested / resident / missing slabs | 101,532 / 39,293 / 62,239 |
| Full hits / extensions / cold fills | 5,294 / 459 / 12,164 |
| Evictions / unavailable | 11,069 / 0 |
| Completed / abort / non-prefix | 17,917 / 0 / 0 |
| Executed compact graph depths | 118,971 |
| Logical requested / suppressed payload | 55,976,374,272 / 21,614,896,128 B |
| Compact packs | 12,623 |
| Payload H2D | 34,361,478,144 B |
| Direct physical sidecar reads | 22,095,003,648 B |

Against the off path, B cuts payload H2D by 38.61% and direct physical reads by
8.34%. Its component-major pack counter falls from the off-bracket 1.548112612
seconds to 0.969386244 seconds. Those necessary resource gates pass, but they
do not override the performance gates.

## Exactness, deletion and restoration

All final arms retain the P42c/P42d logit SHA-256
`cce1bd031e90eb13928ffddfb7e9329d75d55419a8f73b6479a920fe6c561a69` and traffic
SHA-256 `e2eb5fcca9e0138d326892710977f4bd5dad1b7166d37cce6ef3675b0a911f13`.

The measured P44 snapshot is presently preserved only at
`/tmp/p44a_final_measured_before_removal`; its nine-artifact manifest is
`9b44a7186374fba10cd03dfecd215bb9fd06f65810d5ae3643962a5c5be89ecd` (9/9
verified). P44's isolated delta versus P42d was +859/−82 production raw lines,
+619/−3 test lines and +8 CMake lines: +1,401 raw net and +1,354 pure lines.
Removal is therefore a real −1,354-pure-line result, not an archive move. A
durable archive of that `/tmp` source snapshot is still required; its temporary
location is the remaining process gap noted by the independent audit.

The removal restores the qualified P42d runner
`1c9a203030ce5650034196cadb9b4c4d83413e20ab4048093a077600e5c5a2a2` and its
source hashes. Local and remote searches find zero P44 strings. Local and remote
gates pass: provider; ordered join on both Lucebox4 GPUs; MoE compute on both;
and sparse-K 68/68 on each GPU.

## Decision

P44a proves an exact cache can reduce delivery resource counters. It does not
prove a throughput gain or earn retention, a broad run, default enablement or
any deletion of P42d/P41 controls. Keep the frozen evidence and reusable
test/API seam only. Pursue a batch or persistent-execution boundary that
attacks launches, graph work and routed preparation together; the immediate
pre-speculative target is **at least 2.0 true AR/s**.
