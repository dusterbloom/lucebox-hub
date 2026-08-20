# P59A — K3 grouped compact graph discriminator

## Verdict

**BYTE-EXACT COMPUTE ROOFLINE; GROUPING NO-GO. P59B GATED OFF.**

P59A tested the smallest arithmetic-preserving grouped-execution boundary behind
P58A: place independent existing P41 compact branches for one causal
layer-position bucket in one GGML graph, without changing selected bytes, row
arithmetic or output layout. The corrected Lucebox4 GPU0 run is byte-exact for
every registered causal geometry cell and projects **35.822693 positions/s**
with all compact payloads resident. That clears the 20 positions/s minimum and
24 positions/s preferred *absolute compute-roofline* thresholds.

It does not clear the activation gate. Exact P56 weighting gives only
**1.081548x** over serial submission, below the 1.16x minimum and 1.20x
preferred speedups. Alternating controls also span 8.31% for serial and 9.26%
for grouped execution, above the registered 2% validity ceiling. The benchmark
therefore exits 2 and records `NO_GO`. GPU1 was not run, as preregistered.

This is an all-resident compute discriminator, not an integrated prefill result.
It contains no storage or payload H2D inside the timed windows, and HIP graph
capture was not compiled on the qualified build. The measured operation is
honestly a grouped GGML submission, not persistent native-graph replay.

## Exact causal workload

The corpus is derived from the frozen P56 broad-12 prefill trace. P59 does not
group jobs across tokens, layers or layer-position boundaries. Within each
causal layer-position group, jobs are partitioned by complete compact spec and
selected prefix depth. One representative uses the first real natural mask and
natural-to-compact map observed for each `(spec, depth, bucket size)` cell; the
measured cell time is then weighted by that cell's exact trace count.

| quantity | frozen P56 value |
|---|---:|
| prompt positions | 552 |
| routed layer-position groups | 50,784 |
| compact jobs | 223,579 |
| causal full-spec/depth buckets | 151,947 |
| observed `(spec, depth, bucket size)` cells | 131 |
| representative jobs | 459 |
| singleton buckets | 112,791 / 151,947 = 74.2305% |
| jobs in singleton buckets | 112,791 / 223,579 = 50.4479% |

The exact bucket-size histogram is:

```text
1:112791  2:26408  3:6483  4:2391  5:1155
6:819     7:410    8:389   9:206   10:283
11:103    12:271   13:125  14:101  15:12
```

The trace, manifest and jobs TSV SHA-256 values are respectively
`248548c73d396eca091aca34a02f5435df02eace471268be002eb5af319268e1`,
`6655058414d94c2b2667ff278ac3b59324b58b774f5332a92b4a038d35861707`
and `4428473c66679278f1936b61d03bef54d887f3774615235ebebb963f18729a14`.
The 131-cell fingerprint is `4145dd556b954ea4`; representative real maps hash
to `9c099cb4948005dc`.

This is exact causal-geometry weighting with representative real maps. It is not
an exact replay of all 223,579 route maps.

## The first failure and its root cause

The first GPU0 implementation failed byte parity. The failure was not evidence
against grouped arithmetic. Its retained P41 `CompactEntry` graphs left
input/gate/up/down/map tensors inside a gallocr-managed graph allocation.
Gallocr may reuse an input leaf's storage after its last child. Static allocator
review identified output/gate-leaf reuse as the strongest concrete mechanism.
Production P41 reuploads those leaves before every compute, while the
all-resident benchmark intentionally did not.

The decisive A0 discriminator replayed the same singleton serial graph
immediately after its teacher, with no poison, no grouped graph and no reupload.
It already failed. That proves the retained-leaf residency assumption was
invalid before dummy nodes, grouping or graph capture could matter. No retained
diagnostic log records the candidate gate/output addresses, so the specific
overlap remains an allocator-lifetime diagnosis rather than a directly archived
observation. The prediagnostic source snapshot is
`/tmp/p59-gpu0-failure-6ed55b4-20260820.tar`, SHA-256
`360199344e9f3911ab4cfcf7725b351496a1e98b84ffe5af0baf96c10d62153c`.
The path is ephemeral; the hash and diagnosis are retained here.

## Corrected exact boundary

The reviewed correction used two test-owned metadata contexts:

1. a static context owned only op-NONE input, gate, up, down, map and dummy
   leaves, allocated once in a backend buffer;
2. a compute context owned P41 operations, intermediates and cgraphs;
3. outputs lived in a dedicated aligned backend arena and were marked outputs;
4. one gallocr allocated only the combined grouped graph's intermediates;
5. serial and grouped graphs referenced the same operation and output pointers;
6. one causal input was shared within a cell, while weights, maps and outputs
   remained distinct per job; and
7. leaf hashes were checked around two serial replays with no upload, outputs
   were poisoned, synchronized, then compared after grouped execution.

The singleton and mixed-map multi-job sentinels passed before the complete
131-cell corpus. All cells remained byte-equal to a fresh synchronous P41
teacher. Static leaves, persistent outputs and gallocr scratch peaked at
45,299,712 bytes combined, below the 512-MiB cap. Timed payload H2D and storage
were both zero.

## Lucebox4 measurement

The Release build used base HEAD
`6ed55b4528b8a29888e91a840af58c9282e80ac8`, the dedicated default-off P59
option, `CCACHE_DISABLE=1` and at most four build jobs. The Release contract
self-test passed all provenance, representative-map, memory-limit and graph-env
negative checks. The model-free benchmark ran serialized on GPU0 `gfx1201`.

| metric | serial | grouped |
|---|---:|---:|
| projected 552-position device window | 16,665.818960 ms | 15,409.226588 ms |
| projected submissions | 223,579 | 151,947 |
| projected synchronizations | 223,579 | 151,947 |
| projected positions/s | — | **35.822693426** |
| weighted speedup | — | **1.081548049x** |

The unweighted micro-cell medians were 0.199679 ms serial and 0.161839 ms
grouped, or 1.181642x. The exact workload weighting is binding: it reflects the
large singleton population and reduces the aggregate benefit to 1.081548x.

| alternating trial | serial projected ms | grouped projected ms | speedup |
|---:|---:|---:|---:|
| 0 | 17,978.802450 | 16,586.219937 | 1.083960x |
| 1 | 17,291.716333 | 16,311.155530 | 1.060116x |
| 2 | 16,866.634860 | 15,531.452434 | 1.085966x |
| 3 | 17,337.653143 | 15,403.480313 | 1.125567x |
| 4 | 16,610.084483 | 15,390.715250 | 1.079228x |
| 5 | 16,701.106547 | 15,359.148035 | 1.087372x |
| 6 | 16,577.119815 | 15,159.303673 | 1.093528x |

The process completed in 18.70 seconds with 406,824 KiB maximum RSS. No KFD
process existed before or after. The host carried 2.3 MiB of old swap occupancy,
but `pswpin=12274` and `pswpout=15119` were unchanged and the timed command
reported zero swaps.

## Source removal and durable evidence

The complete uncommitted P59 implementation added 2,106 raw lines and deleted
six across its analyzer, tests, CMake hook, provider hook and two benchmark
sources. Closing the discriminator removes those 2,106 experimental lines and
restores the six HEAD lines. No P59 production or test hook remains.

The corrected remote evidence root is
`/home/duster/kimi-k3-deploy/p59a-g0-corrected-20260820`. Its artifact manifest
hash is `fad791670956f41b946a3d9a83caae17166017b0a8253d0d05358808d8b56523`;
`gpu0.json` is
`6c57eb7ce36eae385e387faa5576957a6ac79b7e1566de9c6a81afe4746fbc7e`.
The machine-readable durable summary is
`results/k3_p59a_grouped_graph_discriminator.json`.

## Decision

Do not run GPU1, integrate the benchmark, build P59B's custom descriptor kernel,
or claim a prefill speedup. P59A proves that resident exact compact arithmetic
has sufficient absolute capacity for the 16.280351 positions/s target, but it
also shows that merely grouping the existing branches saves only 8.15% on the
real causal distribution and cannot carry the campaign.

The next bounded decision is P60: measure and, only if justified, collapse GPU1
router/policy/job-descriptor preparation and host-visible submission overhead
while preserving the P58A exact seam. P59B may reopen only with new evidence
that a persistent descriptor executor removes a separately measured critical
path large enough to clear the 1.16x gate. Otherwise the campaign should compare
P60's attainable gain with exact multi-token proposal/verification work rather
than continue polishing grouped submission.
