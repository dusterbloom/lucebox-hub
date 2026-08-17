# Progressive slab-page cache lane

STATUS: REVISION40 TRACE GO / RUNTIME INTEGRATION OPEN

This lane evaluates a byte-bounded host LRU without changing the K3 runtime.
The simulator consumes a frozen I/O trace and the matching calibrated-runtime
provenance manifest. Every cache key contains the model checksum-registry
SHA-256, exact source-artifact SHA-256, aligned source offset, and aligned
source length. Equal offsets from another model or artifact cannot alias, and
overlapping or adjacent ranges are not hits.

Capacity counts exact cached source bytes. Python bookkeeping is not included,
so a runtime implementation must budget its key and allocator metadata inside
the configured host-memory ceiling.

The simulator never changes slab selection, exact-fallback decisions, natural
positions, activation masks, mean-tail arithmetic, expert order, or output
accumulation. Exact fallbacks are always uncached. Default runtime behavior is
unchanged because this is metadata replay, not provider integration.

## Decision trace: revision40

The decision-relevant input is the 12-prompt broad 1.22 GiB trace:

- trace SHA-256:
  `bd0c7c10a27cb2ef01c4a005c7990c9e587e2a2ed9645c8c6bc2c15b6be25e49`;
- 682 total positions across 12 independently served prompts;
- 1,505,856 aligned slab-page reads and 994,139 mean-tail reads;
- 9,765 exact fallbacks, totaling 64,887,631,872 bytes, kept uncached;
- 287,040 slab-page reads occur in decode and form the slab-only denominator.

The causal policy clears the cache at each `base_pos` reset, processes prompt
positions to warm the cache, and scores only positions at or beyond each
prompt's registered token count. Thus no later prompt benefits from an earlier
prompt, while decode can legitimately reuse bytes read during its own prefill.

| Total capacity | Slab-only byte hits | Slab bytes avoided | Unified slab+mean byte hits | Unified bytes avoided |
|---:|---:|---:|---:|---:|
| 2 GiB | 35.7035% | 56,815,255,552 | 31.2772% | 59,981,406,208 |
| 4 GiB | 51.0728% | 81,272,389,632 | 49.1643% | 94,284,128,256 |
| 8 GiB | 64.4903% | 102,623,821,824 | 62.3755% | 119,619,796,992 |
| 16 GiB | 74.9701% | 119,300,431,872 | 73.8307% | 141,587,865,600 |

The percentages have different denominators: slab-only scores 159,130,583,040
decode slab bytes and leaves all mean reads outside the cache; unified scores
191,773,655,040 decode slab-plus-mean bytes under one total capacity. Unified
therefore has a lower percentage at each point but avoids more total decode
bytes. Those are trace estimates, not measured latency, RSS, or physical-I/O
reductions.

## Why the P27 control says 0/0/49.56%

The earlier P27 result is valid but answers a different question. It uses one
32-position calibrated96 trajectory, includes its initial/prefill position in
the score, and carries one slab-only LRU across the entire trace without a
reset. Its aligned slab traffic is 156,478,537,728 bytes, about 4.56 GiB per
position, so a 2 or 4 GiB LRU evicts the preceding working set before reuse.

| Total capacity | P27 whole-trace slab byte hits |
|---:|---:|
| 2 GiB | 0.0000% |
| 4 GiB | 0.0000% |
| 8 GiB | 49.5571% |
| 16 GiB | 62.8574% |

Revision40 uses a much smaller 1.22 GiB policy, 12 prompt-local cache
lifetimes, prefill warming, and a decode-only denominator. Its
35.70/51.07/64.49% slab-only figures therefore do not contradict P27's
0/0/49.56%; workload, phase, and reset policy all changed.

## Verdict

Revision40 earns an isolated opt-in runtime A/B, with 8 GiB slab-only as the
clearest first point: it avoids 64.49% of traced decode slab reads, while 16
GiB adds 10.48 percentage points at twice the capacity. Unified caching has a
larger absolute ceiling but also couples mean-tail delivery into the cache and
is not the minimal first implementation.

No runtime cache is implemented here. Before promotion, an integrated lane
must pass byte-parity logits, unchanged fallback and selection fingerprints,
unchanged mean-tail and deterministic accumulation, hard resident-byte
accounting, and measured physical I/O/latency/RSS gates.

## Reproduction

```bash
python3 scripts/k3_progressive_io_replay.py \
  /mnt/kimi-k3/results/kimi-h23-broad-moonshot1p2-v1-20260817/io_trace.tsv \
  --mode bounded-lru-sim \
  --identity-manifest \
    /mnt/kimi-k3/artifacts/kimi-h23-calibrated96-runtime-10000/all_layers_calibrated96_manifest.json \
  --sequence-manifest \
    /mnt/kimi-k3/results/kimi-h23-broad-moonshot1p2-v1-20260817/suite/suite-manifest.json \
  --phase decode --reset-policy sequence \
  --cache-scope slab-only unified --cache-gib 2 4 8 16 \
  --output results/k3_revision40_slab_page_cache.json

python3 -m unittest server.tests.test_k3_progressive_io_replay -v
```

Machine-readable results are in
`results/k3_revision40_slab_page_cache.json`; the differently scoped P27
control is in `results/k3_p27_slab_page_cache_control.json`.
