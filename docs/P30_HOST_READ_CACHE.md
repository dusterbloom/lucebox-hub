# P30 bounded host read cache

## Verdict

**MEASURED NO-GO as a runtime default.** The trace simulator correctly found
substantial cross-token byte reuse, but its storage-only ceiling does not
survive the integrated host lookup/copy, transfer, scatter, core, and model
costs on this machine.

The opt-in cache stores immutable aligned progressive-slab records and
calibrated mean records in a bounded host LRU. It excludes native exact-fallback
weights, resets between independent prompts, and does not change routing,
selection, mean-tail semantics, GPU layout, or arithmetic. Every test used the
same 53-token official-template code prompt and generated 24 tokens.

| Host cache | Provider reads | Reduction | Prefill | Decode | Decode speedup |
|---:|---:|---:|---:|---:|---:|
| 0 MiB | 112.179 GB | — | 83.865 s | 34.292 s | 1.000x |
| 256 MiB | 112.179 GB | 0.00% | 85.805 s | 35.479 s | 0.967x |
| 2,048 MiB | 75.121 GB | 33.03% | 88.283 s | 34.556 s | 0.992x |
| 8,192 MiB | 48.367 GB | 56.88% | 90.068 s | 33.811 s | **1.014x** |

All four full-logit files have the identical SHA-256
`e1282eac9f7de57cb731680971ef9655f357026d813b96d669571fd4a36db93d`;
generated tokens are also identical. The 8 GiB run served 63.813 GB from the
cache, but reduced measured direct-I/O exposure by only about 1.11 seconds and
slowed prefill by 7.40%. Peak anonymous resident memory grew to about 9.28 GiB;
the run completed safely, but that capacity buys too little integrated speed.

## What this falsifies

The earlier 8 GiB unified trace simulation remains a valid cache-hit and
storage-roofline calculation. It is not an achievable integrated decode-rate
prediction. Its roughly 8.85 token/s value assumed cached bytes were free and
that storage dominated the remaining transition. This real implementation
falsifies both assumptions for the current host-copy design.

Therefore P30 stops before a broad suite. A future GPU-resident cache should be
considered only after stage accounting shows that eliminating its H2D/scatter
work has a material ceiling; it must not inherit the 8.85 token/s claim.

Machine-readable evidence is registered in
`results/k3_p30_host_read_cache.json`.
