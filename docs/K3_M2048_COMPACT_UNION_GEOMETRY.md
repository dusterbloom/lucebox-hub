# K3 M=2048 compact/union-mask geometry

This is an offline reconstruction of one retained P65 causal prefix, not a
runtime result.  It identifies how much compact expert-weight payload could be
reused at `M=2048`; it does not establish storage, GPU, or end-to-end speed.

Source: `p65-a0-long-2048`, 2,048 causal positions and 92 routed layers.  The
trace SHA-256 is
`38b4af38ef2405c23f30a78d2d6d3b2e0c573ae1e6196fed35bb25eef5df66c6`.

## Reconstruction and validity

The causal route key is `(base_pos + token_index, model_layer, expert_id)`.
This matters because `token_index` resets while `base_pos` carries the global
causal position.  We retained every natural-sidecar `gate` record marked
`exact_fallback=0`, including those with `explicit_read_bytes=0`; cached rows
therefore remain in the geometry.  Per-layer sidecar manifests reconstruct the
natural slab index from file offset, payload offset, expert record size, and
slab size.

The result contains 1,239,227 nonzero compact routes from 4,521,984 gate
events.  The trace also contains 1,020,552 `exact_fallback=1` gate events,
which are outside this compact-executor candidate.  Reconstruction had zero
invalid offsets, out-of-range slab IDs, or duplicate slabs.

## Exact complete-mask chunks

The exact compact identity is the frozen sidecar generation plus
layer/expert/spec and the complete reconstructed natural selected-slab mask.
There are 66,753 identities; their route multiplicity is p05/p50/p95/max
`1 / 2 / 70 / 2027`.

Using one compact upload per route is 2,493,066,844,000 bytes (2,321.85 GiB).
If an exact graph reuses only identical complete masks, the offline payload
accounting is:

| Width | Full routes | Tails | Payload reduction |
| --- | ---: | ---: | ---: |
| 2 | 1,196,674 (96.566%) | 42,553 | 47.647% |
| 4 | 1,143,512 (92.276%) | 95,715 | 67.313% |
| 8 | 1,076,144 (86.840%) | 163,083 | 72.770% |

At width 8 that is 678,859,959,328 payload bytes, saving
1,814,206,884,672 bytes.  A layer's largest complete-mask unique-payload pool
is 4,759,795,072 bytes.  These are payload counts, not measured H2D time.

## Per-expert union-mask candidate

The stronger exact graph would upload the union of selected natural slabs once
per `(layer, expert)`, then process rows in chunks of at most eight while using
row-specific sparse-down maps and canonical joins.

There are 22,809 expert cells.  Their routed row count `M_e` is
p05/p50/p95/max `1 / 6 / 245 / 2035`; union slab count is
`1 / 2 / 12 / 12`.  The single-upload union payload is 61,697,204,000 bytes
(57.46 GiB), a 97.525% reduction versus the one-upload-per-route compact
payload.  Width-eight chunking covers 1,173,480 full-chunk rows (94.695%) and
leaves 65,747 tails across 167,887 chunks.

The cost is real: gate/up operations would see 9,315,461 union slab-rows
instead of 4,521,984 selected slab-rows, a `2.0600x` widening.  This graph is
only exact if its sparse-down map remains row-specific and joins remain in
canonical order.  Holding all union payloads for the largest layer would be
1,703,796,064 bytes (1.587 GiB); a single union image is at most 7,827,488
bytes.

The machine-readable figures are in
`results/k3_m2048_compact_union_geometry.json`.

## Arithmetic follow-up

The geometry earned an actual-shaped GPU discriminator, but width-eight did
not earn exact runtime integration.  A full-12, 245-row union candidate was
nominally `3.561165x` faster and reduced its test payload H2D by 99.012%, yet
failed byte identity against the current one-row compact teacher.  A
six-slab identical-mask width-eight candidate likewise failed byte identity.

The offline byte opportunity remains valid; the assumption that MMVQ width
eight can realize it on the exact lane does not.  The failed experimental
executor was removed.  Exact follow-up is bounded to width two or one-row
arithmetic plus layer-major storage/order; wider execution is quality-gated.
See `results/k3_compact_schedule_discriminator.json` and
`results/k3_full12_union_long.json`.
