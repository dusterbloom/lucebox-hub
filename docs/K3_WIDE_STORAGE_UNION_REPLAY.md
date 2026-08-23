# K3 wide storage union replay

## Verdict

**GO.** The existing XG7000 can service the retained 2048-position P65
selected-slab geometry far above the registered 20-position/s continuation
gate when the same physical ranges are scheduled layer-major and read once per
macroblock.

This is a model-free storage result. It is not K3 prefill throughput, route
`M_e` evidence, or proof that the current exact compact expert executor can
consume the schedule.

## Frozen plan

- Trace SHA-256:
  `38b4af38ef2405c23f30a78d2d6d3b2e0c573ae1e6196fed35bb25eef5df66c6`
- Manifest SHA-256:
  `1a92808a09abd905e3e8f3837003d7b13a2a3f1178d88aa4242396a2b3f44e11`
- Replay script SHA-256:
  `d504cc8664b8c4e0ae11fd27d56f48f8a2aa347635167e68cf9a5c3427895b2f`
- Macro width: 2048 positions
- Dependency boundary: one macro/model-layer
- Ordering: sequence, macro, layer, file, offset
- Union rule: merge overlapping or adjacent page-aligned ranges per file
- Plan fingerprint:
  `d407673fe2905771340874d5a3a6ef00f7fbbc84b0c10710ceac5704814f0fcc`

Plan-only validation reproduced exactly 92 layer groups, 35,772 physical
requests and 74,656,587,776 submitted bytes. The union plan is
0.0339498855 GiB/position, 73.740% fewer bytes than the 284,295,323,648-byte
recorded physical trace and 86.310% fewer bytes than a full 507.9-GiB bank
scan.

## Qualified physical replay

The preregistered arm order was QD4 then QD16, one pass each and no
post-observation reruns. Both arms used `O_DIRECT`, persistent aligned buffers,
the performance platform profile, no cache drop, no model and no build.

| QD | seconds | GiB/s | storage positions/s | p50 | p95 | p99 |
|---:|---:|---:|---:|---:|---:|---:|
| 4 | 14.734819 | 4.718712 | 138.990514 | 1.340 ms | 3.464 ms | 5.224 ms |
| 16 | 14.053003 | 4.947652 | **145.733976** | 5.319 ms | 10.526 ms | 17.082 ms |

For both arms, process physical bytes matched submitted bytes within the
registered two-percent band, background reads were zero, swap-in/out were
zero, and no direct read was short. Device writes were 77,824 bytes and
3,108,864 bytes respectively. Peak reported NVMe temperature was 49.85 C.

QD16 reached 88.665% of the previously qualified 5.580138451-GiB/s raw
sequential ceiling. Its 145.734 storage positions/s is 7.287 times the
registered 20-position/s continuation gate and 8.952 times the
16.28035098-position/s long-term prefill target. Storage therefore has ample
margin for this selected-slab schedule.

## Engineering decision

Do not implement a full-bank scan: at M=2048 its analytical raw ceiling is
only about 22.5 positions/s, whereas the measured union plan is both smaller
and already near raw device service.

The next gate is arithmetic and representation, not another storage replay.
The current exact P41/P58 payload cannot be bound directly to the common
multi-row expert graph: it uses compact selected slabs, natural-K mapping and
mean-tail correction, while the common graph expects a complete native expert.
The existing split-slab bridge was already rejected by H17 because splitting
the 3072-K down reduction changes floating-point association.

Before adding a production-wide provider, use the existing stream-compute test
surface to compare the established one-row quantized path against widths
2/4/8. Width four crosses the current MMVQ-to-MMQ dispatch boundary. Only a
passing exactness gate earns the small default-off service adapter and the
accepted-prompt/live-commit P58 extension; the recurrent core must remain
causal and execute once.

Machine-readable preregistration and output are in
`results/k3_wide_storage_union_replay_prereg.json` and
`results/k3_wide_storage_union_replay.json`.
