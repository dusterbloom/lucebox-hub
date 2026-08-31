# K3 terminal-BWS v2

## Provenance lock

The experiment branch is `experiment/k3-terminal-kl-bws-v2` at
`fac048c090c74e5f8f989bffcda3aadc0bc8c266`, created from the clean commit
instead of the dirty production worktree.  The research-only reference is
`perf/k3-layer-major-prefill` at `102cf35ab2d86dc37e63faeece180267b2e6`;
its terminal-state adoption parent is `04ce074a17f37285b9de0e7169b1ee307f9f1000`.

The imported evidence objects are immutable Git blobs:

| Object | Blob SHA-1 |
| --- | --- |
| `docs/H16_PROGRESSIVE_FISHER_SLABS.md` | `ec28130efc354dfb9b6dc291b51bebd85f390ca6` |
| `docs/K3_H22_LAYER_ADAPTIVE_ALLOCATION.md` | `8f7ea9083da71e6299da6530f7279697c093dd23` |
| `docs/H23_ADAPTIVE_BYTE_FRONTIER.md` | `049a369fdcd771bd131e6ecd98edee9d6862ccdb` |
| `docs/H23_10K_CALIBRATION_RUNBOOK.md` | `432b73b7a72d33b3fd45cc12641ab063f558e1f3` |
| `docs/K3_LAYER_MAJOR_RANGE_M64.md` | `388f2ab8e1ef5d9fcb79741b88d969e652df5acd` |
| `results/h23_10k_projected_byte_frontier.json` | `604392ceb60ed4e1e93e89a23319fcadad9d7fa6` |
| `results/h23_10k_aggressive1p8_quality.json` | `81f82ce50dce6eca958234d4ace12236e8a6cccf` |
| `results/h23_10k_moonshot1p2_quality.json` | `1960f193ce812f73f35c6c8c99ed4b2521f52c38` |

## Decisive first experiment

For each representative isolated routed layer, retain the existing exact
teacher trajectory and evaluate one changed slab group at a time.  Each plan
row must name the layer, route/expert, natural slab IDs, byte count, local
residual/cosine, selector score, and paired full-vocabulary trace paths.
`scripts/analyze_kimi_terminal_slab_screen.py` rejects unaligned traces and
produces mean/median/p95/max terminal KL, top-1 changes, logit-margin change,
and conditional KL-recovered-per-byte.  It reports rank correlation only after
three real interventions; missing estimators remain `null`, never zero.

Screen independent marginals first, then use the measured winners for a greedy
conditional restoration series at budgets 8/12/16/24/32.  The only Phase-A
GO criterion is lower measured terminal KL at equal-or-lower authoritative
bytes than the frozen local Budget24 control.  No projection is a GO.

## Frozen comparison facts

The 10K trace has exact geometry 9.069580078125 GiB/position, a measured
fallback floor 0.009017229080200195 GiB/position, and a registered Budget24
floor 1.1427147388458252 GiB/position.  The measured 1.8-GiB H23 candidate
used 1.8344892914762203 GiB/position with mean KL 0.2963024675818419 and 6/6
exact sequences on the small suite.  The measured 1.2-GiB candidate used
1.2200937172801225 GiB/position with mean KL 0.7417508484726244 and 5/6 exact
sequences.  These are controls, not Phase-A results.

M64 layer-major was an exact, storage-fast end-to-end NO-GO: only 0.65% total
improvement.  Its causal/state invariants may be reused later; its shape must
not be tuned as a prefill claim.

## Lucebox4 scalar control (2026-08-31)

The complete deployed K3 teacher was found and used on Lucebox4:
`Kimi-K3-KDA-HYBRID-Q2-MIDLATE-00001-of-00014.gguf` (first-shard SHA-256
`93cf4fa551660cbc390e6bdea12caf67403cdb0091d82a17b14e39d253708fc4`), with
the calibrated96 manifest SHA-256
`1189c85743a1e5f41901f09b090ea3a4a567c2020bce53400e9ae5f039409f56` and
natural-sidecar manifest SHA-256
`86216b66256f61bdcbe8d796dc2e9316e3dccdeba801ea296b2e779cf86394e9`.

The production smoke harness must run as `HIP_VISIBLE_DEVICES=1` with logical
GPU 0.  Addressing physical GPU 1 directly reproducibly faulted inside ROCm
graph capture's fused RMSNorm before a forward result.  This is a hardware
topology/harness limitation, not a BWS observation.

On the fixed five-token raw fixture `According to all known laws`, an exact
terminal capture predicted token ID 11 (`,`).  The frozen H23 moonshot table
(all 92 layers at Budget24) measured 5.668487549 GiB logical provider bytes
for the five prefill positions, or 1.133697510 GiB/position; its terminal KL
was 1.158164302 and its top token was ID 318 (` of`).  This is a one-row
plumbing control and must not be interpreted as a quality-suite score.

The new immutable plan trace showed that at the final token, terminal-sensitive
layer 92 spent all 24 records on slabs 0--11 of experts 382 and 4.  Forcing
one omitted-route candidate (layer 92, expert 369, rank 11) at the same bytes
made terminal KL **worse**, 1.193684550 (+0.035520248), while retaining the
same wrong top token.  It falsifies the simple claim that an arbitrary omitted
route/slab is a terminal-quality rescue.  It is one conditional datum only;
no terminal ranking, rank correlation, or below-Budget24 conclusion is earned.

`results/k3_terminal_bws_v2_lucebox4_scalar_20260831.json` is the immutable
result summary; the raw F32 logits, plan TSV, traffic TSV, stdout/stderr and
SHA256SUMS live under the absolute artifact roots named there.
