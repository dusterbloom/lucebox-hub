# qwen4exp MTP end-to-end on lucebox4 (gfx1151): retained verify graphs (#827 port)

Box: lucebox4, HIP_VISIBLE_DEVICES=1, `--max-ctx 20480 --prefix-cache-slots 0`, model
`Qwen3.8-Flash-Next-UD-Q4_K_XL` (4 shards) + sidecar `MTP/mtp-Qwen3.8-Flash-Next-shared-Q8_0.gguf` (`--draft`; the
loader's auto-discovery looks two levels up from the model, lucebox-models keeps MTP/ one level up). Every request is
a cold prompt (`prefix_len=0` on every arm, checked in the server logs), 256 greedy tokens, one dropped warm-up per
workload per restart, arms alternating across restarts. Workloads: code16k (textparity p16384 slice, 16421 tokens),
prose16k (20037 tokens), counting (72 tokens, near-perfect acceptance). Harness: `mtp_ab.sh` + `mtp_absum.py`.

## What changed (commit e8572aa7)

The verify forward (T = k+1 rows, one row per drafted token) was rebuilt, re-allocated and evaluated eagerly on every
call (`ggml_backend_cuda_graph_invalidate_range` per call -- VERIFY-PROFILE.md measured ~18 ms of idle at k=4). It is
now a retained graph per width, `cache.verify_workspace[T]`, replayed like T=1 decode: inputs uploaded async, the
graph HIP-captured once and replayed (`[cuda-graph-stats]` for the 7234-node k=4 graph: 8 calls = 1 eager + 1 capture
+ 6 replays). Per-row stable QSA inputs (visibility + 10 I32 params per row) keep every row on the T=1 decode
arithmetic, so the outputs are bit-identical to the legacy path (smoke: tokens_identical=1 logits_identical=1,
rollback 20/20). Kill switch: `LUCE_VERIFY_GRAPH_LEGACY=1`. Main's at-once QSA path (tokenwise indexer score,
multi-row top_k_qsa) is not ported -- our ggml lacks those ops.

Decode copies (`LUCE_DECODE_REQ=q5_k LUCE_DECODE_REQ_ATTN=q6_k`) are selected by `dec(w, copy, T)` = `T == 1 && copy`:
they apply to T=1 decode (and the k=1 steps of MTP) only; verify rows (T>1) read the exact Q8_0 weights on both the
legacy and the retained path. Unchanged here (acceptance would need re-measuring before changing it).

## Verify call cost (bench_verify_realtok, 8K ids, 20 reps after 3 warm-ups, max_ctx 12000, median ms per call)

| width | HEAD binary 938bea37 | legacy (same binary as retained) | retained run 1 | retained run 2 |
|---|---|---|---|---|
| k=1 (decode) | 44.03 | 41.39 | 39.92 | 41.94 |
| k=2 | 64.21 | 59.86 | 55.00 | 58.34 |
| k=3 | 77.34 | 72.50 | 65.76 | 70.73 |
| k=4 | 81.11 | 80.42 | 76.70 | 78.25 |
| k=8 | 128.28 | 127.50 | 122.71 | 126.62 |

Retained vs legacy: k=2 -2.5..-4.9 ms, k=3 -1.8..-6.7 ms, k=4 -2.2..-3.7 ms, k=8 -0.9..-4.8 ms. Run-to-run spread of
the bench itself is 2-5 ms (compare the two retained runs and k=1), so the per-call saving is real but small: the
graph build/alloc the legacy path paid per call was mostly overlapped with GPU work; the verify call is GPU-bound.
`builds=2` at k=4 is the QSA bucket growing past 8192 mid-sweep (one rebuild, then replays).

## Baseline e2e (HEAD binary 938bea37, 1 round x 3 reps, client decode tok/s, median)

| arm | code16k | prose16k | counting | p4096 | p16384 |
|---|---|---|---|---|---|
| MTP adaptive | 32.10 (server 31.02, accept 0.709, 2.63 tok/step, 77% k=2) | 28.53 (27.66, 0.619, 2.16, 32% k=1 / 51% k=2) | 60.58 (53.78, 0.995, 7.29, 64% k=7) | 1318 | 1382 |
| MTP fixed k=4 | 31.18 (29.69, 0.584, 3.31) | 27.39 (26.50, 0.494, 2.97) | crashed (below) | - | - |
| AR, no sidecar | 25.97 | 26.47 | 29.80 | 1351 | 1408 |

MTP adaptive is +24% / +8% / +103% over AR at 16K code / 16K prose / counting. The client tok/s counts the stream
wall; the server's `decode=` excludes the draft and sampling overhead -- both are reported.

Pre-existing crash (HEAD binary, legacy path, `--verify-width 5`): after the two 16K prompts, the 72-token counting
prompt hits `ggml-cuda.cu:6631 GGML_ASSERT(upmix_row8 && upmix_row8->pairs.size() == 96)` (exact-stack
LUCE_QWEN_HC_UPMIX_ROW8 plan on the T=1 decode graph rebuilt for the short dense context with the MTP hidden-state
output). A fresh server on the counting prompt alone does not crash (new, legacy, LUCE_QWEN_HC_UPMIX_ROW8=0 all fine).
Not touched here; see the A/B below for whether the retained path hits it.

## A/B (one binary luce_server dda1dc0e / ggml-hip 60dcc204, 2 rounds x 5 reps, arms alternating, `ab/`)

Client decode tok/s, median of 10 (round medians' spread in parentheses); server `decode=` / acceptance /
tokens per step / width histogram from the `[qwen4exp-mtp]` lines.

| arm | code16k | prose16k | counting |
|---|---|---|---|
| retained, adaptive (`new`) | **32.28** (0.9%) -- 31.21, 0.717, 2.18, k1 59% k2 26% k3 10% | **29.80** (0.1%) -- 28.88, 0.654, 1.94, k1 71% k2 16% k3 10% | **61.31** (0.3%) -- 54.48, 0.979, 7.29, k7 68% |
| legacy, adaptive | 31.86 (0.2%) -- 30.80, 0.709, 2.63, k1 1% k2 77% k3 15% | 28.15 (0.3%) -- 27.31, 0.615, 2.14, k1 34% k2 50% k3 14% | 60.39 (0.2%) -- 53.62, 0.995, 7.29, k7 65% |
| retained, fixed k=4 | **32.35** (0.2%) -- 30.68, 0.584, 3.31 | **28.40** (0.2%) -- 27.38, 0.494, 2.97 | crash (below) |
| legacy, fixed k=4 | 31.05 (0.1%) -- 29.56, 0.584, 3.31 | 27.31 (0.1%) -- 26.43, 0.494, 2.97 | crash (below) |
| AR, no sidecar | 26.47 (0.1%) | 26.53 (0.2%) | 29.82 (0.1%) |

Retained vs legacy: adaptive **+1.3% / +5.9% / +1.5%**, fixed k=4 **+4.2% / +4.0%** (code16k / prose16k / counting);
every delta is > 2x the round spread. Same acceptance per width on both paths (the fixed-k rows: identical
rate / tokens per step), and the greedy-256 texts are identical across new / legacy / AR on all three workloads.
MTP vs AR after the port: +22% code16k, +12% prose16k, +106% counting.

The adaptive policy (`Qwen4ExpMtpWidth`, cycle cost fit base + slope*k) picks k=1 far more often on the retained path
(59% / 71% vs 1% / 34%): the retained graphs cut the small widths most (k=1..2 -4..-7 ms, k=4 -3 ms), so the fitted
slope steepens and the utility `commits / (base + slope*k)` favours narrow windows at the 0.6-0.7 acceptance of the
16K prompts. Fewer tokens per step (2.18 vs 2.63) at a higher tok/s, so the policy is doing its job; counting (0.98
acceptance) still sits at k=7.

Prefill (tok/s, median of 10): p4096 new 1298 (0.5%) / legacy 1359 (3.5%) / AR 1393 (0.1%); p16384 new 1376 (0.03%)
/ legacy 1404 (2.2%) / AR 1424 (0.3%). new vs legacy -4.7% / -2.0% with legacy's own round spread at 3.5% / 2.2%:
inside the 2x-spread band, same direction both sizes. chunk-auto is identical on both arms (11520; AR gets 15616
without the sidecar's buffers). Candidate mechanism: the retained verify workspaces (one ~90 MB arena + HIP graph per
width used, allocated after the weights) -- the decode-requant control measured ~2% per extra post-weight allocation
on this uncompacted box. Prefill-only re-run below.

Fixed k=4 on counting after the two 16K prompts crashes on BOTH paths (same `upmix_row8 pairs.size() == 96` assert
as the HEAD binary): pre-existing, independent of this port.

## Second A/B, 3 rounds x 10 reps (`abpre/`, same binary; meant as prefill-only, the harness ran everything)

Decode: code16k new 32.10 (0.85%) vs legacy 31.84 (0.36%) **+0.8%**; prose16k 29.85 (0.31%) vs 28.26 (1.40%)
**+5.3%**; counting 61.41 (0.07%) vs 60.38 (0.25%) **+1.7%** (n=30 each, greedy texts identical). Prefill with n=30:
p4096 new 1360.6 (0.54%) vs legacy 1359.9 (1.36%) **-0.05%**; p16384 1419.2 (0.22%) vs 1418.2 (0.32%) **-0.07%**.
The -4.7% / -2.0% prefill of the 2x5 run was noise (that run's legacy spread was 3.5% / 2.2%): prefill is unchanged.

## Remaining verify cost (ranked)

From the unprofiled bench (retained path): k=1 40-42 ms, then **~12 ms per extra row** at k<=4 (k=4 77-78 ms) and
~11.8 ms/row to k=8 (123-127 ms). VERIFY-PROFILE.md measured the GPU-busy growth at +6.75 ms/row (expert weight reads
per row: every row routes to its own top-k experts; route-union reuse r_4 ~ 0.69, ROUTE-UNION.md), so ~5 ms/row is
still launch/replay overhead and small per-row kernels, not bytes:

1. Routed expert reads per row (~6.75 ms/row GPU busy): grouped route-union expert GEMV (ROUTE-UNION.md) was the
   estimated lever on bytes (8-9 ms per k=4 window). **Tried and measured, no saving** -- see "Route union" below:
   the per-pair launch already runs at DRAM speed on the union byte count, the duplicate expert reads are served by
   L2/MALL inside the launch, and a kernel that reads each expert once is 9 ms slower per k=4 call.
2. Per-node replay overhead: the k=4 graph has 7234 nodes vs 5298 at k=1 (+484 nodes per row); at this box's
   ~1.8 us per replayed node that is ~0.9 ms/row, ~13 ms per k=4 call in total. Fewer nodes per row (the per-row
   attention sub-graphs: QSA score / top-k / gather / wmma / merge per row, kv set_rows, rope) is the lever; main's
   at-once QSA path (tokenwise indexer score + multi-row top_k_qsa) does exactly that and needs those two ops.
3. The rest (~4 ms/row): T>1 GEMV columns run batch-invariant (`mmvq_batch_invariant`, one column per row for the
   exactness guarantee), so the dense trunk GEMVs scale with rows instead of sharing the weight read.

A rocprofv3 --kernel-trace of the retained k=4 call is not usable for the idle question: the tracer inflates the
graph replay 3x (221 ms per call under the tracer vs 77 unprofiled), so a per-kernel ledger of the replayed verify
needs a different method (per-op timing with graphs disabled, or event pairs around the replay).

## Pre-existing crash, standalone status

`smoke_qwen4exp_forward MODEL 600 --draft MTP --mtp 64 --mtp-draft 4` (dense, short context) passes on the retained
path (builds=1 replays=29, tokens and logits identical, rollback 20/20), so the `upmix_row8` assert needs the server's
sequence (16K QSA prompts, then the 72-token prompt at fixed k=4): `mtp_ab.sh out 1 1 "k4=/--verify-width 5"`
reproduces it on both paths and on the HEAD binary. Owner: the exact-stack fusion plan (LUCE_QWEN_HC_UPMIX_ROW8).

Resolved (follow-up commit): the crash is the T=1 decode graph that also returns the MTP hidden row, which the
server builds at any k=0 step (the last token when exactly one remains -- the counting prompt at fixed k=4 hits it,
the 16K prompts happened to end on k=1/k=2 steps -- and every step with n_gen <= 2 or teacher forcing). With a hidden
output `hc_inject_hint` is off, the allocation differs, all 97 HC up-mix sites validate (the plain decode graph has 96:
site 10 aliases its input) and layer 0's GDN A/B outputs overlap its activation; the exact-stack contracts asserted the
plain graph's counts. The probes now engage the sites that validate and record that count per sealed graph (reported as
`[exact-stack-capture] ... noncanonical sites: ab=35/36 upmix=97/96`), the plain T=1 graph still captures 96/36/97.
`smoke_qwen4exp_forward --mtp` gained the hidden-decode check (T=1 with the hidden row: plain decode's logits bit for
bit, verify path's hidden row bit for bit); it aborts on the old library and passes on the fix. A fresh server on the
counting prompt alone does crash at its last step (the note above was wrong).

## With the decode copies (`LUCE_DECODE_REQ=q5_k LUCE_DECODE_REQ_ATTN=q6_k`, retained path, 1 x 5, `abreq/`)

| arm | code16k | prose16k | counting | p4096 | p16384 |
|---|---|---|---|---|---|
| MTP adaptive + copies | 33.11 (31.93, 0.702, 2.63, k2 77%) | 28.76 (27.78, 0.642, 1.90, k1 74%) | 61.36 (54.58, 0.981, 7.29) | 1268 | 1361 |
| MTP fixed k=4 + copies | 32.35 (31.08, 0.595, 3.36) | 28.14 (27.15, 0.486, 2.93) | crash | - | - |
| AR + copies, no sidecar | 29.19 | 29.45 | 33.01 | 1393 | 1425 |

The copies apply to T=1 only (`dec()`), so they speed up AR (+10% code16k, +11% prose16k, +11% counting vs exact AR)
and the draft, but not one verify row: MTP + copies is only +2.6% / -3.5% / +0.1% over MTP exact, and on prose16k
MTP + copies (28.76) is now *slower* than AR + copies (29.45). The semantics also differ: AR + copies samples the
requantised model, MTP + copies commits exact-weight verify logits, so the greedy texts diverge at the first token
(code16k, prose16k; counting identical). Extending the copies to verify rows would make MTP + copies consistent with
AR + copies and recover the ~10% on the verify call, but it changes the verified distribution; left as is, acceptance
and KL would have to be re-measured first (decode-requant/RESULTS.md has the KL gate for T=1).

## Route union (LUCE_MMID_UNION, rejected, kept as mmid_union.patch)

Goal: read each distinct expert once per verify window (top-10 union over 4 rows is 0.69-0.73 x 4x10 experts). Done
as a new MUL_MAT_ID kernel `mul_mat_vec_q_moe_union` (mmvq.cu, `[TAG_MMID_UNION]`): `mmid_group_prep` sorts the
(token,slot) pairs by expert and cuts runs of up to 8 pairs with the same expert; one wave per run loads each weight
block once and dots it against every pair's q8_1 row, same kbx order, same `warp_reduce_sum`, so each row's result
is bit-identical to the T=1 kernel. Gate/up fused GLU and down, types Q4_0/Q5_1/Q8_0/Q4_K/Q5_K/Q6_K, 2..16 rows,
`LUCE_MMID_UNION_ROWS` 1/2/4 rows per wave. `test_mmid_union_parity` (qwen4exp shapes, 64 experts, top-10; window /
same / random routings at T=2,3,4,5,8, duplicate and sentinel(-1) ids at T=4; gate/up Q4_K + swiglu + down Q5_1 and
Q8_0): every batch row memcmp-equal to its T=1 run, 34 cases x {on, rows=1, rows=4, off}, all PASS on lucebox4.

### Ledger of the k=4 verify (rocprofv3 kernel-trace, eager verify `LUCE_VERIFY_GRAPH_LEGACY=1`, T=5, per layer x48)

| class | per layer (ms) | per k=4 call | union arm |
|---|---|---|---|
| dense trunk mmvq Q8_0 (batch-invariant, one column per row) | 0.55 | 26.4 | 0.55 |
| routed gate/up Q4_K (`mul_mat_vec_q_moe`, fused GLU, rows_per_block 2) | 0.27 | 13.0 | 0.39 |
| routed down Q5_1 / Q8_0 (`mul_mat_vec_q_moe`) | 0.18 | 8.6 | 0.30 |
| other | 0.14 | 6.7 | 0.14 |
| attention / QSA / router | 0.10 | 4.8 | 0.10 |
| elementwise / norm / copy | 0.08 | 3.8 | 0.08 |
| GDN | 0.04 | 1.9 | 0.04 |
| quantize | 0.02 | 1.0 | 0.02 |
| busy / idle | 1.38 / 0.25 | 66 / 12 | 1.62 / 0.25 |

What the existing grouped path does in the verify graph: nothing by default on HIP (`LUCE_MMID_GROUPED` is off);
the gfx1151 dispatcher sends T=2..4 Q4_K and T=2..8 Q5_1/Q8_0 to `mul_mat_vec_q_moe` (one wave per (token,slot)
pair, gate/up GLU fused), Q4_K at T=5..8 to the generic tokenwise `mul_mat_vec_q`, T=9 to MMQ. The grouped kernel
with all type bits (`LUCE_MMID_GROUPED=1 LUCE_MMID_GROUPED_TYPES=133`) only sorts pairs by expert: k=4 76.49 ms vs
76.65 off, nothing to gain.

Why the union kernel loses: the baseline gate/up launch moves 50 pairs x 2 x 921.6 KB = 92 MB nominal in 0.27 ms =
340 GB/s, above the 220 GB/s DRAM floor; at the measured 0.65 union ratio that is 222 GB/s, i.e. the per-pair launch
already streams exactly the union bytes -- the duplicate expert reads hit the 2 MB L2 / 32 MB MALL within the
launch. The union kernel cannot cut bytes further and lengthens each wave's dependent chain (the routed waves are
only 2-3 K-iterations long, latency-bound; serialising up to 8 pairs per wave makes them longer, not fewer):
+0.12 ms per routed launch, +0.24 ms per layer, +9-12 ms per k=4 call. No scratch spills (fused Q4_K rows=2: 120
VGPRs, occupancy 12); rows=1/4 do not help.

### bench_verify_realtok (8K ids, 20 reps, max_ctx 12000, median ms per call; off/on/on/off arms)

| k | off | union | union | off |
|---|---|---|---|---|
| 1 | 41.89 | 39.98 | 40.04 | 41.89 |
| 2 | 58.42 | 60.72 | 60.67 | 58.31 |
| 3 | 70.68 | 73.46 | 73.47 | 70.65 |
| 4 | 77.38 | 86.71 | 86.76 | 77.51 |
| 8 | 123.19 | 137.46 | 137.48 | 123.71 |

k=4 matrix (10 reps): off 76.65, grouped 76.49, union rows=1 89.36, rows=2 86.01, rows=4 86.07. The union path is not
engaged at T=1 (telemetry: 0 `variant=union` launches at k=1), so the k=1 deltas are run spread (40.0-44.1 ms across
arms with identical settings). `LUCE_PRINT_ARGMAX=1` sums identical on every arm at every k (k=1 16049, k=2 25088,
k=3 22992, k=4 66549, k=8 115903; matrix k=1 619, k=4 20992).

Decision: stop rule (reachable saving < 1 ms per window; it is negative). The kernel, its `LUCE_MMID_UNION` switch
and `test_mmid_union_parity` were taken out of the tree and kept as `bench/exact/mtp/mmid_union.patch` (applies
cleanly; apply it to rerun the A/B on other targets). The e2e mtp_ab A/B was not run: the verify call is its direct
input and is slower at every width. The largest remaining class is the dense Q8_0 trunk at T=5 (0.55 ms/layer, ~26 ms
per k=4 call, 2x the routed gate/up): under `mmvq_batch_invariant` each row is one column through the generic
kernel, so a multi-column kernel that keeps the per-row reduction order (same kbx stride and per-lane accumulation,
rows interleaved the way the union kernel does) would share the weight read across the 5 rows; that is the lever
to try next, not the experts.
