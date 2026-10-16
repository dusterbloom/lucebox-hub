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
3. The rest (~4 ms/row): the dense Q8_0 trunk GEMVs at T>1. They already share the weight read across the rows
   (one width-invariant launch per GEMV) but ran 1.45-1.8x the T=1 launch time on the same bytes; two weight rows
   per wave recover 3.8 ms per T=5 call and staging the T activation columns in LDS another 4.6 ms, which puts the
   class within 0.7 ms per T=5 forward of the sum of the T=1 launches -- the two "Dense Q8_0 trunk at T=5" sections
   below.

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
per k=4 call, 2x the routed gate/up); the next section takes it up. (The guess here was that each row runs as its own
column through the generic kernel; it does not -- see below.)

## Dense Q8_0 trunk at T=5: rows per wave (LUCE_MMVQ_INVARIANT_ROWS, shipped)

Same box, binary luce_server 9a720376 / ggml-hip 4e1b1748 (the tree at this commit), `bench_verify_realtok` relinked
against it. Traces: `rocprofv3 --kernel-trace` of the eager verify (`LUCE_VERIFY_GRAPH_LEGACY=1`) at k=1 and k=5
(T=5), per launch, grouped by kernel template and grid; the window is the last 5 forward calls (`op_sgn` is the
once-per-forward marker). Scratch on the box: `~/qwen4exp-f16/scratch/densever/` (`trace.sh`, `launches.py`,
`sweep.sh`).

### Launch structure (what the tree already did)

`mmvq_batch_invariant` does **not** run one column per row for Q8_0. `mul_mat_vec_q_switch_type` has a Q8_0-only
branch (`c_width_invariant=true`): one launch per GEMV with all T rows as columns, the single-column block shape
(gfx1151 RDNA2 table: one wave per weight row, 8 blocks per iteration) and `tmp[j]` per column, so each wave reads
its weight row once and dots it against the T activation columns; per column the kbx traversal, per-lane
accumulation and `warp_reduce_sum` are those of the T=1 kernel, which is the bit-exactness guarantee. The trace
shows it: at T=5 every dense Q8_0 GEMV is `mul_mat_vec_q<Q8_0, 5, ..., width_invariant=true>` with the same grid as
its T=1 launch (attn_qkv 10240 waves, attn_q 12288, attn_gate 6144, ssm_out / attn_output 2560, hc_up 10240, hc_down
320, shexp 640 / 2560, attn_k / v 512, lm head 248320). The other types (Q4_K / Q5_K / Q6_K dense) run one column
per row; this model's dense trunk is all Q8_0, so that path is not exercised here.

### Bytes ledger (read-once floor at 220 GB/s)

Per layer: GDN layer attn_qkv 26.6 + attn_gate 15.9 + ssm_out 15.9 = 58.4 MiB; attention layer attn_q 31.9 +
attn_output 15.9 + attn_k / v 2.7 = 50.5 MiB; every layer hc_attn / hc_ffn up + down 13.3 MiB and the shared
expert 5.0 MiB. Per forward 3586 MiB = 17.1 ms, plus the Q8_0 lm head 644 MiB = 3.1 ms (T=5 needs all five rows'
logits), ~20 ms floor. The weights are read once at every width, so "read each weight once for all rows" had
nothing left to save; the headroom is the T=5 launch itself: same grid, same bytes, 1.45-1.8x the T=1 time.

### Per launch, T=1 vs T=5 (median us, same grid, one row per wave)

| GEMV | MiB | T=1 | GB/s | T=5 | ratio |
|---|---|---|---|---|---|
| attn_qkv [2560 x 10240] | 26.6 | 124.0 | 225 | 178.5 | 1.44 |
| attn_q [2560 x 12288] | 31.9 | 147.7 | 226 | 218.4 | 1.48 |
| attn_gate [2560 x 6144] | 15.9 | 74.7 | 224 | 130.7 | 1.75 |
| ssm_out / attn_output [6144 x 2560] | 15.9 | 76.7 | 218 | 137.2 | 1.79 |
| lm head [2560 x 248320] | 644 | 2987 | 226 | 4336 | 1.45 |
| hc_up [320 x 10240] | 3.3 | 21.1 (hc_upmix_row8_exact) | 165 | 36.1 | 1.71 |
| hc_down [10240 x 320] | 3.3 | 21.5 (hc_down_inject_mixed) | 162 | 45.1 | 2.10 |
| shexp gate + up [2560 x 640] | 3.3 | 19.3 (fused pair) | 180 | 2 x 15.3 | 1.59 |
| shexp down [640 x 2560] | 1.7 | 16.7 | 104 | 16.0 | 0.96 |
| attn_k / v [2560 x 512] | 1.3 | 8.6 | 162 | 13.6 | 1.58 |

Q8_0 class per forward: T=1 22.4 ms (18.3 ms of `mul_mat_vec_q<Q8_0,1>` + 4.1 ms of the fused HC kernels), T=5
35.2 ms: +12.9 ms per k=4 call for zero extra DRAM bytes. The large launches sit at the DRAM floor at T=1 and at
~155 GB/s at T=5. What a T=5 wave does per 8-byte weight chunk and lane: one weight load, then for each of the five
columns the q8_1 `qs` and `ds` loads (from L0/L1: every wave streams the whole 5-column activation, 14.5 KB, again)
and the two dp4a -- the launch is bound by the activation loads and VMEM issue, not by weight bytes. VGPR 16 at T=1,
32 at T=5, no scratch.

### The knob: rows per wave (`c_invariant_rpb`, LUCE_MMVQ_INVARIANT_ROWS)

`mul_mat_vec_q` gained a trailing template parameter `c_invariant_rpb` (1 / 2 / 4), used only on the
width-invariant launch: a wave handles 2 or 4 weight rows and shares the per-column activation loads across them.
`rows_per_cuda_block` already parameterises the kernel (upstream uses 2 on the generic table); the per-row chain --
same kbx sequence, same per-lane `tmp[j][i]` accumulation, same `warp_reduce_sum`, `nwarps` stays 1 so no
shared-memory combine -- is untouched, only the row-to-wave assignment and the grid change. Host side:
`calc_launch_params(..., width_invariant, invariant_rpb)` and `mmvq_invariant_rows()` (override API
`ggml_backend_cuda_set_mmvq_invariant_rows_override`, then `LUCE_MMVQ_INVARIANT_ROWS`, then the default 2; only when
the device table runs one wave per row and `nrows % rows == 0`, else 1). T=1 and every non-Q8_0 or non-invariant
launch are not in this branch. `test_mmvq_invariant_rows_parity` (HIP-gated like the other GPU unit tests): the
eleven qwen4exp dense Q8_0 shapes plus 1002- and 1001-row fallbacks, T = 2, 3, 4, 5, 8, rows 1 / 2 / 4, every column
memcmp-equal to its own one-column MUL_MAT (invariance off, the T=1 kernel): 165 cases PASS on lucebox4.
`test_qwen4exp_mtp` (4236 cases) and `smoke_qwen4exp_forward --mtp` (tokens / logits / rollback / hidden decode
identical) pass on the final library.

Per launch at T=5 (median us), rows 1 / 2 / 4: attn_qkv 178.5 / **144.4** / 162.3, attn_q 218.4 / **173.1** / 202.7,
attn_gate 130.7 / **124.0** / 147.5, ssm_out / attn_output 137.2 / **130.9** / 146.6, lm head 4336 / **3289** / 4391,
hc_up 36.1 / 33.6 / 32.7, hc_down 45.1 / 44.6 / 50.8, shexp gate / up 15.3 / 14.7 / 17.2, shexp down 16.0 / 14.8 /
14.3, attn_k / v 13.6 / 13.6 / 15.7. Q8_0 class per T=5 call: 35.2 / **31.4** / 35.8 ms. Two rows win on every
shape (-3.8 ms per call, -1.05 ms of it the lm head); four rows compile to 80 VGPRs (32 at two), which caps the
occupancy at 6 waves per SIMD instead of 16, and the 320-640-row projections are left with 80-160 waves: the large
launches give the gain back and the small ones lose. Default: 2.

### bench_verify_realtok (8K ids, 20 reps, max_ctx 12000, median ms per call; arms rows=1, 2, 4, 4, 2, 1)

| k | rows=1 | rows=2 | rows=4 | rows=4 | rows=2 | rows=1 |
|---|---|---|---|---|---|---|
| 1 (decode) | 40.11 | 41.95 | 40.09 | 41.79 | 39.96 | 39.91 |
| 2 | 55.29 | 58.03 | 57.72 | 62.18 | 54.64 | 55.06 |
| 3 | 65.97 | 68.80 | 67.47 | 74.22 | 63.81 | 65.78 |
| 4 | 77.37 | 74.94 | 78.65 | 80.36 | 74.26 | 76.93 |
| 5 | 89.51 | 86.01 | 89.03 | 90.63 | 85.04 | 89.89 |
| 8 | 122.94 | 119.12 | 121.88 | 124.22 | 117.36 | 123.03 |

rows=2 vs rows=1 at k=4 / 5 / 8: -2.6 / -4.2 / -4.8 ms (both rows=2 runs below both rows=1 runs); k=2 / 3 inside
the run spread (the Q8_0 class at T=2..3 is ~1/3 of the T=5 one). `LUCE_PRINT_ARGMAX=1` sums identical on all six
runs at every k (k=1 16049, k=2 25088, k=3 22992, k=4 66549, k=5 56401, k=8 99096; k=1..4 and 8 equal the route-union
table above). The k=1 decode path does not enter the branch (ncols_dst = 1).

### e2e (mtp_ab.sh, 2 rounds x 5 reps, arms alternating, PREFILL=1, `densever/ab/`)

One binary (luce_server 9a720376 / ggml-hip 4e1b1748), `new` = default (two rows per wave), `old` =
`LUCE_MMVQ_INVARIANT_ROWS=1`, fixed k=4 arms the same, `ar` = `--verify-width 1 --no-draft`; `prefix_len=0` on every
request of every arm (server logs). Client decode tok/s, median of 10 (round-median spread in parentheses); server
`decode=` / acceptance / tokens per step / width histogram.

| arm | code16k | prose16k | counting |
|---|---|---|---|
| new, adaptive | **31.37** (1.0%) -- 30.28, 0.656, 2.03, k1 68% k2 16% k3 9% k4 6% | **31.24** (0.6%) -- 30.19, 0.680, 2.07, k1 60% k2 26% k3 13% | **64.72** (0.03%) -- 57.45, 0.995, 7.29, k7 74% k6 15% |
| old, adaptive | 31.09 (2.0%) -- 29.94, 0.665, 2.17, k1 53% k2 27% k3 12% k4 7% | 30.66 (2.2%) -- 29.68, 0.686, 2.03, k1 66% k2 21% k3 12% | 61.44 (0.1%) -- 54.59, 0.978, 7.29, k7 69% k6 17% |
| new, fixed k=4 | **28.36** (0.4%) -- 27.30, 0.457, 2.82 | **30.92** (0.2%) -- 29.77, 0.520, 3.07 | **57.01** (0.2%) -- 51.59, 0.995, 4.90 |
| old, fixed k=4 | 27.25 (0.2%) -- 26.22, 0.457, 2.82 | 29.62 (0.0%) -- 28.52, 0.522, 3.08 | 54.70 (0.0%) -- 49.47, 0.995, 4.90 |
| AR, no sidecar | 26.58 (0.3%) | 26.56 (0.3%) | 29.87 (0.3%) |

new vs old: fixed k=4 **+4.1% / +4.4% / +4.2%** (code16k / prose16k / counting; every delta > 2x both arms' round
spread, same acceptance and tokens per step on both), adaptive **+0.9% / +1.9% / +5.3%** (the adaptive arms' round
spread is 1-2% on the 16K prompts, so only counting and prose are clearly above it: the adaptive policy runs k=1 on
53-68% of the steps there, where the branch is not entered). Fixed k=4 on counting no longer crashes (73f55611).
Greedy-256 texts identical across new / old / fixed k=4 / AR on all three workloads. AR 26.58 / 26.56 / 29.87 and
prefill p4096 new 1337 (0.03%) vs old 1338 (0.08%), p16384 1389 (0.2%) vs 1391 (0.9%): untouched (neither enters the
branch). MTP vs AR now +18% code16k, +18% prose16k, +117% counting (adaptive).

What the adaptive policy picks now (`Qwen4ExpMtpWidth`): code16k k1 68% / k2 16% / k3 9% / k4 6% (2.03 tokens
per step at 0.656 acceptance), prose16k k1 60% / k2 26% / k3 13% (2.07 at 0.680), counting k7 74% / k6 15% (7.29 at
0.995). Cheaper wide windows nudge it toward k=1 even more on code16k (53% -> 68%) because the cycle-cost fit's base
also moved; the 16K prompts' 0.65-0.68 acceptance keeps the utility at narrow windows regardless.

Decision: shipped (default 2 rows per wave; `LUCE_MMVQ_INVARIANT_ROWS=1` restores one). What remained of the T=5
Q8_0 class was 31.4 ms per call vs 22.4 at T=1 on the same bytes; the activation stream per wave is the lever, taken
in the next section.

## Dense Q8_0 trunk at T=5: LDS activation staging (LUCE_MMVQ_INVARIANT_LDS, shipped)

Same box, binary luce_server 9a720376 / ggml-hip cd77718d (the tree at this commit; the change is library-only),
`bench_verify_realtok` relinked against it. Scratch on the box: `~/qwen4exp-f16/scratch/actstage/` (`trace.sh`,
`launches2.py` / `launches3.py`, `sweep.sh`, `mb.sh`, `chain.sh`, `ab/`, `sw1/`).

### Where the T=5 time went (after two rows per wave)

Per T=5 wave and 8-byte weight chunk the kernel loaded the weight once and then, for each of the five columns, the
q8_1 `qs` pair and `ds` -- the whole 5-column activation (14.5 KB for K=2560) streamed through L0/L1 again for every
weight-row pair. Measured in isolation (`bench/exact/mtp/bench_mmvq_invariant.cpp`: each shape as a T-column MUL_MAT
on the backend with invariance on, 16 weight copies rotated so every launch streams from DRAM past the 32 MB MALL,
kernel time under rocprofv3, median of 25; `mb.sh`), the T=5 launch with two rows per wave vs the T=1 launch:

| GEMV | T=1 us | T=5 rows 2 (previous section's default) | T=5 + LDS 16 (this section's default) |
|---|---|---|---|
| attn_qkv [2560 x 10240] | 121.5 | 143.2 | **126.4** |
| attn_gate [2560 x 6144] | 73.4 | 93.1 | **77.3** |
| ssm_out / attn_output [6144 x 2560] | 73.6 | 89.2 | **77.2** |
| attn_q [2560 x 12288] | 145.4 | 169.4 | **150.7** |
| attn_k / v [2560 x 512] | 4.0 | 9.3 | **6.4** |
| shexp down [640 x 2560] | 4.6 | 8.4 | **6.5** |
| shexp up [2560 x 640] | 3.4 | 9.0 | **6.1** |
| hc_up [320 x 10240] | 18.7 | 31.9 | **24.5** |
| hc_down [10240 x 320] | 20.3 | 43.1 | **26.7** |
| lm head [2560 x 248320] | 2957 | 3256 | **2964** |
| sum | 3422 | 3853 | **3466** |

VGPRs 16 / 32 / 72, no scratch. With the activation in LDS the large launches are back at the T=1 DRAM floor
(attn_qkv 220 GB/s at T=5) and the whole set is within 1.3% of the T=1 launches; the small projections (hc_up /
hc_down / shexp / attn_k) keep 2-6 us each of staging cost (a 16-wave block copies 10-14 KB before it starts) and
of too few waves (320-640 rows are 10-20 blocks).

### The change (`c_invariant_lds_waves`, LUCE_MMVQ_INVARIANT_LDS)

`mul_mat_vec_q` gained a second trailing template parameter: a block of W waves (each with its `c_invariant_rpb`
weight rows) stages the T columns' `block_q8_1` blocks in dynamic LDS, their 36-byte memory layout verbatim, and
`vec_dot_q8_0_q8_1_lds` takes the q8_1 ints and the `ds` scale from that copy instead of global memory --
the same `vec_dot_q8_0_q8_1_impl` chain on the same values, so each column's kbx sequence, per-lane accumulation and
`warp_reduce_sum` stay those of the one-column kernel (the bit-exactness contract of the previous section). Two
layouts on the device:

- Whole-K (`T * blocks_per_row * 36 <= 64 KiB`, every qwen4exp shape at T <= 8 except hc_down's K=10240 at T >= 6,
  no fused gate): the block stages the columns once, then its waves take weight-row groups from a block-local
  `atomicAdd` counter; block b owns row groups b, b + gridDim.x, b + 2 gridDim.x ..., so the launch sweeps the
  weight matrix front to back like the one-column kernel. (The first version gave each block a contiguous share;
  on attn_qkv a share is 256 rows = 696320 bytes, a multiple of 4 KiB, so all 40 blocks' streams ran at the same
  page phase and the launch cost 30% more: 176 vs 126 us.) The grid is `LUCE_MMVQ_INVARIANT_LDS_BLOCKS` persistent blocks (default 2 x the
  multiprocessor count, 40 on gfx1151) when the matrix has at most `MMVQ_INVARIANT_LDS_PERSIST_MAX_ROWS` (32768)
  rows -- the 2560-12288-row projections gain 5-10% over a block per 16 row groups -- and a block per
  `W * rpb` rows above it: the 248320-row lm head loses 4% to the per-group dispensing (3120 vs 2990 us) and runs
  7760 blocks of 16 waves.
- Chunked (otherwise, and with the fused gate): `MMVQ_INVARIANT_LDS_CHUNK` = 128 blocks per column at a time, two
  `__syncthreads` per chunk, one row group per wave, the kernel's usual epilogue.

Host side: `mmvq_invariant_lds_waves()` (override API `ggml_backend_cuda_set_mmvq_invariant_lds_override`, then
`LUCE_MMVQ_INVARIANT_LDS`, then the default 16; `0` is the kill switch; W in 4 / 8 / 16 / 32 with rows 2, 8 with
rows 1, 16 with rows 4; only when the device table runs one wave per row and `nrows % (rows * W) == 0`, else the
per-wave stream of the previous section), `calc_launch_params(..., invariant_rpb, invariant_lds_waves)` (blocks
of `warp_size x W`), `mul_mat_vec_q_switch_fusion` passes the dynamic LDS size and, on the whole-K persistent
launch, the row count in `ids_stride` (ids is null there). Instantiated (rows, W): (1,0) (2,0) (4,0) (1,8) (2,4)
(2,8) (2,16) (2,32) (4,16). T=1 and every non-Q8_0 or non-invariant launch are not in this branch.

`test_mmvq_invariant_rows_parity` now runs rows 1 / 2 / 4 x LDS 0 / 4 / 8 / 16 / 32 on the eleven qwen4exp dense
shapes, the 1002 / 1001-row fallbacks and a 40960-row shape (above the persistent limit), T = 2, 3, 4, 5, 8: 900
cases, every column memcmp-equal to its own one-column MUL_MAT, PASS on lucebox4. `test_qwen4exp_mtp` (4236 cases)
and `smoke_qwen4exp_forward --mtp` (tokens / logits / rollback / hidden decode identical) pass on the final library.

### In the model (eager verify at k=5, rocprofv3, 5 calls, Q8_0 class per T=5 call)

Same session, final library, `trace.sh 5 5` with the default and with `LUCE_MMVQ_INVARIANT_LDS=0`, per launch
median us. The persistent launches share one grid, so the default arm's mid-size launches are matched by their
position in the forward and split by rank into the known per-forward counts (`launches3.py`); the LDS 0 arm's grids
separate the shapes (its 15.9 MiB class splits 35 launches at 133 and 13 at 95).

| GEMV | n per call | LDS 0 | LDS 16 |
|---|---|---|---|
| attn_q | 12 | 173.5 | **154.0** |
| attn_qkv | 36 | 143.7 | **128.9** |
| attn_gate / ssm_out / attn_output (15.9 MiB) | 84 | 124.9 / 132.9 / 94.5 | **115.2** (82-128) |
| hc_up | 96 | 33.4 | **24.8** |
| hc_down | 97 | 44.5 | **28.8** |
| shexp up | 96 | 14.9 | **11.9** |
| shexp down | 48 | 14.8 | **12.8** |
| attn_k / v | 24 | 13.5 | **10.7** |
| lm head | 1 | 3278 | **2972** |
| Q8_0 class per T=5 call | | 31.41 ms | **26.77 ms** |

Verify k=5 under the tracer (5 eager calls) 92.8 -> 88.5 ms. The 15.9 MiB class runs 25-40 us above its standalone
time on both arms (attn_gate 124.9 in the model vs 93.1 standalone at LDS 0, 115 vs 77 at LDS 16) while attn_qkv
and attn_q sit at their standalone times; rocm-smi sampled during the k=5 and the k=1 runs shows sclk 970-1160 MHz
on both, so it is not a verify-specific clock state. Not pursued here: that class is 84 launches x ~38 us = 3.2 ms
of the 4.4 ms the T=5 class still has over the 22.4 ms of the T=1 launches in the model.

### bench_verify_realtok (8K ids, 20 reps, max_ctx 12000, median ms per call; arms LDS 0, 16, 16, 0, `sw1/`)

| k | LDS 0 | LDS 16 | LDS 16 | LDS 0 |
|---|---|---|---|---|
| 1 (decode, not in the branch) | 41.82 | 39.91 | 39.96 | 41.85 |
| 2 | 57.83 | 53.97 | 54.08 | 57.84 |
| 3 | 68.79 | 62.17 | 61.91 | 69.33 |
| 4 | 74.78 | 70.40 | 70.75 | 74.65 |
| 5 | 85.78 | 79.28 | 79.35 | 85.43 |
| 8 | 118.85 | 104.59 | 104.57 | 120.74 |

LDS 16 vs 0: k=2 / 3 / 4 / 5 / 8 **-3.8 / -7.0 / -4.1 / -6.3 / -15.2 ms** per call (means of the pairs) (both LDS 16 runs below both
LDS 0 runs at every k >= 2). `LUCE_PRINT_ARGMAX=1` sums identical on all four runs at every k (16049 / 25088 /
22992 / 66549 / 56401 / 99096, the previous section's values). A separate k=1-only run (arms 16 / 0 / 16 / 0, `sw_k1/`) gave 39.99 / 41.34 / 41.78 / 43.69 ms: the
decode medians drift with the session, not with the arm (the T=1 launch is not in the branch).

### e2e (mtp_ab.sh, 2 rounds x 5 reps, arms alternating, PREFILL=1, `actstage/ab/`)

One binary (luce_server 9a720376 / ggml-hip cd77718d), `new` = default (LDS 16, two rows per wave), `old` =
`LUCE_MMVQ_INVARIANT_LDS=0` (the previous section's default), fixed k=4 arms the same, `ar` = `--verify-width 1
--no-draft`; `prefix_len=0` on all 610 requests (server logs). Client decode tok/s, median of 10 (round-median spread
in parentheses); server `decode=` / acceptance / tokens per step / width histogram.

| arm | code16k | prose16k | counting |
|---|---|---|---|
| new, adaptive | **31.94** (0.4%) -- 30.86, 0.658, 2.00, k1 70% k2 15% k3 9% k4 5% | 31.37 (0.3%) -- 30.44, 0.689, 1.92, k1 80% k2 8% k3 11% | **72.46** (0.05%) -- 64.12, 0.995, 7.50, k7 76% k6 12% |
| old, adaptive | 31.27 (0.5%) -- 30.18, 0.657, 2.04, k1 67% k2 17% k3 9% k4 6% | 31.84 (3.0%) -- 30.75, 0.684, 2.07, k1 60% k2 25% k3 13% | 64.74 (0.03%) -- 57.25, 0.995, 7.29, k7 74% k6 15% |
| new, fixed k=4 | **30.94** (3.3%) -- 29.73, 0.457, 2.82 | **33.22** (1.2%) -- 31.96, 0.520, 3.07 | **60.84** (0.1%) -- 55.09, 0.995, 4.90 |
| old, fixed k=4 | 28.38 (0.4%) -- 27.31, 0.457, 2.82 | 30.87 (0.2%) -- 29.73, 0.520, 3.07 | 57.03 (0.04%) -- 51.61, 0.995, 4.90 |
| AR, no sidecar | 26.54 (0.3%) | 26.56 (0.3%) | 29.79 (0.4%) |

new vs old: fixed k=4 **+9.0% / +7.6% / +6.7%** (code16k / prose16k / counting; same acceptance and tokens per step
on both arms; code16k's new arm has a 3.3% round spread, its slower round 30.3 is still 1.9 above old's best round),
adaptive **+2.1% / -1.5% / +11.9%**. Prose16k adaptive is the policy, not the kernel: the new arm's
`Qwen4ExpMtpWidth` picked k=1 on 80% of the steps (1.92 tokens per step) against 60% on the old arm (2.07), the
old arm's rounds spread 3.0% (32.3 / 31.4) and fixed k=4 on the same prompt is +7.6%; the kernel is not entered at
k=1. Greedy-256 texts identical across new / old / fixed k=4 / AR on all three workloads. AR 26.54 / 26.56 / 29.79
(previous section 26.58 / 26.56 / 29.87) and prefill p4096 new 1311 (4.3%) vs old 1296 (4.7%), p16384 1382 (1.8%)
vs 1364 (2.4%): untouched (neither enters the branch). MTP vs AR now +20% code16k, +18% prose16k, +143% counting
(adaptive); fixed k=4 +17% / +25% / +104%.

What the adaptive policy picks now: code16k k1 70% / k2 15% / k3 9% / k4 5% / k5 1% (2.00 tokens per step at 0.658
acceptance), prose16k k1 80% / k2 8% / k3 11% / k4 2% (1.92 at 0.689), counting k7 76% / k6 12% / k4 6% (7.50 at
0.995). The 16K prompts' 0.66-0.69 acceptance keeps it at k=1 most of the time whatever the wide windows cost; with
fixed k=4 now above adaptive on prose16k (33.2 vs 31.4) and the wide windows 4-14 ms cheaper, the cycle-cost fit
that feeds the policy is the next thing to look at, not the kernels.

Decision: shipped (default LDS 16 waves per block, two rows per wave; `LUCE_MMVQ_INVARIANT_LDS=0` restores the
per-wave activation stream, `LUCE_MMVQ_INVARIANT_ROWS=1 LUCE_MMVQ_INVARIANT_LDS=0` the kernel of 0a6a47a7's parent).
The T=5 Q8_0 class is now 1.3% above the sum of the T=1 launches on the same bytes standalone (3466 vs 3422 us) and 4.4 ms above in the model (26.8 vs 22.4 ms per call, the 15.9 MiB class above); what is left in the
verify call is the per-row attention sub-graphs and the routed experts (items 1-2 at the top of this file).
