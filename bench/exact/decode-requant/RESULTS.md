# Decode-only requantized copies (lucebox4 gfx1151, 2026-10-15)

Opt-in, lossy, off by default. `LUCE_DECODE_REQ=<type>` (GDN layers; also the attention layers unless
`LUCE_DECODE_REQ_ATTN=<type|0>` overrides). Commits 77317f11, b3182d8a, 2c2daad2 and the allocation-order fix below on qwen4exp-exact-stack.

## What gufo does (GUFO_DECODE_Q5, default on)

device_model.cpp:553-590: every Q8_0 tensor named `ssm_in`, `ssm_out`, `attn_qkv`, `attn_out` with
>= 2560*6144 elements gets a Q5_1 copy (`RequantQ8_0ToQ5_1`, ggml's reference Q5_1: 32 levels over the block
[min,max]) that `Executor::Dense` reads when `q.n <= 8`. Logs "decode Q5_1 copies: 1.87 GiB".

Ours (ggml [ne0=K, ne1=rows]): the 36 GDN layers' `attn_qkv` [2560x10240], `attn_gate` [2560x6144],
`ssm_out` [6144x2560]; the 12 attention layers' `attn_q` ([q|gate], [2560x12288]), `attn_k` / `attn_v`
[2560x512], `attn_output` [6144x2560]. 156 tensors, 2.674 G elements = 2.65 GiB Q8_0, 2.84 GB of the ~6.3 GB a
decode token streams. hc_up / hc_down (320-wide) and the experts stay Q8_0 / Q4_K.

Bytes saved per token and the ceiling at 220 GB/s (the 34.97 ms exact token):

| copy type | B/elem | copies | GB/token saved | ceiling |
|---|---|---|---|---|
| Q8_0 (control) | 1.0625 | 2.65 GiB | 0 | 0 |
| Q6_K | 0.8203 | 2.05 GiB | 0.650 | 2.95 ms |
| Q5_1 (gufo) | 0.75 | 1.87 GiB | 0.836 | 3.80 ms |
| Q5_K GDN + Q6_K attn | - | 1.79 GiB | 0.923 | 4.20 ms |
| Q5_K / Q5_0 | 0.6875 | 1.71 GiB | 1.003 | 4.56 ms |
| Q4_K | 0.5625 | 1.40 GiB | 1.340 | 6.09 ms |

gufo realises 3.51 of its 3.80 ms (33.87 -> 30.36 ms, GUFO_DECODE_Q5=0 vs default, same session).

## How it is built

Load time, host threads: each Q8_0 source is dequantized row-chunk-wise and re-quantized with
`ggml_quantize_chunk` (ggml's reference quantizer) into `<name>.dec` in its own backend buffer
(`decode_req_ctx` / `decode_req_buf`); one upload per tensor; ~10 s on 16 threads. Nothing on disk.
The graph reads the copy only at T == 1 (`dec()` at the 9 mm sites of build_linear_attn / build_full_attn);
prefill, T > 1, the batched builder and reference mode read the GGUF tensors. The five Q8_0-or-Q6_K
allowlists the T=1 fusions keep (producer_q8_reserve, gdn_q8_match, hc_upmix_row8_consumer,
hc_upmix_row8_pair_quick_valid, the mmvq fixed-q8 producer assert) go through
`ggml_cuda_dense_decode_weight_ok`; all T=1 fusions stay engaged on the copies (96 HC up-mix sites,
48 shared-epilogue sites captured on every arm). Unit test `test_qwen4exp_decode_requant` (LUCE_TESTS):
env parser, byte parity of the requant with `ggml_quantize_chunk`, block-size rejects, [2560 x 64] GEMV on
each copy type vs the F64 product of its own dequantized rows on CPU and HIP.

## Quality gate (teacher-forced, 2399 positions)

`gate_all.sh` / `qgate.sh` / `teacher_client.py` / `kl_gate.py`: four 3600-char code slices as the
*assistant* turn (in the user turn the model's top-1 is `<|im_end|>` at most positions), the last 600 tokens of
each forced through the T=1 decode path (`LUCE_TEACHER_TAIL=600 LUCE_TEACHER_DROP=7`), per-step logits
dumped as F16 (`LUCE_TEACHER_LOGITS`), scored against the exact Q8_0 dump. Determinism floor: a second Q8_0
run gives KL exactly 0. Baseline next-token accuracy 93.16 %, NLL 0.2765.

| arm (gdn:attn) | GB/token | KL mean | KL median | KL p99 | KL max | top-1 | top-5 | NLL |
|---|---|---|---|---|---|---|---|---|
| q6_k:0 | 0.503 | 0.00618 | 0.00028 | 0.104 | 0.502 | 98.87 % | 92.09 % | 0.2777 |
| q6_k | 0.650 | 0.00728 | 0.00035 | 0.116 | 0.903 | 98.96 % | 90.88 % | 0.2777 |
| q5_1:0 | 0.649 | 0.00945 | 0.00040 | 0.155 | 0.598 | 98.67 % | 90.04 % | 0.2866 |
| q5_k:0 | 0.779 | 0.00882 | 0.00036 | 0.137 | 0.629 | 98.21 % | 89.96 % | 0.2821 |
| 0:q5_k (attn only) | 0.224 | 0.00820 | 0.00031 | 0.143 | 0.486 | 98.54 % | 90.51 % | 0.2843 |
| q5_1:q6_k | 0.794 | 0.01118 | 0.00043 | 0.149 | 1.037 | 98.67 % | 89.62 % | 0.2893 |
| **q5_1 (gufo's choice)** | 0.836 | 0.01358 | 0.00054 | 0.237 | 1.148 | 98.33 % | 88.46 % | 0.2853 |
| **q5_k:q6_k (chosen)** | 0.923 | 0.00997 | 0.00047 | 0.160 | 0.854 | 98.25 % | 89.55 % | 0.2808 |
| q5_k | 1.003 | 0.01296 | 0.00047 | 0.227 | 0.720 | 98.21 % | 88.78 % | 0.2911 |
| q5_0 | 1.003 | 0.01833 | 0.00070 | 0.271 | 1.231 | 98.12 % | 87.26 % | 0.2871 |
| q4_k | 1.340 | 0.02931 | 0.00123 | 0.518 | 1.769 | 96.71 % | 83.55 % | 0.2874 |

The attention layers are ~3.5x as KL-sensitive per byte as the GDN layers (0.224 GB -> KL 0.0082 vs 0.779 GB ->
0.0088), hence the per-class knob. Q5_K GDN + Q6_K attention saves 10 % more bytes than gufo's Q5_1 at 27 %
less KL and a better NLL; Q5_K everywhere saves the most but has the worst NLL of the 5-bit arms.

## Speed (one binary, arms alternating across restarts, prefix_len=0 on luce, cache=miss on gufo)

Main run `ab/`, 2 rounds x 5 reps, binary fa83942c, model 1958631667 (no sudo on the box: no
drop_caches / compact_memory, warm-up dropped; VRAM counters are GTT-only on this APU and useless):

| arm | p4096 tok/s | p16384 tok/s | decode tok/s | ms/token | vs A |
|---|---|---|---|---|---|
| A luce exact | 1420.0 (rounds 1435.6 / 1411.4, spread 1.7 %) | 1438.1 (0.1 %) | 28.54 (0.3 %) | 35.04 | - |
| B luce q5_k:q6_k | 1374.5 (1388.1 / 1368.6, 1.4 %) | 1421.8 (0.0 %) | 30.81 (0.3 %) | 32.46 | p4096 -3.2 %, p16384 -1.1 %, **decode +8.0 %** |
| C gufo default (Q5_1) | 1604.8 | 1620.6 | 32.94 | 30.36 | +13.0 / +12.7 / +15.4 % |
| D gufo GUFO_DECODE_Q5=0 | - | - | 29.52 | 33.87 | +3.5 % |

C vs B: decode +6.9 % (was +15.4 % vs exact); prefill +16.8 / +14.0 % (unchanged story, gufo's prefill lead is
elsewhere). Greedy-64: A vs B diverge at char 86 (p4096) / 96 (d512), the lossy copies change the sampled
text as gufo's do; B vs C and A vs C diverge at char 0 (they always did).

Control `abq8/` (`LUCE_DECODE_REQ=q8_0`: same bytes, same graph, separate buffer; greedy-64 byte-identical on
both prompts), 2 rounds x 5, binary 43d080d1: decode 28.56 -> 28.02 (-1.9 %), p4096 1414.4 -> 1383.0 (-2.2 %),
p16384 1431.3 -> 1394.2 (-2.6 %). (Superseded by the allocation-order section below: it was when the buffer
was allocated, not that it was.) So ~2 % of every B arm, prefill and decode alike, is the extra allocation
(every buffer allocated after the weights, copies / KV / compute, lands on worse pages on this uncompacted
box; the memory note puts uncompacted decode ~5 % slow and we cannot run compact_memory here). Net of that
control the quant itself is 28.02 -> 30.81 = +10.0 % (3.2 of the 4.2 ms ceiling, 77 %; gufo gets 92 % of its
3.8 ms).

Single-round B-only speed arms (1 x 5, decode median; prefill from one round is within the ~3 % day-to-day
spread and is not read):

| arm | GB/token | decode tok/s | ms/token |
|---|---|---|---|
| q5_1 | 0.836 | 31.15 | 32.10 |
| q5_k | 1.003 | 31.16 | 32.10 |
| q5_1:q6_k | 0.794 | 30.75 | 32.52 |
| q5_k:q6_k | 0.923 | 30.81 | 32.46 |

Q5_K saves 20 % more bytes than Q5_1 for the same speed: the k-quant mmvq dot (Q5_K / Q6_K) costs more ALU per
byte than Q5_1's on gfx1151, and Q6_K attention costs ~0.35 ms against Q5_K / Q5_1 attention. A Q5_K / Q6_K GEMV
tuned for this part is the follow-up if the last ~1 ms matters.

RAM: +1.79 GiB device memory for the chosen arm (loader line `decode copies (gdn q5_K, attn q6_K): 156
tensors, 1.79 GiB`); VmRSS 1.25 -> 1.33-1.36 GB (host staging freed after upload).

## The control-arm cost: allocation order, not footprint (fixed)

The ~2 % the Q8_0 control paid above was blamed on "every buffer allocated after the weights lands on worse
pages". Three one-binary 2x5 runs (`ab2.sh`, arms alternating across restarts, warm-up dropped, prefix_len=0 on
every arm, greedy-64 byte-identical on every arm) pin it down. Decode is the only axis with signal: its round
spread is 0.01-0.27 %, prefill's is 0.9-3.5 % on this box, so every prefill delta below is inside its own noise.

1. Not the prefill chunk. The chunk auto-sizer (`qwen4exp_select_chunk`) reads the free GTT (96 GiB
   `amdgpu.gttsize` cap, weights 76.2 GiB) and the copies drop it 20.2 -> 17.3 GB, so the control got chunk
   13056 against 15616 for exact. `chunk/` (binary 43d080d1): exact `--chunk 13056` is 28.55 vs 28.59 tok/s
   (-0.13 %), p4096 -1.3 %, p16384 +0.5 %; the control with `--chunk 15616` is still 28.03 (-1.95 %), same as
   the control at 13056 (28.03). Chunk is not it.
2. Not GTT page quality as such. `pages/pagebw.hip` (hipMalloc 76 GiB, mmap-read the three shards through the
   page cache, then allocate 2.65 + 1 + 2 GiB) reads every late buffer at 241-244 GB/s streaming and 224-226 GB/s
   with an mmvq-like one-warp-per-2720-byte-row kernel, same as the early buffer and as the weights, on a
   fresh box and on a fragmented one (order-10 free blocks 5176 -> 0, GPUReclaim pool 62 GB -> 0).
3. Not the footprint. `place/` (binary 82cecfbb, the loader with two experiment knobs since removed): a 2.65 GiB
   device buffer allocated after the upload and never read (`PAD`) decodes at 28.60 vs 28.58 (+0.07 %).
4. It is when the copy buffer is allocated. The old loader allocated it after the 76 GiB of shards had streamed
   through the page cache; moving the allocation to right after the weight buffer (before any shard is mapped)
   makes the same-bytes control cost nothing: `Q8` 28.59 vs 28.58 (+0.03 %), and inside the weight buffer
   (`Q8I`, one contiguous allocation) 28.65 (+0.22 %). Old placement, three runs: 28.02 / 28.03 / 28.03 vs
   28.56 / 28.59 / 28.59 exact (-1.9 / -1.95 / -1.95 %). The micro-probe cannot reproduce the late-buffer penalty,
   so the hardware mechanism (page order / GPU TLB fragments is the candidate; `/sys/kernel/debug/dri` is root-
   only here) stays unproven; the effect itself is reproduced five times with the server.

Fix (`qwen4exp_loader.cpp`): the decode-copy tensors are selected and their buffer allocated right after the
weight buffer, before the shard mmaps; the requant + upload stays where it was (after the trunk upload, host
threads). The exact path is untouched (no env: no ctx, no buffer, same code order), greedy-64 byte-identical.
Separate buffer kept (the contiguous variant was no better and needs offset bookkeeping).

After the fix, `fix/` (binary 938bea37, 2x5, same session):

| arm | p4096 tok/s | p16384 tok/s | decode tok/s | ms/token | vs exact |
|---|---|---|---|---|---|
| exact | 1382.6 (rounds 1354.4 / 1402.2, spread 3.5 %) | 1388.9 (3.3 %) | 28.58 (0.08 %) | 34.99 | - |
| q8_0 control | 1405.0 (3.0 %) | 1425.9 (0.6 %) | 28.62 (0.19 %) | 34.94 | +1.6 / +2.7 / **+0.14 %** |
| q5_k:q6_k | 1344.8 (2.0 %) | 1429.8 (0.9 %) | 31.57 (0.23 %) | 31.68 | -2.7 / +3.0 / **+10.44 %** |

Before (old placement, `abq8/` and `ab/`): control -2.2 / -2.6 / -1.9 %; q5_k:q6_k -3.2 / -1.1 / +8.0 % (30.81
tok/s, 32.46 ms). The quant arm now shows its full +10.4 % (3.31 of the 4.20 ms ceiling, 79 %; gufo's Q5_1
realises 92 % of 3.80 ms) and its prefill deltas sit inside the 2-3.5 % round spread of the exact arm
(p16384 +3.0 % is noise: the exact arm's round 1 was the slow one, 1361.8 vs 1408.0).

## The last 0.9 ms: per-kernel ledger and the K-quant GEMV kernel search (negative result)

Question: the q5_k:q6_k arm realises 3.31 of the 4.20 ms byte ceiling; is the rest in the Q5_K / Q6_K
`mul_mat_vec_q` dot (the earlier reading: "k-quant mmvq costs more ALU per byte than Q5_1")?

### Ledger (rocprofv3 --kernel-trace on luce_server, `prof.sh` / `ktledger.py`, 120 steady decode tokens)

Both arms, same binary 938bea37, same session, profiler on (+9 % wall). Everything outside the copy GEMVs is
identical across arms (routed gate/up 5.0-5.1 ms, hc_* 5.1 ms, GDN 1.26 ms, ...); the delta is entirely the
156 copy GEMVs. Grid = rows x 32 work-items, one 32-lane wave per row (RDNA3.5 takes the RDNA2 mmvq table:
nwarps 1, rows_per_block 1), VGPR 16 (Q8_0) / 24 (Q6_K) / 32 (Q5_K), no LDS, no scratch:

| tensor (per token) | exact Q8_0 us/launch (GB/s) | requant us/launch (GB/s) | saved us/token |
|---|---|---|---|
| attn_qkv [2560 x 10240] x36 | 123.6 (225) | Q5_K 81.9 (220) | 1503 |
| attn_gate [2560 x 6144] x36 | 74.5 (224) | Q5_K 51.6 (210) | 825 |
| ssm_out [6144 x 2560] x36 + attn_output x12 | 76.3 (219) | Q5_K 53.6 (202) / Q6_K 62.3 (207) | 985 |
| attn_q [2560 x 12288] x12 | 147.4 (227) | Q6_K 115.5 (223) | 383 |
| attn_k, attn_v [2560 x 512] x24 | 8.5 (164) | Q6_K 8.9 (121) | -9 |
| total | 12.77 ms | 9.08 ms | 3.69 ms (wall 38.47 -> 34.94, 3.53 ms) |

The copies stream 1.92 GB per token; at the Q8_0 kernels' own rate (219-227 GB/s on the same shapes) they would
take 8.6-8.7 ms, so Q8_0-parity of the k-quant kernels is worth ~0.4 ms/token, the asymptotic 231 GB/s of this
kernel family ~0.5 ms, and the 243 GB/s streaming peak (pagebw.hip) ~1.2 ms. The k/v GEMVs are latency-bound
(512 waves) on both arms: 0.2 ms/token together, nothing to recover there.

### Micro-benchmark (`gemvbench.cpp` / `gemv.sh` / `ktsum.py`: 8 distinct copies > the 32 MiB MALL, rocprofv3 kernel medians)

Reproduces the server rates to 1-2 %: Q8_0 227-230, Q5_1 231, Q5_K 211-222, Q6_K 216-227 GB/s on the decode
shapes. Row-count sweep at K=2560: at 40960 rows Q8_0 / Q5_K / Q6_K all converge to **230.3 / 230.8 / 231.0
GB/s** (the dot costs nothing per byte asymptotically); at 2560 rows Q8_0 221 vs Q5_K 207; at K=6144 x 10240 rows
(195 us) Q5_K still 221 vs Q8_0 232. So the k-quant deficit is a per-launch / per-wave effect on the smaller and
longer-K launches (2560-row ssm_out / attn_output are 2 rounds of the 1280 resident waves), not ALU per byte.

Kernel search, `mmvq.cu mul_mat_vec_q_kq_dense<type, nwarps, unroll, rows_per_wave>` and
`mul_mat_vec_q_kq_dense_pipe<type>` (gfx1151, dense, ncols_dst 1, no ids / fusion; `LUCE_KQ_DENSE=w<n>u<u>r<r>`
or `pipe`), same lane -> block mapping and per-lane vec_dot as the generic kernel, so the one-wave variants are
bit-identical to it (unit test `test_qwen4exp_decode_requant`, every variant on [2560|6144 x 256], split-K
within the generic error). Median us per launch, generic first (`gemv/sweep.log` on lucebox4):

| variant | q5_K 2560x10240 | q5_K 2560x6144 | q5_K 6144x2560 | q6_K 2560x12288 | q6_K 6144x2560 |
|---|---|---|---|---|---|
| generic (legacy) | 81.2 | 49.9 | 51.6 | 113.9 | 59.7 |
| w1u1r1 (same shape, own kernel) | 81.0 | 49.9 | 52.1 | 112.9 | 59.4 |
| w1u2r1 (K loop unrolled 2) | 82.5 | 51.4 | 52.6 | 112.0 | 58.0 |
| w1u4r1 | 83.3 | 52.5 | 53.4 | 113.4 | 58.8 |
| w1u2r2 (2 rows per wave) | 84.8 | 55.1 | 56.0 | 114.0 | 59.8 |
| w2u1r1 (split-K, 2 waves per row) | 89.1 | 54.5 | 50.7 | 116.3 | 58.7 |
| w2u2r1 | 90.2 | 55.9 | 53.5 | 113.8 | 57.4 |
| w4u2r1 | 108.9 | 67.8 | 56.1 | 123.6 | 57.0 |
| w8u1r1 | 111.2 | 64.5 | 54.4 | 138.1 | 59.3 |
| pipe (next block's words loaded before this one's dot) | 86.6 | 52.4 | 54.1 | 114.4 | 60.1 |

Nothing beats the generic kernel beyond the 1-3 % run-to-run spread: more loads in flight per wave (unroll,
pipeline, 2 rows) is neutral to worse, split-K is worse (each wave's stream becomes strided), the only positive
cells are Q6_K unroll 2 at +1.5-3 % (~40 us/token, under the e2e decode noise). The one-wave-per-row generic
kernel is at the rate this memory system gives the pattern; the 0.9 ms is the per-launch ramp / tail of 156
small GEMVs per token (5-8 us each on every type: 6.96 MB Q8_0 at 2560 rows loses 1.3 us to its own asymptote,
4.5 MB Q5_K 2.3 us), which a vec_dot rewrite cannot touch. Below the task's 0.35 ms bar, so no e2e A/B was run
and the variants stay opt-in (unset = generic kernel, the production path is unchanged).

Follow-up if the 0.9 ms matters: cut launches, not bytes -- attn_qkv + attn_gate read the same `cur` in every
GDN layer and could be one launch (36 ramps/token, ~0.15-0.25 ms), or run on two streams; the same applies to
the exact arm.

## Recommendation

Keep it opt-in (lossy; the user decides). If enabled, `LUCE_DECODE_REQ=q5_k LUCE_DECODE_REQ_ATTN=q6_k`:
+10.4 % decode measured after the allocation-order fix (31.57 tok/s, 31.68 ms vs 34.99), KL 0.0100 / top-1
98.25 % / NLL 0.2808 vs 0.2765 exact, strictly better quality and more bytes saved than gufo's Q5_1, prefill
inside the day's spread. `LUCE_DECODE_REQ=q5_k` alone is the speed point (NLL 0.2911). The remaining cost is
1.79 GiB of device memory, which also lowers the auto prefill chunk on a GTT-capped box (15616 -> 13824 here,
measured as no cost).
