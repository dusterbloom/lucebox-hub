# Decode-only requantized copies (lucebox4 gfx1151, 2026-10-15)

Opt-in, lossy, off by default. `LUCE_DECODE_REQ=<type>` (GDN layers; also the attention layers unless
`LUCE_DECODE_REQ_ATTN=<type|0>` overrides). Commits 77317f11, b3182d8a, 2c2daad2 on qwen4exp-exact-stack.

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
p16384 1431.3 -> 1394.2 (-2.6 %). So ~2 % of every B arm, prefill and decode alike, is the extra allocation
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

## Recommendation

Keep it opt-in (lossy; the user decides). If enabled, `LUCE_DECODE_REQ=q5_k LUCE_DECODE_REQ_ATTN=q6_k`:
+8.0 % decode measured (+10 % net of the allocation penalty), KL 0.0100 / top-1 98.25 % / NLL 0.2808 vs
0.2765 exact, strictly better quality and more bytes saved than gufo's Q5_1. `LUCE_DECODE_REQ=q5_k` alone is the
speed point (+9.2 %, NLL 0.2911). The ~2 % allocation penalty on prefill is the one real cost; allocating the
copies after the KV / compute buffers (or a compacted box) is the thing to try before making anything default.
