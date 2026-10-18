# Selector v2 on the GPU: design

Status 2026-10-18: in the server behind `LUCE_KVFLASH_SCORER_DEVICE=gpu` (2f1247c0, module `src/common/kvflash_gpu_mass.h`).
End to end on the 3090 it is a WIN and passes the cost line: scorer median 80.9 ms per reselect, prefill 888 ms against on6ref's 814
(`evidence/agentic/selv2/gpu/e2e/README.md`).

## Why

Selector v2 won end to end (`evidence/agentic/selv2/e2e/README.md`, same binary and session as on6ref):

- beyond-pool answers: 138 -> 176/207 (+18.4 [+10.9, +25.6] pts)
- within-pool answers: 709 -> 712/744
- overall: 847 -> 888/951

It failed only the cost line. The AVX2 CPU scorer takes a median 1720 ms per reselect; the line is a 200 ms median. Prefill went from 848 to 2190 ms and client wall from 1.40 to 2.81 s, against on6ref.

The work is exact chunk mass for 64 queries x 384 (layer, q-head) planes over every sealed key:

- softmax over all sealed keys at scale 1/16, summed per 64-token chunk;
- per (query, chunk): mean over planes, mean of the top 8 planes, max plane (`KvFlashMassPlanes`);
- then `kvflash_select_scores` and `kvflash_recall_pick`, unchanged.

At the E2E median (514 sealed chunks, 33K tokens) that is 207 GFLOP of q·k. The CPU cannot do it in 200 ms. The GPU can, and it already holds 254 of the 514 chunks' K in the pool.

## Design (step 1: stock ggml ops, what the bench measures)

One side graph per reselect on the target backend (the gallocr pattern of `flashprefill_q8.cpp`):

1. **Keys.**
   - Resident chunks are read in place: the pool tensors `cache_.attn_k[l]` ([D, pool, Hkv], q4_0) are graph leaves.
   - Evicted chunks are memcpy'd in parallel, from the pager's pinned per-chunk host copies (`host_k_seg`), into one pinned staging buffer [D, E·64, Hkv, L].
   - The staging buffer goes up in one H2D.
2. **Per layer:**
   - `mul_mat(pool_k[l], q)` and `mul_mat(staging[l], q)`, with q laid out as [D, G·N, Hkv] so each KV head serves its 6 q-heads;
   - `concat` the two products;
   - `add` a [T] bias: 0 on blocks holding a chunk < sealed, -inf on free blocks and on blocks holding chunks >= sealed, i.e. this turn's first pass;
   - `soft_max_ext` at scale 1/sqrt(D);
   - `sum_rows` over [64, chunks];
   - `cpy` into the output [384 planes, chunks, N].
3. **Back on the host.** One D2H of the output. Then, in parallel over queries:
   - mean of the 384 planes;
   - top 8 via `nth_element`;
   - the max plane.

   These fill `KvFlashMassPlanes`; selection is the existing code.

Everything is stock ops. The same code runs on CUDA, HIP and Vulkan, and on the ggml CPU backend for unit tests.

### Measured (bench_kvflash_gpu_mass, 3090, 64 queries, median of 8-10 reps)

Raw output is in `evidence/agentic/selv2/gpu/bench.out`. The server's K is q4_0 (its scorer pool copy is 151 MB = 144 B rows).

**q4_0 K (the server's type).**

| sealed chunks | where in E2E | gather | H2D (MB) | graph | D2H | host agg | **total** | first call | scratch | CPU oracle |
|---|---|---|---|---|---|---|---|---|---|---|
| 514 | median | 8.6 | 11.8 (153) | 53.2 | 4.4 | 15.7 | **96.7** | 297 | 637 MB | 1945 |
| 576 | max | 10.7 | 14.7 (190) | 61.7 | 5.2 | 18.1 | **113.2** | 293 | 728 MB | 2205 |
| 816 | max-ctx 52224 | 18.1 | 25.3 (331) | 88.8 | 7.8 | 23.5 | **167.6** | 417 | 1082 MB | 3347 |

All times are ms.

**Other configurations at 514 sealed** (all times ms):

- 16 queries, q4_0 K: total 41.5 against 358 on the CPU.
- q8_0 K: 111.8 (H2D 290 MB).
- q8_0 K at 1024 sealed: 259.5, scratch 1.8 GB.

**Parity against the CPU oracle.** Every row picks the same 30 units in the same order. Select-score max relative error is 0.008 for q4_0 and 0.003 for q8_0. Both scorers recover the same planted units (29 of 30).

**Projection onto the E2E run.** `predict_gpu.py` substitutes the bench cost per request, using each request's own sealed count, in the v2 arm's logs (`predict_gpu.out`):

| | median scorer per reselect | prefill (ms) | client wall (s) |
|---|---|---|---|
| on6ref | 31 | 848 | 1.40 |
| v2, CPU scorer | 1729 | 2190 | 2.81 |
| v2, GPU scorer (projected) | 98.6 (max 113) | ~941 | ~1.47 |

- Decode is untouched: 43.8 tok/s against 42.9 for on6ref.
- The cost line passes with 2x margin.

**VRAM** (`vram_probe.out`). The v2 server peaks at 19.6 GB of 24.6 GB on a 43.9K pooled prefill, which leaves 4.8-5.0 GB free. The scorer's working set at max-ctx 52224 is under 1.5 GB:

- scratch 1.08 GB;
- staging 0.33 GB;
- output 0.08 GB;
- q 0.03 GB.

**Where the graph time goes** (nsys, q8_0, 514, shares of kernel time):

- softmax 39%, bias add 17%, concat 15%, sum_rows 10%;
- the two matmuls with their dequant/convert: 19%.

The graph is memory-bound on the logits passes. Each pass moves 3.2 GB (32896 keys x 1536 rows x 16 layers x 4 B), about 26 GB in all, so its floor at 936 GB/s is about 28 ms against 53 measured. The staging matmul runs on cuBLAS f16 (dequant -> f16 gemm -> convert) instead of MMQ.

**Rejected variants:**

- f32 math: graph 381 ms.
- One K tensor plus mask instead of concat: 64 per-head uploads cost more than the graph saved (+16 / -4 ms).

## Implementation plan

1. **`server/src/common/kvflash_gpu_mass.{h,cpp}`.** Entry point:

   `bool kvflash_gpu_mass_planes(backend, pool K tensors, block_of, host_k_seg, sealed, qs, planes&, timing*)`

   - It holds its ggml context, gallocr and pinned staging buffer across calls, grows them on demand, and frees them on backend shutdown.
   - It calls `synchronize_paging()` before it reads the pool, as the CPU path does.
   - Staging is one 3D tensor per layer, so both matmuls take MMQ; measure it.
2. **Unit test on the ggml CPU backend** against `kvflash_exact_mass_scores_fast_multi`, with no GPU in ctest. The synthetic generator moves out of the bench into a shared test header. Cases:
   - q4_0 and q8_0;
   - all resident;
   - all evicted but sinks;
   - fewer than 64 queries;
   - pool blocks holding chunks >= sealed (must get -inf; the bench never has them);
   - free blocks.

   Pass: same picks, select relative error < 1e-2.
3. **Server.**
   - `LUCE_KVFLASH_SCORER_DEVICE=gpu` (default cpu) routes `kvflash_prefill_score` to the GPU path when selector v2 is on. Any failure (allocation, a free-VRAM check, a graph error) logs once and falls back to the CPU scorer.
   - Check free memory before the first allocation. WSL oversubscribes VRAM into shared memory without an error, so a too-large scratch would show up as slow, not as OOM.
   - Log line: `[kvflash] gpu scorer: n_q sealed resident | gather h2d graph d2h agg total ms`.
4. **Shadow arm on the 3090.** `LUCE_KVFLASH_SCORER_SHADOW=1` with the GPU device runs both scorers and logs pick overlap and max relative error per reselect. It runs on the smoke ids.
5. **Preregistered E2E arm.** v2+gpu against on6ref: same binary, same session, same harness, the same WIN rule as `PREREG_e2e.md`, and the cost line. Predictions:
   - scorer median 90-140 ms;
   - prefill 880-1050 ms (on6ref 848);
   - wall 1.42-1.60 s;
   - decode within 3%;
   - beyond 168-184/207.

## Later, only if a measurement asks for it

- **First-call cost (~200 ms once per server process).** gallocr reserve, cuBLAS and kernel load. Fix: pre-warm at the first pooled prefill.
- **Scratch grows with context x queries** (3.2 GB at 128K, 64 queries). Fix: tile the queries so scratch fits the free-VRAM check. 16 queries cost 41 ms at 514 sealed.
- **Fused mass kernel (CUDA/HIP).**
  - One pass over K with int8 MMA.
  - Per (row, chunk) it keeps the chunk-local max and exp-sum, then combines them exactly: mass_c = s_c·e^(m_c−M) / Σ s_c'·e^(m_c'−M).
  - Top-8/max aggregation on device.
  - Expected: graph 53 -> ~5-10 ms, D2H 50 MB -> 0.3 MB, no host aggregation, scratch ~MBs; about 35 ms total at the median.
  - Worth it at 128K contexts or on 220 GB/s boxes (lucebox4), where the stock graph's ~26 GB of logits traffic alone is over 100 ms.
- **Overlapping the evicted-K gather+H2D (20 ms) with the first prefill pass.** Evicted K never changes. A device mirror of evicted K would remove the copy but makes VRAM O(context), against KVFlash's O(pool) contract, so it is not the default.

## Reproduce

```
cd server
g++ -std=c++17 -O2 -march=native -Wall -Wextra -I src/common -I src -I deps/llama.cpp/ggml/include \
  test/bench/bench_kvflash_gpu_mass.cpp <selv2-cuda build>/deps/llama.cpp/ggml/src/ggml-cuda/libggml-cuda.so \
  <build>/deps/llama.cpp/ggml/src/libggml.so <build>/deps/llama.cpp/ggml/src/libggml-base.so \
  -Wl,-rpath,<build>/deps/llama.cpp/ggml/src/ggml-cuda:<build>/deps/llama.cpp/ggml/src -lpthread -o <bench>
bash ../optimizations/kvflash/evidence/agentic/selv2/gpu/run_bench.sh <bench>          # 3090 lock per run
python3 -I ../optimizations/kvflash/evidence/agentic/selv2/gpu/predict_gpu.py /mnt/kimi-k3/split-work/selv2/e2e
```
