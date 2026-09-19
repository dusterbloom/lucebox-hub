# qwen4exp quality & benchmark rigour — proposal (draft for discussion)

Status: draft for review, 2026-09-19. Scope: how we measure qwen4exp (and the
rest of lucebox) so numbers are reproducible and trustworthy, covering
generation correctness, agentic coding, the special tools (KVFlash / PFlash /
paged / spec), and decode + speculative-decoding SOTA.

## 1. How CIRU measures (ciru-ai/Qwen3.8-Flash-CIRU-STRIX-IU4)

CIRU's rigour has two pillars: a **frozen numerical-fidelity benchmark** and
**per-release qualification** with matched arms.

- **Fidelity** (`benchmarks/fidelity/`): 16 windows, each 2048 input tokens +
  the next-token label, score positions 1024-2047 → 16,384 full-vocabulary
  distributions, against a frozen BF16 reference (`Qwen/Qwen3.8-Flash-Next`,
  fixed revision). Primary metrics: **strict argmax agreement** and **forward
  KL(reference || candidate)** in nats. CIs from 10,000 bootstrap resamples over
  **eight repository clusters** (does not pretend positions are independent).
  The corpus, baseline numbers and hashes are committed; the run captures
  `result.json`, per-window results, runtime/env hashes and the raw
  `logits.f32le` for audit. They explicitly warn the % is corpus-specific and
  **not** comparable across corpora, and that lower KL ≠ higher agent score.
- **Qualification** (`docs/qualification/vX/`): matched control/candidate arms
  (e.g. stock/corrected/corrected/stock with four independent server loads),
  paired protocols, **draft acceptance counters** (`accepted/drafted`), a
  losslessness argument for spec decode, and workload-specific caveats. The
  agentic eval is **HermesAgent-20** (score/100, full-score tasks, 32-turn
  allowance, production sampling, native xhigh thinking), run twice per model.
- **Reported decode/spec numbers (with protocol):** v4.4 IU4 MTP depth 3 —
  **44.53 tok/s @12.96K**, 21.14 @245K cold, 60.35 short HE0-9. v4.3 Hermes-20
  pooled 34.74 tok/s, **68.6% acceptance**. v4.2 prompt 984-990, decode 21.2
  after the attention correction. MTP depth is now **3-4**, not 6 (depth 6 was
  16.9% slower).

Takeaways to copy: frozen corpus + committed baselines; cluster-bootstrap CIs;
`argmax` + `KL` as the primary fidelity metrics; matched arms with a control;
acceptance counters; hashes/env in an evidence bundle; refuse partial scores.

## 2. How lucebox measures (this repo)

- **`server/RESULTS.md`** (919 lines): the canonical ledger — one section per
  hardware/architecture with its own stated protocol and reproduction command
  (`scripts/bench_llm.py`, `:145`), spec-decode AL and acceptance, NIAH
  long-context configs, determinism claims.
- **`docs/specs/speed-profile.md`** (+ `.github/workflows/speed-profile.yml`):
  the CI speed-profiler contract — headline prefill/TTFT/decode/AL, per-step
  phases, nsys kernels, fixed defaults `--budget 22 --n-gen 128 --reps 5
  --noise-rsd-pct 0.05`, and a **losslessness gate** (greedy spec vs greedy AR,
  with an AR-vs-AR determinism control; FAIL only when the control matches and
  spec diverges). This is our best existing "trust" artifact.
- **`harness/`**: `client_test_runner.py bench` (math `\boxed{}`, GSM numeric,
  **HumanEval gold-test execution**, HTTP tool probes), `concurrency/`
  (canonical/ragged, C=1..16, DDTree acceptance/AL, published-run requirements),
  `clients/` (codex / claude_code / opencode / Open WebUI launchers + smoke
  probes), `qualification/deepseek4`.
- **`server/scripts/`**: `bench_llm.py` (AR vs DFlash + HE/GSM/Math),
  `bench_he.py`, `bench_agent.py` (2K/8K/24K agent buckets), `bench_agent_loop.py`
  (real Claude Code turns, cold vs warm prefix), `bench_server.py`,
  `bench_ds4_decode.py` (DSpark spec + accept_rate), `benchmark_tool_prefix_cache.py`
  (cold/warm tool-prefix cache identity contract), `profile.py` (CI profile +
  losslessness).
- **`server/eval/`**: `quality_ab_simple.py` (sameness A/B with a
  `baseline_2` determinism control), `quality_humaneval_plus.py` (EvalPlus
  pass@1 — real capability).
- **Special tools**: `optimizations/{pflash,kvflash,spark,megakernel,paged_attention}/{RESULTS,DESIGN}.md`;
  `server/test/test_kvflash*.cpp` (verification suites A-F, bit-exact
  page_out/page_in, reselect/recall); `bench_paged_attention.cpp` validates both
  layouts against a **double-precision CPU oracle before timing**.

## 3. Gaps (relative to CIRU)

1. **The qwen4exp correctness gate is not in-repo.** `planted.py` + `rep_hc.sh`
   live only in `/tmp` (referenced in the handoff). No committed gate.
2. **No numerical-fidelity benchmark** against a frozen reference for the
   hand-written qwen4exp graph — our strongest correctness claims are
   planted-fact + eyeballed NIAH.
3. **No soak** (staggered admission / slot reuse / cancellation).
4. **No unified entry point** — no `make bench`, each suite is a separate script.
5. **MTP is RED-phase** (`test_mtp_{e2e,converter}.sh` expected to fail), and
   there is no green decode/spec benchmark on qwen4exp.
6. **No agentic task-success scoring** — `bench_agent` measures TTFT/tok/s +
   regex, not resolution (explicitly narrower than SWE-bench).
7. **Special-tool benches are manual**; only `speed-profile.yml` is a CI gate.

## 4. Proposal — measure something others can trust

### Tier 0 — correctness & fidelity (the trust anchor)
- **Commit the correctness gates** for qwen4exp into `server/test/`:
  `test_qwen4exp_planted.cpp` (or a Python runner) + a determinism control
  (same prompt twice; compare hashes), and a qwen4exp NIAH gate. Removes the
  `/tmp` dependency.
- **Build a frozen fidelity harness** (port of CIRU's idea): a committed corpus
  of windows, a frozen reference = logits captured from a known-good engine on
  the same windows (e.g. `strix-ref` llama.cpp or our own dense path), committed
  baseline + hashes, and a scorer that reports **strict argmax agreement +
  forward KL** with **cluster-bootstrap CIs**. Capture raw logits for audit.
  This is the single most credible artifact we can publish.

### Tier 1 — generation & agentic
- Add an **agentic success metric** alongside `bench_agent`: wire
  **HermesAgent-20** (or an equivalent tool/apply-patch suite) into `harness/`,
  reporting pass rate, full-score tasks, decode tok/s and acceptance.
- Keep `bench_agent_loop.py` (real multi-turn Claude Code turns) as the
  latency/prefix-cache counterpart.

### Tier 2 — special tools
- One runner per tool with an explicit **correctness-vs-oracle gate before
  timing** (KVFlash NIAH + bit-exact relocation; PFlash NIAH; paged vs CPU
  oracle; tool-prefix identity contract), promoted to CI like
  `paged_attn_wmma_route`.

### Tier 3 — decode & speculative decoding (SOTA)
- Implement/bench **MTP on qwen4exp**: draft head, depth sweep **{2,3,4}**,
  instrument `draft_n` / `draft_n_accepted` / mean accept length, and report
  `tg_spec / tg_serial` at matched context and temperature.
- **Targets to beat** (with protocol):
  - pwilkin open `llama-bench -p16384 -n128 -d0,40000 -r3`: **26.28 tg @d0 /
    16.63 @40K**, 1204 pp @d0 (reproducible; we already match the pp).
  - CIRU v4.4 IU4 MTP3: **44.53 @12.96K, 21.14 @245K**; 60.35 short HE0-9.
  - halogen (vendor): ~41.7 served @32K, 45-50 short; EngramHalo open:
    39.3 code @d0.
- **Expectations:** ~1.5-1.9× on code/deterministic short prompts, ~1.3-1.5× at
  long context, depth 2-4; gains shrink and can go negative on high-temp prose
  and >128K context. Lead with a matched spec-vs-AR arm and acceptance counters,
  mirroring our `speed-profile` losslessness gate.

## 5. Recommended first three shipments
1. Commit qwen4exp correctness gates (planted + determinism + NIAH) — cheap,
   unblocks trust immediately.
2. Stand up the frozen fidelity harness (argmax + KL + cluster-bootstrap CIs +
   raw-logit audit) — the artifact others can reproduce.
3. MTP decode + accept-length telemetry and a matched spec-vs-AR arm — the
   decode SOTA target (short-context 1.5-1.9×).

## 6. Reference data (decode / spec-decode)

| Source | Config | pp | tg serial | tg spec | accept / AL |
|---|---|---|---|---|---|
| pwilkin llama-bench (open) | IQ4_NL, f16, d0 | 1204 ±2 | **26.28** | — | — |
| pwilkin llama-bench | d40k | 1086 | **16.63** | — | — |
| CIRU v4.4 IU4 | 12.96K, MTP3 | — | — | **44.53** | — |
| CIRU v4.4 IU4 | 245K cold, MTP3 | 761 | 7.62 ctrl | **21.14** | — |
| CIRU v4.4 HE0-9 | short, MTP3 | — | — | **60.35** | — |
| CIRU v4.3 | Hermes-20, 262K | 548 | — | 34.74 | **68.6%** |
| CIRU v4.2 | 12.96K, MTP6 | 984-990 | 21.2 | — | 30.3% |
| halogen (vendor) | served @32K, MTP | — | 34.1 | **41.7** | 1.63 tok/round |
| EngramHalo (open) | IQ3, q8KV, draft-mtp n=4 | 496 | 24.4 | **39.3** code | — |
| slb350 (pwilkin) | 128K/256K, MTP3 | — | — | 31.68 / 28.49 | — |
| **lucebox (this box)** | IQ4_NL, ~16.4K, no spec | **988** | **~27** | — | — |

Metrics definitions: `pp` = prompt tokens / prefill wall; `tg` = generated
tokens / decode wall (prefill excluded); acceptance = accepted/proposed; AL =
tokens committed per verify round; effective speedup = `tg_spec/tg_serial` at
matched context/temp, with a spec losslessness check.

Confounds to match: quant/bpw, KV type (F16 vs q8_0), context depth, MTP depth,
n-gram/prompt-lookup on/off, temperature, power envelope, resident vs SSD
tables, warmup, single vs multi-stream.

## 7. Tier 2 verification (2026-09-19, box `lucebox4`)

Built the existing special-tool gates (`ninja -C server/build-hip test_server_unit
test_kvflash bench_paged_attention test_paged_attn_wmma`) and ran the targeted
subset on gfx1151 (`HIP_VISIBLE_DEVICES=1`):

| gate | command | result |
|---|---|---|
| KVFlash qk / placement / pool-sizing / pager identity-sync | `ctest -R "kvflash\|KvflashPlacement\|KvflashPoolSizing"` | **pass** |
| Spec-decode acceptance accounting (tree/chain/AR-tail) | `ctest -R SpecAcceptance` | **pass** |
| Adaptive spec width | `ctest -R AdaptiveSpec` | **pass** |
| Paged-KV offload (quantized payload remap, budget/restore, cancel) | `ctest -R PagedKv` | **pass** |
| paged-attn vs double-precision CPU oracle (+ timing) | `bench_paged_attention` | **pass**, oracle max-abs-error 5e-6–1e-5 (limit 0.005) |
| paged-attn WMMA route | `test_paged_attn_wmma` | SKIP on gfx1151 (RDNA4-only); aborts on the box's gfx1201 dGPU (pre-existing, not our target) |

Notes: `ctest -R "kvflash|PagedKv|AdaptiveSpec|SpecAccept|paged_attn"` is 50/52
(the 2 are the RDNA4-only WMMA route). The host-side gates run in ~0.3 s and
could run on a CPU CI runner; the GPU ones need gfx1151. `test_kvflash` (the
model-driven NIAH/longab suite) needs a Qwen3.5/3.6 GGUF, not qwen4exp, so it was
not run here. None of KVFlash/PFlash/paged are integrated with the hand-written
qwen4exp graph yet — that is a separate question from whether the tools work.

Run recipe:
```
cd server/build-hip
ninja test_server_unit test_kvflash bench_paged_attention test_paged_attn_wmma
HIP_VISIBLE_DEVICES=1 ctest -R "kvflash|PagedKv|AdaptiveSpec|SpecAccept" --output-on-failure
HIP_VISIBLE_DEVICES=1 ./bench_paged_attention
```
