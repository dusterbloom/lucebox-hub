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

## 8. Tier 3 — MTP draft head (reference design, 2026-09-19)

Chosen: **MTP via llama.cpp `--spec-type draft-mtp`**. The head ships inside
`Qwen/Qwen3.8-Flash-Next` (converter PR 27742 drops it) and is republished as
`dzannotti/Qwen3.8-Flash-Next-MTP-GGUF` (Qwen Community 1.0): `...-MTP-Q4_K_M.gguf`
2.5 GB (use this; match the target's quant — a Q8_0 head measured worse), `...-MTP-BF16.gguf`
7.8 GB, and the reference patch `patches/qwen4exp-mtp-draft-head.patch` + converter
`patches/merge-mtp-shard.py`. Alternative head: `EasiiX/...-MTP-Strix-Halo-GGUF`
(Q8_0, gfx1151-tuned); `agentionai/...-MTP-Q8_0-GGUF` (ROCmFP4). DFlash2/EAGLE3 have
no Flash-Next drafter.

**Head layout (34 tensors, `block_count=49`, `nextn_predict_layers=1`):** one full
`qwen4exp` block `blk.48.*` (attn + indexer + 512-expert MoE + hyper-connections),
`blk.48.nextn.{eh_proj [2*n_embd,n_embd], enorm [n_embd], hnorm [hc*n_embd]}`,
plus shared `token_embd`/`output`/`output_hc_*` (and, MTP-only, the head's own
`hyper_connection_mixer` as the final mixer).

**Draft graph (from the patch):**
- inputs: `tokens` (shared embed) and `h` = the trunk's wide residual **before the
  final mix** (`res_hc`, `[hc*n_embd, T]`); trunk sets `t_h_nextn = res_hc`.
- `h_norm = rms(h, hnorm)` spans `hc*n_embd` then reshapes to `[n_embd, hc, T]`;
  `e_norm = rms(tok_embd, enorm)` is one `[n_embd,T]` stream **repeated across hc**.
- `inpL = eh_proj @ concat(e_norm, h_norm)` (concat on dim0 → `[2*n_embd, hc, T]`);
  so `eh_proj` contracts **per HC stream**, no pre-pooling (PR 27836's key rule).
- then the normal block: `hc_mix → dense attention → hc_combine → hc_mix → MoE →
  hc_combine`; the draft attends **dense** (its indexer weights exist but are skipped).
- output: reshape to `[hc*n_embd, T]` (`t_h_nextn` for chained steps), then the head's
  final mixer (`hc_head_norm/down/up`) → shared `output` → logits.

**Runtime contract:** `--spec-type draft-mtp --spec-draft-n-max 3 --spec-draft-p-min 0.75`
(depth 3–4; `p-min` matters more than depth — deeper/adaptive drafting is slower on
this bandwidth-bound iGPU because the head carries its own MoE so every draft token is
a real forward pass). `LLAMA_ATTN_ROT_DISABLE=1` (qwen4exp attention rejects upstream's
quantized-KV rotation). Target PLE n-gram table stays in host memory
(`-ot per_layer_token_embd=CPU`).

**Reference numbers (Strix Halo, temp 0, 300 tok):** ROCm UD-Q4_K_XL 20.3 → **35.8 code
/ 22.6 prose** (accept 0.90 / 0.74); ROCm UD-IQ4_XS 18.0 → 32.8 / 22.1 (0.84 / 0.68);
Vulkan 24.2 → 37.2 / 30.3 (0.88 / 0.82). Profiling: target pass ~47 ms/token, each
extra verified token ~+4.4 ms, head step ~3.4 ms, ~1/3 of a pass is kernel-launch gaps
(target graph ~8000 nodes).

**Port plan into our graph (server/src/qwen4exp/):**
1. Loader: extend beyond `n_layer` to load one MTP block (`blk.48.*`, `nextn.*`,
   `block_count=49`, `compress_ratios += 0`); keep the trunk path unchanged.
2. Forward: add `qwen4exp_mtp_forward(hidden_wide, tokens)` building the block above,
   reusing `hc_mix`/`build_full_attn` (dense)/`build_moe`; return draft logits + the
   next wide state.
3. Trunk: expose `res_hc` (pre-final-mix) to the spec loop.
4. Backend: draft-and-verify loop (chain), accept/reject with target logits;
   report `draft_n`/`draft_n_accepted`/AL.
5. Measure with harness `agent`/`he` + acceptance telemetry; target ≈1.5–1.8× code,
   1.3–1.5× long-context, prose near break-even.

## 9. Spec-decoder decision for Qwen3.8-Flash-Next (2026-09-19)

### HF evidence (re-checked)

- **No DFlash and no DSpark drafter exists for Flash-Next.** The only DFlash
  artifact is `PixelML/Qwen3.8-Flash-Next-NVFP4-DFlash` — NVFP4, vLLM-only
  (3 control-flow patches + adapter overlay), not a GGUF, not loadable by our
  HIP/llama.cpp engine. `DSpark` search returns nothing for this model.
- **DFlash2 exists only for the dense 27B**: `z-lab/Qwen3.8-27B-DFlash2` /
  `incoai/Qwen3.8-27B-DFlash2` (block 8, two-tap dynamic conv + candidate path
  selector). Not for the Flash-Next hybrid.
- **MTP heads are plentiful** (the head ships in the checkpoint): `dzannotti`,
  `EasiiX`, `agentionai`, `ToPo-ToPo`, `drluoto`, `ashbash`, etc.
- **Lucebox's own drafter experience** (HF `Lucebox/`):
  `Qwen3.6-27B-DFlash-GGUF`, `gemma-4-{26B-A4B,31B}-it-DFlash-GGUF`,
  `Laguna-XS.2-DFlash-GGUF` (with `dflash_aux_heads.pt` + `GATE_RESULTS.md`),
  `DeepSeek-V4-Flash-0731-DSpark-GGUF`, `Kimi-K3-DSpark-Q8_0-GGUF`. The Laguna
  card states the method: **"full continued training from v23-step18000 on 60k
  clean regenerated rows (10k @16k ctx + 50k Open-PerfectBlend @4k ctx)"**, and
  the GGUF carries **DSpark Markov/confidence aux heads** (our
  `src/common/dspark_head.cpp`, `src/deepseek4/deepseek4_dspark*.cpp`).

### What DFlash/DFlash2/DSpark are

- **DFlash** (z-lab, ICML'26, arXiv 2602.06036): a lightweight **block-diffusion**
  drafter. It conditions on the last target token + last ~5 captured target hidden
  states and denoises a block (16) of MASK tokens in one forward. Loss = masked-
  position cross-entropy. `z-lab/dflash` ships `pip install dflash` + the training
  code; draft arch is ~5 layers / hidden 5120 / MASK token / RoPE θ1e6.
- **DFlash2** (Inco AI, Aug'26): same idea, block 8, **two-tap dynamic convs** in
  the backbone + a **candidate path selector** so the block doesn't decay toward
  the end. Beats MTP and DSpark on H200 (GSM8K AL 5.46 vs 5.02 MTP, 4.36 DSpark;
  3.43× vs 2.59×/2.69× concurrency-1 GSM8K).
- **DSpark** = DFlash + Markov/confidence **aux heads** (adaptive verify width /
  confidence-gated top-k); we already run this for DeepSeek V4.

### Recommended plan

Sequence: **ship MTP now** (zero training; head + reference patch already in
hand — §8), and **train a DFlash2-style drafter for Flash-Next** as the SOTA path,
reusing the Lucebox pipeline.

1. **Bootstrap the trainer.** Start from `z-lab/dflash` (`pip install dflash`) and
   the Lucebox Laguna run's config (60k regenerated rows; `dflash_aux_heads.pt`
   trainer). Our inference side already loads DFlash drafts
   (`gguf_draft_loader.cpp`, `--draft`, DDTree, fast rollback, KVFlash-on-pool).
2. **Data.** Regenerate ~60k rows *from Qwen3.8-Flash-Next itself*: 10k @16k ctx +
   50k Open-PerfectBlend @4k, plus code/math/agent mixes, **thinking disabled**
   (RESULTS.md: a drafter trained on non-thinking output predicts it better), and
   cache the target hidden states + logits needed for the conditioning.
3. **Architecture.** A Flash-Next drafter must consume the target's hidden state.
   Condition on the trunk's post-final-mix hidden (or the wide `res_hc`, as the MTP
   head does) via an `fc`/`hidden_norm` pair, then the DFlash2 block-diffusion
   backbone (block 8, two-tap conv, path selector). Flash-Next is a hybrid
   (HC/QSA/GDN/MoE); the drafter itself can be a plain small transformer — only the
   conditioning interface must match. Reuse `qwen35-dflash-draft` loader shape and
   extend it to a `qwen4exp-dflash-draft` arch.
4. **Train.** Block-diffusion masked-CE, 4k→16k ctx curriculum, target frozen; add
   the DSpark Markov/confidence aux heads for adaptive width and confidence-gated
   verification. Init from scratch if the hidden interface diverges from
   Qwen3.8-27B-DFlash2, else fine-tune from it.
5. **Eval.** Acceptance length + tok/s on gsm8k/math500/humaneval/mbpp/mt-bench
   (z-lab harness) at concurrency 1 and 8, then the Lucebox serving gate (our
   `harness/benchmarks`, `speed-profile` losslessness) — target ≈3× dense-class,
   and clearly above MTP's ~1.5–1.8× code on this bandwidth-bound iGPU.
6. **Compute.** A ~5-layer / hidden ~5k drafter over 60k 4k–16k rows is a
   multi-GPU-day job (z-lab used Modal/InnoMatrix; Lucebox trained Laguna in-house).

Decision: **MTP for immediate SOTA-ish decode; DFlash2-style as the trained
upgrade; skip DSpark-for-Flash-Next** (no public drafter, and DFlash2 already
subsumes it in the Inco eval). If a public Flash-Next DFlash appears, revisit.

## 10. Quality vs the model card, DeepSWE feasibility, local models (2026-09-19)

### Quality vs the original model card

The `Qwen/Qwen3.8-Flash-Next` card does **not** report GSM8K / Math500 /
HumanEval. It reports agentic + coding: DeepSWE 1.1 **58.7**, SWE-bench Pro 62.5,
SWE-bench Multilingual 81.0, NL2Repo-Bench 48.1, LiveCodeBench v6 91.9, GPQA
Diamond 91.7, HLE 35.9, IFBench 81.3 (plus vision). So our harness suites have
**no direct card reference**.

Our harness numbers were also **wrong, not the model**: the GSM extractor grabbed
an intermediate number (model-correct 260/160/120 scored as 20/40/4) and the Math
normalizer didn't equate `[2, 5)` with `[2,5)` or `\frac{20}{3}` with `20/3`.
After fixing `harness/math_scoring.py` + `_score_gsm_response` (`32d1fa66`):

| suite | reported | corrected |
|---|---|---|
| HE (gold-test scored) | 10/10 | 10/10 |
| GSM8K | 7/10 | **10/10** |
| Math500 | 7/10 | **9/10** (1: max_tokens truncation at 2048) |
| planted recall 13.6k/24k | 2/2 | 2/2 |

n=10 per suite, so these are smoke-level, not capability claims. To compare with
the card we must run an agentic/coding eval (DeepSWE or LiveCodeBench), which is
the real open item.

### DeepSWE — can we compare, and how much work?

DeepSWE (`datacurve/deep-swe`): **113 long-horizon SWE tasks** (TS/Go/Py/JS/Rust)
in the **Harbor** format with Docker task environments + program-based verifiers.
It is **gated** (must accept access terms). Official runs use **Pier**
(`pier run -p deep-swe/tasks --agent mini-swe-agent --model ...`), with
`mini-swe-agent` / `claude-code` / `codex` / `opencode`, **256K context, temp 1.0,
top_p 0.95**, and were produced on **Modal** sandboxes. Card: 58.7 for Flash-Next.

Blockers on `lucebox4` today:
1. **No Docker** on the box (task environments / verifiers need it).
2. **No 256K serving** — chunked QSA + indexer cache get 64K working; native is
   262144 but our graph alloc OOMs well before that, and the 28.8 GB PLE n-gram
   table needs offload.
3. **No Pier/multi-agent harness or dataset access** on the box.
4. Tool-call reliability under the Qwen `<tool_call>` format is untested (the
   harness has `client_test_runner probe` → `chat.tools_accepted`).
5. On Strix Halo a 256K agentic task is hours of GPU; 113 tasks is days.

Work to a *trustworthy* number: get dataset access + Pier + Docker; make our
server serve **256K with correct tool calls** (finish the OOM/memory path, PLE
offload, QSA at depth); run a **subset** (`--n-tasks 10 --sample-seed 0`) with
mini-swe-agent pointed at our OpenAI endpoint; score with the program verifiers.
That is a multi-day-to-week integration, and even then only a subset is
practical on this single iGPU. Recommendation: treat DeepSWE as a **later,
subset-only** target; meanwhile add the tool-call probe as a signed-off gate and
keep HE/GSM/Math/recall as the routine suite.

### Models available locally (`lucebox4:~/models`)

| model | file(s) | size | notes |
|---|---|---|---|
| **Qwen3.8-Flash-Next IQ4_NL** (what we run) | 3-shard | 94 GB | hand-written qwen4exp graph |
| Qwen3.8-Flash-Next GSQ-RCO-IQ3_XXS | 2-shard | 71 GB | another Flash-Next quant to try |
| Qwen3.8-Flash-Next IQ4_NL "unc" | 1 file | 119 GB | **mislabeled Q8_0** (handoff) |
| Qwen3.8-27B UD-IQ4_XS (dense) | 1 file | 13.3 GB | + DFlash2 below |
| **Qwen3.8-27B DFlash2 drafter** | `qwen38-dflash2-f16.gguf`, `-q8_0.gguf`, `dflash2/` | 3.6/1.9 GB | **spec-decode testbed** |

The 27B + its **DFlash2 drafter** is locally available and is the best way to
exercise our DFlash/DDTree/KVFlash spec machinery end-to-end now, while any
Flash-Next drafter would have to be trained (§9).
