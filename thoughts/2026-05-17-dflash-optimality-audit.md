# DFlash Optimality Audit — 2026-05-17

**Author:** momus | **Reviewer scope:** DFlash standalone bench path (test_dflash.cpp + bench_agent.py)

## TL;DR — the headline finding

The DFlash hot loop in `dflash/test/test_dflash.cpp` calls `sync_us()` — which issues `ggml_backend_synchronize(target_backend)` (and a second sync on the draft backend in split-GPU mode) — **27 times per decode step**, purely to feed the `[timing]` per-stage breakdown that `bench_agent.py` parses.

At RTX 3090 sync cost ~50-200µs and SWE-bench AL=4.23 (~60 steps for n_gen=256), that's **1.6k forced pipeline drains per prompt = ~80-320 ms of pure observer overhead on a ~5.5 s decode = 1.5-6% of headline tok/s left on the table for free.**

On top of that, there are 2-3 unconditional `std::printf` calls per step (lines 3662, 3677, 3974, 4010) writing to a pipe captured by Python — more sync points.

**The single cheapest experiment in the world**: build with `sync_us` neutered (`#ifdef DFLASH_TIMING` or env-gated) and the four debug printfs `#if 0`'d out, re-run the 262/301/350 W three-point sweep. Prediction: **+2-5% across all power points**, and possibly recovers most of the 262 W decay because forced syncs serialise the pipeline and hurt more when compute is already power-constrained.

## Untried CLI flags

The bench passes only `--fast-rollback --ddtree --ddtree-budget=22 --max-ctx=N`. Everything else default.

| Flag | Default | What it does | SWE impact | Test cost |
|---|---|---|---|---|
| `--ddtree-temp=T` | 1.0 (`test_dflash.cpp:1560`) | Sharpens draft logits before top-K extract | Mid-AL workloads benefit from sharpening (T=0.7-0.8 worth a sweep). RESULTS.md budget sweep was T=1.0 only. | 30 min |
| `--ddtree-no-chain-seed` | chain seeding ON | Disables full-chain pre-seed; pure best-first | RESULTS.md credits chain pre-seed with +~5 AL on HumanEval. Probably bad on SWE; confirm. | 5 min |
| `--draft-swa=N` / `DFLASH27B_DRAFT_SWA` | 0/unset | Per-layer SWA window on draft model | Should sweep 1024/4096 — narrower draft attention = faster draft compute but may collapse AL on long agentic prompts. | 1 h |
| `--draft-ctx-max=N` | 4096 (clamped low) | Caps draft attention slice over `cache.target_feat` | At 8K/24K SWE buckets draft sees only 2048-4096 of context; raising MAY raise AL by letting draft "see" more recent tool outputs. | 1 h |
| `--seq-verify` | off | Sequential single-token decodes instead of batched verify | If AL jumps on seq-verify, there's a verify correctness bug eating SWE AL (z-lab #57). | 20 min |
| `-ctk q4_0 -ctv q4_0` | (env-set q8_0/q8_0) | Lower KV bitwidth | RESULTS.md: ~3% cost at short context but enables 128K. At 24K may be net positive due to bandwidth. | 30 min |
| `-ctk bf16 -ctv bf16` | n/a | BF16 KV (no quant) | Bandwidth-heavier than Q8/Q8 but skips dequant. At 24K could matter. | 30 min |
| `DFLASH27B_FA_WINDOW` | 2048 | Target FA sliding window | Not set in bench_agent.py. RESULTS.md ablation suggests 4096 helps long contexts. Never swept on SWE. | 30 min |
| `DFLASH27B_DRAFT_BLOCK_SIZE` | 16 (`dflash27b.h:37`, compile-time) | Tokens per draft step | Budget 22 assumes block_size=16. For low-AL workloads (SWE 4.23, chain ceiling ~7), smaller block could reduce wasted draft compute. Requires recompile. | 4 h |
| `--profile-scaling` | off | Microbench: target forward time vs N | Diagnostic — tells you where ddtree-budget plateau really is for SWE. | 20 min |

## Code-level optimization opportunities (ranked)

### 1. **`sync_us()` observer effect (HEADLINE)** — 0.25 days, expected +2-5%
`test_dflash.cpp:3275-3279` defines `sync_us` as a function that ALWAYS calls `ggml_backend_synchronize(target_backend)`. Called from 27 sites inside the per-step loop (grep-verified). No `#ifdef` guard. RESULTS.md headline AND our bench numbers all eat this cost.

Fix: wrap `sync_us` in `#ifdef DFLASH_TIMING` or `if (g_timing_enabled)`; default off. Gate the `[timing]` block + 27 accumulators on same flag.

### 2. **Unconditional debug printf in hot loop** — 0.1 days, expected +0.5-1%
- `test_dflash.cpp:3662-3673` — `[dbg sib step]` (DDTree)
- `test_dflash.cpp:3677` — `[step N] committed=…`
- `test_dflash.cpp:3974` — `[step N] committed=…`
- `test_dflash.cpp:4010` — `[step N] accept_n=…`

Each `printf` to a Python-captured pipe is a write syscall. ~50µs each × 4 per step × 60 steps × 8 prompts = ~100ms per bench run.

Fix: same `g_timing_enabled` gate.

### 3. **`extract_draft_topk` runs CPU OpenMP per step** — 0.5 days, expected +1-2%
`test_dflash.cpp:3486-3498`. Budget 22 falls in path where `ddtree_K = 8` and full logits transferred via `ggml_backend_tensor_get` (9 MB D2H per step at vocab 152k × 15). RESULTS.md credits K=32→8 with +2.1 tok/s; could push to K=4.

Fix: lower `ddtree_K` from 8 to 4, validate AL unchanged.

### 4. **Per-step `std::vector` allocations in DDTree path** — 0.5 days, <1%
`test_dflash.cpp:3567,3572,3579,3619,3638,3645` — `flat_tokens`, `tree_embed`, `pos4`, `parent_ids`, `posterior`, `accepted` allocated per step. Not hoisted like `mask_buf` is. Small but real.

### 5. **`ddtree-budget=22` tuned on HumanEval (AL=8.88), SWE plateaus at AL=4.23** — 0.5 days, expected +3-8% on SWE
RESULTS.md sweep table at budget 22 plateaus at AL 8.88. SWE plateaus at AL 4.23 — half the AL means roughly half the tree depth utilization. **Budget 22 is almost certainly oversized for SWE.** Each unused node costs verify time, buys zero AL. Predicted SWE optimum: budget 12-16.

Fix: sweep `--ddtree-budget=12,14,16,18,20,22` on SWE-2K n=8.

### 6. **Noise-embedding `[last_tok, MASK*15]`** — not a code fix, requires drafter retrain
Drafter trained on this exact pattern. RESULTS.md HumanEval AL 8.31 vs SWE AL 4.23 likely reflects drafter's training distribution being closer to HumanEval-style than SWE-bench agentic. No code knob.

## Workload-specific knobs we haven't tuned

| Hypothesis | Cheap test |
|---|---|
| SWE AL=4.23 means budget 22 wastes nodes → smaller budget = faster verify same AL | Sweep `--ddtree-budget=10,12,14,16,18,20,22` |
| Draft sees only last 2048 of 24K prompt — agentic context decays | Sweep `DFLASH27B_DRAFT_SWA=1024,2048,4096,8192` + `--draft-ctx-max` |
| `--ddtree-temp=0.7-0.9` sharpens draft so sibling allocation hits higher-prob tokens | Sweep `--ddtree-temp=0.5,0.7,0.85,1.0,1.2` |
| Batched verify divergence silently truncating SWE AL | One run with `--seq-verify` |
| Q4_0 KV at 24K reduces verify bandwidth more than it costs in dequant | One run `-ctk q4_0 -ctv q4_0` on 24K bucket |

## What HumanEval config has that SWE doesn't

`bench_he.py:238-262` vs `bench_agent.py:201-217`:

| HumanEval | SWE |
|---|---|
| Exposes `--ddtree-temp` | Not exposed |
| Exposes `--ddtree-no-chain-seed` | Not exposed |
| Exposes `--draft-feature-mirror`, `--peer-access`, multi-GPU flags, IPC, `--prefill-ubatch` | None exposed |
| Default n_gen = 128 | Default n_gen = 256 |
| Does NOT pass `-ctk q8_0 -ctv q8_0` | Does NOT pass either |

**Both rely on `resolve_kv_types()` defaults via `DFLASH27B_KV_K/V` env. The SWE↔HumanEval RESULTS.md comparison is not apples-to-apples on KV** if env was set differently between runs.

## Final verdict

**Run ONE experiment: gate `sync_us` + 4 hot-loop `printf`s behind `DFLASH_TIMING=1`, default off; rebuild; re-run 301W SWE-2K n=8.**

```bash
# After patching + rebuilding test_dflash:
python3 scripts/bench_agent.py --bucket 2k --n-sample 8 \
  --out /tmp/dflash_no_observer_301w.json
```

Compare to baseline `dflash/bench/results/2026-05-17T13-55-36_0925bea/swe_bench_2k_x_dflash_b22.json` (median 48.99 tok/s).

**Decision rule:**
- ≥2% gain → ship the gate, re-baseline all three power points
- unchanged → next experiment is `--ddtree-budget=12,14,16,18` sweep at 301W (+3-8% predicted)

**If only ONE other thing today**: the budget sweep. Cheapest single-knob test mathematically motivated by RESULTS.md's own AL/budget plateau table.

## Reference paths

- `dflash/test/test_dflash.cpp` — hot loop (`sync_us` 3275-3279; printfs 3662,3677,3974,4010; ddtree 3547-3925; chain 3935-4140; arg parsing 1623-1813)
- `dflash/src/common/dflash_spec_decode.cpp` — layer-split spec decode (not used single-GPU SWE)
- `dflash/src/common/ddtree.cpp:12-30` — OpenMP top-K
- `dflash/scripts/bench_agent.py:208-211` — SWE bench cmd
- `dflash/scripts/bench_he.py:312-357` — HumanEval bench cmd (more flags exposed)
- `dflash/RESULTS.md:105-120` — budget sweep table
- `dflash/RESULTS.md:122-137` — kernel wins history
- `dflash/include/dflash27b.h:37` — `DFLASH27B_DRAFT_BLOCK_SIZE 16` (compile-time)
