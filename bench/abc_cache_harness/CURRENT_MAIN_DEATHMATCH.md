# Current-Main Deathmatch Runbook

Branch: `codex/deathmatch-current-main-review`  
Remote: `origin = https://github.com/Luce-Org/lucebox-hub.git`  
Base: `origin/main@b44c872b`

Verified on 2026-07-05: `git fetch origin main` returned the same
`b44c872b` tip for `HEAD`, `origin/main`, and `FETCH_HEAD`.

## Goal

Correctness is the gate. Speed rows only count after the run is valid:

- no request errors or censored turns
- expected tool calls are structured and named correctly
- unexpected tool-call rate is tracked
- output-token equality is used only for pinned speed rows
- natural tool-stop rows are not mixed with pinned equal-output claims
- wall, prefill, decode, GPU memory, and energy are reported together

## What Was Ported

This branch restores the prior ABC replay/deathmatch harness onto current main:

- `replay_harness.py`
- trace exporters/structurers
- 38-turn tool and mixed-session traces
- the 2026-07-02 PC64 evidence ledger

The harness also adds optional energy sampling:

```bash
--power-sample-interval 1.0 --power-gpu-index 0
```

That records `energy_j`, `avg_power_w`, `max_power_w`,
`mean_gpu_util_pct`, and `max_memory_used_mib` in the raw JSON, markdown
report, and console summary.

## Claim Boundaries

Historical rows from `bench/upstream-pr469-473-plus-468` are evidence, not
current-main proof. They were produced by an older branch/binary, so all
merge/deathmatch claims need fresh runs on this branch even though this branch
is currently at the fetched `origin/main` tip.

Use two complementary rows:

1. Natural tool behavior: cap2048 mixed trace, no pinning. This gates tool
   validity and natural stop behavior.
2. Equal-output speed: same mixed trace at cap256 with `--pin-decode-length`.
   This gates wall/decode at identical requested output length. Tool rows in
   pinned mode are speed-only because forced length conflicts with natural
   tool-stop behavior.

## Build

```bash
CUDAHOSTCXX=/usr/bin/g++-11 cmake -S server -B server/build-current-main-gcc11-cuda126 \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_CUDA_ARCHITECTURES=86 \
  -DCMAKE_C_COMPILER=/usr/bin/gcc-11 \
  -DCMAKE_CXX_COMPILER=/usr/bin/g++-11 \
  -DCMAKE_CUDA_COMPILER=/usr/local/cuda-12.6/bin/nvcc \
  -DCMAKE_CUDA_HOST_COMPILER=/usr/bin/g++-11

CCACHE_DISABLE=1 cmake --build server/build-current-main-gcc11-cuda126 \
  --target test_server_unit -j 4

server/build-current-main-gcc11-cuda126/test_server_unit

CCACHE_DISABLE=1 cmake --build server/build-current-main-gcc11-cuda126 \
  --target dflash_server -j 4
```

The initial unpinned configure used `/usr/bin/nvcc`/CUDA 12.0 with GCC 13 and
failed during CUDA compiler detection on glibc `_Float32`/`_Float64` types.
The validated build pins CUDA 12.6 and GCC/G++ 11. `test_server_unit` passed
2039 assertions with 0 failures, and `dflash_server` built successfully.

## Regenerate Private Mixed Trace

The source transcript is intentionally not committed. Regenerate locally:

```bash
python3 bench/abc_cache_harness/structure_claude_replay_trace.py \
  --in /home/peppi/.claude/projects/-home-peppi-Dev-lucebox-hub/b324020e-f90c-45f3-8055-55dd5fe723c3.jsonl \
  --out /tmp/luce_mixed_candidate_0_fixed_38.jsonl \
  --turns 38 \
  --max-tokens 256 \
  --source-kind raw-session

python3 - <<'PY'
import json
src = "/tmp/luce_mixed_candidate_0_fixed_38.jsonl"
dst = "/tmp/luce_mixed_candidate_0_fixed_38_cap2048.jsonl"
with open(src, encoding="utf-8") as f, open(dst, "w", encoding="utf-8") as out:
    for line in f:
        row = json.loads(line)
        row["max_tokens"] = 2048
        out.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
PY
```

## Gate A: MoE Natural Tool Behavior

```bash
python3 bench/abc_cache_harness/replay_harness.py \
  --arm AR_35B_KVF_FORCE \
  --trace /tmp/luce_mixed_candidate_0_fixed_38_cap2048.jsonl \
  --n 1 \
  --max-ctx 131072 \
  --binary server/build-current-main-gcc11-cuda126/dflash_server \
  --power-sample-interval 1.0

python3 bench/abc_cache_harness/replay_harness.py \
  --arm AR_LLAMA_35B_SLOTCACHE \
  --trace /tmp/luce_mixed_candidate_0_fixed_38_cap2048.jsonl \
  --n 1 \
  --max-ctx 131072 \
  --power-sample-interval 1.0
```

Pass condition: Lucebox has no errors, better expected-tool validity, lower
unexpected-tool rate, and lower wall/energy for comparable work. Output lengths
may differ; do not cite this as equal-output speed.

## Gate B: MoE Equal-Output Speed

```bash
python3 bench/abc_cache_harness/replay_harness.py \
  --arm AR_35B_KVF_FORCE \
  --trace /tmp/luce_mixed_candidate_0_fixed_38.jsonl \
  --pin-decode-length \
  --n 1 \
  --max-ctx 131072 \
  --binary server/build-current-main-gcc11-cuda126/dflash_server \
  --power-sample-interval 1.0

python3 bench/abc_cache_harness/replay_harness.py \
  --arm AR_LLAMA_35B_SLOTCACHE \
  --trace /tmp/luce_mixed_candidate_0_fixed_38.jsonl \
  --pin-decode-length \
  --n 1 \
  --max-ctx 131072 \
  --power-sample-interval 1.0
```

Pass condition: both arms emit the same requested output tokens with no errors,
and Lucebox wins wall, weighted decode, and energy. Treat expected tool rows as
speed-only under pinning.

## Gate C: Spark Variant

Run Gate A and Gate B again with:

```bash
--arm AR_35B_SPARK_KVF
```

This checks whether constrained Spark improves hardware efficiency without
regressing correctness.

## Gate D: Dense Qwen3.6-27B

Use the same trace pair with:

```bash
--arm AR_27B_KVF
--arm AR_LLAMA_27B_SLOTCACHE
```

This separates dense-model behavior from the MoE/Spark path.

## Escalation To Long Deathmatch

After 38-turn gates pass, extend the generated trace to 60, 120, and 240 turns.
Keep the same two-row discipline: cap2048 natural tool behavior and pinned
equal-output speed. Do not merge the two claims into a single row unless one
trace/run simultaneously satisfies natural tool correctness and equal output.
