# Full-Harness Validation — ee7 + Adaptive Bandit

Evidence priority order (user-set, per plan):
1. **P1 — Agentic + multi-turn**: closest to production; headline result.
2. **P2 — Bandit adaptive multi-turn**: proves keep_ratio convergence under real load.
3. **P3 — NIAH**: no-regression safety check; already largely proven; runs last.

## Client Matrix

| Client | P1: Agentic | P2: Bandit 5-turn | P3: NIAH 32K | P3: NIAH 1K/128K |
|---|---|---|---|---|
| claude_code         | ee7 | bandit | ee7 | ee7 |
| codex               | ee7 | bandit | ee7 | ee7 |
| pi                  | ee7 | bandit | ee7 | ee7 |
| hermes              | ee7 | bandit | ee7 | ee7 |
| opencode            | ee7 | bandit | ee7 | ee7 |
| openwebui           | ee7 | —      | ee7 | —   |
| openwebui_tools     | ee7 | —      | ee7 | —   |
| openclaw            | ee7 | —      | ee7 | ee7 |
| claude_llamacpp_matrix        | ee7 | — | — | — |
| claude_llamacpp_decode_check  | ee7 | — | — | — |
| backend_pair        | ee7 | —      | —   | —  |

Bandit column applies to primary 5 only. The remaining 6 clients use
`MODEL_SERVER=lucebox` by default and exercise ee7 transparently, but have no
natural multi-turn session continuity — bandit keep_ratio tracking is not
applicable for them.

## How to Run

```bash
# P1 — agentic, all 11 clients
for c in claude_code codex pi hermes opencode openwebui openwebui_tools openclaw \
         claude_llamacpp_matrix claude_llamacpp_decode_check backend_pair; do
  bash server/bench/run_agentic_ee7_passbv.py --client "$c" \
    --output server/bench/results/2026-05-27_full_harness/$c/
done

# P2 — bandit 5-turn, primary 5 clients
for c in claude_code codex pi hermes opencode; do
  PFLASH_SESSION_ID="$(uuidgen)" \
  bash server/bench/run_bandit_5turn.py --client "$c" \
    --output server/bench/results/2026-05-27_full_harness/$c/
done

# P3 — NIAH no-regression, broad (1K-16K)
python3 server/bench/run_niah_ee7_broad.py \
  --output server/bench/results/2026-05-27_full_harness/niah_broad/

# P3 — NIAH long-context (32K-128K)
python3 server/bench/run_niah_ee7_longctx.py \
  --output server/bench/results/2026-05-27_full_harness/niah_longctx/
```

## Server Contention Rule

Bench scripts use port `:19099`+ (alternate). The user's interactive server on
`:18099` is left untouched. Bench scripts hold `flock /tmp/lucebox-bench.lock`
to prevent collisions between parallel runs.

## Reproduction

**Required env vars:**

```bash
export PFLASH_DRAFTER_EARLY_EXIT_N=7
export PFLASH_DRAFTER_SCORE_LAYERS=7
export PFLASH_SESSION_ID="$(uuidgen)"   # bandit runs only
```

**Model paths** (RTX 3090, 24 GB):
- Target: `~/models/Qwen3.6-27B-Q3_K_S.gguf`
- Drafter: `~/models/Qwen2.5-0.5B-Instruct-BF16.gguf`
- Chat template: `~/models/qwen3-coder-chat-template.jinja`

**Server launch** (bench scripts spawn their own server on alternate port, but
for manual verification):

```bash
./dflash_server \
  --model ~/models/Qwen3.6-27B-Q3_K_S.gguf \
  --draft-model ~/models/Qwen2.5-0.5B-Instruct-BF16.gguf \
  --chat-template-file ~/models/qwen3-coder-chat-template.jinja \
  --pflash-early-exit 7 \
  --pflash-score-layers 7 \
  --port 19099
```

**Wiring proof** (per-client verification that proxy injects session_id):

```bash
PFLASH_SESSION_ID=test123 bash harness/clients/run_codex.sh "hello world"
# expect: [run_codex] session-inject proxy up on ... and [pflash-bandit] in server logs
```
