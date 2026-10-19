# Prereg: selector v2 with the pinned preamble and user-question queries on real agent decision steps (3090)

Fixed before any eval row (2026-10-19). Only the smoke in `smoke/` ran before this file (one trajectory, smoke ids, not eval rows); it was used to read the preamble length and to check the logs, nothing else.

## Question
On real agent steps whose prompt ends with a tool result (no appended question naming the source), cut past the pool, does selector v2 with the system+tools preamble pinned outside the pool (`LUCE_KVFLASH_PREAMBLE_ROWS`) and the latest human question as extra queries (`LUCE_KVFLASH_UQ_QUERIES`) keep the model's next action at least as close to full KV as the shipped ON6 setting does?

## Data
- `trajectories.jsonl` sha256 `797d312458a7a83f1e7afa8f8e2d0e84c855b1f78d4d30229705decb7d903db3`, `tasks_all60.jsonl` sha256 `a87111ddd7c1844868eefafe87f0ba6c8e0a44e246ebe3936a27f3a00a34175d`: the 60 SWE-agent traces behind the 951 probes. All 60 share one system+tools block; 59 have one human message (the task), one has two.
- Decision step: a `kind=decision` task (the prompt ends with a tool message; 1438 in total, 1269 cut past 16,384 tokens). Eligible: `cut_tok > 16384 + R` = 22,528, so every arm pages (a pinned arm's pool is 16K + R rows and would hold a shorter cut whole). 1045 eligible.
- Subset: 400 steps, `random.Random(20261019).sample` over the eligible list sorted by id, covering all 60 trajectories (`decision_steps.py select 6144 400`). `decision_ids.txt` sha256 `9a137ba7caf1016d3e7f2aa7393243ee3ae594541e7c25b58c41f6c4f304d544`, `decision_tasks.jsonl` sha256 `07bfe8b95e553fa54706c516e57515eed245625aad82d689a178dd5acfbd7ce8`.
- Session growth: each trajectory starts with one warm-up (`max_tokens` 1, `kind=warmup`, not scored) at its last tool-message boundary that fits 16,384 tokens (10,948-16,364), so the system+tools block and the task are prefilled densely as in a real session; then its steps run in cut order.
- Request: the trajectory up to the cut, the trajectory's tools, `qwen36_template.jinja` with thinking off, temperature 0, seed 1, `max_tokens` 256 (the harness default for decisions), `--max-ctx 52224`, `--no-kv-cache` behaviour as in every earlier arm (`disk_hit=false` checked).

## Binary and model
`kvflash/2.0c-pin-uq` @ `28806315`, frozen at `/mnt/kimi-k3/build/pinuq-cuda-28806315`: `luce_server` md5 `f956a5e1dcac2485a37184bab3cab4ef`, libggml-cuda `1e725f2f4138b5daf4de8f9b060e610c` (all md5s in that dir's `PROVENANCE`). Model `Qwen3.8-27B-UD-Q4_K_M.gguf` md5 `76f527022f010d42bc0dff5905c60444`. One binary for every arm. Unit suite 822 pass / 0 fail / 3 skip at this commit.

## Arms (the 3090, one session, this order, `drop_caches` + `compact_memory` and an idle-GPU check before each)
| Arm | Pool | Selector | `PREAMBLE_ROWS` | `UQ_QUERIES` |
|---|---|---|---|---|
| FULL | none (reference) | none | 0 | 0 |
| ON6 | 16,384 | shipped rlru4k-v2 (mode 4, k = 6, exact scorer, recency pool) | 0 | 0 |
| V2 | 16,384 | selector v2 (ON6 + GPU scorer, 64 queries, unit 2) | 0 | 0 |
| V2PU | 16,384 + 6,144 | selector v2 | 6,144 | 64 |
| V2P | 16,384 + 6,144 | selector v2 | 6,144 | 0 |
| OFFP | 16,384 + 6,144 | none (no reselect) | 6,144 | 0 |

R = 6,144: the system+tools block is 5,921 tokens (smoke log `preamble pin end=5921`), rounded up to 256. The pinned arms hold R extra rows by design. Rows of the reserve the 5,921-token preamble does not use (223) hold history. V2P - V2 therefore measures "pin plus R rows", not pinning at equal memory.

Then the probe guard, same binary: the 710 retained probes cut past 16,384 tokens (`retained_paged.json`, sha256 `e2fe7d03513e7c9faf37a9331512a2cdde7c435ce21d9f1245b436932a2d4a0f`; 207 of them beyond the pool) on ON6 and V2PU. Probes cut at or under 16,384 tokens fit whole in a pinned arm's pool and cannot lose there, so they are not rerun.

## Metric (`analyze_decision.py`, selftest included)
- Agreement with FULL per step: the first tool call's name and arguments (JSON parsed, keys sorted, ids dropped) equal FULL's; if FULL emits no tool call, the stripped text must be equal.
- Secondary: tool-name agreement.
- Paired over the steps that have a row on every arm. 95% CI by bootstrap over trajectories (10,000 draws, seed 20261019).

## Decision
- Primary: V2PU - ON6. **WIN** if the CI lower bound > 0. **NONINFERIOR** if the lower bound >= -2.0 points. **LOSS** otherwise.
- V2PU becomes the candidate default only on WIN or NONINFERIOR **and** if the probe guard holds: V2PU - ON6 on the 710 paged probes has a CI lower bound >= -2.0 points (`analyze_e2e.py` method, exact answer match).
- Secondary, reported, not deciding: V2 - ON6 (v2 alone on decisions, the audit's open question); V2PU - V2P (user-question queries; V2PU also cuts a prefill batch at the question's end, so batch-boundary numerics are part of it); V2P - V2 (pin plus R rows); OFFP vs FULL (how much decisions depend on old context: if OFFP agrees with FULL on >= 95% of steps, the set cannot separate selectors and the primary is reported as underpowered, not as a win).
- Instrumentation gates (else the arm is void): V2PU logs `uq hit` on >= 90% of its reselects; every pinned arm logs `preamble pin … pinned_chunks=93` on every pooled request; V2/V2P/V2PU log the GPU scorer (no fallback).

## Cost (reported, not deciding)
Per arm: prefill ms, decode tok/s (steps with >= 32 tokens) and client wall s, each as median, mean, best, min; GPU scorer median ms. Smoke: the scorer reads 128 queries with UQ and took 179-218 ms (64 queries: ~88 ms in the 10-18 run), so V2PU may exceed the earlier 200 ms cost line.

## Predictions (written before rows)
- OFFP exact agreement with FULL: 60% (tool name 88%).
- V2PU - ON6: +1.0 points (between -2 and +4).
- V2 - ON6: -0.5 points.
- V2PU - V2P: +0.5 points.
- UQ hit rate: >= 99%.
- GPU scorer median: V2 85 ms, V2PU 170 ms.
- Probe guard V2PU - ON6 on the 710 paged probes: +3 points (v2's beyond-pool gain), lower bound > -2.

## Artifacts
`decision_steps.py`, `decision_ids.txt`, `retained_paged.json`, `run_decision.sh`, `analyze_decision.py`, `run_arm.patch` (the harness copy differs from `rlru4/agentic_eval/run_arm.py` only in honouring a task's `max_tokens`), `smoke/`, then per-arm JSONL, server logs and `run_decision.log`.
