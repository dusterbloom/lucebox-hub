# B0 on the 3090: agentic coding traces, full KV vs KVFlash paging without compaction

Run 2026-10-14 12:25-20:50 on the local RTX 3090 with harness `agentic_eval/` (`run_all.sh`, `score.py`, `report.py`).
60 OpenHands trajectories. Two tasks:
- b. recall probes: ask for a fact from earlier in the session;
- c. next tool call: compare each arm's call to FULL's own call.

Arms:
- FULL: no KVFlash.
- POOL8K, POOL16K: `--kvflash 8192` / `16384`, `KVFLASH_STATE_LADDER=1 KVFLASH_LIVE_RESTORE=1`. This is 2.0a session restore with no compaction.
- FULLB: FULL again on the first 20 trajectories, as a control.

## Parity

| Axis | Value |
|---|---|
| Model | `Qwen3.8-27B-UD-Q4_K_M.gguf`, same file on all arms |
| Binary | `luce_server` md5 13c65b5cdce446f00f348a4101432de9, same on all arms |
| KV type | default on all arms |
| `--max-ctx` | 52224 on all arms |
| Sampling | temperature 0, seed 1 |
| Caching | prefix-cache hit fraction on decision requests: 0.984 FULL, 0.985 POOL8K, 0.985 POOL16K |
| Decision-request prefilled tokens | 2,106,075 / 2,147,699 / 2,141,276 (the pool arms reprocess about 2% more) |

## Findings

- **The noise floor is zero.** FULLB matches FULL on 330/330 probes and 393/393 decisions, so every drop in the pool arms comes from paging.
- **Recall:**
  - POOL8K keeps 46.5% of the facts FULL knew (−53.5 points); POOL16K keeps 75.6% (−24.4 points).
  - Facts inside the pool mostly survive: 87-96%.
  - Facts older than the pool are lost: POOL8K 12% at 8-16K tokens back and 2% beyond 16K; POOL16K 6% beyond 16K.
- **Next tool call:** POOL8K agrees with FULL 64.6% of the time, POOL16K 76.2%.
- **Speed, like for like.** Only the decision requests are the same 1,438 on FULL, POOL8K and POOL16K. Medians from `score.json` `timing.*.decision`:

| Arm | Prefill tok/s | Decode tok/s | Wall s |
|---|---|---|---|
| FULL | 816 | 36.9 | 3.80 |
| POOL8K | 1096 | 42.4 | 3.25 |
| POOL16K | 1020 | 40.4 | 3.49 |

  The "all requests" medians in the report below are not like for like. The pool arms skip the 903 probes FULL got wrong, and FULLB is a subset.
- **Bar for 2.0b compaction:** win back recall beyond the pool while staying close to the pool arms' speed.

Rows: `<ARM>.jsonl`, one per request. Scores: `score.json`. Retained probe set: `retained.json`. Server logs are kept on `/mnt/kimi-k3/artifacts/agentic-eval/full`.

## Report (`report.py`)

### Counts
- Trajectories: 60. Probe candidates 1854, FULL attempted 1854, retained (FULL correct) 951 (yield 51.3%).
- Decision points 1438; FULL produced a parsable call at 1204 (excluded 234, all `finish=length` at 256 tokens).
- FULL vs gold (OpenHands/Qwen3-Coder call) full key 780/1204, tool name 881/1204.
- Request errors per arm: {'FULL': 0, 'POOL8K': 0, 'POOL16K': 0, 'FULLB': 0}

Retained probes by type (retained / attempted): defname 174/181, line 452/808, traceback 22/23, failed_test 18/18, cmdline_path 145/367, cmdline_text 140/457

### b. Recall retention, P(correct | FULL correct)
| arm | n | correct | retention | Wilson 95% | arm - FULL (cluster bootstrap 95%) | McNemar b/c, exact p |
|---|---|---|---|---|---|---|
| POOL8K | 951 | 442 | 46.5% | 43.3-49.7% | -53.5 pts [-57.6, -49.8] | 509/0, 1.2e-153 |
| POOL16K | 951 | 719 | 75.6% | 72.8-78.2% | -24.4 pts [-27.2, -21.8] | 232/0, 2.9e-70 |
| FULLB | 330 | 330 | 100.0% | 98.8-100.0% | +0.0 pts [+0.0, +0.0] | 0/0, 1 |

Retention by type (correct/n):
| arm | defname | line | traceback | failed_test | cmdline_path | cmdline_text |
|---|---|---|---|---|---|---|
| POOL8K | 85/174 (49%) | 241/452 (53%) | 11/22 (50%) | 8/18 (44%) | 52/145 (36%) | 45/140 (32%) |
| POOL16K | 134/174 (77%) | 349/452 (77%) | 17/22 (77%) | 14/18 (78%) | 103/145 (71%) | 102/140 (73%) |
| FULLB | 56/56 (100%) | 156/156 (100%) | 6/6 (100%) | 5/5 (100%) | 45/45 (100%) | 62/62 (100%) |

Retention by P (K tokens) (correct/n):
| arm | 12 | 24 | 36 |
|---|---|---|---|
| POOL8K | 224/241 (93%) | 153/351 (44%) | 65/359 (18%) |
| POOL16K | 230/241 (95%) | 329/351 (94%) | 160/359 (45%) |
| FULLB | 86/86 (100%) | 123/123 (100%) | 121/121 (100%) |

Retention by distance of fact from cut (tokens) (correct/n):
| arm | 0-4096 | 4096-8192 | 8192-16384 | 16384-inf |
|---|---|---|---|---|
| POOL8K | 275/296 (93%) | 125/144 (87%) | 38/304 (12%) | 4/207 (2%) |
| POOL16K | 283/296 (96%) | 131/144 (91%) | 293/304 (96%) | 12/207 (6%) |
| FULLB | 108/108 (100%) | 55/55 (100%) | 106/106 (100%) | 61/61 (100%) |

### c. Next-tool-call agreement with FULL's own call
| arm | n | agree (name+key) | agreement | Wilson 95% | arm - FULL (cluster bootstrap 95%) | name-only agreement | name-only arm - FULL | arm parse failures | McNemar b/c, exact p |
|---|---|---|---|---|---|---|---|---|---|
| POOL8K | 1204 | 778 | 64.6% | 61.9-67.3% | -35.4 pts [-38.2, -32.6] | 70.2% | -29.8 pts [-32.4, -27.3] | 209 | 426/0, 1.2e-128 |
| POOL16K | 1204 | 918 | 76.2% | 73.8-78.6% | -23.8 pts [-26.4, -21.2] | 79.7% | -20.3 pts [-22.9, -17.8] | 164 | 286/0, 1.6e-86 |
| FULLB | 393 | 393 | 100.0% | 99.0-100.0% | +0.0 pts [+0.0, +0.0] | 100.0% | +0.0 pts [+0.0, +0.0] | 0 | 0/0, 1 |

### Server-side timings (usage.timings: prefill_ms, decode_ms; wall = prefill + decode)
| arm | requests | prefill s (sum) | decode s (sum) | wall s (sum) | wall per trajectory | median prefill s | median decode s | median wall s | median prefill tok/s* | median decode tok/s* | prefilled tokens (sum) |
|---|---|---|---|---|---|---|---|---|---|---|---|
| FULL | 3447 | 3243 | 5270 | 8513 | 142 s | 0.37 | 0.70 | 1.09 | 819 | 37.9 | 2530265 |
| POOL8K | 2544 | 2462 | 4355 | 6817 | 114 s | 0.56 | 1.22 | 2.15 | 1086 | 42.6 | 2737832 |
| POOL16K | 2544 | 2624 | 4672 | 7296 | 122 s | 0.60 | 1.26 | 2.27 | 1011 | 40.6 | 2714730 |
| FULLB | 864 | 981 | 1623 | 2604 | 130 s | 0.67 | 1.33 | 2.53 | 847 | 38.5 | 832133 |

*tok/s medians over requests with prefill > 500 tokens and > 0.5 s (prefill) or >= 8 completion tokens and > 0.3 s (decode).

By request kind (sum wall s / median wall s):
| arm | probe | decision | primer |
|---|---|---|---|
| FULL | 1363 / 0.67 (n=1854) | 6901 / 3.80 (n=1438) | 249 / 0.58 (n=155) |
| POOL8K | 628 / 0.59 (n=951) | 5956 / 3.25 (n=1438) | 233 / 0.37 (n=155) |
| POOL16K | 685 / 0.65 (n=951) | 6374 / 3.49 (n=1438) | 237 / 0.37 (n=155) |
| FULLB | 220 / 0.61 (n=330) | 2294 / 3.77 (n=480) | 90 / 0.73 (n=54) |
