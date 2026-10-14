# Gate A on R9700 (lucebox6), session-restore-main 17ac4dae - 2026-10-14

Result: PASS. Plain 10/10 identical, DFlash2 10/10 identical (first run, no rerun needed), scenarios S1/S2/S3 all probes EQUAL to cold reference.

## Parity contract
GPU: R9700 (HIP_VISIBLE_DEVICES=0), lucebox6. One luce_server at a time (flock; pgrep empty before each run). Model ~/models/Qwen3.8-27B-UD-Q4_K_M.gguf, default KV type, --max-ctx 131072, --kvflash 8192, temp 0 seed 1, turn-1 ~11.6k tokens, 10 turns, max_tokens 80. Arm A = KVFLASH_SESSION=0 (cold re-prefill), arm B = KVFLASH_SESSION=1 (session restore), same prompts. DFlash2 drafter ~/models/Qwen3.8-27B-DFlash2-Q8_0.gguf on both arms. Scripts: ~/kvflash-gate/run_gate.sh, run_scenarios.sh (outputs in gateA_r9700/).
Binaries: plain luce_server-17ac4dae md5 5c3ecaa7b5022f2f3bc9902ccdb56741; dflash2 luce_server-17ac4dae-d2 md5 a5101df1459b7d6a35e9fd28a99d0e9a.

## Server-side stats (turn 1 = cold in both arms; warm-up request excluded; n=10)
| gate | arm | prefill s med/mean/min | decode tok/s med/mean/min/max | wall s med/mean/min |
|---|---|---|---|---|
| plain | A | 12.90 / 12.96 / 11.70 | 26.9 / 26.99 / 26.9 / 27.8 | 13.90 / 13.98 / 12.60 |
| plain | B | 0.55 / 1.74 / 0.40 | 27.1 / 27.14 / 27.0 / 27.9 | 1.75 / 2.88 / 1.60 |
| dflash2 | A | 13.60 / 13.61 / 11.90 | 65.8 / 64.36 / 46.9 / 70.1 | 14.00 / 14.03 / 12.30 |
| dflash2 | B | 0.75 / 1.90 / 0.50 | 66.15 / 64.76 / 47.8 / 70.6 | 1.75 / 2.81 / 1.40 |

B means include the cold turn 1 (12.4-12.5 s prefill); medians are the headline. Decode min for dflash2 is the cold turn 1 (spec warm-up).
DFlash2 draft acceptance: A 268/312 = 85.9%, B 268/312 = 85.9% (identical, as expected for byte-identical outputs).

## pt / prefix_len per turn (pt identical on both arms)
pt: 11606, 11958, 12404, 12613, 12898, 13265, 13711, 13921, 14205, 14597
A prefix_len: 0 on all 10 turns (both gates).
B prefix_len: 0, 11602, 11954, 12400, 12609, 12894, 13261, 13707, 13917, 14201 (both gates; replay 213-450 tokens/turn).

## Scenarios (luce_server-17ac4dae, plain)
S1: P1 EQUAL, P2 EQUAL. S2: P EQUAL, P2 EQUAL. S3: T EQUAL. All PASS (scenarios.out; live-restore and slot 63 CACHE hits confirmed in logs).

## Comparison with 3090 (ratios only; different GPU and binaries)
| metric | 3090 plain | R9700 plain | 3090 dflash2 | R9700 dflash2 |
|---|---|---|---|---|
| prefill A/B | 9.45/0.50 = 18.9x | 12.90/0.55 = 23.5x | 9.75/0.45 = 21.7x | 13.60/0.75 = 18.1x |
| wall A/B | 10.89/1.18 = 9.2x | 13.90/1.75 = 7.9x | 10.15/1.05 = 9.7x | 14.00/1.75 = 8.0x |
| decode B/A | 57.7/57.4 = 1.005 | 27.1/26.9 = 1.007 | 83.2/83.6 = 0.995 | 66.15/65.8 = 1.005 |

Same qualitative picture: restore removes about 95% of warm-turn prefill, decode unchanged between arms. On the 3090 dflash2 B accepted/proposed was 267/320 vs A 268/312; here A and B are identical. Absolute speeds differ by GPU and are not comparable.

## Raw artifacts
gateA_r9700/{plain,dflash2}-gateA/ (cmp.out = cmp_gate.py output, gate4_{A,B}.log server logs, provenance.txt), gateA_r9700/scenarios-gateA/ (scenarios.out, per-scenario live/cold logs and json). Full prompt JSON (gate4_*.json) left on lucebox6 in ~/kvflash-gate/out/.
