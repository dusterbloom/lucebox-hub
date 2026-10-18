# Selector v2 with the GPU scorer end to end on the 3090 (2026-10-18): WIN, cost line passes

Preregistered in `PREREG_gpu.md` (commit 09c93d26, before any row, no amendment). One binary for every phase:
selv2gpu-cuda `luce_server` 20e33b31 (code 2f1247c0). Harness QK16KND, 951 probes, 16K pool, k=6, mode 4, exact scorer.
gpu = ON6 env + `LUCE_KVFLASH_SELECT_QUERIES=64 LUCE_KVFLASH_SELECT_UNIT=2 LUCE_KVFLASH_SCORER_DEVICE=gpu`; on6ref = ON6 env.

Validity:

- **Shadow** (smoke ids, GPU and CPU scorer both run): 6 of 6 reselects pick the same 30 units; 0 structural mismatches.
- **gpu log gate:**
  - 827 of 827 reselects ran selector v2 (64 queries on 822 of them);
  - 827 `gpu scorer:` lines and no CPU fallback;
  - no failed reselect, no disk_hit.
- **Structural:** 1106 of 1106 requests, both arms.
- **Flags-off identity:** on6ref answers equal the 497cbeca on6ref on 951 of 951 (`cmp_noid.py`). The gate's 950 of 951 is one tool call whose random `call_` id differs; content and arguments are equal. on6ref also equals prs8 ON6 on 951 of 951.
- **Errors:** 951 of 951 probes, 0 errors, median pt 349 on both arms.

| | on6ref | gpu | gpu - on6ref (95% cluster bootstrap) | CPU-scorer v2 (`../../e2e`, 497cbeca) |
|---|---|---|---|---|
| beyond (207) | 138 | **177** | +18.8 [+11.8, +25.8] pts; discordant 45 vs 6, sign p 1.8e-8 | 176 |
| within (744) | 709 | 710 | +0.1 [-1.0, +1.3] | 712 |
| within excl. 27 (717) | 685 | 684 | -0.1 [-1.3, +1.0] | 686 |
| overall (951) | 847 | **887** | +4.2 [+2.4, +6.1]; discordant 60 vs 20 | 888 |
| beyond facts fully held (206 out of window) | 147 | 185 | missed 57 -> 15 | 186 |
| scorer ms per reselect (median / max) | 33.5 / 60.1 | **80.9 / 238.5** | | 1720 / 3094 |
| prefill ms (median) | 814 | 888 | +74 | 2190 |
| decode tok/s (median) | 43.6 | 43.0 | -1.4% | 43.8 |
| client wall s (median) | 1.37 | 1.42 | +0.06 | 2.81 |

The CPU-scorer column's speeds were measured on another day and session. Read them as context, not as a paired comparison.

On the answers themselves, the gpu arm agrees with the CPU-scorer v2 arm on 932 of 951 probes when tool-call ids are ignored (predicted >= 95%).

**Predictions:**
- Inside the predicted range: accuracy, prefill, wall, decode, and the scorer max (<= 450 ms).
- Better than predicted: the scorer median, 80.9 ms against 90-140.

**Verdict:** WIN. The cost line passes (median 80.9 ms <= 200 ms), so selector v2 with the GPU scorer is shippable as the default on this config.

**Files:**
- `run_gpu.sh`, `gate_gpu.py`: phases and checks 1-3.
- `analyze_e2e.py`: unchanged from `../../e2e`.
- `analyze_e2e.out`, `report_e2e.json`: the read-out.
- `cmp_noid.py`: answer agreement with tool-call ids ignored.
- `run_gpu.log`, `provenance_gpu.txt`: the run log and binary hashes.

Arm dirs stay on /mnt/kimi-k3/split-work/selv2/gpu (v2/ = gpu, on6ref/, shadow_pre/).
