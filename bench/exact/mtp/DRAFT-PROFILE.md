# MTP draft-call profile (2026-10-09, lucebox4, Q8 GGUF + Q8_0 MTP sidecar, 16K ctx, k=3)

Env-gated steady_clock checkpoints in `mtp_forward_batch`. Scratch build only, not committed. Run with:
`LUCE_QWEN_MTP_PROF=1 smoke_qwen4exp_forward <gguf> 16384 --mtp 64 --mtp-draft 3 --chunk 2048`

| rank | calls | nodes | build ms | gallocr ms | upload ms | compute+sync ms | total ms |
|---|---|---|---|---|---|---|---|
| 0 (catch-up, n<=k+1) | 35 | 151 | 0.019 | 0.032 | 0.047 | 2.344 | 2.442 |
| >0 (chained T=1) | 37 | 149 | 0.011 | 0.022 | 0.015 | 2.265 | 2.312 |

- **Window cost:** a k=3 window costs ~7.1 ms, or 2.35 ms per drafted token. The weight-byte floor is ~2 ms per call: ~420 MB at 220 GB/s.
- **Host overhead:** graph build, gallocr and upload together are <0.1 ms per call.
- **Old 5.9 ms figure:** the earlier `draft_ms=5.9` is stale, from an older build. The draft cost is not the MTP gap. Strata's whole draft pass is 8.8-9.1 ms.
- **Requantizing the drafter experts:** the ten Q8_0 experts are ~50 MB per call, so this buys <=0.2 ms per call.
- **Where the gap is:** verify, ~79 ms vs ~55 ms at T=4.
