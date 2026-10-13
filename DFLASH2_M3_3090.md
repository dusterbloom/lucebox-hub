# DFlash 2 (Qwen3.8-27B-DFlash2 upstream GGUF) on RTX 3090, M3 evidence

Date 2026-10-13. Branch `dflash2/qwen38-upstream-gguf` (based on fork/main 5f402c61), commit with M1+M2: 88473180.
Raw logs and per-request JSON: `/mnt/kimi-k3/artifacts/dflash2/m3_*`.

## Parity contract (bench-engine-parity, both arms)

| Axis | Value (identical on both arms) |
|---|---|
| Binary | `/mnt/kimi-k3/build/dflash2-cuda/luce_server`, md5 2e7f074e4c2655b3fec5c32567dfd21c on both |
| Target | `/mnt/kimi-k3/home-models/qwen3.8-27b/Qwen3.8-27B-UD-Q4_K_M.gguf` (sha256 matches expected.sha256) |
| Drafter (spec arm only) | `Qwen3.8-27B-DFlash2-Q8_0.gguf` (upstream, arch `dflash`) |
| max-ctx / KV | `--max-ctx 4096`, `--cache-type-k q8_0 --cache-type-v q8_0` |
| Sampling | temperature 0, seed 1, max_tokens 2048, streaming, thinking template default |
| Caching | `--prefix-cache-slots 0`; every request `prefix_len=0`, `cached_prefix_tokens=0`, prefilled tokens equal on both arms (82/43/72/51/125/72/58/87) |
| Prompts | first 8 of `lucebox_bench/workloads/gsm8k.jsonl` (GSM8K test order, gsm_01 = Janet's ducks) |
| GPU | RTX 3090, one server at a time; foreign-process monitor (5 s) saw only the server's own compute app during both runs |

## Hparams asserted at load (from the spec-arm server log)

```
[draft GGUF] upstream DFlash 2: block=8 conv=k2/g16 selector=r256/k16 mask_id=248070 window=2048 causal=0 capture_ids=5 19 33 47 61
[draft GGUF] non-causal attention: 5/5 layers sliding-window (window=2048)
[draft]  target capture layers from drafter GGUF: 5 19 33 47 61
```

GGUF stores `dflash.target_layers=[6,20,34,48,62]`; the loader maps to capture ids `[5,19,33,47,61]`
(unit-tested, and the capture-id log line above is the runtime assertion).

## Result (medians over the 8 prompts)

| Arm | Prefill tok/s | Decode tok/s | Wall s | Acceptance length |
|---|---|---|---|---|
| AR (no drafter) | 932.2 | 49.06 | 4.95 | n/a |
| DFlash 2 spec (block 8) | 938.9 | 142.9 | 1.90 | 6.10 tokens/step (2170 tokens / 356 steps, per-request 4.85 to 7.28) |

Decode speedup 2.91x, wall speedup 2.61x. The PR's 5.08 (Q8_0) was sampled at temperature 1.0; ours is greedy, so
6.10 is above that loose range, not a like-for-like comparison. Context stays under 400 tokens in this test, so the
2048 sliding window never truncates anything here; the per-query window mask is covered by the unit test only.

Prefill tok/s uses the server-reported `prefill_ms` and prefilled tokens; the first request of each run is a warm-up
(177 and 125 tok/s) and the median is unaffected.

## Byte-identity versus AR: NOT met (4 of 8 identical)

| Run vs AR | gsm_01..08 |
|---|---|
| AR rerun | `========` |
| spec (block 8) | `X=X=X==X` |
| spec rerun | `X=X=X==X` |
| spec `--no-fast-rollback` | `X==XXX=X` |
| spec `--draft-block-size 6` | `X=X=XX=X` |

Both arms are individually deterministic (AR == AR rerun, spec == spec rerun). The divergence points move with the
verify configuration (different prompts diverge for fast-rollback off and for block 6), and every divergence is a
mid-sentence synonym choice (for example "eggs laid per day" vs "eggs used personally"). That is the signature of
the batched-verify versus one-token-decode FP effect that `docs/specs/speed-profile.md` documents (verify scores
the draft as a batch, AR decodes one token; GEMM reduction order differs; a near-tie argmax flips and everything after
follows). The drafter cannot change committed tokens, because every committed token comes from the target's verify
pass, and that path is untouched by this change. No logprob endpoint exists to measure the logit gap at the first
mismatch, so the near-tie classification is by inference, not measurement. Treat the byte-identity criterion as
unmet and open for triage; the gap measurement is the follow-up the speed-profile doc already asks for.

## Control: DFlash 1 drafter (Arm D1) decides it is batched-verify numerics

Same binary, prompts and flags; drafter `/home/peppi/models/qwen3.6-27b-dflash/dflash-draft-3.6-q8_0.gguf` on the Qwen3.8
target (loaded fine; avg 84 tok/s decode median 83.3). Raw: `m3_d1_*`. First divergent position per pair (character
index in reasoning+text, approximate token index in parentheses; `=` is identical; no token ids are exposed):

| Prompt | D1 vs AR | D1 vs DFlash2 | DFlash2 vs AR |
|---|---|---|---|
| gsm_01 | chr125 (~38) | = | chr125 (~38) |
| gsm_02 | = | = | = |
| gsm_03 | = | chr348 (~113) | chr348 (~113) |
| gsm_04 | chr61 (~19) | chr61 (~19) | = |
| gsm_05 | chr630 (~190) | chr133 (~40) | chr133 (~40) |
| gsm_06 | chr38 (~14) | chr38 (~14) | = |
| gsm_07 | = | = | = |
| gsm_08 | chr322 (~108) | chr167 (~55) | chr167 (~56) |

D1 diverges from AR on 5 of 8 prompts (DFlash2: 4 of 8), at similar shallow depths (tens to low hundreds of tokens).
Conclusion: the mismatch is inherent to batched verify numerics on this target, not DFlash2-specific. Code left alone.

## Optional tensor byte-compare vs our own converter: skipped

The HF `Qwen3.8-27B-DFlash2` safetensors are not present locally (only the GGUF is), so there was nothing to convert;
a download was not made.

## Unit tests

Baseline on this tree before changes (ctest, 866 tests): 863 pass, 3 fail (`server_unit_entrypoint`, and two
`test_server_unit` cases that are load-flaky and passed on rerun). After (868 tests): 868 minus `server_unit_entrypoint`
pass; `deepseek4_unit` timed out once under parallel load and passed on rerun. New: `test_dflash2_upstream_schema`
(42 checks) and `test_draft_attn_mask` (1770 checks).
