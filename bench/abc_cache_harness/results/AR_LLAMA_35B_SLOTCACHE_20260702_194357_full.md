# ABC Cache Harness — AR_LLAMA_35B_SLOTCACHE
Generated: 2026-07-02 17:51 UTC
Mode: FULL (all turns, N=1)

## Provenance
```
{
  "binary": "/home/peppi/llama.cpp/build-cuda/bin/llama-server",
  "binary_sha256": "feedd55326b13fd4156dd0c7d7086fb94201cceeda5ef3eabc43fb26e2adc06b",
  "git_branch": "bench/upstream-pr469-473-plus-468",
  "git_commit": "e0e8573ac59a43fdb108005ef2bf9082dec3c629",
  "arm": "AR_LLAMA_35B_SLOTCACHE",
  "arm_description": "llama.cpp 35B-A3B Q4_K_M + prompt/cache-reuse/checkpoints + slot save/restore; pure AR; q4_0 KV",
  "arm_extra_args": [
    "-fa",
    "on",
    "--cache-prompt",
    "--cache-reuse",
    "256",
    "--ctx-checkpoints",
    "64",
    "--checkpoint-min-step",
    "512",
    "--cache-ram",
    "8192",
    "--cache-idle-slots",
    "--slot-save-path",
    "/tmp/llama35b_slot_cache"
  ],
  "arm_env": {},
  "server_type": "llama_cpp",
  "model_target": "/home/peppi/models/qwen3.6-35b-a3b/Qwen3.6-35B-A3B-UD-Q4_K_M.gguf",
  "model_draft_decode": "/home/peppi/models/qwen3.6-27b-dflash/dflash-draft-3.6-bf16-reconverted.gguf",
  "model_draft_prefill": "/home/peppi/models/Qwen3-0.6B-BF16.gguf",
  "chat_template": "/home/peppi/models/qwen3-coder-chat-template.jinja",
  "max_ctx": 131072,
  "cache_type_k": "q4_0",
  "cache_type_v": "q4_0",
  "temperature": 0.7,
  "seed_requested": 42,
  "n_repeats": 1,
  "smoke": false,
  "restart_per_turn": false,
  "quality_probe": false,
  "pin_decode_length": false,
  "trace_max_tokens_unique": [],
  "pin_decode_tokens_process": null,
  "pin_decode_mechanism": null,
  "port": 19099,
  "trace": "bench/abc_cache_harness/traces/deep_tool_structured_38_cap2048.jsonl",
  "n_turns_in_trace": 38,
  "timestamp_utc": "2026-07-02T17:43:57.776156+00:00",
  "slot_save_dir": "/tmp/llama35b_slot_cache",
  "slot_save_file": "ar35b_slot.bin"
}
```

## Run Quality

- ok: True
- censored: False
- valid_turns: 38/38
- error_count: 0
- tool_expected_valid: 20/38
- unexpected_tool_call_rate: None

## Per-Turn Cache Trace
turn      pt  eff_in  prefix_len  fresh_pf     hr   pf_s  pf_tps   out/req  dec_tps  mode  accept  pflash%  tool  wall_s
------------------------------------------------------------------------------------------------------------------------
   1   31484    None        None     31484      ?  10.90    2888   44/2048    105.2  None      AR        -     Y    11.4
   2    4342    None        None      4342      ?   1.73    2516  144/2048    101.9  None      AR        -     N    3.21
   3    2845    None        None      2845      ?   1.19    2393   60/2048    101.5  None      AR        -     Y    1.86
   4    9510    None        None      9510      ?   4.00    2379 1283/2048     91.3  None      AR        -     N   20.07
   5    2089    None        None      2089      ?   1.00    2097   72/2048     94.4  None      AR        -     Y    1.86
   6    2050    None        None      2050      ?   0.97    2123  190/2048     94.4  None      AR        -     N    3.09
   7    9012    None        None      9012      ?   4.03    2236  636/2048     86.9  None      AR        -     N   11.45
   8    4847    None        None      4847      ?   2.38    2033 2048/2048     87.4  None      AR        -     Y   27.86
   9    3493    None        None      3493      ?   1.71    2042 1727/2048     85.2  None      AR        -     N    24.0
  10    1007    None        None      1007      ?   0.52    1935  124/2048     89.7  None      AR        -     N    2.02
  11     214    None        None       214      ?   0.20    1081 2048/2048     84.9  None      AR        -     Y   26.33
  12     178    None        None       178      ?   0.18     989 2048/2048     87.1  None      AR        -     Y    23.8
  13     231    None        None       231      ?   0.22    1030  100/2048     84.4  None      AR        -     Y    1.53
  14     105    None        None       105      ?   0.16     647 1568/2048     85.9  None      AR        -     Y   20.41
  15    2413    None        None      2413      ?   1.19    2023 1176/2048     83.3  None      AR        -     Y   15.43
  16    2732    None        None      2732      ?   1.41    1937  179/2048     82.9  None      AR        -     N    5.67
  17    2547    None        None      2547      ?   1.34    1903  165/2048     80.9  None      AR        -     Y     3.5
  18     552    None        None       552      ?   0.38    1456   43/2048     82.9  None      AR        -     Y    1.07
  19     351    None        None       351      ?   0.27    1325  116/2048     82.3  None      AR        -     Y     1.8
  20     133    None        None       133      ?   0.17     769  384/2048     81.2  None      AR        -     N    5.03
  21      80    None        None        80      ?   0.14     582  176/2048     81.4  None      AR        -     N    2.43
  22    9125    None        None      9125      ?   4.69    1946  322/2048     77.8  None      AR        -     N    8.98
  23    3611    None        None      3611      ?   1.95    1848 2048/2048     74.9  None      AR        -     N   31.03
  24    1092    None        None      1092      ?   0.70    1562 1767/2048     74.3  None      AR        -     N   26.27
  25     331    None        None       331      ?   0.29    1156 1564/2048     74.1  None      AR        -     Y   23.18
  26    3364    None        None      3364      ?   1.91    1760  390/2048     73.7  None      AR        -     Y    7.36
  27    1005    None        None      1005      ?   0.64    1581   99/2048     73.2  None      AR        -     N    2.14
  28    1752    None        None      1752      ?   1.09    1606  149/2048     71.9  None      AR        -     Y    3.32
  29    2704    None        None      2704      ?   1.64    1653   58/2048     72.6  None      AR        -     N     2.6
  30    1016    None        None      1016      ?   0.64    1590  747/2048     71.5  None      AR        -     Y   12.88
  31    4417    None        None      4417      ?   2.61    1693 1620/2048     70.1  None      AR        -     Y   25.88
  32    3687    None        None      3687      ?   2.24    1645 1096/2048     69.6  None      AR        -     Y    19.8
  33    3094    None        None      3094      ?   1.87    1653  246/2048     69.5  None      AR        -     N    5.59
  34    2491    None        None      2491      ?   1.59    1569  253/2048     68.8  None      AR        -     N    5.44
  35     348    None        None       348      ?   0.31    1121   63/2048     68.7  None      AR        -     N    1.41
  36    1058    None        None      1058      ?   0.76    1398 1528/2048     67.6  None      AR        -     N   25.19
  37     377    None        None       377      ?   0.33    1141 1195/2048     68.0  None      AR        -     Y   19.71
  38    2228    None        None      2228      ?   1.39    1602  290/2048     67.2  None      AR        -     Y    5.88

## Arm Aggregate
```
{
  "ok": true,
  "censored": false,
  "expected_turns": 38,
  "valid_turns": 38,
  "total_attempts": 38,
  "error_count": 0,
  "total_wall_s": 440.5,
  "total_prefill_s": 58.718,
  "total_decode_s": 353.721,
  "sum_prompt_tokens": 121915,
  "sum_effective_in_tokens": null,
  "sum_fresh_prefill_tokens": 121915,
  "sum_requested_out_tokens": 77824,
  "sum_out_tokens": 27766,
  "out_token_mismatch_count": 34,
  "out_tokens_match_requested": false,
  "pin_decode_ok": false,
  "mean_cache_hit_ratio": null,
  "mean_prefill_tps": 1655.461,
  "mean_decode_tps": 80.753,
  "weighted_prompt_prefill_tps": 2076.28,
  "weighted_effective_prefill_tps": null,
  "weighted_fresh_prefill_tps": 2076.28,
  "weighted_decode_tps": 78.497,
  "spec_engagement_rate": 0.0,
  "mean_accept_rate": null,
  "mean_disk_hit_rate": null,
  "sum_tool_expected_turns": 38,
  "sum_tool_expected_valid_turns": 20,
  "tool_call_valid_rate": 0.526,
  "unexpected_tool_call_rate": null,
  "charbench_valid_rate": null
}
```
