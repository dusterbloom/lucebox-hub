# ABC Cache Harness — AR_LLAMA_35B_SLOTCACHE
Generated: 2026-07-05 10:20 UTC
Mode: FULL (all turns, N=1)

## Provenance
```
{
  "binary": "/home/peppi/llama.cpp/build-cuda/bin/llama-server",
  "binary_sha256": "feedd55326b13fd4156dd0c7d7086fb94201cceeda5ef3eabc43fb26e2adc06b",
  "git_branch": "codex/deathmatch-current-main-review",
  "git_commit": "b44c872b725061ca16a546c9d8f332784ddf6ae6",
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
  "pin_decode_length": true,
  "trace_max_tokens_unique": [
    256
  ],
  "pin_decode_tokens_process": 256,
  "pin_decode_mechanism": "ignore_eos+n_predict",
  "power_sampling_enabled": true,
  "power_sample_interval_s": 1.0,
  "power_gpu_index": 0,
  "port": 19099,
  "trace": "/tmp/luce_mixed_candidate_0_fixed_38.jsonl",
  "n_turns_in_trace": 38,
  "timestamp_utc": "2026-07-05T10:16:45.899137+00:00",
  "slot_save_dir": "/tmp/llama35b_slot_cache",
  "slot_save_file": "ar35b_slot.bin",
  "power": {
    "gpu_index": 0,
    "sample_interval_s": 1.0,
    "duration_s": 212.196,
    "sample_count": 201,
    "valid_sample_count": 201,
    "energy_j": 58236.406,
    "avg_power_w": 274.446,
    "max_power_w": 301.58,
    "mean_gpu_util_pct": 80.741,
    "max_memory_used_mib": 23274,
    "first_error": null,
    "samples": [
      {
        "power_w": 121.64,
        "gpu_util_pct": 0,
        "memory_used_mib": 953,
        "t_s": 0.0
      },
      {
        "power_w": 120.62,
        "gpu_util_pct": 0,
        "memory_used_mib": 1208,
        "t_s": 1.159
      },
      {
        "power_w": 120.01,
        "gpu_util_pct": 0,
        "memory_used_mib": 1208,
        "t_s": 2.221
      },
      {
        "power_w": 119.99,
        "gpu_util_pct": 0,
        "memory_used_mib": 1208,
        "t_s": 3.278
      },
      {
        "power_w": 119.79,
        "gpu_util_pct": 0,
        "memory_used_mib": 1208,
        "t_s": 4.34
      },
      {
        "power_w": 119.62,
        "gpu_util_pct": 0,
        "memory_used_mib": 1208,
        "t_s": 5.399
      },
      {
        "power_w": 119.54,
        "gpu_util_pct": 0,
        "memory_used_mib": 1208,
        "t_s": 6.454
      },
      {
        "power_w": 119.49,
        "gpu_util_pct": 0,
        "memory_used_mib": 1208,
        "t_s": 7.508
      },
      {
        "power_w": 134.61,
        "gpu_util_pct": 93,
        "memory_used_mib": 21792,
        "t_s": 8.568
      },
      {
        "power_w": 152.15,
        "gpu_util_pct": 6,
        "memory_used_mib": 21792,
        "t_s": 9.628
      },
      {
        "power_w": 154.27,
        "gpu_util_pct": 39,
        "memory_used_mib": 21792,
        "t_s": 10.684
      },
      {
        "power_w": 155.78,
        "gpu_util_pct": 14,
        "memory_used_mib": 23248,
        "t_s": 11.745
      },
      {
        "power_w": 243.7,
        "gpu_util_pct": 100,
        "memory_used_mib": 23268,
        "t_s": 12.793
      },
      {
        "power_w": 300.2,
        "gpu_util_pct": 100,
        "memory_used_mib": 23268,
        "t_s": 13.848
      },
      {
        "power_w": 299.67,
        "gpu_util_pct": 100,
        "memory_used_mib": 23268,
        "t_s": 14.91
      },
      {
        "power_w": 300.03,
        "gpu_util_pct": 100,
        "memory_used_mib": 23268,
        "t_s": 15.958
      },
      {
        "power_w": 299.89,
        "gpu_util_pct": 100,
        "memory_used_mib": 23268,
        "t_s": 17.012
      },
      {
        "power_w": 300.39,
        "gpu_util_pct": 100,
        "memory_used_mib": 23268,
        "t_s": 18.065
      },
      {
        "power_w": 300.57,
        "gpu_util_pct": 100,
        "memory_used_mib": 23268,
        "t_s": 19.114
      },
      {
        "power_w": 300.86,
        "gpu_util_pct": 100,
        "memory_used_mib": 23268,
        "t_s": 20.164
      },
      {
        "power_w": 300.48,
        "gpu_util_pct": 100,
        "memory_used_mib": 23268,
        "t_s": 21.207
      },
      {
        "power_w": 299.91,
        "gpu_util_pct": 100,
        "memory_used_mib": 23268,
        "t_s": 22.254
      },
      {
        "power_w": 293.38,
        "gpu_util_pct": 86,
        "memory_used_mib": 23270,
        "t_s": 23.305
      },
      {
        "power_w": 300.24,
        "gpu_util_pct": 93,
        "memory_used_mib": 23270,
        "t_s": 24.356
      },
      {
        "power_w": 299.83,
        "gpu_util_pct": 93,
        "memory_used_mib": 23270,
        "t_s": 25.41
      },
      {
        "power_w": 246.78,
        "gpu_util_pct": 93,
        "memory_used_mib": 23270,
        "t_s": 26.457
      },
      {
        "power_w": 299.83,
        "gpu_util_pct": 92,
        "memory_used_mib": 23270,
        "t_s": 27.506
      },
      {
        "power_w": 268.37,
        "gpu_util_pct": 44,
        "memory_used_mib": 23270,
        "t_s": 28.554
      },
      {
        "power_w": 277.56,
        "gpu_util_pct": 100,
        "memory_used_mib": 23272,
        "t_s": 29.601
      },
      {
        "power_w": 299.85,
        "gpu_util_pct": 93,
        "memory_used_mib": 23272,
        "t_s": 30.657
      },
      {
        "power_w": 299.65,
        "gpu_util_pct": 93,
        "memory_used_mib": 23272,
        "t_s": 31.72
      },
      {
        "power_w": 283.16,
        "gpu_util_pct": 42,
        "memory_used_mib": 23272,
        "t_s": 32.772
      },
      {
        "power_w": 259.84,
        "gpu_util_pct": 96,
        "memory_used_mib": 23272,
        "t_s": 33.814
      },
      {
        "power_w": 298.89,
        "gpu_util_pct": 91,
        "memory_used_mib": 23272,
        "t_s": 34.862
      },
      {
        "power_w": 301.44,
        "gpu_util_pct": 91,
        "memory_used_mib": 23272,
        "t_s": 35.906
      },
      {
        "power_w": 244.77,
        "gpu_util_pct": 90,
        "memory_used_mib": 23272,
        "t_s": 36.96
      },
      {
        "power_w": 297.67,
        "gpu_util_pct": 89,
        "memory_used_mib": 23272,
        "t_s": 38.018
      },
      {
        "power_w": 301.58,
        "gpu_util_pct": 93,
        "memory_used_mib": 23272,
        "t_s": 39.075
      },
      {
        "power_w": 244.92,
        "gpu_util_pct": 88,
        "memory_used_mib": 23272,
        "t_s": 40.129
      },
      {
        "power_w": 300.61,
        "gpu_util_pct": 92,
        "memory_used_mib": 23272,
        "t_s": 41.18
      },
      {
        "power_w": 299.98,
        "gpu_util_pct": 90,
        "memory_used_mib": 23272,
        "t_s": 42.232
      },
      {
        "power_w": 240.76,
        "gpu_util_pct": 100,
        "memory_used_mib": 23272,
        "t_s": 43.284
      },
      {
        "power_w": 298.98,
        "gpu_util_pct": 94,
        "memory_used_mib": 23272,
        "t_s": 44.342
      },
      {
        "power_w": 298.32,
        "gpu_util_pct": 91,
        "memory_used_mib": 23272,
        "t_s": 45.451
      },
      {
        "power_w": 299.85,
        "gpu_util_pct": 92,
        "memory_used_mib": 23272,
        "t_s": 46.51
      },
      {
        "power_w": 243.67,
        "gpu_util_pct": 11,
        "memory_used_mib": 23272,
        "t_s": 47.57
      },
      {
        "power_w": 298.85,
        "gpu_util_pct": 91,
        "memory_used_mib": 23272,
        "t_s": 48.627
      },
      {
        "power_w": 299.55,
        "gpu_util_pct": 92,
        "memory_used_mib": 23272,
        "t_s": 49.68
      },
      {
        "power_w": 253.61,
        "gpu_util_pct": 0,
        "memory_used_mib": 23272,
        "t_s": 50.748
      },
      {
        "power_w": 291.12,
        "gpu_util_pct": 92,
        "memory_used_mib": 23272,
        "t_s": 51.8
      },
      {
        "power_w": 299.26,
        "gpu_util_pct": 92,
        "memory_used_mib": 23272,
        "t_s": 52.854
      },
      {
        "power_w": 249.1,
        "gpu_util_pct": 0,
        "memory_used_mib": 23272,
        "t_s": 53.909
      },
      {
        "power_w": 294.94,
        "gpu_util_pct": 89,
        "memory_used_mib": 23274,
        "t_s": 54.959
      },
      {
        "power_w": 300.19,
        "gpu_util_pct": 91,
        "memory_used_mib": 23274,
        "t_s": 56.015
      },
      {
        "power_w": 281.17,
        "gpu_util_pct": 75,
        "memory_used_mib": 23274,
        "t_s": 57.091
      },
      {
        "power_w": 265.57,
        "gpu_util_pct": 92,
        "memory_used_mib": 23274,
        "t_s": 58.138
      },
      {
        "power_w": 299.14,
        "gpu_util_pct": 89,
        "memory_used_mib": 23274,
        "t_s": 59.189
      },
      {
        "power_w": 290.22,
        "gpu_util_pct": 61,
        "memory_used_mib": 23274,
        "t_s": 60.246
      },
      {
        "power_w": 267.04,
        "gpu_util_pct": 91,
        "memory_used_mib": 23274,
        "t_s": 61.295
      },
      {
        "power_w": 299.4,
        "gpu_util_pct": 92,
        "memory_used_mib": 23274,
        "t_s": 62.352
      },
      {
        "power_w": 301.04,
        "gpu_util_pct": 92,
        "memory_used_mib": 23274,
        "t_s": 63.406
      },
      {
        "power_w": 244.87,
        "gpu_util_pct": 94,
        "memory_used_mib": 23274,
        "t_s": 64.461
      },
      {
        "power_w": 300.58,
        "gpu_util_pct": 92,
        "memory_used_mib": 23274,
        "t_s": 65.523
      },
      {
        "power_w": 299.73,
        "gpu_util_pct": 92,
        "memory_used_mib": 23274,
        "t_s": 66.574
      },
      {
        "power_w": 257.37,
        "gpu_util_pct": 0,
        "memory_used_mib": 23274,
        "t_s": 67.626
      },
      {
        "power_w": 298.74,
        "gpu_util_pct": 92,
        "memory_used_mib": 23274,
        "t_s": 68.674
      },
      {
        "power_w": 299.86,
        "gpu_util_pct": 92,
        "memory_used_mib": 23274,
        "t_s": 69.721
      },
      {
        "power_w": 300.74,
        "gpu_util_pct": 93,
        "memory_used_mib": 23274,
        "t_s": 70.78
      },
      {
        "power_w": 245.94,
        "gpu_util_pct": 100,
        "memory_used_mib": 23274,
        "t_s": 71.832
      },
      {
        "power_w": 297.6,
        "gpu_util_pct": 100,
        "memory_used_mib": 23274,
        "t_s": 72.876
      },
      {
        "power_w": 298.32,
        "gpu_util_pct": 92,
        "memory_used_mib": 23274,
        "t_s": 73.926
      },
      {
        "power_w": 299.71,
        "gpu_util_pct": 93,
        "memory_used_mib": 23274,
        "t_s": 74.974
      },
      {
        "power_w": 298.68,
        "gpu_util_pct": 93,
        "memory_used_mib": 23274,
        "t_s": 76.037
      },
      {
        "power_w": 245.58,
        "gpu_util_pct": 100,
        "memory_used_mib": 23274,
        "t_s": 77.09
      },
      {
        "power_w": 298.59,
        "gpu_util_pct": 93,
        "memory_used_mib": 23274,
        "t_s": 78.141
      },
      {
        "power_w": 299.98,
        "gpu_util_pct": 93,
        "memory_used_mib": 23274,
        "t_s": 79.196
      },
      {
        "power_w": 287.71,
        "gpu_util_pct": 54,
        "memory_used_mib": 23274,
        "t_s": 80.249
      },
      {
        "power_w": 254.03,
        "gpu_util_pct": 88,
        "memory_used_mib": 23274,
        "t_s": 81.299
      },
      {
        "power_w": 299.88,
        "gpu_util_pct": 92,
        "memory_used_mib": 23274,
        "t_s": 82.351
      },
      {
        "power_w": 300.37,
        "gpu_util_pct": 93,
        "memory_used_mib": 23274,
        "t_s": 83.402
      },
      {
        "power_w": 235.42,
        "gpu_util_pct": 94,
        "memory_used_mib": 23274,
        "t_s": 84.45
      },
      {
        "power_w": 299.17,
        "gpu_util_pct": 100,
        "memory_used_mib": 23274,
        "t_s": 85.502
      },
      {
        "power_w": 299.44,
        "gpu_util_pct": 91,
        "memory_used_mib": 23274,
        "t_s": 86.556
      },
      {
        "power_w": 300.32,
        "gpu_util_pct": 92,
        "memory_used_mib": 23274,
        "t_s": 87.612
      },
      {
        "power_w": 299.51,
        "gpu_util_pct": 93,
        "memory_used_mib": 23274,
        "t_s": 88.662
      },
      {
        "power_w": 243.11,
        "gpu_util_pct": 95,
        "memory_used_mib": 23274,
        "t_s": 89.708
      },
      {
        "power_w": 296.96,
        "gpu_util_pct": 89,
        "memory_used_mib": 23274,
        "t_s": 90.76
      },
      {
        "power_w": 299.97,
        "gpu_util_pct": 92,
        "memory_used_mib": 23274,
        "t_s": 91.811
      },
      {
        "power_w": 300.67,
        "gpu_util_pct": 91,
        "memory_used_mib": 23274,
        "t_s": 92.868
      },
      {
        "power_w": 240.52,
        "gpu_util_pct": 0,
        "memory_used_mib": 23274,
        "t_s": 93.92
      },
      {
        "power_w": 298.24,
        "gpu_util_pct": 93,
        "memory_used_mib": 23274,
        "t_s": 94.973
      },
      {
        "power_w": 300.57,
        "gpu_util_pct": 92,
        "memory_used_mib": 23274,
        "t_s": 96.025
      },
      {
        "power_w": 300.42,
        "gpu_util_pct": 93,
        "memory_used_mib": 23274,
        "t_s": 97.069
      },
      {
        "power_w": 241.71,
        "gpu_util_pct": 100,
        "memory_used_mib": 23274,
        "t_s": 98.121
      },
      {
        "power_w": 298.09,
        "gpu_util_pct": 100,
        "memory_used_mib": 23274,
        "t_s": 99.189
      },
      {
        "power_w": 299.92,
        "gpu_util_pct": 93,
        "memory_used_mib": 23274,
        "t_s": 100.253
      },
      {
        "power_w": 298.73,
        "gpu_util_pct": 93,
        "memory_used_mib": 23274,
        "t_s": 101.306
      },
      {
        "power_w": 296.99,
        "gpu_util_pct": 93,
        "memory_used_mib": 23274,
        "t_s": 102.358
      },
      {
        "power_w": 245.88,
        "gpu_util_pct": 95,
        "memory_used_mib": 23274,
        "t_s": 103.404
      },
      {
        "power_w": 300.36,
        "gpu_util_pct": 93,
        "memory_used_mib": 23274,
        "t_s": 104.456
      },
      {
        "power_w": 299.98,
        "gpu_util_pct": 92,
        "memory_used_mib": 23274,
        "t_s": 105.512
      },
      {
        "power_w": 277.64,
        "gpu_util_pct": 26,
        "memory_used_mib": 23274,
        "t_s": 106.565
      },
      {
        "power_w": 260.61,
        "gpu_util_pct": 90,
        "memory_used_mib": 23274,
        "t_s": 107.629
      },
      {
        "power_w": 299.05,
        "gpu_util_pct": 93,
        "memory_used_mib": 23274,
        "t_s": 108.681
      },
      {
        "power_w": 300.07,
        "gpu_util_pct": 90,
        "memory_used_mib": 23274,
        "t_s": 109.736
      },
      {
        "power_w": 299.4,
        "gpu_util_pct": 91,
        "memory_used_mib": 23274,
        "t_s": 110.788
      },
      {
        "power_w": 240.86,
        "gpu_util_pct": 100,
        "memory_used_mib": 23274,
        "t_s": 111.843
      },
      {
        "power_w": 298.46,
        "gpu_util_pct": 100,
        "memory_used_mib": 23274,
        "t_s": 112.897
      },
      {
        "power_w": 298.79,
        "gpu_util_pct": 93,
        "memory_used_mib": 23274,
        "t_s": 113.948
      },
      {
        "power_w": 299.81,
        "gpu_util_pct": 94,
        "memory_used_mib": 23274,
        "t_s": 114.999
      },
      {
        "power_w": 300.19,
        "gpu_util_pct": 93,
        "memory_used_mib": 23274,
        "t_s": 116.052
      },
      {
        "power_w": 254.16,
        "gpu_util_pct": 100,
        "memory_used_mib": 23274,
        "t_s": 117.097
      },
      {
        "power_w": 299.12,
        "gpu_util_pct": 100,
        "memory_used_mib": 23274,
        "t_s": 118.178
      },
      {
        "power_w": 297.38,
        "gpu_util_pct": 93,
        "memory_used_mib": 23274,
        "t_s": 119.229
      },
      {
        "power_w": 298.78,
        "gpu_util_pct": 93,
        "memory_used_mib": 23274,
        "t_s": 120.282
      },
      {
        "power_w": 299.36,
        "gpu_util_pct": 93,
        "memory_used_mib": 23274,
        "t_s": 121.337
      },
      {
        "power_w": 239.53,
        "gpu_util_pct": 14,
        "memory_used_mib": 23274,
        "t_s": 122.39
      },
      {
        "power_w": 296.93,
        "gpu_util_pct": 93,
        "memory_used_mib": 23274,
        "t_s": 123.45
      },
      {
        "power_w": 300.74,
        "gpu_util_pct": 100,
        "memory_used_mib": 23274,
        "t_s": 124.516
      },
      {
        "power_w": 298.34,
        "gpu_util_pct": 94,
        "memory_used_mib": 23274,
        "t_s": 125.565
      },
      {
        "power_w": 298.55,
        "gpu_util_pct": 90,
        "memory_used_mib": 23274,
        "t_s": 126.618
      },
      {
        "power_w": 299.94,
        "gpu_util_pct": 93,
        "memory_used_mib": 23274,
        "t_s": 127.668
      },
      {
        "power_w": 256.33,
        "gpu_util_pct": 0,
        "memory_used_mib": 23274,
        "t_s": 128.721
      },
      {
        "power_w": 291.18,
        "gpu_util_pct": 89,
        "memory_used_mib": 23274,
        "t_s": 129.774
      },
      {
        "power_w": 300.18,
        "gpu_util_pct": 93,
        "memory_used_mib": 23274,
        "t_s": 130.852
      },
      {
        "power_w": 299.7,
        "gpu_util_pct": 94,
        "memory_used_mib": 23274,
        "t_s": 131.915
      },
      {
        "power_w": 299.58,
        "gpu_util_pct": 94,
        "memory_used_mib": 23274,
        "t_s": 132.965
      },
      {
        "power_w": 239.74,
        "gpu_util_pct": 100,
        "memory_used_mib": 23274,
        "t_s": 134.014
      },
      {
        "power_w": 299.08,
        "gpu_util_pct": 100,
        "memory_used_mib": 23274,
        "t_s": 135.068
      },
      {
        "power_w": 295.68,
        "gpu_util_pct": 88,
        "memory_used_mib": 23274,
        "t_s": 136.119
      },
      {
        "power_w": 299.81,
        "gpu_util_pct": 94,
        "memory_used_mib": 23274,
        "t_s": 137.178
      },
      {
        "power_w": 299.94,
        "gpu_util_pct": 94,
        "memory_used_mib": 23274,
        "t_s": 138.229
      },
      {
        "power_w": 267.69,
        "gpu_util_pct": 42,
        "memory_used_mib": 23274,
        "t_s": 139.279
      },
      {
        "power_w": 266.76,
        "gpu_util_pct": 100,
        "memory_used_mib": 23274,
        "t_s": 140.328
      },
      {
        "power_w": 296.51,
        "gpu_util_pct": 94,
        "memory_used_mib": 23274,
        "t_s": 141.376
      },
      {
        "power_w": 299.07,
        "gpu_util_pct": 91,
        "memory_used_mib": 23274,
        "t_s": 142.427
      },
      {
        "power_w": 299.7,
        "gpu_util_pct": 93,
        "memory_used_mib": 23274,
        "t_s": 143.483
      },
      {
        "power_w": 300.04,
        "gpu_util_pct": 94,
        "memory_used_mib": 23274,
        "t_s": 144.532
      },
      {
        "power_w": 237.66,
        "gpu_util_pct": 100,
        "memory_used_mib": 23274,
        "t_s": 145.582
      },
      {
        "power_w": 300.19,
        "gpu_util_pct": 100,
        "memory_used_mib": 23274,
        "t_s": 146.632
      },
      {
        "power_w": 299.75,
        "gpu_util_pct": 100,
        "memory_used_mib": 23274,
        "t_s": 147.684
      },
      {
        "power_w": 299.56,
        "gpu_util_pct": 100,
        "memory_used_mib": 23274,
        "t_s": 148.734
      },
      {
        "power_w": 299.33,
        "gpu_util_pct": 100,
        "memory_used_mib": 23274,
        "t_s": 149.797
      },
      {
        "power_w": 299.56,
        "gpu_util_pct": 100,
        "memory_used_mib": 23274,
        "t_s": 150.847
      },
      {
        "power_w": 300.17,
        "gpu_util_pct": 100,
        "memory_used_mib": 23274,
        "t_s": 151.899
      },
      {
        "power_w": 298.71,
        "gpu_util_pct": 93,
        "memory_used_mib": 23274,
        "t_s": 152.953
      },
      {
        "power_w": 300.04,
        "gpu_util_pct": 93,
        "memory_used_mib": 23274,
        "t_s": 154.012
      },
      {
        "power_w": 299.88,
        "gpu_util_pct": 92,
        "memory_used_mib": 23274,
        "t_s": 155.086
      },
      {
        "power_w": 243.12,
        "gpu_util_pct": 0,
        "memory_used_mib": 23274,
        "t_s": 156.29
      },
      {
        "power_w": 290.57,
        "gpu_util_pct": 99,
        "memory_used_mib": 23274,
        "t_s": 157.362
      },
      {
        "power_w": 298.89,
        "gpu_util_pct": 93,
        "memory_used_mib": 23274,
        "t_s": 158.419
      },
      {
        "power_w": 298.35,
        "gpu_util_pct": 94,
        "memory_used_mib": 23274,
        "t_s": 159.47
      },
      {
        "power_w": 299.54,
        "gpu_util_pct": 93,
        "memory_used_mib": 23274,
        "t_s": 160.531
      },
      {
        "power_w": 299.72,
        "gpu_util_pct": 93,
        "memory_used_mib": 23274,
        "t_s": 161.615
      },
      {
        "power_w": 227.65,
        "gpu_util_pct": 100,
        "memory_used_mib": 23274,
        "t_s": 162.671
      },
      {
        "power_w": 294.6,
        "gpu_util_pct": 88,
        "memory_used_mib": 23274,
        "t_s": 163.742
      },
      {
        "power_w": 299.4,
        "gpu_util_pct": 94,
        "memory_used_mib": 23274,
        "t_s": 164.795
      },
      {
        "power_w": 299.9,
        "gpu_util_pct": 94,
        "memory_used_mib": 23274,
        "t_s": 165.853
      },
      {
        "power_w": 300.1,
        "gpu_util_pct": 93,
        "memory_used_mib": 23274,
        "t_s": 166.922
      },
      {
        "power_w": 234.73,
        "gpu_util_pct": 100,
        "memory_used_mib": 23274,
        "t_s": 168.001
      },
      {
        "power_w": 297.25,
        "gpu_util_pct": 91,
        "memory_used_mib": 23274,
        "t_s": 169.052
      },
      {
        "power_w": 300.13,
        "gpu_util_pct": 94,
        "memory_used_mib": 23274,
        "t_s": 170.105
      },
      {
        "power_w": 299.92,
        "gpu_util_pct": 93,
        "memory_used_mib": 23274,
        "t_s": 171.156
      },
      {
        "power_w": 299.99,
        "gpu_util_pct": 93,
        "memory_used_mib": 23274,
        "t_s": 172.213
      },
      {
        "power_w": 234.11,
        "gpu_util_pct": 97,
        "memory_used_mib": 23274,
        "t_s": 173.267
      },
      {
        "power_w": 299.07,
        "gpu_util_pct": 100,
        "memory_used_mib": 23274,
        "t_s": 174.336
      },
      {
        "power_w": 300.28,
        "gpu_util_pct": 100,
        "memory_used_mib": 23274,
        "t_s": 175.385
      },
      {
        "power_w": 299.9,
        "gpu_util_pct": 100,
        "memory_used_mib": 23274,
        "t_s": 176.434
      },
      {
        "power_w": 298.9,
        "gpu_util_pct": 100,
        "memory_used_mib": 23274,
        "t_s": 177.491
      },
      {
        "power_w": 299.54,
        "gpu_util_pct": 100,
        "memory_used_mib": 23274,
        "t_s": 178.539
      },
      {
        "power_w": 299.58,
        "gpu_util_pct": 100,
        "memory_used_mib": 23274,
        "t_s": 179.591
      },
      {
        "power_w": 298.27,
        "gpu_util_pct": 100,
        "memory_used_mib": 23274,
        "t_s": 180.659
      },
      {
        "power_w": 296.34,
        "gpu_util_pct": 100,
        "memory_used_mib": 23274,
        "t_s": 181.71
      },
      {
        "power_w": 300.42,
        "gpu_util_pct": 94,
        "memory_used_mib": 23274,
        "t_s": 182.77
      },
      {
        "power_w": 300.42,
        "gpu_util_pct": 94,
        "memory_used_mib": 23274,
        "t_s": 183.839
      },
      {
        "power_w": 300.47,
        "gpu_util_pct": 94,
        "memory_used_mib": 23274,
        "t_s": 184.889
      },
      {
        "power_w": 246.62,
        "gpu_util_pct": 0,
        "memory_used_mib": 23274,
        "t_s": 185.939
      },
      {
        "power_w": 299.52,
        "gpu_util_pct": 94,
        "memory_used_mib": 23274,
        "t_s": 187.003
      },
      {
        "power_w": 300.0,
        "gpu_util_pct": 94,
        "memory_used_mib": 23274,
        "t_s": 188.064
      },
      {
        "power_w": 299.78,
        "gpu_util_pct": 94,
        "memory_used_mib": 23274,
        "t_s": 189.115
      },
      {
        "power_w": 300.01,
        "gpu_util_pct": 93,
        "memory_used_mib": 23274,
        "t_s": 190.179
      },
      {
        "power_w": 226.34,
        "gpu_util_pct": 89,
        "memory_used_mib": 23274,
        "t_s": 191.244
      },
      {
        "power_w": 300.25,
        "gpu_util_pct": 92,
        "memory_used_mib": 23274,
        "t_s": 192.307
      },
      {
        "power_w": 299.81,
        "gpu_util_pct": 93,
        "memory_used_mib": 23274,
        "t_s": 193.359
      },
      {
        "power_w": 300.38,
        "gpu_util_pct": 95,
        "memory_used_mib": 23274,
        "t_s": 194.418
      },
      {
        "power_w": 300.2,
        "gpu_util_pct": 95,
        "memory_used_mib": 23274,
        "t_s": 195.5
      },
      {
        "power_w": 225.48,
        "gpu_util_pct": 6,
        "memory_used_mib": 23274,
        "t_s": 196.569
      },
      {
        "power_w": 298.82,
        "gpu_util_pct": 100,
        "memory_used_mib": 23274,
        "t_s": 197.643
      },
      {
        "power_w": 300.38,
        "gpu_util_pct": 93,
        "memory_used_mib": 23274,
        "t_s": 198.715
      },
      {
        "power_w": 299.7,
        "gpu_util_pct": 94,
        "memory_used_mib": 23274,
        "t_s": 199.786
      },
      {
        "power_w": 300.2,
        "gpu_util_pct": 94,
        "memory_used_mib": 23274,
        "t_s": 200.838
      },
      {
        "power_w": 265.48,
        "gpu_util_pct": 21,
        "memory_used_mib": 23274,
        "t_s": 201.888
      },
      {
        "power_w": 293.48,
        "gpu_util_pct": 94,
        "memory_used_mib": 23274,
        "t_s": 202.938
      },
      {
        "power_w": 299.52,
        "gpu_util_pct": 94,
        "memory_used_mib": 23274,
        "t_s": 203.991
      },
      {
        "power_w": 299.97,
        "gpu_util_pct": 91,
        "memory_used_mib": 23274,
        "t_s": 205.043
      },
      {
        "power_w": 300.11,
        "gpu_util_pct": 93,
        "memory_used_mib": 23274,
        "t_s": 206.097
      },
      {
        "power_w": 189.4,
        "gpu_util_pct": 0,
        "memory_used_mib": 953,
        "t_s": 207.148
      },
      {
        "power_w": 146.2,
        "gpu_util_pct": 0,
        "memory_used_mib": 953,
        "t_s": 208.276
      },
      {
        "power_w": 124.42,
        "gpu_util_pct": 0,
        "memory_used_mib": 953,
        "t_s": 209.442
      },
      {
        "power_w": 124.09,
        "gpu_util_pct": 0,
        "memory_used_mib": 953,
        "t_s": 210.6
      },
      {
        "power_w": 123.74,
        "gpu_util_pct": 0,
        "memory_used_mib": 953,
        "t_s": 211.755
      }
    ]
  }
}
```

## Run Quality

- ok: True
- censored: False
- valid_turns: 38/38
- error_count: 0
- energy_j: 58236.406
- avg_power_w: 274.446
- max_power_w: 301.58
- mean_gpu_util_pct: 80.741
- max_memory_used_mib: 23274
- pin_decode_claim_scope: mixed_tool_and_non_tool
- pin_decode_tool_turns: 29
- pin_decode_non_tool_turns: 9
- pin_decode_tool_stop_conflict: True
- pin_decode_non_tool_ok: True
- tool_expected_valid: 19/29
- unexpected_tool_call_rate: 0.667

## Per-Turn Cache Trace
turn      pt  eff_in  prefix_len  fresh_pf     hr   pf_s  pf_tps   out/req  dec_tps  mode  accept  pflash%  tool  wall_s
------------------------------------------------------------------------------------------------------------------------
   1   31362    None        None     31362      ?  11.14    2816   256/256    110.8  None      AR        -     Y   13.52
   2     193    None        None       193      ?   0.17    1142   256/256    112.1  None      AR        -     Y    4.63
   3    3835    None        None      3835      ?   1.58    2425   256/256    108.4  None      AR        -     Y    4.01
   4    1455    None        None      1455      ?   0.68    2148   256/256    107.6  None      AR        -     Y    3.13
   5     500    None        None       500      ?   0.29    1742   256/256    107.7  None      AR        -     Y    2.74
   6     834    None        None       834      ?   0.45    1866   256/256    105.7  None      AR        -     Y    2.94
   7    4011    None        None      4011      ?   1.76    2284   256/256    103.4  None      AR        -     Y    4.31
   8     877    None        None       877      ?   0.50    1765   256/256    102.3  None      AR        -     Y    3.08
   9     433    None        None       433      ?   0.27    1631   256/256    102.4  None      AR        -     Y    2.85
  10    1003    None        None      1003      ?   0.49    2047   256/256    100.9  None      AR        -     Y    3.11
  11     393    None        None       393      ?   0.26    1488   256/256    101.9  None      AR        -     Y    4.96
  12    1066    None        None      1066      ?   0.57    1875   256/256    100.4  None      AR        -     Y     3.2
  13    1449    None        None      1449      ?   0.73    1992   256/256     99.4  None      AR        -     N    3.39
  14    1175    None        None      1175      ?   0.60    1972   256/256    100.1  None      AR        -     N    3.24
  15    5310    None        None      5310      ?   2.37    2240   256/256     96.2  None      AR        -     N    5.13
  16    1639    None        None      1639      ?   0.83    1981   256/256     95.3  None      AR        -     Y    3.61
  17     911    None        None       911      ?   0.52    1766   256/256     95.2  None      AR        -     N     3.3
  18    3512    None        None      3512      ?   1.67    2099   256/256     92.5  None      AR        -     Y    4.54
  19    3376    None        None      3376      ?   1.60    2113   256/256     90.8  None      AR        -     N    6.62
  20    1518    None        None      1518      ?   0.76    1996   256/256     90.1  None      AR        -     N    3.71
  21    3123    None        None      3123      ?   1.55    2018   256/256     88.9  None      AR        -     N    4.53
  22    1497    None        None      1497      ?   0.78    1926   256/256     87.7  None      AR        -     N     3.8
  23    2266    None        None      2266      ?   1.19    1903   256/256     86.8  None      AR        -     N    4.25
  24    3244    None        None      3244      ?   1.66    1953   256/256     85.8  None      AR        -     N    4.77
  25    4587    None        None      4587      ?   2.34    1963   256/256     83.6  None      AR        -     N    5.51
  26    5790    None        None      5790      ?   2.98    1944   256/256     81.4  None      AR        -     N    8.35
  27    1644    None        None      1644      ?   0.96    1708   256/256     80.1  None      AR        -     Y    4.29
  28    4581    None        None      4581      ?   2.44    1877   256/256     78.7  None      AR        -     N    5.82
  29    3067    None        None      3067      ?   1.72    1784   256/256     77.4  None      AR        -     N    5.16
  30   12796    None        None     12796      ?   7.28    1759   256/256     70.9  None      AR        -     N   13.04
  31    3592    None        None      3592      ?   2.15    1672   256/256     73.6  None      AR        -     N     5.8
  32    1818    None        None      1818      ?   1.14    1600   256/256     72.2  None      AR        -     Y    4.83
  33    1621    None        None      1621      ?   1.11    1464   256/256     69.6  None      AR        -     Y    4.94
  34   14353    None        None     14353      ?   8.85    1622   256/256     65.2  None      AR        -     N   12.93
  35     731    None        None       731      ?   0.58    1264   256/256     65.5  None      AR        -     Y    6.79
  36    1569    None        None      1569      ?   1.09    1445   256/256     65.7  None      AR        -     Y    5.14
  37    1754    None        None      1754      ?   1.19    1478   256/256     64.9  None      AR        -     N    5.29
  38     393    None        None       393      ?   0.37    1062   256/256     64.4  None      AR        -     N    4.51

## Arm Aggregate
```
{
  "ok": true,
  "censored": false,
  "expected_turns": 38,
  "valid_turns": 38,
  "total_attempts": 38,
  "error_count": 0,
  "total_wall_s": 195.8,
  "total_prefill_s": 66.556,
  "total_decode_s": 112.469,
  "sum_prompt_tokens": 133278,
  "sum_effective_in_tokens": null,
  "sum_fresh_prefill_tokens": 133278,
  "sum_requested_out_tokens": 9728,
  "sum_out_tokens": 9728,
  "out_token_mismatch_count": 0,
  "out_tokens_match_requested": true,
  "pin_decode_ok": true,
  "pin_decode_turns": 38,
  "pin_decode_tool_turns": 29,
  "pin_decode_non_tool_turns": 9,
  "pin_decode_non_tool_mismatch_count": 0,
  "pin_decode_non_tool_ok": true,
  "pin_decode_tool_stop_conflict": true,
  "pin_decode_claim_scope": "mixed_tool_and_non_tool",
  "mean_cache_hit_ratio": null,
  "mean_prefill_tps": 1837.613,
  "mean_decode_tps": 89.095,
  "weighted_prompt_prefill_tps": 2002.494,
  "weighted_effective_prefill_tps": null,
  "weighted_fresh_prefill_tps": 2002.494,
  "weighted_decode_tps": 86.495,
  "spec_engagement_rate": 0.0,
  "mean_accept_rate": null,
  "mean_disk_hit_rate": null,
  "sum_tool_expected_turns": 29,
  "sum_tool_expected_valid_turns": 19,
  "tool_call_valid_rate": 0.655,
  "unexpected_tool_call_rate": 0.667,
  "charbench_valid_rate": null,
  "energy_j": 58236.406,
  "avg_power_w": 274.446,
  "max_power_w": 301.58,
  "mean_gpu_util_pct": 80.741,
  "max_memory_used_mib": 23274
}
```
