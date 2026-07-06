# ABC Cache Harness — AR_LLAMA_27B_SLOTCACHE
Generated: 2026-07-06 10:54 UTC
Mode: FULL (all turns, N=1)

## Provenance
```
{
  "binary": "/home/peppi/llama.cpp/build-cuda/bin/llama-server",
  "binary_sha256": "feedd55326b13fd4156dd0c7d7086fb94201cceeda5ef3eabc43fb26e2adc06b",
  "git_branch": "wip/deathmatch-current-main-snapshot-20260706",
  "git_commit": "8b946106e0e43826ff8594bb673cf06729f63a84",
  "arm": "AR_LLAMA_27B_SLOTCACHE",
  "arm_description": "llama.cpp 27B Q4_K_M + prompt/cache-reuse/checkpoints + slot save/restore; pure AR; q4_0 KV",
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
    "/tmp/llama27b_slot_cache"
  ],
  "arm_env": {},
  "server_type": "llama_cpp",
  "model_target": "/home/peppi/models/qwen3.6-27b-q4km/Qwen3.6-27B-Q4_K_M.gguf",
  "model_draft_decode": "/home/peppi/models/qwen3.6-27b-dflash/dflash-draft-3.6-bf16-reconverted.gguf",
  "model_draft_prefill": "/home/peppi/models/Qwen3-0.6B-BF16.gguf",
  "chat_template": "/home/peppi/models/qwen3-coder-chat-template.jinja",
  "max_ctx": 131072,
  "cache_type_k": "q4_0",
  "cache_type_v": "q4_0",
  "temperature": 0.7,
  "seed_requested": 42,
  "seed_sent": true,
  "n_repeats": 1,
  "smoke": false,
  "restart_per_turn": false,
  "quality_probe": false,
  "pin_decode_length": false,
  "trace_max_tokens_unique": [],
  "pin_decode_tokens_process": null,
  "pin_decode_mechanism": null,
  "power_sampling_enabled": true,
  "power_sample_interval_s": 1.0,
  "power_gpu_index": 0,
  "port": 19099,
  "trace": "/tmp/deep_tool_structured_38_cap2048.jsonl",
  "n_turns_in_trace": 38,
  "timestamp_utc": "2026-07-06T10:28:32.001528+00:00",
  "slot_save_dir": "/tmp/llama27b_slot_cache",
  "slot_save_file": "ar27b_slot.bin",
  "power": {
    "gpu_index": 0,
    "sample_interval_s": 1.0,
    "duration_s": 1470.164,
    "sample_count": 1394,
    "valid_sample_count": 1394,
    "energy_j": 436187.654,
    "avg_power_w": 296.693,
    "max_power_w": 302.56,
    "mean_gpu_util_pct": 95.57,
    "max_memory_used_mib": 20037,
    "first_error": null,
    "samples": [
      {
        "power_w": 113.71,
        "gpu_util_pct": 0,
        "memory_used_mib": 595,
        "t_s": 0.0
      },
      {
        "power_w": 112.54,
        "gpu_util_pct": 0,
        "memory_used_mib": 851,
        "t_s": 1.115
      },
      {
        "power_w": 111.93,
        "gpu_util_pct": 0,
        "memory_used_mib": 851,
        "t_s": 2.17
      },
      {
        "power_w": 111.88,
        "gpu_util_pct": 0,
        "memory_used_mib": 851,
        "t_s": 3.229
      },
      {
        "power_w": 114.65,
        "gpu_util_pct": 98,
        "memory_used_mib": 13331,
        "t_s": 4.283
      },
      {
        "power_w": 151.13,
        "gpu_util_pct": 47,
        "memory_used_mib": 16197,
        "t_s": 5.349
      },
      {
        "power_w": 164.08,
        "gpu_util_pct": 33,
        "memory_used_mib": 19871,
        "t_s": 6.415
      },
      {
        "power_w": 142.1,
        "gpu_util_pct": 0,
        "memory_used_mib": 19871,
        "t_s": 7.465
      },
      {
        "power_w": 196.5,
        "gpu_util_pct": 100,
        "memory_used_mib": 19889,
        "t_s": 8.514
      },
      {
        "power_w": 300.03,
        "gpu_util_pct": 100,
        "memory_used_mib": 19889,
        "t_s": 9.565
      },
      {
        "power_w": 300.46,
        "gpu_util_pct": 100,
        "memory_used_mib": 19889,
        "t_s": 10.616
      },
      {
        "power_w": 299.28,
        "gpu_util_pct": 100,
        "memory_used_mib": 19889,
        "t_s": 11.666
      },
      {
        "power_w": 300.29,
        "gpu_util_pct": 100,
        "memory_used_mib": 19889,
        "t_s": 12.723
      },
      {
        "power_w": 300.63,
        "gpu_util_pct": 100,
        "memory_used_mib": 19889,
        "t_s": 13.773
      },
      {
        "power_w": 299.66,
        "gpu_util_pct": 100,
        "memory_used_mib": 19889,
        "t_s": 14.825
      },
      {
        "power_w": 299.93,
        "gpu_util_pct": 100,
        "memory_used_mib": 19889,
        "t_s": 15.878
      },
      {
        "power_w": 299.62,
        "gpu_util_pct": 100,
        "memory_used_mib": 19889,
        "t_s": 16.929
      },
      {
        "power_w": 299.88,
        "gpu_util_pct": 100,
        "memory_used_mib": 19889,
        "t_s": 18.153
      },
      {
        "power_w": 299.63,
        "gpu_util_pct": 100,
        "memory_used_mib": 19889,
        "t_s": 19.207
      },
      {
        "power_w": 300.02,
        "gpu_util_pct": 100,
        "memory_used_mib": 19889,
        "t_s": 20.256
      },
      {
        "power_w": 299.69,
        "gpu_util_pct": 100,
        "memory_used_mib": 19889,
        "t_s": 21.31
      },
      {
        "power_w": 299.88,
        "gpu_util_pct": 100,
        "memory_used_mib": 19889,
        "t_s": 22.362
      },
      {
        "power_w": 299.81,
        "gpu_util_pct": 100,
        "memory_used_mib": 19889,
        "t_s": 23.414
      },
      {
        "power_w": 299.57,
        "gpu_util_pct": 100,
        "memory_used_mib": 19889,
        "t_s": 24.469
      },
      {
        "power_w": 300.38,
        "gpu_util_pct": 99,
        "memory_used_mib": 19889,
        "t_s": 25.522
      },
      {
        "power_w": 300.32,
        "gpu_util_pct": 100,
        "memory_used_mib": 19889,
        "t_s": 26.572
      },
      {
        "power_w": 299.66,
        "gpu_util_pct": 100,
        "memory_used_mib": 19889,
        "t_s": 27.625
      },
      {
        "power_w": 299.79,
        "gpu_util_pct": 100,
        "memory_used_mib": 19889,
        "t_s": 28.678
      },
      {
        "power_w": 300.29,
        "gpu_util_pct": 100,
        "memory_used_mib": 19889,
        "t_s": 29.727
      },
      {
        "power_w": 299.88,
        "gpu_util_pct": 100,
        "memory_used_mib": 19889,
        "t_s": 30.776
      },
      {
        "power_w": 299.62,
        "gpu_util_pct": 100,
        "memory_used_mib": 19889,
        "t_s": 31.838
      },
      {
        "power_w": 300.02,
        "gpu_util_pct": 100,
        "memory_used_mib": 19889,
        "t_s": 32.911
      },
      {
        "power_w": 297.08,
        "gpu_util_pct": 93,
        "memory_used_mib": 19891,
        "t_s": 33.963
      },
      {
        "power_w": 301.88,
        "gpu_util_pct": 95,
        "memory_used_mib": 19893,
        "t_s": 35.013
      },
      {
        "power_w": 300.92,
        "gpu_util_pct": 97,
        "memory_used_mib": 19893,
        "t_s": 36.064
      },
      {
        "power_w": 240.42,
        "gpu_util_pct": 93,
        "memory_used_mib": 19893,
        "t_s": 37.116
      },
      {
        "power_w": 299.45,
        "gpu_util_pct": 100,
        "memory_used_mib": 19893,
        "t_s": 38.171
      },
      {
        "power_w": 299.64,
        "gpu_util_pct": 100,
        "memory_used_mib": 19893,
        "t_s": 39.222
      },
      {
        "power_w": 299.53,
        "gpu_util_pct": 100,
        "memory_used_mib": 19893,
        "t_s": 40.272
      },
      {
        "power_w": 299.82,
        "gpu_util_pct": 94,
        "memory_used_mib": 19895,
        "t_s": 41.321
      },
      {
        "power_w": 299.98,
        "gpu_util_pct": 97,
        "memory_used_mib": 19895,
        "t_s": 42.372
      },
      {
        "power_w": 299.68,
        "gpu_util_pct": 96,
        "memory_used_mib": 19895,
        "t_s": 43.419
      },
      {
        "power_w": 300.01,
        "gpu_util_pct": 96,
        "memory_used_mib": 19895,
        "t_s": 44.466
      },
      {
        "power_w": 300.3,
        "gpu_util_pct": 98,
        "memory_used_mib": 19895,
        "t_s": 45.52
      },
      {
        "power_w": 300.02,
        "gpu_util_pct": 97,
        "memory_used_mib": 19895,
        "t_s": 46.573
      },
      {
        "power_w": 300.35,
        "gpu_util_pct": 97,
        "memory_used_mib": 19895,
        "t_s": 47.625
      },
      {
        "power_w": 300.46,
        "gpu_util_pct": 98,
        "memory_used_mib": 19895,
        "t_s": 48.674
      },
      {
        "power_w": 300.49,
        "gpu_util_pct": 97,
        "memory_used_mib": 19895,
        "t_s": 49.725
      },
      {
        "power_w": 300.39,
        "gpu_util_pct": 95,
        "memory_used_mib": 19895,
        "t_s": 50.784
      },
      {
        "power_w": 300.17,
        "gpu_util_pct": 97,
        "memory_used_mib": 19895,
        "t_s": 51.836
      },
      {
        "power_w": 300.51,
        "gpu_util_pct": 97,
        "memory_used_mib": 19895,
        "t_s": 52.888
      },
      {
        "power_w": 300.48,
        "gpu_util_pct": 97,
        "memory_used_mib": 19895,
        "t_s": 53.937
      },
      {
        "power_w": 300.12,
        "gpu_util_pct": 97,
        "memory_used_mib": 19895,
        "t_s": 54.988
      },
      {
        "power_w": 300.42,
        "gpu_util_pct": 96,
        "memory_used_mib": 19895,
        "t_s": 56.036
      },
      {
        "power_w": 299.98,
        "gpu_util_pct": 97,
        "memory_used_mib": 19895,
        "t_s": 57.085
      },
      {
        "power_w": 300.59,
        "gpu_util_pct": 97,
        "memory_used_mib": 19895,
        "t_s": 58.141
      },
      {
        "power_w": 299.92,
        "gpu_util_pct": 97,
        "memory_used_mib": 19895,
        "t_s": 59.191
      },
      {
        "power_w": 243.96,
        "gpu_util_pct": 100,
        "memory_used_mib": 19895,
        "t_s": 60.241
      },
      {
        "power_w": 299.84,
        "gpu_util_pct": 100,
        "memory_used_mib": 19895,
        "t_s": 61.294
      },
      {
        "power_w": 295.0,
        "gpu_util_pct": 96,
        "memory_used_mib": 19895,
        "t_s": 62.34
      },
      {
        "power_w": 301.76,
        "gpu_util_pct": 96,
        "memory_used_mib": 19895,
        "t_s": 63.411
      },
      {
        "power_w": 300.7,
        "gpu_util_pct": 97,
        "memory_used_mib": 19895,
        "t_s": 64.463
      },
      {
        "power_w": 300.82,
        "gpu_util_pct": 97,
        "memory_used_mib": 19895,
        "t_s": 65.526
      },
      {
        "power_w": 300.01,
        "gpu_util_pct": 97,
        "memory_used_mib": 19895,
        "t_s": 66.578
      },
      {
        "power_w": 300.44,
        "gpu_util_pct": 96,
        "memory_used_mib": 19895,
        "t_s": 67.631
      },
      {
        "power_w": 300.02,
        "gpu_util_pct": 96,
        "memory_used_mib": 19895,
        "t_s": 68.686
      },
      {
        "power_w": 300.27,
        "gpu_util_pct": 97,
        "memory_used_mib": 19895,
        "t_s": 69.739
      },
      {
        "power_w": 300.08,
        "gpu_util_pct": 97,
        "memory_used_mib": 19895,
        "t_s": 70.801
      },
      {
        "power_w": 300.66,
        "gpu_util_pct": 94,
        "memory_used_mib": 19895,
        "t_s": 71.867
      },
      {
        "power_w": 300.05,
        "gpu_util_pct": 96,
        "memory_used_mib": 19895,
        "t_s": 72.918
      },
      {
        "power_w": 300.33,
        "gpu_util_pct": 96,
        "memory_used_mib": 19895,
        "t_s": 73.968
      },
      {
        "power_w": 300.41,
        "gpu_util_pct": 96,
        "memory_used_mib": 19895,
        "t_s": 75.022
      },
      {
        "power_w": 254.74,
        "gpu_util_pct": 0,
        "memory_used_mib": 19895,
        "t_s": 76.074
      },
      {
        "power_w": 286.8,
        "gpu_util_pct": 100,
        "memory_used_mib": 19895,
        "t_s": 77.119
      },
      {
        "power_w": 299.95,
        "gpu_util_pct": 100,
        "memory_used_mib": 19895,
        "t_s": 78.169
      },
      {
        "power_w": 299.82,
        "gpu_util_pct": 100,
        "memory_used_mib": 19895,
        "t_s": 79.222
      },
      {
        "power_w": 299.57,
        "gpu_util_pct": 100,
        "memory_used_mib": 19895,
        "t_s": 80.268
      },
      {
        "power_w": 299.01,
        "gpu_util_pct": 100,
        "memory_used_mib": 19895,
        "t_s": 81.333
      },
      {
        "power_w": 299.5,
        "gpu_util_pct": 100,
        "memory_used_mib": 19895,
        "t_s": 82.382
      },
      {
        "power_w": 298.92,
        "gpu_util_pct": 100,
        "memory_used_mib": 19895,
        "t_s": 83.432
      },
      {
        "power_w": 299.3,
        "gpu_util_pct": 99,
        "memory_used_mib": 19895,
        "t_s": 84.481
      },
      {
        "power_w": 300.1,
        "gpu_util_pct": 100,
        "memory_used_mib": 19895,
        "t_s": 85.531
      },
      {
        "power_w": 299.75,
        "gpu_util_pct": 94,
        "memory_used_mib": 19895,
        "t_s": 86.582
      },
      {
        "power_w": 299.63,
        "gpu_util_pct": 97,
        "memory_used_mib": 19895,
        "t_s": 87.639
      },
      {
        "power_w": 299.99,
        "gpu_util_pct": 97,
        "memory_used_mib": 19895,
        "t_s": 88.691
      },
      {
        "power_w": 300.43,
        "gpu_util_pct": 97,
        "memory_used_mib": 19895,
        "t_s": 89.741
      },
      {
        "power_w": 300.3,
        "gpu_util_pct": 97,
        "memory_used_mib": 19895,
        "t_s": 90.789
      },
      {
        "power_w": 300.32,
        "gpu_util_pct": 97,
        "memory_used_mib": 19895,
        "t_s": 91.832
      },
      {
        "power_w": 300.58,
        "gpu_util_pct": 97,
        "memory_used_mib": 19895,
        "t_s": 92.883
      },
      {
        "power_w": 300.24,
        "gpu_util_pct": 97,
        "memory_used_mib": 19895,
        "t_s": 93.936
      },
      {
        "power_w": 300.36,
        "gpu_util_pct": 97,
        "memory_used_mib": 19895,
        "t_s": 94.984
      },
      {
        "power_w": 300.34,
        "gpu_util_pct": 97,
        "memory_used_mib": 19895,
        "t_s": 96.034
      },
      {
        "power_w": 300.12,
        "gpu_util_pct": 97,
        "memory_used_mib": 19895,
        "t_s": 97.086
      },
      {
        "power_w": 300.05,
        "gpu_util_pct": 96,
        "memory_used_mib": 19895,
        "t_s": 98.143
      },
      {
        "power_w": 300.38,
        "gpu_util_pct": 97,
        "memory_used_mib": 19895,
        "t_s": 99.195
      },
      {
        "power_w": 300.37,
        "gpu_util_pct": 96,
        "memory_used_mib": 19895,
        "t_s": 100.241
      },
      {
        "power_w": 300.36,
        "gpu_util_pct": 97,
        "memory_used_mib": 19895,
        "t_s": 101.291
      },
      {
        "power_w": 300.41,
        "gpu_util_pct": 97,
        "memory_used_mib": 19895,
        "t_s": 102.343
      },
      {
        "power_w": 299.88,
        "gpu_util_pct": 96,
        "memory_used_mib": 19895,
        "t_s": 103.392
      },
      {
        "power_w": 299.84,
        "gpu_util_pct": 96,
        "memory_used_mib": 19895,
        "t_s": 104.443
      },
      {
        "power_w": 300.43,
        "gpu_util_pct": 96,
        "memory_used_mib": 19895,
        "t_s": 105.494
      },
      {
        "power_w": 300.19,
        "gpu_util_pct": 97,
        "memory_used_mib": 19895,
        "t_s": 106.546
      },
      {
        "power_w": 300.46,
        "gpu_util_pct": 97,
        "memory_used_mib": 19895,
        "t_s": 107.6
      },
      {
        "power_w": 300.39,
        "gpu_util_pct": 97,
        "memory_used_mib": 19895,
        "t_s": 108.65
      },
      {
        "power_w": 300.27,
        "gpu_util_pct": 97,
        "memory_used_mib": 19895,
        "t_s": 109.705
      },
      {
        "power_w": 242.42,
        "gpu_util_pct": 100,
        "memory_used_mib": 19895,
        "t_s": 110.759
      },
      {
        "power_w": 299.47,
        "gpu_util_pct": 100,
        "memory_used_mib": 19895,
        "t_s": 111.81
      },
      {
        "power_w": 299.07,
        "gpu_util_pct": 98,
        "memory_used_mib": 19895,
        "t_s": 112.859
      },
      {
        "power_w": 300.11,
        "gpu_util_pct": 97,
        "memory_used_mib": 19895,
        "t_s": 113.917
      },
      {
        "power_w": 299.93,
        "gpu_util_pct": 97,
        "memory_used_mib": 19895,
        "t_s": 114.967
      },
      {
        "power_w": 300.23,
        "gpu_util_pct": 97,
        "memory_used_mib": 19895,
        "t_s": 116.017
      },
      {
        "power_w": 243.22,
        "gpu_util_pct": 100,
        "memory_used_mib": 19895,
        "t_s": 117.068
      },
      {
        "power_w": 297.44,
        "gpu_util_pct": 100,
        "memory_used_mib": 19895,
        "t_s": 118.129
      },
      {
        "power_w": 297.15,
        "gpu_util_pct": 100,
        "memory_used_mib": 19895,
        "t_s": 119.196
      },
      {
        "power_w": 300.53,
        "gpu_util_pct": 97,
        "memory_used_mib": 19895,
        "t_s": 120.252
      },
      {
        "power_w": 300.87,
        "gpu_util_pct": 96,
        "memory_used_mib": 19895,
        "t_s": 121.302
      },
      {
        "power_w": 300.21,
        "gpu_util_pct": 97,
        "memory_used_mib": 19895,
        "t_s": 122.354
      },
      {
        "power_w": 300.48,
        "gpu_util_pct": 98,
        "memory_used_mib": 19895,
        "t_s": 123.404
      },
      {
        "power_w": 300.18,
        "gpu_util_pct": 96,
        "memory_used_mib": 19895,
        "t_s": 124.485
      },
      {
        "power_w": 299.87,
        "gpu_util_pct": 97,
        "memory_used_mib": 19895,
        "t_s": 125.551
      },
      {
        "power_w": 300.42,
        "gpu_util_pct": 96,
        "memory_used_mib": 19895,
        "t_s": 126.603
      },
      {
        "power_w": 300.2,
        "gpu_util_pct": 97,
        "memory_used_mib": 19895,
        "t_s": 127.653
      },
      {
        "power_w": 300.78,
        "gpu_util_pct": 96,
        "memory_used_mib": 19895,
        "t_s": 128.705
      },
      {
        "power_w": 300.25,
        "gpu_util_pct": 97,
        "memory_used_mib": 19895,
        "t_s": 129.755
      },
      {
        "power_w": 300.38,
        "gpu_util_pct": 97,
        "memory_used_mib": 19895,
        "t_s": 130.805
      },
      {
        "power_w": 300.23,
        "gpu_util_pct": 97,
        "memory_used_mib": 19895,
        "t_s": 131.86
      },
      {
        "power_w": 300.8,
        "gpu_util_pct": 97,
        "memory_used_mib": 19895,
        "t_s": 132.916
      },
      {
        "power_w": 240.1,
        "gpu_util_pct": 100,
        "memory_used_mib": 19895,
        "t_s": 133.965
      },
      {
        "power_w": 298.55,
        "gpu_util_pct": 100,
        "memory_used_mib": 19895,
        "t_s": 135.008
      },
      {
        "power_w": 295.88,
        "gpu_util_pct": 100,
        "memory_used_mib": 19895,
        "t_s": 136.054
      },
      {
        "power_w": 298.77,
        "gpu_util_pct": 100,
        "memory_used_mib": 19895,
        "t_s": 137.095
      },
      {
        "power_w": 297.3,
        "gpu_util_pct": 100,
        "memory_used_mib": 19895,
        "t_s": 138.161
      },
      {
        "power_w": 298.19,
        "gpu_util_pct": 100,
        "memory_used_mib": 19895,
        "t_s": 139.223
      },
      {
        "power_w": 293.46,
        "gpu_util_pct": 100,
        "memory_used_mib": 19895,
        "t_s": 140.273
      },
      {
        "power_w": 294.99,
        "gpu_util_pct": 100,
        "memory_used_mib": 19895,
        "t_s": 141.322
      },
      {
        "power_w": 298.09,
        "gpu_util_pct": 100,
        "memory_used_mib": 19895,
        "t_s": 142.381
      },
      {
        "power_w": 298.64,
        "gpu_util_pct": 100,
        "memory_used_mib": 19895,
        "t_s": 143.431
      },
      {
        "power_w": 299.9,
        "gpu_util_pct": 97,
        "memory_used_mib": 19895,
        "t_s": 144.492
      },
      {
        "power_w": 299.96,
        "gpu_util_pct": 97,
        "memory_used_mib": 19895,
        "t_s": 145.545
      },
      {
        "power_w": 300.8,
        "gpu_util_pct": 97,
        "memory_used_mib": 19895,
        "t_s": 146.593
      },
      {
        "power_w": 300.18,
        "gpu_util_pct": 97,
        "memory_used_mib": 19895,
        "t_s": 147.64
      },
      {
        "power_w": 299.69,
        "gpu_util_pct": 96,
        "memory_used_mib": 19895,
        "t_s": 148.685
      },
      {
        "power_w": 300.28,
        "gpu_util_pct": 97,
        "memory_used_mib": 19895,
        "t_s": 149.74
      },
      {
        "power_w": 300.63,
        "gpu_util_pct": 97,
        "memory_used_mib": 19895,
        "t_s": 150.792
      },
      {
        "power_w": 300.28,
        "gpu_util_pct": 98,
        "memory_used_mib": 19895,
        "t_s": 151.839
      },
      {
        "power_w": 300.42,
        "gpu_util_pct": 98,
        "memory_used_mib": 19895,
        "t_s": 152.887
      },
      {
        "power_w": 300.26,
        "gpu_util_pct": 98,
        "memory_used_mib": 19895,
        "t_s": 153.935
      },
      {
        "power_w": 300.45,
        "gpu_util_pct": 98,
        "memory_used_mib": 19895,
        "t_s": 154.982
      },
      {
        "power_w": 300.31,
        "gpu_util_pct": 97,
        "memory_used_mib": 19895,
        "t_s": 156.024
      },
      {
        "power_w": 299.9,
        "gpu_util_pct": 98,
        "memory_used_mib": 19895,
        "t_s": 157.071
      },
      {
        "power_w": 300.2,
        "gpu_util_pct": 97,
        "memory_used_mib": 19895,
        "t_s": 158.12
      },
      {
        "power_w": 300.32,
        "gpu_util_pct": 97,
        "memory_used_mib": 19895,
        "t_s": 159.167
      },
      {
        "power_w": 300.54,
        "gpu_util_pct": 97,
        "memory_used_mib": 19895,
        "t_s": 160.213
      },
      {
        "power_w": 300.34,
        "gpu_util_pct": 97,
        "memory_used_mib": 19895,
        "t_s": 161.258
      },
      {
        "power_w": 300.32,
        "gpu_util_pct": 97,
        "memory_used_mib": 19895,
        "t_s": 162.313
      },
      {
        "power_w": 300.39,
        "gpu_util_pct": 97,
        "memory_used_mib": 19895,
        "t_s": 163.362
      },
      {
        "power_w": 300.18,
        "gpu_util_pct": 98,
        "memory_used_mib": 19895,
        "t_s": 164.409
      },
      {
        "power_w": 300.33,
        "gpu_util_pct": 98,
        "memory_used_mib": 19895,
        "t_s": 165.457
      },
      {
        "power_w": 300.12,
        "gpu_util_pct": 95,
        "memory_used_mib": 19895,
        "t_s": 166.504
      },
      {
        "power_w": 300.34,
        "gpu_util_pct": 97,
        "memory_used_mib": 19895,
        "t_s": 167.551
      },
      {
        "power_w": 300.08,
        "gpu_util_pct": 97,
        "memory_used_mib": 19895,
        "t_s": 168.598
      },
      {
        "power_w": 300.21,
        "gpu_util_pct": 97,
        "memory_used_mib": 19895,
        "t_s": 169.646
      },
      {
        "power_w": 300.38,
        "gpu_util_pct": 98,
        "memory_used_mib": 19895,
        "t_s": 170.694
      },
      {
        "power_w": 300.39,
        "gpu_util_pct": 97,
        "memory_used_mib": 19895,
        "t_s": 171.743
      },
      {
        "power_w": 300.52,
        "gpu_util_pct": 97,
        "memory_used_mib": 19895,
        "t_s": 172.787
      },
      {
        "power_w": 299.96,
        "gpu_util_pct": 96,
        "memory_used_mib": 19895,
        "t_s": 173.837
      },
      {
        "power_w": 300.12,
        "gpu_util_pct": 97,
        "memory_used_mib": 19895,
        "t_s": 174.902
      },
      {
        "power_w": 300.28,
        "gpu_util_pct": 97,
        "memory_used_mib": 19895,
        "t_s": 175.956
      },
      {
        "power_w": 299.92,
        "gpu_util_pct": 98,
        "memory_used_mib": 19895,
        "t_s": 177.009
      },
      {
        "power_w": 300.42,
        "gpu_util_pct": 97,
        "memory_used_mib": 19895,
        "t_s": 178.06
      },
      {
        "power_w": 241.08,
        "gpu_util_pct": 52,
        "memory_used_mib": 19895,
        "t_s": 179.115
      },
      {
        "power_w": 299.05,
        "gpu_util_pct": 100,
        "memory_used_mib": 19895,
        "t_s": 180.166
      },
      {
        "power_w": 299.05,
        "gpu_util_pct": 100,
        "memory_used_mib": 19895,
        "t_s": 181.224
      },
      {
        "power_w": 297.45,
        "gpu_util_pct": 100,
        "memory_used_mib": 19895,
        "t_s": 182.274
      },
      {
        "power_w": 298.62,
        "gpu_util_pct": 100,
        "memory_used_mib": 19895,
        "t_s": 183.326
      },
      {
        "power_w": 295.68,
        "gpu_util_pct": 91,
        "memory_used_mib": 19895,
        "t_s": 184.373
      },
      {
        "power_w": 302.56,
        "gpu_util_pct": 98,
        "memory_used_mib": 19895,
        "t_s": 185.432
      },
      {
        "power_w": 300.08,
        "gpu_util_pct": 96,
        "memory_used_mib": 19895,
        "t_s": 186.48
      },
      {
        "power_w": 299.61,
        "gpu_util_pct": 97,
        "memory_used_mib": 19895,
        "t_s": 187.533
      },
      {
        "power_w": 300.43,
        "gpu_util_pct": 97,
        "memory_used_mib": 19895,
        "t_s": 188.586
      },
      {
        "power_w": 300.61,
        "gpu_util_pct": 97,
        "memory_used_mib": 19895,
        "t_s": 189.629
      },
      {
        "power_w": 300.11,
        "gpu_util_pct": 95,
        "memory_used_mib": 19895,
        "t_s": 190.679
      },
      {
        "power_w": 300.7,
        "gpu_util_pct": 97,
        "memory_used_mib": 19895,
        "t_s": 191.732
      },
      {
        "power_w": 300.45,
        "gpu_util_pct": 96,
        "memory_used_mib": 19895,
        "t_s": 192.78
      },
      {
        "power_w": 300.37,
        "gpu_util_pct": 97,
        "memory_used_mib": 19895,
        "t_s": 193.841
      },
      {
        "power_w": 300.26,
        "gpu_util_pct": 98,
        "memory_used_mib": 19895,
        "t_s": 194.896
      },
      {
        "power_w": 300.25,
        "gpu_util_pct": 97,
        "memory_used_mib": 19895,
        "t_s": 195.948
      },
      {
        "power_w": 300.46,
        "gpu_util_pct": 96,
        "memory_used_mib": 19895,
        "t_s": 197.16
      },
      {
        "power_w": 300.55,
        "gpu_util_pct": 98,
        "memory_used_mib": 19895,
        "t_s": 198.213
      },
      {
        "power_w": 300.21,
        "gpu_util_pct": 97,
        "memory_used_mib": 19895,
        "t_s": 199.309
      },
      {
        "power_w": 300.15,
        "gpu_util_pct": 98,
        "memory_used_mib": 19895,
        "t_s": 200.366
      },
      {
        "power_w": 299.93,
        "gpu_util_pct": 98,
        "memory_used_mib": 19895,
        "t_s": 201.416
      },
      {
        "power_w": 300.29,
        "gpu_util_pct": 97,
        "memory_used_mib": 19895,
        "t_s": 202.47
      },
      {
        "power_w": 300.37,
        "gpu_util_pct": 97,
        "memory_used_mib": 19895,
        "t_s": 203.526
      },
      {
        "power_w": 300.45,
        "gpu_util_pct": 98,
        "memory_used_mib": 19895,
        "t_s": 204.581
      },
      {
        "power_w": 300.62,
        "gpu_util_pct": 97,
        "memory_used_mib": 19895,
        "t_s": 205.624
      },
      {
        "power_w": 299.96,
        "gpu_util_pct": 97,
        "memory_used_mib": 19895,
        "t_s": 206.677
      },
      {
        "power_w": 300.29,
        "gpu_util_pct": 97,
        "memory_used_mib": 19895,
        "t_s": 207.727
      },
      {
        "power_w": 300.08,
        "gpu_util_pct": 94,
        "memory_used_mib": 19895,
        "t_s": 208.777
      },
      {
        "power_w": 300.32,
        "gpu_util_pct": 97,
        "memory_used_mib": 19895,
        "t_s": 209.826
      },
      {
        "power_w": 300.45,
        "gpu_util_pct": 97,
        "memory_used_mib": 19895,
        "t_s": 210.881
      },
      {
        "power_w": 300.1,
        "gpu_util_pct": 97,
        "memory_used_mib": 19895,
        "t_s": 211.923
      },
      {
        "power_w": 300.11,
        "gpu_util_pct": 97,
        "memory_used_mib": 19895,
        "t_s": 212.998
      },
      {
        "power_w": 300.29,
        "gpu_util_pct": 98,
        "memory_used_mib": 19895,
        "t_s": 214.047
      },
      {
        "power_w": 300.11,
        "gpu_util_pct": 97,
        "memory_used_mib": 19895,
        "t_s": 215.096
      },
      {
        "power_w": 300.15,
        "gpu_util_pct": 97,
        "memory_used_mib": 19895,
        "t_s": 216.147
      },
      {
        "power_w": 299.93,
        "gpu_util_pct": 96,
        "memory_used_mib": 19895,
        "t_s": 217.2
      },
      {
        "power_w": 300.43,
        "gpu_util_pct": 97,
        "memory_used_mib": 19895,
        "t_s": 218.25
      },
      {
        "power_w": 300.23,
        "gpu_util_pct": 96,
        "memory_used_mib": 19895,
        "t_s": 219.297
      },
      {
        "power_w": 300.7,
        "gpu_util_pct": 97,
        "memory_used_mib": 19895,
        "t_s": 220.359
      },
      {
        "power_w": 300.0,
        "gpu_util_pct": 97,
        "memory_used_mib": 19895,
        "t_s": 221.499
      },
      {
        "power_w": 300.21,
        "gpu_util_pct": 97,
        "memory_used_mib": 19895,
        "t_s": 222.55
      },
      {
        "power_w": 300.12,
        "gpu_util_pct": 98,
        "memory_used_mib": 19895,
        "t_s": 223.602
      },
      {
        "power_w": 300.56,
        "gpu_util_pct": 97,
        "memory_used_mib": 19895,
        "t_s": 224.657
      },
      {
        "power_w": 300.3,
        "gpu_util_pct": 97,
        "memory_used_mib": 19895,
        "t_s": 225.712
      },
      {
        "power_w": 300.27,
        "gpu_util_pct": 95,
        "memory_used_mib": 19895,
        "t_s": 226.762
      },
      {
        "power_w": 300.24,
        "gpu_util_pct": 96,
        "memory_used_mib": 19895,
        "t_s": 227.814
      },
      {
        "power_w": 300.35,
        "gpu_util_pct": 97,
        "memory_used_mib": 19895,
        "t_s": 228.864
      },
      {
        "power_w": 300.3,
        "gpu_util_pct": 97,
        "memory_used_mib": 19895,
        "t_s": 229.921
      },
      {
        "power_w": 300.47,
        "gpu_util_pct": 97,
        "memory_used_mib": 19895,
        "t_s": 230.974
      },
      {
        "power_w": 300.3,
        "gpu_util_pct": 97,
        "memory_used_mib": 19895,
        "t_s": 232.025
      },
      {
        "power_w": 300.15,
        "gpu_util_pct": 98,
        "memory_used_mib": 19895,
        "t_s": 233.079
      },
      {
        "power_w": 300.21,
        "gpu_util_pct": 97,
        "memory_used_mib": 19895,
        "t_s": 234.134
      },
      {
        "power_w": 300.57,
        "gpu_util_pct": 98,
        "memory_used_mib": 19895,
        "t_s": 235.187
      },
      {
        "power_w": 300.26,
        "gpu_util_pct": 97,
        "memory_used_mib": 19895,
        "t_s": 236.236
      },
      {
        "power_w": 300.21,
        "gpu_util_pct": 97,
        "memory_used_mib": 19895,
        "t_s": 237.291
      },
      {
        "power_w": 300.25,
        "gpu_util_pct": 97,
        "memory_used_mib": 19895,
        "t_s": 238.348
      },
      {
        "power_w": 300.26,
        "gpu_util_pct": 98,
        "memory_used_mib": 19895,
        "t_s": 239.392
      },
      {
        "power_w": 300.17,
        "gpu_util_pct": 97,
        "memory_used_mib": 19895,
        "t_s": 240.443
      },
      {
        "power_w": 300.12,
        "gpu_util_pct": 97,
        "memory_used_mib": 19895,
        "t_s": 241.493
      },
      {
        "power_w": 300.24,
        "gpu_util_pct": 98,
        "memory_used_mib": 19895,
        "t_s": 242.542
      },
      {
        "power_w": 299.98,
        "gpu_util_pct": 97,
        "memory_used_mib": 19895,
        "t_s": 243.587
      },
      {
        "power_w": 300.19,
        "gpu_util_pct": 98,
        "memory_used_mib": 19895,
        "t_s": 244.642
      },
      {
        "power_w": 300.15,
        "gpu_util_pct": 97,
        "memory_used_mib": 19895,
        "t_s": 245.694
      },
      {
        "power_w": 300.61,
        "gpu_util_pct": 97,
        "memory_used_mib": 19895,
        "t_s": 246.743
      },
      {
        "power_w": 300.55,
        "gpu_util_pct": 97,
        "memory_used_mib": 19895,
        "t_s": 247.793
      },
      {
        "power_w": 300.12,
        "gpu_util_pct": 97,
        "memory_used_mib": 19895,
        "t_s": 248.845
      },
      {
        "power_w": 299.94,
        "gpu_util_pct": 95,
        "memory_used_mib": 19895,
        "t_s": 249.896
      },
      {
        "power_w": 300.42,
        "gpu_util_pct": 97,
        "memory_used_mib": 19895,
        "t_s": 250.953
      },
      {
        "power_w": 300.02,
        "gpu_util_pct": 97,
        "memory_used_mib": 19895,
        "t_s": 252.004
      },
      {
        "power_w": 300.02,
        "gpu_util_pct": 97,
        "memory_used_mib": 19895,
        "t_s": 253.06
      },
      {
        "power_w": 300.66,
        "gpu_util_pct": 96,
        "memory_used_mib": 19895,
        "t_s": 254.109
      },
      {
        "power_w": 300.14,
        "gpu_util_pct": 98,
        "memory_used_mib": 19895,
        "t_s": 255.163
      },
      {
        "power_w": 299.95,
        "gpu_util_pct": 95,
        "memory_used_mib": 19895,
        "t_s": 256.218
      },
      {
        "power_w": 300.45,
        "gpu_util_pct": 98,
        "memory_used_mib": 19895,
        "t_s": 257.274
      },
      {
        "power_w": 234.36,
        "gpu_util_pct": 100,
        "memory_used_mib": 19895,
        "t_s": 258.326
      },
      {
        "power_w": 297.82,
        "gpu_util_pct": 100,
        "memory_used_mib": 19895,
        "t_s": 259.372
      },
      {
        "power_w": 298.36,
        "gpu_util_pct": 100,
        "memory_used_mib": 19895,
        "t_s": 260.416
      },
      {
        "power_w": 299.5,
        "gpu_util_pct": 100,
        "memory_used_mib": 19895,
        "t_s": 261.467
      },
      {
        "power_w": 299.35,
        "gpu_util_pct": 97,
        "memory_used_mib": 19895,
        "t_s": 262.521
      },
      {
        "power_w": 299.31,
        "gpu_util_pct": 97,
        "memory_used_mib": 19895,
        "t_s": 263.576
      },
      {
        "power_w": 300.47,
        "gpu_util_pct": 98,
        "memory_used_mib": 19895,
        "t_s": 264.631
      },
      {
        "power_w": 300.49,
        "gpu_util_pct": 98,
        "memory_used_mib": 19895,
        "t_s": 265.693
      },
      {
        "power_w": 300.5,
        "gpu_util_pct": 97,
        "memory_used_mib": 19895,
        "t_s": 266.742
      },
      {
        "power_w": 300.26,
        "gpu_util_pct": 97,
        "memory_used_mib": 19895,
        "t_s": 267.792
      },
      {
        "power_w": 300.34,
        "gpu_util_pct": 97,
        "memory_used_mib": 19895,
        "t_s": 268.846
      },
      {
        "power_w": 300.17,
        "gpu_util_pct": 97,
        "memory_used_mib": 19895,
        "t_s": 269.907
      },
      {
        "power_w": 300.58,
        "gpu_util_pct": 97,
        "memory_used_mib": 19895,
        "t_s": 270.959
      },
      {
        "power_w": 300.4,
        "gpu_util_pct": 98,
        "memory_used_mib": 19895,
        "t_s": 272.011
      },
      {
        "power_w": 300.07,
        "gpu_util_pct": 98,
        "memory_used_mib": 19895,
        "t_s": 273.069
      },
      {
        "power_w": 300.26,
        "gpu_util_pct": 97,
        "memory_used_mib": 19895,
        "t_s": 274.12
      },
      {
        "power_w": 300.24,
        "gpu_util_pct": 97,
        "memory_used_mib": 19895,
        "t_s": 275.171
      },
      {
        "power_w": 300.45,
        "gpu_util_pct": 98,
        "memory_used_mib": 19895,
        "t_s": 276.225
      },
      {
        "power_w": 300.47,
        "gpu_util_pct": 97,
        "memory_used_mib": 19895,
        "t_s": 277.274
      },
      {
        "power_w": 300.26,
        "gpu_util_pct": 97,
        "memory_used_mib": 19895,
        "t_s": 278.391
      },
      {
        "power_w": 300.41,
        "gpu_util_pct": 97,
        "memory_used_mib": 19895,
        "t_s": 279.442
      },
      {
        "power_w": 299.75,
        "gpu_util_pct": 96,
        "memory_used_mib": 19895,
        "t_s": 280.494
      },
      {
        "power_w": 300.87,
        "gpu_util_pct": 97,
        "memory_used_mib": 19895,
        "t_s": 281.545
      },
      {
        "power_w": 300.64,
        "gpu_util_pct": 98,
        "memory_used_mib": 19895,
        "t_s": 282.595
      },
      {
        "power_w": 224.1,
        "gpu_util_pct": 100,
        "memory_used_mib": 19895,
        "t_s": 283.647
      },
      {
        "power_w": 296.34,
        "gpu_util_pct": 93,
        "memory_used_mib": 19895,
        "t_s": 284.696
      },
      {
        "power_w": 300.9,
        "gpu_util_pct": 97,
        "memory_used_mib": 19895,
        "t_s": 285.758
      },
      {
        "power_w": 300.62,
        "gpu_util_pct": 97,
        "memory_used_mib": 19895,
        "t_s": 286.81
      },
      {
        "power_w": 300.62,
        "gpu_util_pct": 97,
        "memory_used_mib": 19895,
        "t_s": 287.869
      },
      {
        "power_w": 300.45,
        "gpu_util_pct": 97,
        "memory_used_mib": 19895,
        "t_s": 288.924
      },
      {
        "power_w": 300.16,
        "gpu_util_pct": 97,
        "memory_used_mib": 19895,
        "t_s": 289.975
      },
      {
        "power_w": 300.13,
        "gpu_util_pct": 97,
        "memory_used_mib": 19895,
        "t_s": 291.026
      },
      {
        "power_w": 300.32,
        "gpu_util_pct": 97,
        "memory_used_mib": 19895,
        "t_s": 292.077
      },
      {
        "power_w": 300.44,
        "gpu_util_pct": 97,
        "memory_used_mib": 19895,
        "t_s": 293.127
      },
      {
        "power_w": 299.97,
        "gpu_util_pct": 97,
        "memory_used_mib": 19895,
        "t_s": 294.181
      },
      {
        "power_w": 300.14,
        "gpu_util_pct": 97,
        "memory_used_mib": 19895,
        "t_s": 295.236
      },
      {
        "power_w": 300.7,
        "gpu_util_pct": 97,
        "memory_used_mib": 19895,
        "t_s": 296.288
      },
      {
        "power_w": 300.34,
        "gpu_util_pct": 96,
        "memory_used_mib": 19895,
        "t_s": 297.339
      },
      {
        "power_w": 300.36,
        "gpu_util_pct": 97,
        "memory_used_mib": 19895,
        "t_s": 298.391
      },
      {
        "power_w": 300.54,
        "gpu_util_pct": 97,
        "memory_used_mib": 19895,
        "t_s": 299.444
      },
      {
        "power_w": 300.61,
        "gpu_util_pct": 96,
        "memory_used_mib": 19895,
        "t_s": 300.496
      },
      {
        "power_w": 300.0,
        "gpu_util_pct": 97,
        "memory_used_mib": 19895,
        "t_s": 301.566
      },
      {
        "power_w": 300.59,
        "gpu_util_pct": 97,
        "memory_used_mib": 19895,
        "t_s": 302.618
      },
      {
        "power_w": 300.14,
        "gpu_util_pct": 95,
        "memory_used_mib": 19895,
        "t_s": 303.671
      },
      {
        "power_w": 300.23,
        "gpu_util_pct": 98,
        "memory_used_mib": 19895,
        "t_s": 304.713
      },
      {
        "power_w": 300.46,
        "gpu_util_pct": 97,
        "memory_used_mib": 19895,
        "t_s": 305.767
      },
      {
        "power_w": 300.53,
        "gpu_util_pct": 97,
        "memory_used_mib": 19895,
        "t_s": 306.826
      },
      {
        "power_w": 300.37,
        "gpu_util_pct": 97,
        "memory_used_mib": 19895,
        "t_s": 307.877
      },
      {
        "power_w": 300.52,
        "gpu_util_pct": 97,
        "memory_used_mib": 19895,
        "t_s": 308.928
      },
      {
        "power_w": 300.37,
        "gpu_util_pct": 97,
        "memory_used_mib": 19895,
        "t_s": 309.987
      },
      {
        "power_w": 300.22,
        "gpu_util_pct": 97,
        "memory_used_mib": 19895,
        "t_s": 311.038
      },
      {
        "power_w": 300.39,
        "gpu_util_pct": 97,
        "memory_used_mib": 19895,
        "t_s": 312.093
      },
      {
        "power_w": 300.01,
        "gpu_util_pct": 97,
        "memory_used_mib": 19895,
        "t_s": 313.146
      },
      {
        "power_w": 300.75,
        "gpu_util_pct": 97,
        "memory_used_mib": 19895,
        "t_s": 314.201
      },
      {
        "power_w": 300.47,
        "gpu_util_pct": 97,
        "memory_used_mib": 19895,
        "t_s": 315.255
      },
      {
        "power_w": 300.46,
        "gpu_util_pct": 97,
        "memory_used_mib": 19895,
        "t_s": 316.309
      },
      {
        "power_w": 300.47,
        "gpu_util_pct": 97,
        "memory_used_mib": 19895,
        "t_s": 317.362
      },
      {
        "power_w": 300.31,
        "gpu_util_pct": 97,
        "memory_used_mib": 19895,
        "t_s": 318.416
      },
      {
        "power_w": 300.47,
        "gpu_util_pct": 97,
        "memory_used_mib": 19895,
        "t_s": 319.473
      },
      {
        "power_w": 300.25,
        "gpu_util_pct": 97,
        "memory_used_mib": 19895,
        "t_s": 320.54
      },
      {
        "power_w": 300.2,
        "gpu_util_pct": 98,
        "memory_used_mib": 19895,
        "t_s": 321.595
      },
      {
        "power_w": 300.27,
        "gpu_util_pct": 97,
        "memory_used_mib": 19895,
        "t_s": 322.676
      },
      {
        "power_w": 300.25,
        "gpu_util_pct": 97,
        "memory_used_mib": 19895,
        "t_s": 323.726
      },
      {
        "power_w": 300.42,
        "gpu_util_pct": 97,
        "memory_used_mib": 19895,
        "t_s": 324.777
      },
      {
        "power_w": 300.14,
        "gpu_util_pct": 97,
        "memory_used_mib": 19895,
        "t_s": 325.835
      },
      {
        "power_w": 300.67,
        "gpu_util_pct": 97,
        "memory_used_mib": 19895,
        "t_s": 326.889
      },
      {
        "power_w": 300.44,
        "gpu_util_pct": 97,
        "memory_used_mib": 19895,
        "t_s": 327.943
      },
      {
        "power_w": 300.31,
        "gpu_util_pct": 97,
        "memory_used_mib": 19895,
        "t_s": 328.99
      },
      {
        "power_w": 300.4,
        "gpu_util_pct": 97,
        "memory_used_mib": 19895,
        "t_s": 330.047
      },
      {
        "power_w": 300.15,
        "gpu_util_pct": 94,
        "memory_used_mib": 19895,
        "t_s": 331.104
      },
      {
        "power_w": 300.72,
        "gpu_util_pct": 97,
        "memory_used_mib": 19895,
        "t_s": 332.158
      },
      {
        "power_w": 300.11,
        "gpu_util_pct": 97,
        "memory_used_mib": 19895,
        "t_s": 333.213
      },
      {
        "power_w": 300.4,
        "gpu_util_pct": 97,
        "memory_used_mib": 19895,
        "t_s": 334.277
      },
      {
        "power_w": 300.49,
        "gpu_util_pct": 97,
        "memory_used_mib": 19895,
        "t_s": 335.335
      },
      {
        "power_w": 300.29,
        "gpu_util_pct": 97,
        "memory_used_mib": 19895,
        "t_s": 336.388
      },
      {
        "power_w": 300.55,
        "gpu_util_pct": 97,
        "memory_used_mib": 19895,
        "t_s": 337.444
      },
      {
        "power_w": 300.34,
        "gpu_util_pct": 97,
        "memory_used_mib": 19895,
        "t_s": 338.496
      },
      {
        "power_w": 300.22,
        "gpu_util_pct": 97,
        "memory_used_mib": 19895,
        "t_s": 339.548
      },
      {
        "power_w": 300.38,
        "gpu_util_pct": 97,
        "memory_used_mib": 19895,
        "t_s": 340.607
      },
      {
        "power_w": 238.94,
        "gpu_util_pct": 0,
        "memory_used_mib": 19895,
        "t_s": 341.673
      },
      {
        "power_w": 301.84,
        "gpu_util_pct": 97,
        "memory_used_mib": 19895,
        "t_s": 342.733
      },
      {
        "power_w": 300.44,
        "gpu_util_pct": 97,
        "memory_used_mib": 19895,
        "t_s": 343.788
      },
      {
        "power_w": 300.7,
        "gpu_util_pct": 97,
        "memory_used_mib": 19895,
        "t_s": 344.843
      },
      {
        "power_w": 300.46,
        "gpu_util_pct": 97,
        "memory_used_mib": 19895,
        "t_s": 345.899
      },
      {
        "power_w": 300.5,
        "gpu_util_pct": 97,
        "memory_used_mib": 19895,
        "t_s": 346.95
      },
      {
        "power_w": 299.98,
        "gpu_util_pct": 97,
        "memory_used_mib": 19895,
        "t_s": 348.007
      },
      {
        "power_w": 299.94,
        "gpu_util_pct": 97,
        "memory_used_mib": 19895,
        "t_s": 349.059
      },
      {
        "power_w": 300.18,
        "gpu_util_pct": 97,
        "memory_used_mib": 19895,
        "t_s": 350.109
      },
      {
        "power_w": 300.45,
        "gpu_util_pct": 98,
        "memory_used_mib": 19895,
        "t_s": 351.161
      },
      {
        "power_w": 300.61,
        "gpu_util_pct": 97,
        "memory_used_mib": 19895,
        "t_s": 352.213
      },
      {
        "power_w": 300.51,
        "gpu_util_pct": 97,
        "memory_used_mib": 19895,
        "t_s": 353.264
      },
      {
        "power_w": 300.71,
        "gpu_util_pct": 98,
        "memory_used_mib": 19895,
        "t_s": 354.324
      },
      {
        "power_w": 300.22,
        "gpu_util_pct": 97,
        "memory_used_mib": 19895,
        "t_s": 355.375
      },
      {
        "power_w": 300.81,
        "gpu_util_pct": 97,
        "memory_used_mib": 19895,
        "t_s": 356.426
      },
      {
        "power_w": 300.1,
        "gpu_util_pct": 97,
        "memory_used_mib": 19895,
        "t_s": 357.481
      },
      {
        "power_w": 300.75,
        "gpu_util_pct": 96,
        "memory_used_mib": 19895,
        "t_s": 358.531
      },
      {
        "power_w": 300.21,
        "gpu_util_pct": 98,
        "memory_used_mib": 19895,
        "t_s": 359.58
      },
      {
        "power_w": 299.96,
        "gpu_util_pct": 97,
        "memory_used_mib": 19895,
        "t_s": 360.631
      },
      {
        "power_w": 300.38,
        "gpu_util_pct": 97,
        "memory_used_mib": 19895,
        "t_s": 361.685
      },
      {
        "power_w": 300.42,
        "gpu_util_pct": 95,
        "memory_used_mib": 19895,
        "t_s": 362.738
      },
      {
        "power_w": 300.73,
        "gpu_util_pct": 98,
        "memory_used_mib": 19895,
        "t_s": 363.795
      },
      {
        "power_w": 300.27,
        "gpu_util_pct": 97,
        "memory_used_mib": 19895,
        "t_s": 364.85
      },
      {
        "power_w": 300.38,
        "gpu_util_pct": 97,
        "memory_used_mib": 19895,
        "t_s": 365.9
      },
      {
        "power_w": 300.72,
        "gpu_util_pct": 97,
        "memory_used_mib": 19895,
        "t_s": 366.956
      },
      {
        "power_w": 300.45,
        "gpu_util_pct": 97,
        "memory_used_mib": 19895,
        "t_s": 368.007
      },
      {
        "power_w": 300.21,
        "gpu_util_pct": 97,
        "memory_used_mib": 19895,
        "t_s": 369.069
      },
      {
        "power_w": 300.18,
        "gpu_util_pct": 97,
        "memory_used_mib": 19895,
        "t_s": 370.131
      },
      {
        "power_w": 300.58,
        "gpu_util_pct": 98,
        "memory_used_mib": 19895,
        "t_s": 371.186
      },
      {
        "power_w": 300.29,
        "gpu_util_pct": 97,
        "memory_used_mib": 19895,
        "t_s": 372.241
      },
      {
        "power_w": 300.12,
        "gpu_util_pct": 95,
        "memory_used_mib": 19895,
        "t_s": 373.29
      },
      {
        "power_w": 300.36,
        "gpu_util_pct": 97,
        "memory_used_mib": 19895,
        "t_s": 374.344
      },
      {
        "power_w": 300.25,
        "gpu_util_pct": 97,
        "memory_used_mib": 19895,
        "t_s": 375.407
      },
      {
        "power_w": 300.26,
        "gpu_util_pct": 98,
        "memory_used_mib": 19895,
        "t_s": 376.458
      },
      {
        "power_w": 300.52,
        "gpu_util_pct": 97,
        "memory_used_mib": 19895,
        "t_s": 377.513
      },
      {
        "power_w": 300.44,
        "gpu_util_pct": 98,
        "memory_used_mib": 19895,
        "t_s": 378.565
      },
      {
        "power_w": 300.22,
        "gpu_util_pct": 97,
        "memory_used_mib": 19895,
        "t_s": 379.628
      },
      {
        "power_w": 300.08,
        "gpu_util_pct": 96,
        "memory_used_mib": 19895,
        "t_s": 380.773
      },
      {
        "power_w": 300.41,
        "gpu_util_pct": 97,
        "memory_used_mib": 19895,
        "t_s": 381.828
      },
      {
        "power_w": 300.29,
        "gpu_util_pct": 97,
        "memory_used_mib": 19895,
        "t_s": 382.897
      },
      {
        "power_w": 300.23,
        "gpu_util_pct": 97,
        "memory_used_mib": 19895,
        "t_s": 383.957
      },
      {
        "power_w": 300.45,
        "gpu_util_pct": 97,
        "memory_used_mib": 19895,
        "t_s": 385.008
      },
      {
        "power_w": 300.21,
        "gpu_util_pct": 97,
        "memory_used_mib": 19895,
        "t_s": 386.058
      },
      {
        "power_w": 300.5,
        "gpu_util_pct": 95,
        "memory_used_mib": 19895,
        "t_s": 387.108
      },
      {
        "power_w": 300.49,
        "gpu_util_pct": 97,
        "memory_used_mib": 19895,
        "t_s": 388.165
      },
      {
        "power_w": 300.37,
        "gpu_util_pct": 97,
        "memory_used_mib": 19895,
        "t_s": 389.215
      },
      {
        "power_w": 300.41,
        "gpu_util_pct": 98,
        "memory_used_mib": 19895,
        "t_s": 390.267
      },
      {
        "power_w": 300.33,
        "gpu_util_pct": 97,
        "memory_used_mib": 19895,
        "t_s": 391.317
      },
      {
        "power_w": 299.49,
        "gpu_util_pct": 97,
        "memory_used_mib": 19895,
        "t_s": 392.369
      },
      {
        "power_w": 300.45,
        "gpu_util_pct": 97,
        "memory_used_mib": 19895,
        "t_s": 393.421
      },
      {
        "power_w": 300.18,
        "gpu_util_pct": 97,
        "memory_used_mib": 19895,
        "t_s": 394.469
      },
      {
        "power_w": 300.33,
        "gpu_util_pct": 97,
        "memory_used_mib": 19895,
        "t_s": 395.52
      },
      {
        "power_w": 300.11,
        "gpu_util_pct": 97,
        "memory_used_mib": 19895,
        "t_s": 396.574
      },
      {
        "power_w": 300.28,
        "gpu_util_pct": 96,
        "memory_used_mib": 19895,
        "t_s": 397.624
      },
      {
        "power_w": 300.55,
        "gpu_util_pct": 98,
        "memory_used_mib": 19895,
        "t_s": 398.676
      },
      {
        "power_w": 300.03,
        "gpu_util_pct": 96,
        "memory_used_mib": 19895,
        "t_s": 399.735
      },
      {
        "power_w": 300.4,
        "gpu_util_pct": 94,
        "memory_used_mib": 19895,
        "t_s": 400.805
      },
      {
        "power_w": 300.59,
        "gpu_util_pct": 98,
        "memory_used_mib": 19895,
        "t_s": 401.873
      },
      {
        "power_w": 299.94,
        "gpu_util_pct": 97,
        "memory_used_mib": 19895,
        "t_s": 402.926
      },
      {
        "power_w": 300.67,
        "gpu_util_pct": 97,
        "memory_used_mib": 19895,
        "t_s": 403.976
      },
      {
        "power_w": 300.44,
        "gpu_util_pct": 97,
        "memory_used_mib": 19895,
        "t_s": 405.029
      },
      {
        "power_w": 300.23,
        "gpu_util_pct": 97,
        "memory_used_mib": 19895,
        "t_s": 406.083
      },
      {
        "power_w": 300.04,
        "gpu_util_pct": 97,
        "memory_used_mib": 19895,
        "t_s": 407.142
      },
      {
        "power_w": 300.48,
        "gpu_util_pct": 98,
        "memory_used_mib": 19895,
        "t_s": 408.193
      },
      {
        "power_w": 300.04,
        "gpu_util_pct": 98,
        "memory_used_mib": 19895,
        "t_s": 409.251
      },
      {
        "power_w": 300.79,
        "gpu_util_pct": 97,
        "memory_used_mib": 19895,
        "t_s": 410.312
      },
      {
        "power_w": 300.74,
        "gpu_util_pct": 97,
        "memory_used_mib": 19895,
        "t_s": 411.365
      },
      {
        "power_w": 300.33,
        "gpu_util_pct": 97,
        "memory_used_mib": 19895,
        "t_s": 412.417
      },
      {
        "power_w": 299.73,
        "gpu_util_pct": 97,
        "memory_used_mib": 19895,
        "t_s": 413.473
      },
      {
        "power_w": 300.14,
        "gpu_util_pct": 97,
        "memory_used_mib": 19895,
        "t_s": 414.532
      },
      {
        "power_w": 288.61,
        "gpu_util_pct": 59,
        "memory_used_mib": 19895,
        "t_s": 415.585
      },
      {
        "power_w": 253.57,
        "gpu_util_pct": 92,
        "memory_used_mib": 19895,
        "t_s": 416.629
      },
      {
        "power_w": 299.83,
        "gpu_util_pct": 98,
        "memory_used_mib": 19895,
        "t_s": 417.683
      },
      {
        "power_w": 299.67,
        "gpu_util_pct": 96,
        "memory_used_mib": 19895,
        "t_s": 418.739
      },
      {
        "power_w": 300.31,
        "gpu_util_pct": 97,
        "memory_used_mib": 19895,
        "t_s": 419.792
      },
      {
        "power_w": 300.06,
        "gpu_util_pct": 97,
        "memory_used_mib": 19895,
        "t_s": 420.845
      },
      {
        "power_w": 300.56,
        "gpu_util_pct": 97,
        "memory_used_mib": 19895,
        "t_s": 421.895
      },
      {
        "power_w": 300.78,
        "gpu_util_pct": 97,
        "memory_used_mib": 19895,
        "t_s": 422.946
      },
      {
        "power_w": 300.32,
        "gpu_util_pct": 95,
        "memory_used_mib": 19895,
        "t_s": 424.003
      },
      {
        "power_w": 300.34,
        "gpu_util_pct": 96,
        "memory_used_mib": 19895,
        "t_s": 425.052
      },
      {
        "power_w": 300.12,
        "gpu_util_pct": 97,
        "memory_used_mib": 19895,
        "t_s": 426.107
      },
      {
        "power_w": 300.25,
        "gpu_util_pct": 98,
        "memory_used_mib": 19895,
        "t_s": 427.179
      },
      {
        "power_w": 300.24,
        "gpu_util_pct": 97,
        "memory_used_mib": 19895,
        "t_s": 428.228
      },
      {
        "power_w": 300.61,
        "gpu_util_pct": 98,
        "memory_used_mib": 19895,
        "t_s": 429.28
      },
      {
        "power_w": 300.34,
        "gpu_util_pct": 98,
        "memory_used_mib": 19895,
        "t_s": 430.329
      },
      {
        "power_w": 300.26,
        "gpu_util_pct": 97,
        "memory_used_mib": 19895,
        "t_s": 431.38
      },
      {
        "power_w": 300.4,
        "gpu_util_pct": 97,
        "memory_used_mib": 19895,
        "t_s": 432.429
      },
      {
        "power_w": 300.44,
        "gpu_util_pct": 97,
        "memory_used_mib": 19895,
        "t_s": 433.48
      },
      {
        "power_w": 300.1,
        "gpu_util_pct": 97,
        "memory_used_mib": 19895,
        "t_s": 434.531
      },
      {
        "power_w": 300.56,
        "gpu_util_pct": 97,
        "memory_used_mib": 19895,
        "t_s": 435.581
      },
      {
        "power_w": 300.23,
        "gpu_util_pct": 97,
        "memory_used_mib": 19895,
        "t_s": 436.632
      },
      {
        "power_w": 300.24,
        "gpu_util_pct": 97,
        "memory_used_mib": 19895,
        "t_s": 437.683
      },
      {
        "power_w": 300.43,
        "gpu_util_pct": 97,
        "memory_used_mib": 19895,
        "t_s": 438.735
      },
      {
        "power_w": 300.51,
        "gpu_util_pct": 97,
        "memory_used_mib": 19895,
        "t_s": 439.784
      },
      {
        "power_w": 300.34,
        "gpu_util_pct": 98,
        "memory_used_mib": 19895,
        "t_s": 440.835
      },
      {
        "power_w": 300.48,
        "gpu_util_pct": 97,
        "memory_used_mib": 19895,
        "t_s": 441.886
      },
      {
        "power_w": 300.31,
        "gpu_util_pct": 97,
        "memory_used_mib": 19895,
        "t_s": 442.928
      },
      {
        "power_w": 299.65,
        "gpu_util_pct": 96,
        "memory_used_mib": 19895,
        "t_s": 443.979
      },
      {
        "power_w": 300.67,
        "gpu_util_pct": 98,
        "memory_used_mib": 19895,
        "t_s": 445.034
      },
      {
        "power_w": 300.45,
        "gpu_util_pct": 97,
        "memory_used_mib": 19895,
        "t_s": 446.086
      },
      {
        "power_w": 300.59,
        "gpu_util_pct": 98,
        "memory_used_mib": 19895,
        "t_s": 447.136
      },
      {
        "power_w": 300.38,
        "gpu_util_pct": 98,
        "memory_used_mib": 19895,
        "t_s": 448.187
      },
      {
        "power_w": 300.21,
        "gpu_util_pct": 98,
        "memory_used_mib": 19895,
        "t_s": 449.238
      },
      {
        "power_w": 300.64,
        "gpu_util_pct": 97,
        "memory_used_mib": 19895,
        "t_s": 450.289
      },
      {
        "power_w": 300.33,
        "gpu_util_pct": 97,
        "memory_used_mib": 19895,
        "t_s": 451.34
      },
      {
        "power_w": 300.14,
        "gpu_util_pct": 98,
        "memory_used_mib": 19895,
        "t_s": 452.394
      },
      {
        "power_w": 300.27,
        "gpu_util_pct": 97,
        "memory_used_mib": 19895,
        "t_s": 453.446
      },
      {
        "power_w": 300.5,
        "gpu_util_pct": 98,
        "memory_used_mib": 19895,
        "t_s": 454.498
      },
      {
        "power_w": 300.28,
        "gpu_util_pct": 97,
        "memory_used_mib": 19895,
        "t_s": 455.558
      },
      {
        "power_w": 300.43,
        "gpu_util_pct": 97,
        "memory_used_mib": 19895,
        "t_s": 456.617
      },
      {
        "power_w": 300.28,
        "gpu_util_pct": 96,
        "memory_used_mib": 19895,
        "t_s": 457.668
      },
      {
        "power_w": 300.32,
        "gpu_util_pct": 97,
        "memory_used_mib": 19895,
        "t_s": 458.736
      },
      {
        "power_w": 300.07,
        "gpu_util_pct": 97,
        "memory_used_mib": 19895,
        "t_s": 459.788
      },
      {
        "power_w": 300.27,
        "gpu_util_pct": 95,
        "memory_used_mib": 19895,
        "t_s": 460.832
      },
      {
        "power_w": 300.41,
        "gpu_util_pct": 97,
        "memory_used_mib": 19895,
        "t_s": 461.876
      },
      {
        "power_w": 300.54,
        "gpu_util_pct": 97,
        "memory_used_mib": 19895,
        "t_s": 462.936
      },
      {
        "power_w": 300.66,
        "gpu_util_pct": 97,
        "memory_used_mib": 19895,
        "t_s": 463.986
      },
      {
        "power_w": 300.44,
        "gpu_util_pct": 98,
        "memory_used_mib": 19895,
        "t_s": 465.036
      },
      {
        "power_w": 300.22,
        "gpu_util_pct": 97,
        "memory_used_mib": 19895,
        "t_s": 466.086
      },
      {
        "power_w": 300.19,
        "gpu_util_pct": 97,
        "memory_used_mib": 19895,
        "t_s": 467.136
      },
      {
        "power_w": 300.3,
        "gpu_util_pct": 97,
        "memory_used_mib": 19895,
        "t_s": 468.187
      },
      {
        "power_w": 300.54,
        "gpu_util_pct": 97,
        "memory_used_mib": 19895,
        "t_s": 469.234
      },
      {
        "power_w": 300.25,
        "gpu_util_pct": 98,
        "memory_used_mib": 19895,
        "t_s": 470.286
      },
      {
        "power_w": 300.21,
        "gpu_util_pct": 97,
        "memory_used_mib": 19895,
        "t_s": 471.338
      },
      {
        "power_w": 300.57,
        "gpu_util_pct": 97,
        "memory_used_mib": 19895,
        "t_s": 472.388
      },
      {
        "power_w": 300.38,
        "gpu_util_pct": 98,
        "memory_used_mib": 19895,
        "t_s": 473.431
      },
      {
        "power_w": 300.37,
        "gpu_util_pct": 97,
        "memory_used_mib": 19895,
        "t_s": 474.482
      },
      {
        "power_w": 300.12,
        "gpu_util_pct": 97,
        "memory_used_mib": 19895,
        "t_s": 475.535
      },
      {
        "power_w": 294.54,
        "gpu_util_pct": 97,
        "memory_used_mib": 19895,
        "t_s": 476.588
      },
      {
        "power_w": 260.63,
        "gpu_util_pct": 96,
        "memory_used_mib": 19895,
        "t_s": 477.635
      },
      {
        "power_w": 299.42,
        "gpu_util_pct": 97,
        "memory_used_mib": 19895,
        "t_s": 478.687
      },
      {
        "power_w": 300.32,
        "gpu_util_pct": 98,
        "memory_used_mib": 19895,
        "t_s": 479.736
      },
      {
        "power_w": 300.33,
        "gpu_util_pct": 96,
        "memory_used_mib": 19895,
        "t_s": 480.787
      },
      {
        "power_w": 300.48,
        "gpu_util_pct": 97,
        "memory_used_mib": 19895,
        "t_s": 481.841
      },
      {
        "power_w": 299.55,
        "gpu_util_pct": 95,
        "memory_used_mib": 19895,
        "t_s": 482.892
      },
      {
        "power_w": 301.05,
        "gpu_util_pct": 98,
        "memory_used_mib": 19895,
        "t_s": 483.954
      },
      {
        "power_w": 301.02,
        "gpu_util_pct": 97,
        "memory_used_mib": 19895,
        "t_s": 485.001
      },
      {
        "power_w": 300.57,
        "gpu_util_pct": 97,
        "memory_used_mib": 19895,
        "t_s": 486.048
      },
      {
        "power_w": 300.23,
        "gpu_util_pct": 97,
        "memory_used_mib": 19895,
        "t_s": 487.095
      },
      {
        "power_w": 300.27,
        "gpu_util_pct": 98,
        "memory_used_mib": 19895,
        "t_s": 488.152
      },
      {
        "power_w": 300.47,
        "gpu_util_pct": 98,
        "memory_used_mib": 19895,
        "t_s": 489.203
      },
      {
        "power_w": 300.26,
        "gpu_util_pct": 98,
        "memory_used_mib": 19895,
        "t_s": 490.251
      },
      {
        "power_w": 300.25,
        "gpu_util_pct": 98,
        "memory_used_mib": 19895,
        "t_s": 491.298
      },
      {
        "power_w": 300.58,
        "gpu_util_pct": 98,
        "memory_used_mib": 19895,
        "t_s": 492.345
      },
      {
        "power_w": 300.3,
        "gpu_util_pct": 96,
        "memory_used_mib": 19895,
        "t_s": 493.395
      },
      {
        "power_w": 300.5,
        "gpu_util_pct": 98,
        "memory_used_mib": 19895,
        "t_s": 494.447
      },
      {
        "power_w": 300.5,
        "gpu_util_pct": 98,
        "memory_used_mib": 19895,
        "t_s": 495.497
      },
      {
        "power_w": 300.22,
        "gpu_util_pct": 97,
        "memory_used_mib": 19895,
        "t_s": 496.544
      },
      {
        "power_w": 300.33,
        "gpu_util_pct": 98,
        "memory_used_mib": 19895,
        "t_s": 497.592
      },
      {
        "power_w": 300.36,
        "gpu_util_pct": 97,
        "memory_used_mib": 19895,
        "t_s": 498.641
      },
      {
        "power_w": 300.07,
        "gpu_util_pct": 96,
        "memory_used_mib": 19895,
        "t_s": 499.69
      },
      {
        "power_w": 300.24,
        "gpu_util_pct": 98,
        "memory_used_mib": 19895,
        "t_s": 500.741
      },
      {
        "power_w": 300.48,
        "gpu_util_pct": 97,
        "memory_used_mib": 19895,
        "t_s": 501.793
      },
      {
        "power_w": 300.5,
        "gpu_util_pct": 97,
        "memory_used_mib": 19895,
        "t_s": 502.838
      },
      {
        "power_w": 300.47,
        "gpu_util_pct": 97,
        "memory_used_mib": 19895,
        "t_s": 503.881
      },
      {
        "power_w": 300.65,
        "gpu_util_pct": 98,
        "memory_used_mib": 19895,
        "t_s": 504.936
      },
      {
        "power_w": 300.17,
        "gpu_util_pct": 97,
        "memory_used_mib": 19895,
        "t_s": 505.979
      },
      {
        "power_w": 300.43,
        "gpu_util_pct": 98,
        "memory_used_mib": 19895,
        "t_s": 507.031
      },
      {
        "power_w": 300.07,
        "gpu_util_pct": 97,
        "memory_used_mib": 19895,
        "t_s": 508.086
      },
      {
        "power_w": 299.95,
        "gpu_util_pct": 98,
        "memory_used_mib": 19895,
        "t_s": 509.134
      },
      {
        "power_w": 300.74,
        "gpu_util_pct": 98,
        "memory_used_mib": 19895,
        "t_s": 510.183
      },
      {
        "power_w": 300.67,
        "gpu_util_pct": 97,
        "memory_used_mib": 19895,
        "t_s": 511.229
      },
      {
        "power_w": 300.07,
        "gpu_util_pct": 97,
        "memory_used_mib": 19895,
        "t_s": 512.279
      },
      {
        "power_w": 300.56,
        "gpu_util_pct": 97,
        "memory_used_mib": 19895,
        "t_s": 513.335
      },
      {
        "power_w": 300.54,
        "gpu_util_pct": 98,
        "memory_used_mib": 19895,
        "t_s": 514.385
      },
      {
        "power_w": 300.71,
        "gpu_util_pct": 97,
        "memory_used_mib": 19895,
        "t_s": 515.426
      },
      {
        "power_w": 300.37,
        "gpu_util_pct": 98,
        "memory_used_mib": 19895,
        "t_s": 516.474
      },
      {
        "power_w": 299.87,
        "gpu_util_pct": 97,
        "memory_used_mib": 19895,
        "t_s": 517.515
      },
      {
        "power_w": 300.37,
        "gpu_util_pct": 97,
        "memory_used_mib": 19895,
        "t_s": 518.566
      },
      {
        "power_w": 300.45,
        "gpu_util_pct": 97,
        "memory_used_mib": 19895,
        "t_s": 519.622
      },
      {
        "power_w": 300.13,
        "gpu_util_pct": 98,
        "memory_used_mib": 19895,
        "t_s": 520.673
      },
      {
        "power_w": 300.28,
        "gpu_util_pct": 97,
        "memory_used_mib": 19895,
        "t_s": 521.717
      },
      {
        "power_w": 300.58,
        "gpu_util_pct": 97,
        "memory_used_mib": 19895,
        "t_s": 522.768
      },
      {
        "power_w": 300.02,
        "gpu_util_pct": 98,
        "memory_used_mib": 19895,
        "t_s": 523.821
      },
      {
        "power_w": 300.28,
        "gpu_util_pct": 97,
        "memory_used_mib": 19895,
        "t_s": 524.871
      },
      {
        "power_w": 299.71,
        "gpu_util_pct": 97,
        "memory_used_mib": 19895,
        "t_s": 525.994
      },
      {
        "power_w": 300.64,
        "gpu_util_pct": 97,
        "memory_used_mib": 19895,
        "t_s": 527.038
      },
      {
        "power_w": 300.73,
        "gpu_util_pct": 97,
        "memory_used_mib": 19895,
        "t_s": 528.08
      },
      {
        "power_w": 300.35,
        "gpu_util_pct": 97,
        "memory_used_mib": 19895,
        "t_s": 529.129
      },
      {
        "power_w": 300.35,
        "gpu_util_pct": 97,
        "memory_used_mib": 19895,
        "t_s": 530.18
      },
      {
        "power_w": 300.08,
        "gpu_util_pct": 97,
        "memory_used_mib": 19895,
        "t_s": 531.236
      },
      {
        "power_w": 301.0,
        "gpu_util_pct": 98,
        "memory_used_mib": 19895,
        "t_s": 532.288
      },
      {
        "power_w": 300.15,
        "gpu_util_pct": 96,
        "memory_used_mib": 19895,
        "t_s": 533.332
      },
      {
        "power_w": 300.67,
        "gpu_util_pct": 97,
        "memory_used_mib": 19895,
        "t_s": 534.385
      },
      {
        "power_w": 300.45,
        "gpu_util_pct": 96,
        "memory_used_mib": 19895,
        "t_s": 535.441
      },
      {
        "power_w": 300.19,
        "gpu_util_pct": 98,
        "memory_used_mib": 19895,
        "t_s": 536.483
      },
      {
        "power_w": 300.09,
        "gpu_util_pct": 97,
        "memory_used_mib": 19895,
        "t_s": 537.528
      },
      {
        "power_w": 299.96,
        "gpu_util_pct": 97,
        "memory_used_mib": 19895,
        "t_s": 538.581
      },
      {
        "power_w": 300.66,
        "gpu_util_pct": 98,
        "memory_used_mib": 19895,
        "t_s": 539.632
      },
      {
        "power_w": 300.29,
        "gpu_util_pct": 97,
        "memory_used_mib": 19895,
        "t_s": 540.681
      },
      {
        "power_w": 300.42,
        "gpu_util_pct": 97,
        "memory_used_mib": 19895,
        "t_s": 541.732
      },
      {
        "power_w": 300.44,
        "gpu_util_pct": 97,
        "memory_used_mib": 19895,
        "t_s": 542.783
      },
      {
        "power_w": 300.31,
        "gpu_util_pct": 96,
        "memory_used_mib": 19895,
        "t_s": 543.836
      },
      {
        "power_w": 299.89,
        "gpu_util_pct": 98,
        "memory_used_mib": 19895,
        "t_s": 544.89
      },
      {
        "power_w": 241.89,
        "gpu_util_pct": 40,
        "memory_used_mib": 19895,
        "t_s": 545.942
      },
      {
        "power_w": 298.55,
        "gpu_util_pct": 98,
        "memory_used_mib": 19895,
        "t_s": 547.001
      },
      {
        "power_w": 300.73,
        "gpu_util_pct": 96,
        "memory_used_mib": 19895,
        "t_s": 548.052
      },
      {
        "power_w": 300.58,
        "gpu_util_pct": 97,
        "memory_used_mib": 19895,
        "t_s": 549.104
      },
      {
        "power_w": 300.26,
        "gpu_util_pct": 97,
        "memory_used_mib": 19895,
        "t_s": 550.159
      },
      {
        "power_w": 300.16,
        "gpu_util_pct": 97,
        "memory_used_mib": 19895,
        "t_s": 551.21
      },
      {
        "power_w": 300.08,
        "gpu_util_pct": 97,
        "memory_used_mib": 19895,
        "t_s": 552.259
      },
      {
        "power_w": 299.92,
        "gpu_util_pct": 97,
        "memory_used_mib": 19895,
        "t_s": 553.308
      },
      {
        "power_w": 300.38,
        "gpu_util_pct": 97,
        "memory_used_mib": 19895,
        "t_s": 554.361
      },
      {
        "power_w": 300.51,
        "gpu_util_pct": 98,
        "memory_used_mib": 19895,
        "t_s": 555.41
      },
      {
        "power_w": 300.37,
        "gpu_util_pct": 97,
        "memory_used_mib": 19895,
        "t_s": 556.463
      },
      {
        "power_w": 300.23,
        "gpu_util_pct": 98,
        "memory_used_mib": 19895,
        "t_s": 557.525
      },
      {
        "power_w": 300.4,
        "gpu_util_pct": 98,
        "memory_used_mib": 19895,
        "t_s": 558.578
      },
      {
        "power_w": 300.37,
        "gpu_util_pct": 97,
        "memory_used_mib": 19895,
        "t_s": 559.63
      },
      {
        "power_w": 300.51,
        "gpu_util_pct": 97,
        "memory_used_mib": 19895,
        "t_s": 560.682
      },
      {
        "power_w": 300.17,
        "gpu_util_pct": 97,
        "memory_used_mib": 19895,
        "t_s": 561.733
      },
      {
        "power_w": 300.24,
        "gpu_util_pct": 97,
        "memory_used_mib": 19895,
        "t_s": 562.792
      },
      {
        "power_w": 299.7,
        "gpu_util_pct": 97,
        "memory_used_mib": 19895,
        "t_s": 563.841
      },
      {
        "power_w": 300.6,
        "gpu_util_pct": 98,
        "memory_used_mib": 19895,
        "t_s": 564.896
      },
      {
        "power_w": 300.62,
        "gpu_util_pct": 97,
        "memory_used_mib": 19895,
        "t_s": 565.949
      },
      {
        "power_w": 300.45,
        "gpu_util_pct": 98,
        "memory_used_mib": 19895,
        "t_s": 567.0
      },
      {
        "power_w": 300.21,
        "gpu_util_pct": 97,
        "memory_used_mib": 19895,
        "t_s": 568.053
      },
      {
        "power_w": 300.39,
        "gpu_util_pct": 97,
        "memory_used_mib": 19895,
        "t_s": 569.128
      },
      {
        "power_w": 300.35,
        "gpu_util_pct": 97,
        "memory_used_mib": 19895,
        "t_s": 570.181
      },
      {
        "power_w": 300.5,
        "gpu_util_pct": 98,
        "memory_used_mib": 19895,
        "t_s": 571.235
      },
      {
        "power_w": 300.32,
        "gpu_util_pct": 95,
        "memory_used_mib": 19895,
        "t_s": 572.288
      },
      {
        "power_w": 300.58,
        "gpu_util_pct": 97,
        "memory_used_mib": 19895,
        "t_s": 573.338
      },
      {
        "power_w": 300.27,
        "gpu_util_pct": 97,
        "memory_used_mib": 19895,
        "t_s": 574.387
      },
      {
        "power_w": 300.45,
        "gpu_util_pct": 97,
        "memory_used_mib": 19895,
        "t_s": 575.441
      },
      {
        "power_w": 300.09,
        "gpu_util_pct": 98,
        "memory_used_mib": 19895,
        "t_s": 576.492
      },
      {
        "power_w": 300.18,
        "gpu_util_pct": 97,
        "memory_used_mib": 19895,
        "t_s": 577.542
      },
      {
        "power_w": 300.7,
        "gpu_util_pct": 97,
        "memory_used_mib": 19895,
        "t_s": 578.592
      },
      {
        "power_w": 300.54,
        "gpu_util_pct": 97,
        "memory_used_mib": 19895,
        "t_s": 579.636
      },
      {
        "power_w": 300.36,
        "gpu_util_pct": 97,
        "memory_used_mib": 19895,
        "t_s": 580.688
      },
      {
        "power_w": 300.71,
        "gpu_util_pct": 97,
        "memory_used_mib": 19895,
        "t_s": 581.746
      },
      {
        "power_w": 300.41,
        "gpu_util_pct": 98,
        "memory_used_mib": 19895,
        "t_s": 582.795
      },
      {
        "power_w": 300.01,
        "gpu_util_pct": 97,
        "memory_used_mib": 19895,
        "t_s": 583.846
      },
      {
        "power_w": 300.11,
        "gpu_util_pct": 96,
        "memory_used_mib": 19895,
        "t_s": 584.893
      },
      {
        "power_w": 300.67,
        "gpu_util_pct": 98,
        "memory_used_mib": 19895,
        "t_s": 585.945
      },
      {
        "power_w": 300.35,
        "gpu_util_pct": 96,
        "memory_used_mib": 19895,
        "t_s": 586.998
      },
      {
        "power_w": 300.66,
        "gpu_util_pct": 97,
        "memory_used_mib": 19895,
        "t_s": 588.05
      },
      {
        "power_w": 300.44,
        "gpu_util_pct": 97,
        "memory_used_mib": 19895,
        "t_s": 589.1
      },
      {
        "power_w": 300.44,
        "gpu_util_pct": 98,
        "memory_used_mib": 19895,
        "t_s": 590.151
      },
      {
        "power_w": 300.5,
        "gpu_util_pct": 98,
        "memory_used_mib": 19895,
        "t_s": 591.202
      },
      {
        "power_w": 299.99,
        "gpu_util_pct": 98,
        "memory_used_mib": 19895,
        "t_s": 592.255
      },
      {
        "power_w": 300.15,
        "gpu_util_pct": 97,
        "memory_used_mib": 19895,
        "t_s": 593.302
      },
      {
        "power_w": 300.6,
        "gpu_util_pct": 97,
        "memory_used_mib": 19895,
        "t_s": 594.358
      },
      {
        "power_w": 300.36,
        "gpu_util_pct": 98,
        "memory_used_mib": 19895,
        "t_s": 595.408
      },
      {
        "power_w": 300.36,
        "gpu_util_pct": 97,
        "memory_used_mib": 19895,
        "t_s": 596.46
      },
      {
        "power_w": 300.3,
        "gpu_util_pct": 98,
        "memory_used_mib": 19895,
        "t_s": 597.508
      },
      {
        "power_w": 300.61,
        "gpu_util_pct": 98,
        "memory_used_mib": 19895,
        "t_s": 598.561
      },
      {
        "power_w": 300.23,
        "gpu_util_pct": 97,
        "memory_used_mib": 19895,
        "t_s": 599.614
      },
      {
        "power_w": 300.92,
        "gpu_util_pct": 97,
        "memory_used_mib": 19895,
        "t_s": 600.682
      },
      {
        "power_w": 300.31,
        "gpu_util_pct": 98,
        "memory_used_mib": 19895,
        "t_s": 601.735
      },
      {
        "power_w": 300.44,
        "gpu_util_pct": 97,
        "memory_used_mib": 19895,
        "t_s": 602.79
      },
      {
        "power_w": 300.48,
        "gpu_util_pct": 98,
        "memory_used_mib": 19895,
        "t_s": 603.84
      },
      {
        "power_w": 300.39,
        "gpu_util_pct": 98,
        "memory_used_mib": 19895,
        "t_s": 604.889
      },
      {
        "power_w": 300.39,
        "gpu_util_pct": 97,
        "memory_used_mib": 19895,
        "t_s": 605.938
      },
      {
        "power_w": 299.99,
        "gpu_util_pct": 97,
        "memory_used_mib": 19895,
        "t_s": 606.994
      },
      {
        "power_w": 300.34,
        "gpu_util_pct": 97,
        "memory_used_mib": 19895,
        "t_s": 608.043
      },
      {
        "power_w": 300.32,
        "gpu_util_pct": 96,
        "memory_used_mib": 19895,
        "t_s": 609.087
      },
      {
        "power_w": 300.62,
        "gpu_util_pct": 98,
        "memory_used_mib": 19895,
        "t_s": 610.129
      },
      {
        "power_w": 300.31,
        "gpu_util_pct": 98,
        "memory_used_mib": 19895,
        "t_s": 611.172
      },
      {
        "power_w": 300.69,
        "gpu_util_pct": 97,
        "memory_used_mib": 19895,
        "t_s": 612.224
      },
      {
        "power_w": 300.11,
        "gpu_util_pct": 97,
        "memory_used_mib": 19895,
        "t_s": 613.275
      },
      {
        "power_w": 238.7,
        "gpu_util_pct": 0,
        "memory_used_mib": 19895,
        "t_s": 614.325
      },
      {
        "power_w": 298.39,
        "gpu_util_pct": 100,
        "memory_used_mib": 19895,
        "t_s": 615.373
      },
      {
        "power_w": 296.52,
        "gpu_util_pct": 100,
        "memory_used_mib": 19895,
        "t_s": 616.419
      },
      {
        "power_w": 293.79,
        "gpu_util_pct": 80,
        "memory_used_mib": 19895,
        "t_s": 617.468
      },
      {
        "power_w": 299.73,
        "gpu_util_pct": 96,
        "memory_used_mib": 19895,
        "t_s": 618.512
      },
      {
        "power_w": 300.62,
        "gpu_util_pct": 98,
        "memory_used_mib": 19895,
        "t_s": 619.574
      },
      {
        "power_w": 299.94,
        "gpu_util_pct": 97,
        "memory_used_mib": 19895,
        "t_s": 620.623
      },
      {
        "power_w": 300.67,
        "gpu_util_pct": 97,
        "memory_used_mib": 19895,
        "t_s": 621.67
      },
      {
        "power_w": 299.87,
        "gpu_util_pct": 98,
        "memory_used_mib": 19895,
        "t_s": 622.713
      },
      {
        "power_w": 300.0,
        "gpu_util_pct": 97,
        "memory_used_mib": 19895,
        "t_s": 623.766
      },
      {
        "power_w": 300.49,
        "gpu_util_pct": 97,
        "memory_used_mib": 19895,
        "t_s": 624.814
      },
      {
        "power_w": 300.38,
        "gpu_util_pct": 98,
        "memory_used_mib": 19895,
        "t_s": 625.863
      },
      {
        "power_w": 300.64,
        "gpu_util_pct": 97,
        "memory_used_mib": 19895,
        "t_s": 626.917
      },
      {
        "power_w": 300.6,
        "gpu_util_pct": 97,
        "memory_used_mib": 19895,
        "t_s": 627.967
      },
      {
        "power_w": 300.64,
        "gpu_util_pct": 98,
        "memory_used_mib": 19895,
        "t_s": 629.018
      },
      {
        "power_w": 300.45,
        "gpu_util_pct": 97,
        "memory_used_mib": 19895,
        "t_s": 630.074
      },
      {
        "power_w": 300.42,
        "gpu_util_pct": 97,
        "memory_used_mib": 19895,
        "t_s": 631.123
      },
      {
        "power_w": 299.21,
        "gpu_util_pct": 96,
        "memory_used_mib": 19895,
        "t_s": 632.18
      },
      {
        "power_w": 300.22,
        "gpu_util_pct": 97,
        "memory_used_mib": 19895,
        "t_s": 633.231
      },
      {
        "power_w": 300.71,
        "gpu_util_pct": 97,
        "memory_used_mib": 19895,
        "t_s": 634.283
      },
      {
        "power_w": 300.84,
        "gpu_util_pct": 97,
        "memory_used_mib": 19895,
        "t_s": 635.333
      },
      {
        "power_w": 300.24,
        "gpu_util_pct": 97,
        "memory_used_mib": 19895,
        "t_s": 636.385
      },
      {
        "power_w": 300.29,
        "gpu_util_pct": 97,
        "memory_used_mib": 19895,
        "t_s": 637.435
      },
      {
        "power_w": 300.17,
        "gpu_util_pct": 97,
        "memory_used_mib": 19895,
        "t_s": 638.493
      },
      {
        "power_w": 300.52,
        "gpu_util_pct": 97,
        "memory_used_mib": 19895,
        "t_s": 639.543
      },
      {
        "power_w": 300.63,
        "gpu_util_pct": 97,
        "memory_used_mib": 19895,
        "t_s": 640.596
      },
      {
        "power_w": 300.02,
        "gpu_util_pct": 98,
        "memory_used_mib": 19895,
        "t_s": 641.66
      },
      {
        "power_w": 300.41,
        "gpu_util_pct": 97,
        "memory_used_mib": 19895,
        "t_s": 642.711
      },
      {
        "power_w": 300.51,
        "gpu_util_pct": 97,
        "memory_used_mib": 19895,
        "t_s": 643.763
      },
      {
        "power_w": 300.41,
        "gpu_util_pct": 98,
        "memory_used_mib": 19895,
        "t_s": 644.816
      },
      {
        "power_w": 300.41,
        "gpu_util_pct": 97,
        "memory_used_mib": 19895,
        "t_s": 645.863
      },
      {
        "power_w": 300.46,
        "gpu_util_pct": 97,
        "memory_used_mib": 19895,
        "t_s": 646.916
      },
      {
        "power_w": 300.29,
        "gpu_util_pct": 98,
        "memory_used_mib": 19895,
        "t_s": 647.963
      },
      {
        "power_w": 300.48,
        "gpu_util_pct": 97,
        "memory_used_mib": 19895,
        "t_s": 649.013
      },
      {
        "power_w": 300.21,
        "gpu_util_pct": 94,
        "memory_used_mib": 19895,
        "t_s": 650.06
      },
      {
        "power_w": 300.46,
        "gpu_util_pct": 97,
        "memory_used_mib": 19895,
        "t_s": 651.114
      },
      {
        "power_w": 300.5,
        "gpu_util_pct": 97,
        "memory_used_mib": 19895,
        "t_s": 652.158
      },
      {
        "power_w": 300.45,
        "gpu_util_pct": 97,
        "memory_used_mib": 19895,
        "t_s": 653.208
      },
      {
        "power_w": 300.17,
        "gpu_util_pct": 98,
        "memory_used_mib": 19895,
        "t_s": 654.254
      },
      {
        "power_w": 300.03,
        "gpu_util_pct": 97,
        "memory_used_mib": 19895,
        "t_s": 655.305
      },
      {
        "power_w": 300.58,
        "gpu_util_pct": 97,
        "memory_used_mib": 19895,
        "t_s": 656.357
      },
      {
        "power_w": 300.68,
        "gpu_util_pct": 98,
        "memory_used_mib": 19895,
        "t_s": 657.409
      },
      {
        "power_w": 300.35,
        "gpu_util_pct": 97,
        "memory_used_mib": 19895,
        "t_s": 658.465
      },
      {
        "power_w": 300.18,
        "gpu_util_pct": 98,
        "memory_used_mib": 19895,
        "t_s": 659.527
      },
      {
        "power_w": 300.32,
        "gpu_util_pct": 98,
        "memory_used_mib": 19895,
        "t_s": 660.582
      },
      {
        "power_w": 300.11,
        "gpu_util_pct": 97,
        "memory_used_mib": 19895,
        "t_s": 661.629
      },
      {
        "power_w": 300.58,
        "gpu_util_pct": 97,
        "memory_used_mib": 19895,
        "t_s": 662.683
      },
      {
        "power_w": 239.68,
        "gpu_util_pct": 44,
        "memory_used_mib": 19895,
        "t_s": 663.736
      },
      {
        "power_w": 297.51,
        "gpu_util_pct": 100,
        "memory_used_mib": 19895,
        "t_s": 664.792
      },
      {
        "power_w": 299.12,
        "gpu_util_pct": 99,
        "memory_used_mib": 19897,
        "t_s": 665.853
      },
      {
        "power_w": 297.25,
        "gpu_util_pct": 100,
        "memory_used_mib": 19899,
        "t_s": 666.904
      },
      {
        "power_w": 299.48,
        "gpu_util_pct": 97,
        "memory_used_mib": 19899,
        "t_s": 667.957
      },
      {
        "power_w": 299.61,
        "gpu_util_pct": 96,
        "memory_used_mib": 19899,
        "t_s": 669.016
      },
      {
        "power_w": 300.42,
        "gpu_util_pct": 97,
        "memory_used_mib": 19899,
        "t_s": 670.078
      },
      {
        "power_w": 300.07,
        "gpu_util_pct": 97,
        "memory_used_mib": 19899,
        "t_s": 671.134
      },
      {
        "power_w": 300.45,
        "gpu_util_pct": 97,
        "memory_used_mib": 19899,
        "t_s": 672.186
      },
      {
        "power_w": 300.08,
        "gpu_util_pct": 97,
        "memory_used_mib": 19899,
        "t_s": 673.238
      },
      {
        "power_w": 300.58,
        "gpu_util_pct": 97,
        "memory_used_mib": 19899,
        "t_s": 674.294
      },
      {
        "power_w": 300.46,
        "gpu_util_pct": 97,
        "memory_used_mib": 19899,
        "t_s": 675.348
      },
      {
        "power_w": 300.7,
        "gpu_util_pct": 98,
        "memory_used_mib": 19899,
        "t_s": 676.405
      },
      {
        "power_w": 300.18,
        "gpu_util_pct": 97,
        "memory_used_mib": 19899,
        "t_s": 677.458
      },
      {
        "power_w": 300.26,
        "gpu_util_pct": 98,
        "memory_used_mib": 19899,
        "t_s": 678.518
      },
      {
        "power_w": 300.51,
        "gpu_util_pct": 97,
        "memory_used_mib": 19899,
        "t_s": 679.568
      },
      {
        "power_w": 300.1,
        "gpu_util_pct": 97,
        "memory_used_mib": 19899,
        "t_s": 680.616
      },
      {
        "power_w": 300.36,
        "gpu_util_pct": 97,
        "memory_used_mib": 19899,
        "t_s": 681.669
      },
      {
        "power_w": 300.2,
        "gpu_util_pct": 97,
        "memory_used_mib": 19899,
        "t_s": 682.728
      },
      {
        "power_w": 300.11,
        "gpu_util_pct": 97,
        "memory_used_mib": 19899,
        "t_s": 683.782
      },
      {
        "power_w": 299.87,
        "gpu_util_pct": 95,
        "memory_used_mib": 19899,
        "t_s": 684.836
      },
      {
        "power_w": 300.51,
        "gpu_util_pct": 97,
        "memory_used_mib": 19899,
        "t_s": 685.897
      },
      {
        "power_w": 300.46,
        "gpu_util_pct": 97,
        "memory_used_mib": 19899,
        "t_s": 686.951
      },
      {
        "power_w": 300.28,
        "gpu_util_pct": 98,
        "memory_used_mib": 19899,
        "t_s": 688.001
      },
      {
        "power_w": 300.13,
        "gpu_util_pct": 98,
        "memory_used_mib": 19899,
        "t_s": 689.052
      },
      {
        "power_w": 300.18,
        "gpu_util_pct": 97,
        "memory_used_mib": 19899,
        "t_s": 690.111
      },
      {
        "power_w": 300.38,
        "gpu_util_pct": 97,
        "memory_used_mib": 19899,
        "t_s": 691.162
      },
      {
        "power_w": 300.38,
        "gpu_util_pct": 98,
        "memory_used_mib": 19899,
        "t_s": 692.217
      },
      {
        "power_w": 300.24,
        "gpu_util_pct": 96,
        "memory_used_mib": 19899,
        "t_s": 693.271
      },
      {
        "power_w": 300.29,
        "gpu_util_pct": 97,
        "memory_used_mib": 19899,
        "t_s": 694.323
      },
      {
        "power_w": 300.15,
        "gpu_util_pct": 97,
        "memory_used_mib": 19899,
        "t_s": 695.375
      },
      {
        "power_w": 300.31,
        "gpu_util_pct": 97,
        "memory_used_mib": 19899,
        "t_s": 696.435
      },
      {
        "power_w": 300.69,
        "gpu_util_pct": 98,
        "memory_used_mib": 19899,
        "t_s": 697.492
      },
      {
        "power_w": 300.12,
        "gpu_util_pct": 97,
        "memory_used_mib": 19899,
        "t_s": 698.544
      },
      {
        "power_w": 300.21,
        "gpu_util_pct": 97,
        "memory_used_mib": 19899,
        "t_s": 699.602
      },
      {
        "power_w": 300.61,
        "gpu_util_pct": 97,
        "memory_used_mib": 19899,
        "t_s": 700.655
      },
      {
        "power_w": 300.59,
        "gpu_util_pct": 97,
        "memory_used_mib": 19899,
        "t_s": 701.71
      },
      {
        "power_w": 300.2,
        "gpu_util_pct": 97,
        "memory_used_mib": 19899,
        "t_s": 702.763
      },
      {
        "power_w": 300.24,
        "gpu_util_pct": 97,
        "memory_used_mib": 19899,
        "t_s": 703.818
      },
      {
        "power_w": 300.08,
        "gpu_util_pct": 98,
        "memory_used_mib": 19899,
        "t_s": 704.871
      },
      {
        "power_w": 300.33,
        "gpu_util_pct": 97,
        "memory_used_mib": 19899,
        "t_s": 705.933
      },
      {
        "power_w": 300.21,
        "gpu_util_pct": 97,
        "memory_used_mib": 19899,
        "t_s": 706.994
      },
      {
        "power_w": 300.49,
        "gpu_util_pct": 97,
        "memory_used_mib": 19899,
        "t_s": 708.049
      },
      {
        "power_w": 300.61,
        "gpu_util_pct": 98,
        "memory_used_mib": 19899,
        "t_s": 709.102
      },
      {
        "power_w": 300.25,
        "gpu_util_pct": 97,
        "memory_used_mib": 19899,
        "t_s": 710.156
      },
      {
        "power_w": 300.6,
        "gpu_util_pct": 98,
        "memory_used_mib": 19899,
        "t_s": 711.204
      },
      {
        "power_w": 229.05,
        "gpu_util_pct": 100,
        "memory_used_mib": 19899,
        "t_s": 712.253
      },
      {
        "power_w": 297.43,
        "gpu_util_pct": 100,
        "memory_used_mib": 19899,
        "t_s": 713.307
      },
      {
        "power_w": 296.57,
        "gpu_util_pct": 100,
        "memory_used_mib": 19899,
        "t_s": 714.358
      },
      {
        "power_w": 298.11,
        "gpu_util_pct": 92,
        "memory_used_mib": 19899,
        "t_s": 715.406
      },
      {
        "power_w": 300.69,
        "gpu_util_pct": 97,
        "memory_used_mib": 19899,
        "t_s": 716.455
      },
      {
        "power_w": 300.31,
        "gpu_util_pct": 97,
        "memory_used_mib": 19899,
        "t_s": 717.502
      },
      {
        "power_w": 299.98,
        "gpu_util_pct": 97,
        "memory_used_mib": 19899,
        "t_s": 718.554
      },
      {
        "power_w": 300.58,
        "gpu_util_pct": 98,
        "memory_used_mib": 19899,
        "t_s": 719.606
      },
      {
        "power_w": 300.62,
        "gpu_util_pct": 98,
        "memory_used_mib": 19899,
        "t_s": 720.655
      },
      {
        "power_w": 233.62,
        "gpu_util_pct": 100,
        "memory_used_mib": 19899,
        "t_s": 721.704
      },
      {
        "power_w": 300.05,
        "gpu_util_pct": 98,
        "memory_used_mib": 19899,
        "t_s": 722.755
      },
      {
        "power_w": 299.61,
        "gpu_util_pct": 98,
        "memory_used_mib": 19899,
        "t_s": 723.807
      },
      {
        "power_w": 299.83,
        "gpu_util_pct": 97,
        "memory_used_mib": 19899,
        "t_s": 724.854
      },
      {
        "power_w": 300.29,
        "gpu_util_pct": 98,
        "memory_used_mib": 19899,
        "t_s": 725.911
      },
      {
        "power_w": 300.75,
        "gpu_util_pct": 97,
        "memory_used_mib": 19899,
        "t_s": 726.96
      },
      {
        "power_w": 300.81,
        "gpu_util_pct": 98,
        "memory_used_mib": 19899,
        "t_s": 728.012
      },
      {
        "power_w": 300.64,
        "gpu_util_pct": 98,
        "memory_used_mib": 19899,
        "t_s": 729.061
      },
      {
        "power_w": 299.99,
        "gpu_util_pct": 97,
        "memory_used_mib": 19899,
        "t_s": 730.112
      },
      {
        "power_w": 299.45,
        "gpu_util_pct": 98,
        "memory_used_mib": 19899,
        "t_s": 731.193
      },
      {
        "power_w": 300.47,
        "gpu_util_pct": 98,
        "memory_used_mib": 19899,
        "t_s": 732.244
      },
      {
        "power_w": 300.26,
        "gpu_util_pct": 98,
        "memory_used_mib": 19899,
        "t_s": 733.295
      },
      {
        "power_w": 241.14,
        "gpu_util_pct": 65,
        "memory_used_mib": 19899,
        "t_s": 734.346
      },
      {
        "power_w": 299.53,
        "gpu_util_pct": 90,
        "memory_used_mib": 19899,
        "t_s": 735.397
      },
      {
        "power_w": 299.98,
        "gpu_util_pct": 97,
        "memory_used_mib": 19899,
        "t_s": 736.451
      },
      {
        "power_w": 282.82,
        "gpu_util_pct": 95,
        "memory_used_mib": 19899,
        "t_s": 737.51
      },
      {
        "power_w": 244.03,
        "gpu_util_pct": 96,
        "memory_used_mib": 19899,
        "t_s": 738.563
      },
      {
        "power_w": 300.14,
        "gpu_util_pct": 98,
        "memory_used_mib": 19899,
        "t_s": 739.619
      },
      {
        "power_w": 299.63,
        "gpu_util_pct": 97,
        "memory_used_mib": 19899,
        "t_s": 740.679
      },
      {
        "power_w": 300.07,
        "gpu_util_pct": 98,
        "memory_used_mib": 19899,
        "t_s": 741.739
      },
      {
        "power_w": 299.76,
        "gpu_util_pct": 97,
        "memory_used_mib": 19899,
        "t_s": 742.795
      },
      {
        "power_w": 300.37,
        "gpu_util_pct": 98,
        "memory_used_mib": 19899,
        "t_s": 743.848
      },
      {
        "power_w": 300.54,
        "gpu_util_pct": 98,
        "memory_used_mib": 19899,
        "t_s": 744.933
      },
      {
        "power_w": 300.84,
        "gpu_util_pct": 98,
        "memory_used_mib": 19899,
        "t_s": 745.982
      },
      {
        "power_w": 300.23,
        "gpu_util_pct": 98,
        "memory_used_mib": 19899,
        "t_s": 747.034
      },
      {
        "power_w": 300.14,
        "gpu_util_pct": 96,
        "memory_used_mib": 19899,
        "t_s": 748.084
      },
      {
        "power_w": 300.2,
        "gpu_util_pct": 97,
        "memory_used_mib": 19899,
        "t_s": 749.133
      },
      {
        "power_w": 300.3,
        "gpu_util_pct": 98,
        "memory_used_mib": 19899,
        "t_s": 750.194
      },
      {
        "power_w": 299.73,
        "gpu_util_pct": 97,
        "memory_used_mib": 19899,
        "t_s": 751.245
      },
      {
        "power_w": 300.2,
        "gpu_util_pct": 98,
        "memory_used_mib": 19905,
        "t_s": 752.305
      },
      {
        "power_w": 249.51,
        "gpu_util_pct": 0,
        "memory_used_mib": 19905,
        "t_s": 753.359
      },
      {
        "power_w": 298.82,
        "gpu_util_pct": 97,
        "memory_used_mib": 19905,
        "t_s": 754.415
      },
      {
        "power_w": 300.13,
        "gpu_util_pct": 97,
        "memory_used_mib": 19907,
        "t_s": 755.466
      },
      {
        "power_w": 300.84,
        "gpu_util_pct": 97,
        "memory_used_mib": 19920,
        "t_s": 756.523
      },
      {
        "power_w": 300.31,
        "gpu_util_pct": 97,
        "memory_used_mib": 19921,
        "t_s": 757.568
      },
      {
        "power_w": 300.46,
        "gpu_util_pct": 97,
        "memory_used_mib": 19914,
        "t_s": 758.623
      },
      {
        "power_w": 300.5,
        "gpu_util_pct": 97,
        "memory_used_mib": 19914,
        "t_s": 759.677
      },
      {
        "power_w": 300.4,
        "gpu_util_pct": 97,
        "memory_used_mib": 19914,
        "t_s": 760.73
      },
      {
        "power_w": 300.34,
        "gpu_util_pct": 96,
        "memory_used_mib": 19914,
        "t_s": 761.777
      },
      {
        "power_w": 300.32,
        "gpu_util_pct": 98,
        "memory_used_mib": 19913,
        "t_s": 762.833
      },
      {
        "power_w": 300.48,
        "gpu_util_pct": 97,
        "memory_used_mib": 19913,
        "t_s": 763.889
      },
      {
        "power_w": 300.51,
        "gpu_util_pct": 95,
        "memory_used_mib": 19912,
        "t_s": 764.944
      },
      {
        "power_w": 300.38,
        "gpu_util_pct": 97,
        "memory_used_mib": 19924,
        "t_s": 766.0
      },
      {
        "power_w": 300.23,
        "gpu_util_pct": 97,
        "memory_used_mib": 19922,
        "t_s": 767.055
      },
      {
        "power_w": 300.24,
        "gpu_util_pct": 97,
        "memory_used_mib": 19922,
        "t_s": 768.112
      },
      {
        "power_w": 300.4,
        "gpu_util_pct": 97,
        "memory_used_mib": 19923,
        "t_s": 769.169
      },
      {
        "power_w": 300.55,
        "gpu_util_pct": 98,
        "memory_used_mib": 19923,
        "t_s": 770.226
      },
      {
        "power_w": 300.01,
        "gpu_util_pct": 98,
        "memory_used_mib": 19922,
        "t_s": 771.286
      },
      {
        "power_w": 254.31,
        "gpu_util_pct": 0,
        "memory_used_mib": 19921,
        "t_s": 772.347
      },
      {
        "power_w": 293.66,
        "gpu_util_pct": 100,
        "memory_used_mib": 19923,
        "t_s": 773.409
      },
      {
        "power_w": 298.42,
        "gpu_util_pct": 100,
        "memory_used_mib": 19935,
        "t_s": 774.457
      },
      {
        "power_w": 298.49,
        "gpu_util_pct": 100,
        "memory_used_mib": 19933,
        "t_s": 775.537
      },
      {
        "power_w": 296.48,
        "gpu_util_pct": 100,
        "memory_used_mib": 19931,
        "t_s": 776.588
      },
      {
        "power_w": 297.61,
        "gpu_util_pct": 100,
        "memory_used_mib": 19917,
        "t_s": 777.641
      },
      {
        "power_w": 296.7,
        "gpu_util_pct": 100,
        "memory_used_mib": 19917,
        "t_s": 778.693
      },
      {
        "power_w": 298.19,
        "gpu_util_pct": 100,
        "memory_used_mib": 19925,
        "t_s": 779.746
      },
      {
        "power_w": 296.58,
        "gpu_util_pct": 100,
        "memory_used_mib": 19923,
        "t_s": 780.8
      },
      {
        "power_w": 296.79,
        "gpu_util_pct": 100,
        "memory_used_mib": 19911,
        "t_s": 781.864
      },
      {
        "power_w": 296.55,
        "gpu_util_pct": 100,
        "memory_used_mib": 19917,
        "t_s": 782.917
      },
      {
        "power_w": 297.15,
        "gpu_util_pct": 100,
        "memory_used_mib": 19917,
        "t_s": 783.972
      },
      {
        "power_w": 298.16,
        "gpu_util_pct": 100,
        "memory_used_mib": 19917,
        "t_s": 785.022
      },
      {
        "power_w": 299.7,
        "gpu_util_pct": 97,
        "memory_used_mib": 19913,
        "t_s": 786.072
      },
      {
        "power_w": 300.47,
        "gpu_util_pct": 97,
        "memory_used_mib": 19912,
        "t_s": 787.124
      },
      {
        "power_w": 300.57,
        "gpu_util_pct": 98,
        "memory_used_mib": 19911,
        "t_s": 788.177
      },
      {
        "power_w": 300.08,
        "gpu_util_pct": 97,
        "memory_used_mib": 19909,
        "t_s": 789.227
      },
      {
        "power_w": 300.33,
        "gpu_util_pct": 97,
        "memory_used_mib": 19909,
        "t_s": 790.279
      },
      {
        "power_w": 299.97,
        "gpu_util_pct": 95,
        "memory_used_mib": 19909,
        "t_s": 791.338
      },
      {
        "power_w": 300.82,
        "gpu_util_pct": 97,
        "memory_used_mib": 19909,
        "t_s": 792.394
      },
      {
        "power_w": 300.42,
        "gpu_util_pct": 97,
        "memory_used_mib": 19909,
        "t_s": 793.446
      },
      {
        "power_w": 300.2,
        "gpu_util_pct": 97,
        "memory_used_mib": 19909,
        "t_s": 794.499
      },
      {
        "power_w": 300.69,
        "gpu_util_pct": 98,
        "memory_used_mib": 19909,
        "t_s": 795.55
      },
      {
        "power_w": 300.31,
        "gpu_util_pct": 97,
        "memory_used_mib": 19909,
        "t_s": 796.601
      },
      {
        "power_w": 300.17,
        "gpu_util_pct": 97,
        "memory_used_mib": 19905,
        "t_s": 797.656
      },
      {
        "power_w": 300.3,
        "gpu_util_pct": 97,
        "memory_used_mib": 19905,
        "t_s": 798.71
      },
      {
        "power_w": 228.57,
        "gpu_util_pct": 15,
        "memory_used_mib": 19905,
        "t_s": 799.765
      },
      {
        "power_w": 297.82,
        "gpu_util_pct": 100,
        "memory_used_mib": 19905,
        "t_s": 800.822
      },
      {
        "power_w": 296.43,
        "gpu_util_pct": 100,
        "memory_used_mib": 19905,
        "t_s": 801.877
      },
      {
        "power_w": 295.52,
        "gpu_util_pct": 100,
        "memory_used_mib": 19919,
        "t_s": 802.93
      },
      {
        "power_w": 299.89,
        "gpu_util_pct": 95,
        "memory_used_mib": 19919,
        "t_s": 803.989
      },
      {
        "power_w": 292.86,
        "gpu_util_pct": 100,
        "memory_used_mib": 19919,
        "t_s": 805.043
      },
      {
        "power_w": 298.18,
        "gpu_util_pct": 97,
        "memory_used_mib": 19917,
        "t_s": 806.099
      },
      {
        "power_w": 300.37,
        "gpu_util_pct": 97,
        "memory_used_mib": 19913,
        "t_s": 807.147
      },
      {
        "power_w": 300.72,
        "gpu_util_pct": 97,
        "memory_used_mib": 19913,
        "t_s": 808.19
      },
      {
        "power_w": 300.37,
        "gpu_util_pct": 97,
        "memory_used_mib": 19911,
        "t_s": 809.241
      },
      {
        "power_w": 299.73,
        "gpu_util_pct": 97,
        "memory_used_mib": 19909,
        "t_s": 810.339
      },
      {
        "power_w": 299.67,
        "gpu_util_pct": 97,
        "memory_used_mib": 19916,
        "t_s": 811.391
      },
      {
        "power_w": 300.67,
        "gpu_util_pct": 97,
        "memory_used_mib": 19917,
        "t_s": 812.446
      },
      {
        "power_w": 300.01,
        "gpu_util_pct": 97,
        "memory_used_mib": 19917,
        "t_s": 813.498
      },
      {
        "power_w": 300.3,
        "gpu_util_pct": 97,
        "memory_used_mib": 19917,
        "t_s": 814.549
      },
      {
        "power_w": 300.51,
        "gpu_util_pct": 97,
        "memory_used_mib": 19913,
        "t_s": 815.597
      },
      {
        "power_w": 300.66,
        "gpu_util_pct": 97,
        "memory_used_mib": 19913,
        "t_s": 816.643
      },
      {
        "power_w": 300.04,
        "gpu_util_pct": 96,
        "memory_used_mib": 19912,
        "t_s": 817.689
      },
      {
        "power_w": 300.23,
        "gpu_util_pct": 97,
        "memory_used_mib": 19909,
        "t_s": 818.739
      },
      {
        "power_w": 299.47,
        "gpu_util_pct": 96,
        "memory_used_mib": 19909,
        "t_s": 819.78
      },
      {
        "power_w": 299.27,
        "gpu_util_pct": 98,
        "memory_used_mib": 19909,
        "t_s": 820.828
      },
      {
        "power_w": 300.2,
        "gpu_util_pct": 98,
        "memory_used_mib": 19909,
        "t_s": 821.887
      },
      {
        "power_w": 300.32,
        "gpu_util_pct": 97,
        "memory_used_mib": 19909,
        "t_s": 822.935
      },
      {
        "power_w": 300.34,
        "gpu_util_pct": 97,
        "memory_used_mib": 19909,
        "t_s": 823.998
      },
      {
        "power_w": 300.56,
        "gpu_util_pct": 97,
        "memory_used_mib": 19909,
        "t_s": 825.058
      },
      {
        "power_w": 300.76,
        "gpu_util_pct": 98,
        "memory_used_mib": 19909,
        "t_s": 826.11
      },
      {
        "power_w": 300.3,
        "gpu_util_pct": 98,
        "memory_used_mib": 19908,
        "t_s": 827.162
      },
      {
        "power_w": 299.98,
        "gpu_util_pct": 98,
        "memory_used_mib": 19908,
        "t_s": 828.211
      },
      {
        "power_w": 300.56,
        "gpu_util_pct": 97,
        "memory_used_mib": 19917,
        "t_s": 829.26
      },
      {
        "power_w": 300.4,
        "gpu_util_pct": 94,
        "memory_used_mib": 19917,
        "t_s": 830.31
      },
      {
        "power_w": 300.5,
        "gpu_util_pct": 97,
        "memory_used_mib": 19917,
        "t_s": 831.367
      },
      {
        "power_w": 300.44,
        "gpu_util_pct": 97,
        "memory_used_mib": 19911,
        "t_s": 832.422
      },
      {
        "power_w": 300.15,
        "gpu_util_pct": 97,
        "memory_used_mib": 19911,
        "t_s": 833.473
      },
      {
        "power_w": 300.37,
        "gpu_util_pct": 98,
        "memory_used_mib": 19911,
        "t_s": 834.52
      },
      {
        "power_w": 300.16,
        "gpu_util_pct": 98,
        "memory_used_mib": 19911,
        "t_s": 835.575
      },
      {
        "power_w": 300.35,
        "gpu_util_pct": 98,
        "memory_used_mib": 19919,
        "t_s": 836.625
      },
      {
        "power_w": 300.04,
        "gpu_util_pct": 98,
        "memory_used_mib": 19911,
        "t_s": 837.681
      },
      {
        "power_w": 300.2,
        "gpu_util_pct": 97,
        "memory_used_mib": 19917,
        "t_s": 838.736
      },
      {
        "power_w": 300.38,
        "gpu_util_pct": 97,
        "memory_used_mib": 19916,
        "t_s": 839.789
      },
      {
        "power_w": 299.65,
        "gpu_util_pct": 96,
        "memory_used_mib": 19908,
        "t_s": 840.841
      },
      {
        "power_w": 300.13,
        "gpu_util_pct": 97,
        "memory_used_mib": 19908,
        "t_s": 841.895
      },
      {
        "power_w": 300.24,
        "gpu_util_pct": 98,
        "memory_used_mib": 19907,
        "t_s": 842.949
      },
      {
        "power_w": 300.21,
        "gpu_util_pct": 97,
        "memory_used_mib": 19906,
        "t_s": 844.006
      },
      {
        "power_w": 299.98,
        "gpu_util_pct": 97,
        "memory_used_mib": 19913,
        "t_s": 845.057
      },
      {
        "power_w": 300.61,
        "gpu_util_pct": 97,
        "memory_used_mib": 19917,
        "t_s": 846.108
      },
      {
        "power_w": 300.36,
        "gpu_util_pct": 96,
        "memory_used_mib": 19915,
        "t_s": 847.159
      },
      {
        "power_w": 300.28,
        "gpu_util_pct": 97,
        "memory_used_mib": 19916,
        "t_s": 848.204
      },
      {
        "power_w": 300.16,
        "gpu_util_pct": 96,
        "memory_used_mib": 19918,
        "t_s": 849.254
      },
      {
        "power_w": 299.21,
        "gpu_util_pct": 97,
        "memory_used_mib": 19936,
        "t_s": 850.335
      },
      {
        "power_w": 300.22,
        "gpu_util_pct": 96,
        "memory_used_mib": 19994,
        "t_s": 851.417
      },
      {
        "power_w": 299.47,
        "gpu_util_pct": 95,
        "memory_used_mib": 20007,
        "t_s": 852.493
      },
      {
        "power_w": 300.49,
        "gpu_util_pct": 97,
        "memory_used_mib": 20002,
        "t_s": 853.619
      },
      {
        "power_w": 299.64,
        "gpu_util_pct": 97,
        "memory_used_mib": 19989,
        "t_s": 854.67
      },
      {
        "power_w": 300.37,
        "gpu_util_pct": 97,
        "memory_used_mib": 19991,
        "t_s": 855.734
      },
      {
        "power_w": 300.4,
        "gpu_util_pct": 97,
        "memory_used_mib": 19991,
        "t_s": 856.818
      },
      {
        "power_w": 300.38,
        "gpu_util_pct": 98,
        "memory_used_mib": 19990,
        "t_s": 857.872
      },
      {
        "power_w": 300.35,
        "gpu_util_pct": 98,
        "memory_used_mib": 19986,
        "t_s": 858.926
      },
      {
        "power_w": 300.0,
        "gpu_util_pct": 97,
        "memory_used_mib": 19986,
        "t_s": 859.985
      },
      {
        "power_w": 300.13,
        "gpu_util_pct": 97,
        "memory_used_mib": 19977,
        "t_s": 861.039
      },
      {
        "power_w": 301.04,
        "gpu_util_pct": 97,
        "memory_used_mib": 19974,
        "t_s": 862.096
      },
      {
        "power_w": 300.34,
        "gpu_util_pct": 97,
        "memory_used_mib": 19972,
        "t_s": 863.15
      },
      {
        "power_w": 300.29,
        "gpu_util_pct": 98,
        "memory_used_mib": 19987,
        "t_s": 864.208
      },
      {
        "power_w": 300.18,
        "gpu_util_pct": 97,
        "memory_used_mib": 19986,
        "t_s": 865.261
      },
      {
        "power_w": 300.05,
        "gpu_util_pct": 97,
        "memory_used_mib": 19984,
        "t_s": 866.319
      },
      {
        "power_w": 300.53,
        "gpu_util_pct": 98,
        "memory_used_mib": 19980,
        "t_s": 867.366
      },
      {
        "power_w": 300.54,
        "gpu_util_pct": 97,
        "memory_used_mib": 19980,
        "t_s": 868.418
      },
      {
        "power_w": 300.05,
        "gpu_util_pct": 97,
        "memory_used_mib": 19983,
        "t_s": 869.492
      },
      {
        "power_w": 300.38,
        "gpu_util_pct": 98,
        "memory_used_mib": 19985,
        "t_s": 870.545
      },
      {
        "power_w": 299.87,
        "gpu_util_pct": 98,
        "memory_used_mib": 19986,
        "t_s": 871.595
      },
      {
        "power_w": 300.57,
        "gpu_util_pct": 98,
        "memory_used_mib": 19979,
        "t_s": 872.648
      },
      {
        "power_w": 300.35,
        "gpu_util_pct": 98,
        "memory_used_mib": 20002,
        "t_s": 873.698
      },
      {
        "power_w": 300.22,
        "gpu_util_pct": 97,
        "memory_used_mib": 20009,
        "t_s": 874.758
      },
      {
        "power_w": 300.49,
        "gpu_util_pct": 98,
        "memory_used_mib": 20003,
        "t_s": 875.806
      },
      {
        "power_w": 300.34,
        "gpu_util_pct": 98,
        "memory_used_mib": 19999,
        "t_s": 876.854
      },
      {
        "power_w": 300.4,
        "gpu_util_pct": 97,
        "memory_used_mib": 19994,
        "t_s": 877.902
      },
      {
        "power_w": 300.29,
        "gpu_util_pct": 97,
        "memory_used_mib": 19983,
        "t_s": 878.955
      },
      {
        "power_w": 299.98,
        "gpu_util_pct": 97,
        "memory_used_mib": 19981,
        "t_s": 880.003
      },
      {
        "power_w": 300.24,
        "gpu_util_pct": 97,
        "memory_used_mib": 19989,
        "t_s": 881.055
      },
      {
        "power_w": 300.04,
        "gpu_util_pct": 98,
        "memory_used_mib": 19976,
        "t_s": 882.119
      },
      {
        "power_w": 300.07,
        "gpu_util_pct": 97,
        "memory_used_mib": 19980,
        "t_s": 883.171
      },
      {
        "power_w": 300.25,
        "gpu_util_pct": 98,
        "memory_used_mib": 19980,
        "t_s": 884.223
      },
      {
        "power_w": 300.67,
        "gpu_util_pct": 97,
        "memory_used_mib": 19977,
        "t_s": 885.272
      },
      {
        "power_w": 289.33,
        "gpu_util_pct": 62,
        "memory_used_mib": 19984,
        "t_s": 886.322
      },
      {
        "power_w": 247.66,
        "gpu_util_pct": 91,
        "memory_used_mib": 19987,
        "t_s": 887.381
      },
      {
        "power_w": 294.68,
        "gpu_util_pct": 100,
        "memory_used_mib": 19988,
        "t_s": 888.434
      },
      {
        "power_w": 298.55,
        "gpu_util_pct": 98,
        "memory_used_mib": 20004,
        "t_s": 889.488
      },
      {
        "power_w": 300.3,
        "gpu_util_pct": 98,
        "memory_used_mib": 20001,
        "t_s": 890.542
      },
      {
        "power_w": 300.01,
        "gpu_util_pct": 95,
        "memory_used_mib": 19993,
        "t_s": 891.595
      },
      {
        "power_w": 301.15,
        "gpu_util_pct": 98,
        "memory_used_mib": 19981,
        "t_s": 892.659
      },
      {
        "power_w": 300.79,
        "gpu_util_pct": 97,
        "memory_used_mib": 19974,
        "t_s": 893.764
      },
      {
        "power_w": 299.9,
        "gpu_util_pct": 97,
        "memory_used_mib": 19972,
        "t_s": 894.817
      },
      {
        "power_w": 299.74,
        "gpu_util_pct": 97,
        "memory_used_mib": 19968,
        "t_s": 895.872
      },
      {
        "power_w": 300.0,
        "gpu_util_pct": 97,
        "memory_used_mib": 19968,
        "t_s": 896.928
      },
      {
        "power_w": 300.41,
        "gpu_util_pct": 97,
        "memory_used_mib": 19968,
        "t_s": 897.973
      },
      {
        "power_w": 300.22,
        "gpu_util_pct": 97,
        "memory_used_mib": 19966,
        "t_s": 899.027
      },
      {
        "power_w": 300.38,
        "gpu_util_pct": 97,
        "memory_used_mib": 19966,
        "t_s": 900.091
      },
      {
        "power_w": 263.32,
        "gpu_util_pct": 18,
        "memory_used_mib": 19962,
        "t_s": 901.154
      },
      {
        "power_w": 298.41,
        "gpu_util_pct": 94,
        "memory_used_mib": 19962,
        "t_s": 902.208
      },
      {
        "power_w": 300.67,
        "gpu_util_pct": 97,
        "memory_used_mib": 19962,
        "t_s": 903.263
      },
      {
        "power_w": 300.49,
        "gpu_util_pct": 97,
        "memory_used_mib": 19962,
        "t_s": 904.318
      },
      {
        "power_w": 299.95,
        "gpu_util_pct": 97,
        "memory_used_mib": 19962,
        "t_s": 905.374
      },
      {
        "power_w": 300.06,
        "gpu_util_pct": 97,
        "memory_used_mib": 19962,
        "t_s": 906.448
      },
      {
        "power_w": 300.13,
        "gpu_util_pct": 98,
        "memory_used_mib": 19971,
        "t_s": 907.5
      },
      {
        "power_w": 300.63,
        "gpu_util_pct": 97,
        "memory_used_mib": 19971,
        "t_s": 908.555
      },
      {
        "power_w": 300.13,
        "gpu_util_pct": 97,
        "memory_used_mib": 19971,
        "t_s": 909.61
      },
      {
        "power_w": 300.32,
        "gpu_util_pct": 97,
        "memory_used_mib": 19969,
        "t_s": 910.67
      },
      {
        "power_w": 299.96,
        "gpu_util_pct": 97,
        "memory_used_mib": 19973,
        "t_s": 911.723
      },
      {
        "power_w": 300.53,
        "gpu_util_pct": 98,
        "memory_used_mib": 19973,
        "t_s": 912.785
      },
      {
        "power_w": 300.26,
        "gpu_util_pct": 97,
        "memory_used_mib": 19974,
        "t_s": 913.84
      },
      {
        "power_w": 299.78,
        "gpu_util_pct": 98,
        "memory_used_mib": 19967,
        "t_s": 914.896
      },
      {
        "power_w": 300.39,
        "gpu_util_pct": 97,
        "memory_used_mib": 19966,
        "t_s": 915.949
      },
      {
        "power_w": 300.11,
        "gpu_util_pct": 97,
        "memory_used_mib": 19963,
        "t_s": 917.003
      },
      {
        "power_w": 300.19,
        "gpu_util_pct": 96,
        "memory_used_mib": 19968,
        "t_s": 918.052
      },
      {
        "power_w": 300.16,
        "gpu_util_pct": 97,
        "memory_used_mib": 19984,
        "t_s": 919.116
      },
      {
        "power_w": 300.19,
        "gpu_util_pct": 98,
        "memory_used_mib": 19986,
        "t_s": 920.199
      },
      {
        "power_w": 300.6,
        "gpu_util_pct": 97,
        "memory_used_mib": 20004,
        "t_s": 921.256
      },
      {
        "power_w": 300.18,
        "gpu_util_pct": 97,
        "memory_used_mib": 20005,
        "t_s": 922.306
      },
      {
        "power_w": 300.33,
        "gpu_util_pct": 97,
        "memory_used_mib": 20002,
        "t_s": 923.364
      },
      {
        "power_w": 300.05,
        "gpu_util_pct": 97,
        "memory_used_mib": 19999,
        "t_s": 924.437
      },
      {
        "power_w": 300.28,
        "gpu_util_pct": 98,
        "memory_used_mib": 19993,
        "t_s": 925.492
      },
      {
        "power_w": 300.29,
        "gpu_util_pct": 97,
        "memory_used_mib": 19973,
        "t_s": 926.548
      },
      {
        "power_w": 300.46,
        "gpu_util_pct": 98,
        "memory_used_mib": 19973,
        "t_s": 927.6
      },
      {
        "power_w": 300.05,
        "gpu_util_pct": 97,
        "memory_used_mib": 19969,
        "t_s": 928.652
      },
      {
        "power_w": 300.33,
        "gpu_util_pct": 98,
        "memory_used_mib": 19969,
        "t_s": 929.707
      },
      {
        "power_w": 300.15,
        "gpu_util_pct": 97,
        "memory_used_mib": 19970,
        "t_s": 930.761
      },
      {
        "power_w": 300.41,
        "gpu_util_pct": 96,
        "memory_used_mib": 19974,
        "t_s": 931.818
      },
      {
        "power_w": 300.53,
        "gpu_util_pct": 97,
        "memory_used_mib": 19974,
        "t_s": 932.871
      },
      {
        "power_w": 300.28,
        "gpu_util_pct": 97,
        "memory_used_mib": 19980,
        "t_s": 933.933
      },
      {
        "power_w": 300.63,
        "gpu_util_pct": 97,
        "memory_used_mib": 19973,
        "t_s": 935.013
      },
      {
        "power_w": 300.01,
        "gpu_util_pct": 97,
        "memory_used_mib": 19971,
        "t_s": 936.072
      },
      {
        "power_w": 300.63,
        "gpu_util_pct": 97,
        "memory_used_mib": 19963,
        "t_s": 937.13
      },
      {
        "power_w": 300.46,
        "gpu_util_pct": 98,
        "memory_used_mib": 19959,
        "t_s": 938.196
      },
      {
        "power_w": 299.75,
        "gpu_util_pct": 97,
        "memory_used_mib": 19969,
        "t_s": 939.248
      },
      {
        "power_w": 300.37,
        "gpu_util_pct": 97,
        "memory_used_mib": 19971,
        "t_s": 940.3
      },
      {
        "power_w": 300.4,
        "gpu_util_pct": 97,
        "memory_used_mib": 19968,
        "t_s": 941.352
      },
      {
        "power_w": 300.28,
        "gpu_util_pct": 97,
        "memory_used_mib": 19968,
        "t_s": 942.404
      },
      {
        "power_w": 299.83,
        "gpu_util_pct": 96,
        "memory_used_mib": 19983,
        "t_s": 943.461
      },
      {
        "power_w": 300.2,
        "gpu_util_pct": 96,
        "memory_used_mib": 20000,
        "t_s": 944.533
      },
      {
        "power_w": 300.02,
        "gpu_util_pct": 97,
        "memory_used_mib": 20001,
        "t_s": 945.586
      },
      {
        "power_w": 300.14,
        "gpu_util_pct": 97,
        "memory_used_mib": 19999,
        "t_s": 946.639
      },
      {
        "power_w": 300.45,
        "gpu_util_pct": 98,
        "memory_used_mib": 19994,
        "t_s": 947.69
      },
      {
        "power_w": 300.39,
        "gpu_util_pct": 98,
        "memory_used_mib": 19994,
        "t_s": 948.745
      },
      {
        "power_w": 300.48,
        "gpu_util_pct": 97,
        "memory_used_mib": 19991,
        "t_s": 949.798
      },
      {
        "power_w": 300.32,
        "gpu_util_pct": 98,
        "memory_used_mib": 19991,
        "t_s": 950.849
      },
      {
        "power_w": 300.54,
        "gpu_util_pct": 97,
        "memory_used_mib": 19989,
        "t_s": 951.902
      },
      {
        "power_w": 300.61,
        "gpu_util_pct": 98,
        "memory_used_mib": 19988,
        "t_s": 952.962
      },
      {
        "power_w": 300.1,
        "gpu_util_pct": 98,
        "memory_used_mib": 19975,
        "t_s": 954.017
      },
      {
        "power_w": 300.15,
        "gpu_util_pct": 97,
        "memory_used_mib": 19977,
        "t_s": 955.072
      },
      {
        "power_w": 300.62,
        "gpu_util_pct": 97,
        "memory_used_mib": 19978,
        "t_s": 956.135
      },
      {
        "power_w": 300.17,
        "gpu_util_pct": 97,
        "memory_used_mib": 19977,
        "t_s": 957.194
      },
      {
        "power_w": 300.25,
        "gpu_util_pct": 97,
        "memory_used_mib": 19976,
        "t_s": 958.244
      },
      {
        "power_w": 300.36,
        "gpu_util_pct": 97,
        "memory_used_mib": 19979,
        "t_s": 959.294
      },
      {
        "power_w": 300.22,
        "gpu_util_pct": 98,
        "memory_used_mib": 19986,
        "t_s": 960.351
      },
      {
        "power_w": 299.96,
        "gpu_util_pct": 97,
        "memory_used_mib": 19986,
        "t_s": 961.404
      },
      {
        "power_w": 300.3,
        "gpu_util_pct": 97,
        "memory_used_mib": 19986,
        "t_s": 962.473
      },
      {
        "power_w": 300.29,
        "gpu_util_pct": 97,
        "memory_used_mib": 19985,
        "t_s": 963.527
      },
      {
        "power_w": 300.08,
        "gpu_util_pct": 98,
        "memory_used_mib": 19985,
        "t_s": 964.586
      },
      {
        "power_w": 300.16,
        "gpu_util_pct": 97,
        "memory_used_mib": 19985,
        "t_s": 965.644
      },
      {
        "power_w": 300.47,
        "gpu_util_pct": 97,
        "memory_used_mib": 19999,
        "t_s": 966.719
      },
      {
        "power_w": 299.96,
        "gpu_util_pct": 96,
        "memory_used_mib": 19997,
        "t_s": 967.768
      },
      {
        "power_w": 300.36,
        "gpu_util_pct": 97,
        "memory_used_mib": 20003,
        "t_s": 968.825
      },
      {
        "power_w": 300.6,
        "gpu_util_pct": 97,
        "memory_used_mib": 20003,
        "t_s": 969.879
      },
      {
        "power_w": 300.18,
        "gpu_util_pct": 97,
        "memory_used_mib": 19997,
        "t_s": 970.934
      },
      {
        "power_w": 300.67,
        "gpu_util_pct": 97,
        "memory_used_mib": 19997,
        "t_s": 971.988
      },
      {
        "power_w": 299.89,
        "gpu_util_pct": 93,
        "memory_used_mib": 19997,
        "t_s": 973.041
      },
      {
        "power_w": 300.07,
        "gpu_util_pct": 96,
        "memory_used_mib": 19996,
        "t_s": 974.103
      },
      {
        "power_w": 300.32,
        "gpu_util_pct": 97,
        "memory_used_mib": 19996,
        "t_s": 975.165
      },
      {
        "power_w": 300.15,
        "gpu_util_pct": 97,
        "memory_used_mib": 19980,
        "t_s": 976.218
      },
      {
        "power_w": 300.56,
        "gpu_util_pct": 97,
        "memory_used_mib": 19981,
        "t_s": 977.269
      },
      {
        "power_w": 300.25,
        "gpu_util_pct": 97,
        "memory_used_mib": 19981,
        "t_s": 978.324
      },
      {
        "power_w": 300.3,
        "gpu_util_pct": 97,
        "memory_used_mib": 19981,
        "t_s": 979.378
      },
      {
        "power_w": 300.37,
        "gpu_util_pct": 98,
        "memory_used_mib": 19982,
        "t_s": 980.431
      },
      {
        "power_w": 279.2,
        "gpu_util_pct": 74,
        "memory_used_mib": 19977,
        "t_s": 981.487
      },
      {
        "power_w": 248.95,
        "gpu_util_pct": 100,
        "memory_used_mib": 19977,
        "t_s": 982.538
      },
      {
        "power_w": 295.49,
        "gpu_util_pct": 100,
        "memory_used_mib": 19990,
        "t_s": 983.598
      },
      {
        "power_w": 294.71,
        "gpu_util_pct": 100,
        "memory_used_mib": 19995,
        "t_s": 984.678
      },
      {
        "power_w": 292.9,
        "gpu_util_pct": 100,
        "memory_used_mib": 19982,
        "t_s": 985.732
      },
      {
        "power_w": 296.06,
        "gpu_util_pct": 100,
        "memory_used_mib": 19982,
        "t_s": 986.786
      },
      {
        "power_w": 300.69,
        "gpu_util_pct": 96,
        "memory_used_mib": 19981,
        "t_s": 987.845
      },
      {
        "power_w": 300.35,
        "gpu_util_pct": 97,
        "memory_used_mib": 19989,
        "t_s": 988.895
      },
      {
        "power_w": 300.66,
        "gpu_util_pct": 97,
        "memory_used_mib": 19990,
        "t_s": 989.947
      },
      {
        "power_w": 300.34,
        "gpu_util_pct": 98,
        "memory_used_mib": 19989,
        "t_s": 991.003
      },
      {
        "power_w": 300.42,
        "gpu_util_pct": 98,
        "memory_used_mib": 19988,
        "t_s": 992.061
      },
      {
        "power_w": 300.37,
        "gpu_util_pct": 97,
        "memory_used_mib": 19995,
        "t_s": 993.116
      },
      {
        "power_w": 299.86,
        "gpu_util_pct": 97,
        "memory_used_mib": 19995,
        "t_s": 994.174
      },
      {
        "power_w": 300.55,
        "gpu_util_pct": 97,
        "memory_used_mib": 19995,
        "t_s": 995.23
      },
      {
        "power_w": 300.39,
        "gpu_util_pct": 97,
        "memory_used_mib": 19996,
        "t_s": 996.287
      },
      {
        "power_w": 300.17,
        "gpu_util_pct": 97,
        "memory_used_mib": 19989,
        "t_s": 997.356
      },
      {
        "power_w": 300.7,
        "gpu_util_pct": 98,
        "memory_used_mib": 19990,
        "t_s": 998.413
      },
      {
        "power_w": 300.44,
        "gpu_util_pct": 97,
        "memory_used_mib": 19990,
        "t_s": 999.464
      },
      {
        "power_w": 300.36,
        "gpu_util_pct": 97,
        "memory_used_mib": 19994,
        "t_s": 1000.517
      },
      {
        "power_w": 231.75,
        "gpu_util_pct": 43,
        "memory_used_mib": 19994,
        "t_s": 1001.571
      },
      {
        "power_w": 294.68,
        "gpu_util_pct": 100,
        "memory_used_mib": 19992,
        "t_s": 1002.623
      },
      {
        "power_w": 297.92,
        "gpu_util_pct": 97,
        "memory_used_mib": 19992,
        "t_s": 1003.679
      },
      {
        "power_w": 299.66,
        "gpu_util_pct": 98,
        "memory_used_mib": 19992,
        "t_s": 1004.729
      },
      {
        "power_w": 299.88,
        "gpu_util_pct": 97,
        "memory_used_mib": 19992,
        "t_s": 1005.782
      },
      {
        "power_w": 300.31,
        "gpu_util_pct": 97,
        "memory_used_mib": 19991,
        "t_s": 1006.86
      },
      {
        "power_w": 300.43,
        "gpu_util_pct": 97,
        "memory_used_mib": 19979,
        "t_s": 1007.917
      },
      {
        "power_w": 300.35,
        "gpu_util_pct": 97,
        "memory_used_mib": 20001,
        "t_s": 1008.974
      },
      {
        "power_w": 300.32,
        "gpu_util_pct": 97,
        "memory_used_mib": 20012,
        "t_s": 1010.034
      },
      {
        "power_w": 300.3,
        "gpu_util_pct": 98,
        "memory_used_mib": 20019,
        "t_s": 1011.088
      },
      {
        "power_w": 300.08,
        "gpu_util_pct": 97,
        "memory_used_mib": 20018,
        "t_s": 1012.14
      },
      {
        "power_w": 299.88,
        "gpu_util_pct": 97,
        "memory_used_mib": 20021,
        "t_s": 1013.208
      },
      {
        "power_w": 299.67,
        "gpu_util_pct": 97,
        "memory_used_mib": 20022,
        "t_s": 1014.264
      },
      {
        "power_w": 300.41,
        "gpu_util_pct": 97,
        "memory_used_mib": 20021,
        "t_s": 1015.322
      },
      {
        "power_w": 300.72,
        "gpu_util_pct": 97,
        "memory_used_mib": 20008,
        "t_s": 1016.379
      },
      {
        "power_w": 300.62,
        "gpu_util_pct": 97,
        "memory_used_mib": 20003,
        "t_s": 1017.446
      },
      {
        "power_w": 300.27,
        "gpu_util_pct": 97,
        "memory_used_mib": 20006,
        "t_s": 1018.608
      },
      {
        "power_w": 300.24,
        "gpu_util_pct": 97,
        "memory_used_mib": 20007,
        "t_s": 1019.658
      },
      {
        "power_w": 300.22,
        "gpu_util_pct": 98,
        "memory_used_mib": 20007,
        "t_s": 1020.73
      },
      {
        "power_w": 300.27,
        "gpu_util_pct": 97,
        "memory_used_mib": 20007,
        "t_s": 1021.784
      },
      {
        "power_w": 300.67,
        "gpu_util_pct": 97,
        "memory_used_mib": 20007,
        "t_s": 1022.836
      },
      {
        "power_w": 300.27,
        "gpu_util_pct": 97,
        "memory_used_mib": 20005,
        "t_s": 1023.891
      },
      {
        "power_w": 299.75,
        "gpu_util_pct": 97,
        "memory_used_mib": 20005,
        "t_s": 1024.954
      },
      {
        "power_w": 299.75,
        "gpu_util_pct": 97,
        "memory_used_mib": 20002,
        "t_s": 1026.008
      },
      {
        "power_w": 300.43,
        "gpu_util_pct": 97,
        "memory_used_mib": 20003,
        "t_s": 1027.075
      },
      {
        "power_w": 300.65,
        "gpu_util_pct": 98,
        "memory_used_mib": 20011,
        "t_s": 1028.129
      },
      {
        "power_w": 300.29,
        "gpu_util_pct": 97,
        "memory_used_mib": 20011,
        "t_s": 1029.208
      },
      {
        "power_w": 300.73,
        "gpu_util_pct": 96,
        "memory_used_mib": 20014,
        "t_s": 1030.262
      },
      {
        "power_w": 300.55,
        "gpu_util_pct": 98,
        "memory_used_mib": 20014,
        "t_s": 1031.326
      },
      {
        "power_w": 300.56,
        "gpu_util_pct": 97,
        "memory_used_mib": 20021,
        "t_s": 1032.38
      },
      {
        "power_w": 299.88,
        "gpu_util_pct": 98,
        "memory_used_mib": 20016,
        "t_s": 1033.435
      },
      {
        "power_w": 301.02,
        "gpu_util_pct": 98,
        "memory_used_mib": 20017,
        "t_s": 1034.492
      },
      {
        "power_w": 300.22,
        "gpu_util_pct": 95,
        "memory_used_mib": 20016,
        "t_s": 1035.536
      },
      {
        "power_w": 300.22,
        "gpu_util_pct": 96,
        "memory_used_mib": 20021,
        "t_s": 1036.589
      },
      {
        "power_w": 300.23,
        "gpu_util_pct": 97,
        "memory_used_mib": 20037,
        "t_s": 1037.647
      },
      {
        "power_w": 299.82,
        "gpu_util_pct": 97,
        "memory_used_mib": 20036,
        "t_s": 1038.698
      },
      {
        "power_w": 300.36,
        "gpu_util_pct": 97,
        "memory_used_mib": 20036,
        "t_s": 1039.751
      },
      {
        "power_w": 300.42,
        "gpu_util_pct": 98,
        "memory_used_mib": 20036,
        "t_s": 1040.804
      },
      {
        "power_w": 300.24,
        "gpu_util_pct": 97,
        "memory_used_mib": 20016,
        "t_s": 1041.859
      },
      {
        "power_w": 300.07,
        "gpu_util_pct": 97,
        "memory_used_mib": 20015,
        "t_s": 1042.907
      },
      {
        "power_w": 300.36,
        "gpu_util_pct": 98,
        "memory_used_mib": 20014,
        "t_s": 1043.963
      },
      {
        "power_w": 300.27,
        "gpu_util_pct": 98,
        "memory_used_mib": 20014,
        "t_s": 1045.015
      },
      {
        "power_w": 300.22,
        "gpu_util_pct": 98,
        "memory_used_mib": 19999,
        "t_s": 1046.073
      },
      {
        "power_w": 300.48,
        "gpu_util_pct": 98,
        "memory_used_mib": 19999,
        "t_s": 1047.127
      },
      {
        "power_w": 300.07,
        "gpu_util_pct": 98,
        "memory_used_mib": 19994,
        "t_s": 1048.179
      },
      {
        "power_w": 300.24,
        "gpu_util_pct": 97,
        "memory_used_mib": 19994,
        "t_s": 1049.231
      },
      {
        "power_w": 300.35,
        "gpu_util_pct": 97,
        "memory_used_mib": 20001,
        "t_s": 1050.296
      },
      {
        "power_w": 300.28,
        "gpu_util_pct": 97,
        "memory_used_mib": 20001,
        "t_s": 1051.352
      },
      {
        "power_w": 300.34,
        "gpu_util_pct": 98,
        "memory_used_mib": 20003,
        "t_s": 1052.404
      },
      {
        "power_w": 300.23,
        "gpu_util_pct": 98,
        "memory_used_mib": 20003,
        "t_s": 1053.46
      },
      {
        "power_w": 300.38,
        "gpu_util_pct": 98,
        "memory_used_mib": 20004,
        "t_s": 1054.513
      },
      {
        "power_w": 300.61,
        "gpu_util_pct": 97,
        "memory_used_mib": 20003,
        "t_s": 1055.567
      },
      {
        "power_w": 300.18,
        "gpu_util_pct": 97,
        "memory_used_mib": 20003,
        "t_s": 1056.62
      },
      {
        "power_w": 300.17,
        "gpu_util_pct": 98,
        "memory_used_mib": 19995,
        "t_s": 1057.678
      },
      {
        "power_w": 300.74,
        "gpu_util_pct": 97,
        "memory_used_mib": 20001,
        "t_s": 1058.736
      },
      {
        "power_w": 300.29,
        "gpu_util_pct": 98,
        "memory_used_mib": 20018,
        "t_s": 1059.786
      },
      {
        "power_w": 300.38,
        "gpu_util_pct": 97,
        "memory_used_mib": 20016,
        "t_s": 1060.849
      },
      {
        "power_w": 300.2,
        "gpu_util_pct": 97,
        "memory_used_mib": 20007,
        "t_s": 1061.903
      },
      {
        "power_w": 300.3,
        "gpu_util_pct": 98,
        "memory_used_mib": 20007,
        "t_s": 1062.962
      },
      {
        "power_w": 300.36,
        "gpu_util_pct": 97,
        "memory_used_mib": 20005,
        "t_s": 1064.022
      },
      {
        "power_w": 300.39,
        "gpu_util_pct": 97,
        "memory_used_mib": 20010,
        "t_s": 1065.079
      },
      {
        "power_w": 300.3,
        "gpu_util_pct": 98,
        "memory_used_mib": 20020,
        "t_s": 1066.134
      },
      {
        "power_w": 300.58,
        "gpu_util_pct": 98,
        "memory_used_mib": 20020,
        "t_s": 1067.187
      },
      {
        "power_w": 300.21,
        "gpu_util_pct": 97,
        "memory_used_mib": 20011,
        "t_s": 1068.24
      },
      {
        "power_w": 300.21,
        "gpu_util_pct": 98,
        "memory_used_mib": 20018,
        "t_s": 1069.295
      },
      {
        "power_w": 300.3,
        "gpu_util_pct": 97,
        "memory_used_mib": 20016,
        "t_s": 1070.346
      },
      {
        "power_w": 300.37,
        "gpu_util_pct": 97,
        "memory_used_mib": 20017,
        "t_s": 1071.398
      },
      {
        "power_w": 300.12,
        "gpu_util_pct": 97,
        "memory_used_mib": 20012,
        "t_s": 1072.45
      },
      {
        "power_w": 299.98,
        "gpu_util_pct": 97,
        "memory_used_mib": 20007,
        "t_s": 1073.516
      },
      {
        "power_w": 299.61,
        "gpu_util_pct": 95,
        "memory_used_mib": 20011,
        "t_s": 1074.57
      },
      {
        "power_w": 300.2,
        "gpu_util_pct": 97,
        "memory_used_mib": 20014,
        "t_s": 1075.634
      },
      {
        "power_w": 300.62,
        "gpu_util_pct": 98,
        "memory_used_mib": 20014,
        "t_s": 1076.692
      },
      {
        "power_w": 300.31,
        "gpu_util_pct": 98,
        "memory_used_mib": 20014,
        "t_s": 1077.742
      },
      {
        "power_w": 300.24,
        "gpu_util_pct": 98,
        "memory_used_mib": 20015,
        "t_s": 1078.806
      },
      {
        "power_w": 300.32,
        "gpu_util_pct": 97,
        "memory_used_mib": 20011,
        "t_s": 1079.861
      },
      {
        "power_w": 300.51,
        "gpu_util_pct": 98,
        "memory_used_mib": 20011,
        "t_s": 1080.916
      },
      {
        "power_w": 299.81,
        "gpu_util_pct": 97,
        "memory_used_mib": 20007,
        "t_s": 1081.976
      },
      {
        "power_w": 300.34,
        "gpu_util_pct": 97,
        "memory_used_mib": 20004,
        "t_s": 1083.036
      },
      {
        "power_w": 300.21,
        "gpu_util_pct": 97,
        "memory_used_mib": 20007,
        "t_s": 1084.102
      },
      {
        "power_w": 300.03,
        "gpu_util_pct": 97,
        "memory_used_mib": 20007,
        "t_s": 1085.157
      },
      {
        "power_w": 300.96,
        "gpu_util_pct": 98,
        "memory_used_mib": 20005,
        "t_s": 1086.21
      },
      {
        "power_w": 222.56,
        "gpu_util_pct": 100,
        "memory_used_mib": 20002,
        "t_s": 1087.282
      },
      {
        "power_w": 295.1,
        "gpu_util_pct": 100,
        "memory_used_mib": 19998,
        "t_s": 1088.331
      },
      {
        "power_w": 298.25,
        "gpu_util_pct": 93,
        "memory_used_mib": 20003,
        "t_s": 1089.382
      },
      {
        "power_w": 300.76,
        "gpu_util_pct": 97,
        "memory_used_mib": 20003,
        "t_s": 1090.433
      },
      {
        "power_w": 300.47,
        "gpu_util_pct": 97,
        "memory_used_mib": 20001,
        "t_s": 1091.492
      },
      {
        "power_w": 300.69,
        "gpu_util_pct": 97,
        "memory_used_mib": 19997,
        "t_s": 1092.554
      },
      {
        "power_w": 298.82,
        "gpu_util_pct": 86,
        "memory_used_mib": 19996,
        "t_s": 1093.637
      },
      {
        "power_w": 277.55,
        "gpu_util_pct": 21,
        "memory_used_mib": 19995,
        "t_s": 1094.702
      },
      {
        "power_w": 265.12,
        "gpu_util_pct": 100,
        "memory_used_mib": 20000,
        "t_s": 1095.751
      },
      {
        "power_w": 294.16,
        "gpu_util_pct": 86,
        "memory_used_mib": 20000,
        "t_s": 1096.878
      },
      {
        "power_w": 295.3,
        "gpu_util_pct": 100,
        "memory_used_mib": 20001,
        "t_s": 1097.929
      },
      {
        "power_w": 294.86,
        "gpu_util_pct": 100,
        "memory_used_mib": 20001,
        "t_s": 1098.983
      },
      {
        "power_w": 298.88,
        "gpu_util_pct": 97,
        "memory_used_mib": 20001,
        "t_s": 1100.06
      },
      {
        "power_w": 300.1,
        "gpu_util_pct": 98,
        "memory_used_mib": 20000,
        "t_s": 1101.128
      },
      {
        "power_w": 300.48,
        "gpu_util_pct": 97,
        "memory_used_mib": 20000,
        "t_s": 1102.182
      },
      {
        "power_w": 300.43,
        "gpu_util_pct": 97,
        "memory_used_mib": 20000,
        "t_s": 1103.236
      },
      {
        "power_w": 300.57,
        "gpu_util_pct": 98,
        "memory_used_mib": 20006,
        "t_s": 1104.29
      },
      {
        "power_w": 300.3,
        "gpu_util_pct": 98,
        "memory_used_mib": 20006,
        "t_s": 1105.348
      },
      {
        "power_w": 300.03,
        "gpu_util_pct": 97,
        "memory_used_mib": 20004,
        "t_s": 1106.398
      },
      {
        "power_w": 300.07,
        "gpu_util_pct": 98,
        "memory_used_mib": 20000,
        "t_s": 1107.457
      },
      {
        "power_w": 300.14,
        "gpu_util_pct": 97,
        "memory_used_mib": 20003,
        "t_s": 1108.509
      },
      {
        "power_w": 300.51,
        "gpu_util_pct": 98,
        "memory_used_mib": 20024,
        "t_s": 1109.561
      },
      {
        "power_w": 300.21,
        "gpu_util_pct": 97,
        "memory_used_mib": 20023,
        "t_s": 1110.61
      },
      {
        "power_w": 299.57,
        "gpu_util_pct": 97,
        "memory_used_mib": 20017,
        "t_s": 1111.659
      },
      {
        "power_w": 300.76,
        "gpu_util_pct": 98,
        "memory_used_mib": 20007,
        "t_s": 1112.711
      },
      {
        "power_w": 300.54,
        "gpu_util_pct": 97,
        "memory_used_mib": 20003,
        "t_s": 1113.767
      },
      {
        "power_w": 300.6,
        "gpu_util_pct": 97,
        "memory_used_mib": 20001,
        "t_s": 1114.819
      },
      {
        "power_w": 300.37,
        "gpu_util_pct": 97,
        "memory_used_mib": 19999,
        "t_s": 1115.869
      },
      {
        "power_w": 300.5,
        "gpu_util_pct": 98,
        "memory_used_mib": 19999,
        "t_s": 1116.918
      },
      {
        "power_w": 300.01,
        "gpu_util_pct": 97,
        "memory_used_mib": 19999,
        "t_s": 1117.972
      },
      {
        "power_w": 300.51,
        "gpu_util_pct": 97,
        "memory_used_mib": 19999,
        "t_s": 1119.021
      },
      {
        "power_w": 300.11,
        "gpu_util_pct": 98,
        "memory_used_mib": 19995,
        "t_s": 1120.089
      },
      {
        "power_w": 300.17,
        "gpu_util_pct": 97,
        "memory_used_mib": 19995,
        "t_s": 1121.139
      },
      {
        "power_w": 300.12,
        "gpu_util_pct": 97,
        "memory_used_mib": 19994,
        "t_s": 1122.191
      },
      {
        "power_w": 300.49,
        "gpu_util_pct": 98,
        "memory_used_mib": 20002,
        "t_s": 1123.245
      },
      {
        "power_w": 300.26,
        "gpu_util_pct": 98,
        "memory_used_mib": 20002,
        "t_s": 1124.298
      },
      {
        "power_w": 300.14,
        "gpu_util_pct": 97,
        "memory_used_mib": 20008,
        "t_s": 1125.35
      },
      {
        "power_w": 300.35,
        "gpu_util_pct": 98,
        "memory_used_mib": 20008,
        "t_s": 1126.403
      },
      {
        "power_w": 300.43,
        "gpu_util_pct": 98,
        "memory_used_mib": 20006,
        "t_s": 1127.453
      },
      {
        "power_w": 300.1,
        "gpu_util_pct": 97,
        "memory_used_mib": 20003,
        "t_s": 1128.502
      },
      {
        "power_w": 300.87,
        "gpu_util_pct": 97,
        "memory_used_mib": 20002,
        "t_s": 1129.551
      },
      {
        "power_w": 299.94,
        "gpu_util_pct": 98,
        "memory_used_mib": 20001,
        "t_s": 1130.606
      },
      {
        "power_w": 300.59,
        "gpu_util_pct": 96,
        "memory_used_mib": 19993,
        "t_s": 1131.655
      },
      {
        "power_w": 300.27,
        "gpu_util_pct": 97,
        "memory_used_mib": 19992,
        "t_s": 1132.711
      },
      {
        "power_w": 300.45,
        "gpu_util_pct": 98,
        "memory_used_mib": 19990,
        "t_s": 1133.763
      },
      {
        "power_w": 300.52,
        "gpu_util_pct": 98,
        "memory_used_mib": 19990,
        "t_s": 1134.816
      },
      {
        "power_w": 300.26,
        "gpu_util_pct": 97,
        "memory_used_mib": 19990,
        "t_s": 1135.865
      },
      {
        "power_w": 300.42,
        "gpu_util_pct": 97,
        "memory_used_mib": 19990,
        "t_s": 1136.918
      },
      {
        "power_w": 228.75,
        "gpu_util_pct": 100,
        "memory_used_mib": 19990,
        "t_s": 1137.972
      },
      {
        "power_w": 294.84,
        "gpu_util_pct": 100,
        "memory_used_mib": 19990,
        "t_s": 1139.031
      },
      {
        "power_w": 296.73,
        "gpu_util_pct": 97,
        "memory_used_mib": 19990,
        "t_s": 1140.083
      },
      {
        "power_w": 300.84,
        "gpu_util_pct": 97,
        "memory_used_mib": 19990,
        "t_s": 1141.136
      },
      {
        "power_w": 300.92,
        "gpu_util_pct": 97,
        "memory_used_mib": 19990,
        "t_s": 1142.191
      },
      {
        "power_w": 300.56,
        "gpu_util_pct": 97,
        "memory_used_mib": 19990,
        "t_s": 1143.235
      },
      {
        "power_w": 300.33,
        "gpu_util_pct": 97,
        "memory_used_mib": 20005,
        "t_s": 1144.296
      },
      {
        "power_w": 300.33,
        "gpu_util_pct": 98,
        "memory_used_mib": 20002,
        "t_s": 1145.38
      },
      {
        "power_w": 300.23,
        "gpu_util_pct": 97,
        "memory_used_mib": 20004,
        "t_s": 1146.433
      },
      {
        "power_w": 300.14,
        "gpu_util_pct": 97,
        "memory_used_mib": 20004,
        "t_s": 1147.485
      },
      {
        "power_w": 300.73,
        "gpu_util_pct": 97,
        "memory_used_mib": 20000,
        "t_s": 1148.537
      },
      {
        "power_w": 300.35,
        "gpu_util_pct": 97,
        "memory_used_mib": 20000,
        "t_s": 1149.583
      },
      {
        "power_w": 300.17,
        "gpu_util_pct": 98,
        "memory_used_mib": 19996,
        "t_s": 1150.648
      },
      {
        "power_w": 300.44,
        "gpu_util_pct": 97,
        "memory_used_mib": 19995,
        "t_s": 1151.704
      },
      {
        "power_w": 300.55,
        "gpu_util_pct": 97,
        "memory_used_mib": 19995,
        "t_s": 1152.755
      },
      {
        "power_w": 300.28,
        "gpu_util_pct": 98,
        "memory_used_mib": 19999,
        "t_s": 1153.803
      },
      {
        "power_w": 300.5,
        "gpu_util_pct": 97,
        "memory_used_mib": 19999,
        "t_s": 1154.846
      },
      {
        "power_w": 300.35,
        "gpu_util_pct": 97,
        "memory_used_mib": 19999,
        "t_s": 1155.897
      },
      {
        "power_w": 300.42,
        "gpu_util_pct": 97,
        "memory_used_mib": 19999,
        "t_s": 1156.949
      },
      {
        "power_w": 300.56,
        "gpu_util_pct": 97,
        "memory_used_mib": 19997,
        "t_s": 1157.992
      },
      {
        "power_w": 299.94,
        "gpu_util_pct": 98,
        "memory_used_mib": 19998,
        "t_s": 1159.036
      },
      {
        "power_w": 300.21,
        "gpu_util_pct": 97,
        "memory_used_mib": 19998,
        "t_s": 1160.088
      },
      {
        "power_w": 300.29,
        "gpu_util_pct": 97,
        "memory_used_mib": 19996,
        "t_s": 1161.137
      },
      {
        "power_w": 300.17,
        "gpu_util_pct": 97,
        "memory_used_mib": 20003,
        "t_s": 1162.188
      },
      {
        "power_w": 300.16,
        "gpu_util_pct": 97,
        "memory_used_mib": 19995,
        "t_s": 1163.236
      },
      {
        "power_w": 300.53,
        "gpu_util_pct": 98,
        "memory_used_mib": 19994,
        "t_s": 1164.28
      },
      {
        "power_w": 300.42,
        "gpu_util_pct": 98,
        "memory_used_mib": 19996,
        "t_s": 1165.327
      },
      {
        "power_w": 300.28,
        "gpu_util_pct": 98,
        "memory_used_mib": 19996,
        "t_s": 1166.386
      },
      {
        "power_w": 299.91,
        "gpu_util_pct": 97,
        "memory_used_mib": 19998,
        "t_s": 1167.436
      },
      {
        "power_w": 300.64,
        "gpu_util_pct": 97,
        "memory_used_mib": 19996,
        "t_s": 1168.485
      },
      {
        "power_w": 300.23,
        "gpu_util_pct": 97,
        "memory_used_mib": 19996,
        "t_s": 1169.538
      },
      {
        "power_w": 300.09,
        "gpu_util_pct": 98,
        "memory_used_mib": 20008,
        "t_s": 1170.604
      },
      {
        "power_w": 300.53,
        "gpu_util_pct": 97,
        "memory_used_mib": 20008,
        "t_s": 1171.689
      },
      {
        "power_w": 299.95,
        "gpu_util_pct": 98,
        "memory_used_mib": 20010,
        "t_s": 1172.742
      },
      {
        "power_w": 300.15,
        "gpu_util_pct": 97,
        "memory_used_mib": 20006,
        "t_s": 1173.793
      },
      {
        "power_w": 300.21,
        "gpu_util_pct": 98,
        "memory_used_mib": 20013,
        "t_s": 1174.862
      },
      {
        "power_w": 300.48,
        "gpu_util_pct": 98,
        "memory_used_mib": 20016,
        "t_s": 1175.928
      },
      {
        "power_w": 300.47,
        "gpu_util_pct": 97,
        "memory_used_mib": 20016,
        "t_s": 1176.978
      },
      {
        "power_w": 300.4,
        "gpu_util_pct": 97,
        "memory_used_mib": 20016,
        "t_s": 1178.028
      },
      {
        "power_w": 300.77,
        "gpu_util_pct": 98,
        "memory_used_mib": 20014,
        "t_s": 1179.079
      },
      {
        "power_w": 300.5,
        "gpu_util_pct": 97,
        "memory_used_mib": 20003,
        "t_s": 1180.129
      },
      {
        "power_w": 300.08,
        "gpu_util_pct": 97,
        "memory_used_mib": 19995,
        "t_s": 1181.178
      },
      {
        "power_w": 300.59,
        "gpu_util_pct": 97,
        "memory_used_mib": 19998,
        "t_s": 1182.232
      },
      {
        "power_w": 300.53,
        "gpu_util_pct": 97,
        "memory_used_mib": 19998,
        "t_s": 1183.282
      },
      {
        "power_w": 300.07,
        "gpu_util_pct": 98,
        "memory_used_mib": 19998,
        "t_s": 1184.33
      },
      {
        "power_w": 300.48,
        "gpu_util_pct": 98,
        "memory_used_mib": 19999,
        "t_s": 1185.378
      },
      {
        "power_w": 231.1,
        "gpu_util_pct": 100,
        "memory_used_mib": 19997,
        "t_s": 1186.427
      },
      {
        "power_w": 295.63,
        "gpu_util_pct": 100,
        "memory_used_mib": 19999,
        "t_s": 1187.475
      },
      {
        "power_w": 292.11,
        "gpu_util_pct": 100,
        "memory_used_mib": 20000,
        "t_s": 1188.537
      },
      {
        "power_w": 294.47,
        "gpu_util_pct": 100,
        "memory_used_mib": 20000,
        "t_s": 1189.601
      },
      {
        "power_w": 293.86,
        "gpu_util_pct": 100,
        "memory_used_mib": 20000,
        "t_s": 1190.654
      },
      {
        "power_w": 293.7,
        "gpu_util_pct": 100,
        "memory_used_mib": 20000,
        "t_s": 1191.705
      },
      {
        "power_w": 295.82,
        "gpu_util_pct": 100,
        "memory_used_mib": 19996,
        "t_s": 1192.76
      },
      {
        "power_w": 299.64,
        "gpu_util_pct": 97,
        "memory_used_mib": 19993,
        "t_s": 1193.813
      },
      {
        "power_w": 300.0,
        "gpu_util_pct": 97,
        "memory_used_mib": 19994,
        "t_s": 1194.868
      },
      {
        "power_w": 300.5,
        "gpu_util_pct": 97,
        "memory_used_mib": 19988,
        "t_s": 1195.924
      },
      {
        "power_w": 299.71,
        "gpu_util_pct": 97,
        "memory_used_mib": 19988,
        "t_s": 1196.982
      },
      {
        "power_w": 300.84,
        "gpu_util_pct": 97,
        "memory_used_mib": 19988,
        "t_s": 1198.041
      },
      {
        "power_w": 300.24,
        "gpu_util_pct": 97,
        "memory_used_mib": 19996,
        "t_s": 1199.094
      },
      {
        "power_w": 300.57,
        "gpu_util_pct": 97,
        "memory_used_mib": 20000,
        "t_s": 1200.149
      },
      {
        "power_w": 300.2,
        "gpu_util_pct": 97,
        "memory_used_mib": 20002,
        "t_s": 1201.203
      },
      {
        "power_w": 299.96,
        "gpu_util_pct": 96,
        "memory_used_mib": 20001,
        "t_s": 1202.257
      },
      {
        "power_w": 300.51,
        "gpu_util_pct": 98,
        "memory_used_mib": 20001,
        "t_s": 1203.316
      },
      {
        "power_w": 300.43,
        "gpu_util_pct": 98,
        "memory_used_mib": 20001,
        "t_s": 1204.369
      },
      {
        "power_w": 300.65,
        "gpu_util_pct": 98,
        "memory_used_mib": 19995,
        "t_s": 1205.422
      },
      {
        "power_w": 300.19,
        "gpu_util_pct": 97,
        "memory_used_mib": 19996,
        "t_s": 1206.477
      },
      {
        "power_w": 300.12,
        "gpu_util_pct": 97,
        "memory_used_mib": 19995,
        "t_s": 1207.532
      },
      {
        "power_w": 300.5,
        "gpu_util_pct": 97,
        "memory_used_mib": 19994,
        "t_s": 1208.585
      },
      {
        "power_w": 299.96,
        "gpu_util_pct": 97,
        "memory_used_mib": 19992,
        "t_s": 1209.637
      },
      {
        "power_w": 300.5,
        "gpu_util_pct": 96,
        "memory_used_mib": 19992,
        "t_s": 1210.692
      },
      {
        "power_w": 300.64,
        "gpu_util_pct": 98,
        "memory_used_mib": 19992,
        "t_s": 1211.743
      },
      {
        "power_w": 300.59,
        "gpu_util_pct": 98,
        "memory_used_mib": 19992,
        "t_s": 1212.802
      },
      {
        "power_w": 299.68,
        "gpu_util_pct": 97,
        "memory_used_mib": 19990,
        "t_s": 1213.87
      },
      {
        "power_w": 300.55,
        "gpu_util_pct": 97,
        "memory_used_mib": 19990,
        "t_s": 1214.925
      },
      {
        "power_w": 300.35,
        "gpu_util_pct": 97,
        "memory_used_mib": 19989,
        "t_s": 1215.979
      },
      {
        "power_w": 300.41,
        "gpu_util_pct": 98,
        "memory_used_mib": 19989,
        "t_s": 1217.04
      },
      {
        "power_w": 300.65,
        "gpu_util_pct": 97,
        "memory_used_mib": 19989,
        "t_s": 1218.098
      },
      {
        "power_w": 300.26,
        "gpu_util_pct": 97,
        "memory_used_mib": 19989,
        "t_s": 1219.153
      },
      {
        "power_w": 300.26,
        "gpu_util_pct": 97,
        "memory_used_mib": 19989,
        "t_s": 1220.209
      },
      {
        "power_w": 300.22,
        "gpu_util_pct": 97,
        "memory_used_mib": 19989,
        "t_s": 1221.262
      },
      {
        "power_w": 300.36,
        "gpu_util_pct": 97,
        "memory_used_mib": 19989,
        "t_s": 1222.318
      },
      {
        "power_w": 300.73,
        "gpu_util_pct": 97,
        "memory_used_mib": 19988,
        "t_s": 1223.371
      },
      {
        "power_w": 300.3,
        "gpu_util_pct": 97,
        "memory_used_mib": 19989,
        "t_s": 1224.425
      },
      {
        "power_w": 300.3,
        "gpu_util_pct": 97,
        "memory_used_mib": 19989,
        "t_s": 1225.483
      },
      {
        "power_w": 300.54,
        "gpu_util_pct": 97,
        "memory_used_mib": 19989,
        "t_s": 1226.542
      },
      {
        "power_w": 300.56,
        "gpu_util_pct": 97,
        "memory_used_mib": 19989,
        "t_s": 1227.596
      },
      {
        "power_w": 300.37,
        "gpu_util_pct": 97,
        "memory_used_mib": 19990,
        "t_s": 1228.652
      },
      {
        "power_w": 300.08,
        "gpu_util_pct": 97,
        "memory_used_mib": 19994,
        "t_s": 1229.709
      },
      {
        "power_w": 300.34,
        "gpu_util_pct": 97,
        "memory_used_mib": 19997,
        "t_s": 1230.769
      },
      {
        "power_w": 300.22,
        "gpu_util_pct": 96,
        "memory_used_mib": 19996,
        "t_s": 1231.83
      },
      {
        "power_w": 300.36,
        "gpu_util_pct": 97,
        "memory_used_mib": 19992,
        "t_s": 1232.911
      },
      {
        "power_w": 300.43,
        "gpu_util_pct": 98,
        "memory_used_mib": 19991,
        "t_s": 1233.967
      },
      {
        "power_w": 299.74,
        "gpu_util_pct": 97,
        "memory_used_mib": 19991,
        "t_s": 1235.025
      },
      {
        "power_w": 300.17,
        "gpu_util_pct": 97,
        "memory_used_mib": 19990,
        "t_s": 1236.079
      },
      {
        "power_w": 300.52,
        "gpu_util_pct": 97,
        "memory_used_mib": 19990,
        "t_s": 1237.175
      },
      {
        "power_w": 300.05,
        "gpu_util_pct": 97,
        "memory_used_mib": 20000,
        "t_s": 1238.232
      },
      {
        "power_w": 300.33,
        "gpu_util_pct": 97,
        "memory_used_mib": 20001,
        "t_s": 1239.288
      },
      {
        "power_w": 300.57,
        "gpu_util_pct": 97,
        "memory_used_mib": 19999,
        "t_s": 1240.343
      },
      {
        "power_w": 300.62,
        "gpu_util_pct": 97,
        "memory_used_mib": 19991,
        "t_s": 1241.396
      },
      {
        "power_w": 300.18,
        "gpu_util_pct": 97,
        "memory_used_mib": 19991,
        "t_s": 1242.448
      },
      {
        "power_w": 300.38,
        "gpu_util_pct": 97,
        "memory_used_mib": 19991,
        "t_s": 1243.505
      },
      {
        "power_w": 300.13,
        "gpu_util_pct": 98,
        "memory_used_mib": 19986,
        "t_s": 1244.565
      },
      {
        "power_w": 300.76,
        "gpu_util_pct": 98,
        "memory_used_mib": 19986,
        "t_s": 1245.632
      },
      {
        "power_w": 300.21,
        "gpu_util_pct": 97,
        "memory_used_mib": 19986,
        "t_s": 1246.716
      },
      {
        "power_w": 300.26,
        "gpu_util_pct": 97,
        "memory_used_mib": 19986,
        "t_s": 1247.762
      },
      {
        "power_w": 300.21,
        "gpu_util_pct": 97,
        "memory_used_mib": 19986,
        "t_s": 1248.818
      },
      {
        "power_w": 300.3,
        "gpu_util_pct": 97,
        "memory_used_mib": 19986,
        "t_s": 1249.877
      },
      {
        "power_w": 300.41,
        "gpu_util_pct": 98,
        "memory_used_mib": 19984,
        "t_s": 1250.937
      },
      {
        "power_w": 300.34,
        "gpu_util_pct": 98,
        "memory_used_mib": 19993,
        "t_s": 1251.991
      },
      {
        "power_w": 300.11,
        "gpu_util_pct": 98,
        "memory_used_mib": 19997,
        "t_s": 1253.052
      },
      {
        "power_w": 300.28,
        "gpu_util_pct": 97,
        "memory_used_mib": 19997,
        "t_s": 1254.105
      },
      {
        "power_w": 300.54,
        "gpu_util_pct": 96,
        "memory_used_mib": 19994,
        "t_s": 1255.151
      },
      {
        "power_w": 300.09,
        "gpu_util_pct": 97,
        "memory_used_mib": 19993,
        "t_s": 1256.204
      },
      {
        "power_w": 300.53,
        "gpu_util_pct": 96,
        "memory_used_mib": 19992,
        "t_s": 1257.26
      },
      {
        "power_w": 300.49,
        "gpu_util_pct": 98,
        "memory_used_mib": 19988,
        "t_s": 1258.317
      },
      {
        "power_w": 300.0,
        "gpu_util_pct": 97,
        "memory_used_mib": 19988,
        "t_s": 1259.372
      },
      {
        "power_w": 300.58,
        "gpu_util_pct": 97,
        "memory_used_mib": 19988,
        "t_s": 1260.428
      },
      {
        "power_w": 300.47,
        "gpu_util_pct": 97,
        "memory_used_mib": 19988,
        "t_s": 1261.493
      },
      {
        "power_w": 300.47,
        "gpu_util_pct": 98,
        "memory_used_mib": 19988,
        "t_s": 1262.548
      },
      {
        "power_w": 300.42,
        "gpu_util_pct": 93,
        "memory_used_mib": 19988,
        "t_s": 1263.608
      },
      {
        "power_w": 300.37,
        "gpu_util_pct": 97,
        "memory_used_mib": 19988,
        "t_s": 1264.666
      },
      {
        "power_w": 299.73,
        "gpu_util_pct": 98,
        "memory_used_mib": 19988,
        "t_s": 1265.719
      },
      {
        "power_w": 300.34,
        "gpu_util_pct": 96,
        "memory_used_mib": 19988,
        "t_s": 1266.774
      },
      {
        "power_w": 300.71,
        "gpu_util_pct": 98,
        "memory_used_mib": 19988,
        "t_s": 1267.825
      },
      {
        "power_w": 300.38,
        "gpu_util_pct": 97,
        "memory_used_mib": 19988,
        "t_s": 1268.883
      },
      {
        "power_w": 225.76,
        "gpu_util_pct": 67,
        "memory_used_mib": 19988,
        "t_s": 1269.942
      },
      {
        "power_w": 294.24,
        "gpu_util_pct": 100,
        "memory_used_mib": 19988,
        "t_s": 1270.996
      },
      {
        "power_w": 292.3,
        "gpu_util_pct": 100,
        "memory_used_mib": 19988,
        "t_s": 1272.048
      },
      {
        "power_w": 291.59,
        "gpu_util_pct": 98,
        "memory_used_mib": 19988,
        "t_s": 1273.103
      },
      {
        "power_w": 295.26,
        "gpu_util_pct": 100,
        "memory_used_mib": 19988,
        "t_s": 1274.165
      },
      {
        "power_w": 292.86,
        "gpu_util_pct": 89,
        "memory_used_mib": 19991,
        "t_s": 1275.22
      },
      {
        "power_w": 297.55,
        "gpu_util_pct": 97,
        "memory_used_mib": 19992,
        "t_s": 1276.288
      },
      {
        "power_w": 300.37,
        "gpu_util_pct": 98,
        "memory_used_mib": 19992,
        "t_s": 1277.34
      },
      {
        "power_w": 300.3,
        "gpu_util_pct": 97,
        "memory_used_mib": 19992,
        "t_s": 1278.396
      },
      {
        "power_w": 299.76,
        "gpu_util_pct": 97,
        "memory_used_mib": 19993,
        "t_s": 1279.452
      },
      {
        "power_w": 300.14,
        "gpu_util_pct": 97,
        "memory_used_mib": 20003,
        "t_s": 1280.508
      },
      {
        "power_w": 301.03,
        "gpu_util_pct": 97,
        "memory_used_mib": 20003,
        "t_s": 1281.566
      },
      {
        "power_w": 300.53,
        "gpu_util_pct": 98,
        "memory_used_mib": 20003,
        "t_s": 1282.794
      },
      {
        "power_w": 300.58,
        "gpu_util_pct": 97,
        "memory_used_mib": 19997,
        "t_s": 1283.85
      },
      {
        "power_w": 300.35,
        "gpu_util_pct": 98,
        "memory_used_mib": 19993,
        "t_s": 1284.911
      },
      {
        "power_w": 222.0,
        "gpu_util_pct": 75,
        "memory_used_mib": 19993,
        "t_s": 1285.965
      },
      {
        "power_w": 292.97,
        "gpu_util_pct": 100,
        "memory_used_mib": 19992,
        "t_s": 1287.016
      },
      {
        "power_w": 292.56,
        "gpu_util_pct": 100,
        "memory_used_mib": 19992,
        "t_s": 1288.097
      },
      {
        "power_w": 296.11,
        "gpu_util_pct": 100,
        "memory_used_mib": 19992,
        "t_s": 1289.161
      },
      {
        "power_w": 296.38,
        "gpu_util_pct": 100,
        "memory_used_mib": 19992,
        "t_s": 1290.236
      },
      {
        "power_w": 297.86,
        "gpu_util_pct": 97,
        "memory_used_mib": 19992,
        "t_s": 1291.295
      },
      {
        "power_w": 300.66,
        "gpu_util_pct": 98,
        "memory_used_mib": 19992,
        "t_s": 1292.354
      },
      {
        "power_w": 299.83,
        "gpu_util_pct": 97,
        "memory_used_mib": 19997,
        "t_s": 1293.414
      },
      {
        "power_w": 299.94,
        "gpu_util_pct": 95,
        "memory_used_mib": 19997,
        "t_s": 1294.471
      },
      {
        "power_w": 300.23,
        "gpu_util_pct": 97,
        "memory_used_mib": 19997,
        "t_s": 1295.528
      },
      {
        "power_w": 300.3,
        "gpu_util_pct": 95,
        "memory_used_mib": 19997,
        "t_s": 1296.581
      },
      {
        "power_w": 300.72,
        "gpu_util_pct": 97,
        "memory_used_mib": 19997,
        "t_s": 1297.634
      },
      {
        "power_w": 300.52,
        "gpu_util_pct": 97,
        "memory_used_mib": 19997,
        "t_s": 1298.691
      },
      {
        "power_w": 300.49,
        "gpu_util_pct": 97,
        "memory_used_mib": 19991,
        "t_s": 1299.744
      },
      {
        "power_w": 300.07,
        "gpu_util_pct": 96,
        "memory_used_mib": 19998,
        "t_s": 1300.799
      },
      {
        "power_w": 232.98,
        "gpu_util_pct": 100,
        "memory_used_mib": 20002,
        "t_s": 1301.852
      },
      {
        "power_w": 293.41,
        "gpu_util_pct": 100,
        "memory_used_mib": 20002,
        "t_s": 1302.91
      },
      {
        "power_w": 288.87,
        "gpu_util_pct": 100,
        "memory_used_mib": 20002,
        "t_s": 1303.956
      },
      {
        "power_w": 299.58,
        "gpu_util_pct": 100,
        "memory_used_mib": 19994,
        "t_s": 1305.016
      },
      {
        "power_w": 294.95,
        "gpu_util_pct": 97,
        "memory_used_mib": 19997,
        "t_s": 1306.071
      },
      {
        "power_w": 300.41,
        "gpu_util_pct": 97,
        "memory_used_mib": 19996,
        "t_s": 1307.129
      },
      {
        "power_w": 300.12,
        "gpu_util_pct": 97,
        "memory_used_mib": 19989,
        "t_s": 1308.196
      },
      {
        "power_w": 300.03,
        "gpu_util_pct": 97,
        "memory_used_mib": 19991,
        "t_s": 1309.253
      },
      {
        "power_w": 299.82,
        "gpu_util_pct": 97,
        "memory_used_mib": 19989,
        "t_s": 1310.308
      },
      {
        "power_w": 300.08,
        "gpu_util_pct": 97,
        "memory_used_mib": 19989,
        "t_s": 1311.364
      },
      {
        "power_w": 300.22,
        "gpu_util_pct": 98,
        "memory_used_mib": 19989,
        "t_s": 1312.417
      },
      {
        "power_w": 300.19,
        "gpu_util_pct": 98,
        "memory_used_mib": 19989,
        "t_s": 1313.472
      },
      {
        "power_w": 300.24,
        "gpu_util_pct": 97,
        "memory_used_mib": 19989,
        "t_s": 1314.529
      },
      {
        "power_w": 300.48,
        "gpu_util_pct": 97,
        "memory_used_mib": 19989,
        "t_s": 1315.576
      },
      {
        "power_w": 300.55,
        "gpu_util_pct": 97,
        "memory_used_mib": 19988,
        "t_s": 1316.63
      },
      {
        "power_w": 300.09,
        "gpu_util_pct": 97,
        "memory_used_mib": 19987,
        "t_s": 1317.692
      },
      {
        "power_w": 300.2,
        "gpu_util_pct": 97,
        "memory_used_mib": 19987,
        "t_s": 1318.747
      },
      {
        "power_w": 299.88,
        "gpu_util_pct": 97,
        "memory_used_mib": 19987,
        "t_s": 1319.802
      },
      {
        "power_w": 300.45,
        "gpu_util_pct": 97,
        "memory_used_mib": 19991,
        "t_s": 1320.86
      },
      {
        "power_w": 231.31,
        "gpu_util_pct": 100,
        "memory_used_mib": 19995,
        "t_s": 1321.924
      },
      {
        "power_w": 299.32,
        "gpu_util_pct": 97,
        "memory_used_mib": 19999,
        "t_s": 1322.976
      },
      {
        "power_w": 300.91,
        "gpu_util_pct": 97,
        "memory_used_mib": 20001,
        "t_s": 1324.03
      },
      {
        "power_w": 300.78,
        "gpu_util_pct": 97,
        "memory_used_mib": 20001,
        "t_s": 1325.083
      },
      {
        "power_w": 300.02,
        "gpu_util_pct": 97,
        "memory_used_mib": 20000,
        "t_s": 1326.132
      },
      {
        "power_w": 300.01,
        "gpu_util_pct": 97,
        "memory_used_mib": 19997,
        "t_s": 1327.19
      },
      {
        "power_w": 300.38,
        "gpu_util_pct": 97,
        "memory_used_mib": 19992,
        "t_s": 1328.243
      },
      {
        "power_w": 300.58,
        "gpu_util_pct": 98,
        "memory_used_mib": 19992,
        "t_s": 1329.29
      },
      {
        "power_w": 260.02,
        "gpu_util_pct": 5,
        "memory_used_mib": 19991,
        "t_s": 1330.342
      },
      {
        "power_w": 263.63,
        "gpu_util_pct": 100,
        "memory_used_mib": 19991,
        "t_s": 1331.394
      },
      {
        "power_w": 281.97,
        "gpu_util_pct": 100,
        "memory_used_mib": 19991,
        "t_s": 1332.452
      },
      {
        "power_w": 300.24,
        "gpu_util_pct": 98,
        "memory_used_mib": 19991,
        "t_s": 1333.514
      },
      {
        "power_w": 300.34,
        "gpu_util_pct": 98,
        "memory_used_mib": 19998,
        "t_s": 1334.575
      },
      {
        "power_w": 300.89,
        "gpu_util_pct": 98,
        "memory_used_mib": 19998,
        "t_s": 1335.654
      },
      {
        "power_w": 300.12,
        "gpu_util_pct": 97,
        "memory_used_mib": 19997,
        "t_s": 1336.708
      },
      {
        "power_w": 300.0,
        "gpu_util_pct": 97,
        "memory_used_mib": 19997,
        "t_s": 1337.761
      },
      {
        "power_w": 299.47,
        "gpu_util_pct": 97,
        "memory_used_mib": 19997,
        "t_s": 1338.942
      },
      {
        "power_w": 300.54,
        "gpu_util_pct": 98,
        "memory_used_mib": 19997,
        "t_s": 1340.035
      },
      {
        "power_w": 300.54,
        "gpu_util_pct": 97,
        "memory_used_mib": 19997,
        "t_s": 1341.087
      },
      {
        "power_w": 300.44,
        "gpu_util_pct": 98,
        "memory_used_mib": 19997,
        "t_s": 1342.14
      },
      {
        "power_w": 300.77,
        "gpu_util_pct": 97,
        "memory_used_mib": 19997,
        "t_s": 1343.198
      },
      {
        "power_w": 300.39,
        "gpu_util_pct": 97,
        "memory_used_mib": 19997,
        "t_s": 1344.257
      },
      {
        "power_w": 300.79,
        "gpu_util_pct": 98,
        "memory_used_mib": 19997,
        "t_s": 1345.309
      },
      {
        "power_w": 300.56,
        "gpu_util_pct": 98,
        "memory_used_mib": 19997,
        "t_s": 1346.37
      },
      {
        "power_w": 300.43,
        "gpu_util_pct": 97,
        "memory_used_mib": 19997,
        "t_s": 1347.424
      },
      {
        "power_w": 300.31,
        "gpu_util_pct": 97,
        "memory_used_mib": 19997,
        "t_s": 1348.483
      },
      {
        "power_w": 300.26,
        "gpu_util_pct": 98,
        "memory_used_mib": 19997,
        "t_s": 1349.543
      },
      {
        "power_w": 300.5,
        "gpu_util_pct": 97,
        "memory_used_mib": 19997,
        "t_s": 1350.606
      },
      {
        "power_w": 300.42,
        "gpu_util_pct": 98,
        "memory_used_mib": 19997,
        "t_s": 1351.658
      },
      {
        "power_w": 300.68,
        "gpu_util_pct": 97,
        "memory_used_mib": 19997,
        "t_s": 1352.713
      },
      {
        "power_w": 300.31,
        "gpu_util_pct": 97,
        "memory_used_mib": 19997,
        "t_s": 1353.763
      },
      {
        "power_w": 300.27,
        "gpu_util_pct": 97,
        "memory_used_mib": 19997,
        "t_s": 1354.824
      },
      {
        "power_w": 300.4,
        "gpu_util_pct": 97,
        "memory_used_mib": 19997,
        "t_s": 1355.876
      },
      {
        "power_w": 300.27,
        "gpu_util_pct": 97,
        "memory_used_mib": 19997,
        "t_s": 1356.932
      },
      {
        "power_w": 300.5,
        "gpu_util_pct": 97,
        "memory_used_mib": 19997,
        "t_s": 1357.994
      },
      {
        "power_w": 300.62,
        "gpu_util_pct": 96,
        "memory_used_mib": 19997,
        "t_s": 1359.046
      },
      {
        "power_w": 300.52,
        "gpu_util_pct": 97,
        "memory_used_mib": 19997,
        "t_s": 1360.098
      },
      {
        "power_w": 300.45,
        "gpu_util_pct": 97,
        "memory_used_mib": 19997,
        "t_s": 1361.158
      },
      {
        "power_w": 300.56,
        "gpu_util_pct": 98,
        "memory_used_mib": 19997,
        "t_s": 1362.22
      },
      {
        "power_w": 299.84,
        "gpu_util_pct": 97,
        "memory_used_mib": 19997,
        "t_s": 1363.273
      },
      {
        "power_w": 300.44,
        "gpu_util_pct": 97,
        "memory_used_mib": 19997,
        "t_s": 1364.328
      },
      {
        "power_w": 300.73,
        "gpu_util_pct": 97,
        "memory_used_mib": 19997,
        "t_s": 1365.383
      },
      {
        "power_w": 300.33,
        "gpu_util_pct": 97,
        "memory_used_mib": 19997,
        "t_s": 1366.443
      },
      {
        "power_w": 300.86,
        "gpu_util_pct": 97,
        "memory_used_mib": 19997,
        "t_s": 1367.497
      },
      {
        "power_w": 300.42,
        "gpu_util_pct": 97,
        "memory_used_mib": 19997,
        "t_s": 1368.548
      },
      {
        "power_w": 300.08,
        "gpu_util_pct": 97,
        "memory_used_mib": 19997,
        "t_s": 1369.603
      },
      {
        "power_w": 300.18,
        "gpu_util_pct": 97,
        "memory_used_mib": 19997,
        "t_s": 1370.657
      },
      {
        "power_w": 300.5,
        "gpu_util_pct": 98,
        "memory_used_mib": 19997,
        "t_s": 1371.729
      },
      {
        "power_w": 300.3,
        "gpu_util_pct": 97,
        "memory_used_mib": 19997,
        "t_s": 1372.782
      },
      {
        "power_w": 300.29,
        "gpu_util_pct": 98,
        "memory_used_mib": 19997,
        "t_s": 1373.832
      },
      {
        "power_w": 300.43,
        "gpu_util_pct": 98,
        "memory_used_mib": 19997,
        "t_s": 1374.888
      },
      {
        "power_w": 300.56,
        "gpu_util_pct": 97,
        "memory_used_mib": 19997,
        "t_s": 1375.937
      },
      {
        "power_w": 300.37,
        "gpu_util_pct": 97,
        "memory_used_mib": 19997,
        "t_s": 1376.992
      },
      {
        "power_w": 300.51,
        "gpu_util_pct": 97,
        "memory_used_mib": 19997,
        "t_s": 1378.062
      },
      {
        "power_w": 300.44,
        "gpu_util_pct": 97,
        "memory_used_mib": 19997,
        "t_s": 1379.116
      },
      {
        "power_w": 300.41,
        "gpu_util_pct": 98,
        "memory_used_mib": 19997,
        "t_s": 1380.167
      },
      {
        "power_w": 300.33,
        "gpu_util_pct": 98,
        "memory_used_mib": 19997,
        "t_s": 1381.219
      },
      {
        "power_w": 300.35,
        "gpu_util_pct": 98,
        "memory_used_mib": 19997,
        "t_s": 1382.274
      },
      {
        "power_w": 231.5,
        "gpu_util_pct": 100,
        "memory_used_mib": 19997,
        "t_s": 1383.327
      },
      {
        "power_w": 299.85,
        "gpu_util_pct": 97,
        "memory_used_mib": 19997,
        "t_s": 1384.381
      },
      {
        "power_w": 300.47,
        "gpu_util_pct": 98,
        "memory_used_mib": 19997,
        "t_s": 1385.433
      },
      {
        "power_w": 300.72,
        "gpu_util_pct": 98,
        "memory_used_mib": 19997,
        "t_s": 1386.485
      },
      {
        "power_w": 300.66,
        "gpu_util_pct": 97,
        "memory_used_mib": 19997,
        "t_s": 1387.535
      },
      {
        "power_w": 299.91,
        "gpu_util_pct": 98,
        "memory_used_mib": 19997,
        "t_s": 1388.594
      },
      {
        "power_w": 300.47,
        "gpu_util_pct": 95,
        "memory_used_mib": 19997,
        "t_s": 1389.65
      },
      {
        "power_w": 300.55,
        "gpu_util_pct": 98,
        "memory_used_mib": 19997,
        "t_s": 1390.705
      },
      {
        "power_w": 300.23,
        "gpu_util_pct": 98,
        "memory_used_mib": 19997,
        "t_s": 1391.753
      },
      {
        "power_w": 300.4,
        "gpu_util_pct": 98,
        "memory_used_mib": 19997,
        "t_s": 1392.806
      },
      {
        "power_w": 300.43,
        "gpu_util_pct": 97,
        "memory_used_mib": 19997,
        "t_s": 1393.871
      },
      {
        "power_w": 300.56,
        "gpu_util_pct": 97,
        "memory_used_mib": 19997,
        "t_s": 1394.928
      },
      {
        "power_w": 300.58,
        "gpu_util_pct": 97,
        "memory_used_mib": 19997,
        "t_s": 1395.977
      },
      {
        "power_w": 300.47,
        "gpu_util_pct": 98,
        "memory_used_mib": 19997,
        "t_s": 1397.029
      },
      {
        "power_w": 300.29,
        "gpu_util_pct": 98,
        "memory_used_mib": 19997,
        "t_s": 1398.09
      },
      {
        "power_w": 300.55,
        "gpu_util_pct": 98,
        "memory_used_mib": 19997,
        "t_s": 1399.138
      },
      {
        "power_w": 299.66,
        "gpu_util_pct": 96,
        "memory_used_mib": 19997,
        "t_s": 1400.184
      },
      {
        "power_w": 300.46,
        "gpu_util_pct": 98,
        "memory_used_mib": 19997,
        "t_s": 1401.235
      },
      {
        "power_w": 300.6,
        "gpu_util_pct": 98,
        "memory_used_mib": 19997,
        "t_s": 1402.289
      },
      {
        "power_w": 300.33,
        "gpu_util_pct": 98,
        "memory_used_mib": 19997,
        "t_s": 1403.338
      },
      {
        "power_w": 300.42,
        "gpu_util_pct": 98,
        "memory_used_mib": 19997,
        "t_s": 1404.378
      },
      {
        "power_w": 300.51,
        "gpu_util_pct": 96,
        "memory_used_mib": 19997,
        "t_s": 1405.43
      },
      {
        "power_w": 300.44,
        "gpu_util_pct": 98,
        "memory_used_mib": 19997,
        "t_s": 1406.474
      },
      {
        "power_w": 300.48,
        "gpu_util_pct": 97,
        "memory_used_mib": 19997,
        "t_s": 1407.525
      },
      {
        "power_w": 300.48,
        "gpu_util_pct": 98,
        "memory_used_mib": 19997,
        "t_s": 1408.577
      },
      {
        "power_w": 300.46,
        "gpu_util_pct": 98,
        "memory_used_mib": 19997,
        "t_s": 1409.629
      },
      {
        "power_w": 300.3,
        "gpu_util_pct": 98,
        "memory_used_mib": 19997,
        "t_s": 1410.679
      },
      {
        "power_w": 300.15,
        "gpu_util_pct": 97,
        "memory_used_mib": 19996,
        "t_s": 1411.742
      },
      {
        "power_w": 300.57,
        "gpu_util_pct": 97,
        "memory_used_mib": 19996,
        "t_s": 1412.868
      },
      {
        "power_w": 300.02,
        "gpu_util_pct": 97,
        "memory_used_mib": 19996,
        "t_s": 1413.924
      },
      {
        "power_w": 300.37,
        "gpu_util_pct": 96,
        "memory_used_mib": 19996,
        "t_s": 1414.979
      },
      {
        "power_w": 300.34,
        "gpu_util_pct": 98,
        "memory_used_mib": 19996,
        "t_s": 1416.039
      },
      {
        "power_w": 300.67,
        "gpu_util_pct": 98,
        "memory_used_mib": 19996,
        "t_s": 1417.093
      },
      {
        "power_w": 300.31,
        "gpu_util_pct": 96,
        "memory_used_mib": 19996,
        "t_s": 1418.146
      },
      {
        "power_w": 300.45,
        "gpu_util_pct": 98,
        "memory_used_mib": 19996,
        "t_s": 1419.201
      },
      {
        "power_w": 300.38,
        "gpu_util_pct": 98,
        "memory_used_mib": 19996,
        "t_s": 1420.254
      },
      {
        "power_w": 300.38,
        "gpu_util_pct": 98,
        "memory_used_mib": 19996,
        "t_s": 1421.305
      },
      {
        "power_w": 300.43,
        "gpu_util_pct": 96,
        "memory_used_mib": 19996,
        "t_s": 1422.37
      },
      {
        "power_w": 300.48,
        "gpu_util_pct": 97,
        "memory_used_mib": 19996,
        "t_s": 1423.422
      },
      {
        "power_w": 300.36,
        "gpu_util_pct": 96,
        "memory_used_mib": 19996,
        "t_s": 1424.471
      },
      {
        "power_w": 300.26,
        "gpu_util_pct": 97,
        "memory_used_mib": 19996,
        "t_s": 1425.524
      },
      {
        "power_w": 300.4,
        "gpu_util_pct": 98,
        "memory_used_mib": 19996,
        "t_s": 1426.575
      },
      {
        "power_w": 300.31,
        "gpu_util_pct": 97,
        "memory_used_mib": 19996,
        "t_s": 1427.625
      },
      {
        "power_w": 300.75,
        "gpu_util_pct": 98,
        "memory_used_mib": 19996,
        "t_s": 1428.673
      },
      {
        "power_w": 300.41,
        "gpu_util_pct": 97,
        "memory_used_mib": 19996,
        "t_s": 1429.729
      },
      {
        "power_w": 300.53,
        "gpu_util_pct": 97,
        "memory_used_mib": 19996,
        "t_s": 1430.789
      },
      {
        "power_w": 299.72,
        "gpu_util_pct": 97,
        "memory_used_mib": 19996,
        "t_s": 1431.844
      },
      {
        "power_w": 299.63,
        "gpu_util_pct": 96,
        "memory_used_mib": 19996,
        "t_s": 1432.894
      },
      {
        "power_w": 300.85,
        "gpu_util_pct": 98,
        "memory_used_mib": 19996,
        "t_s": 1433.947
      },
      {
        "power_w": 300.95,
        "gpu_util_pct": 98,
        "memory_used_mib": 19996,
        "t_s": 1434.998
      },
      {
        "power_w": 300.44,
        "gpu_util_pct": 98,
        "memory_used_mib": 19996,
        "t_s": 1436.06
      },
      {
        "power_w": 300.04,
        "gpu_util_pct": 95,
        "memory_used_mib": 19996,
        "t_s": 1437.107
      },
      {
        "power_w": 300.46,
        "gpu_util_pct": 97,
        "memory_used_mib": 19996,
        "t_s": 1438.175
      },
      {
        "power_w": 300.14,
        "gpu_util_pct": 97,
        "memory_used_mib": 19996,
        "t_s": 1439.239
      },
      {
        "power_w": 300.75,
        "gpu_util_pct": 97,
        "memory_used_mib": 19996,
        "t_s": 1440.299
      },
      {
        "power_w": 300.49,
        "gpu_util_pct": 97,
        "memory_used_mib": 19996,
        "t_s": 1441.351
      },
      {
        "power_w": 300.6,
        "gpu_util_pct": 98,
        "memory_used_mib": 19996,
        "t_s": 1442.401
      },
      {
        "power_w": 300.22,
        "gpu_util_pct": 98,
        "memory_used_mib": 19996,
        "t_s": 1443.454
      },
      {
        "power_w": 299.86,
        "gpu_util_pct": 97,
        "memory_used_mib": 19996,
        "t_s": 1444.506
      },
      {
        "power_w": 300.25,
        "gpu_util_pct": 97,
        "memory_used_mib": 19996,
        "t_s": 1445.555
      },
      {
        "power_w": 300.26,
        "gpu_util_pct": 97,
        "memory_used_mib": 19996,
        "t_s": 1446.601
      },
      {
        "power_w": 300.43,
        "gpu_util_pct": 98,
        "memory_used_mib": 19996,
        "t_s": 1447.651
      },
      {
        "power_w": 300.64,
        "gpu_util_pct": 97,
        "memory_used_mib": 19996,
        "t_s": 1448.7
      },
      {
        "power_w": 232.52,
        "gpu_util_pct": 0,
        "memory_used_mib": 19996,
        "t_s": 1449.752
      },
      {
        "power_w": 287.17,
        "gpu_util_pct": 100,
        "memory_used_mib": 19996,
        "t_s": 1450.8
      },
      {
        "power_w": 295.07,
        "gpu_util_pct": 100,
        "memory_used_mib": 19996,
        "t_s": 1451.852
      },
      {
        "power_w": 285.33,
        "gpu_util_pct": 100,
        "memory_used_mib": 19996,
        "t_s": 1452.912
      },
      {
        "power_w": 292.03,
        "gpu_util_pct": 89,
        "memory_used_mib": 19996,
        "t_s": 1453.962
      },
      {
        "power_w": 300.67,
        "gpu_util_pct": 98,
        "memory_used_mib": 19996,
        "t_s": 1455.013
      },
      {
        "power_w": 300.9,
        "gpu_util_pct": 97,
        "memory_used_mib": 19996,
        "t_s": 1456.069
      },
      {
        "power_w": 300.51,
        "gpu_util_pct": 97,
        "memory_used_mib": 19996,
        "t_s": 1457.122
      },
      {
        "power_w": 299.95,
        "gpu_util_pct": 97,
        "memory_used_mib": 19996,
        "t_s": 1458.173
      },
      {
        "power_w": 300.51,
        "gpu_util_pct": 98,
        "memory_used_mib": 19996,
        "t_s": 1459.226
      },
      {
        "power_w": 300.56,
        "gpu_util_pct": 98,
        "memory_used_mib": 19996,
        "t_s": 1460.273
      },
      {
        "power_w": 300.59,
        "gpu_util_pct": 97,
        "memory_used_mib": 19996,
        "t_s": 1461.323
      },
      {
        "power_w": 300.25,
        "gpu_util_pct": 97,
        "memory_used_mib": 19996,
        "t_s": 1462.38
      },
      {
        "power_w": 300.49,
        "gpu_util_pct": 98,
        "memory_used_mib": 19996,
        "t_s": 1463.434
      },
      {
        "power_w": 274.88,
        "gpu_util_pct": 67,
        "memory_used_mib": 19996,
        "t_s": 1464.485
      },
      {
        "power_w": 153.08,
        "gpu_util_pct": 0,
        "memory_used_mib": 692,
        "t_s": 1465.534
      },
      {
        "power_w": 135.94,
        "gpu_util_pct": 0,
        "memory_used_mib": 692,
        "t_s": 1466.66
      },
      {
        "power_w": 125.11,
        "gpu_util_pct": 0,
        "memory_used_mib": 692,
        "t_s": 1467.788
      },
      {
        "power_w": 123.89,
        "gpu_util_pct": 0,
        "memory_used_mib": 692,
        "t_s": 1468.922
      },
      {
        "power_w": 121.85,
        "gpu_util_pct": 0,
        "memory_used_mib": 692,
        "t_s": 1470.049
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
- fair_claim_class: natural_correctness
- fair_row_usable: False
- correctness_gate_ok: False
- equal_output_speed_ok: None
- energy_j: 436187.654
- avg_power_w: 296.693
- max_power_w: 302.56
- mean_gpu_util_pct: 95.57
- max_memory_used_mib: 20037
- tool_expected_valid: 17/38
- unexpected_tool_call_rate: None

## Per-Turn Cache Trace
turn      pt  eff_in  prefix_len  fresh_pf     hr   pf_s  pf_tps   out/req  dec_tps  mode  accept  pflash%  tool  wall_s
------------------------------------------------------------------------------------------------------------------------
   1   31484    None        None     31484      ?  26.49    1188   60/2048     34.2  None      AR        -     Y   29.97
   2    4342    None        None      4342      ?   4.44     978  606/2048     33.1  None      AR        -     N   24.47
   3    2845    None        None      2845      ?   3.08     925  417/2048     32.6  None      AR        -     N   15.96
   4    9510    None        None      9510      ?  10.30     923  726/2048     30.9  None      AR        -     N   35.58
   5    2089    None        None      2089      ?   2.51     832  107/2048     30.7  None      AR        -     Y    6.09
   6    2050    None        None      2050      ?   2.48     827  420/2048     30.4  None      AR        -     N   18.07
   7    9012    None        None      9012      ?  10.69     843 1003/2048     29.1  None      AR        -     N   46.95
   8    4847    None        None      4847      ?   6.11     793 2048/2048     28.3  None      AR        -     N   83.69
   9    3493    None        None      3493      ?   4.57     764  574/2048     28.1  None      AR        -     N   26.81
  10    1007    None        None      1007      ?   1.46     691 1579/2048     27.9  None      AR        -     N   59.86
  11     214    None        None       214      ?   0.43     501 2048/2048     27.9  None      AR        -     N   79.34
  12     178    None        None       178      ?   0.40     448 1677/2048     27.9  None      AR        -     N    64.2
  13     231    None        None       231      ?   0.46     499 1897/2048     27.9  None      AR        -     N   72.25
  14     105    None        None       105      ?   0.30     351 1888/2048     27.9  None      AR        -     N   71.81
  15    2413    None        None      2413      ?   3.15     765 1276/2048     27.9  None      AR        -     Y   50.94
  16    2732    None        None      2732      ?   3.82     716 1201/2048     27.3  None      AR        -     Y   52.44
  17    2547    None        None      2547      ?   3.40     749  164/2048     28.6  None      AR        -     Y    9.25
  18     552    None        None       552      ?   0.89     618  315/2048     27.4  None      AR        -     Y    12.5
  19     351    None        None       351      ?   0.64     551   69/2048     25.9  None      AR        -     Y    5.72
  20     133    None        None       133      ?   0.35     376  409/2048     27.6  None      AR        -     N   15.28
  21      80    None        None        80      ?   0.25     325  487/2048     26.5  None      AR        -     N   21.02
  22    9125    None        None      9125      ?  12.71     718  368/2048     26.3  None      AR        -     N   28.87
  23    3611    None        None      3611      ?   5.55     651 2048/2048     25.3  None      AR        -     N    91.2
  24    1092    None        None      1092      ?   1.89     579  301/2048     24.6  None      AR        -     Y   16.76
  25     331    None        None       331      ?   0.64     515 1986/2048     25.0  None      AR        -     Y   84.67
  26    3364    None        None      3364      ?   5.28     637  345/2048     24.7  None      AR        -     Y   21.65
  27    1005    None        None      1005      ?   1.68     597 2048/2048     24.6  None      AR        -     Y   89.56
  28    1752    None        None      1752      ?   2.85     615  121/2048     24.9  None      AR        -     Y    9.84
  29    2704    None        None      2704      ?   4.34     623  923/2048     24.5  None      AR        -     N   44.35
  30    1016    None        None      1016      ?   1.74     585 1125/2048     24.3  None      AR        -     Y   52.47
  31    4417    None        None      4417      ?   7.15     618 1820/2048     23.9  None      AR        -     Y   87.77
  32    3687    None        None      3687      ?   6.17     598  220/2048     23.8  None      AR        -     N   17.76
  33    3094    None        None      3094      ?   5.19     597  233/2048     23.5  None      AR        -     N    15.3
  34    2491    None        None      2491      ?   4.43     562  351/2048     22.7  None      AR        -     Y    22.2
  35     348    None        None       348      ?   0.79     441  179/2048     23.4  None      AR        -     N    8.63
  36    1058    None        None      1058      ?   2.02     523 1145/2048     23.0  None      AR        -     N   55.91
  37     377    None        None       377      ?   0.83     453 1538/2048     23.4  None      AR        -     Y   71.61
  38    2228    None        None      2228      ?   4.01     556  243/2048     23.1  None      AR        -     Y    14.7

## Arm Aggregate
```
{
  "ok": true,
  "censored": false,
  "expected_turns": 38,
  "valid_turns": 38,
  "total_attempts": 38,
  "error_count": 0,
  "total_wall_s": 1535.5,
  "total_prefill_s": 153.49,
  "total_decode_s": 1286.607,
  "sum_prompt_tokens": 121915,
  "sum_effective_in_tokens": null,
  "sum_fresh_prefill_tokens": 121915,
  "sum_requested_out_tokens": 77824,
  "sum_out_tokens": 33965,
  "out_token_mismatch_count": 34,
  "out_tokens_match_requested": false,
  "pin_decode_ok": null,
  "pin_decode_turns": null,
  "pin_decode_tool_turns": null,
  "pin_decode_non_tool_turns": null,
  "pin_decode_non_tool_mismatch_count": null,
  "pin_decode_non_tool_ok": null,
  "pin_decode_tool_stop_conflict": null,
  "pin_decode_claim_scope": null,
  "mean_cache_hit_ratio": null,
  "mean_prefill_tps": 645.542,
  "mean_decode_tps": 26.818,
  "weighted_prompt_prefill_tps": 794.286,
  "weighted_effective_prefill_tps": null,
  "weighted_fresh_prefill_tps": 794.286,
  "weighted_decode_tps": 26.399,
  "spec_engagement_rate": 0.0,
  "mean_accept_rate": null,
  "mean_disk_hit_rate": null,
  "sum_tool_expected_turns": 38,
  "sum_tool_expected_valid_turns": 17,
  "tool_call_valid_rate": 0.447,
  "unexpected_tool_call_rate": null,
  "charbench_valid_rate": null,
  "energy_j": 436187.654,
  "avg_power_w": 296.693,
  "max_power_w": 302.56,
  "mean_gpu_util_pct": 95.57,
  "max_memory_used_mib": 20037,
  "fair_contract": {
    "claim_class": "natural_correctness",
    "row_usable": false,
    "generation_ok": true,
    "natural_correctness_ok": false,
    "correctness_gate_ok": false,
    "equal_output_speed_ok": null,
    "tool_expected_ok": false,
    "unexpected_tool_ok": null,
    "charbench_ok": null,
    "note": null
  }
}
```
