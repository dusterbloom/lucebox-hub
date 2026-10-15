M6 gate, 3090, Qwen3.8-27B-UD-Q4_K_M, --kvflash 8192, --no-kv-cache, 5 x ~18.8K-token prompts, temp 0, 5 interleaved runs per arm.
Arms: main 6f2b1795 vs kvflash/2.0b-m6-region-b with LUCE_KVFLASH_B_TOKENS=2048 (B allocated, empty). See PROVENANCE.txt (md5s) and cmp.out.
Result: 4/5 prompts byte-identical; prompt 4 diverges at char 285 (one near-tie token), deterministic in all 5 m6 runs.
Control m6_B0 (B_TOKENS=0): all 5 prompts identical to main. Caches were not dropped (no sudo).
