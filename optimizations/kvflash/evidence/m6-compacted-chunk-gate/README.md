M6 (Revision 2 item 3) gate after the Codex review fixes, 3090, Qwen3.8-27B-UD-Q4_K_M, --kvflash 8192, --no-kv-cache,
5 x ~18.8K-token prompts, temp 0, 5 interleaved rounds of three arms: main 6f2b1795, m6c (this branch, nothing
compacted), neutral (same binary, LUCE_KVFLASH_TEST_COMPACT=20:0: one chunk marked compacted, K/V untouched,
beta 0). All 5 rounds of raw JSON per arm are included. See PROVENANCE.txt and cmp.out (ab_cmp.py exits non-zero on
any mismatch or missing row). The neutral arm's driver (ab_m6.py) also fails unless every prompt logged a marked chunk,
a readback match and the in-engine graph check (every layer's flash-attention node carries its own kv_bias_<il> when
the bias is on, none when off).
smoke_neutral2 / sanity_beta_neg2: 2 x ~11K-token prompts; the second uses beta -1000. bias_lines.txt names the chunk
that was marked; oracle.txt shows the text changes against beta 0, so the bias reaches attention.
Caches were not dropped (no sudo). The ggml .so md5s of main and m6c differ (separate build trees).
