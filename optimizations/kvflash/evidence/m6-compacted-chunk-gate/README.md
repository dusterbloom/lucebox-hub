M6 (Revision 2 item 3) gate, 3090, Qwen3.8-27B-UD-Q4_K_M, --kvflash 8192, --no-kv-cache, 5 x ~18.8K-token prompts, temp 0,
5 interleaved rounds of three arms: main 6f2b1795, m6c (this branch, nothing compacted), neutral (same binary,
LUCE_KVFLASH_TEST_COMPACT=20:0: one chunk marked compacted, K/V untouched, beta 0). See PROVENANCE.txt and cmp.out.
readback_beta025: beta 0.25 on the same chunk, device kv_bias read back equals the pager's table.
sanity_beta_neg: beta -1000 on a chunk changes the text, so the bias reaches attention.
Caches were not dropped (no sudo). The ggml .so md5s of main and m6c differ (separate build trees).
