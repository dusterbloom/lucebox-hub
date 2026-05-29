# anchor transitive scan

`scan_and_force_transitive` (anchor_scan.cpp) expands the query pool with
tokens from newly-forced chunks and re-runs `scan_and_force` until fixed
point or max_iters (default 3) is reached.

Improves multi-hop retrieval: enables discovery of intermediate context
chunks whose tokens do not appear in the original query but connect
query-to-needle via shared rare tokens.

Empirical result: F1=0.587 on LongBench HotpotQA at ee7 + keep=0.10
(3× reproduced; see bench/2026-05-28_adaptive_stack/E_rerun_pflash_anchor_093712/SUMMARY.md).
Uncompressed apples-to-apples N=5: F1=0.547 (tied within noise).
Prior F1=0.628 is a real result at keep=0.05 on the prior cascade-gated binary
(the 0.697 figure is unverified). 0.587 and 0.628 both sit inside the verified
55–65% published range for the Q4_K_M+tq3_0 stack.

On by default. Disable via PFLASH_COMPRESS_ANCHOR_TRANSITIVE=0.
