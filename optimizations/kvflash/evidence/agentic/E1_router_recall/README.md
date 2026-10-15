# E1: router recall of the target-QK index at prefill end (diagnostic)

Run 2026-10-15 on the local RTX 3090, 18:36-19:40, preregistered in `../../am/PREREG.md` ("E1: router recall of the target-QK index at prefill end"). A diagnostic: it decides nothing and changes no default.

**Question.** At prefill end, does luce_server's target-QK residency index (`LUCE_KVFLASH_POLICY=qk`) rank the 64-token chunk that holds the source of a recalled fact inside its top k?

**Answer, in one line.** Weakly. For the 207 facts older than 16K tokens (the ones POOL16K lost), the fact's chunk is in the top 64 of about 570 sealed chunks for 45% of probes, in the top 128 for 65%, in the top 8 for 4%. Chance is 11% and 23%. The median rank is 74 (about the 13th percentile). The index is far above chance and far from a precise router.

## Result (variant (a): the last prefill token's query)

Hit = any chunk overlapping the fact's own token span is in the top k. n_chunks per probe is 155-577 (median 382 for all, 570 for the old facts).

| subset | n | recall@8 | recall@16 | recall@32 | recall@64 | recall@128 |
|---|---|---|---|---|---|---|
| older than 16K (primary) | 207 | 8 (3.9%) | 19 (9.2%) | 49 (23.7%) | 94 (45.4%) | 135 (65.2%) |
| within 16K | 744 | 77 (10.3%) | 168 (22.6%) | 379 (50.9%) | 545 (73.3%) | 689 (92.6%) |
| all | 951 | 85 (8.9%) | 187 (19.7%) | 428 (45.0%) | 639 (67.2%) | 824 (86.6%) |
| chance, older than 16K | | 1.4% | 2.9% | 5.7% | 11.4% | 22.8% |
| chance, all | | 2.4% | 4.9% | 9.7% | 19.4% | 38.9% |

Rank of the best covering chunk, old facts (n = 207): 1: 0, 2-4: 2, 5-8: 6, 9-16: 11, 17-32: 30, 33-64: 45, 65-128: 41, over 128: 72; median 74, quartiles 34 and 172.

## Result (variant (b): max over the question's prefill tokens)

The max is taken per chunk over the queries of every prefill token in the last 512 prompt tokens (the question and its template, median 349 tokens, up to 610).

| subset | n | recall@8 | recall@16 | recall@32 | recall@64 | recall@128 |
|---|---|---|---|---|---|---|
| older than 16K (primary) | 207 | 0 (0.0%) | 2 (1.0%) | 14 (6.8%) | 39 (18.8%) | 84 (40.6%) |
| within 16K | 744 | 41 (5.5%) | 104 (14.0%) | 221 (29.7%) | 409 (55.0%) | 612 (82.3%) |
| all | 951 | 41 (4.3%) | 106 (11.1%) | 235 (24.7%) | 448 (47.1%) | 696 (73.2%) |

Variant (b) is worse than (a) at every k: median old-fact rank 173 against 74. Taking the max over many question tokens lets generic tokens of the template and the question words lift unrelated chunks. The last token carries the cleaner signal.

## Resident at prefill end

- Facts within 16K: 744 of 744 are resident (the LRU window). Their rank does not matter to a router.
- Facts older than 16K: 12 of 207 are resident, the same 12 POOL16K recalls. So 195 old facts sit in evicted chunks, and recall over those is 8, 19, 46, 86, 126 of 195 (4.1%, 9.7%, 23.6%, 44.1%, 64.6%) at k = 8, 16, 32, 64, 128 (variant (a)).
- Under variant (b) the same 195 give 0, 2, 13, 38, 80 (0.0%, 1.0%, 6.7%, 19.5%, 41.0%).

## Other hit definitions (same dump, `tables.md`)

- `lit`, the literal PREREG rule, chunks covering [src_tok, src_tok + gold length]: old facts 4.8%, 12.6%, 30.9%, 61.8%, 88.4% at k = 8..128 (variant (a)). `src_tok` is the start of the assistant call that produced the tool message, not the fact (see Deviations), so this measures where the message begins. It is higher probably because the start of a message carries the file path or command that the question names.
- `msg`, any chunk overlapping the whole source message (median 1,215 tokens, 19 chunks; up to 8,788 tokens): old facts 21.3%, 48.8%, 76.3%, 94.7%, 99.5%. A router that fetches the whole message once it finds any of it would need 64 chunks of budget for 95%.

## Deviations from the PREREG, and what contradicts the brief

1. **The B0 binary was not built from 2.0a-E (09bde623).** The B0 `luce_server` (md5 13c65b5cdce446f00f348a4101432de9, built 2026-10-14 01:08, `CMAKE_HOME_DIRECTORY` = this worktree's `server/`) matches `server/` at 0697848b "turn cuts default on with live restore", with no server change since. 09bde623 and 0697848b diverge (19 and 50 commits, 173 server files). The engine branch `kvflash/2.0r-e1-router-recall` is therefore based on **0697848b**, not 09bde623.
2. **`src_tok` is not the fact position.** In the tasks it is the token position where the assistant message that made the tool call starts (all 951 equal the harness boundary). The PREREG rule as written tests that point. The primary hit uses the fact's own span instead: the answer string located in the tool message (for `defname`, on the stated line), mapped to tokens with the harness template and tokenizer (`analyze_e1.py`, `locate_facts`; all 951 located, widest fact span 56 tokens). The literal rule is reported too (above).
3. **The prefill graph does not capture queries by default.** `q_cap` holds the last token of the graph, and `capture_qk` was on only for decode steps. With `KVFLASH_QK_DUMP` set the prefill graph captures the last token (variant (a)) and up to 512 tokens per batch into a new `q_cap_all` tensor (variant (b)); both are feasible. The final batch of a probe prefill holds only the few tokens after the last assistant header (turn cuts are on), so (b) folds the queries of every batch ending in the last 512 prompt tokens, not only the final batch.
4. **The replay is not B0's request sequence.** It sends probes only (with the primers the harness sends before each probe group); B0 interleaved the decision requests. The answers differ from B0's POOL16K on 168 of 951 probes: recall within 16K is 688/744 against 707/744, older than 16K 12/207 against 12/207 (`answer_parity.json`). This is a request-history effect: on a 5-probe smoke sequence the B0 binary, the E1 binary with LRU and no variable, the E1 binary with policy qk and no dump, and the E1 binary with the dump all gave identical answers, and all four differed from B0's own rows on 2 of 5. The dump and the policy do not change an answer on that check; the full run's agreement with B0 on the old facts (12 and 12) holds.
5. **Disk KV cache.** On this lineage the disk cache is opt-in (`--kv-cache-dir`) and there is no `--no-kv-cache` flag. B0 and E1 both ran without it: no `--kv-cache-dir` on the command line, `disk_hit=false` on 1106 of 1106 requests.
6. **Pooled keys cover every chunk.** `n_pooled == n_chunks` in all 1106 dump lines. Keys are pooled at seal time and kept across the live restore (`kvflash_live_restore` keeps the pool up to the cut), so evicted chunks are scored on real keys. The consuming-restore path (`kvflash_restore_pooled`) would lose them, but B0's flags do not use it.
7. **No reselect ran** (no `reselect` line in the server log). Probe answers are at most 64 tokens, as in the brief.
8. **Chunks, not rows.** `n_chunks` is the sealed chunks at prefill end, including the chunks that hold the question. The unsealed tail is always resident and never ranked (none of the 951 facts fell in it).

## Parity and provenance

| Item | Value |
|---|---|
| Question set | the 951 retained B0 probes (`B0_3090/retained.json`), `tasks_all60.jsonl`, B0 `trajectories.jsonl`; order as `run_arm.py` |
| Engine | branch `kvflash/2.0r-e1-router-recall`, commits 11f3a076 and d77c90f6 on top of 0697848b; pushed to `fork` |
| Binary | `/mnt/kimi-k3/build/e1-cuda/luce_server`, md5 c43a8341377147a383ffd399c2ec4657 (B0 binary 13c65b5cdce446f00f348a4101432de9) |
| ggml .so (E1 build) | libggml.so.0 93b382f49f59cd0bf56b6aa77ff35331; libggml-cuda.so.0 c9e2230e5ca205ad1da9d7f22dcfa76b; libggml-cpu.so.0 251c07daf02e327ab144b334904e2955; libggml-base.so.0 f8905880aa4aa63766be5b8cb370b2b7. The B0 build's libs differ in md5 (build path); the vendored ggml source is the same tree |
| Model | `Qwen3.8-27B-UD-Q4_K_M.gguf`, same path as B0 |
| Flags | as POOL16K: `--kvflash 16384`, `KVFLASH_STATE_LADDER=1 KVFLASH_LIVE_RESTORE=1`, `--max-ctx 52224`, `--prefix-cache-max-mib 12288 --prefix-cache-slots 4`, same template file, temperature 0, seed 1. Added: `LUCE_KVFLASH_POLICY=qk`, `KVFLASH_QK_DUMP=/mnt/kimi-k3/e1-router-recall/dump` |
| Machine | 3090 under `flock /mnt/kimi-k3/build/3090.lock`, GPU idle (compute apps 0) before the start, one server |
| Rows | 1106 requests (951 probes, 155 primers), 0 errors, one server start; raw files under `/mnt/kimi-k3/e1-router-recall/` |
| Caching state | probe cache-hit fraction 0.985 (B0 POOL16K on the same probes 0.987); `cached_tokens` median 24064 of 24448 prompt tokens; prefilled tokens median 349 (B0 307, max 8290 against 610); `provenance.json` has request samples with `prefix_len` |
| Disk KV cache | off (see 5), `disk_hit=false` 1106/1106 |
| Speeds | not the subject. Probe prefill median 1028 ms (B0 360 ms; E1 includes the CPU score fold), decode median 43.0 tok/s (B0 43.6), client wall median 1.66 s (B0 0.91 s) |
| Match of rows to dump lines | in order, `prompt_len == prompt_tokens`; 1106 of 1106 matched, 0 skipped |
| Statistics | one row per probe; no confidence intervals (a diagnostic with no decision) |

An earlier launch of the same run (single-threaded score fold, about 2x slower) was stopped after 133 rows and its output set aside (`/mnt/kimi-k3/e1-router-recall/aborted1_*`). Nothing from it is used.

## Engine change and tests

- `server/src/common/kvflash_qk_dump.h` (pure): `kvflash_qk_fold_max`, `kvflash_qk_dump_line`, `kvflash_qk_dump_append`, `kvflash_qk_prompt_hash`. Hooks in `qwen35_backend.cpp` (`kvflash_qk_dump_fold` after each late prefill batch, `kvflash_qk_dump_prefill` before the first decode token), `q_cap_all` in `qwen35_target_graph.cpp` and `internal.h`.
- Test `server/test/unit/test_kvflash_qk_dump.cpp`: the dumped `scores_last` and `scores_max` equal `kvflash_qk_chunk_scores` on synthetic data bit for bit, missing keys dump as -2, the threaded fold equals a sequential max over 37 queries, the unset variable gives an empty directory and writes no file, the set variable appends one line per call.
- Red: the test did not compile before the header existed (`kvflash_qk_dump.h: No such file or directory`). Green: `ctest -R kvflash_qk` in the E1 build, 2 of 2 passed.
- With the variable unset, the prefill graph is built with `capture_qk=false` and `q_cap_all` is not allocated, as before. Behaviour check: on the 5-probe smoke the E1 binary with LRU and no variable gave the B0 binary's answers.

## Files

- `analyze_e1.py`: fact location, rank and recall (run: `python3 -I analyze_e1.py rows dump summary.json`).
- `answer_parity.py`, `provenance.py`, `make_tables.py`.
- `summary.json` (all variants, subsets, histograms), `summary.per_probe.json` (rank of each probe), `tables.md` (all tables), `answer_parity.json`, `provenance.json`, `QK16K.meta.json`.
- `qk_dump.jsonl.xz`: the 1106 dump lines (every sealed chunk's score, resident list). `rows_QK16K.jsonl.xz`: the harness rows.
- Harness change: `agentic_eval/run_arm.py` gets arms `QK16K` (dump on) and `QK16KND` (policy qk, no dump), `--probes-only`, and `E1_BIN` for the binary.
