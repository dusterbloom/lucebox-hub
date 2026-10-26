# F1 trunk retention and F3 incremental bundles: design

Branch `trackc/f1-f3-design` off `f7ff3ba6` (the C1.5 binary under test). Design only: no engine code on this branch.
Scope: the Qwen3.8-27B hybrid slot engine (`Qwen35SeqEngine`, `KvFlashSlotPager`, `KvFlashSlotBundle`). Constants are
measured on the S0 qualification of 2026-10-26 (luce_box-kvflash `trackc/C1_5_PROGRESS.md`, findings 10 and 12).

## 0. What the engine holds today (f7ff3ba6)

- A parked session is `ParkedSession {history, tick, valid, ckpt{pos, slab}}` (`qwen35_seq_engine.h:272-281`): the tokens its
  KV rows hold, and **one** recurrent rung, the prompt-end checkpoint. The recurrent (SSM + conv) state cannot be sliced back,
  so a rewind to P needs a slab at P (`kvflash_slot_ladder.h` header comment).
- Resume (`qwen35_seq_engine.cpp:559-606`): `kvflash_pick_session` (longest lcp), then
  `L = kvflash_resume_point(lcp, len, ckpt, prompt_len)` (`kvflash_slot_turn.h`): `len` when the prompt extends the whole
  history **and** `len < prompt_len`, else `ckpt` when `lcp >= ckpt`, else -1 (cold). `L < len` restores the ckpt slab and
  truncates the pager; then `ckpt_[slot] = std::move(ps.ckpt)` (line 576): the restored rung stays as the fallback until the
  turn takes its own checkpoint.
- The turn's checkpoint, two regimes (`kvflash_sticky_ckpt_pos`):
  - LRU turn (prompt + reserve fits the 16,384-token pool): the state at the last `<|im_start|>` of the prompt (the assistant
    header), snapshotted **in place** into `ckpt_[slot]` at the cut ending there (`qwen35_seq_engine.cpp:2420-2423`).
  - Recall turn (mode 4): the state at R, the `<|im_start|>` of the prompt's last user turn (`kvflash_last_user_turn`), kept
    from the first pass; `ckpt_[slot] = SlotCkpt{r, snap}` replaces the fallback (`qwen35_seq_engine.cpp:822`).
- Park (`retire`, lines 2497-2505): `ParkedSession{history, ++tick, true, ckpt}`, then `save_bundle_`.
- Bundle (`kvflash_slot_bundle.h`): `slot<i>.kvb` = head (magic, identity, `ckpt_pos`, the history tokens, the end-of-history
  slab, the ckpt slab) + `KvFlashPager::serialize()` (44 B + nc x (2,228,224 + 8) B). The whole blob is rewritten at every
  park (tmp + fsync + rename, writer thread, one staged job per slot: the next park's `wait(slot)` stalls the step thread until
  the previous blob is on disk). Startup reads the whole file, unpacks, `adopt_parked`, restores the end slab.
- Measured (S0K, `bundle staged` lines and the file size): 2.125 MiB per 64-token chunk (34,816 B/token = 16 attention layers
  x K,V x 8 KV heads x 128 x q8_0), 149.6 MiB per slab; 346.0 / 450.1 / 594.7 MiB at H = 1,372 / 4,516 / 8,849; the 594.7 MiB
  blob took 921 ms to stage and 1,711 ms to write to `/mnt/c` (about 350 MB/s).

## 1. F1: trunk retention

### 1.1 The fork pattern and what each regime does with it

At a LIVECODE cut c with prompt T_c (its last token run is the assistant header `<|im_start|>assistant\n`), four requests fork
at the same point F_c = the last `<|im_start|>` of T_c:

| request | prompt | lcp with the parked history |
|---|---|---|
| H_c (history, max_tokens 1) | T_c | the resume from H_{c-1} |
| P2 | == T_c | len == prompt_len (identical prompt) |
| P1 | T_c[:F_c] + `<\|im_start\|>user` + question + header | F_c + 1 |
| H_{c+1} | T_c[:F_c] + header + the real reply + new messages | F_c + 2 (the reply text differs from the probe) |

LRU regime (prompts under about 16K tokens): H_c parks with ckpt = F_c. P2 resumes at F_c (the identical prompt cannot resume
at `len`), re-prefills 7 tokens and parks with ckpt = F_c again. P1 resumes at F_c and parks with ckpt = F' = its own header,
F' > F_c, **snapshotted into the same `SlotCkpt` in place: the F_c slab is destroyed** (line 2420-2423). H_{c+1} has
lcp = F_c + 2 < F' and `kvflash_resume_point` returns -1: `session not resumable`, and with one slot the cold admit drops the
session (`slot needed for a new session`) and prefills T_{c+1} from zero. This is the "branch returns cold in the LRU regime"
finding of C1_5_PLAN.md section 7. It is **not** reachable by the LIVECODE arms: every probe sits at 69K+ tokens. It is
reachable by S0-style fixtures with a history request after the probes, by short sessions and by any live client that
regenerates or branches below 16K.

Recall regime (the LIVECODE probes): H_c parks with ckpt = R_c, the last user turn of H_c's new tokens. For ses_f88 cut 21 that
is 13,399 tokens before the end (tool output after it). P2 resumes at R_c and re-prefills 13,399 tokens (54 s; the serial
engine re-decodes 4 tokens in 5.5 s); its own turn's R is R_c again (the suffix's last user turn is at its start), so it parks
with ckpt = R_c. P1 resumes at R_c (13,516 tokens, 29 s) and parks with ckpt = its question's `<|im_start|>` = F_c. H_{c+1}
then resumes at F_c: warm. So in this regime the branch returns are warm but every branch pays `[R_c, end)` first: C1.5
finding 10. The rung that would make P2 and P1 cheap, the state at F_c after H_c's replay, is never taken in a recall turn
(`kvflash_slot_turn_step` sets `ckpt` only for `!t.recall()`), by the C1.4 choice that the replay state at the prompt end is
shaped by the picks and should not seed the next turn's first pass.

### 1.2 The ancestor snapshot, named

The ancestor of every branch at cut c is **the recurrent state at F_c, the assistant header of T_c, after H_c's pass over
T_c**. Where it comes from:

- LRU regime: it already exists. It is H_c's checkpoint (`kvflash_slot_turn_step.ckpt` on the cut ending at `t.ckpt`,
  `qwen35_seq_engine.cpp:2420-2423`), it reaches the fork as `ps.ckpt` and is moved into `ckpt_[slot]` at line 576 when P2 or
  P1 resume; it is lost when the branch's own checkpoint is written over it in place. No new GPU work: F1 keeps it.
- Recall regime: it does not exist, and today no replay cut ends at F_c: `kvflash_slot_turn_next_cut` stops a cut at `t.ckpt`
  only for `!t.recall()` (`kvflash_slot_turn.h:91`) and `kvflash_slot_turn_step` sets `ckpt` only for `!t.recall()` (line
  109); the replay runs plain U cuts to `end` and crosses the header. F1 changes the replay's cut planning (a cut boundary at
  `t.ckpt`, the first-pass cuts and the capture boundary unchanged) and takes one more `snapshot_slot_state` there (a 149.6 MiB
  D2H, the same call the LRU rung makes). Taken only under the F7 policy of section 3.2 (one rule for both: a header rung on
  a recall turn exists when `end - R >= LUCE_KVFLASH_END_RUNG_MIN`).

### 1.3 Design: two rungs per parked session

`ParkedSession` and the live slot carry two rungs, `trunk` and `ckpt`, with `0 < trunk.pos < ckpt.pos <= history.size()`
(either may be absent, pos -1):

- **Rule A, a checkpoint at p** (LRU: the header; recall: R from the first pass, and now also the header at the replay's end):
  if the current `ckpt_[slot]` has `0 < q < p`, it becomes `trunk_[slot]` (the old trunk is dropped); if `q == p` it is
  replaced; `q > p` cannot happen after a truncation (assert). The slab is moved, not copied: no new memory for the LRU case.
- **Rule B, resume**: `kvflash_resume_point(lcp, len, ckpt, trunk, prompt_len)` returns the largest of `{len, ckpt, trunk}`
  that is `> 0`, `<= lcp` and `< prompt_len`; -1 when none. Rungs above L are dropped (their rows are truncated away), the
  rung at L is restored and stays as the fallback (today's line 576), the rung below L stays as the trunk. `L == len` restores
  nothing and keeps both.
- **Rule C, park**: `ParkedSession{history, tick, true, ckpt_[slot], trunk_[slot]}`.
- **Recall turns** keep the R rung (`trunk`, what the next turn's first pass starts from when the prompt diverges inside the
  turn) and add the header rung (`ckpt`) when the F7 threshold allows it. After H_c with the rung: trunk = R_c, ckpt = F_c.
  P2 resumes at F_c (7 tokens instead of 13,399); P1 at F_c (its question instead of 13,516); H_{c+1} at F_c (its new
  messages). A prompt that diverges before F_c but after R_c resumes at R_c as today. Without the rung (below the threshold)
  nothing changes against today.

With the rules above the LRU sequence reads: H_c parks {trunk = F_{c-1}'s rung or none, ckpt = F_c}; P2 resumes at F_c, finds
no `<|im_start|>` strictly above its resume point (`kvflash_slot_turn_plan` searches `i > kv_offset`, line 61: `t.ckpt = -1`),
takes no snapshot and parks with the admitted fallback {trunk as before, ckpt = F_c}; P1 resumes at F_c and parks
{trunk = F_c, ckpt = F'}; H_{c+1} has lcp = F_c + 2 and resumes at trunk F_c: warm, re-prefilling the reply and the new
messages. The cold branch return is gone with one slab moved.

Scope of two rungs: Rule A keeps the **two latest** checkpoints, not the fork ancestor as such. It covers a one-level fork
whose new user turn begins at the trunk (the LIVECODE P1 and a client's regenerate: the question starts at F_c, so the
branch's own R is F_c and no third position appears). A branch with a later user turn R' > F_c promotes R' at its first pass
and its header at its replay: F_c is dropped, and a return to the original history with lcp < R' is cold, as today. A nested
fork needs a third rung; out of scope, the fields are two named `SlotCkpt`, not an array.

### 1.4 Costs and risks

- Host memory: at most one more slab per parked session, 149.6 MiB (LRU: moved, so +1 only when the turn forked at the
  trunk; recall: +1 when the header rung is taken). The charge today (`qwen35_backend.cpp:1743`) is
  `ladder_bytes = slab x (recall + sticky) + staging`, with `staging = 3 x slab + ...` (the end snapshot and a head that
  lends two slabs). F1 adds one slab to the ladder (the trunk) and one to the staging (a head with three slabs): +2 slabs per
  slot with bundles on, +1 without; the peak counts a pending writer job (section 2.3). The read cap
  `kvflash_bundle_max_bytes` (`kvflash_slot_bundle.h:117`, two slabs) becomes three slabs or a near-maximum v2 file is
  rejected before unpacking (`init_bundles_`, line 674).
- Lifecycle: `trunk_` follows `ckpt_` at every site: assigned with it (`qwen35_seq_engine.cpp:95`), cleared on a cold admit
  (622), moved out and cleared before either retirement outcome (2496-2498), gone with `drop_session_` (639, after
  `wait(slot)`). A tainted or failed retirement must not leave an uncharged trunk slab behind.
- GPU time: one more `snapshot_slot_state` per recall turn that takes the rung (a D2H of 149.6 MiB, order 10 ms on the 3090's
  x16 link; measure).
- Recall quality: the header rung of a recall turn is the replay state, shaped by that turn's picks (the C1.4 objection). Any
  next prompt that cannot resume at `len` but covers the header (an identical prompt, a fork at the header, a reply the client
  re-tokenized differently) then starts its first pass from it instead of from R with the recent window, and a later turn
  inherits that state even when it is itself below the threshold. So G-E byte identity against `458fec57` holds only on
  turns where no header rung was taken; where one was, the gate is G-D (P1 norm, P2 path_hit) across whole trajectories, not
  per turn. If it costs recall, the fallback is Rule B with the header rung restricted to identical or near-identical prompts
  (`lcp >= prompt_len - 8`), which still covers P2.
- Bundles: `KvFlashSlotBundle` gains `trunk_pos` and the trunk slab (magic `KVFSLOT2`; a v1 file is skipped as "not a bundle of
  this server" and the server writes v2 from its first park). Without F3 that is +149.6 MiB per park on a whole-blob design:
  land F3 first or together (section 2.5).

### 1.5 Edits (for the implementation branch, not this one; about 60 engine lines plus tests)

`kvflash_slot_turn.h`: `kvflash_resume_point` with the trunk argument and a pure `kvflash_rungs_after_resume(ckpt, trunk, L)`
-> `(ckpt', trunk')`, `kvflash_rungs_after_ckpt(ckpt, trunk, p)` -> which rung the new slab takes and whether the old one
moves; `kvflash_slot_turn_next_cut` stops a replay cut at `t.ckpt` when the rung is wanted (a `KvFlashSlotTurn::end_rung`
flag set by the plan from `end - r` and the threshold); `kvflash_slot_turn_step` sets `ckpt` on that cut.
`qwen35_seq_engine.h`: `trunk` in `ParkedSession`, `trunk_` beside `ckpt_`. `qwen35_seq_engine.cpp`: lines 95, 563-576
(resume), 622, 639, 822 (recall R rung), 2420-2423 (header rung, both regimes), 2496-2504 (park), `save_bundle_` /
`init_bundles_` (the trunk in and out of the bundle, the read cap). `kvflash_slot_bundle.h`: v2 fields, `kvflash_bundle_max_bytes`
with three slabs. Tests first: `test_kvflash_slot_turn.cpp` for the helpers (lcp between trunk and ckpt -> trunk; lcp >= ckpt
-> ckpt; identical prompt -> ckpt below `prompt_len`; rung bookkeeping on resume at len / ckpt / trunk and on a checkpoint at
p == q, p > q; the replay cut at `t.ckpt` with and without the flag, including a header inside the last U cut as in
`cur=4800, end=5000, ckpt=4990`); `test_kvflash_slot_bundle.cpp` for v2 pack/unpack and the cap. GPU gate: the S0 fixture plus
one history request after the probes on S1: no `session not resumable` on the three branch returns, `analyze_c15.py` G-B
`branch_returns` with L == F_c, and G-E byte identity against the pre-F1 binary on the turns where no header rung was taken.

## 2. F3: incremental bundles

### 2.1 Why

Per park the whole blob is rewritten: 299.3 MiB (two slabs) + 4H + nc x 2.125 MiB. For ses_3dae8731 (101 history cuts, 6
probes, 111,761 tokens at the last cut) that is 357 MiB at cut 1, 2.9 GiB at the median park and 4.0 GiB at the last one;
302 GiB per session; 11 s of writer time per park at the end, which the next park waits for on the step thread. The K arm is
possible on a 20-request fixture only because of this.

### 2.2 Layout: a per-slot directory `slot<i>/`

- `chunks.<gen>.log`: append-only log of sealed 64-token chunk KV records, one generation live at a time, its name in the
  head. 16 B file header (magic, generation), then records: 16 B tag `{u32 logical chunk, u32 park generation, u64 crc64 of
  the payload}` + the chunk's KV bytes (2,228,224 B here; the pager derives it from the tensor strides, `kvflash_pager.h:169`,
  the format carries it in the head). Records are only ever appended (`pwrite` at the end); a record the committed head
  references is never overwritten. The file header means offset 0 is never a record, so 0 in the map is a hole.
- The ledger is **not** in the records: `was_resident` changes below L on every resume (`begin_resume_` calls
  `reselect_recent`, `qwen35_seq_engine.cpp:755`; `serialize` records the residency of the moment, `kvflash_pager.h:935`;
  `deserialize` rejects a blob whose resident count exceeds the pool, line 974). Stale per-record flags would restore a
  residency the pager never had. The whole current ledger (8 B x nc, 14 KB at 112K tokens) goes into the head at every park.
  The KV bytes below L are immutable: paging copies bytes, and `truncate` zeroes only the boundary chunk, which is inside the
  dirty range (`kvflash_pager.h:288`); scores live in `rc.scores` / `score_hook`, not in the chunk state.
- `slab.<pos>.<gen>`: one recurrent slab (the `ssm` and `conv` lists in today's `put_list` form), written once, tmp + fsync +
  rename. Three are live at most: the end-of-history slab, the ckpt rung, the trunk rung (F1).
- `head`: the commit point, tmp + fsync + rename, small. Magic `KVFSLOT3`, identity, the history tokens (4H), `ckpt_pos`,
  `trunk_pos`, the log file name, the three slab file names, the pager's blob header (44 B: 8 B magic + 9 x u32,
  `kvflash_pager.h:1011`), `n_chunks`, `sealed` (= floor(H / 64)), the map (`sealed` x u64 log offset, 0 = a hole: a zero
  chunk), the ledger, the partial tail chunk (< 64 tokens: its KV bytes, at most 2.125 MiB). About 2.6 MiB at H = 112K
  (0.45 MiB tokens + 14 KB map + 14 KB ledger + the tail).
- `.lock` and the identity key as today (one server per directory, a bundle of another server is skipped).

### 2.3 What a park writes: the KV dirty range is `[floor(L / 64), nc)`, the ledger and the slabs of the turn

Every KV row at or after the turn's resume point L was (re)written this turn (a cold admit has L = 0: everything). The
engine knows L (`kvf_kv_->set_next_resume`) and H at park, so no per-chunk dirty tracking is needed: the sealed chunks in
`[floor(L / 64), floor(H / 64))` are appended as new records with the park's generation, the tail chunk and the whole ledger
go into the head. A rewind inside the turn (mode 4: replay from R >= L) is inside that range already.

Which slabs a park writes: the end slab always; a rung slab whenever its snapshot was taken this turn. On a recall turn the
R snapshot is new every time (`ckpt_[slot] = SlotCkpt{r, snap}`, line 822, from this turn's first pass), so with F1 the park
writes R + end, and R + header + end when the header rung was taken: two or three slabs, not "the previous park's ckpt
file by name" (that holds only for a rung the turn did not retake, e.g. an LRU P2 that kept its fallback). The byte floor
per park is therefore two slabs on the recall path (section 2.5).

The job owns its bytes: the slabs are **copied** into the job (today's head packing lends the checkpoint vectors and gets them
back before `submit`, line 728; an asynchronous job that borrowed them would race the `std::move` of an immediate resume at
line 576), about 0.1 s of memcpy for three slabs. The writer's `done_` callback records, per slot, the generation of the last
**successfully committed** head; `wait(slot)` only orders (`kvflash_slot_bundle.h:211`). A job that fails leaves its
records and slab files unreferenced; the next park (or startup) unlinks them and never reuses a generation number.

Commit order per park, on the writer thread: (1) append the dirty records; (2) write the new slab files; (3) `fsync` the log,
the slab files and the directory (the names are published); (4) write `head.tmp`, `fsync`, rename over `head`, `fsync` the
directory; (5) unlink the slab files the new head does not reference and any `*.tmp.*`; (6) compaction when the dead records
in the log outnumber the live ones: write `chunks.<gen+1>.log` with the live records, then a head that names it with the new
offsets (step 4 again), then unlink `chunks.<gen>.log`. The log name and its map always travel in the same head commit: a
kill at any point leaves a head whose offsets address its own log (a renamed log under an old head, or a new head over an old
log, can address a wrong but valid record, which no crc catches). Compaction is rare: a fork rewrites about 1K tokens = 16
records per probe; a cold admit of a long session is the only big one.

### 2.4 Restore and crash cases

Startup reads `head`, checks magic and identity, reads the referenced slab files, assembles the pager blob in memory (the 44 B
header, then for each logical chunk the record at its offset via `pread`, with its tag's chunk index, its length against the
file size and its crc checked, or zeros for a hole, then the head's ledger) and calls `deserialize` as today, then
`restore_slot_state` with the end slab. The same bytes as today's whole-file read, once, at startup only; a streaming
`deserialize_from(reader)` is a follow-up, not needed for correctness. Orphans from a kill are cleaned at startup: `*.tmp.*`,
slab and log files the head does not name, log records past the head's map (reclaimed by the next compaction).

Three different guarantees, named: (a) process-kill atomicity comes from the order above and `rename`; (b) machine-crash
durability needs the directory `fsync`s of steps (3) and (4) before any dependency of the previous head is deleted (today's
whole-file helper, `kvflash_slot_bundle.h:146`, syncs the file but not the directory, so "as today" is not a durability
claim); (c) the crc64 detects corruption only; it does not replace the index, bounds, length and geometry checks. A kill
during (1)-(3) leaves the old head consistent because its records and files were never touched; during (4) either head is
complete; during (5)-(6) only garbage is left. A crc mismatch on a referenced record, or a missing slab or log file, skips the
bundle with a reason, as a bad magic does today. Test hook for the gate: `LUCE_KVFLASH_BUNDLE_CRASH_AT=<step>` aborts the
writer after that step, including inside (6) between the new log, the new head and the unlink. Oracle: before the head rename
the restart restores the **previous** park; after it the **new** park; byte-identical replies (G-E) from either.

### 2.5 Bytes for ses_3dae8731

Estimate (labelled): the prompt tokens at each cut are interpolated in rendered characters between the probe anchors
(76,132 at cut 98, 100,311 at 192, 111,761 at 214; 1,602 tokens of system prompt and tools at cut 0, the intercept of the
S0 rows); 101 history parks (max_tokens 1, so the end slab is at the prompt end) and 6 probe parks (+130 / +230 tokens);
new tokens per park: median 521, mean 1,050, max 17,788. Slab 149.6 MiB, chunk record 2.125 MiB, 64 tokens per chunk.

| layout | per park: first / median / last | per session (107 parks) | writer time per park at the end (350 MB/s) |
|---|---|---|---|
| today, whole blob | 357 / 2,954 / 4,018 MiB | 302 GiB | 11 s |
| F3: end slab + R rung + dirty records + head (two slabs) | 359 / 321 / 310 MiB | 35.3 GiB | 0.9 s |
| F3 + F1 header rung on that park (three slabs) | 509 / 471 / 460 MiB | +149.6 MiB per park that takes it | 1.3 s |
| F3 without the end slab (option F3b, one slab + the header rung when taken) | 209 / 171 / 161 MiB | 19.7 GiB | 0.5 s |

Under F3 the trunk is a new file whenever the turn retook it (every recall turn, section 2.3); it is free only for a rung the
turn kept. On the INTERIM_3 trio with the F7 threshold at 2,048 the header rung is taken on 3 of 227 turns (section 3.1), so
the two-slab row is the session figure. F3b drops the end-of-history slab: a resume at `len` becomes a resume at the ckpt rung
plus a replay of the reply, which is what every LIVECODE return already does and what a recall turn does by design; a live
client that sends our reply back verbatim pays the reply's re-prefill (one decode's worth of tokens). The floor after F3b is
one new slab per park, 149.6 MiB of f32 recurrent state; halving it (bf16 slabs, 75 MiB) is an option that needs
restore-precision evidence and is not designed here. The sizes are this geometry's: `16 x 2 x 8 x (128 / 32 x 34) x 64 =
2,228,224 B` per chunk (q8_0: 34 B per 32 values) and `48 x [128^2 x 48 + 3 x (6144 + 2 x 16 x 128)] x 4 = 156,893,184 B`
per slab (`qwen35_target_graph.cpp:168, 259`); the format reads them from the tensors, it does not hardcode them.

### 2.6 Edits (implementation branch; about 300 lines plus tests)

`kvflash_slot_bundle.h`: the record and head v3 pack/unpack (pure, length-checked like v1, the ledger in the head), the
dirty-range helper `kvflash_bundle_dirty_chunks(L, H, chunk_tokens)` -> `[lo, hi)`, the map assembly from `(offset, crc)`
pairs, the compaction decision (`dead > live`) and the generation bookkeeping (pure: the next generation never reuses a
failed one). `KvFlashBundleWriter`: a job carries the dirty records, copies of the new slabs and the head; `wait(slot)` as
today but on a job of a few hundred MiB; `done_` reports the committed generation. `qwen35_seq_engine.cpp`: `save_bundle_` builds the job from L and H (the pager's
`serialize_to` already streams one chunk at a time: give it a chunk range), `init_bundles_` assembles the blob from the
directory, the startup cleanup. Tests first in `test_kvflash_slot_bundle.cpp`: v3 round trip, the dirty range (cold, resume at
len, resume at a rung, rewind), a map with holes, the crash cases on a temp directory (extra records, orphan slab, torn
`head.tmp`, missing slab file, crc mismatch), compaction keeps every referenced record.

## 3. F7: the resume cost of the R rung (finding 10)

### 3.1 What is measured

On the S0 fixture both engines run the same mode-4 turn: the same R4 and the same replay length (S0R `mode4 phases`: first
11.1 s + replay 10.9 s on request 4; S0S1: `R4=8914 ... re-prefill 12761`). They differ in where a turn that **cannot resume
at `len`** starts. A plain continuation resumes at `len` on both (`kvflash_resume_point` takes `len` when the prompt extends
the whole parked history, `kvflash_slot_turn.h:150`; S0S1 requests 2-6: `resume L=1372 history=1372`, ..., `L=32336
history=32336`, the same new-token counts as the serial R arm). The R rung is paid only when the prompt does not extend the
history: an identical prompt (P2: `len == prompt_len` is refused), a fork (P1), and in a live agent loop a reply the client
re-rendered from text whose tokens differ from the generated ones (the reason the LRU regime checkpoints at the header; not
measurable on LIVECODE, whose history requests generate one token). There the serial engine resumes at its prompt-end live
cut and the slot engine at R_prev: for ses_f88 cut 21 that is 13,399 tokens on P2 (54.3 s vs 5.5 s) and 13,516 on P1
(28.9 s vs 4.1 s). The earlier reading of finding 10, "every next turn re-prefills `[R, end)`", was wrong.

Estimate on the INTERIM_3 trio (`c15/f7_estimate.py` over the serial ON16K log of 2026-10-25 and its rows, 227 requests,
tool-check follow-ups skipped as D1 does; the slot engine's resume rule applied with the LIVECODE shape: a history request
extends the previous prompt, P2 is identical, P1 and the return after it share the question's `<|im_start|>` + 1; the
user-turn positions come from the serial R4; labelled estimate): 218 of 227 turns are recall turns; the one-slot engine
re-prefills 424,567 tokens today and 405,027 with a header rung on every recall turn: **19,540 tokens saved (4.6%) on 13 of
227 requests** (9 P2, 4 P1 whose question starts at the header; the returns after P1 resume at P1's R on both), a mean of
86 tokens per request, median 860 on the 13, max 8,626 (ses_f88 cut 13, the same session as the 13,399 of S0 at another
cut; the other two sessions' last user turns sit 13-1,281 tokens before their cuts). A threshold of 2,048 tokens on `end - R`
keeps 14,068 of the 19,540 on 3 requests (all ses_f88); 4,096 keeps 8,626 on 1. Against the serial engine the slot's extra
re-prefill over the whole trio is that 19,540 (plus the 4-token re-decode the serial engine does on identical prompts). The
unmeasured case is the re-tokenized reply in a live loop, where the saving per turn is `[R, header)`, i.e. the tool output
after the last user turn, the 13.4K of S0 on a bad turn.

### 3.2 Options and their recall risk

| option | tokens saved on the trio | recall risk |
|---|---|---|
| A. header rung always (F1's header rung on every recall turn; the serial engine's semantics for identical prompts and header forks) | 19,540 on the trio (4.6%); `[R, header)` per re-tokenized reply in a live loop | measured in C1.4 R2-1 (`c5858025`): sticky 15/20 vs cold 19/20 on identical prompts with the replay's header state as the checkpoint, 19/20 with the R rung; answers matched 24/24, pick sets 9/24 (C1_4_PROGRESS.md:63). Fact-table workload: a 4,600-token table, a 2,048-token pool, question turns where `[R, end)` is the question itself |
| B. header rung only when `end - R` exceeds a threshold (else the R rung alone) | 14,068 at 2,048 (3 requests); 8,626 at 4,096 (1) | confined to turns whose last user turn sits far from the end. `end - R` is a span, not a role: a long question or assistant continuation crosses it too, and R is the last user turn **inside the resumed suffix** or L when it has none (`kvflash_prefill_reselect.h:57`). So the gate is G-D across whole trajectories, not a per-turn check |
| C. a rung at the first pass's end E (the state under LRU with the recent window, "what a cold turn has there") | not established: a rung at E cannot serve an identical prompt (`kvflash_resume_point` needs the rung `< prompt_len`) nor a P1 that diverges before E; it would have to sit at the header inside the first pass, which also moves the capture window | the recurrent state from the first pass, the attention rows of `[R, end)` from the replay: two passes in one slot, untested; one more 149.6 MiB D2H per recall turn |
| D. move R to the last turn of any role when `end - R` exceeds the threshold | as B | the engine already captures the final prompt's queries (`kvflash_slot_turn.h:45`, readback at `qwen35_seq_engine.cpp:792`), so this does not change what is captured; it changes the scoring prefix, the snapshot position, the reserve and the replay interval, and may change the capture length. Not proposed |
| E. keep the R rung and skip the first pass over `[R_prev, end_prev)` (reuse the parked rows, pass only the new tokens) | as A | needs a rung at the header to start from, so it is A or B with a different name |

Proposal: B, as `LUCE_KVFLASH_END_RUNG_MIN=<tokens>`, the **one** policy for F1's header rung on recall turns (unset: no
header rung on recall turns, today's behaviour; 0: A; default candidate 2,048), inside F1's two-rung bookkeeping (trunk = R,
ckpt = the header when taken). It changes nothing on a turn where the rung is not taken, and on the turn after a taken rung it
changes the first pass of any prompt that covers the header without extending the history; later turns inherit that state.
G-E against `458fec57` therefore holds on the turns without a rung in their past; the rest is G-D on the night fixture (S1 with
the rung against S1 without it and against serial R, same binary, same prompts, the 9+9 probes of the trio) **and** on the
C1.4 fact-table trajectories and a long-question variant, whole trajectories compared (agreement, not superiority, as in
C1_5_PLAN.md section 4). The identical-prompt case (P2, `lcp == len == prompt_len`) resolves itself with any rung below
`prompt_len`.

## 4. F8: the S1 vs R gap at N = 1

### 4.1 What is measured (S0, same binary, same prompts, one client)

| request | prompt | regime | R prefill s | S1 prefill s | S1 / R |
|---|---|---|---|---|---|
| 1, 2 | 1,372 / 4,516 | LRU | 1.5 / 2.9 | 1.3 / 2.9 | 0.9 / 1.0 |
| 3 | 8,849 | LRU | 3.9 | 4.8 | 1.23 |
| 4 | 21,675 (12,826 new, R4 8,914, replay 12,761) | recall | 22.4 (first 11.1 + replay 10.9) | 39.6 | 1.77 |
| 5 | 32,336 (10,661 new, replay 6,135) | recall | 15.5 | 30.1 | 1.94 |
| 6 | 46,126 (13,790 new, replay 13,399) | recall | 24.7 | 51.8 | 2.10 |
| P2 / P1 decode at 46K | | | 43.5 / 43.7 tok/s | 37.5 / 37.4 tok/s | 0.86 |

Request 4 processes 25,587 tokens on both engines (first pass + replay): 1,142 tok/s serial, 646 tok/s slot. The gap is not
the plan (same R4, same replay) and not the scorer (114-349 ms per turn against a 17-27 s gap); it grows with the resident
context: parity at 4.5K, 1.23x at 8.8K (an LRU turn, no replay), 1.8-2.1x at 21-46K, and decode -14% at 46K. C1.3 measured
parity at 8K.

### 4.2 Suspects

1. The paged-attention kernels over the block table (prefill and decode) once the resident window is full (16,384 rows): the
   serial engine attends over its pooled flat KV with a mask (`qwen35_backend.cpp:3011`, `with_mask`); the slot prefill
   builds without a mask on the paged path (`qwen35_seq_engine.cpp:2164`). The sharper form: **logical span versus resident
   rows**. The paged read's launch bound follows the logical KV length (`qwen35_target_graph.cpp:1383`, `launch_kv_len`,
   bounded by the configured context, not the physical pool), and the block-table column upload follows the logical prefix
   length too (`upload_kvflash_column(slot, kv_len)`, line 1133: 4 entries per chunk, -1 for an evicted one). At 46K tokens
   the kernel walks a 46K-row table of which 16K rows are resident; the serial pool walks 16-22K rows. Fits the growth with
   context, the parity below the pool size and the decode gap.
2. Per-cut page work on the step thread: U = 512 gives 26-28 cuts per turn, each with `settle_kvflash_`, deferred frees and
   `synchronize_paging`; the serial engine pages per cut too but off a different path. Would show as step time outside the graph.
3. Block-table or mask upload per cut: a cost that scales with the arena (16,448 rows), not the context; the parity at 4.5K
   argues against it unless it is per cut (26 cuts at 21K vs 9 at 4.5K).
4. Different first-pass residency: `reselect_recent` before the turn against the serial engine's recent pool; a different
   page-in volume shows in the page-event counts (serial: `128 page events; 64 in (136.0 MiB H2D)` per cut).
5. The scorer: excluded by its own log line.

### 4.3 The profile that separates them

Same binary, the S0 fixture, one client, in this order: (a) `LUCE_KVFLASH_STEP_TRACE=1` on S1, with the caveat that its
per-64-step aggregates cross requests, drop the unfinished aggregate and skip idle returns, and that "graph + readback" holds
graph build, input uploads, synchronization and snapshots, "upload" the staging work, "settle" the rewind and the scorer
(`qwen35_seq_engine.cpp:2161`, 2425); request 4 alone is about 26 cuts per pass, 52 in all. So the trace gains per-request,
per-phase counters (tokens processed; build, compute, readback, snapshot, rewind and page time apart) before it can name a
suspect: a graph share above 90% does not identify attention. Device-side attention timing (the paged read against the masked
flat read at equal token counts) is the direct test of suspect 1. (b) The same fixture with `--kvflash 65536` on both engines
so requests 4-6 stay LRU turns (no first pass, no replay, no scorer): a gap that stays is the kernel or page path; a gap that
closes is mode 4's page work; the pool size also changes residency, recall and resume, so the pair needs matched token work
(same resume points, same cuts) before assigning the difference. (c) Decode at 8.8K and at 46K on both engines (the probe
rows: `decode=...s(...tok/s)`): a gap growing with context is the attention kernel over the logical span, a flat gap is
per-step overhead. (d) Page-event counts per first pass from both logs. Tonight's S1 rows (227 requests, 69-113K prompts) give
the gap per request at full length against R; the plan's G-F threshold stands, this section only names what to measure when
it fails.

## 5. Order

F1 first: small, pure-testable, and it changes what the gates measure (the branch returns and finding 10). Then F3, which makes
bundles usable on the full sessions (the K arm on all 227 requests, not 20). F7 is F1's header rung on recall turns under one
threshold (section 3.2) and is gated by G-D. F8 is a measurement, not a change, until the profile names the suspect. F3 touches
no pick; F1 touches none on a turn without a taken header rung in its past, so G-E byte identity against `458fec57` on those
turns is the regression gate for both, and G-D across whole trajectories covers the rest. Not in scope: slab compression, more
than two rungs, cross-slot deduplication, a streaming deserialize.

## 6. Codex review (f7ff3ba6, 2026-10-26, `c15/review_f1f3.md`) and what changed

Seventeen items, each checked against the engine before it was folded in. Blockers: (1) the ledger is mutable below L
(`reselect_recent` at resume), so F3 keeps the whole ledger in the head and the KV dirty range alone in the log (2.2, 2.3);
(2) a recall turn's R snapshot is new every time, so a recall park writes two or three slabs, the "trunk is never written"
claim and the zero-cost trunk row are gone (2.3, 2.5); (3) compaction publishes a new log generation named in the head, never
a renamed log under an old head (2.3). Errors: (4) continuations resume at `len`; the F7 estimate is recomputed on the
resume rule: 19,540 tokens on 13 of 227 requests, not 231,540 on 213 (3.1); (5) the replay does not stop at `t.ckpt` on recall
turns, F1 changes the cut planning (1.2, 1.5); (6) two rungs keep the two latest checkpoints, the scope is the one-level fork
whose user turn starts at the trunk (1.3); (7) one policy for the header rung, F1 = F7-B, and G-E holds only without a rung in
the past (1.3, 1.4, 3.2, 5); (8) options C and D restated (3.2); (9) the log header makes offset 0 a hole, the pager header is
44 B (2.2). Omissions: (10) the threshold is a span, not a role, so the gate runs over whole trajectories including C1.4's
(3.2); (11) the job owns copies and the writer tracks the last committed generation (2.3); (12) the ladder and staging charge
and the read cap (1.4); (13) the trunk lifecycle sites (1.4, 1.5); (14) directory `fsync`s, the crash oracle, crc is detection
only (2.4); (15) F8's logical-span suspect and the trace's limits (4.2, 4.3); (16) the P2 bookkeeping example (1.3); (17) the
sizes are geometry-derived (2.5). Not taken: none.

## 7. Implementation notes (4df5783b F3, then F1 + F7 + the F3 review folds, 2026-10-26)

Deviations from sections 1-3 as built, with the reason: (1) slab files are `slab.<id>` with one per-slot counter for park
generations and slab ids (a checkpoint that becomes the trunk keeps its file; two slabs written in one park never collide);
after a restart the counter resumes past every id the head names, so nothing a durable head names is reused. (2) The head
carries `log_bytes`, the committed length of its log; a record past it is a torn tail the next park truncates
(`ftruncate` + `fseeko` before an append). (3) Compaction is folded into the next park (copies of the live records into a new
log generation, then the park's records, one head publication), not a separate step. (4) The partial tail chunk is a normal
record. (5) Slab tensor lengths are u32. (6) The engine builds on the last head the writer acknowledged (`SlotBundle.head`),
keeps the park in flight as `staged`, and the next park unlinks a failed park's leftovers (`kvflash_bundle_leftovers`) and
writes a fresh log; a park whose slab snapshot failed marks the slot for a rewrite. (7) A rung's slab file is rewritten when
the acknowledged head does not name it (a rung taken this turn, or every rung after a fresh log). (8) A directory this server
cannot adopt (another identity, over its limits, damaged) is set aside as `slot<i>.skipped`, replacing an earlier one, never
erased. (9) The loader refuses a head whose history or chunk count is over the server's context before it allocates the
pager blob (`KvFlashBundleLimits`), and a map that disagrees with the ledger (a materialized chunk without a record, a record
for a chunk never materialized). (10) `LUCE_KVFLASH_BUNDLE_CRASH_AT=<step>[:<commit>]`: the writer `_Exit(70)`s right after
that commit step of its n-th commit. (11) Directory `fsync` `EINVAL` is tolerated (drvfs): on such a filesystem the
machine-crash guarantee is the filesystem's. (12) The host charge is 2 rungs (+1 recall snapshot) and 4 staging slabs (the
end snapshot and its pack, the checkpoint and trunk packs; packs are reserved at their exact size).
