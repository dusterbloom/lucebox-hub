# Composition Bench Findings — 2026-05-21

## Setup

- Target: Qwen3.6-27B-Q4_K_M (tq3_0/tq3_0 KV cache, max_ctx=33792)
- PFlash drafter: Qwen3-0.6B-BF16
- MTP drafter: Qwen3.6-27B-MTP-Q4_K_M (gamma=2)
- DFlash draft: qwen3.6-27b-dflash (ddtree, budget=16)
- Workload: 3-turn multi-turn HTTP chat (clamp() function explanation chain)

---

## Q1: Does PFlash + MTP compose cleanly?

**Answer: PARTIAL — turn 1 works, server crashes on turn 2+**

Pattern across all 4 MTP+pflash=always cells (keep 0.025/0.05/0.10/0.20):
- Turn 1 (69 prompt tokens): pflash warmup, 100% kept (no real compression), MTP accept_rate=0.85, OK_DONE returned
- Turn 2 (270-280 tokens): pflash compress to 52-53% kept, server crashes silently after [unpark] target restored, before MTP decode starts

No SIGSEGV, no OOM, no stderr output. Server log ends cleanly at [pflash] N -> M -> P tokens.

Hypothesis: unpark + MTP decode re-initialization is broken when compression changes token count. Turn 1 gets 69->66->69 (preserved size). Turn 2 gets 275->142->145 (lossy to 52.7%). The token position mismatch likely corrupts MTP speculation state.

This bug is not keep_ratio-dependent. It triggers on the first request where actual compression occurs (not just warmup).

Yesterday's 0.85 single-turn result holds — single-turn always hits the 100%-kept warmup path and is not affected.

---

## Q2: Does PFlash + DFlash chain compose cleanly?

**Answer: YES**

Both dflash-off and dflash-always-k05 completed 3/3 turns with OK_DONE in every turn.

dflash-always server log shows the full multi-turn pflash cycle:
- Turn 1: 69->66->69 (100% kept), DFlash spec-decode 26.9% acceptance, OK_DONE
- Turn 2: 275->142->145 (52.7% kept), DFlash spec-decode, OK_DONE
- Turn 3: ~530 prompt tokens, OK_DONE

DFlash spec-decode does not crash after real compression. The composition is clean.

Wall overhead: dflash-always (5.76s/turn) vs dflash-off (3.16s/turn) = +82% from park/compress/unpark per turn. Expected at these context sizes (below threshold=32000); every turn compresses in mode=always.

---

## Q3: Does keep_ratio influence decode_tok_s?

**Answer: BLOCKED by server crash**

All MTP+pflash runs only produced 1 usable decode (turn 1, 100% kept regardless of keep_ratio). The sweep data is not meaningful for this question.

---

## Q4: Empirical optimum keep_ratio for MTP composition?

**Answer: BLOCKED** — the crash must be fixed before multi-turn MTP keep_ratio data can be collected.

---

## Summary

| Question | Answer |
|----------|--------|
| PFlash + MTP composes cleanly (multi-turn)? | NO — server crashes on turn 2+ after first real compression |
| PFlash + DFlash composes cleanly (multi-turn)? | YES — 3/3 turns, all OK_DONE |
| keep_ratio influences decode_tok_s? | BLOCKED |
| Empirical optimum keep_ratio? | BLOCKED |

---

## Bug: MTP + pflash=always server crash after first real compression

Trigger: context large enough that pflash actually drops tokens (< 100% kept). Specifically turn 2 in multi-turn where accumulated history exceeds forced-keep threshold.

Not affected: pflash=off + MTP (3/3 turns OK), DFlash + pflash=always (3/3 turns OK).

Evidence: server logs end at [pflash] N -> M -> P tokens with no error and no [mtp_decode] line for turn 2.

Next step: reproduce with a minimal 2-turn sequence. Check MTP state re-initialization in pflash unpark path — the prefix cache or KV slot mapping is likely not reset after lossy compression changes token positions.
