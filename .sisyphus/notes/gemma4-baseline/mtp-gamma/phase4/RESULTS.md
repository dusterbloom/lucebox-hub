# Phase 4 — γ × ctx sweep results

Date: 2026-05-11 00:13–00:28 CEST
Binary: `dflash/build/test_gemma4_dflash` (commit `d8ebd12`)
Approach: A (re-capture single token when accept_drafts < K)

## Setup

- Model: Dense 31B + assistant Q4_K_M
- KV: TQ3_0 / TQ3_0
- `--temp 0 --ignore-eos --n-predict 64`
- Position mode: const (Google reference)
- Prompts:
  - 4K: 145-token essay (inline `--prompt`)
  - 16K: `prose_12288.txt` (12K tokens)
  - 64K: `long_code_50k.txt` (50K tokens)

## Decode tok/s

| ctx  | no-MTP | γ=1   | γ=2    | γ=4   | γ=8   |
|------|--------|-------|--------|-------|-------|
| 4K   | 19.63  | 19.18 | **20.21** | 16.54 | 16.54 |
| 16K  | **12.99** | 10.20 |  8.40  |  5.37 |  6.86 |
| 64K  |  6.55  |  5.54 | **8.42** |  6.54 |  5.33 |

## Accept rate (drafts accepted / drafts proposed)

| ctx  | γ=1   | γ=2   | γ=4   | γ=8   |
|------|-------|-------|-------|-------|
| 4K   | 0.64  | 0.58  | 0.38  | 0.20  |
| 16K  | 0.64  | 0.33  | 0.13  | 0.20  |
| 64K  | 0.69  | **0.73** | 0.49  | 0.28  |

## Headline findings

1. **γ=2 is the sweet spot at both 4K and 64K**.
   - 4K: 20.21 tok/s (+3% over no-MTP)
   - 64K: 8.42 tok/s (+29% over no-MTP)
2. **MTP accept rate at 64K is HIGH (0.69–0.73)**, contradicting the pre-rebase OPEN_QUESTIONS.md figure of 0.02. The post-rebase Bug-2 fix (submodule `daef232a6`) plus our γ>1 chain produces strong long-context accept rates.
3. **16K is the dead zone** — every MTP config loses to no-MTP. The medium-context regime has the worst overhead-to-benefit ratio: drafter cost is fixed, target verify saves little because the chain doesn't accept much (γ=2 accept drops from 0.58 at 4K to 0.33 at 16K), and re-capture overhead is non-trivial.
4. **γ=4 / γ=8 collapse at long context** under approach A. The re-capture single-token forward at 64K reads 64K KV slots (~80 ms), and partial accept fires almost every chain, so re-capture is paid most of the time.

## Approach B projection

Removing re-capture (capture all K+1 rows in one verify pass; host-side pick):
- 64K γ=4: 6.54 → ~9 tok/s (saves ~80 ms/chain, accept_rate stays 0.49)
- 64K γ=8: 5.33 → ~8 tok/s
- 16K γ=2: 8.40 → ~12 tok/s (saves ~30 ms/chain)
- 4K γ=2: 20.21 → ~22 tok/s (saves ~10 ms/chain, marginal)

Approach B is most valuable at long context where the re-capture forward is most expensive.

## Recommended user-facing defaults

Until approach B lands, the recommended config per ctx range:

| Ctx     | Best config                       | Decode tok/s |
|---------|-----------------------------------|--------------|
| ≤8K     | `--draft-method mtp --gamma 2`    | ≥20          |
| 8K–32K  | no MTP                            | ≈13          |
| ≥32K    | `--draft-method mtp --gamma 2`    | ≈8 at 64K    |

The 8K–32K dead zone is real and worth documenting. Above 64K we haven't tested but the trajectory suggests γ=2 stays the best.

## Logs

Per-cell logs in this directory: `none_g0_ctx*.log`, `mtp_g{1,2,4,8}_ctx{4096,16384,65536}.log`.

Sweep script: `../run_sweep.sh`.

## Open questions raised

1. Why is 16K so bad? Verify the masking / attention path isn't doing something pathological in the medium-ctx regime.
2. Is γ=2 at 128K / 256K / 1M still a win? Worth a follow-on sweep on Dense 31B (the prior dense-tq3-frontier work hit a 64K cache-fill anomaly that has since been resolved — would be useful to revisit with γ=2 now).
3. Approach B remains the obvious next implementation step (Phase 3.5).
