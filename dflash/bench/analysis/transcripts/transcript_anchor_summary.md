# Transcript Anchor Coverage Summary

Total user turns analyzed: **168**

## Overall Anchor-Zero Rates

| N-gram | Zero-hit turns | % of total |
|--------|---------------|------------|
| 2-gram | 0 | 0.0% |
| 4-gram | 11 | 6.5% |
| 6-gram | 40 | 23.8% |

## 4-Gram Anchor-Zero Rate by Body-Token Bucket

| Body-token bucket | Total turns | Zero-hit | Zero% |
|-------------------|-------------|----------|-------|
| <=16K | 95 | 8 | 8.4% |
| 16-32K | 45 | 3 | 6.7% |
| 32-64K | 26 | 0 | 0.0% |
| 64-128K | 2 | 0 | 0.0% |

## 4-Gram Hit Count Distribution

| Hits | Turns |
|------|-------|
| 0 | 11 |
| 1 | 3 |
| 2 | 7 |
| 3 | 2 |
| 5 | 6 |
| 6 | 5 |
| 7 | 2 |
| 8 | 2 |
| 9 | 2 |
| 10 | 4 |
| 11 | 2 |
| 12 | 4 |
| 13 | 4 |
| 14 | 4 |
| 15 | 2 |
| 16 | 4 |
| 17 | 2 |
| 19 | 4 |
| 20 | 2 |
| 21 | 1 |
| ... (66 more values) | |

## Multi-Resolution Rescue Analysis

Of turns where 4-gram hits == 0:

| Rescue | Cases | % of 4-gram-zero |
|--------|-------|-----------------|
| 2-gram | 11 | 100.0% |
| 6-gram | 0 | 0.0% |
| Neither (true dead zone) | 0 | 0.0% |

## Qualitative: Representative Anchor-Zero Queries

10 anonymized turns where 4-gram anchor hits == 0:

| body_tokens | query (80 chars, paths redacted) |
|-------------|----------------------------------|
| 392 | `<task-notification> <task-id>abd5ab9c5093a0f80</task-id> <tool-use-id>toolu_01RH` |
| 19,407 | `I beg your pardon .... pFlash should work maybe not on this PR but we have it wi` |
| 125 | ` /resume_handoff  thoughts/shared/handoffs/general/2026-05-18_12-49-19_mtp-warm-` |
| 8,368 | `[SISYPHUS MODE ACTIVATED]    ## Orchestration Instructions  You are now operatin` |
| 28,288 | `This session is being continued from a previous conversation that ran out of con` |
| 11,702 | `ok stop you have opus, sonnet, haiku. use them as direct subagent you are task w` |
| 82 | `So to recap we have now fixed pFlash and TQ3. Can we do a benchmark against q8 b` |
| 26,373 | `Base directory for this skill: /home/<user>/.claude/skills/commit  # Commit Chan` |
| 11,821 | `Big question from Howard0su ```please think about if we can normalize mtp and dr` |
| 1,146 | `<command-message>gpt-5-4-prompting</command-message> <command-name>gpt-5-4-promp` |

Anchor-zero corpus size (for cosine-pool testing): **11** turns

## Qualitative Interpretation

Detected patterns in anchor-zero user turns:

- **tool_result_only**: 1 turns (9.1% of anchor-zero)
- **continuation_request**: 1 turns (9.1% of anchor-zero)
- **acknowledgment**: 1 turns (9.1% of anchor-zero)

The anchor-zero problem predominantly affects short or abstract queries that lack literal repetition of any 4-token sequence from the body. This is the exact workload gap that the cosine-pool backstop is designed to cover: semantic similarity between query intent and relevant body chunks that share no verbatim n-gram overlap.

