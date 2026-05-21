# Per-Client Anchor Coverage Analysis
## Client × Anchor Coverage
| client | turns | mean body tok | p95 body tok | % anchor-zero | 2g rescue % | 6g rescue % |
|---|---|---|---|---|---|---|
| claude_code | 3 | 78 | 123 | 0.0% | 0.0% | 0.0% |
| opencode | 1 | 79 | 79 | 0.0% | 0.0% | 0.0% |

## 4-gram Hit Histograms
### claude_code
  hits= 1:   2 turns  ##
  hits= 2:   1 turns  #
### opencode
  hits= 1:   1 turns  #

## Representative Anchor-Zero Queries
### claude_code
### opencode

## Notes

### Hermes
No captured session data found in harness-runs/. The run_hermes.sh script stores logs
under `$LOG_DIR/hermes-home/` but no hermes-home directory exists in any harness run.
Hermes was not exercised in these runs.

### Corpus size caveat
The JSONL/DB session corpus is tiny: 3 CC turns + 1 opencode turn, all S < 96 tokens.
For S < QUERY_TOKENS (96), the C++ anchor scan produces a guaranteed self-match at p=q=0
(search_end=0, q0=0, ids[0:4] matches itself), so anchor_zero=0% is structurally
unavoidable at this scale. The anchor-zero question only matters for S >= 96 tokens.

### What the server log tells us
All 20 production drafter-skip entries (S: min=20 mean=8130 max=23726) had
forced_anchors > 0. These ran under pflash=ALWAYS — the [drafter-skip] path only fires
when skip_drafter=true. In AUTO mode, anchor_zero requests would fall through to
drafter-forward (not logged here). So the server log proves anchor coverage was 100% on
all captured requests under ALWAYS mode, but gives no information on how often AUTO mode
would have skipped (requires S >= 32K AND anchor_hits > 0).

## Server Log [drafter-skip] Evidence
Direct production anchor counts (from server.log):

### claude_code (20 requests)
forced-anchor distribution:
  anchors=  1: 1 requests
  anchors=  3: 3 requests
  anchors=  6: 2 requests
  anchors=  9: 1 requests
  anchors= 10: 1 requests
  anchors= 12: 8 requests
  anchors= 37: 2 requests
  anchors= 45: 1 requests
  anchors= 55: 1 requests
body size S: min=20 mean=8130 max=23725

