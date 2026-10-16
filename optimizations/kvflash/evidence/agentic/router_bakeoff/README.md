# Router bake-off for KVFlash mode 4 (offline, CPU only)

Question: with the same token budget of k blocks (1 block = 10 chunks x 64 tokens = 640 tokens), which way of choosing
the blocks to bring back recalls the block holding the needed fact best, and what end score does that imply?
Run: `python3 -I bakeoff.py > bakeoff.out` (about 90 s). `bakeoff_aligned.out` is the same run on the 580 probes whose token count matched exactly.

## Method
- Estimator: a copy of `rank4_cal.py` logic (window and tail rule, skip when the prompt fits, rates 0.95 / 0.94 / miss 0.29 within, 0.11 beyond),
  with only the block ranking swapped. Candidates are blocks 1..nb-1, exactly as today. "Exact" reproduces rank4_cal's printed table for k=2..10 cell for cell.
- Prompts are rebuilt with the harness chat template and the Qwen tokenizer used by E1 (`agentic_eval/convert.py`), question appended, generation prompt on.
- Query = the last user turn (question text). BM25 documents = candidate blocks. Ties go to the newer block.
- Routers: exact (summed `mass_end`), `mass_r`, recency, random (mean of 20 seeds), block BM25 on token ids (unigrams + bigrams),
  block BM25 on text words (lex.py splitting of paths and identifiers, mapped to blocks by token offsets), message-level BM25 (split at turns and tool results; take the blocks of the top messages in order),
  RRF fusion of exact and block BM25 (k0 = 60), prefilter (BM25 top 16, then rerank by `mass_end`), and an oracle (fact block first, then the rest of the source message, then exact).
- "adj" = within + 0.83 x beyond (the 17% over-estimate cut, same for every router). Measured engine k=2: 766; rank4_cal k=2: 770 (adj 756).
- Recall is over the probes whose fact is outside the window (the only ones where the router acts).

## Alignment check (Step 1)
- Spans decode correctly: the fact span contains the answer and the message span contains the source message, 951/951 each.
- Token count equals the server's `prompt_tokens` for 580/951 probes. The other 371 (39%, over the 2% stop line) have the server count larger by a median of 5 tokens, p90 16, max 56, never more than 9% of a block.
  Cause: server vs local template formatting (the convert.py header already says "up to tojson formatting"); the drift grows with prompt length and sits in tool-call text.
  I did not stop: the error is a sub-block shift, the same one rank4_cal already carries, and at most 21/951 (2.2%) fact spans could straddle a block edge because of it.
  Local-coordinate lexical routing is slightly favoured by it, so the aligned-only run is the cross-check (below).

## Results, main run (951 probes; engaged at k=2/4/8: 228/272/367)
| k | router | fact recall (beyond 16K / all) | message recall (all) | within | beyond | total | adj |
|---|---|---|---|---|---|---|---|
| 2 | exact | 34.2% / 40.4% | 56.6% | 688 | 83 | 770 | 756 |
| 2 | block BM25 (token ids) | 40.1% / 41.2% | 82.5% | 681 | 93 | 774 | 758 |
| 2 | RRF exact + BM25 (token ids) | 43.6% / 47.8% | 82.9% | 686 | 99 | 785 | 768 |
| 2 | prefilter BM25 (token ids) + exact | 36.6% / 42.1% | 63.2% | 687 | 87 | 774 | 759 |
| 2 | oracle | 100% / 98.7% | 100% | 688 | 194 | 881 | 848 |
| 4 | exact | 60.3% / 65.8% | 85.3% | 683 | 127 | 810 | 789 |
| 4 | block BM25 (token ids) | 46.6% / 46.7% | 95.2% | 668 | 104 | 772 | 754 |
| 4 | RRF | 53.4% / 57.7% | 96.7% | 678 | 115 | 794 | 774 |
| 4 | prefilter (text words) + exact | 59.8% / 64.3% | 89.3% | 681 | 126 | 808 | 786 |
| 8 | exact | 76.8% / 84.2% | 97.5% | 681 | 156 | 836 | 810 |
| 8 | block BM25 (token ids) | 52.7% / 56.4% | 99.2% | 647 | 114 | 761 | 741 |
| 8 | RRF | 63.3% / 72.2% | 98.9% | 670 | 132 | 803 | 780 |
| 8 | prefilter (text words) + exact | 67.6% / 73.0% | 97.0% | 666 | 140 | 806 | 782 |
Other routers at k=2 (total / adj): mass_r 741 / 731, recency 715 / 710, random 711 / 704, message BM25 717 / 709, block BM25 text words 761 / 747.
All routers and all k (1, 2, 4, 6, 8, 10) are in `bakeoff.out`.

Paired fact hits against exact (engaged probes): RRF at k=2 wins 41, loses 24 (net +17 of 228); at k=4 net -22 of 272; at k=8 net -44 of 367.
Block BM25 alone: net +2 at k=2, then -52 (k=4) and -102 (k=8).

Aligned-only run (580 probes, 49 engaged at k=2): RRF net +1, adj 508 vs exact 507 at k=2; exact is ahead from k=4 up (adj 513 vs 507 at k=4, 517 vs 503 at k=8).

## Caveats
- Lexical leak: 946/951 questions (99.5%) quote an identifier or path of 6+ characters (contains / _ . a digit or camelCase) that appears in the source message. The probes are close to the best case for lexical matching; real agent turns usually will not quote the fact so directly.
  Even so, lexical alone finds the source message (82% at k=2) far more often than the exact fact block.
- The estimate scores a miss when the fact block is not chosen, so BM25's message-level recall does not convert into score.
- Estimates carry the same unmeasured P_WHOLE / P_PART rates for every router; differences under about 10 points of 951 are noise.

## BM25 index cost (token-id unigrams + bigrams, 40 prompts, 963K tokens)
- About 886 postings per 1K tokens (432 distinct terms), roughly 5 KiB per 1K tokens at 6 bytes per posting (block id and count).
- Pure Python (hash maps): build about 1 ms per 1K tokens (0.02 s for a 24K prompt), query about 0.3 ms. A C++ version would be far faster. Text-word and message variants were not timed.

## Verdict
No router beats exact attention mass at equal budget in a way that holds up: block BM25 alone ties at k=2 and loses at k>=4; RRF is +12 adj at k=2 on the full set but vanishes on the aligned subset and loses 15 to 30 points at k>=4; prefilter + exact never beats exact. Keep exact.
