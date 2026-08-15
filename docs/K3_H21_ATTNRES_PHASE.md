# H21 — AttnRes phase pilot

VERDICT: PHASE MATTERS; BLOCK ENDS ARE PROMISING BUT NOT A UNIVERSAL TOKEN FIREBREAK.

The same four-complete-route intervention was placed at three equal-size,
pre-registered layer sets.  All other routed layers remained native exact.

| Seven-layer phase | Aviation mean / max KL | top choice | first generated token |
|---|---:|---:|---:|
| Block ends: 11,23,35,47,59,71,83 | 0.02528 / 0.06254 at one token | 5/5 | native |
| Block starts: 12,24,36,48,60,72,84 | 0.00764 / 0.01978 | 4/5 | differs |
| Block middles: 6,18,30,42,54,66,78 | 0.01064 / 0.02990 | 5/5 | native |

Block ends were extended on-policy to four aviation tokens.  All 8/8 scored
top choices and 4/4 generated IDs matched native; mean KL was 0.04101 and the
maximum was 0.15532.  The corresponding logical routed-byte saving is only
5.20%, because 85 of 92 routed layers remain exact.

The second-subject control used the frozen Japan-capital prompt.  Block ends
changed native wording: the candidate answered `Tokyo` immediately instead of
`The answer is Tokyo`.  On the nine shared prompt-trajectory rows, mean KL was
0.02895 and top choice agreement was 7/9.  The candidate remained correct and
non-degenerate, but was not token-exact.  KL after that history divergence is
not reported.

This is stronger than a generic “fewer approximated layers” result: block
starts flipped the aviation decision even though their mean KL was lower than
both ends and middles.  The phase changes which terminal margin directions are
perturbed.  It does not yet prove that block ends always attenuate error.

The next earned experiment is a small registered multi-prompt, one-token phase
screen.  Do not grow backward from block ends or call this an architectural law
until that replication passes.
