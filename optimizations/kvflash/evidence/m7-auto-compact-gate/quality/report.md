questions: modal 357, orig_engine 357, F_engine 357, missing 0

| arm | correct | accuracy |
|---|---|---|
| orig_engine | 314/357 | 0.8796 |
| F_engine (auto 0.10, P=0 W=0) | 302/357 | 0.8459 |
| orig_Modal | 319/357 | 0.8936 |
| F_Modal (F-UNI-LOCAL) | 309/357 | 0.8655 |

D_engine = F_engine - orig_engine = -0.0336 (-12 questions), 95% cluster bootstrap (20 articles, 10000 resamples, seed 0) [-0.0634, -0.0055]
D_Modal  = F_Modal - orig_Modal   = -0.0280, [-0.0519, -0.0028]

per-question agreement of the parsed choice: F_engine vs F_Modal 339/357 (0.950); orig_engine vs orig_Modal 343/357 (0.961)
agreement F_engine vs orig_engine 319/357; F_Modal vs orig_Modal 331/357

## Parity and safety checks
- orig_engine: prefix_len==0 and disk_hit false and restore false in 357/357; server prompt_tokens == HF tokens in 326/357; non-empty nan_layers in 0; unparsed answers 3
- F_engine: prefix_len==0 and disk_hit false and restore false in 357/357; server prompt_tokens == HF tokens in 326/357; non-empty nan_layers in 0; unparsed answers 0
- F_engine compaction: sealed blocks per prompt min/max 6/11; compacted==sealed in 249/357; failed blocks 0; skipped blocks 108
- sealed context blocks per question equal the reference's ctx_len//640 in 357/357

## Cost (server-side, per question)
- orig_engine: prefill_s median/min/max (5.1, 3.3, 6.0); decode tok/s median/min/max (76.2, 33.1, 79.6); wall_s (5.174840211868286, 3.3543334007263184, 13.276179552078247)
- F_engine: prefill_s median/min/max (8.2, 5.1, 10.0); decode tok/s median/min/max (84.2, 45.5, 85.0); wall_s (8.379040002822876, 5.1138081550598145, 13.88108777999878)
- compaction per sealed block (ms, total = select + fit + io + publish): median/min/max (313.0, 296.7, 382.3); select (46.525, 44.37777777777777, 55.279999999999994); fit (207.45999999999998, 200.95555555555555, 247.18); io (57.27, 50.03333333333333, 78.59)
