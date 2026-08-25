# K3 X8 baseline reconciliation

## Verdict

Lane B's performance verdict is suspended. X8 did not introduce the AR
regression. The regression was already present in X7's automatic
`exact-m1024` profile.

The automatic profile replaced the qualified scalar budget of 24 slabs with
96 slabs and ran width-8 prefill. On the same 53-token P55 `code-function`
prefix and the same 23 true AR transitions, that changed the whole-request
traffic and result as follows:

| Runtime | Physical bytes | Logical bytes | Physical GB / transition | AR tok/s | Traffic | Output |
| --- | ---: | ---: | ---: | ---: | --- | --- |
| retained P30 closure | 37,039,996,928 | 96,217,497,600 | 1.610435 | 2.129991 today | `8475c234...` | qualified 24 IDs |
| clean X7 production default | 203,778,605,056 | 374,433,669,120 | 8.859939 | 1.073751 | `76147313...` | diverged at output position 14 |
| X8 binary, oracle disabled | 203,776,557,056 | 374,433,669,120 | 8.859850 | 1.022236 | `76147313...` | identical to X7 |
| clean X7 with P30 policy | 37,039,996,928 | 96,217,497,600 | 1.610435 | 2.332025 | `8475c234...` | qualified 24 IDs |
| corrected server default | 37,039,996,928 | 96,217,497,600 | 1.610435 | **2.355260** | `8475c234...` | qualified 24 IDs |

The byte counters cover the whole request: 53 prompt positions plus 23 decode
transitions. Dividing by 23 preserves the convention in the reported P30 and
X8 comparison. Because every matched arm uses the same prefix and transition
count, prompt amortization cannot explain the difference.

## First changed mechanism

The first divergence occurs at provider initialization, before cache or GPU
execution:

```text
P30: requested-budget=table:...h23_moonshot_1_2gib.txt
X7:  requested-budget=96
```

The H23 table has 92 entries and every entry is 24. The X7 profile therefore
quadrupled requested logical slab records. Measured logical traffic rose
`3.8915x`; P40 and macro-union cache behavior then raised physical traffic
`5.5016x` on this matched request.

X7 and X8-disabled produced byte-identical traffic and final logits. X8 is not
the source of this regression.

## Second changed mechanism

Using budget 24 for scalar rows while retaining width-8 prefill recovered AR
to 2.116 tok/s but still changed the generated stream late. Width-8 prefill
had only been compared with the same explicit wide profile in the prior
production-default gate. It had not been compared against the scalar P55/P30
causal state across subsequent decode.

Therefore wide prefill is retained as an explicit experimental profile, not
the unconditional exact server default. It must pass scalar-reference state
and logit parity before it can be promoted again.

## Production correction

The automatic profile now resolves to `exact-scalar`:

- scalar budget 24, or the operator-supplied H22 table;
- prefill width 1;
- P30 16-GiB borrowed-record cache;
- P40 and macro-union disabled;
- P41/P42/P45/P46 retained.

The corrected automatic profile needs only the calibrated aux and sidecar
asset paths. It reproduced the P30 traffic hash, all 24 output IDs, and the
captured X7-P30 final-logit hash at 2.355260 true AR tok/s.

The scalar runner does not emit a whole-request expert-wall counter. This
report records direct-provider wall and P45 device-window counters without
mislabeling either as expert wall. The older P55 broad expert timing is not
substituted into this different run.

Machine-readable counters, full launch environments, source/binary hashes,
and retained artifact hashes are in
`results/k3_x8_baseline_reconciliation.json`.
