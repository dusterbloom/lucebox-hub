# H22 — behaviorally priced layer-adaptive progressive slabs

STATUS: BEHAVIORAL ATLAS MEASURED / HELD-OUT COMPOSITION IN PROGRESS

Uniform calibrated96 is an archived control and will not be rerun to select
this policy.  H22 asks whether the same progressive representation becomes
materially better when exact bytes are moved from behaviorally tolerant layers
to behaviorally sensitive layers.

## Allocation calibration

The existing paired K3 runner loads the model once.  It then sweeps model
layers 1–92 on the registered `raw-hi` stream.  At each row only the named
routed layer uses the frozen calibrated selector at budget 96; all other routed
layers and the paired teacher pass are exact.  The paired cache is restored, so
the intervention always receives the native trajectory.

For layer `l`, terminal teacher-to-candidate KL at budget 96 defines a
behavioral weight.  The already frozen local calibration supplies an omitted
residual proxy `q_l(b)` at budgets 48,72,96,120,144,168,192.  Before any
end-to-end result is observed, the planning cost is fixed as

```text
cost_l(b) = KL_l(96) * (q_l(b) / q_l(96))^2.
```

An exact dynamic program emits three immutable 92-row tables with average
nominal budgets 96, 120 and 144.  These are projections, not quality claims;
errors from different layers need not add independently.

## Held-out decision

The first end-to-end candidate is the average-96 behavioral table because it
tests allocation at the same nominal slab count as the archived uniform
control.  It is run on-policy on two held-out subjects with native exact as the
teacher.  If it clearly fails, the average-120 table is the one predeclared
escalation; no atlas refitting is allowed.  Average 144 is retained as the
safety frontier.

Report generated text, full-vocabulary terminal KL on aligned histories,
top-one agreement, exact fallback decisions, logical and physical provider
bytes, and actual decode time.  A table is not accepted merely because its
isolated-layer prediction is favorable.

## Limits

- The 2,048-token per-layer calibration remains a pilot with exact fallbacks.
- The one-stream atlas can discover large sensitivity differences but cannot
  certify cross-domain robustness.
- AttnRes phase is observed metadata, not a hard allocation rule.
- The exact teacher path, slab order, means and fallback threshold do not
  change in H22.

## Measured atlas

All 92 paired rows completed from one model load.  The exact teacher trace was
byte-identical across every repetition, and all 92 isolated budget-96
interventions retained the same top token.  Terminal KL nevertheless ranged
from `0.000398` at layer 88 to `0.014350` at layer 92, a roughly 36x range.
Layers adjacent in depth often differed by an order of magnitude, and locally
difficult layers 12 and 24 were behaviorally quiet on this stream.

The frozen average-96 allocation is:

| budget | layers |
|---:|---:|
| 48 | 8 |
| 72 | 27 |
| 96 | 28 |
| 120 | 15 |
| 144 | 14 |

At the same nominal total as uniform 96, the preregistered additive cost model
falls from `0.32334` to `0.22961`, a projected reduction of about 29%.  This is
not an end-to-end quality claim.  The immutable table SHA-256 is
`907c3a37026f3a2803e10e872f6cefe0c434b94acf8e23f450575f2bd877c8e4`.
