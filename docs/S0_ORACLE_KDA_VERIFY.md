# S0 — Oracle multi-token K3 verification ceiling

## Verdict

**STOP AT WIDTH FOUR.** The existing K3 target has a real causal multi-token
forward and ReplaySSM commit path. Width two is bit-exact and gives a measured
1.55x committed speedup on the frozen calibrated96/P27 path. Width four is not
numerically correct. The same width-four failure occurs with native streamed
experts, so it is in the shared K3 multi-token core rather than progressive
slab batching.

No width-eight run was performed after the mandatory correctness gate failed.
No speculative drafter work is earned yet.

## Frozen input convention

The shared speculative runtime passes the current accepted seed token as row
zero and subsequent proposed tokens as later rows. Each target row predicts the
token after its input row. S0 therefore used the frozen P27 trace:

- prompt token IDs: `18699` (`Hi`);
- verifier input IDs: `11,374,4936,261,814,2742,316,374`;
- source: `/mnt/kimi-k3/results/kimi-p27-direct-pinned-32-row-20260816/stdout.log`.

Both arms rebuild the prompt from a cleared cache. The sequential arm executes
one input row per K3 forward. The oracle arm executes the same rows in one
causal forward with recurrent-state writes disabled, then commits the accepted
prefix through ReplaySSM. The timers exclude prompt rebuild and state hashing.

## Results

| provider | width | sequential | verify | commit | committed V | logits | KDA/conv | MLA rows |
|---|---:|---:|---:|---:|---:|---|---|---|
| calibrated96/P27 | 2 | 7.113 s | 4.145 s | 0.454 s | **1.547x** | bit-equal | hash-equal | hash-equal |
| calibrated96/P27 | 4 | 13.139 s | 8.714 s | 0.510 s | 1.424x | **FAIL**, relL2 0.02329 | **FAIL** | **FAIL** |
| native streamed | 4 | 15.920 s | 9.876 s | 0.503 s | 1.534x | **FAIL**, relL2 0.02440 | **FAIL** | **FAIL** |

All width-four argmax rows happened to agree, but that does not satisfy the
state or logit parity gate. Maximum logit error was 0.6267 for calibrated96 and
0.4812 for native streamed experts.

Physical `/proc/self/io` deltas also show why verification amortization remains
interesting after correctness is repaired:

| provider/width | sequential reads | verify+commit reads |
|---|---:|---:|
| calibrated96 / 2 | 12.999 GB | 9.793 GB |
| calibrated96 / 4 | 22.726 GB | 21.481 GB |
| native / 4 | 39.469 GB | 26.113 GB |

These are operating-system physical-read counters, not logical provider or H2D
bytes. The existing P20 I/O and traffic traces remain authoritative for those
categories.

## Interpretation

The native control falsifies the hypothesis that calibrated96's token loop is
causing the width-four numerical divergence. The likely fault boundary is the
shared width-dependent KDA/causal-batch execution (not yet localized); K3 uses
a convolution width of four, making that boundary a useful first diagnostic.
That is a hypothesis, not a measured cause.

AttnRes has no persistent cross-token cache. Exact terminal rows cover its
accepted-boundary result. Persistent parity is checked separately for every
KDA convolution/state tensor and every newly written MLA cache row.

## Reproduction

Base revision: `17fc1e970b9f88e5dfb1167736f8d9f4355f9300`, plus the S0 harness in this
change.

Build (CUDA build parallelism is deliberately capped):

```bash
env CCACHE_DISABLE=1 cmake --build server/build-k3-p20-cuda126b -j4 \
  --target run_kimi_k3_s0_oracle
```

The runner is:

```text
run_kimi_k3_s0_oracle MODEL PROMPT_IDS ORACLE_IDS RESULT_JSON \
  GPU CORE EXPERT_GPU MAX_WIDTH MIN_WIDTH
```

Use the same P27 environment recorded in the P27 telemetry. `MAX_WIDTH=2` is
the passing gate. A gated `MAX_WIDTH=8 MIN_WIDTH=2` run now stops automatically
at the first parity failure and cannot silently proceed to width eight.

Authoritative artifacts and SHA-256:

- calibrated96 m2: `f7fd44f37f99294f2ab284e1616d14678c3bd7e39d92501f3dcf57417b243154`;
- calibrated96 m4: `33b2c28d9cf731c6377770f1465a3600db3b1f2907528fdded46a8c8c866842b`;
- native m4: `aa2cc22ea6fe64608ce02985d594f36a73a088557ff7852ab43b54f0af7793ab`.

## Next single S0 action

Localize the first sequential-versus-width-four divergence inside one KDA
layer, beginning with convolution inputs/outputs and state at the four-token
boundary. Do not build a drafter or run width eight until width four is exact.
