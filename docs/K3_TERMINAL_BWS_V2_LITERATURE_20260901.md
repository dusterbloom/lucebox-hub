# K3 terminal-BWS v2 literature sweep — 2026-09-01

## Scope and current evidence

This sweep asks only what should be tested next after the held-out layer-92
result at experiment commit
`afe57d6a540ba6ba09f3395f7be63348d111c0c3`:

- direct terminal labels can reduce held-out isolated-layer B24 KL from
  `0.1062990854` to `0.0607545120` at the same 14,278,656 logical bytes;
- that 42.8% KL reduction still selects the wrong top token;
- independent slab marginals interact non-monotonically;
- route/rank/residual metadata fails the preregistered held-out correlation
  gate.

The sweep is not evidence that any cited method works on Kimi K3. It is used
to choose the cheapest discriminating experiment. All papers below are
primary sources.

## Findings

### 1. Global KL curvature is the right geometry, but not yet the right implementation

[YAQA / Model-Preserving Adaptive Rounding](https://arxiv.org/abs/2505.22988)
constructs Kronecker-factored approximations of a layer Hessian with respect
to full-model KL rather than immediate layer reconstruction. It reports about
30% lower KL than conventional local objectives across quantizers. This
supports the K3 hypothesis that downstream output geometry matters.

[BaKron](https://arxiv.org/abs/2608.06291) and
[KronQ](https://arxiv.org/abs/2607.07964) make the same point from two-sided
curvature: input activation covariance alone treats output directions as
equally important, while gradient covariance distinguishes them. BaKron
reduces the adaptive-rounding work to GPTQ-like cubic complexity.

These are weight-rounding systems. They do not establish that a static
Kronecker sketch can rank prompt-conditioned K3 route/slab interventions, and
they do not solve the observed top-token failure. Building their complete
rounding machinery now would be premature. Their earned contribution is the
objective: a displacement should be evaluated in downstream KL/Fisher
geometry, not Euclidean local-output geometry.

### 2. The exact-screen result is activation patching, and first-order attribution has known failure modes

[AtP*](https://arxiv.org/abs/2403.00745) formalizes a cheap approximation to
activation patching as a directional derivative, `gradient dot activation
displacement`. It finds false negatives from downstream nonlinearity and from
cancellation between paths. It also warns that coarser interventions are less
linear and recommends exact verification of the candidates selected by the
cheap screen.

[When Attribution Patching Lies](https://arxiv.org/abs/2606.09899) attributes
the dominant first-order error to downstream network curvature. For scalar
metric `M` and patch direction `delta`, it uses:

```text
first order       = grad(M) dot delta
second order      = 0.5 * delta^T H delta
reliability ratio = abs(delta^T H delta) /
                    (2 * abs(grad(M) dot delta))
```

It proposes screen–flag–fix: screen with the first-order score, flag directions
whose quadratic term rivals the first-order term, and spend HVP work only on
those directions. A multi-step HVP is reserved for large patches where one
Taylor step overshoots.

This directly explains why adding more individually positive K3 slabs did not
monotonically improve the held-out margin. It also rejects a giant Fisher
framework as the first move: exact interventions already provide the gold
labels, so the approximation must earn its complexity against those labels.

### 3. Layer 92 offers a narrower experiment than full K3 backpropagation

K3 code inspection shows that the final routed layer is followed by:

1. the routed latent join, including a routed RMS normalization and up
   projection;
2. a final AttnRes mixture whose score for the current hidden is nonlinear;
3. final RMS normalization;
4. the vocabulary projection.

The downstream tail is not linear, but it is much smaller than the complete
93-layer graph. This makes a captured layer-92 tail discriminator scientifically
preferable to adding general K3 autograd.

[Does Circuit Analysis Interpretability Scale?](https://arxiv.org/abs/2307.09458)
shows why direct logit attribution through a residual stream can work when
the final RMS scale is nearly fixed, but explicitly notes that late-layer
ablation can violate that assumption. K3 must therefore record the RMS/AttnRes
change or validate a fixed-scale approximation; it cannot silently assume
linearity at the final layer.

### 4. KL and decision correctness need separate measured objectives

The K3 held-out result is stronger evidence than an abstract warning: mean
full-vocabulary KL fell by 42.8% while the top token remained wrong. A
distributional score is still primary, but a boundary-aware score is required
near small teacher margins.

The next scorer should therefore report at least:

- teacher-distribution Fisher/KL directional damage;
- teacher top-token versus candidate-contender logit-margin damage;
- exact top-1 after the equal-byte conditional group intervention.

The margin term is not a replacement for full-vocabulary KL. It prevents a
selector from spending its byte budget on many harmless vocabulary changes
while missing one structured-token or identifier boundary.

### 5. Progressive fidelity is supported, but entropy is not an adequate risk oracle

[FlexQuant](https://arxiv.org/abs/2506.12024) and
[Progressive Mixed-Precision Decoding](https://arxiv.org/abs/2410.13461)
support prompt/token-adaptive precision rather than one request-wide setting.
Their evidence is directionally relevant to B24 plus selective hydration, not
proof of a K3 policy.

The K3 tool boundary was confidently wrong at B24, so softmax entropy alone is
already falsified as a sufficient escalation signal. Cheap/richer disagreement,
predicted terminal damage, route instability, and grammar/tool state remain
eligible.

[DraftExpert](https://arxiv.org/abs/2607.24434) and
[EcoSpec](https://arxiv.org/abs/2607.12696) show that the union of newly loaded
experts must enter speculative-decoding decisions. That is a later systems
opportunity after the representation reaches the byte frontier; it does not
fix the present selector.

### 6. GSQ is the earned complement fallback; Trellis is not yet earned

[GSQ](https://arxiv.org/abs/2604.18556) optimizes discrete assignments while
retaining symmetric scalar group quantization and existing scalar decoder
structure. Its Kimi-K2.5 experiment quantizes non-shared routed experts from
about 4.5 to 2.13 bpw and largely preserves coding/reasoning, but GPQA drops
substantially. The paper attributes part of that gap to calibration-data
composition and shows that GSQ can improve existing GGUF K-quant assignments.

This supports an offline K3 experiment on the omitted BWS complement. It does
not justify a runtime kernel yet, and it warns that the calibration suite must
cover science, multilingual, tools, code, and structured output rather than
only reasoning/code.

[QTIP](https://arxiv.org/abs/2406.11235) demonstrates a stronger low-bit
Trellis frontier, but its decoder and representation are a larger systems bet.
It is earned only if an equal-byte captured-state proxy materially beats GSQ.

### 7. arXiv:2608.28444 is orthogonal to the first discriminator

The paper at [arXiv:2608.28444](https://arxiv.org/abs/2608.28444) concerns a
different attention/cache execution axis. It may inform a later KDA/MLA or
sliding-window state experiment, but it provides neither a routed slab value
function nor authoritative expert-byte reduction. It should not displace the
current terminal-selection test.

## Preregistered next decision

The shortest defensible order is:

1. **Existing-artifact boundary reanalysis.** From the completed 192-arm
   held-out logits, rank isolated equal-byte swaps by recovery of the exact
   teacher-versus-B24-contender logit margin. Run only small conditional groups
   (2/4/8 and the positive-gain crossover) at B24. These are mechanistic oracle
   results, never validation.
2. **Gate A.** If no margin-targeted equal-byte group recovers top-1, static
   B24 selection alone is not an adequate explanation of the boundary. Move
   next to an offline GSQ complement or progressive richer-pass rescue rather
   than building general autograd.
3. **Gate B.** If the margin-targeted group recovers top-1, implement the
   smallest captured layer-92 tail scorer. Compare first-order directional
   scores, Fisher/KL quadratic scores, and explicit curvature flags against the
   two complete 192-arm screens. Correct only flagged directions.
4. **Acceptance.** A prompt-conditioned scorer must beat the existing local
   ranking on held-out Spearman and equal-byte terminal KL, and must recover
   the boundary on a fixture whose teacher is valid. Otherwise it is a NO-GO.
5. Do not build a GSQ/Trellis HIP decoder, a general second-order framework, or
   a production selector before those gates pass.

## Trade-offs disclosed

- Margin-ranked groups use held-out labels and can only diagnose whether the
  representation contains the needed decision information; they cannot be
  called a deployable selector.
- Layer 92 is deliberately the easiest downstream tail. Passing there would
  justify, not replace, tests on earlier KDA/MLA neighborhoods.
- A Fisher quadratic is naturally aligned with infinitesimal KL around the
  teacher. K3 slab swaps are finite, so exact patches and curvature flags remain
  mandatory.
- No result in this document changes full-model bytes/token or serving speed.
- Research stays on `experiment/k3-terminal-kl-bws-v2`. A passing primitive is
  narrowly reimplemented and requalified on `perf/k3-production-ponytail`;
  neither research branch is merged wholesale.

## GLM-5.3-Flash adversarial modification

The requested read-only `zai-coding-plan/glm-5.3-flash` review returned
**MODIFY** in OpenCode session `ses_fa6051828ffeYAEJ6Ptf4g93p2`. Its binding
corrections are:

- exact terminal margin is an offline label or post-compute verifier, not a
  runtime selector, because it requires computing the omitted slab;
- a runtime-rankable proxy must use resident state and resident sidecar/mean
  information before any unranked authoritative slab is read;
- the teacher margin must first be non-degenerate;
- group choices must be frozen before their joint results are observed;
- no more than four planned full-model group runs are allowed, with a hard cap
  of six only for invalid/capacity-contended repeats;
- if no group recovers top-1 or makes the teacher-versus-contender margin
  positive, captured-tail/JVP work is not earned.

The teacher margin is `0.0924091339`, above the preregistered numerical floor
of `0.001`. The four frozen compositions and their exact targets are in
`results/k3_terminal_bws_v2_margin_groups_prereg_20260901.json`. They are
post-heldout mechanistic oracle tests, not validation.
