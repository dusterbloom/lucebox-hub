# Panel Approximation of Stable LatentMoE — v2, Corrected Against the K3 Tech Report

arXiv:2607.24653 resolves all three assumptions v1 was resting on. **Two of my
results were wrong, both in the direction that favours the idea.** One number in
the original claim is now provably impossible for K3, by arithmetic rather than
extrapolation.

---

## §0 What the report actually says

Table 1 of the report, with my v1 guesses beside it:

| quantity | v1 assumed | **actual** |
|---|---|---|
| layers | 61 | **93** (69 KDA + 24 MLA), 1 dense → **92 MoE layers** |
| hidden $d$ | 7168 | **7168** ✓ |
| latent $\ell$ | 3584 | **3584** ✓ (exactly $0.5d$) |
| expert hidden | — | **3072** |
| routed / active | 896 / 16 | ✓ |
| shared experts | — | **2, full width** |
| activation | SwiGLU | **SiTU-GLU**, bounded by $\beta_1\beta_2=100$ |
| residual | $h_{\ell+1}=h_\ell+\Delta_\ell$ | **Block AttnRes**, 8 blocks of 12 |

Equation 11 of the report gives the block exactly:

$$u = \sum_{i \in T_k(x)} p_i\, E_i^{\text{routed}}(W_{\downarrow}x), \qquad
y = \sum_{j=1}^{2} E_j^{\text{shared}}(x) + W_{\uparrow}\,\mathrm{RMSNorm}(u)$$

with $W_\downarrow : \mathbb{R}^d\!\to\!\mathbb{R}^\ell$, $W_\uparrow : \mathbb{R}^\ell\!\to\!\mathbb{R}^d$,
$E_i^{\text{routed}} : \mathbb{R}^\ell\!\to\!\mathbb{R}^\ell$.

**Verified parameter split.** Routed expert $= 3\ell h = 33.03$M each;
$896 \times 33.03\text{M} \times 92 = 2.723$T $= 97.9\%$ of 2.78T. Everything else
is 57.3B. Both match the derivation from published totals in v1 — the accounting is closed.

---

## §1 Theorem 1 is confirmed, with equality

> **Theorem 1 (exact, no longer conditional).** $W_\uparrow$ is shared across all 896
> routed experts in a layer. Therefore for all inputs,
> $$\Delta_{\text{routed}} \in \operatorname{col}(W_\uparrow), \qquad \operatorname{rank} \le \ell = 3584 = d/2 .$$
> The optimal panel is $D = W_\uparrow$, **read off the weights, zero calibration,
> zero error** — and it is already stored in high precision, since §4.1.4 keeps
> latent MoE projections out of MXFP4.

The "surprising low-dimensional subspace" is $W_\uparrow$. It was never hidden.

> **Theorem 1′ (RMSNorm kills the magnitude).** Because the block applies
> $\mathrm{RMSNorm}$ between aggregation and up-projection,
> $$\Delta_{\text{routed}} = W_\uparrow\big(\gamma \odot u/\|u\|\big)\sqrt{\ell}$$
> depends on $u$ **only through its direction**. Any surrogate $\hat u = c\,u$,
> $c>0$, is exact.

*Consequence for the objective.* Least squares on $u$ is the **wrong loss** — it
spends capacity on a degree of freedom the architecture discards. The correct
recovery problem is a Rayleigh quotient,

$$\max_{\hat u(\cdot)} \; \mathbb{E}\left[\frac{\langle \hat u, u\rangle}{\|\hat u\|\,\|u\|}\right]
\quad\text{not}\quad \min\mathbb{E}\|\hat u - u\|^2 ,$$

which for a linear $\hat u = Wz$ is a generalized eigenvalue problem in the same
Gram matrices — still closed-form, still one pass. One free dimension out of 3584
is negligible; what matters is that **scale error is free**, so the shrinkage
tuning of v1 §4.1 is unnecessary on the routed branch.

---

## §2 Correction 1: Block AttnRes destroys the depth argument (in your favour)

v1 assumed $h_{\ell+1} = h_\ell + \Delta_\ell$ and derived catastrophic
amplification. K3 does not do this. From §2.2 of the report, layer $l$ forms

$$h_l = \sum_{i=0}^{l-1}\alpha_{i\to l}\, v_i, \qquad \sum_i \alpha_{i\to l}=1,$$

a **softmax-weighted convex combination** of preceding layer outputs, with
RMSNorm on the keys. Block AttnRes sums within blocks of $S=12$ and attends
across $N=9$ block representations.

> **Theorem 5 (revised).** Under Block AttnRes with sequential calibration
> (errors conditionally mean-zero given the state):
> * **Intra-block**, over $S=12$ summed layers, errors add incoherently:
>   $\|\varepsilon_{b_n}\| \approx \sqrt{S}\,\bar\varepsilon$, while
>   $\|b_n\| \approx \sqrt{S}\,\|\bar f\|$ — **relative error is preserved**, not amplified.
> * **Inter-block**, $\delta h_L = \sum_n \alpha_n \varepsilon_{b_n}$ with $\sum\alpha_n = 1$.
>   For independent errors, $\|\delta h_L\|^2 = \sum \alpha_n^2\|\varepsilon_{b_n}\|^2$
>   and $\|h_L\|^2 \approx \sum\alpha_n^2\|b_n\|^2$, so again the ratio is preserved.
>
> Net amplification $G \approx 1$, versus the $G \approx 278$ that a plain
> depth-92 residual stream would give at $\lambda = 0.05$.

| architecture | naive | sequential |
|---|---|---|
| plain residual, depth 92 | 1760× | 278× |
| **Block AttnRes (K3)** | ~12× | **~1×** |

**This is the single most important correction.** Convex mixing is a normalizer:
it cannot amplify a weighted average of bounded errors. The per-layer fidelity
budget for 2% end-to-end drift moves accordingly:

$$\text{v1 (wrong): } R^2 \ge 0.99999957 \qquad\longrightarrow\qquad
\boxed{\text{v2: } R^2 \ge 0.9996}$$

**Four orders of magnitude easier.** This changes the verdict from "requires a
miracle" to "requires a good but not unprecedented fit."

Two caveats that keep this honest: the $\alpha_{i\to l}$ are themselves
data-dependent softmax weights, so there is a feedback loop of the same kind as
the router (perturb $h$, shift the depth-attention, change which layers are
read); and the argument assumes incoherent errors, which is exactly what
sequential calibration buys and naive calibration does not. Sequential
calibration remains mandatory — it is now what makes $G\approx1$ true rather than
merely what reduces $G$ from 1760 to 278.

---

## §3 Correction 2: SiTU-GLU and Quantile Balancing retire two of my risk flags

**Bounded activations.** SiTU-GLU satisfies $|f(x)| \le \beta_1\beta_2 = 100$ by
construction (§2.3.2), so routed-expert outputs are bounded. The
massive-activation / outlier-feature risk from v1 does not apply to the routed
branch, and unweighted SVD will not have its rank hijacked by a few dimensions.

**Uniform expert load.** Quantile Balancing (§2.3.3) sets each expert's bias from
the router-score quantile matching its target load $q = mk/n$, and the bias is
frozen at inference. Load is therefore near-exactly uniform: each expert sees
$16/896 = 1/56$ of tokens. The "rare but critical expert" risk from v1 is largely
retired — there is no long tail to destroy.

**Sample complexity, corrected.** Uniform load means $N/56$ calibration tokens
per expert. A *dense* per-expert map would need $N \gtrsim 4$M tokens. But the
diagonal parameterisation below decouples into $\ell$ independent two-parameter
univariate regressions per expert, each using all $N/56$ samples, so
$N/56 \gtrsim 30$ suffices for stability: **$N \approx 50$k–150k tokens**, an
afternoon of forward passes. The diagonal structure is not just cheap in
parameters, it is cheap in data.

---

## §4 The K3-native surrogate

Everything needed is already resident in high precision. $z = W_\downarrow h \in \mathbb{R}^\ell$
is computed anyway; the router is kept; $W_\uparrow$ is the panel. So replace only
$E_i^{\text{routed}}$:

$$\hat u = \sum_{i\in T_k(h)} p_i\big(v_i + s_i \odot z\big), \qquad
\hat\Delta = W_\uparrow\,\mathrm{RMSNorm}(\hat u)$$

Each expert becomes a **codeword plus a diagonal gain on the shared latent
projection**: $2\ell = 7168$ parameters, against 33.03M for the real expert —
**4608×**.

> **Theorem 4″ (rank ceiling, K3 form).** With codewords alone ($s_i=0$),
> $\operatorname{span}\{u\} \subseteq \operatorname{span}\{v_1..v_{896}\}$, so rank $\le 896$.
> The diagonal term contributes $(\sum_i p_i s_i)\odot z$, which spans all $\ell = 3584$
> dimensions. The diagonal gain is therefore not a refinement — it is what removes
> the rank ceiling, at a cost of 3584 parameters per expert.

**Budget.**

| | params | bf16 |
|---|---|---|
| codebook $v$ + gains $s$, 92 layers | 591 M | **1.18 GB** |
| replaces routed experts | 2.723 T | 1361 GB @ MXFP4 |
| ratio on the routed branch | | **1152×** |
| irreducible remainder ($W_\downarrow, W_\uparrow$, routers, **shared experts**, attention, embeddings) | 57.3 B | 115 GB bf16 / **57 GB fp8** |

> **Theorem 7 (the 12.9 GB figure is impossible for K3).** The method compresses
> only $E_i^{\text{routed}}$. The remaining 57.3B parameters include two
> **full-width** shared experts per layer (132.1M/layer, 12.2B total) which are
> dense FFNs, not experts, and which §4.1.4 deliberately keeps out of MXFP4. At
> 1 byte/param this floor is **57 GB**. No panel, at any $k$, at any fidelity,
> reaches 12.9 GB on Kimi K3. $\square$

**The honest headline is 58–116 GB**, i.e. **5–10× smaller than Unsloth's 594 GB
UD-IQ1_S**, not 46× smaller. That still moves K3 from a two-machine cluster onto
a single 128 GB Mac Studio — which is the outcome that actually matters — but it
is a different claim from the one in the video.

---

## §5 What the audit now says

The v1 rate–distortion comparison stands as arithmetic but its force is reduced,
because §2 removed the depth catastrophe that made the required fidelity absurd:

| | bits per routed-expert param | measured / required |
|---|---|---|
| Unsloth UD-IQ1_S | 1.697 | 78.9% top-1 |
| Unsloth UD-Q2_K_XL | 2.460 | 90.4% top-1 |
| this surrogate (1.18 GB) | 0.0035 | $R^2 \ge 0.9996$ needed |

The surrogate still stores ~485× fewer bits per routed-expert parameter than the
best calibrated quantiser. It survives that comparison only if the
distribution-specificity argument (v1 §5.3) is worth ~three orders of magnitude —
and it now has three structural advantages the generic panel didn't:
$W_\uparrow$ is exact and free, magnitude error is free, and depth is
non-amplifying.

**I would not bet on $R^2 \ge 0.9996$ from a diagonal map.** But it is now a
plausible-enough target that measuring beats arguing, and the measurement is
cheap.

---

## §6 The experiment, restated for K3

One MoE layer, ~100k tokens, no full-model forward pass needed:

1. Load layer $\ell$'s $W_\downarrow$, router, and the 896 routed experts from the
   GGUF. Stream $z = W_\downarrow h$ and $u = \sum p_i E_i(z)$.
2. Fit per-expert $(v_i, s_i)$ by univariate least squares — closed form, $\ell$
   independent 2-parameter fits per expert.
3. Report **$\mathbb{E}[\cos(\hat u, u)]$**, not MSE (Theorem 1′).

| $\mathbb{E}[\cos]$ | verdict |
|---|---|
| $\ge 0.9998$ | $R^2 \ge 0.9996$ cleared. Build it. |
| 0.99–0.9998 | Too lossy for full replacement; excellent for speculative fetch (v1 §6.2). |
| $< 0.99$ | Diagonal is insufficient; go to rank-$m$ per expert and re-cost. |

Ladder if the diagonal fails: $E_i(z) \approx v_i + B_i A_i z$ with
$\operatorname{rank} m$. Cost per expert $2m\ell$; at $m{=}8$, 57k params/expert
(576× compression, 9.4 GB total); at $m{=}64$, 459k (72×, 76 GB) — at which point
you are back in the regime the published literature occupies and below Unsloth
only by attacking the shared-expert floor, which this method cannot touch.

---

## Revision log vs v1

| v1 claim | status |
|---|---|
| A3 (shared $W_\uparrow$) assumed | **confirmed exactly**, Eq. 11 |
| $L = 61$ | **wrong**, 93 layers / 92 MoE |
| Amplification 278× at depth, sequential | **wrong** — Block AttnRes gives $\approx 1\times$ |
| $R^2 \ge 0.99999957$ required | **wrong** — $R^2 \ge 0.9996$ |
| Massive-activation risk | **retired** — SiTU-GLU is bounded |
| Rare-expert risk | **retired** — Quantile Balancing forces uniform load |
| MSE objective | **superseded** — RMSNorm makes it a Rayleigh quotient |
| 12.9 GB target | **provably impossible**: 57 GB floor from shared experts + projections |
| Panel ≈ constant per expert (Thm 4′) | **stands**, now with the diagonal fix that removes the rank ceiling |
