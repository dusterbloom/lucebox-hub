**The transferable idea is routing profiling. The reported speedups remove inter-GPU activation traffic; they do not reduce expert-weight reads. Your strongest opportunity is weight reuse during batched MTP verification.**

Read the [article](https://www.zyphra.com/our-work/expert-coupling-in-moe-pretraining), its [technical report](https://arxiv.org/html/2610.09372v1), and the directly linked [Megatron-LM paper](https://arxiv.org/pdf/1909.08053). No specific coupling implementation is linked. I also fetched the report’s [DeepEP](https://github.com/deepseek-ai/DeepEP) and [MoonEP](https://github.com/MoonshotAI/MoonEP) code references. No files were created or changed.

**Reasoning:** (1) Identify which correlations they measure. (2) Separate network activation bytes from your DRAM weight bytes. (3) Require measured prediction accuracy or temporal reuse before implementing anything.

**1. What expert coupling means**

Coupling means correlated *selection*, without implying shared parameters or interchangeable outputs. They measure:
- Within-layer co-selection counts \(W_\ell(i,j)=\sum_t1[i,j\in S_{\ell,t}]\), and conditional probabilities \(P(j\in S_\ell\mid i\in S_\ell)\).
- Cross-layer conditionals \(P(j\in S_{\ell+1}\mid i\in S_\ell)\); their planner also conditions on routes from two preceding layers.
- Placement outcomes: distinct destination GPUs per token, duplicate activation rows removed, and the fraction of selected experts local to the token. [Report, §§II/V](https://arxiv.org/html/2610.09372v1#S2.SS2).

Their models have **128 experts, top-2/top-6 routing, 12 layers**, and train for **2.1B tokens**:
- Layer 8, top-2: **64/8,128 pairs (0.8%) serve 42% of tokens**, versus **1.6%** under independent routing with matching loads.
- Layers 8→9: **112/128 experts** have a next-layer partner selected by at least **30%** of their tokens; the median strongest-partner probability is **48%**. This is not full-set prediction accuracy. [Report, §II-B](https://arxiv.org/html/2610.09372v1#S2.SS2).
- With **4,096 sampled tokens**, matrix Pearson correlations reach **0.93/0.98** for top-2/top-6; placement/prediction results are within approximately **one percentage point** of using 524k tokens.
- Tables from training step **199** deliver roughly half the final benefit; step **1000** is within **2–3 points**, and step **1400+** is nearly final. These demonstrate stable statistical relationships across samples and training—not identical routes on consecutive tokens or identical coupling in every layer. [Article](https://www.zyphra.com/our-work/expert-coupling-in-moe-pretraining).

They change **placement and dispatch**, preserving router decisions and weights: colocate correlated experts, send each activation once per destination GPU, and predict token ownership inside existing sequence-parallel collectives. At EP8, duplicate-row removal rises **6%→26%** for top-2 and **25%→58%** for top-6; token shuffling raises top-2 locality **12.5%→59%**. Maximum speedups are **2.63× for all-to-all** and **1.41× overall training**. [Article](https://www.zyphra.com/our-work/expert-coupling-in-moe-pretraining).

**2. Transfer to your inference**

Your supplied numbers imply **26.16 ms** of weight streaming and a **12.24 ms** gap to 38.4 ms/token. That gap includes computation, dequantization, launches, and bandwidth inefficiency. Keeping 6.33 GB/token imposes an ideal speedup ceiling of **1.47×**.

Let \(R\) be routed-expert weight bytes/token and \(D=6.33-R\) GB cover shared experts and other weights. Measure this split from actual quantized tensor sizes.

| Idea | Bytes/token | Launches/token and practical implication |
|---|---|---|
| Predict/prefetch next-layer experts | **No compulsory-byte reduction.** Wrong predictions add reads. Predicting 10 with 8 correct adds approximately **0.2R**, assuming correct prefetched weights survive until use. | Same expert computation; potentially extra planner/prefetch launches. Useful only when reads overlap otherwise idle time and cache retention works. |
| Put coupled experts contiguously | **No compulsory-byte reduction.** Ten distinct FFNs still require ten weight sets. | No launch reduction by layout alone. Possible translation/streaming improvement requires a benchmark; strong coupling alone proves nothing about DRAM bandwidth. |
| Fuse coupled expert GEMVs | **No weight-byte reduction.** | If currently launched separately, grouping all ten saves **432 launches/token per projection**: \(48(10-1)\). Already grouped kernels give **zero** savings. Coupling is unnecessary for dynamic grouping. |
| Reuse across batched verification | **Potentially substantial reduction**, determined by each layer’s expert union and actual kernel reuse. | Group tokens by expert and process them together; separate GEMVs reread weights. Sequential drafts get no automatic saving. |
| Expert caching | **No benefit from duplicating resident weights elsewhere in unified RAM.** A faster-cache hit could save bytes. | No intrinsic launch reduction. Between adjacent-token uses of the same layer, the intervening stream is approximately **6.33 GB**—far beyond GPU cache capacity. |

For fusion, two/three separate projection stages would save **864/1,296 launches**. At an illustrative *measured* 2 µs critical-path cost each, that is **1.73/2.59 ms**. The shared expert remains separate in this accounting; hc=4 does not multiply the 480 routed selections.

For a verification window of \(w\) tokens, compute \(U_\ell(w)=|\bigcup_{t=1}^{w}S_{\ell,t}|\). With expert byte sizes \(b_{\ell,e}\), the ideal weight traffic is
\[
B_{\rm window}=D+\sum_\ell\sum_{e\in\cup_tS_{\ell,t}}b_{\ell,e}.
\]
This assumes each tensor streams once per window; actual traffic can be higher. With equal expert sizes, routed traffic per verified token becomes \(R\bar U(w)/(10w)\).

For **w=4**, mean unions **20/24/30** imply **50%/40%/25% routed-weight savings**. Saving 25% of each **1 GB of R** removes **0.25 GB/token**, equivalent to **1.03 ms** at 242 GB/s.

Uniform independent routing gives unions **19.80/38.84/74.74** for windows **2/4/8**: only **0.98%/2.89%/6.58%** routed reuse. Critically, **within-token coupling alone does not lower expected union at fixed expert marginals**; temporal persistence or concentrated expert popularity does.

For speculation, divide window traffic by actual emitted tokens \(a\), and include drafting and rejection recovery: require **\((T_{\rm draft}+T_{\rm verify}+T_{\rm recovery})/a<38.4\) ms**. Routed bytes alone improve only when **\(\bar U<10a\)**. MTP-head and target experts can reuse weights only when they reference the same tensors. GDN state handling, QSA attention, and hc execution also affect realized latency.

**3. Cheap measurements and decision gates**

Capture **256 decode tokens × 48 layers × 10 IDs**: **240 KiB** using uint16. Buffer on-device and copy once after the run; time performance separately. Keep layer identities distinct and exclude the always-active shared expert.

| Measurement | Exact statistic | Suggested decision gate |
|---|---|---|
| Co-selection and stability | Compute \(W\), \(n_i=\sum_t1[i\in S_t]\), and \(W(i,j)/n_i\). Rank pairs on a fit split; on held-out tokens measure their pair mass \(\sum W/(N\binom{10}{2})\). Compare with a fixed-top-10, load-preserving randomization; correlate off-diagonal matrices across splits. | Stable excess co-selection identifies groups. Layout still needs ≥5% measured MoE-time improvement; fusion should depend on the launch trace. |
| Next-layer prediction | Compute \(A_\ell(i,j)=\sum_t1[i\in S_{\ell,t},j\in S_{\ell+1,t}]/n_{\ell,i}\). Rank \(j\) by \(\sum_{i\in S_{\ell,t}}A_\ell(i,j)\); evaluate held-out top-10 recall and byte-weighted misses against a popularity-only predictor. | Rough initial gate: **≥90% byte recall**, plus retained prefetched tiles and latency savings exceeding prediction, wasted-read, and cache-pollution costs. Fit each GDN/QSA transition separately. |
| Verification reuse | For windows 2/4/8, compute unique bytes and \(r_w=\sum_{\ell,e\in\cup S}b_{\ell,e}/\sum_{t,\ell,e\in S_t}b_{\ell,e}\). Compare consecutive windows with shuffled-token windows preserving the original routing sets. | **\(r_4≤0.75\)** is promising if four tokens are useful; then measure actual draft-window routes, DRAM traffic, and acceptance. |
| Caching | Compute byte reuse distances for identical layer/expert/tensor keys, including intervening \(D\) streams; simulate the actual faster-cache capacity. | Proceed only if capacity supports meaningful hits and measured DRAM bytes fall. Repeated IDs alone are insufficient. |

The paper’s destination statistic is \(|\{\mathrm{GPU}(e):e\in S_t\}|\), not a temporal expert union. On your GPU it is always **1**. Also, 256 tokens average only **five observations/expert/layer** under uniform loads: adequate reconnaissance for unions, sparse evidence for conditional tables. Expand to 1–4k tokens and held-out prompts if promising.

**Ranked worth trying / not worth it**
1. **Worth trying:** measure verification-window unions; implement expert-major batched reuse if bytes and acceptance justify it.
2. **Worth trying conditionally:** grouped GEMV fusion, if the launch trace shows separate expert launches.
3. **Low priority:** next-layer prefetch, after held-out prediction and cache-retention evidence.
4. **Usually not worth it:** physical expert regrouping without measured streaming/translation improvement.
5. **Not worth it for resident serial decode:** a unified-RAM expert cache; their distributed token-shuffling optimization has zero network traffic to remove here.

**Recommended first experiment:** capture 256 greedy-token routes and report byte-weighted consecutive versus shuffled unions for windows 2/4/8, per layer and overall. Treat this as the accepted-path opportunity estimate; if \(r_4≤0.75\), repeat on a second prompt and actual MTP verification windows.