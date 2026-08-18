# P34 — isolated KDA accelerator ceiling

## Verdict

**The KDA kernel is accelerator-friendly, but splitting each selected layer at
the CPU/GPU boundary is a measured NO-GO.** The split runtime improves the
frozen transition rate only 2.98%, below the preregistered 5% continuation
gate, and is not retained as a production mode.

## Isolated control

The benchmark copies exactly one complete recurrent KDA layer (14 tensors,
253,542,272 bytes in the P32 core), uses the same deterministic one-token
normalized hidden vector and zero recurrent state on both backends, and times
graph creation, allocation, input, compute, and output. It does not change the
model or runtime placement.

| model layer | CPU ms | RTX 3090 ms | speedup | relative L2 | cosine |
|---:|---:|---:|---:|---:|---:|
| 1 | 7.095 | 1.408 | 5.040x | 0.02329 | 0.999730 |
| 40 | 7.075 | 1.657 | 4.269x | 0.01504 | 0.999887 |
| 73 | 7.291 | 1.600 | 4.557x | 0.01870 | 0.999826 |
| 88 | 7.303 | 1.564 | 4.671x | 0.01418 | 0.999900 |

The arithmetic difference is expected CPU-versus-CUDA quantized reduction
sensitivity. These local cosines are diagnostics, not a quality claim.

A second CPU backend copy of layer 88 was bit-identical and measured
6.778 ms versus 6.795 ms mapped. Therefore an ordinary CPU copy does not
activate a useful hidden repack path on this machine.

## Integrated stop

The smallest capacity-safe production experiment placed the final 18
recurrent layers on the RTX 3090. Their weights occupy 4.250 GiB; peak VRAM was
20,539 MiB. It deliberately kept the P32 core, calibrated 1.22-GiB slab policy,
mean tail, exact fallbacks, CPU router, latent/shared placement, and native
expert math fixed.

| metric | P32 reference | late-18 split |
|---|---:|---:|
| median transition | 1,328.166 ms | 1,289.787 ms |
| transition rate | 0.7529/s | 0.7753/s |
| change | — | **+2.98%** |

The fixed 24-token code continuation diverged at token 14 and ended one token
too early to complete its modulus predicate. Aligned mean/p95/max KL was
0.0446/0.2543/0.6735 with 62/66 top-one agreement. This one truncated output
does not prove broad quality failure, but the sub-5% systems result already
closes the split implementation before a broad suite.

The cause is visible in the stage trace: a selected layer needs a CPU graph to
produce the normalized KDA input, a CUDA KDA graph, state copies, and a second
CPU graph for the downstream residual/router boundary. Those crossings consume
most of the isolated kernel gain.

## Decision

- Keep the isolated benchmark as a hardware ceiling tool.
- Do not ship or broaden the split KDA runtime.
- Do not allocate a larger KDA bank under the same split design.
- A future accelerator attempt must execute the selected layer preparation as
  one graph, including its residual mixing and existing accelerator MoE core,
  then pass only the already-required expert boundary to the host.

Machine-readable evidence is in
`results/k3_p34_kda_offload_ceiling.json`; the raw integrated artifact is
`/mnt/kimi-k3/results/kimi-k3-p34-kda-offload-late18-code-20260818`.
