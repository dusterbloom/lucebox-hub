# K3 experiment scratchpad — 2026-09-01

## Decisions to retain

- Run the route12/Budget16 V16 target-only oracle before building a real
  draft or rescue runtime. The gate is `<= 1.55 s` for the 16-row target;
  `1.55–1.75 s` is only a retained optimization lead.
- Freeze the candidate IDs from a scalar route12/Budget16 run. Do not compare
  the V16 target against a native trajectory after the approximate policy has
  diverged.
- Require all 16 target argmax rows, the next scalar continuation, terminal
  full-vocabulary logits, and the complete committed recurrent/MLA state to
  match the scalar control.
- Test B16→B20 incremental hydration only after the V16 timing and correctness
  gate earns it. Do not implement a broad progressive subsystem first.
- Separately screen the existing generic chunked delta-net helper at M64. Its
  prior Qwen state drift is a preregistered exactness risk. A failure closes
  only this helper inside the current Core8-grouped executor, not a true
  prompt-wide DPLR/FlashKDA implementation.
- Keep `perf/k3-production-ponytail` at
  `fac048c090c74e5f8f989bffcda3aadc0bc8c266` untouched. A narrowly passing
  primitive still needs production requalification before transplant.

## Current provenance

- Experiment branch: `experiment/k3-terminal-kl-bws-v2`
- Committed experiment head before this patch:
  `85727b24535eb22e8bcf977a506cf898c820d872`
- V16 preregistration SHA-256:
  `af210eaa8365af576b74743118943d96616cd5ed4df8c4479f1641ceeb8b0dd6`
- Chunked-KDA preregistration SHA-256:
  `d89d71b99cee4f46b1a82ae1c4d32aaedd4eb1b034ca0785186e3e8d4264df06`

## Verification status

- `git diff --cached --check`: PASS.
- Modified backend, graph, and calibrated-provider translation units: syntax
  compile PASS using the available local CUDA headers with only the existing
  HIP-only `cudaDeviceProp::gcnArchName` field mapped to CUDA's `name` field.
- Qualified HIP build: PENDING on Lucebox4.
- Lucebox4 measurement: NOT STARTED.

## External blocker

The local Tailscale daemon disappeared during the turn and later restarted.
The local client is now healthy enough to reach another direct peer, but
Lucebox4 still has no handshake or return traffic: Tailscale ping, Tailscale
userspace `nc`, and direct SSH all time out. No GPU work started. Resume after
Lucebox4 itself is online/reachable, then first verify whether the staged patch reached
`/home/duster/k3-terminal-kl-bws-v2-guarded` before applying it again.
