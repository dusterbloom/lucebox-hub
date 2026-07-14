# Vendored llama.cpp/ggml snapshot

This directory contains the ggml-only subset used by Lucebox Hub.

- Source repository: https://github.com/Luce-Org/lucebox-ggml
- Source base branch: `luce-dflash`
- Source base commit: `6fbe72d67069136bbd370be703e1d4f441b5e942`
- Included merged PR: `#35` (`0fe65d9354b7c5da52a7741d2e37ba85f0d0c925`)
- Included test PR: `#37` (`0699be81480428f01b9b7ac49a09a2d51c77f8df`)
- Reconstruction: `luce-dflash@6fbe72d67069136bbd370be703e1d4f441b5e942` plus cherry-picked PRs `#35` and `#37`
- Focused CUDA Q1_0 import: upstream llama.cpp PR `#21629` at
  `7e72b38bc186deeee41b4d518a42bb50e1d0ba36` (type traits, conversion,
  dequantization, GET_ROWS, MMVQ/MMQ dispatch, and template instance only)
- Q1_0 MMVQ correctness follow-up: the Q1 hunk from Prism llama.cpp PR `#55`
  at `5eaa904930fb` (deferred affine reconstruction after the DP4A dot product)
- Vendored paths: `LICENSE`, `common/jinja`, `common/log.h`, `common/unicode.*`, `ggml`, `gguf-py`

Open ggml feature PRs are intentionally not included until they are merged, except for explicitly listed hub test PRs.
