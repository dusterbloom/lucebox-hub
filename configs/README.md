# configs — Declarative Inference Profiles

This directory contains declarative TOML profiles and backend definitions for
running Gemma-4 inference on lucebox-hub. Each profile captures a specific
(model, context length, speculative decode method, hardware) combination along
with measured performance floors, so every run is reproducible and comparable.

## Why this exists

Ad-hoc shell commands diverge over time. Profiles make the connection between
a benchmark log and the exact flags used to produce it explicit and machine-checkable.

## Directory layout

```
configs/
  profiles/        — one .toml per (model, ctx, method, hw) combination
  backends/        — one .toml per inference binary variant
```

## Quick start

```bash
# Lint everything (exits 0 if no errors, prints warnings)
python dflash/scripts/config_lint.py

# Dry-run a profile (validates env, paths, backend; does NOT run inference)
python dflash/scripts/profile_run.py --profile rtx3090-moe26b-dflash-256k --dry-run

# Print the resolved command (for inspection or shell scripting)
python dflash/scripts/profile_run.py --profile rtx3090-moe26b-dflash-256k --print-cmd

# Run (execvp — replaces the Python process)
LUCEBOX_ROOT=/your/root python dflash/scripts/profile_run.py --profile rtx3090-dense31b-mtp-64k

# Override a single field at runtime
python dflash/scripts/profile_run.py --profile rtx3090-moe26b-dflash-256k \
    --override runtime.ctx=131072

# Verify a running server meets the floors declared in the profile
python dflash/scripts/verify_server.py --profile rtx3090-moe26b-dflash-256k \
    --base-url http://127.0.0.1:8080 --runs 5
```

## Required environment variables

| Profile | Variable | Purpose |
|---------|----------|---------|
| rtx3090-dense31b-mtp-64k | `LUCEBOX_ROOT` | Root containing models/ |
| rtx3090-moe26b-dflash-256k | `HOME` (auto-set) | Root for ~/models/ paths |
| rtx3090-moe26b-mtp-1m | `HOME` (auto-set) | Root for ~/models/ paths |
| llama-upstream backend | `LUCEBOX_LLAMA_BIN` | Path to llama-server or llama-cli |

## Shipped profiles

| Profile | Model | Method | CTX | Measured decode | Floor |
|---------|-------|--------|-----|-----------------|-------|
| rtx3090-dense31b-mtp-64k | Gemma-4 31B dense Q4_K_M | MTP γ=2 | 64K | 10.07 tok/s | 9.5 tok/s |
| rtx3090-moe26b-dflash-256k | Gemma-4 26B-A4B MoE Q4_K_M | DFlash dm=4+pflash | 256K | 67.95 tok/s / 55ms TTFT | 65.0 tok/s / 65ms |
| rtx3090-moe26b-mtp-1m | Gemma-4 26B-A4B MoE Q4_K_M | MTP γ=2+pflash | 1M | 23.65 tok/s / 108ms TTFT | 22.0 tok/s / 120ms |

All measurements taken on RTX 3090 (24 GB VRAM) running WSL2 (peppi-rtx3090-wsl).

## Backends

| Backend | Binary | Spec methods |
|---------|--------|-------------|
| dflash | `dflash/build/test_gemma4_dflash` (in-tree) | none, mtp, dflash |
| llama-upstream | `$LUCEBOX_LLAMA_BIN` (external) | none |

## Schema reference

See `configs/CONTRIBUTING.md` for the full schema and contribution guide.
