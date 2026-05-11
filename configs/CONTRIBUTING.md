# Contributing a New Profile

Thank you for contributing a benchmark result. Follow these five steps exactly.

---

## Step 1 — Run the benchmark and save the log

Run the inference command and capture stdout/stderr to a file:

```bash
<your binary> <flags> 2>&1 | tee .sisyphus/notes/<experiment>/<your-run>.log
```

The log file is the ground truth for the numbers in your profile. Without it
the profile will be rejected.

---

## Step 2 — Create the profile TOML

Copy the template from `configs/profiles/base.toml` or the most similar
existing profile. Name it `<hw>-<model>-<method>-<ctx>.toml`, e.g.:
`rtx4090-dense31b-mtp-128k.toml`.

### Required keys

```toml
extends = "base"          # or another profile stem
backend = "dflash"        # must match a file in configs/backends/

[hardware]
gpu = "RTX 4090"
sm = 89

[model]
target = "${LUCEBOX_ROOT}/models/your-model.gguf"
# mtp_assistant required when spec.method = "mtp"
# dflash_draft  required when spec.method = "dflash"

[runtime]
ctx = 131072
kv_k = "tq3_0"
kv_v = "tq3_0"

[runtime.spec]
method = "mtp"      # "none" | "mtp" | "dflash"
gamma = 2           # required for mtp
# draft_max = 4     # required for dflash

[expected_floors]
decode_tok_s = 15.0
# ttft_ms_max = 80.0
# prefill_tok_s = 500.0

[provenance]
source_log = ".sisyphus/notes/<experiment>/<your-run>.log"
measured_at = "2026-01-15"       # ISO date
hardware_id = "yourname-rtx4090-linux"
commit = "abc1234"               # optional git SHA
```

### Auto-rejection rules (the linter will reject these)

- `provenance.source_log = "<NEEDS_RUN>"` — fill in the real log path
- Hardcoded `/absolute/paths` anywhere — use `${VAR}/...` or relative paths
- `spec.method = "mtp"` without `model.mtp_assistant`
- `spec.method = "dflash"` without `model.dflash_draft`
- Empty `[expected_floors]` — set at least one floor
- Missing `[provenance]` section or any of its three required fields
- `source_log` pointing to a file that does not exist (warning, not error,
  but reviewers will ask you to provide it)

---

## Step 3 — Lint before submitting

```bash
python dflash/scripts/config_lint.py --profile <your-profile-stem>
```

Must exit 0 (warnings about missing binaries are OK).

For strict checking (promotes warnings to errors):

```bash
python dflash/scripts/config_lint.py --profile <your-profile-stem> --strict
```

---

## Step 4 — Add or validate the backend

If your profile uses a backend that already exists, skip this step.

To add a backend, create `configs/backends/<name>.toml`:

```toml
name = "my-backend"      # must match filename stem exactly
upstream = "https://..."
build_hint = "..."       # optional build instructions

[binary]
# exactly one of:
in_tree = "path/relative/to/git/root"
# env_var = "MY_BINARY_VAR"

[supports]
spec_types = ["none", "mtp"]   # which methods this binary supports
kv_quants = ["q8_0", "tq3_0"]

[flags]
# map canonical key -> CLI flag string
model = "--model"
ctx = "--ctx-size"
kv_k = "--kv-k"
kv_v = "--kv-v"
# if "mtp" in spec_types:
spec_model = "--mtp"
spec_gamma = "--gamma"
# if "dflash" in spec_types:
# draft_model = "--draft"
# draft_max  = "--draft-max"

[stdout_parse]
tok_s   = "eval time.*?([0-9]+\.[0-9]+) tokens per second"
ttft_ms = "time to first token.*?([0-9]+\.[0-9]+) ms"
```

Backend validation rules:
- `name` must equal the filename stem
- Exactly one of `binary.in_tree` or `binary.env_var` must be set
- All required flags for declared `spec_types` must be present

---

## Step 5 — Open a pull request

Include in the PR body:
- A snippet from the log file showing the measured tok/s and TTFT
- The exact hardware (GPU model, driver version, VRAM)
- The date of measurement
- Confirmation that `config_lint.py --strict` exits 0

### Disclosure requirement

If any part of the profile, code, or PR description was AI-generated, state
this explicitly. PRs with AI-generated content that is not disclosed will be
closed.

---

## Schema reference summary

### Profile keys

| Key | Type | Required | Notes |
|-----|------|----------|-------|
| extends | string | yes | parent profile stem or "" for none |
| backend | string | yes | stem of a file in configs/backends/ |
| hardware.gpu | string | yes | GPU model name |
| hardware.sm | int | yes | CUDA SM version (e.g. 86 for Ampere) |
| model.target | path | yes | main model GGUF |
| model.mtp_assistant | path | when method=mtp | MTP assistant GGUF |
| model.dflash_draft | path | when method=dflash | DFlash draft GGUF |
| runtime.ctx | int | yes | context length in tokens |
| runtime.kv_k | string | yes | KV cache key quantization |
| runtime.kv_v | string | yes | KV cache value quantization |
| runtime.spec.method | string | yes | "none", "mtp", or "dflash" |
| runtime.spec.gamma | int | when method=mtp | speculative tokens per step |
| runtime.spec.draft_max | int | when method=dflash | max draft tokens |
| runtime.flash_attn | bool | no | enable flash attention |
| runtime.pflash | bool | no | enable pflash (MoE models) |
| expected_floors | table | yes | at least one floor metric |
| provenance.source_log | path | yes | path to benchmark log |
| provenance.measured_at | date | yes | ISO 8601 date |
| provenance.hardware_id | string | yes | unique hardware identifier |
| provenance.commit | string | no | git SHA of code under test |
