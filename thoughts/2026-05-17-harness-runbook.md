# Qwen3.6-27B Speculative Decoding — Agentic Harness Runbook

**For any agent picking this up cold.** Last verified 2026-05-17 on RTX 3090 (sm_86), WSL2.
This doc describes how to start, smoke-test, bench, and verify quality on three speculative-decoding paths:
1. **AR baseline** (no speculation, ground truth for byte-identity checks)
2. **MTP chain D=3** — autoregressive MTP head re-fed with own pre-norm hidden
3. **DFlash standalone** — separate small drafter model with q_len=16 parallel candidates

If any command in this doc fails, the harness is broken — file an issue with the exact error before doing anything else.

---

## 0. Prerequisites

### Hardware
- NVIDIA GPU with compute capability ≥ sm_86 (RTX 3090, A6000, A100, H100 all qualify)
- ≥24 GB VRAM (the Q4 target + Q4 draft loaded simultaneously use ~18 GB)
- WSL2 OR bare-metal Linux. CUDA 12.x driver + toolkit.

### Software
- `cmake` (build the test binary)
- Python 3.11+ with `transformers`, `gguf`, `pandas`, `pyarrow`
- `nvidia-smi` working

### Model files (verified paths and identities)

All paths verified 2026-05-17. If any file moves or its metadata changes, this harness will give wrong numbers silently.

| Role | Path | Size | Architecture | Key metadata |
|---|---|---|---|---|
| **MTP target+head (Q4)** | `/home/peppi/models/qwen3.6-27b-mtp-q4/Qwen3.6-27B-MTP-Q4_K_M.gguf` | 17.1 GB | `qwen35` | `block_count=65`, `nextn_predict_layers=1`, `rope.freq_base=1e7`, `context_length=262144` |
| MTP target+head (Q2) | `/home/peppi/models/qwen3.6-27b-mtp/Qwen3.6-27B-UD-Q2_K_XL.gguf` | 11.0 GB | `qwen35` | same shape as Q4 |
| **DFlash target** | `/home/peppi/Dev/lucebox-hub/dflash/models/Qwen3.6-27B-Q4_K_XL.gguf` (symlink → `~/models/qwen3.6-27b/`) | 19 GB | `qwen35` | Unsloth dynamic quant |
| **DFlash drafter** | `/home/peppi/models/qwen3.6-27b-dflash/dflash-draft-3.6-q4_k_m.gguf` | 1.0 GB | `dflash-draft` | `block_count=5`, `context_length=262144`, `rope.freq_base=1e7`, `attention.sliding_window=2048` |

**Wrong files to AVOID (silent failures):**
- `/home/peppi/Dev/lucebox-hub/dflash/models/draft-3.6/dflash-draft-3.6-luce.gguf` — Qwen3.5 arch, 32K ctx, NO SWA. Architecturally incompatible with Qwen3.6.
- `/home/peppi/Dev/lucebox-hub/dflash/models/Qwen3.5-27B-Q4_K_M.gguf` — legacy Qwen3.5 target, not Qwen3.6.
- `Qwen3.6-27B-MTP-Q4_K_M.gguf` from a multi-wget race (file may parse but contain NaN in F32 norm/SSM weights). Defensive scanner in `gguf_target_loader.cpp` will fail-fast with the tensor name on load.

### Build the test binary

```bash
cd /home/peppi/Dev/lucebox-hub/dflash
cmake -B build -DDFLASH_GPU_TESTS=ON
cmake --build build -j$(nproc) --target test_dflash test_mtp_chain_strict_spec test_mtp_chain_runner test_mtp_step_graph_cuda test_mtp_step_graph_cuda_nocast test_mtp_step_graph_cache
```

Expected: zero errors, zero warnings. If it warns about deprecated APIs, ignore (transient gcc/CUDA mismatch).

### Verify regression suite passes before doing anything else

```bash
cd /home/peppi/Dev/lucebox-hub
for t in test_mtp_chain_strict_spec test_mtp_chain_runner test_mtp_step_graph_cache test_mtp_step_graph_cuda test_mtp_step_graph_cuda_nocast; do
  echo "=== $t ==="
  dflash/build/$t 2>&1 | tail -3
  echo "exit=$?"
done
```

All 5 must exit 0. The CUDA tests print `absmax=0.000276` and `PASS`. If any fails, **do not bench** — file an issue.

---

## 1. Smoke test (60 seconds, validates harness end-to-end)

This runs a tiny generation through MTP chain D=3 on one short prompt. If it produces sensible output, the harness works.

```bash
cd /home/peppi/Dev/lucebox-hub
BENCH_THINKING=1 \
  MTP_GGUF=/home/peppi/models/qwen3.6-27b-mtp-q4/Qwen3.6-27B-MTP-Q4_K_M.gguf \
  DFLASH_BIN=/home/peppi/Dev/lucebox-hub/dflash/build/test_dflash \
  python3 -u dflash/scripts/bench_agent_mtp.py \
    --bucket 2k --n-sample 1 --n-runs 1 --n-gen 32 \
    --draft-source chain --chain-depth 3 2>&1 | tail -10
```

**Expected output (verified 2026-05-17):**
```
RESULT_JSON {... "tok_s": ~50, "accepted": ~12, "proposed": ~22, "accept_rate": ~0.55-0.78}
MTP run 1 (chain_depth=3): tok_s=~50  tokens=32  ...  accept=~55-78%
```

**Smoke test PASSES if:**
- `tok_s` between 35 and 70 (huge range to accommodate hardware variance)
- `accept` rate between 50% and 85%
- Exit code 0
- No NaN, no crashes

If `tok_s < 20` or `accept < 30%` → something is structurally wrong, do not proceed.

---

## 2. Headline benchmark (10-15 min, reproduces today's numbers)

This is the apples-to-apples comparison on 8 aligned SWE-bench prompts with 3 runs each (n=24 measurements per path).

### 2a. MTP chain D=3 (aligned with DFlash row selection)

```bash
cd /home/peppi/Dev/lucebox-hub
BENCH_THINKING=1 \
  MTP_GGUF=/home/peppi/models/qwen3.6-27b-mtp-q4/Qwen3.6-27B-MTP-Q4_K_M.gguf \
  DFLASH_BIN=/home/peppi/Dev/lucebox-hub/dflash/build/test_dflash \
  python3 -u dflash/scripts/bench_agent_mtp.py \
    --bucket 2k --n-sample 8 --n-runs 3 --n-gen 128 \
    --draft-source chain --chain-depth 3 2>&1 | tee /tmp/harness_mtp.log
```

**Expected (2026-05-17 reference):**
- Median MTP: 49.51 tok/s
- Median decSp: 1.574×
- Median accept: 67.0%
- Per-prompt CV: ~7.5% (rock-stable)

### 2b. DFlash standalone (same 8 prompts, q4_k_m draft, SWA env)

```bash
cd /home/peppi/Dev/lucebox-hub
DFLASH27B_DRAFT_SWA=2048 \
  DFLASH_TARGET=/home/peppi/Dev/lucebox-hub/dflash/models/Qwen3.6-27B-Q4_K_XL.gguf \
  DFLASH_DRAFT=/home/peppi/models/qwen3.6-27b-dflash/dflash-draft-3.6-q4_k_m.gguf \
  DFLASH_BIN=/home/peppi/Dev/lucebox-hub/dflash/build/test_dflash \
  python3 -u dflash/scripts/bench_agent.py --bucket 2k --n-sample 8 2>&1 | tee /tmp/harness_dflash.log
```

**Expected (2026-05-17 reference):**
- Mean DFlash: 48.42 tok/s, decSp 1.58×
- Median DFlash: 42.40 tok/s, decSp 1.39×
- Distribution: wide (13-82 tok/s range, AL 1.33-6.92)

### 2c. Parse + compare

```bash
python3 << 'PY'
import json, statistics
def medians(path, key):
    with open(path) as f:
        data = [json.loads(l[len("CELL_JSON "):]) for l in f if l.startswith("CELL_JSON")]
    return [d[key] for d in data]
mtp_tokps = medians('/tmp/harness_mtp.log', 'tok_s')
mtp_mtp = [t for t in mtp_tokps if t > 35]  # filter AR (gamma=0) cells
print(f"MTP median: {statistics.median(mtp_mtp):.2f} tok/s (n={len(mtp_mtp)})")
PY
```

For DFlash, use the printed `[2k] mean:` summary line directly.

**Acceptance criterion for harness validity:**
- MTP median within ±10% of 49.51 → harness is calibrated
- DFlash median within ±15% of 42.40 → harness is calibrated (wider tolerance due to DFlash's intrinsic variance)
- If either is way off → check `nvidia-smi` for thermal throttling, check `BENCH_THINKING=1` is set for MTP, check `DFLASH27B_DRAFT_SWA=2048` is set for DFlash.

---

## 3. Quality verification (10 min, confirms outputs are coherent)

The throughput bench measures speed, not quality. To verify MTP outputs match AR semantically:

```bash
cd /home/peppi/Dev/lucebox-hub
# Generate same prompt with both paths, dump tokens
PROMPT=/tmp/dflash_bench/mtp_agent_2k_1.bin  # 3051-token SWE prompt, auto-created by bench
GGUF=/home/peppi/models/qwen3.6-27b-mtp-q4/Qwen3.6-27B-MTP-Q4_K_M.gguf
BIN=/home/peppi/Dev/lucebox-hub/dflash/build/test_dflash

# AR (gamma=0)
$BIN $GGUF --mtp-gguf $GGUF --prompt-bin $PROMPT --n-gen 128 \
  --gamma 0 --prompt-id 101 --max-ctx 4096 -ctk q8_0 -ctv q8_0 \
  --draft-source chain --out /tmp/ar_tokens.bin 2>&1 | grep RESULT

# MTP D=3
$BIN $GGUF --mtp-gguf $GGUF --prompt-bin $PROMPT --n-gen 128 \
  --gamma 3 --prompt-id 101 --max-ctx 4096 -ctk q8_0 -ctv q8_0 \
  --draft-source chain --out /tmp/mtp_tokens.bin 2>&1 | grep RESULT

# Compare
python3 << 'PY'
import struct
ar = list(struct.unpack(f'{len(open("/tmp/ar_tokens.bin","rb").read())//4}i', open("/tmp/ar_tokens.bin","rb").read()))
mtp = list(struct.unpack(f'{len(open("/tmp/mtp_tokens.bin","rb").read())//4}i', open("/tmp/mtp_tokens.bin","rb").read()))
prompt_len = 3051
ar_gen, mtp_gen = ar[prompt_len:], mtp[prompt_len:]
ident = sum(1 for i in range(min(len(ar_gen), len(mtp_gen))) if ar_gen[i] == mtp_gen[i])
first_diff = next((i for i in range(min(len(ar_gen), len(mtp_gen))) if ar_gen[i] != mtp_gen[i]), -1)
print(f'AR len={len(ar_gen)}, MTP len={len(mtp_gen)}')
print(f'First-position match: {ident}/{min(len(ar_gen),len(mtp_gen))} ({100*ident/min(len(ar_gen),len(mtp_gen)):.1f}%)')
print(f'First divergence at generated idx: {first_diff}')
PY
```

**Expected (2026-05-17 reference):**
- First ~47 generated tokens byte-identical
- Then divergence due to q8/q8 KV sub-ULP logit drift flipping argmax on near-tied positions
- Both continuations should be coherent SWE-bench analyses (manual inspection)

**Strict-spec failure mode:** if first 5 tokens already diverge, the MTP path has a real bug. File an issue.

---

## 4. Detokenize for human inspection

```bash
cd /home/peppi/Dev/lucebox-hub
python3 << 'PY'
import struct, gguf
r = gguf.GGUFReader('/home/peppi/models/qwen3.6-27b-mtp-q4/Qwen3.6-27B-MTP-Q4_K_M.gguf')
fields = {f.name: f for f in r.fields.values()}
tok_field = fields['tokenizer.ggml.tokens']
vocab = {i: bytes(tok_field.parts[idx]) for i, idx in enumerate(tok_field.data)}

# GPT2 byte-level decoder
def gpt2_byte_decoder():
    bs = list(range(ord('!'), ord('~')+1)) + list(range(ord('¡'), ord('¬')+1)) + list(range(ord('®'), ord('ÿ')+1))
    cs = bs[:]
    n = 0
    for b in range(256):
        if b not in bs:
            bs.append(b); cs.append(256+n); n += 1
    return {chr(c): b for b, c in zip(bs, cs)}

bd = gpt2_byte_decoder()
def decode(ids):
    out = bytearray()
    for tid in ids:
        for ch in vocab[tid].decode('utf-8', errors='replace'):
            out.append(bd.get(ch, ord('?')))
    return out.decode('utf-8', errors='replace')

ar = list(struct.unpack(f'{len(open("/tmp/ar_tokens.bin","rb").read())//4}i', open("/tmp/ar_tokens.bin","rb").read()))
mtp = list(struct.unpack(f'{len(open("/tmp/mtp_tokens.bin","rb").read())//4}i', open("/tmp/mtp_tokens.bin","rb").read()))
print('=== AR generation (128 tokens) ==='); print(decode(ar[3051:]))
print('\n=== MTP D=3 generation (128 tokens) ==='); print(decode(mtp[3051:]))
PY
```

---

## 5. Troubleshooting

| Symptom | Likely cause | Fix |
|---|---|---|
| `qwen36-mtp prefill produced invalid token` | Corrupt GGUF (NaN F32 weights) | Re-download the GGUF; the loader's NaN scanner (commit `238e503`) will name the bad tensor |
| `tok_s < 20` with no errors | GPU thermal throttle OR another process competing | `nvidia-smi`; kill competitors; cool GPU; retry |
| `accept < 30%` on MTP | `BENCH_THINKING=1` not set OR wrong GGUF (use the MTP-fused one, not bare backbone) | Set the env, double-check `MTP_GGUF` path |
| `Got CUDA error` on test_dflash startup | CUDA driver version mismatch | Check `nvidia-smi` shows the same driver version as `nvcc --version` was built against |
| DFlash sample drops to ~13 tok/s on one prompt | Draft model distribution mismatch on that specific prompt | Known DFlash behavior. Median across n=8 is the headline number, not single-sample. |

---

## 6. Reference numbers (2026-05-17 baseline)

| Path | Median tok/s | Median decSp | Accept/AL | Variance (CV) |
|---|---|---|---|---|
| AR baseline (Q4) | 31.45 | 1.00× | — | ~3% |
| MTP chain D=3 (Q4) | 49.51 | 1.574× | 67.0% | 7.5% |
| MTP chain D=3 (Q2) | 56.05 (single best prompt) | 1.668× | 75.2% | ~5% |
| DFlash standalone (Q4) | 42.40 | 1.39× | AL 4.32 | ~50% (wild) |

**Hardware:** RTX 3090 (sm_86), WSL2, single GPU, 24 GB VRAM.
**Commit:** `9052aae` on `feature/mtp-foundation-v2`.
**Prompts:** SWE-bench Verified, 2K bucket, 8 rows selected via `df.sample(n=8, random_state=42)`.

If your numbers differ by more than ±15% on median, something in your harness setup is different from the reference. Most common causes: (a) thermal throttle, (b) wrong GGUF, (c) missing env var, (d) bare-metal vs WSL2.

---

## 7. Next experiments queued (NOT yet executed)

| ID | What | Time | Owner |
|---|---|---|---|
| F16 KV smoke | Test if `-ctk f16 -ctv q8_0` raises accept rate | 5 min | any agent |
| MTP-Q4-as-DFlash-target | DFlash bench using MTP-Q4 GGUF backbone for true apples-to-apples | 10 min | any agent |
| Adaptive depth | Per-position chain depth based on `p(top1)` | 1 day | sisyphus |
| L6 hybrid TDD spike | MTP spine + DFlash widening + tree-verify, full TDD with kill gates | 8 hours | sisyphus (plan at `/tmp/momus_l6_tdd_spike.md`) |

---

## 8. Where to file issues

- Bench numbers off by >15%: include `nvidia-smi`, `uname -a`, commit SHA, full stdout
- Build error: include `cmake --build` full stderr + `nvcc --version`
- Quality regression (output garbled): include both `--out` token bins + decoded text for human review
- Reference doc: `thoughts/2026-05-16-mtp-optimization-day.md`, `thoughts/2026-05-16-ship-plan-handoff.md`
