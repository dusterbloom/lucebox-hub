# CLI Setup Quickstart — Reproduce the Dual-Speculator Session

Tested 2026-05-19 against the dual-speculator server (commit ~f5a7112+ on
`feat/mtp-prefix-warm-ghost`) running on `http://127.0.0.1:18080`.

## Server (one-shot launch)

```bash
python3 dflash/scripts/server.py --host 127.0.0.1 --port 18080 \
  --target /home/peppi/models/qwen3.6-27b-mtp-q4/Qwen3.6-27B-MTP-Q4_K_M.gguf \
  --bin /home/peppi/Dev/lucebox-hub/dflash/build/test_dflash \
  --mtp-gguf /home/peppi/models/qwen3.6-27b-mtp-q4/Qwen3.6-27B-MTP-Q4_K_M.gguf \
  --mtp-gamma 3 --mtp-draft-source chain \
  --draft /home/peppi/models/qwen3.6-27b-dflash/dflash-draft-3.6-q4_k_m.gguf \
  --budget 22 --verify-mode ddtree \
  --prefill-compression auto --prefill-threshold 32000 --prefill-keep-ratio 0.05 \
  --prefill-drafter /home/peppi/models/Qwen3-0.6B-BF16.gguf --prefill-skip-park \
  --max-ctx 65536 --fa-window 4096 \
  --cache-type-k q8_0 --cache-type-v q8_0 \
  --prefix-cache-slots 4
```

Banner should show `[speculators] dual-mode: DFlash+MTP both loaded
(auto_select threshold=4096 tokens)`. VRAM ~21.6 GB on a 24 GB 3090.
Server is OpenAI-compatible at `/v1/chat/completions` AND Anthropic-
compatible at `/v1/messages`.

## CLIs that work today (4/5)

### 1. claude (Claude Code) ✅

`claude --bare` is **required** to bypass the keychain OAuth so the env
vars actually take effect:

```bash
ANTHROPIC_BASE_URL=http://127.0.0.1:18080 \
ANTHROPIC_AUTH_TOKEN=sk-luce \
ANTHROPIC_MODEL=luce-dflash \
DISABLE_AUTOUPDATER=1 DISABLE_TELEMETRY=1 \
claude --bare --print --model luce-dflash "your task"
```

Without `--bare`, claude code talks to itself / uses cloud, ignoring
`ANTHROPIC_BASE_URL`.

### 2. codex (OpenAI Codex CLI) ✅

Two requirements:
- `CODEX_HOME` **must be outside `/tmp`** (codex refuses to drop helper
  binaries in tmpdir). Use e.g. `~/.codex-luce`.
- `wire_api = "responses"` in config.toml (codex no longer supports
  `"chat"`; it falls back automatically when the server doesn't speak
  /v1/responses).

```bash
mkdir -p ~/.codex-luce
cat > ~/.codex-luce/config.toml <<TOML
model = "luce-dflash"
model_provider = "luce"
approval_policy = "never"
sandbox_mode = "danger-full-access"
[model_providers.luce]
name = "Lucebox"
base_url = "http://127.0.0.1:18080/v1"
env_key = "OPENAI_API_KEY"
wire_api = "responses"
TOML

CODEX_HOME=~/.codex-luce OPENAI_API_KEY=sk-luce \
  codex exec --skip-git-repo-check --sandbox danger-full-access \
    --model luce-dflash "your task"
```

### 3. hermes (NousResearch Hermes Agent) ✅

Needs `context_length >= 64000` (hermes refuses smaller). Server must be
launched with `--max-ctx 65536` to match. Hermes config goes in an
isolated HOME:

```bash
HERMES_DIR=~/.hermes-luce; mkdir -p "$HERMES_DIR"
cat > "$HERMES_DIR/config.yaml" <<YAML
model:
  default: "luce-dflash"
  provider: "lucebox"
  base_url: "http://127.0.0.1:18080/v1"
  api_key: "sk-luce"
  api_mode: "chat_completions"
  context_length: 65536
  max_tokens: 2048
custom_providers:
  - name: "lucebox"
    base_url: "http://127.0.0.1:18080/v1"
    api_key: "sk-luce"
    api_mode: "chat_completions"
    models:
      "luce-dflash":
        context_length: 65536
        max_tokens: 2048
terminal:
  backend: "local"
  cwd: "/home/peppi/Dev/lucebox-hub"
  timeout: 180
  lifetime_seconds: 300
YAML

HOME="$HERMES_DIR" HERMES_HOME="$HERMES_DIR" \
OPENAI_API_KEY=sk-luce OPENAI_BASE_URL=http://127.0.0.1:18080/v1 \
HERMES_INFERENCE_PROVIDER=lucebox HERMES_INFERENCE_MODEL=luce-dflash \
HERMES_ACCEPT_HOOKS=1 NO_COLOR=1 \
  hermes chat --quiet --provider lucebox --model luce-dflash \
    --accept-hooks --yolo --max-turns 10 \
    --source lucebox-harness --query "your task"
```

### 4. pi (mariozechner pi-coding-agent) ✅

Pi ships with a `lucebox` provider pre-wired, but pointing at port
**8000** (stale) and api `openai-completions`. Patch the existing config
in `~/.pi/agent/models.json`:

```bash
python3 -c "
import json
p='/home/peppi/.pi/agent/models.json'
d=json.load(open(p))
d['providers']['lucebox']['baseUrl']='http://127.0.0.1:18080/v1'
d['providers']['lucebox']['models'][0]['api']='openai-completions'
json.dump(d, open(p,'w'), indent=2)
"

pi --provider lucebox --model luce-dflash --mode text "your task"
```

Note: pi's supported `api` strings are `openai-completions`,
`openai-responses`, `anthropic`, etc. — **not** `openai-chat`. Stick with
`openai-completions` (legacy completions endpoint, which our server also
serves correctly).

### 5. opencode ⚠️ (reaches server but concurrent-request hang)

Fix: upgrade Node + opencode. The old install (v0.5.x on Node 22) had
provider loading bugs that silenced errors.

```bash
source ~/.nvm/nvm.sh
nvm use --lts                   # Node 24.13.0 LTS at the moment
npm install -g opencode-ai      # 1.15.5+
```

After that, add lucebox to `~/.config/opencode/opencode.json`:

```json
{
  "$schema": "https://opencode.ai/config.json",
  "provider": {
    "lucebox": {
      "npm": "@ai-sdk/openai-compatible",
      "name": "Lucebox (local dual-speculator)",
      "options": {
        "baseURL": "http://127.0.0.1:18080/v1",
        "apiKey": "sk-luce",
        "timeout": 600000,
        "chunkTimeout": 60000
      },
      "models": {
        "luce-dflash": {
          "name": "Lucebox DFlash",
          "limit": { "context": 65536, "output": 2048 }
        }
      }
    }
  }
}
```

Verify the provider is loaded:

```bash
opencode models lucebox
# expected:  lucebox/luce-dflash
```

Then run:

```bash
opencode run --model lucebox/luce-dflash "your task"
```

**Known issue**: opencode fires TWO parallel requests per `run` invocation
— a 557-token title-generation request AND the main 15101+-token request
with all tools included. On our current server, only the first one gets
a `[generate] speculator=` dispatch line; the larger second request
queues but never starts. Probably the daemon's stdin protocol is
serial-only and gets wedged by overlapping reads. Needs server-side
investigation; not a wiring problem.

Workaround until that's fixed: use `--attach` mode against a persistent
opencode server (`opencode serve`) which serializes its own requests.

## Per-CLI server log signature

The `[generate] speculator=…` line in the server log tells you which
speculator path was taken:

| Line                                            | Meaning |
|-------------------------------------------------|---------|
| `[generate] speculator=dflash (req=auto …)`    | auto_select sent it down DFlash (prompt ≤ 4096) |
| `[generate] speculator=mtp (req=auto …)`       | auto_select sent it down MTP (prompt > 4096) |
| `[generate] speculator=dflash (req=dflash …)`  | client forced DFlash via `"speculator": "dflash"` |
| `[generate] speculator=mtp (req=mtp …)`        | client forced MTP via `"speculator": "mtp"` |
| `[spec-decode] tokens=N … accepted=A/P (X%)`   | DFlash drafter accept stats |
| `[mtp_decode] iters=N proposed=… accept_rate=` | MTP γ-chain accept stats |

## Known empirical observations

From a real claude --bare session on this server (2026-05-19, 3-turn
agentic flow against this repo, prompt "explain compress_text_via_daemon"):

| Turn | Prompt tokens | Speculator picked | Decode tok/s | Note |
|------|---------------|-------------------|--------------|------|
| 1    | 92            | DFlash (< 4K)     | 56.3         | initial system+task |
| 2    | 1198          | DFlash (< 4K)     | 33.4         | tool_use turn, accumulated history |
| 3    | 4207          | **MTP (> 4K)**    | **49.3**     | long generation, 1815 output tokens |

The auto-select threshold (`prompt_tokens > 4096 → MTP`) crossed exactly
where it was supposed to in a real session.
