"""_harness_lib.py — shared helpers for bench driver scripts.

Internal module; imported by run_agentic_multiturn.py and run_agentic_ee7_passbv.py.
Do not call directly.
"""

import os
import re
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
HARNESS_DIR = REPO / "harness/clients"
BENCH_LOCK = "/tmp/lucebox-bench.lock"

# All 11 supported harness clients.
# backend_pair is structurally a wrapper over other clients (it drives two backends
# in parallel and compares outputs); it has no --client-facing harness script of
# its own, so it is listed here for completeness but handled specially.
ALL_CLIENTS = [
    "claude_code",
    "codex",
    "pi",
    "hermes",
    "opencode",
    "openwebui",
    "openwebui_tools",
    "openclaw",
    "claude_llamacpp_matrix",
    "claude_llamacpp_decode_check",
    # backend_pair: wrapper over other clients; no natural single-script invocation.
    # Skipped in per-client loops; callers must handle explicitly if needed.
    "backend_pair",
]

# Primary 5: have session_inject_proxy wiring for bandit.
PRIMARY_CLIENTS = ["claude_code", "codex", "pi", "hermes", "opencode"]

# Default model/server configuration (RTX 3090, 24 GB).
DEFAULT_SERVER_ENV = {
    "MODEL_SERVER": "lucebox",
    "LUCEBOX_SERVER_BACKEND": "cpp",
    "DFLASH27B_KV_K": "tq3_0",
    "DFLASH27B_KV_V": "tq3_0",
    "GGML_CUDA_NO_VMM": "1",
    "PFLASH_DRAFTER_EARLY_EXIT_N": "7",
    "PFLASH_DRAFTER_SCORE_LAYERS": "7",
    "TARGET": "/home/peppi/models/qwen3.6-27b-q4km/Qwen3.6-27B-Q4_K_M.gguf",
    "DRAFT": "/home/peppi/models/qwen3.6-27b-dflash/dflash-draft-3.6-q4_k_m.gguf",
    "DFLASH_SERVER_BIN": "/home/peppi/Dev/lucebox-hub/dflash/build/dflash_server",
    "MAX_CTX": "98304",
    "MAX_TOKENS": "512",
    "VERIFY_MODE": "ddtree",
    "BUDGET": "16",
    "REPO_DIR": str(REPO),
    "RUN_DIR": "/tmp/lucebox-bench-runs",
    "EXTRA_SERVER_ARGS": (
        "--prefill-compression always --prefill-keep-ratio 0.05 "
        "--prefill-drafter /home/peppi/models/Qwen3-0.6B-BF16.gguf"
    ),
    "CLAUDE_BIN": "/home/peppi/.local/bin/claude",
    "CLAUDE_TIMEOUT": "600",
    "MARKER": "OK_DONE",
    "CLAUDE_TOOLS": "none",
    "PORT": "19099",
    "HOST": "127.0.0.1",
    "MODEL_ID": "luce-dflash",
    "API_KEY": "sk-lucebox",
}


def harness_for(client: str) -> Path:
    """Return path to the client harness script."""
    return HARNESS_DIR / f"run_{client}.sh"


def client_out_filename(client: str) -> str:
    """Return the expected output filename for a given client."""
    mapping = {
        "claude_code": "claude-code.out",
        "codex": "codex.out",
        "pi": "pi.out",
        "hermes": "hermes.out",
        "opencode": "opencode.out",
        "openwebui": "openwebui.out",
        "openwebui_tools": "openwebui_tools.out",
        "openclaw": "openclaw.out",
        "claude_llamacpp_matrix": "claude-llamacpp-matrix.out",
        "claude_llamacpp_decode_check": "claude-llamacpp-decode-check.out",
        "backend_pair": "backend-pair.out",
    }
    return mapping.get(client, f"{client}.out")


def wait_for_health(base_url: str, timeout_s: int = 120) -> bool:
    """Poll /health until server responds or timeout."""
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        try:
            urllib.request.urlopen(f"{base_url}/health", timeout=2)
            return True
        except Exception:
            time.sleep(1)
    return False


def build_env(overrides: dict | None = None) -> dict:
    """Build subprocess env: os.environ + DEFAULT_SERVER_ENV + overrides."""
    env = os.environ.copy()
    env.update(DEFAULT_SERVER_ENV)
    if overrides:
        env.update(overrides)
    return env


def run_client(
    client: str,
    env: dict,
    run_name: str,
    run_dir: str | None = None,
    timeout_s: int = 600,
) -> dict:
    """Run a harness client; return metrics dict.

    The harness script is responsible for starting and stopping the server;
    env must contain all required variables (PORT, HOST, TARGET, etc.).

    Returns dict with keys: condition, drafter_fwd_s, n_tokens, accept_rate,
    accept_detail, ok_done, run_dir, rc.
    """
    actual_run_dir = run_dir or f"/tmp/lucebox-bench-runs/{run_name}"
    os.makedirs(actual_run_dir, exist_ok=True)

    local_env = env.copy()
    local_env["STAMP"] = run_name
    local_env["RUN_DIR"] = str(Path(actual_run_dir).parent)

    harness = harness_for(client)
    if not harness.exists():
        return {
            "client": client,
            "run_name": run_name,
            "drafter_fwd_s": None,
            "n_tokens": None,
            "accept_rate": None,
            "accept_detail": None,
            "ok_done": False,
            "run_dir": actual_run_dir,
            "rc": -1,
            "error": f"harness not found: {harness}",
        }

    print(f"[harness_lib] running client={client} run={run_name}", flush=True)
    t0 = time.perf_counter()
    try:
        result = subprocess.run(
            ["bash", str(harness)],
            env=local_env,
            capture_output=True,
            text=True,
            timeout=timeout_s,
        )
        elapsed = time.perf_counter() - t0
        combined = result.stdout + result.stderr
        rc = result.returncode
        print(f"[harness_lib] client={client} rc={rc} elapsed={elapsed:.1f}s", flush=True)
    except subprocess.TimeoutExpired:
        elapsed = time.perf_counter() - t0
        print(f"[harness_lib] client={client} TIMEOUT after {elapsed:.0f}s", flush=True)
        return {
            "client": client,
            "run_name": run_name,
            "drafter_fwd_s": None,
            "n_tokens": None,
            "accept_rate": None,
            "accept_detail": None,
            "ok_done": False,
            "run_dir": actual_run_dir,
            "rc": -1,
            "error": "timeout",
        }

    # Parse metrics from combined output (harness prints server tail at end)
    server_log = Path(actual_run_dir) / "server.log"
    drafter_fwd = None
    n_tokens = None
    accept_rate = None
    accept_str = None
    ok_done = False

    # Try server log first, fall back to combined output
    for text_source in [
        (server_log.read_text() if server_log.exists() else ""),
        combined,
    ]:
        if drafter_fwd is None:
            m = re.search(r"\[drafter\] forward\+score in ([\d.]+)s S=(\d+)", text_source)
            if m:
                drafter_fwd = float(m.group(1))
                n_tokens = int(m.group(2))
        if accept_rate is None:
            m2 = re.search(r"accepted=(\d+/\d+) \(([\d.]+)%\)", text_source)
            if m2:
                accept_rate = m2.group(2) + "%"
                accept_str = m2.group(0)

    marker = env.get("MARKER", "OK_DONE")
    # Check client output file
    client_fn = client_out_filename(client)
    client_out_path = Path(actual_run_dir) / client_fn
    if client_out_path.exists():
        ok_done = marker in client_out_path.read_text()
    if not ok_done:
        ok_done = marker in combined

    print(f"  drafter_fwd={drafter_fwd}s n={n_tokens} accept={accept_str} ok_done={ok_done}",
          flush=True)

    return {
        "client": client,
        "run_name": run_name,
        "drafter_fwd_s": drafter_fwd,
        "n_tokens": n_tokens,
        "accept_rate": accept_rate,
        "accept_detail": accept_str,
        "ok_done": ok_done,
        "run_dir": actual_run_dir,
        "rc": rc,
    }
