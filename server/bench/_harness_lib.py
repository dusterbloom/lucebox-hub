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

_PFLASH_DRAFTER = "/home/peppi/models/Qwen3-0.6B-Q8_0.gguf"

# Default model/server configuration (RTX 3090, 24 GB).
DEFAULT_SERVER_ENV = {
    "MODEL_SERVER": "lucebox",
    "LUCEBOX_SERVER_BACKEND": "cpp",
    "DFLASH27B_KV_K": "tq3_0",
    "DFLASH27B_KV_V": "tq3_0",
    "GGML_CUDA_NO_VMM": "1",
    "TARGET": "/home/peppi/models/qwen3.6-27b-q4km/Qwen3.6-27B-Q4_K_M.gguf",
    # decode drafter (dflash speculative decode path)
    "DRAFT": "/home/peppi/models/Qwen3-0.6B-Q8_0.gguf",
    "DFLASH_SERVER_BIN": "/home/peppi/Dev/lucebox-hub/dflash/build/dflash_server",
    "MAX_CTX": "98304",
    "MAX_TOKENS": "512",
    "VERIFY_MODE": "ddtree",
    "BUDGET": "16",
    "REPO_DIR": str(REPO),
    "RUN_DIR": "/tmp/lucebox-bench-runs",
    # EXTRA_SERVER_ARGS is intentionally empty here; callers inject pflash or
    # baseline flags via build_env(overrides={"EXTRA_SERVER_ARGS": ...}).
    "EXTRA_SERVER_ARGS": "",
    "CLAUDE_BIN": "/home/peppi/.local/bin/claude",
    "CLAUDE_TIMEOUT": "600",
    "MARKER": "OK_DONE",
    "CLAUDE_TOOLS": "none",
    "PORT": "19099",
    "HOST": "127.0.0.1",
    "MODEL_ID": "luce-dflash",
    "API_KEY": "sk-lucebox",
}

# pflash arm: prefill compression ON, ee7
# Note: --prefill-anchor-transitive is not in this binary build; omitted.
# DRAFT="" prevents --draft being passed; drafter path is in --prefill-drafter via EXTRA_SERVER_ARGS.
PFLASH_SERVER_EXTRA_ARGS = (
    f"--prefill-compression always --prefill-keep-ratio 0.05 "
    f"--prefill-drafter {_PFLASH_DRAFTER}"
)
PFLASH_ENV_OVERRIDES = {
    "EXTRA_SERVER_ARGS": PFLASH_SERVER_EXTRA_ARGS,
    "PFLASH_DRAFTER_EARLY_EXIT_N": "7",
    "PFLASH_DRAFTER_SCORE_LAYERS": "7",
    "DRAFT": "",
}

# baseline arm: no compression, no early exit, no decode drafter
# DRAFT="" prevents common.sh from passing --draft with an incompatible arch model.
BASELINE_ENV_OVERRIDES: dict[str, str] = {
    "EXTRA_SERVER_ARGS": "",
    "DRAFT": "",
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
    keep_ratio = None
    prefill_s = None
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
        # [pflash] N -> M -> K tokens (X.X% kept)
        if keep_ratio is None:
            m3 = re.search(r"\[pflash\] \d+ -> \d+ -> \d+ tokens \(([\d.]+)% kept\)", text_source)
            if m3:
                keep_ratio = float(m3.group(1)) / 100.0
        # [server] chat DONE ... prefill=X.Xs
        if prefill_s is None:
            m4 = re.search(r"prefill=([\d.]+)s", text_source)
            if m4:
                prefill_s = float(m4.group(1))

    marker = env.get("MARKER", "OK_DONE")
    # Check client output file
    client_fn = client_out_filename(client)
    client_out_path = Path(actual_run_dir) / client_fn
    if client_out_path.exists():
        ok_done = marker in client_out_path.read_text()
    if not ok_done:
        ok_done = marker in combined

    # Parse prompt tokens. Try multiple log formats:
    # 1. [prefill] tokens=N (if server emits this line)
    # 2. prompt_tokens=N in [server] chat msg_... line
    # 3. "input_tokens":N from anthropic client JSON output
    prompt_tokens = None
    for text_source in [
        (server_log.read_text() if server_log.exists() else ""),
        combined,
    ]:
        if prompt_tokens is None:
            for pattern in [
                r"\[prefill\] tokens=(\d+)",
                r"\[server\] chat msg_\S+ \S+ \S+ msgs=\d+ tools=\d+ prompt_tokens=(\d+)",
                r'"prompt_tokens"\s*:\s*(\d+)',
                r'"input_tokens"\s*:\s*(\d+)',
            ]:
                mp = re.search(pattern, text_source)
                if mp:
                    prompt_tokens = int(mp.group(1))
                    break

    print(f"  drafter_fwd={drafter_fwd}s n={n_tokens} prompt_tokens={prompt_tokens} "
          f"keep_ratio={keep_ratio} prefill_s={prefill_s} accept={accept_str} ok_done={ok_done}",
          flush=True)

    return {
        "client": client,
        "run_name": run_name,
        "wall_s": elapsed,
        "drafter_fwd_s": drafter_fwd,
        "n_tokens": n_tokens,
        "prompt_tokens": prompt_tokens,
        "keep_ratio": keep_ratio,
        "prefill_s": prefill_s,
        "accept_rate": accept_rate,
        "accept_detail": accept_str,
        "ok_done": ok_done,
        "run_dir": actual_run_dir,
        "rc": rc,
    }


def run_arm(
    client: str,
    label: str,
    arm_env_overrides: dict,
    base_overrides: dict,
    output_base: Path,
    timeout_s: int = 700,
) -> dict:
    """Run one arm (baseline or pflash) of a two-arm bench.

    Harness output lands at output_base/label/ (e.g. _smoke_v2/baseline/).
    Merges base_overrides + arm_env_overrides, runs the client harness,
    returns the metrics dict with an added 'label' key.
    """
    merged = {**base_overrides, **arm_env_overrides}
    env = build_env(merged)
    # run_client sets RUN_DIR=parent(run_dir), STAMP=run_name → LOG_DIR=parent/run_name
    # So pass run_dir=output_base/label and run_name=label to get LOG_DIR=output_base/label
    arm_dir = output_base / label
    arm_dir.mkdir(parents=True, exist_ok=True)
    result = run_client(
        client=client,
        env=env,
        run_name=label,
        run_dir=str(arm_dir),
        timeout_s=timeout_s,
    )
    result["label"] = label
    return result


def write_two_arm_metrics(baseline: dict, pflash: dict, output_dir: Path) -> None:
    """Write combined metrics.txt for a two-arm bench run."""
    metrics_path = output_dir / "metrics.txt"

    prompt_tokens = pflash.get("prompt_tokens") or baseline.get("prompt_tokens") or "N/A"

    b_wall = baseline.get("wall_s")
    p_wall = pflash.get("wall_s")
    speedup = (b_wall / p_wall) if (b_wall and p_wall and p_wall > 0) else None

    b_drafter = baseline.get("drafter_fwd_s")
    p_drafter = pflash.get("drafter_fwd_s")
    drafter_speedup = (b_drafter / p_drafter) if (b_drafter and p_drafter and p_drafter > 0) else None

    b_prefill = baseline.get("prefill_s")
    p_prefill = pflash.get("prefill_s")
    prefill_speedup = (b_prefill / p_prefill) if (b_prefill and p_prefill and p_prefill > 0) else None

    # keep_ratio from parsed [pflash] line (preferred) or accept_detail
    keep_ratio_val = pflash.get("keep_ratio")
    keep_ratio_str = f"{keep_ratio_val*100:.1f}%" if keep_ratio_val is not None else "N/A"

    with metrics_path.open("w") as f:
        f.write(f"prompt_tokens={prompt_tokens}\n\n")

        f.write("[baseline]\n")
        b_wall_str = f"{b_wall:.1f}s" if b_wall else "N/A"
        b_drafter_str = f"{b_drafter:.2f}s" if b_drafter else "N/A"
        b_prefill_str = f"{b_prefill:.2f}s" if b_prefill else "N/A"
        f.write(f"wall={b_wall_str}   prefill={b_prefill_str}   drafter_wall={b_drafter_str}\n")
        f.write(f"ok_done={'YES' if baseline.get('ok_done') else 'NO'}\n\n")

        f.write("[pflash ee7]\n")
        p_wall_str = f"{p_wall:.1f}s" if p_wall else "N/A"
        p_drafter_str = f"{p_drafter:.2f}s" if p_drafter else "N/A"
        p_prefill_str = f"{p_prefill:.2f}s" if p_prefill else "N/A"
        f.write(
            f"wall={p_wall_str}   prefill={p_prefill_str}   drafter_wall={p_drafter_str}   "
            f"tokens_kept={keep_ratio_str}\n"
        )
        f.write(f"ok_done={'YES' if pflash.get('ok_done') else 'NO'}\n\n")

        f.write("[headline]\n")
        speedup_str = f"{speedup:.2f}x" if speedup else "N/A"
        ds_str = f"{drafter_speedup:.2f}x" if drafter_speedup else "N/A"
        ps_str = f"{prefill_speedup:.2f}x" if prefill_speedup else "N/A"
        both_ok = baseline.get("ok_done") and pflash.get("ok_done")
        f.write(
            f"e2e_speedup={speedup_str}   prefill_speedup={ps_str}   "
            f"drafter_speedup={ds_str}   tokens_kept={keep_ratio_str}   "
            f"OK_DONE={'yes' if both_ok else 'no'}\n"
        )

    print(f"[harness_lib] metrics -> {metrics_path}", flush=True)
