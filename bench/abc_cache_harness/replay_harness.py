#!/usr/bin/env python3
"""Multi-turn agentic composition harness with prefix-cache awareness.

Usage:
    python3 replay_harness.py --arm A_baseline --trace traces/goldgate_fix.jsonl
    python3 replay_harness.py --arm A_baseline --trace traces/goldgate_fix.jsonl --smoke
    python3 replay_harness.py --arm C_always   --trace traces/goldgate_fix.jsonl
    python3 replay_harness.py --arm C_threshold --trace traces/goldgate_fix.jsonl
    python3 replay_harness.py --selftest

Smoke gate (--smoke): runs first 3 turns only, N=1.
Full run: all turns, N=3.

Arms:
  A_baseline      : dFlash decode + prefix cache, NO pFlash, NO stochastic
  A_stochastic    : dFlash decode + prefix cache + DFLASH_STOCHASTIC, NO pFlash
  C_always        : pFlash always + dFlash decode  (ATTN_PRIMARY, stochastic)
  C_threshold     : pFlash auto/32K threshold + dFlash decode
  lucebox         : dFlash 35B-A3B Q4_K_M + dFlash drafter + spec-gate
  llama_cpp_mtp   : llama.cpp b9781 + MTP bundled GGUF + --spec-type draft-mtp
  llama_cpp_ar    : llama.cpp b9781 + autoregressive (no speculation)

Port: 19099 (NEVER 18099)
Max-ctx: 65536
"""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import os
import re
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from statistics import median, mean
from typing import Optional

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

BENCH_DIR = Path(__file__).parent
REPO_DIR = BENCH_DIR.parent.parent

SERVER_BIN = REPO_DIR / "server/build/dflash_server"
TGT = Path("/home/peppi/models/qwen3.6-27b-q4km/Qwen3.6-27B-Q4_K_M.gguf")
DD  = Path("/home/peppi/models/qwen3.6-27b-dflash/dflash-draft-3.6-bf16-reconverted.gguf")
PD  = Path("/home/peppi/models/Qwen3-0.6B-BF16.gguf")
PD_Q8 = Path("/home/peppi/models/Qwen3-0.6B-Q8_0.gguf")  # Q8 compress drafter (no-park arm)
TMPL = Path("/home/peppi/models/qwen3-coder-chat-template.jinja")

HOST = "127.0.0.1"
PORT = 19099
MAX_CTX = 40960
TEMP = 0.7

FORBIDDEN_PORT = 18099  # user's live server — never touch

# 3-way H2H model paths
TGT_35B   = Path("/home/peppi/models/qwen3.6-35b-a3b/Qwen3.6-35B-A3B-UD-Q4_K_M.gguf")
DD_35B    = Path("/home/peppi/models/qwen3.6-35b-a3b-dflash-new/qwen3.6-35b-a3b-dflash-new-bf16-reconv.gguf")
MTP_GGUF  = Path("/home/peppi/models/qwen3.6-35b-a3b-mtp/Qwen3.6-35B-A3B-UD-Q4_K_M.gguf")
MTP_27B_GGUF = Path("/home/peppi/models/qwen3.6-27b-mtp-q4/Qwen3.6-27B-MTP-Q4_K_M.gguf")
LLAMA_BIN     = Path("/home/peppi/llama.cpp/build-cuda/bin/llama-server")
LLAMA_CUDA_LIB = Path("/home/peppi/llama.cpp/build-cuda/bin")

GPU_LOCK_FILE = "/tmp/lucebox_gpu.lock"


def is_dflash_server(server_type: str) -> bool:
    return server_type in {"dflash", "dflash_openai"}


def uses_openai_chat_api(server_type: str) -> bool:
    return server_type in {"llama_cpp", "dflash_openai"}

# ---------------------------------------------------------------------------
# Arm definitions
# ---------------------------------------------------------------------------

ARMS: dict[str, dict] = {
    "A_baseline": {
        "description": "dFlash decode + prefix cache; NO pFlash",
        "extra_args": [
            "--draft", str(DD),
            "--draft-swa", "2048",
        ],
        "env": {},
    },
    "A_stochastic": {
        "description": "dFlash decode + prefix cache + DFLASH_STOCHASTIC; NO pFlash",
        "extra_args": [
            "--draft", str(DD),
            "--draft-swa", "2048",
        ],
        "env": {
            "DFLASH_STOCHASTIC": "1",
        },
    },
    "C_always": {
        "description": "pFlash always (ATTN_PRIMARY+stochastic) + dFlash decode",
        "extra_args": [
            "--draft", str(DD),
            "--draft-swa", "2048",
            "--prefill-drafter", str(PD),
            "--prefill-compression", "always",
            "--prefill-keep-ratio", "0.5",
        ],
        "env": {
            "PFLASH_ATTN_PRIMARY": "1",
            "DFLASH_STOCHASTIC": "1",
        },
    },
    "C_threshold": {
        "description": "pFlash auto/32K threshold + dFlash decode",
        "extra_args": [
            "--draft", str(DD),
            "--draft-swa", "2048",
            "--prefill-drafter", str(PD),
            "--prefill-compression", "auto",
            "--prefill-threshold", "32000",
            "--prefill-keep-ratio", "0.5",
        ],
        "env": {
            "PFLASH_ATTN_PRIMARY": "1",
            "DFLASH_STOCHASTIC": "1",
        },
    },
    # ── FlowKV validation arms ──────────────────────────────────────────
    # A_baseline run with the new FlowKV binary but no pFlash — proves no-op.
    "FLOWKV_OFF": {
        "description": "FlowKV binary, NO pFlash, PFLASH_FREEZE_HISTORY unset (no-op gate)",
        "extra_args": [
            "--draft", str(DD),
            "--draft-swa", "2048",
        ],
        "env": {},
    },
    # pFlash-always + freeze-history=1, hot_window=2.
    # Aged turns (indices 1..N-hot_window-1) compressed once + cached.
    # System (turn 0) + hot tail (last 2 turns) verbatim.
    "FLOWKV_ON": {
        "description": "pFlash-always + PFLASH_FREEZE_HISTORY=1 hot_window=2 per-msg threshold=1000 (FlowKV active)",
        "extra_args": [
            "--draft", str(DD),
            "--draft-swa", "2048",
            "--prefill-drafter", str(PD),
            "--prefill-compression", "always",
            "--prefill-keep-ratio", "0.10",
            "--prefill-threshold", "1000",
        ],
        "env": {
            "PFLASH_FREEZE_HISTORY": "1",
            "PFLASH_FREEZE_HOT_WINDOW": "2",
        },
    },
    # ── #364 disk-prefix-cache composition arms ─────────────────────────
    # DISK364: #364 scoped disk cache ON, NO pFlash, NO freeze.
    #   The #364-alone baseline (momus-required). dFlash decode draft only.
    #   Fresh, empty --kv-cache-dir; rm -rf'd before the run, persists across 7 turns.
    "DISK364": {
        "description": "#364 scoped disk-prefix-cache (auto), dFlash decode; NO pFlash, NO freeze",
        "kv_cache_dir": "/tmp/compose_kv_disk364",
        "extra_args": [
            "--draft", str(DD),
            "--draft-swa", "2048",
            "--disk-prefix-cache", "auto",
            "--kv-cache-dir", "/tmp/compose_kv_disk364",
        ],
        "env": {},
    },
    # DISK364_KV8: DISK364 with q8_0 KV. llama-bench 2026-06-12: tq3_0 KV costs 2x
    #   decode at 63K (14.4 vs 29 tok/s); q8_0 matches f16. Fit: q8 ~= +1.7GB over tq3.
    "DISK364_KV8": {
        "description": "DISK364 + q8_0 KV cache (tq3_0 decode-tax escape test)",
        "kv_cache_dir": "/tmp/compose_kv_disk364_kv8",
        "kv_cache_types": ("q8_0", "q8_0"),
        "extra_args": [
            "--draft", str(DD),
            "--draft-swa", "2048",
            "--disk-prefix-cache", "auto",
            "--kv-cache-dir", "/tmp/compose_kv_disk364_kv8",
        ],
        "env": {},
    },
    # COMPOSE_FLOWKV: unified-gate validation arm. pFlash ALWAYS so should_compress=true,
    #   continuations route to FlowKV-freeze, turn-1 verbatim. #364 disk cache w/ compress.
    #   ee7 cheap drafter (EARLY_EXIT_N=7 + SCORE_LAYERS=7). Fresh tmpdir per run.
    "COMPOSE_FLOWKV": {
        "description": "unified-gate: pFlash always + FlowKV-freeze + #364 disk(compress) + ee7",
        "kv_cache_dir": "/tmp/compose_kv_flowkv_unified",
        "extra_args": [
            "--draft", str(DD),
            "--draft-swa", "2048",
            "--prefill-drafter", str(PD),
            "--prefill-compression", "always",
            "--prefill-keep-ratio", "0.10",
            "--prefill-threshold", "1000",
            "--disk-prefix-cache", "auto",
            "--disk-prefix-cache-compress",
            "--kv-cache-dir", "/tmp/compose_kv_flowkv_unified",
        ],
        "env": {
            "PFLASH_FREEZE_HISTORY": "1",
            "PFLASH_FREEZE_HOT_WINDOW": "2",
            "PFLASH_DRAFTER_EARLY_EXIT_N": "7",
            "PFLASH_DRAFTER_SCORE_LAYERS": "7",
        },
    },
    # COMPOSE_FLOWKV_Q8NOPARK: COMPOSE_FLOWKV with the Q8_0 0.6B compress drafter
    #   AND --prefill-skip-park (target 27B stays RESIDENT during compress; no
    #   park/unpark of the 27B). Drafter also persists via drafter_loaded_.
    #   GGML_CUDA_NO_VMM=1 (set by launch_server) is the 24GB skip-park VMM workaround.
    #   Tests whether eliminating park kills the turn-4 BF16+park prefill spike (374.6s).
    "COMPOSE_FLOWKV_Q8NOPARK": {
        "description": "COMPOSE_FLOWKV + Q8_0 drafter + --prefill-skip-park (27B resident, no park)",
        "kv_cache_dir": "/tmp/compose_kv_flowkv_q8nopark",
        "extra_args": [
            "--draft", str(DD),
            "--draft-swa", "2048",
            "--prefill-drafter", str(PD_Q8),
            "--prefill-compression", "always",
            "--prefill-keep-ratio", "0.10",
            "--prefill-threshold", "1000",
            "--prefill-skip-park",
            "--disk-prefix-cache", "auto",
            "--disk-prefix-cache-compress",
            "--kv-cache-dir", "/tmp/compose_kv_flowkv_q8nopark",
        ],
        "env": {
            "PFLASH_FREEZE_HISTORY": "1",
            "PFLASH_FREEZE_HOT_WINDOW": "2",
            "PFLASH_DRAFTER_EARLY_EXIT_N": "7",
            "PFLASH_DRAFTER_SCORE_LAYERS": "7",
        },
    },
    # COMPOSE_FLOWKV_NOPARK: COMPOSE_FLOWKV (BF16 drafter that ACTUALLY loads) +
    #   --prefill-skip-park. Isolates the park variable: compression still happens
    #   (BF16 loads fine), only park/unpark of the 27B is eliminated. This is the
    #   valid apples-to-apples vs the BF16+park COMPOSE_FLOWKV baseline (turn-4 374.6s).
    "COMPOSE_FLOWKV_NOPARK": {
        "description": "COMPOSE_FLOWKV + BF16 drafter + --prefill-skip-park (27B resident, compression ON)",
        "kv_cache_dir": "/tmp/compose_kv_flowkv_nopark",
        "extra_args": [
            "--draft", str(DD),
            "--draft-swa", "2048",
            "--prefill-drafter", str(PD),
            "--prefill-compression", "always",
            "--prefill-keep-ratio", "0.10",
            "--prefill-threshold", "1000",
            "--prefill-skip-park",
            "--disk-prefix-cache", "auto",
            "--disk-prefix-cache-compress",
            "--kv-cache-dir", "/tmp/compose_kv_flowkv_nopark",
        ],
        "env": {
            "PFLASH_FREEZE_HISTORY": "1",
            "PFLASH_FREEZE_HOT_WINDOW": "2",
            "PFLASH_DRAFTER_EARLY_EXIT_N": "7",
            "PFLASH_DRAFTER_SCORE_LAYERS": "7",
        },
    },
    # COMPOSE_KV8: COMPOSE_FLOWKV_NOPARK (PR372 shipped config, auto residency)
    #   with q8_0 KV. Pairs with DISK364_KV8 for the q8-config A/B of the PR372 win.
    "COMPOSE_KV8": {
        "description": "COMPOSE_FLOWKV_NOPARK + q8_0 KV (PR372 win re-check on q8 config)",
        "kv_cache_dir": "/tmp/compose_kv_flowkv_kv8",
        "kv_cache_types": ("q8_0", "q8_0"),
        "extra_args": [
            "--draft", str(DD),
            "--draft-swa", "2048",
            "--prefill-drafter", str(PD),
            "--prefill-compression", "always",
            "--prefill-keep-ratio", "0.10",
            "--prefill-threshold", "1000",
            "--prefill-skip-park",
            "--disk-prefix-cache", "auto",
            "--disk-prefix-cache-compress",
            "--kv-cache-dir", "/tmp/compose_kv_flowkv_kv8",
        ],
        "env": {
            "PFLASH_FREEZE_HISTORY": "1",
            "PFLASH_FREEZE_HOT_WINDOW": "2",
            "PFLASH_DRAFTER_EARLY_EXIT_N": "7",
            "PFLASH_DRAFTER_SCORE_LAYERS": "7",
        },
    },
    # COMPOSE_FLOWKV_NOPARK_RS: NOPARK + --draft-residency request-scoped.
    #   Discriminator for the turn-4 prefill-rate collapse: frees the pflash
    #   drafter (~1.5-2GB) after compress scoring, before target prefill.
    "COMPOSE_FLOWKV_NOPARK_RS": {
        "description": "COMPOSE_FLOWKV_NOPARK + --draft-residency request-scoped (free drafter before target prefill)",
        "kv_cache_dir": "/tmp/compose_kv_flowkv_nopark_rs",
        "extra_args": [
            "--draft", str(DD),
            "--draft-swa", "2048",
            "--prefill-drafter", str(PD),
            "--prefill-compression", "always",
            "--prefill-keep-ratio", "0.10",
            "--prefill-threshold", "1000",
            "--prefill-skip-park",
            "--draft-residency", "request-scoped",
            "--disk-prefix-cache", "auto",
            "--disk-prefix-cache-compress",
            "--kv-cache-dir", "/tmp/compose_kv_flowkv_nopark_rs",
        ],
        "env": {
            "PFLASH_FREEZE_HISTORY": "1",
            "PFLASH_FREEZE_HOT_WINDOW": "2",
            "PFLASH_DRAFTER_EARLY_EXIT_N": "7",
            "PFLASH_DRAFTER_SCORE_LAYERS": "7",
        },
    },
    # DISK364_NODRAFT_F16: decode-tax discriminator. No draft models, f16 KV.
    #   If decode @63K ~= mainline llama-bench (29 tok/s) -> tax is tq3_0+residency.
    #   If still ~12 -> our serving loop (per-token graph rebuild) is the tax.
    "DISK364_NODRAFT_F16": {
        "description": "#364 disk cache, NO draft, f16 KV (decode-tax discriminator)",
        "kv_cache_dir": "/tmp/disk364_nodraft_f16",
        "kv_cache_types": ("f16", "f16"),
        "extra_args": [
            "--disk-prefix-cache", "auto",
            "--kv-cache-dir", "/tmp/disk364_nodraft_f16",
        ],
        "env": {},
    },
    # COMPOSE_FLOWKV_NOPARK_KV8: NOPARK with q8_0 KV instead of tq3_0.
    #   One-cell sweep: trade freed VRAM (drafter release) for cheaper attention
    #   dequant on decode reads.
    "COMPOSE_FLOWKV_NOPARK_KV8": {
        "description": "COMPOSE_FLOWKV_NOPARK + q8_0 KV cache (dequant-cost sweep)",
        "kv_cache_dir": "/tmp/compose_kv_flowkv_nopark_kv8",
        "kv_cache_types": ("q8_0", "q8_0"),
        "extra_args": [
            "--draft", str(DD),
            "--draft-swa", "2048",
            "--prefill-drafter", str(PD),
            "--prefill-compression", "always",
            "--prefill-keep-ratio", "0.10",
            "--prefill-threshold", "1000",
            "--prefill-skip-park",
            "--disk-prefix-cache", "auto",
            "--disk-prefix-cache-compress",
            "--kv-cache-dir", "/tmp/compose_kv_flowkv_nopark_kv8",
        ],
        "env": {
            "PFLASH_FREEZE_HISTORY": "1",
            "PFLASH_FREEZE_HOT_WINDOW": "2",
            "PFLASH_DRAFTER_EARLY_EXIT_N": "7",
            "PFLASH_DRAFTER_SCORE_LAYERS": "7",
        },
    },
    # COMPOSE: FlowKV (pFlash auto-gated) + #364 disk cache w/ compression.
    #   pFlash fires only on turns >= 65K tokens (same gate as standalone pFlash).
    #   FLOWKV_ON pFlash flags + dFlash decode + disk-prefix-cache-compress.
    #   Fresh, empty --kv-cache-dir; rm -rf'd before the run, persists across 7 turns.
    "COMPOSE": {
        "description": "FlowKV (pFlash auto >=65K + freeze) + #364 disk cache (compress); dFlash decode",
        "kv_cache_dir": "/tmp/compose_kv_compose",
        "extra_args": [
            "--draft", str(DD),
            "--draft-swa", "2048",
            "--prefill-drafter", str(PD),
            "--prefill-compression", "auto",
            "--prefill-keep-ratio", "0.10",
            "--prefill-threshold", "65000",
            "--disk-prefix-cache", "auto",
            "--disk-prefix-cache-compress",
            "--kv-cache-dir", "/tmp/compose_kv_compose",
        ],
        "env": {
            "PFLASH_FREEZE_HISTORY": "1",
            "PFLASH_FREEZE_HOT_WINDOW": "2",
        },
    },
    # COMPOSE_KVQK: arm B of the 372xKVFlash composition probe.
    #   COMPOSE_FLOWKV_NOPARK (PR372 shipped config) + KVFlash bounded residency
    #   pool=max(4096, max_ctx/10)=6553 with the target-QK scorer (env-only policy).
    "COMPOSE_KVQK": {
        "description": "PR372 shipped (FlowKV+disk364+NOPARK) + --kvflash 6553 + QK policy",
        "kv_cache_dir": "/tmp/compose_kv_kvqk",
        "extra_args": [
            "--draft", str(DD),
            "--draft-swa", "2048",
            "--prefill-drafter", str(PD),
            "--prefill-compression", "always",
            "--prefill-keep-ratio", "0.10",
            "--prefill-threshold", "1000",
            "--prefill-skip-park",
            "--disk-prefix-cache", "auto",
            "--disk-prefix-cache-compress",
            "--kv-cache-dir", "/tmp/compose_kv_kvqk",
            "--kvflash", "6553",
        ],
        "env": {
            "PFLASH_FREEZE_HISTORY": "1",
            "PFLASH_FREEZE_HOT_WINDOW": "2",
            "PFLASH_DRAFTER_EARLY_EXIT_N": "7",
            "PFLASH_DRAFTER_SCORE_LAYERS": "7",
            "DFLASH_KVFLASH_POLICY": "qk",
        },
    },
    # KVQK_ONLY: follow-up cell — KVFlash+QK WITHOUT FlowKV/pFlash (lossless lane).
    #   Disk364-style baseline + bounded residency; isolates lossy-vs-lossless retrieval.
    "KVQK_ONLY": {
        "description": "#364 disk cache + dFlash decode + --kvflash 6553 QK; NO pFlash/FlowKV",
        "kv_cache_dir": "/tmp/compose_kv_kvqk_only",
        "extra_args": [
            "--draft", str(DD),
            "--draft-swa", "2048",
            "--disk-prefix-cache", "auto",
            "--kv-cache-dir", "/tmp/compose_kv_kvqk_only",
            "--kvflash", "6553",
        ],
        "env": {
            "DFLASH_KVFLASH_POLICY": "qk",
        },
    },

    # ── Context-size bench arms (ring-cap=65536, gated dFlash, temp=0) ────
    # BENCH_35B: 35B-A3B Q3_K_XL all-hot + Modal drafter.
    #   Past-best config with DFLASH_FEAT_RING_CAP=65536 to avoid ring-wrap at 34K.
    #   DFLASH_SPEC_GATE=1 explicit (it's the default-on; belt-and-suspenders).
    #   fa_window=0 is the server default (full attention); no --fa-window needed.
    #   No --spark/--kvflash: all-hot MoE.
    "BENCH_35B": {
        "description": "35B-A3B Q3_K_XL all-hot + Modal BF16 drafter; ring=65536 gated spec-decode temp=0",
        "model_target":  Path("/home/peppi/models/qwen3.6-35b-a3b/Qwen3.6-35B-A3B-UD-Q3_K_XL.gguf"),
        "model_draft":   Path("/home/peppi/models/qwen3.6-35b-a3b-dflash-new/qwen3.6-35b-a3b-dflash-new-bf16-reconv.gguf"),
        "temperature": 0.0,
        "kv_cache_types": ("f16", "f16"),
        "extra_args": [
            "--draft", str(Path("/home/peppi/models/qwen3.6-35b-a3b-dflash-new/qwen3.6-35b-a3b-dflash-new-bf16-reconv.gguf")),
            "--draft-swa", "2048",
        ],
        "env": {
            "DFLASH_FEAT_RING_CAP": "65536",
            "DFLASH_SPEC_GATE": "1",
            "DFLASH_DRAFT_CTX_MAX": "2048",
        },
    },

    # BENCH_27B: 27B dense Q4_K_M + Lucebox BF16 drafter.
    #   ring=16384: 27B uses fc_in=25600; ring=65536 would allocate 6.7 GB and OOM.
    #   16384*25600*4 = 1.7 GB (same cap as A_baseline default, proven-working config).
    #   gate=1, temp=0.
    "BENCH_27B": {
        "description": "27B Q4_K_M dense + Lucebox drafter; q4_0 KV, cap=2048 cliff fix, ring=4096, gated temp=0",
        "model_target":  Path("/home/peppi/models/qwen3.6-27b-q4km/Qwen3.6-27B-Q4_K_M.gguf"),
        "model_draft":   Path("/home/peppi/models/qwen3.6-27b-dflash/dflash-draft-3.6-bf16-reconverted.gguf"),
        "temperature": 0.0,
        "kv_cache_types": ("q4_0", "q4_0"),
        "extra_args": [
            "--draft", str(Path("/home/peppi/models/qwen3.6-27b-dflash/dflash-draft-3.6-bf16-reconverted.gguf")),
            "--draft-swa", "2048",
        ],
        "env": {
            # draft_ctx capped at 2048 (cliff fix) -> ring only needs ~4096, not max_ctx.
            # 4096*25600*2(bf16) = 0.2 GB (was 16384 -> 1.7 GB); dodges the 27B ring-OOM.
            "DFLASH_FEAT_RING_CAP": "4096",
            "DFLASH_SPEC_GATE": "1",
            "DFLASH_DRAFT_CTX_MAX": "2048",
        },
    },

    # ── 3-way H2H arms ──────────────────────────────────────────────────────
    # lucebox: dFlash server + 35B-A3B Q4_K_M (trunk-only) target + dFlash drafter.
    #   Validated 192117 config: restore-consume snapshot win + ddtree speculation.
    #   GGML_CUDA_NO_VMM in env is idempotent with the dflash launch-path default.
    "lucebox": {
        "description": "dFlash 35B-A3B Q4_K_M + dFlash drafter; spec-gate, kvflash=8192, ring=131072, ddtree",
        "server": "dflash",
        "model_target": TGT_35B,
        "model_draft":  DD_35B,
        "kv_cache_types": ("q4_0", "q4_0"),
        "extra_args": [
            "--draft", str(DD_35B),
            "--draft-swa", "2048",
            "--kvflash", "8192",
            "--ddtree", "--ddtree-budget", "16",
        ],
        "env": {
            "GGML_CUDA_NO_VMM": "1",
            "KVFLASH_RESTORE_CONSUME": "1",
            "DFLASH_SPEC_GATE": "1",
            "DFLASH_FEAT_RING_CAP": "131072",
            "DFLASH_DRAFT_CTX_MAX": "2048",
        },
    },

    "lucebox_spark16_lazy": {
        "description": "dFlash 35B-A3B + dFlash drafter; Spark 16 GiB headroom, lazy draft, kvflash=8192, ddtree",
        "server": "dflash",
        "model_target": TGT_35B,
        "model_draft":  DD_35B,
        "kv_cache_types": ("q4_0", "q4_0"),
        "extra_args": [
            "--draft", str(DD_35B),
            "--draft-swa", "2048",
            "--spark", "--spark-vram", "16",
            "--kvflash", "8192",
            "--ddtree", "--ddtree-budget", "16",
            "--lazy-draft",
        ],
        "env": {
            "GGML_CUDA_NO_VMM": "1",
            "KVFLASH_RESTORE_CONSUME": "1",
            "DFLASH_SPEC_GATE": "1",
            "DFLASH_FEAT_RING_CAP": "16384",
            "DFLASH_DRAFT_CTX_MAX": "2048",
        },
    },

    # lucebox_ddtree22: identical to lucebox EXCEPT --ddtree-budget is 22 (sweep).
    #   Controlled ddtree-budget sweep: 16 vs 22, everything else identical.
    "lucebox_ddtree22": {
        "description": "dFlash 35B-A3B Q4_K_M + dFlash drafter; spec-gate, kvflash=8192, ring=131072, ddtree budget=22",
        "server": "dflash",
        "model_target": TGT_35B,
        "model_draft":  DD_35B,
        "kv_cache_types": ("q4_0", "q4_0"),
        "extra_args": [
            "--draft", str(DD_35B),
            "--draft-swa", "2048",
            "--kvflash", "8192",
            "--ddtree", "--ddtree-budget", "22",
        ],
        "env": {
            "GGML_CUDA_NO_VMM": "1",
            "KVFLASH_RESTORE_CONSUME": "1",
            "DFLASH_SPEC_GATE": "1",
            "DFLASH_FEAT_RING_CAP": "131072",
            "DFLASH_DRAFT_CTX_MAX": "2048",
        },
    },

    # lucebox_no_ddtree: identical to lucebox EXCEPT ddtree is removed.
    #   Controlled ddtree A/B: same target, drafter, kvflash=8192,
    #   KVFLASH_RESTORE_CONSUME=1, DFLASH_SPEC_GATE=1, ring=131072, q4_0 KV.
    #   Differs ONLY by absence of --ddtree --ddtree-budget 16.
    "lucebox_no_ddtree": {
        "description": "dFlash 35B-A3B Q4_K_M + dFlash drafter; spec-gate, kvflash=8192, ring=131072, NO ddtree",
        "server": "dflash",
        "model_target": TGT_35B,
        "model_draft":  DD_35B,
        "kv_cache_types": ("q4_0", "q4_0"),
        "extra_args": [
            "--draft", str(DD_35B),
            "--draft-swa", "2048",
            "--kvflash", "8192",
        ],
        "env": {
            "GGML_CUDA_NO_VMM": "1",
            "KVFLASH_RESTORE_CONSUME": "1",
            "DFLASH_SPEC_GATE": "1",
            "DFLASH_FEAT_RING_CAP": "131072",
            "DFLASH_DRAFT_CTX_MAX": "2048",
        },
    },

    # llama_cpp_mtp: llama.cpp b9781 + MTP bundled GGUF (NextN head embedded).
    #   Speculation via --spec-type draft-mtp, draft-n-max 2.
    "llama_cpp_mtp": {
        "description": "llama.cpp b9781 MTP spec-decode (draft-mtp, n-max=2); q4_0 KV",
        "server": "llama_cpp",
        "binary": str(LLAMA_BIN),
        "model":  str(MTP_GGUF),
        "kv_cache_types": ("q4_0", "q4_0"),
        "extra_args": [
            "--spec-type", "draft-mtp",
            "--spec-draft-n-max", "2",
        ],
        "env": {},
    },

    # llama_cpp_ar: llama.cpp b9781 plain autoregressive baseline (no speculation).
    #   MTP GGUF used so weight match is exact vs llama_cpp_mtp; NextN head unused.
    "llama_cpp_ar": {
        "description": "llama.cpp b9781 autoregressive baseline (no speculation); q4_0 KV",
        "server": "llama_cpp",
        "binary": str(LLAMA_BIN),
        "model":  str(MTP_GGUF),
        "kv_cache_types": ("q4_0", "q4_0"),
        "extra_args": [],
        "env": {},
    },

    "llama_cpp_mtp_27b": {
        "description": "llama.cpp dense 27B MTP spec-decode (draft-mtp, n-max=2); q4_0 KV",
        "server": "llama_cpp",
        "binary": str(LLAMA_BIN),
        "model":  str(MTP_27B_GGUF),
        "kv_cache_types": ("q4_0", "q4_0"),
        "extra_args": [
            "--spec-type", "draft-mtp",
            "--spec-draft-n-max", "2",
        ],
        "env": {},
    },

    "llama_cpp_ar_27b_mtp": {
        "description": "llama.cpp dense 27B autoregressive baseline using MTP GGUF; q4_0 KV",
        "server": "llama_cpp",
        "binary": str(LLAMA_BIN),
        "model":  str(MTP_27B_GGUF),
        "kv_cache_types": ("q4_0", "q4_0"),
        "extra_args": [],
        "env": {},
    },

    # lucebox_ar: dflash_server pure autoregressive baseline.
    #   Same model+KV as the lucebox family (35B-A3B Q4_K_M, q4_0 KV) but with
    #   NO drafter, NO kvflash, NO ddtree, NO spec env vars.
    #   env={} — the launcher unconditionally injects GGML_CUDA_NO_VMM=1; all
    #   DFLASH_SPEC_GATE / FEAT_RING_CAP / DRAFT_CTX_MAX / RESTORE_CONSUME dropped.
    #   Isolates dflash_server's raw AR decode speed on the same model+KV+ctx
    #   so we can compare directly to llama_cpp_ar (139.8 tok/s on stock b9781).
    "lucebox_ar": {
        "description": "dflash_server 35B-A3B Q4_K_M pure AR (no draft/spec/kvflash); q4_0 KV",
        "server": "dflash",
        "model_target": TGT_35B,
        "kv_cache_types": ("q4_0", "q4_0"),
        "extra_args": [],
        "env": {},
    },

    # === A-series AR-parity arms: pure AR both sides, NO MTP/spec, Q4_K_M, q4_0 KV. ===
    # llama side MUST pass -fa on (quantized KV requires FA; lucebox uses FA natively).
    # Both engines load the SAME Q4_K_M gguf per model => true like-for-like same-quant.
    "AR_27B": {
        "description": "dflash 27B dense Q4_K_M pure AR (no draft/spec); q4_0 KV",
        "server": "dflash",
        "model_target": TGT,                       # 27B Q4_K_M
        "kv_cache_types": ("q4_0", "q4_0"),
        "extra_args": [],
        "env": {},
    },
    "AR_27B_KVF": {
        "description": "dflash 27B dense Q4_K_M + KVFlash pooled (qk, 8192); pure AR; q4_0 KV; prefix slots=8",
        "server": "dflash",
        "model_target": TGT,
        "kv_cache_types": ("q4_0", "q4_0"),
        "extra_args": [
            "--kvflash", "8192", "--kvflash-policy", "qk",
            "--prefix-cache-slots", "8",
        ],
        "env": {"KVFLASH_RESTORE_CONSUME": "1"},
    },
    "AR_27B_KVF_DISK": {
        "description": "dflash 27B dense Q4_K_M + KVFlash + disk prefix cache; pure AR; q4_0 KV",
        "server": "dflash",
        "model_target": TGT,
        "kv_cache_dir": "/tmp/ar27b_kvf_disk",
        "kv_cache_types": ("q4_0", "q4_0"),
        "extra_args": [
            "--kvflash", "8192", "--kvflash-policy", "qk",
            "--disk-prefix-cache", "auto",
            "--kv-cache-dir", "/tmp/ar27b_kvf_disk",
        ],
        "env": {"KVFLASH_RESTORE_CONSUME": "1"},
    },
    "AR_27B_KVF_PFLASH": {
        "description": "dflash 27B dense Q4_K_M + KVFlash + pFlash/FlowKV prefill compression; pure AR; q4_0 KV",
        "server": "dflash",
        "model_target": TGT,
        "kv_cache_types": ("q4_0", "q4_0"),
        "extra_args": [
            "--kvflash", "8192", "--kvflash-policy", "qk",
            "--prefill-drafter", str(PD),
            "--prefill-compression", "always",
            "--prefill-keep-ratio", "0.10",
            "--prefill-threshold", "1000",
            "--prefill-skip-park",
        ],
        "env": {
            "KVFLASH_RESTORE_CONSUME": "1",
            "PFLASH_FREEZE_HISTORY": "1",
            "PFLASH_FREEZE_HOT_WINDOW": "2",
        },
    },
    "AR_27B_KVF_PFLASH_EE7": {
        "description": "AR_27B_KVF_PFLASH with pFlash drafter early-exit scoring at layer 7; pure AR; q4_0 KV",
        "server": "dflash",
        "model_target": TGT,
        "kv_cache_types": ("q4_0", "q4_0"),
        "extra_args": [
            "--kvflash", "8192", "--kvflash-policy", "qk",
            "--prefill-drafter", str(PD),
            "--prefill-compression", "always",
            "--prefill-keep-ratio", "0.10",
            "--prefill-threshold", "1000",
            "--prefill-skip-park",
        ],
        "env": {
            "KVFLASH_RESTORE_CONSUME": "1",
            "PFLASH_FREEZE_HISTORY": "1",
            "PFLASH_FREEZE_HOT_WINDOW": "2",
            "PFLASH_DRAFTER_EARLY_EXIT_N": "7",
            "PFLASH_DRAFTER_SCORE_LAYERS": "7",
        },
    },
    "AR_27B_KVF_FLOWKV_EE7": {
        "description": "dflash 27B dense Q4_K_M + KVFlash + FlowKV disk-cache compression; pFlash EE7; pure AR; q4_0 KV",
        "server": "dflash",
        "model_target": TGT,
        "kv_cache_dir": "/tmp/ar27b_kvf_flowkv_ee7",
        "kv_cache_types": ("q4_0", "q4_0"),
        "extra_args": [
            "--kvflash", "8192", "--kvflash-policy", "qk",
            "--prefill-drafter", str(PD),
            "--prefill-compression", "always",
            "--prefill-keep-ratio", "0.10",
            "--prefill-threshold", "1000",
            "--prefill-skip-park",
            "--disk-prefix-cache", "auto",
            "--disk-prefix-cache-compress",
            "--kv-cache-dir", "/tmp/ar27b_kvf_flowkv_ee7",
        ],
        "env": {
            "KVFLASH_RESTORE_CONSUME": "1",
            "PFLASH_FREEZE_HISTORY": "1",
            "PFLASH_FREEZE_HOT_WINDOW": "2",
            "PFLASH_DRAFTER_EARLY_EXIT_N": "7",
            "PFLASH_DRAFTER_SCORE_LAYERS": "7",
        },
    },
    "AR_35B": {
        "description": "dflash 35B-A3B MoE Q4_K_M pure AR (no draft/spec); q4_0 KV",
        "server": "dflash",
        "model_target": TGT_35B,                   # 35B-A3B Q4_K_M
        "kv_cache_types": ("q4_0", "q4_0"),
        "extra_args": [],
        "env": {},
    },
    "AR_LLAMA_27B": {
        "description": "llama.cpp b9781 pure AR, 27B Q4_K_M, -fa on, q4_0 KV (FAIR parity)",
        "server": "llama_cpp",
        "binary": str(LLAMA_BIN),
        "model": str(TGT),                          # same 27B Q4_K_M gguf as AR_27B
        "kv_cache_types": ("q4_0", "q4_0"),
        "extra_args": ["-fa", "on"],
        "env": {},
    },
    "AR_LLAMA_27B_SLOTCACHE": {
        "description": "llama.cpp 27B Q4_K_M + prompt/cache-reuse/checkpoints + slot save/restore; pure AR; q4_0 KV",
        "server": "llama_cpp",
        "binary": str(LLAMA_BIN),
        "model": str(TGT),
        "kv_cache_types": ("q4_0", "q4_0"),
        "slot_save_dir": "/tmp/llama27b_slot_cache",
        "slot_save_file": "ar27b_slot.bin",
        "extra_args": [
            "-fa", "on",
            "--cache-prompt",
            "--cache-reuse", "256",
            "--ctx-checkpoints", "64",
            "--checkpoint-min-step", "512",
            "--cache-ram", "8192",
            "--cache-idle-slots",
            "--slot-save-path", "/tmp/llama27b_slot_cache",
        ],
        "env": {},
    },
    "AR_LLAMA_35B": {
        "description": "llama.cpp b9781 pure AR, 35B-A3B Q4_K_M, -fa on, q4_0 KV (FAIR parity)",
        "server": "llama_cpp",
        "binary": str(LLAMA_BIN),
        "model": str(TGT_35B),                      # same 35B Q4_K_M gguf as AR_35B
        "kv_cache_types": ("q4_0", "q4_0"),
        "extra_args": ["-fa", "on"],
        "env": {},
    },
    "AR_LLAMA_35B_SLOTCACHE": {
        "description": "llama.cpp 35B-A3B Q4_K_M + prompt/cache-reuse/checkpoints + slot save/restore; pure AR; q4_0 KV",
        "server": "llama_cpp",
        "binary": str(LLAMA_BIN),
        "model": str(TGT_35B),
        "kv_cache_types": ("q4_0", "q4_0"),
        "slot_save_dir": "/tmp/llama35b_slot_cache",
        "slot_save_file": "ar35b_slot.bin",
        "extra_args": [
            "-fa", "on",
            "--cache-prompt",
            "--cache-reuse", "256",
            "--ctx-checkpoints", "64",
            "--checkpoint-min-step", "512",
            "--cache-ram", "8192",
            "--cache-idle-slots",
            "--slot-save-path", "/tmp/llama35b_slot_cache",
        ],
        "env": {},
    },
    "AR_35B_KVF": {
        "description": "dflash 35B-A3B MoE Q4_K_M + KVFlash requested (auto-disabled when all-hot fits) + consume-restored; NO spec; q4_0 KV",
        "server": "dflash",
        "model_target": TGT_35B,
        "kv_cache_types": ("q4_0", "q4_0"),
        "extra_args": ["--kvflash", "8192", "--kvflash-policy", "qk"],
        "env": {"KVFLASH_RESTORE_CONSUME": "1"},
    },
    "AR_35B_KVF_OPENAI": {
        "description": "dflash 35B-A3B MoE Q4_K_M + KVFlash requested; OpenAI chat path; NO spec; q4_0 KV",
        "server": "dflash_openai",
        "model_target": TGT_35B,
        "kv_cache_types": ("q4_0", "q4_0"),
        "extra_args": ["--kvflash", "8192", "--kvflash-policy", "qk"],
        "env": {"KVFLASH_RESTORE_CONSUME": "1"},
    },
    "AR_35B_KVF_FORCE": {
        "description": "dflash 35B-A3B MoE Q4_K_M + forced KVFlash pooled-QK decode; pure AR; q4_0 KV",
        "server": "dflash",
        "model_target": TGT_35B,
        "kv_cache_types": ("q4_0", "q4_0"),
        "extra_args": ["--kvflash", "8192", "--kvflash-policy", "qk", "--kvflash-force"],
        "env": {"KVFLASH_RESTORE_CONSUME": "1"},
    },
    "AR_35B_KVF_FORCE_OPENAI": {
        "description": "dflash 35B-A3B MoE Q4_K_M + forced KVFlash pooled-QK decode; pure AR; OpenAI chat path",
        "server": "dflash_openai",
        "model_target": TGT_35B,
        "kv_cache_types": ("q4_0", "q4_0"),
        "extra_args": ["--kvflash", "8192", "--kvflash-policy", "qk", "--kvflash-force"],
        "env": {"KVFLASH_RESTORE_CONSUME": "1"},
    },
    "AR_35B_KVF_DISK": {
        "description": "dflash 35B-A3B MoE Q4_K_M + KVFlash + disk prefix cache; pure AR; q4_0 KV",
        "server": "dflash",
        "model_target": TGT_35B,
        "kv_cache_dir": "/tmp/ar35b_kvf_disk",
        "kv_cache_types": ("q4_0", "q4_0"),
        "extra_args": [
            "--kvflash", "8192", "--kvflash-policy", "qk",
            "--disk-prefix-cache", "auto",
            "--kv-cache-dir", "/tmp/ar35b_kvf_disk",
        ],
        "env": {"KVFLASH_RESTORE_CONSUME": "1"},
    },
    "AR_35B_SPARK_KVF": {
        "description": "dflash 35B-A3B MoE Q4_K_M + constrained Spark (20 GiB) + KVFlash pooled (qk, 8192); pure AR; q4_0 KV",
        "server": "dflash",
        "model_target": TGT_35B,
        "kv_cache_types": ("q4_0", "q4_0"),
        "extra_args": [
            "--spark", "--spark-vram", "20",
            "--kvflash", "8192", "--kvflash-policy", "qk",
        ],
        "env": {"KVFLASH_RESTORE_CONSUME": "1"},
    },
    "AR_35B_SPARK_KVF_DISK": {
        "description": "dflash 35B-A3B MoE Q4_K_M + constrained Spark (20 GiB) + KVFlash + disk prefix cache; pure AR; q4_0 KV",
        "server": "dflash",
        "model_target": TGT_35B,
        "kv_cache_dir": "/tmp/ar35b_spark_kvf_disk",
        "kv_cache_types": ("q4_0", "q4_0"),
        "extra_args": [
            "--spark", "--spark-vram", "20",
            "--kvflash", "8192", "--kvflash-policy", "qk",
            "--disk-prefix-cache", "auto",
            "--kv-cache-dir", "/tmp/ar35b_spark_kvf_disk",
        ],
        "env": {"KVFLASH_RESTORE_CONSUME": "1"},
    },
    "AR_35B_SPARK_KVF_PFLASH": {
        "description": "dflash 35B-A3B MoE Q4_K_M + constrained Spark (20 GiB) + KVFlash + pFlash/FlowKV; pure AR; q4_0 KV",
        "server": "dflash",
        "model_target": TGT_35B,
        "kv_cache_types": ("q4_0", "q4_0"),
        "extra_args": [
            "--spark", "--spark-vram", "20",
            "--kvflash", "8192", "--kvflash-policy", "qk",
            "--prefill-drafter", str(PD),
            "--prefill-compression", "always",
            "--prefill-keep-ratio", "0.10",
            "--prefill-threshold", "1000",
            "--prefill-skip-park",
        ],
        "env": {
            "KVFLASH_RESTORE_CONSUME": "1",
            "PFLASH_FREEZE_HISTORY": "1",
            "PFLASH_FREEZE_HOT_WINDOW": "2",
        },
    },
}

# ---------------------------------------------------------------------------
# Log parsing regexes
# These match the real dflash_server log format confirmed from arm1_baseline logs
# ---------------------------------------------------------------------------

# [server] chat CACHE ... restore=true|false slot=N prefix_len=N effective_prompt=N ...
CACHE_RE = re.compile(
    r'\[server\] chat CACHE\s+\S+\s+'
    r'restore=(\w+)\s+slot=(-?\d+)\s+prefix_len=(\d+)\s+effective_prompt=(\d+)'
    r'(?:\s+prefix_key_len=(\d+))?'
    r'(?:\s+pflash=(\w+))?'
)

# [server] chat DONE ... in=N effective_in=N out=N ... prefix_len=N prefill=Xs decode=Xs(Ytok/s) ...
DONE_RE = re.compile(
    r'\[server\] chat DONE\s+\S+\s+ok=(\w+)\s+'
    r'in=(\d+)\s+effective_in=(\d+)\s+out=(\d+)\s+'
    r'[\d.]+s\s+[\d.]+\s+tok/s\s+finish=(\S+)\s+'
    r'restore=(\w+)\s+slot=(-?\d+)\s+prefix_len=(\d+)\s+'
    r'prefill=([\d.]+)s\s+decode=([\d.]+)s\(([\d.]+)tok/s\)'
    r'(?:\s+error=(\S+))?'
)

# [spec-decode] tokens=N time=Xs speed=Ytok/s steps=N accepted=A/T (P%) avg_commit=C
SPEC_RE = re.compile(
    r'\[spec-decode\] tokens=(\d+) time=([\d.]+) s speed=([\d.]+) tok/s '
    r'steps=\d+ accepted=\d+/\d+ \(([\d.]+)%\) avg_commit=([\d.]+)'
)

# [ar-decode] tokens=N time=Xs speed=Ytok/s
AR_RE = re.compile(
    r'\[ar-decode\] tokens=(\d+) time=([\d.]+) s speed=([\d.]+) tok/s'
)

# [pflash] N -> M -> K tokens (P% kept)
PFLASH_KEPT_RE = re.compile(
    r'\[pflash\] (\d+) -> \d+ -> (\d+) tokens \(([\d.]+)% kept\)'
)

# [pflash] query survival: A/B (P%)
SURVIVAL_RE = re.compile(
    r'\[pflash\] query survival: (\d+)/(\d+) \(([\d.]+)%\)'
)

# llama.cpp MTP per-slot acceptance line (stderr):
#   I slot print_timing: id N | task N | draft acceptance = 0.91176 (   31 accepted /    34 generated), ...
LLAMA_MTP_RE = re.compile(
    r'slot print_timing:.*?draft acceptance\s*=\s*([\d.]+)\s*\(\s*(\d+)\s*accepted\s*/\s*(\d+)\s*generated\)'
)


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def git_info(repo: Path) -> dict:
    try:
        branch = subprocess.check_output(
            ["git", "rev-parse", "--abbrev-ref", "HEAD"],
            cwd=repo, text=True, stderr=subprocess.DEVNULL,
        ).strip()
        commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=repo, text=True, stderr=subprocess.DEVNULL,
        ).strip()
        return {"branch": branch, "commit": commit}
    except Exception:
        return {"branch": "unknown", "commit": "unknown"}


def nvidia_smi_vram() -> tuple[Optional[int], Optional[int]]:
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=memory.used,memory.total",
             "--format=csv,noheader,nounits"],
            text=True, timeout=10,
        ).strip()
        parts = out.split(",")
        return int(parts[0].strip()), int(parts[1].strip())
    except Exception:
        return None, None


def nvidia_smi_power(gpu_index: int = 0) -> dict:
    try:
        out = subprocess.check_output(
            [
                "nvidia-smi",
                f"--id={gpu_index}",
                "--query-gpu=power.draw,utilization.gpu,memory.used",
                "--format=csv,noheader,nounits",
            ],
            text=True,
            timeout=10,
        ).strip().splitlines()[0]
        watts, util, mem = [p.strip() for p in out.split(",")[:3]]
        return {
            "power_w": float(watts),
            "gpu_util_pct": int(float(util)),
            "memory_used_mib": int(float(mem)),
        }
    except Exception as ex:
        return {"error": str(ex)}


class GpuPowerMonitor:
    def __init__(self, interval_s: float, gpu_index: int = 0):
        self.interval_s = max(0.2, float(interval_s))
        self.gpu_index = gpu_index
        self.samples: list[dict] = []
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._start: Optional[float] = None
        self._end: Optional[float] = None

    def start(self) -> None:
        self._start = time.perf_counter()
        self._thread = threading.Thread(target=self._run, name="gpu-power-monitor", daemon=True)
        self._thread.start()

    def _run(self) -> None:
        while not self._stop.is_set():
            now = time.perf_counter()
            sample = nvidia_smi_power(self.gpu_index)
            sample["t_s"] = round(now - (self._start or now), 3)
            self.samples.append(sample)
            self._stop.wait(self.interval_s)

    def stop(self) -> dict:
        self._end = time.perf_counter()
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=max(1.0, self.interval_s * 2.0))

        valid = [
            s for s in self.samples
            if isinstance(s.get("power_w"), (int, float))
        ]
        duration = (
            self._end - self._start
            if self._start is not None and self._end is not None
            else 0.0
        )
        energy_j = None
        if len(valid) >= 2:
            energy = 0.0
            for a, b in zip(valid, valid[1:]):
                dt = max(0.0, float(b["t_s"]) - float(a["t_s"]))
                energy += dt * (float(a["power_w"]) + float(b["power_w"])) / 2.0
            energy_j = round(energy, 3)
        elif len(valid) == 1 and duration > 0:
            energy_j = round(float(valid[0]["power_w"]) * duration, 3)

        avg_power = (
            round(energy_j / duration, 3)
            if energy_j is not None and duration > 0
            else None
        )
        return {
            "gpu_index": self.gpu_index,
            "sample_interval_s": self.interval_s,
            "duration_s": round(duration, 3),
            "sample_count": len(self.samples),
            "valid_sample_count": len(valid),
            "energy_j": energy_j,
            "avg_power_w": avg_power,
            "max_power_w": round(max(float(s["power_w"]) for s in valid), 3) if valid else None,
            "mean_gpu_util_pct": round(mean([float(s["gpu_util_pct"]) for s in valid]), 3) if valid else None,
            "max_memory_used_mib": max(int(s["memory_used_mib"]) for s in valid) if valid else None,
            "first_error": next((s.get("error") for s in self.samples if s.get("error")), None),
            "samples": self.samples,
        }


def tail_log_lines(log_path: Path, n: int = 5) -> list[str]:
    try:
        return log_path.read_text(errors="replace").splitlines()[-n:]
    except Exception:
        return []


def wait_for_server(
    port: int,
    timeout: int = 360,
    proc: Optional[subprocess.Popen] = None,
    log_path: Optional[Path] = None,
) -> tuple[bool, Optional[str]]:
    deadline = time.time() + timeout
    while time.time() < deadline:
        if proc is not None and proc.poll() is not None:
            lines = tail_log_lines(log_path, 5) if log_path is not None else []
            detail = [f"server exited before health check passed (rc={proc.returncode})"]
            if lines:
                detail.append("last 5 log lines:")
                detail.extend(f"  {line}" for line in lines)
            return False, "\n".join(detail)
        try:
            with urllib.request.urlopen(
                f"http://{HOST}:{port}/health", timeout=3
            ) as r:
                if r.status == 200:
                    return True, None
        except Exception:
            pass
        time.sleep(2)
    lines = tail_log_lines(log_path, 5) if log_path is not None else []
    detail = [f"server did not become healthy within {timeout}s"]
    if lines:
        detail.append("last 5 log lines:")
        detail.extend(f"  {line}" for line in lines)
    return False, "\n".join(detail)


def build_llama_cpp_cmd(arm_cfg: dict, port: int, max_ctx: int) -> list[str]:
    """Build the CLI for a llama_cpp arm. Pure function — testable without GPU."""
    binary = arm_cfg.get("binary", str(LLAMA_BIN))
    model  = arm_cfg.get("model",  str(MTP_GGUF))
    ctk, ctv = arm_cfg.get("kv_cache_types", ("q4_0", "q4_0"))
    return [
        binary,
        "-m", model,
        "--host", HOST,
        "--port", str(port),
        "-c", str(max_ctx),
        "-ngl", "999",
        "--cache-type-k", ctk,
        "--cache-type-v", ctv,
        "--chat-template-file", str(TMPL),
        "--jinja",
    ] + arm_cfg.get("extra_args", [])


def parse_llama_cpp_timings(timings: dict) -> dict:
    """Extract prefill/decode metrics from llama.cpp timings dict. Pure function."""
    m: dict = {}
    pn  = timings.get("prompt_n", 0)
    pms = timings.get("prompt_ms", 0.0)
    dn  = timings.get("predicted_n", 0)
    dms = timings.get("predicted_ms", 0.0)
    if pms > 0:
        m["prefill_s"]   = pms / 1000.0
        m["prompt_tokens"] = pn
        if pn > 0:
            m["prefill_tps"] = round(pn / (pms / 1000.0), 1)
    if dms > 0:
        m["decode_s"]   = dms / 1000.0
        m["out_tokens"] = dn
        if dn > 0:
            m["decode_tps"] = round(dn / (dms / 1000.0), 1)
    return m


def parse_llama_cpp_accept(
    log_text: str,
    mtp_offset: int,
    extra_args: list,
) -> tuple[dict, int]:
    """Parse llama.cpp MTP draft-acceptance from server stderr log.

    Returns (metrics_dict, new_mtp_offset).

    If --spec-type draft-mtp is present in extra_args and the log contains a
    matching line, returns accept_rate (0-100 float), spec_engaged=True, and
    decode_mode='spec'.  If spec-type args present but no log line yet found,
    returns spec_engaged=True with accept_rate=None (not 0.0 — avoids false neg).
    If spec-type args absent, returns {} (AR arm, no speculation).
    """
    has_mtp = "--spec-type" in extra_args and "draft-mtp" in extra_args
    if not has_mtp:
        return {}, mtp_offset

    all_matches = LLAMA_MTP_RE.findall(log_text)
    new_matches = all_matches[mtp_offset:]

    if not new_matches:
        # MTP arm but log line not yet found — mark as spec_engaged but rate unknown
        return {"spec_engaged": True, "decode_mode": "spec"}, mtp_offset

    m_frac, m_acc, m_gen = new_matches[0]
    accept_rate = round(float(m_frac) * 100.0, 2)
    accepted    = int(m_acc)
    generated   = int(m_gen)
    return {
        "spec_engaged": True,
        "decode_mode": "spec",
        "accept_rate": accept_rate,
        "draft_accepted": accepted,
        "draft_generated": generated,
    }, mtp_offset + 1


_gpu_lock_fd: Optional[int] = None


def acquire_gpu_lock() -> None:
    global _gpu_lock_fd
    _gpu_lock_fd = open(GPU_LOCK_FILE, "w")
    fcntl.flock(_gpu_lock_fd, fcntl.LOCK_EX)
    print("[gpu-lock] acquired GPU lock")


def release_gpu_lock() -> None:
    global _gpu_lock_fd
    if _gpu_lock_fd is not None:
        fcntl.flock(_gpu_lock_fd, fcntl.LOCK_UN)
        _gpu_lock_fd.close()
        _gpu_lock_fd = None
        print("[gpu-lock] released GPU lock")


def launch_server(
    arm_name: str,
    arm_cfg: dict,
    log_path: Path,
    port: int = PORT,
    max_ctx: int = MAX_CTX,
    pin_decode_tokens: Optional[int] = None,
) -> tuple:
    """Start server for given arm. Returns (proc, log_file_handle).

    Branches on arm_cfg.get('server', 'dflash'):
      'dflash'    — existing dflash_server path (unchanged).
      'llama_cpp' — llama.cpp llama-server, OpenAI-compat API.
    """
    server_type = arm_cfg.get("server", "dflash")

    if server_type == "llama_cpp":
        cmd = build_llama_cpp_cmd(arm_cfg, port, max_ctx)
        env = dict(os.environ)
        # Inject CUDA .so directory so the binary finds libggml-cuda.so etc.
        cuda_lib_dir = str(LLAMA_CUDA_LIB)
        existing_ld = env.get("LD_LIBRARY_PATH", "")
        env["LD_LIBRARY_PATH"] = f"{cuda_lib_dir}:{existing_ld}" if existing_ld else cuda_lib_dir
        # No DFLASH_* / GGML_CUDA_NO_VMM for llama_cpp arms
        print(f"[server] Launching {arm_name} (llama_cpp): {' '.join(cmd)}")
        print(f"[server] LD_LIBRARY_PATH={env['LD_LIBRARY_PATH']}")
        log_f = log_path.open("w")
        proc = subprocess.Popen(cmd, env=env, stdout=log_f, stderr=log_f)
        print(f"[server] PID={proc.pid}  log={log_path}")
        return proc, log_f

    # --- dflash path (unchanged) ---
    env = dict(os.environ)
    env["GGML_CUDA_NO_VMM"] = "1"
    # DFlash decode optimizations (from dFlash drafter integration session):
    # - block_size=16 (model card sweet spot for Qwen3.6 DFlash)
    # - feature ring cap=16384 (default 4096 collapses acceptance at the ring boundary)
    env["DFLASH_DRAFT_BLOCK_SIZE"] = "16"
    env["DFLASH_FEAT_RING_CAP"] = "16384"
    env.update(arm_cfg["env"])
    if pin_decode_tokens is not None:
        env["DFLASH_MIN_TOKENS"] = str(pin_decode_tokens)
        env["DFLASH_DEGENERATE_RUN_TOKENS"] = "0"
        print(f"[pin-decode] DFLASH_MIN_TOKENS={pin_decode_tokens}")
        print("[pin-decode] DFLASH_DEGENERATE_RUN_TOKENS=0")

    # Per-arm model override (used by BENCH_35B / BENCH_27B / lucebox arms)
    target_model = arm_cfg.get("model_target", TGT)

    ctk, ctv = arm_cfg.get("kv_cache_types", ("tq3_0", "tq3_0"))
    cmd = [
        str(SERVER_BIN),
        str(target_model),
        "--host", HOST,
        "--port", str(port),
        "--max-ctx", str(max_ctx),
        "--cache-type-k", ctk,
        "--cache-type-v", ctv,
        "--chat-template-file", str(TMPL),
        "--model-name", "luce-dflash",
    ] + arm_cfg["extra_args"]

    print(f"[server] Launching {arm_name}: {' '.join(cmd)}")
    log_f = log_path.open("w")
    proc = subprocess.Popen(cmd, env=env, stdout=log_f, stderr=log_f)
    print(f"[server] PID={proc.pid}  log={log_path}")
    return proc, log_f


def kill_server(proc, log_f, wait_s: int = 10) -> None:
    if proc and proc.poll() is None:
        proc.terminate()
        try:
            proc.wait(timeout=wait_s)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait(timeout=5)
    if log_f:
        log_f.close()


# ---------------------------------------------------------------------------
# Log parsing: extract per-request metrics from server log
# ---------------------------------------------------------------------------

def parse_log_for_request(
    log_text: str,
    cache_offset: int,
    done_offset: int,
    spec_offset: int,
    ar_offset: int,
    pflash_offset: int,
    survival_offset: int,
) -> tuple[dict, int, int, int, int, int, int]:
    """Parse the NEXT request's metrics from log_text starting at given offsets.

    Returns (metrics_dict, new_cache_offset, new_done_offset, new_spec_offset,
             new_ar_offset, new_pflash_offset, new_survival_offset).
    """
    all_cache    = CACHE_RE.findall(log_text)
    all_done     = DONE_RE.findall(log_text)
    all_spec     = SPEC_RE.findall(log_text)
    all_ar       = AR_RE.findall(log_text)
    all_pflash   = PFLASH_KEPT_RE.findall(log_text)
    all_survival = SURVIVAL_RE.findall(log_text)

    new_cache    = all_cache[cache_offset:]
    new_done     = all_done[done_offset:]
    new_spec     = all_spec[spec_offset:]
    new_ar       = all_ar[ar_offset:]
    new_pflash   = all_pflash[pflash_offset:]
    new_survival = all_survival[survival_offset:]

    m: dict = {}

    # --- chat CACHE line ---
    if new_cache:
        c = new_cache[0]
        m["restore"]          = c[0] == "true"
        m["cache_slot"]       = int(c[1])
        m["prefix_len"]       = int(c[2])
        m["effective_prompt"] = int(c[3])
        if c[4]:
            m["prefix_key_len"] = int(c[4])
        if c[5]:
            m["pflash_active"] = c[5] == "true"

    # --- chat DONE line ---
    if new_done:
        d = new_done[0]
        m["chat_ok"]        = d[0] == "true"
        m["prompt_tokens"]  = int(d[1])
        m["effective_in"]   = int(d[2])
        m["out_tokens"]     = int(d[3])
        m["finish_reason"]  = d[4]
        m["restore_done"]   = d[5] == "true"
        m["prefix_len_done"] = int(d[7])
        m["prefill_s"]      = float(d[8])
        m["decode_s"]       = float(d[9])
        m["decode_tps"]     = float(d[10])
        if len(d) > 11 and d[11] and d[11] != "-":
            m["chat_error"] = d[11]
        # prefix_len from DONE is more reliable; override if we got it
        if "prefix_len" not in m:
            m["prefix_len"] = m["prefix_len_done"]
        # prefill TPS
        pt = m["prompt_tokens"]
        pfs = m["prefill_s"]
        if pt > 0 and pfs > 0:
            m["prefill_tps"] = round(pt / pfs, 1)
        # fresh prefill = effective tokens not served from prefix cache
        eff = m.get("effective_in", m.get("effective_prompt", pt))
        pl  = m.get("prefix_len", 0)
        m["fresh_prefill"] = max(0, eff - pl)
        # cache hit ratio
        if pt > 0:
            m["cache_hit_ratio"] = round(pl / pt, 4)
            if "prefix_key_len" in m:
                m["cache_key_hit_ratio"] = round(m["prefix_key_len"] / pt, 4)

    # --- spec or ar decode ---
    if new_spec:
        s = new_spec[0]
        m["decode_mode"]  = "spec"
        m["accept_rate"]  = float(s[3])
        m["avg_commit"]   = float(s[4])
        if "decode_tps" not in m:
            m["decode_tps"] = float(s[2])
    elif new_ar:
        a = new_ar[0]
        m["decode_mode"] = "ar"
        if "decode_tps" not in m:
            m["decode_tps"] = float(a[2])

    # --- pflash compression info ---
    if new_pflash:
        p = new_pflash[0]
        m["pflash_in_tokens"]   = int(p[0])
        m["pflash_kept_tokens"] = int(p[1])
        m["pflash_kept_pct"]    = float(p[2])
    if new_survival:
        sv = new_survival[0]
        m["query_survival_pct"] = float(sv[2])

    return (
        m,
        cache_offset    + (1 if new_cache    else 0),
        done_offset     + (1 if new_done     else 0),
        spec_offset     + (1 if new_spec     else 0),
        ar_offset       + (1 if new_ar       else 0),
        pflash_offset   + (1 if new_pflash   else 0),
        survival_offset + (1 if new_survival else 0),
    )


# ---------------------------------------------------------------------------
# Check tool_use in response
# ---------------------------------------------------------------------------

def extract_tool_call_names(body: dict) -> list[str]:
    """Return well-formed tool-call names from Anthropic or OpenAI responses."""
    names: list[str] = []
    content = body.get("content", [])
    if isinstance(content, list):
        for block in content:
            if not isinstance(block, dict) or block.get("type") != "tool_use":
                continue
            name = block.get("name", "")
            inp = block.get("input", {})
            if name and isinstance(inp, dict):
                names.append(name)

    choices = body.get("choices", [])
    if choices:
        msg = choices[0].get("message", {})
        for tc in msg.get("tool_calls", []) or []:
            if not isinstance(tc, dict):
                continue
            fn = tc.get("function", {})
            if isinstance(fn, dict) and fn.get("name"):
                names.append(fn["name"])
    return names


def check_tool_call(body: dict, expected_name: Optional[str] = None) -> tuple[bool, str]:
    """Return (valid, detail) for a structured tool call, optionally exact-name."""
    try:
        names = extract_tool_call_names(body)
        if expected_name:
            if expected_name in names:
                return True, f"tool={expected_name}"
            return False, f"expected={expected_name} got={names or 'none'}"
        if names:
            return True, f"tool={names[0]}"
    except Exception as e:
        return False, f"exception: {e}"
    return False, "no tool_use block"


def request_expects_tool_call(req_body: dict) -> Optional[bool]:
    for key in ("expect_tool_call", "tool_call_expected", "should_call_tool"):
        if key in req_body:
            return bool(req_body[key])
    return None


def extract_response_text(body: dict) -> str:
    """Return assistant text from either Anthropic or OpenAI-compatible responses."""
    try:
        content = body.get("content", [])
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            parts = [
                block.get("text", "")
                for block in content
                if isinstance(block, dict) and block.get("type") == "text"
            ]
            if parts:
                return "\n".join(parts)

        choices = body.get("choices", [])
        if choices:
            msg = choices[0].get("message", {})
            content = msg.get("content", "")
            if isinstance(content, str):
                return content
            if isinstance(content, list):
                return "\n".join(
                    block.get("text", "")
                    for block in content
                    if isinstance(block, dict)
                )
    except Exception:
        pass
    return ""


def check_charbench_quality(req_body: dict, body: dict) -> tuple[Optional[bool], Optional[str]]:
    """Small deterministic checks for the two charbench quality-gate prompts."""
    prompt = " ".join(
        str(msg.get("content", ""))
        for msg in req_body.get("messages", [])
        if isinstance(msg, dict)
    )
    text = extract_response_text(body)
    low = text.lower()

    if "def quicksort(arr)" in prompt:
        ok = (
            "def quicksort" in low
            and "return" in low
            and ("quicksort(" in low or "sort(" in low)
            and ("pivot" in low or "left" in low or "right" in low)
        )
        return ok, "code_complete: quicksort structure" if ok else "code_complete: missing quicksort structure"

    tools = req_body.get("tools", [])
    has_read_file_tool = any(
        isinstance(t, dict)
        and (
            t.get("name") == "read_file"
            or t.get("function", {}).get("name") == "read_file"
        )
        for t in tools
    )
    if (("read_file" in prompt or has_read_file_tool) and "/etc/hostname" in prompt):
        structured_tool, _ = check_tool_call(body)
        exact_text_tool = (
            "<tool_call>" in text
            and "read_file" in text
            and "/etc/hostname" in text
            and "\"path\"" in text
        )
        ok = structured_tool or exact_text_tool
        return ok, "tool_call: read_file request" if ok else "tool_call: missing read_file request"

    return None, None


def classify_request_failure(server_type: str, body: dict, metrics: dict) -> Optional[str]:
    """Return a benchmark-invalid reason when transport succeeded but generation failed."""
    if isinstance(body, dict) and body.get("error"):
        return f"response_error: {body.get('error')}"

    if is_dflash_server(server_type):
        finish = metrics.get("finish_reason")
        if metrics.get("chat_ok") is False or finish == "error":
            detail = metrics.get("chat_error") or finish or "unknown"
            return f"dflash_generation_failed: {detail}"

    return None


# ---------------------------------------------------------------------------
# Send one request
# ---------------------------------------------------------------------------

def _anthropic_to_openai_tools(tools: list) -> list:
    """Best-effort remap Anthropic tool defs to OpenAI function format."""
    out = []
    for t in tools:
        if not isinstance(t, dict):
            continue
        if t.get("type") == "function" and isinstance(t.get("function"), dict):
            out.append(t)
            continue
        fn: dict = {
            "name": t.get("name", ""),
            "description": t.get("description", ""),
            "parameters": t.get("input_schema", {}),
        }
        out.append({"type": "function", "function": fn})
    return out


def send_request(
    req_body: dict,
    port: int,
    server_type: str = "dflash",
    pin_decode_length: bool = False,
) -> tuple[dict, float, Optional[str]]:
    """POST request to server. Returns (response_body, wall_s, error_str).

    server_type='dflash'        -> POST /v1/messages  (Anthropic format)
    server_type='dflash_openai' -> POST /v1/chat/completions  (OpenAI format)
    server_type='llama_cpp'     -> POST /v1/chat/completions  (OpenAI format)
    """
    requested_max_tokens = request_max_tokens(req_body)
    if pin_decode_length and requested_max_tokens is None:
        return {}, 0.0, "pin_decode_length requires a positive integer max_tokens"

    if uses_openai_chat_api(server_type):
        # Convert Anthropic trace format -> OpenAI messages format
        sys_prompt = req_body.get("system")
        messages: list = []
        if sys_prompt:
            messages.append({"role": "system", "content": sys_prompt})
        for msg in req_body.get("messages", []):
            content = msg.get("content", "")
            if isinstance(content, list):
                # flatten content blocks to text (tool_use blocks skipped for S1)
                text_parts = [
                    b.get("text", "") for b in content
                    if isinstance(b, dict) and b.get("type") == "text"
                ]
                content = " ".join(text_parts)
            out_msg = {"role": msg["role"], "content": content}
            if "tool_calls" in msg:
                out_msg["tool_calls"] = msg["tool_calls"]
            if "tool_call_id" in msg:
                out_msg["tool_call_id"] = msg["tool_call_id"]
            messages.append(out_msg)

        body_out: dict = {
            "model": req_body.get("model", "luce-dflash"),
            "messages": messages,
            "temperature": req_body.get("temperature", 0),
            "max_tokens": req_body.get("max_tokens", 1024),
            "stream": False,
        }
        if pin_decode_length:
            body_out["max_tokens"] = requested_max_tokens
            if server_type == "llama_cpp":
                # llama.cpp treats max_tokens as an upper bound unless EOS is masked.
                # n_predict wins if both aliases are present in this server build.
                body_out["n_predict"] = requested_max_tokens
                body_out["ignore_eos"] = True
        raw_tools = req_body.get("tools")
        if raw_tools:
            try:
                body_out["tools"] = _anthropic_to_openai_tools(raw_tools)
            except Exception:
                pass  # best-effort; S1 trace has no tools
        if "tool_choice" in req_body:
            body_out["tool_choice"] = req_body["tool_choice"]

        url = f"http://{HOST}:{port}/v1/chat/completions"
        payload = json.dumps(body_out, ensure_ascii=False).encode("utf-8")
        http_req = urllib.request.Request(
            url, data=payload,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        t0 = time.time()
        try:
            with urllib.request.urlopen(http_req, timeout=900) as r:
                resp = json.loads(r.read())
            return resp, time.time() - t0, None
        except urllib.error.HTTPError as ex:
            detail = ex.read().decode("utf-8", errors="replace").strip()
            return {}, time.time() - t0, f"HTTP {ex.code}: {detail[:1000]}"
        except Exception as ex:
            return {}, time.time() - t0, str(ex)

    # --- dflash / Anthropic path (unchanged) ---
    payload = json.dumps(req_body, ensure_ascii=False).encode("utf-8")
    http_req = urllib.request.Request(
        f"http://{HOST}:{port}/v1/messages",
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    t0 = time.time()
    try:
        with urllib.request.urlopen(http_req, timeout=900) as r:
            body = json.loads(r.read())
        return body, time.time() - t0, None
    except urllib.error.HTTPError as ex:
        detail = ex.read().decode("utf-8", errors="replace").strip()
        return {}, time.time() - t0, f"HTTP {ex.code}: {detail[:1000]}"
    except Exception as ex:
        return {}, time.time() - t0, str(ex)


def request_max_tokens(req_body: dict) -> Optional[int]:
    """Return a positive integer request max_tokens, or None when unusable."""
    raw = req_body.get("max_tokens")
    if raw is None or isinstance(raw, bool):
        return None
    try:
        value = int(raw)
    except (TypeError, ValueError):
        return None
    return value if value > 0 else None


def out_tokens_match_requested(metrics: dict, requested_out_tokens: Optional[int]) -> Optional[bool]:
    if requested_out_tokens is None or metrics.get("out_tokens") is None:
        return None
    try:
        actual = int(round(float(metrics["out_tokens"])))
    except (TypeError, ValueError):
        return None
    return actual == requested_out_tokens


def llama_slot_action(
    port: int,
    action: str,
    filename: str,
    slot_id: int = 0,
) -> tuple[bool, str]:
    """Call llama.cpp slot save/restore action for cross-process cache tests."""
    url = f"http://{HOST}:{port}/slots/{slot_id}?action={action}"
    payload = json.dumps({"filename": filename}, ensure_ascii=False).encode("utf-8")
    http_req = urllib.request.Request(
        url,
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(http_req, timeout=120) as r:
            body = r.read().decode("utf-8", errors="replace")
            ok = 200 <= r.status < 300
            return ok, body[:500]
    except Exception as ex:
        return False, str(ex)


# ---------------------------------------------------------------------------
# Run one repeat of the trace
# ---------------------------------------------------------------------------

def run_trace_repeat(
    turns: list[dict],
    port: int,
    log_path: Path,
    arm_name: str,
    smoke: bool = False,
    repeat_idx: int = 0,
    arm_cfg: Optional[dict] = None,
    pin_decode_length: bool = False,
    seed_to_send: Optional[int] = None,
) -> list[dict]:
    """Run all turns in the trace; parse server log per turn.

    Returns list of per-turn metric dicts.
    """
    turns_to_run = turns[:3] if smoke else turns

    arm_temp    = arm_cfg.get("temperature") if arm_cfg else None
    server_type = arm_cfg.get("server", "dflash") if arm_cfg else "dflash"
    is_dflash   = is_dflash_server(server_type)

    # Wait for log file to exist and server to be settled (dflash only)
    if is_dflash:
        deadline = time.time() + 300
        while not log_path.exists() and time.time() < deadline:
            time.sleep(1)

    cache_off = done_off = spec_off = ar_off = pflash_off = survival_off = 0
    llama_mtp_off = 0

    results = []
    for turn_idx, req_body in enumerate(turns_to_run):
        turn_num = turn_idx + 1
        print(f"  [{arm_name} repeat={repeat_idx+1}] turn {turn_num}/{len(turns_to_run)} "
              f"(~{len(json.dumps(req_body))//3:,} est_tok) ...", end=" ", flush=True)

        if arm_temp is not None or seed_to_send is not None:
            req_body = dict(req_body)
            if arm_temp is not None:
                req_body["temperature"] = arm_temp
            if seed_to_send is not None and "seed" not in req_body:
                req_body["seed"] = seed_to_send

        requested_out_tokens = request_max_tokens(req_body)

        resp_body, wall_s, err = send_request(
            req_body,
            port,
            server_type=server_type,
            pin_decode_length=pin_decode_length,
        )

        if err:
            print(f"ERROR: {err}")
            results.append({
                "turn": turn_num,
                "repeat": repeat_idx + 1,
                "error": err,
                "wall_s": wall_s,
                "requested_out_tokens": requested_out_tokens,
            })
            continue

        # Give the server a moment to flush the log lines
        time.sleep(0.3)

        if is_dflash:
            try:
                log_text = log_path.read_text(errors="replace")
            except Exception:
                log_text = ""

            metrics, cache_off, done_off, spec_off, ar_off, pflash_off, survival_off = \
                parse_log_for_request(
                    log_text, cache_off, done_off, spec_off, ar_off, pflash_off, survival_off
                )
        else:
            # llama_cpp: parse timings from response body
            timings = resp_body.get("timings")
            if timings:
                metrics = parse_llama_cpp_timings(timings)
                if "prompt_tokens" in metrics:
                    metrics.setdefault("fresh_prefill", metrics["prompt_tokens"])
            else:
                metrics = {}
                print("[warn] no timings in llama_cpp response", end=" ")

            # Parse MTP draft-acceptance from server log (guarded: llama_cpp arms only)
            extra_args = arm_cfg.get("extra_args", []) if arm_cfg else []
            try:
                log_text = log_path.read_text(errors="replace")
            except Exception:
                log_text = ""
            accept_m, llama_mtp_off = parse_llama_cpp_accept(log_text, llama_mtp_off, extra_args)
            metrics.update(accept_m)

        tool_expected = request_expects_tool_call(req_body)
        expected_tool_name = req_body.get("expected_tool_name")
        # Check tool_call validity.  Replay rows with an expected tool name
        # require that exact tool; "any structured call" is not enough.
        tool_valid, tool_detail = check_tool_call(resp_body, expected_tool_name)
        charbench_valid, charbench_detail = check_charbench_quality(req_body, resp_body)
        response_text = extract_response_text(resp_body)
        request_failure = classify_request_failure(server_type, resp_body, metrics)

        rec = {
            "turn": turn_num,
            "repeat": repeat_idx + 1,
            "wall_s": round(wall_s, 2),
            "requested_out_tokens": requested_out_tokens,
            "out_tokens_match_requested": out_tokens_match_requested(metrics, requested_out_tokens),
            **metrics,
            "tool_call_expected": tool_expected,
            "expected_tool_name": expected_tool_name,
            "tool_call_names": extract_tool_call_names(resp_body),
            "tool_call_valid": tool_valid,
            "tool_detail": tool_detail,
            "charbench_valid": charbench_valid,
            "charbench_detail": charbench_detail,
            "response_text": response_text,
        }
        if request_failure:
            rec["error"] = request_failure
        results.append(rec)

        # Print one-liner
        pt     = metrics.get("prompt_tokens", "?")
        eff    = metrics.get("effective_in", "?")
        pl     = metrics.get("prefix_len", "?")
        fp     = metrics.get("fresh_prefill", "?")
        hr     = metrics.get("cache_hit_ratio")
        pfs    = metrics.get("prefill_s")
        out    = metrics.get("out_tokens")
        dec    = metrics.get("decode_tps")
        mode   = metrics.get("decode_mode", "?")
        acc    = metrics.get("accept_rate")
        pflkpt = metrics.get("pflash_kept_pct")

        hr_str  = f"{hr:.1%}" if hr is not None else "?"
        pfs_str = f"{pfs:.2f}s" if pfs else "?"
        dec_str = f"{dec:.1f}" if dec else "?"
        acc_str = f"{acc:.1f}%" if acc is not None else "AR"
        pfl_str = f"pflash={pflkpt:.1f}%kept" if pflkpt is not None else ""

        print(
            f"wall={wall_s:.1f}s  pt={pt}  eff={eff}  "
            f"prefix_len={pl} ({hr_str} hit)  "
            f"fresh={fp}  prefill={pfs_str}  "
            f"out={out}/{requested_out_tokens if requested_out_tokens is not None else '?'}  "
            f"decode={dec_str}tok/s[{mode}] {acc_str}  "
            f"{pfl_str}  tool={tool_valid}"
        )

    return results


# ---------------------------------------------------------------------------
# Aggregate results across repeats
# ---------------------------------------------------------------------------

def _vals(records: list[dict], key: str) -> list:
    return [r[key] for r in records if key in r and r.get("error") is None]


def aggregate_turns(all_records: list[dict], n_turns: int) -> list[dict]:
    """Return per-turn median metrics aggregated across repeats."""
    agg = []
    for t in range(1, n_turns + 1):
        attempts = [r for r in all_records if r.get("turn") == t]
        error_count = sum(1 for r in attempts if "error" in r)
        recs = [r for r in attempts if "error" not in r]
        if not recs:
            agg.append({
                "turn": t,
                "n_repeats": 0,
                "attempts": len(attempts),
                "error_count": error_count,
            })
            continue

        def med(key):
            vs = _vals(recs, key)
            return round(median(vs), 3) if vs else None

        # disk_hit_rate: fraction of repeats where the disk prefix cache was hit.
        # Only meaningful in --restart-per-turn mode (always None in single-session).
        disk_hits = [r for r in recs if "disk_hit" in r]
        disk_hit_rate = (
            round(sum(1 for r in disk_hits if r["disk_hit"]) / len(disk_hits), 4)
            if disk_hits else None
        )
        out_match_vals = [
            r["out_tokens_match_requested"]
            for r in recs
            if r.get("out_tokens_match_requested") is not None
        ]
        out_match_rate = (
            round(sum(1 for v in out_match_vals if v) / len(out_match_vals), 4)
            if out_match_vals else None
        )
        charbench_vals = [
            r["charbench_valid"]
            for r in recs
            if r.get("charbench_valid") is not None
        ]
        charbench_valid_rate = (
            round(sum(1 for v in charbench_vals if v) / len(charbench_vals), 4)
            if charbench_vals else None
        )
        expected_vals = [
            r.get("tool_call_expected")
            for r in recs
            if r.get("tool_call_expected") is not None
        ]
        if expected_vals:
            expected_count = sum(1 for v in expected_vals if v)
            unexpected_count = len(expected_vals) - expected_count
            expected_valid_count = sum(
                1 for r in recs
                if r.get("tool_call_expected") is True and r.get("tool_call_valid")
            )
            unexpected_valid_count = sum(
                1 for r in recs
                if r.get("tool_call_expected") is False and r.get("tool_call_valid")
            )
            tool_call_valid_rate = (
                expected_valid_count / expected_count if expected_count else None
            )
            unexpected_tool_call_rate = (
                unexpected_valid_count / unexpected_count if unexpected_count else None
            )
            tool_call_expected_rate = expected_count / len(expected_vals)
        else:
            expected_count = None
            expected_valid_count = None
            unexpected_tool_call_rate = None
            tool_call_expected_rate = None
            tool_call_valid_rate = sum(
                1 for r in recs if r.get("tool_call_valid")
            ) / len(recs)
        requested_out_vals = [
            r.get("requested_out_tokens")
            for r in recs
            if r.get("requested_out_tokens") is not None
        ]
        pin_decode_turn = bool(requested_out_vals)
        pin_decode_tool_turn = bool(pin_decode_turn and expected_count and expected_count > 0)
        pin_decode_non_tool_turn = bool(pin_decode_turn and not pin_decode_tool_turn)

        agg.append({
            "turn": t,
            "n_repeats": len(recs),
            "attempts": len(attempts),
            "error_count": error_count,
            "wall_s": med("wall_s"),
            "prompt_tokens": med("prompt_tokens"),
            "effective_in": med("effective_in"),
            "requested_out_tokens": med("requested_out_tokens"),
            "out_tokens": med("out_tokens"),
            "out_tokens_match_requested_rate": out_match_rate,
            "out_token_mismatch_count": (
                sum(1 for v in out_match_vals if not v) if out_match_vals else None
            ),
            "prefix_len": med("prefix_len"),
            "prefix_key_len": med("prefix_key_len"),
            "fresh_prefill": med("fresh_prefill"),
            "cache_hit_ratio": med("cache_hit_ratio"),
            "cache_key_hit_ratio": med("cache_key_hit_ratio"),
            "prefill_s": med("prefill_s"),
            "prefill_tps": med("prefill_tps"),
            "decode_s": med("decode_s"),
            "decode_tps": med("decode_tps"),
            "decode_mode": recs[0].get("decode_mode"),
            "accept_rate": med("accept_rate"),
            "avg_commit": med("avg_commit"),
            "pflash_kept_pct": med("pflash_kept_pct"),
            "disk_hit_rate": disk_hit_rate,
            "tool_call_expected_rate": tool_call_expected_rate,
            "tool_call_expected_count": expected_count,
            "tool_call_expected_valid_count": expected_valid_count,
            "tool_call_valid_rate": tool_call_valid_rate,
            "unexpected_tool_call_rate": unexpected_tool_call_rate,
            "pin_decode_turn": pin_decode_turn,
            "pin_decode_tool_turn": pin_decode_tool_turn,
            "pin_decode_non_tool_turn": pin_decode_non_tool_turn,
            "charbench_valid_rate": charbench_valid_rate,
        })
    return agg


def aggregate_arm(per_turn: list[dict]) -> dict:
    """Compute arm-level aggregates from per-turn medians."""
    valid = [t for t in per_turn if t.get("n_repeats", 0) > 0]
    expected_turns = len(per_turn)
    valid_turns = len(valid)
    total_errors = sum(int(t.get("error_count") or 0) for t in per_turn)
    total_attempts = sum(int(t.get("attempts") or t.get("n_repeats") or 0) for t in per_turn)

    def safe_mean(key):
        vs = [t[key] for t in valid if t.get(key) is not None]
        return round(mean(vs), 3) if vs else None

    def safe_sum(key, ndigits=1):
        vs = [t[key] for t in valid if t.get(key) is not None]
        return round(sum(vs), ndigits) if vs else None

    def safe_int_sum(key):
        vs = [t[key] for t in valid if t.get(key) is not None]
        return int(round(sum(vs))) if vs else None

    def weighted_tps(token_key, seconds_key):
        toks = [t[token_key] for t in valid if t.get(token_key) is not None and t.get(seconds_key)]
        secs = [t[seconds_key] for t in valid if t.get(token_key) is not None and t.get(seconds_key)]
        if not toks or not secs:
            return None
        total_s = sum(secs)
        return round(sum(toks) / total_s, 3) if total_s > 0 else None

    spec_turns = [t for t in valid if t.get("decode_mode") == "spec"]

    # disk_hit_rate: only non-None when --restart-per-turn was used.
    disk_hit_vals = [t["disk_hit_rate"] for t in valid if t.get("disk_hit_rate") is not None]
    mean_disk_hit_rate = round(mean(disk_hit_vals), 4) if disk_hit_vals else None
    out_mismatch_counts = [
        int(t["out_token_mismatch_count"])
        for t in valid
        if t.get("out_token_mismatch_count") is not None
    ]
    out_token_mismatch_count = sum(out_mismatch_counts) if out_mismatch_counts else None
    sum_requested_out_tokens = safe_int_sum("requested_out_tokens")
    censored = total_errors > 0 or valid_turns < expected_turns
    ok = expected_turns > 0 and valid_turns == expected_turns and total_errors == 0
    out_tokens_match_requested = (
        out_token_mismatch_count == 0
        if out_token_mismatch_count is not None and sum_requested_out_tokens is not None
        else None
    )
    pin_decode_ok = (
        ok and out_tokens_match_requested and safe_int_sum("out_tokens") == sum_requested_out_tokens
        if out_tokens_match_requested is not None
        else None
    )
    pin_turns = [t for t in valid if t.get("pin_decode_turn")]
    pin_tool_turns = [t for t in pin_turns if t.get("pin_decode_tool_turn")]
    pin_non_tool_turns = [t for t in pin_turns if t.get("pin_decode_non_tool_turn")]
    pin_non_tool_mismatch_count = sum(
        int(t.get("out_token_mismatch_count") or 0)
        for t in pin_non_tool_turns
    ) if pin_non_tool_turns else None
    pin_non_tool_ok = (
        ok and pin_non_tool_mismatch_count == 0
        if pin_non_tool_mismatch_count is not None
        else None
    )
    pin_tool_stop_conflict = bool(pin_tool_turns)
    pin_decode_claim_scope = None
    if pin_turns:
        if pin_tool_stop_conflict:
            pin_decode_claim_scope = (
                "speed_only_tool_stop_conflict"
                if not pin_non_tool_turns
                else "mixed_tool_and_non_tool"
            )
        else:
            pin_decode_claim_scope = "exact_output_workload"

    return {
        "ok": ok,
        "censored": censored,
        "expected_turns": expected_turns,
        "valid_turns": valid_turns,
        "total_attempts": total_attempts,
        "error_count": total_errors,
        "total_wall_s": safe_sum("wall_s"),
        "total_prefill_s": safe_sum("prefill_s", 3),
        "total_decode_s": safe_sum("decode_s", 3),
        "sum_prompt_tokens": safe_int_sum("prompt_tokens"),
        "sum_effective_in_tokens": safe_int_sum("effective_in"),
        "sum_fresh_prefill_tokens": safe_sum("fresh_prefill"),
        "sum_requested_out_tokens": sum_requested_out_tokens,
        "sum_out_tokens": safe_int_sum("out_tokens"),
        "out_token_mismatch_count": out_token_mismatch_count,
        "out_tokens_match_requested": out_tokens_match_requested,
        "pin_decode_ok": pin_decode_ok,
        "pin_decode_turns": len(pin_turns) if pin_turns else None,
        "pin_decode_tool_turns": len(pin_tool_turns) if pin_turns else None,
        "pin_decode_non_tool_turns": len(pin_non_tool_turns) if pin_turns else None,
        "pin_decode_non_tool_mismatch_count": pin_non_tool_mismatch_count,
        "pin_decode_non_tool_ok": pin_non_tool_ok,
        "pin_decode_tool_stop_conflict": pin_tool_stop_conflict if pin_turns else None,
        "pin_decode_claim_scope": pin_decode_claim_scope,
        "mean_cache_hit_ratio": safe_mean("cache_hit_ratio"),
        "mean_prefill_tps": safe_mean("prefill_tps"),
        "mean_decode_tps": safe_mean("decode_tps"),
        "weighted_prompt_prefill_tps": weighted_tps("prompt_tokens", "prefill_s"),
        "weighted_effective_prefill_tps": weighted_tps("effective_in", "prefill_s"),
        "weighted_fresh_prefill_tps": weighted_tps("fresh_prefill", "prefill_s"),
        "weighted_decode_tps": weighted_tps("out_tokens", "decode_s"),
        "spec_engagement_rate": round(len(spec_turns) / len(valid), 3) if valid else None,
        "mean_accept_rate": safe_mean("accept_rate"),
        "mean_disk_hit_rate": mean_disk_hit_rate,
        "sum_tool_expected_turns": safe_int_sum("tool_call_expected_count"),
        "sum_tool_expected_valid_turns": safe_int_sum("tool_call_expected_valid_count"),
        "tool_call_valid_rate": safe_mean("tool_call_valid_rate"),
        "unexpected_tool_call_rate": safe_mean("unexpected_tool_call_rate"),
        "charbench_valid_rate": safe_mean("charbench_valid_rate"),
    }


def apply_fair_contract(arm_agg: dict, pin_decode_length: bool) -> None:
    """Classify what this row is allowed to prove.

    The contract is deliberately stricter than arm_agg["ok"].  "ok" only says
    the benchmark ran without transport/generation errors.  It does not make a
    pinned mixed tool trace a correctness row: forcing max_tokens masks natural
    tool-stop behavior by design.
    """
    generation_ok = bool(arm_agg.get("ok")) and not bool(arm_agg.get("censored"))
    expected = arm_agg.get("sum_tool_expected_turns")
    expected_valid = arm_agg.get("sum_tool_expected_valid_turns")
    unexpected_rate = arm_agg.get("unexpected_tool_call_rate")
    charbench_rate = arm_agg.get("charbench_valid_rate")

    tool_expected_ok = None
    if expected is not None:
        tool_expected_ok = int(expected_valid or 0) == int(expected)

    unexpected_ok = None
    if unexpected_rate is not None:
        unexpected_ok = float(unexpected_rate) == 0.0

    charbench_ok = None
    if charbench_rate is not None:
        charbench_ok = float(charbench_rate) == 1.0

    observed_checks = [
        v for v in (tool_expected_ok, unexpected_ok, charbench_ok)
        if v is not None
    ]
    natural_correctness_ok = (
        generation_ok and all(observed_checks)
        if observed_checks else generation_ok
    )

    fair_speed_ok = (
        generation_ok and arm_agg.get("pin_decode_ok") is True
        if pin_decode_length else None
    )

    if not generation_ok:
        claim_class = "invalid_censored"
        correctness_ok = False
        row_usable = False
    elif pin_decode_length:
        claim_class = "equal_output_speed_only"
        correctness_ok = False
        row_usable = bool(fair_speed_ok)
    else:
        claim_class = "natural_correctness"
        correctness_ok = bool(natural_correctness_ok)
        row_usable = correctness_ok

    arm_agg["fair_contract"] = {
        "claim_class": claim_class,
        "row_usable": row_usable,
        "generation_ok": generation_ok,
        "natural_correctness_ok": natural_correctness_ok,
        "correctness_gate_ok": correctness_ok,
        "equal_output_speed_ok": fair_speed_ok,
        "tool_expected_ok": tool_expected_ok,
        "unexpected_tool_ok": unexpected_ok,
        "charbench_ok": charbench_ok,
        "note": (
            "Pinned mixed tool rows are speed-only; run an unpinned natural "
            "trace for tool correctness."
            if pin_decode_length and generation_ok else None
        ),
    }


# ---------------------------------------------------------------------------
# Report writing
# ---------------------------------------------------------------------------

def write_report(
    arm_name: str,
    per_turn: list[dict],
    arm_agg: dict,
    provenance: dict,
    out_path: Path,
    smoke: bool,
    run_kind: str = "full",
) -> None:
    ts = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    lines = []
    lines.append(f"# ABC Cache Harness — {arm_name}")
    lines.append(f"Generated: {ts}")
    n_repeats = provenance.get("n_repeats", "?")
    if smoke:
        mode_label = "SMOKE (first 3 turns, N=1)"
    elif run_kind == "quality":
        mode_label = f"QUALITY PROBE (all trace turns, N={n_repeats})"
    else:
        mode_label = f"FULL (all turns, N={n_repeats})"
    lines.append(f"Mode: {mode_label}")
    lines.append("")
    lines.append("## Provenance")
    lines.append("```")
    lines.append(json.dumps(provenance, indent=2))
    lines.append("```")
    lines.append("")
    lines.append("## Run Quality")
    lines.append("")
    lines.append(f"- ok: {arm_agg.get('ok')}")
    if arm_agg.get("invalid_reason"):
        lines.append(f"- invalid_reason: {arm_agg.get('invalid_reason')}")
    lines.append(f"- censored: {arm_agg.get('censored')}")
    lines.append(f"- valid_turns: {arm_agg.get('valid_turns')}/{arm_agg.get('expected_turns')}")
    lines.append(f"- error_count: {arm_agg.get('error_count')}")
    fair = arm_agg.get("fair_contract") or {}
    if fair:
        lines.append(f"- fair_claim_class: {fair.get('claim_class')}")
        lines.append(f"- fair_row_usable: {fair.get('row_usable')}")
        lines.append(f"- correctness_gate_ok: {fair.get('correctness_gate_ok')}")
        lines.append(f"- equal_output_speed_ok: {fair.get('equal_output_speed_ok')}")
        if fair.get("note"):
            lines.append(f"- fair_note: {fair.get('note')}")
    if arm_agg.get("energy_j") is not None:
        lines.append(f"- energy_j: {arm_agg.get('energy_j')}")
        lines.append(f"- avg_power_w: {arm_agg.get('avg_power_w')}")
        lines.append(f"- max_power_w: {arm_agg.get('max_power_w')}")
        lines.append(f"- mean_gpu_util_pct: {arm_agg.get('mean_gpu_util_pct')}")
        lines.append(f"- max_memory_used_mib: {arm_agg.get('max_memory_used_mib')}")
    if arm_agg.get("pin_decode_claim_scope") is not None:
        lines.append(f"- pin_decode_claim_scope: {arm_agg.get('pin_decode_claim_scope')}")
        lines.append(f"- pin_decode_tool_turns: {arm_agg.get('pin_decode_tool_turns')}")
        lines.append(f"- pin_decode_non_tool_turns: {arm_agg.get('pin_decode_non_tool_turns')}")
        lines.append(f"- pin_decode_tool_stop_conflict: {arm_agg.get('pin_decode_tool_stop_conflict')}")
        lines.append(f"- pin_decode_non_tool_ok: {arm_agg.get('pin_decode_non_tool_ok')}")
    if arm_agg.get("sum_tool_expected_turns") is not None:
        lines.append(
            f"- tool_expected_valid: "
            f"{arm_agg.get('sum_tool_expected_valid_turns')}/"
            f"{arm_agg.get('sum_tool_expected_turns')}"
        )
        lines.append(f"- unexpected_tool_call_rate: {arm_agg.get('unexpected_tool_call_rate')}")
    lines.append("")
    lines.append("## Per-Turn Cache Trace")
    has_disk_hit = any(t.get("disk_hit_rate") is not None for t in per_turn)
    has_requested_out = any(t.get("requested_out_tokens") is not None for t in per_turn)
    header = (
        f"{'turn':>4} {'pt':>7} {'eff_in':>7} {'prefix_len':>11} "
        f"{'fresh_pf':>9} {'hr':>6} {'pf_s':>6} {'pf_tps':>7} "
        + (f"{'out/req':>9} " if has_requested_out else "")
        + f"{'dec_tps':>8} {'mode':>5} {'accept':>7} {'pflash%':>8} "
        + (f"{'disk_hit':>9} " if has_disk_hit else "")
        + f"{'tool':>5} {'wall_s':>7}"
    )
    lines.append(header)
    lines.append("-" * len(header))
    for t in per_turn:
        if t.get("n_repeats", 0) == 0:
            lines.append(
                f"  turn {t['turn']}: NO DATA "
                f"(attempts={t.get('attempts', 0)} errors={t.get('error_count', 0)})"
            )
            continue
        hr = t.get("cache_hit_ratio")
        acc = t.get("accept_rate")
        pfl = t.get("pflash_kept_pct")
        dhr = t.get("disk_hit_rate")
        tool_rate = t.get("tool_call_valid_rate")
        pfs_val = t.get('prefill_s')
        ptps_val = t.get('prefill_tps')
        dtps_val = t.get('decode_tps')
        pfs_str2  = f"{pfs_val:.2f}" if pfs_val else "?"
        ptps_str2 = f"{ptps_val:.0f}" if ptps_val else "?"
        dtps_str2 = f"{dtps_val:.1f}" if dtps_val else "?"
        dhr_str   = f"{dhr:.1%}" if dhr is not None else "-"
        out_req_str = (
            f"{str(t.get('out_tokens','?'))}/{str(t.get('requested_out_tokens','?'))}"
            if has_requested_out else ""
        )
        row = (
            f"{t['turn']:>4} "
            f"{str(t.get('prompt_tokens','?')):>7} "
            f"{str(t.get('effective_in','?')):>7} "
            f"{str(t.get('prefix_len','?')):>11} "
            f"{str(t.get('fresh_prefill','?')):>9} "
            f"{f'{hr:.1%}' if hr is not None else '?':>6} "
            f"{pfs_str2:>6} "
            f"{ptps_str2:>7} "
            + (f"{out_req_str:>9} " if has_requested_out else "")
            + f"{dtps_str2:>8} "
            f"{str(t.get('decode_mode','?')):>5} "
            f"{f'{acc:.1f}%' if acc is not None else 'AR':>7} "
            f"{f'{pfl:.1f}%' if pfl is not None else '-':>8} "
            + (f"{dhr_str:>9} " if has_disk_hit else "")
            + f"{'Y' if tool_rate and tool_rate > 0 else 'N':>5} "
            + f"{str(t.get('wall_s','?')):>7}"
        )
        lines.append(row)
    lines.append("")
    lines.append("## Arm Aggregate")
    lines.append("```")
    lines.append(json.dumps(arm_agg, indent=2))
    lines.append("```")
    lines.append("")

    out_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"[report] Written to {out_path}")


# ---------------------------------------------------------------------------
# Per-turn restart mode
# ---------------------------------------------------------------------------

def run_trace_restart_per_turn(
    turns: list[dict],
    port: int,
    arm_name: str,
    arm_cfg: dict,
    log_dir: Path,
    max_ctx: int = MAX_CTX,
    smoke: bool = False,
    repeat_idx: int = 0,
    pin_decode_length: bool = False,
    pin_decode_tokens: Optional[int] = None,
    seed_to_send: Optional[int] = None,
) -> list[dict]:
    """Restart-per-turn mode: each turn gets a fresh server process.

    Sequence per turn:
      1. launch_server() with arm_cfg (same flags including --kv-cache-dir if present)
      2. wait_for_server()
      3. send_request() for this single turn
      4. parse_log_for_request() with fresh offsets (0,0,0,0,0,0)
      5. kill_server()
      6. annotate metrics with disk_hit = metrics.get("restore", False)

    The --kv-cache-dir is NOT wiped between turns (wiped once at arm start by caller).
    This makes the on-disk cache the ONLY cross-turn reuse path.
    """
    turns_to_run = turns[:3] if smoke else turns
    server_type = arm_cfg.get("server", "dflash")
    arm_temp = arm_cfg.get("temperature")
    llama_slot_file = arm_cfg.get("slot_save_file", f"{arm_name}.slot")
    llama_slot_id = int(arm_cfg.get("slot_id", 0))
    use_llama_slot_cache = server_type == "llama_cpp" and bool(arm_cfg.get("slot_save_dir"))
    results = []

    for turn_idx, req_body in enumerate(turns_to_run):
        turn_num = turn_idx + 1
        ts_str = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        log_path = log_dir / f"srv_{arm_name}_rep{repeat_idx+1}_t{turn_num:02d}_{ts_str}.log"

        print(f"  [{arm_name} repeat={repeat_idx+1} restart-per-turn] "
              f"turn {turn_num}/{len(turns_to_run)} "
              f"(~{len(json.dumps(req_body))//3:,} est_tok) — launching server...")

        # Step 1: launch fresh server
        proc, log_f = launch_server(
            arm_name,
            arm_cfg,
            log_path,
            port=port,
            max_ctx=max_ctx,
            pin_decode_tokens=pin_decode_tokens,
        )

        try:
            # Step 2: wait for ready
            ok, start_error = wait_for_server(
                port, timeout=360, proc=proc, log_path=log_path
            )
            if not ok:
                raise RuntimeError(
                    f"server did not come up for turn {turn_num}; log={log_path}\n"
                    f"{start_error or ''}"
                )

            slot_restore_ok = None
            slot_restore_detail = None
            if use_llama_slot_cache and turn_num > 1:
                slot_restore_ok, slot_restore_detail = llama_slot_action(
                    port, "restore", llama_slot_file, slot_id=llama_slot_id
                )
                if not slot_restore_ok:
                    print(f"  ERROR: llama slot restore failed: {slot_restore_detail}")
                    results.append({
                        "turn": turn_num,
                        "repeat": repeat_idx + 1,
                        "error": f"slot_restore_failed: {slot_restore_detail}",
                        "wall_s": 0.0,
                        "disk_hit": False,
                        "slot_restore_ok": False,
                    })
                    continue

            print(f"  READY — sending request...", end=" ", flush=True)

            # Step 3: send the single request
            if arm_temp is not None or seed_to_send is not None:
                req_body = dict(req_body)
                if arm_temp is not None:
                    req_body["temperature"] = arm_temp
                if seed_to_send is not None and "seed" not in req_body:
                    req_body["seed"] = seed_to_send

            requested_out_tokens = request_max_tokens(req_body)

            resp_body, wall_s, err = send_request(
                req_body,
                port,
                server_type=server_type,
                pin_decode_length=pin_decode_length,
            )

            if err:
                print(f"ERROR: {err}")
                results.append({
                    "turn": turn_num,
                    "repeat": repeat_idx + 1,
                    "error": err,
                    "wall_s": wall_s,
                    "disk_hit": False,
                    "requested_out_tokens": requested_out_tokens,
                })
                continue

            # Give server a moment to flush log lines
            time.sleep(0.3)

            # Step 4: parse fresh log (offsets all 0 — new server, new log file)
            try:
                log_text = log_path.read_text(errors="replace")
            except Exception:
                log_text = ""

            if is_dflash_server(server_type):
                metrics, _, _, _, _, _, _ = parse_log_for_request(
                    log_text, 0, 0, 0, 0, 0, 0
                )
            else:
                timings = resp_body.get("timings")
                metrics = parse_llama_cpp_timings(timings) if timings else {}
                if "prompt_tokens" in metrics:
                    metrics.setdefault("fresh_prefill", metrics["prompt_tokens"])
                extra_args = arm_cfg.get("extra_args", [])
                accept_m, _ = parse_llama_cpp_accept(log_text, 0, extra_args)
                metrics.update(accept_m)

            # disk_hit: in restart-per-turn mode, restore=true can ONLY come from the
            # on-disk prefix cache (no in-RAM cache survives a fresh process).
            if is_dflash_server(server_type):
                disk_hit = bool(metrics.get("restore", False))
            else:
                disk_hit = bool(slot_restore_ok)

            slot_save_ok = None
            slot_save_detail = None
            if use_llama_slot_cache:
                slot_save_ok, slot_save_detail = llama_slot_action(
                    port, "save", llama_slot_file, slot_id=llama_slot_id
                )

            tool_expected = request_expects_tool_call(req_body)
            expected_tool_name = req_body.get("expected_tool_name")
            # Check tool_call validity.  Replay rows with an expected tool name
            # require that exact tool; "any structured call" is not enough.
            tool_valid, tool_detail = check_tool_call(resp_body, expected_tool_name)
            charbench_valid, charbench_detail = check_charbench_quality(req_body, resp_body)
            response_text = extract_response_text(resp_body)
            request_failure = classify_request_failure(server_type, resp_body, metrics)

            rec = {
                "turn": turn_num,
                "repeat": repeat_idx + 1,
                "wall_s": round(wall_s, 2),
                "disk_hit": disk_hit,
                "slot_restore_ok": slot_restore_ok,
                "slot_save_ok": slot_save_ok,
                "requested_out_tokens": requested_out_tokens,
                "out_tokens_match_requested": out_tokens_match_requested(metrics, requested_out_tokens),
                **metrics,
                "tool_call_expected": tool_expected,
                "expected_tool_name": expected_tool_name,
                "tool_call_names": extract_tool_call_names(resp_body),
                "tool_call_valid": tool_valid,
                "tool_detail": tool_detail,
                "charbench_valid": charbench_valid,
                "charbench_detail": charbench_detail,
                "response_text": response_text,
            }
            if request_failure:
                rec["error"] = request_failure
            if slot_save_ok is False:
                rec["error"] = f"slot_save_failed: {slot_save_detail}"
            results.append(rec)

            # Print one-liner
            pt     = metrics.get("prompt_tokens", "?")
            eff    = metrics.get("effective_in", "?")
            pl     = metrics.get("prefix_len", "?")
            fp     = metrics.get("fresh_prefill", "?")
            hr     = metrics.get("cache_hit_ratio")
            pfs    = metrics.get("prefill_s")
            out    = metrics.get("out_tokens")
            dec    = metrics.get("decode_tps")
            mode   = metrics.get("decode_mode", "?")
            acc    = metrics.get("accept_rate")
            pflkpt = metrics.get("pflash_kept_pct")

            hr_str  = f"{hr:.1%}" if hr is not None else "?"
            pfs_str = f"{pfs:.2f}s" if pfs else "?"
            dec_str = f"{dec:.1f}" if dec else "?"
            acc_str = f"{acc:.1f}%" if acc is not None else "AR"
            pfl_str = f"pflash={pflkpt:.1f}%kept" if pflkpt is not None else ""

            print(
                f"wall={wall_s:.1f}s  pt={pt}  eff={eff}  "
                f"prefix_len={pl} ({hr_str} hit)  "
                f"fresh={fp}  prefill={pfs_str}  "
                f"out={out}/{requested_out_tokens if requested_out_tokens is not None else '?'}  "
                f"decode={dec_str}tok/s[{mode}] {acc_str}  "
                f"{pfl_str}  disk_hit={disk_hit}  tool={tool_valid}"
            )

        finally:
            # Step 5: kill server before next turn
            print(f"  Killing server PID={proc.pid} after turn {turn_num}...")
            kill_server(proc, log_f, wait_s=15)
            # Brief pause to let the server fully release GPU resources and flush the
            # disk KV cache to --kv-cache-dir before the next turn's server starts.
            time.sleep(3)

    return results


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run_selftest() -> None:
    """No-GPU self-check. Verifies pure functions and trace file correctness."""
    print("=== SELFTEST ===")
    failures = []

    # (a) build_llama_cpp_cmd for llama_cpp_mtp
    arm = ARMS["llama_cpp_mtp"]
    cmd = build_llama_cpp_cmd(arm, PORT, MAX_CTX)
    assert "--spec-type" in cmd and "draft-mtp" in cmd, "missing --spec-type draft-mtp"
    assert "-m" in cmd, "missing -m"
    m_idx = cmd.index("-m")
    assert str(MTP_GGUF) in cmd[m_idx + 1], f"wrong model: {cmd[m_idx+1]}"
    assert "--cache-type-k" in cmd, "missing --cache-type-k"
    k_idx = cmd.index("--cache-type-k")
    assert cmd[k_idx + 1] == "q4_0", f"expected q4_0 cache-type-k, got {cmd[k_idx+1]}"
    print(f"  (a) build_llama_cpp_cmd: PASS  cmd={cmd}")

    # (b) parse_llama_cpp_timings
    sample = {"prompt_n": 100, "prompt_ms": 500.0, "predicted_n": 40, "predicted_ms": 800.0}
    t = parse_llama_cpp_timings(sample)
    assert t["prefill_tps"] == 200.0, f"prefill_tps expected 200.0 got {t['prefill_tps']}"
    assert t["decode_tps"]  ==  50.0, f"decode_tps expected 50.0 got {t['decode_tps']}"
    print(f"  (b) parse_llama_cpp_timings: PASS  prefill_tps={t['prefill_tps']} decode_tps={t['decode_tps']}")

    # (b2) parse_llama_cpp_accept — MTP log line extraction
    mtp_log_line = (
        "0.38.538.542 I slot print_timing: id  3 | task 0 | "
        "draft acceptance = 0.91176 (   31 accepted /    34 generated), "
        "mean acceptance length =  2.82, acceptance rate per position = (1.000, 0.824)"
    )
    mtp_args = ["--spec-type", "draft-mtp", "--spec-draft-n-max", "2"]
    am, new_off = parse_llama_cpp_accept(mtp_log_line, 0, mtp_args)
    assert am.get("spec_engaged") is True, f"expected spec_engaged=True, got {am}"
    assert am.get("decode_mode") == "spec", f"expected decode_mode=spec, got {am}"
    assert abs(am.get("accept_rate", 0) - 91.18) < 0.1, f"accept_rate wrong: {am}"
    assert am.get("draft_accepted") == 31, f"draft_accepted wrong: {am}"
    assert am.get("draft_generated") == 34, f"draft_generated wrong: {am}"
    assert new_off == 1, f"offset should advance to 1, got {new_off}"
    # AR arm (no --spec-type): should return empty dict
    ar_m, ar_off = parse_llama_cpp_accept("some log", 0, [])
    assert ar_m == {}, f"AR arm should return empty dict, got {ar_m}"
    # MTP arm with no log line yet: spec_engaged=True, accept_rate absent
    no_log_m, no_log_off = parse_llama_cpp_accept("no matching line here", 0, mtp_args)
    assert no_log_m.get("spec_engaged") is True, f"expected spec_engaged=True even without log line"
    assert "accept_rate" not in no_log_m, f"accept_rate should be absent when log line not found"
    assert no_log_off == 0, f"offset should not advance when no log line"
    print(f"  (b2) parse_llama_cpp_accept: PASS  {am}")

    # (b2a) strict expected tool matching — a wrong structured tool is still invalid.
    tool_body = {"content": [{"type": "tool_use", "name": "Bash", "input": {"command": "pwd"}}]}
    ok, detail = check_tool_call(tool_body, "Bash")
    assert ok and detail == "tool=Bash", (ok, detail)
    ok, detail = check_tool_call(tool_body, "Read")
    assert not ok and "expected=Read" in detail and "Bash" in detail, (ok, detail)
    print("  (b2a) strict expected tool matching: PASS")

    # (b3) LLAMA_BIN points to build-cuda path
    assert "build-cuda" in str(LLAMA_BIN), f"LLAMA_BIN must point to build-cuda, got {LLAMA_BIN}"
    assert "build-cuda" in str(LLAMA_CUDA_LIB), f"LLAMA_CUDA_LIB must point to build-cuda, got {LLAMA_CUDA_LIB}"
    print(f"  (b3) LLAMA_BIN CUDA path: PASS  {LLAMA_BIN}")

    # (b4) lucebox family: lucebox / lucebox_ddtree22 / lucebox_no_ddtree
    assert "lucebox_no_ddtree" in ARMS, "lucebox_no_ddtree arm missing from ARMS"
    assert "lucebox_ddtree22" in ARMS, "lucebox_ddtree22 arm missing from ARMS"
    lb      = ARMS["lucebox"]
    lb_d22  = ARMS["lucebox_ddtree22"]
    lb_nod  = ARMS["lucebox_no_ddtree"]
    # ddtree flags
    assert "--ddtree" in lb["extra_args"], "lucebox must have --ddtree in extra_args"
    assert "--ddtree-budget" in lb["extra_args"], "lucebox must have --ddtree-budget"
    bud_idx = lb["extra_args"].index("--ddtree-budget")
    assert lb["extra_args"][bud_idx + 1] == "16", f"lucebox ddtree-budget must be 16, got {lb['extra_args'][bud_idx+1]}"
    assert "--ddtree" in lb_d22["extra_args"], "lucebox_ddtree22 must have --ddtree"
    assert "--ddtree-budget" in lb_d22["extra_args"], "lucebox_ddtree22 must have --ddtree-budget"
    bud22_idx = lb_d22["extra_args"].index("--ddtree-budget")
    assert lb_d22["extra_args"][bud22_idx + 1] == "22", f"lucebox_ddtree22 budget must be 22, got {lb_d22['extra_args'][bud22_idx+1]}"
    assert "22" in lb_d22["extra_args"], "lucebox_ddtree22 extra_args must contain '22'"
    assert "--ddtree" not in lb_nod["extra_args"], "lucebox_no_ddtree must NOT have --ddtree"
    # All three share the same env keys and kv type
    assert lb["env"].keys() == lb_d22["env"].keys(), "lucebox_ddtree22 env keys must match lucebox"
    assert lb["env"].keys() == lb_nod["env"].keys(), "lucebox_no_ddtree env keys must match lucebox"
    assert lb.get("kv_cache_types") == lb_d22.get("kv_cache_types"), "lucebox_ddtree22 kv_cache_types must match lucebox"
    assert lb.get("kv_cache_types") == lb_nod.get("kv_cache_types"), "lucebox_no_ddtree kv_cache_types must match lucebox"
    import json as _json
    print(f"  (b4) lucebox arm dict:")
    print(f"       {_json.dumps(lb, indent=6, default=str)}")
    print(f"  (b4) lucebox_ddtree22 arm dict:")
    print(f"       {_json.dumps(lb_d22, indent=6, default=str)}")
    print(f"  (b4) lucebox_no_ddtree arm dict:")
    print(f"       {_json.dumps(lb_nod, indent=6, default=str)}")
    print(f"  (b4) ddtree family (16/22/no) arm check: PASS")

    # (b5) lucebox_ar: pure AR dflash arm — no draft, no spec env vars
    assert "lucebox_ar" in ARMS, "lucebox_ar arm missing from ARMS"
    lb_ar = ARMS["lucebox_ar"]
    assert lb_ar.get("server") == "dflash", "lucebox_ar must use dflash server"
    assert lb_ar.get("model_target") == TGT_35B, "lucebox_ar model_target must be TGT_35B"
    assert lb_ar.get("kv_cache_types") == ("q4_0", "q4_0"), "lucebox_ar must use q4_0 KV"
    assert lb_ar.get("extra_args") == [], "lucebox_ar extra_args must be empty (no --draft)"
    assert lb_ar.get("env") == {}, "lucebox_ar env must be empty (no spec env vars)"
    # Confirm no spec-triggering flags leak into the arm
    for forbidden in ("--draft", "--kvflash", "--ddtree", "--prefill-drafter"):
        assert forbidden not in lb_ar.get("extra_args", []), \
            f"lucebox_ar must NOT have {forbidden} in extra_args"
    for forbidden_env in ("DFLASH_SPEC_GATE", "DFLASH_FEAT_RING_CAP",
                          "DFLASH_DRAFT_CTX_MAX", "KVFLASH_RESTORE_CONSUME"):
        assert forbidden_env not in lb_ar.get("env", {}), \
            f"lucebox_ar env must NOT contain {forbidden_env}"
    print(f"  (b5) lucebox_ar arm dict:")
    print(f"       {_json.dumps(lb_ar, indent=6, default=str)}")
    print(f"  (b5) lucebox_ar pure-AR check: PASS")

    # (b6) cached llama restart arms expose slot-save plumbing.
    for arm_name in ("AR_LLAMA_27B_SLOTCACHE", "AR_LLAMA_35B_SLOTCACHE"):
        slot_arm = ARMS[arm_name]
        assert slot_arm.get("server") == "llama_cpp", f"{arm_name} must be llama_cpp"
        assert slot_arm.get("slot_save_dir"), f"{arm_name} missing slot_save_dir"
        assert "--slot-save-path" in slot_arm.get("extra_args", []), f"{arm_name} missing --slot-save-path"
        assert "--cache-reuse" in slot_arm.get("extra_args", []), f"{arm_name} missing --cache-reuse"
        assert "--ctx-checkpoints" in slot_arm.get("extra_args", []), f"{arm_name} missing --ctx-checkpoints"
    print("  (b6) llama slot-cache arms: PASS")

    # (b7) aggregation classifies partial/error runs and computes weighted rates.
    sample_records = [
        {"turn": 1, "repeat": 1, "wall_s": 1.0, "prompt_tokens": 100,
         "fresh_prefill": 100, "prefill_s": 0.5, "out_tokens": 20,
         "decode_s": 0.25, "decode_tps": 80.0, "tool_call_valid": False},
        {"turn": 2, "repeat": 1, "error": "boom", "wall_s": 0.0},
    ]
    sample_turns = aggregate_turns(sample_records, 2)
    sample_agg = aggregate_arm(sample_turns)
    assert sample_agg["ok"] is False, f"expected ok=false, got {sample_agg}"
    assert sample_agg["censored"] is True, f"expected censored=true, got {sample_agg}"
    assert sample_agg["valid_turns"] == 1, f"expected one valid turn, got {sample_agg}"
    assert sample_agg["error_count"] == 1, f"expected one error, got {sample_agg}"
    assert sample_agg["weighted_fresh_prefill_tps"] == 200.0, f"weighted fresh tps wrong: {sample_agg}"
    assert sample_agg["weighted_decode_tps"] == 80.0, f"weighted decode tps wrong: {sample_agg}"
    print("  (b7) aggregate quality/weighted metrics: PASS")

    # (b7a) Pinned decode accounting: aggregate proves exact output workload.
    assert request_max_tokens({"max_tokens": "20"}) == 20, "string max_tokens should coerce"
    assert request_max_tokens({"max_tokens": 0}) is None, "zero max_tokens must be rejected"
    pin_records = [
        {"turn": 1, "repeat": 1, "wall_s": 1.0, "prompt_tokens": 100,
         "fresh_prefill": 100, "prefill_s": 0.5, "out_tokens": 20,
         "requested_out_tokens": 20, "out_tokens_match_requested": True,
         "decode_s": 0.25, "decode_tps": 80.0, "tool_call_valid": False},
    ]
    pin_turns = aggregate_turns(pin_records, 1)
    pin_agg = aggregate_arm(pin_turns)
    assert pin_turns[0]["out_tokens_match_requested_rate"] == 1.0, f"pin turn bad: {pin_turns}"
    assert pin_agg["sum_requested_out_tokens"] == 20, f"pin requested sum bad: {pin_agg}"
    assert pin_agg["sum_out_tokens"] == 20, f"pin out sum bad: {pin_agg}"
    assert pin_agg["pin_decode_ok"] is True, f"pin ok should be true: {pin_agg}"
    assert pin_agg["pin_decode_claim_scope"] == "exact_output_workload", pin_agg
    assert pin_agg["pin_decode_tool_stop_conflict"] is False, pin_agg
    assert pin_agg["pin_decode_non_tool_ok"] is True, pin_agg
    mismatch_records = [dict(pin_records[0], out_tokens=19, out_tokens_match_requested=False)]
    mismatch_agg = aggregate_arm(aggregate_turns(mismatch_records, 1))
    assert mismatch_agg["out_token_mismatch_count"] == 1, f"mismatch count bad: {mismatch_agg}"
    assert mismatch_agg["pin_decode_ok"] is False, f"pin ok should fail: {mismatch_agg}"
    tool_pin_records = [
        dict(pin_records[0], tool_call_expected=True, tool_call_valid=True),
    ]
    tool_pin_agg = aggregate_arm(aggregate_turns(tool_pin_records, 1))
    assert tool_pin_agg["pin_decode_ok"] is True, f"tool pin exactness bad: {tool_pin_agg}"
    assert tool_pin_agg["pin_decode_tool_stop_conflict"] is True, tool_pin_agg
    assert tool_pin_agg["pin_decode_tool_turns"] == 1, tool_pin_agg
    assert tool_pin_agg["pin_decode_non_tool_turns"] == 0, tool_pin_agg
    assert tool_pin_agg["pin_decode_non_tool_ok"] is None, tool_pin_agg
    assert tool_pin_agg["pin_decode_claim_scope"] == "speed_only_tool_stop_conflict", tool_pin_agg
    apply_fair_contract(tool_pin_agg, pin_decode_length=True)
    assert tool_pin_agg["fair_contract"]["claim_class"] == "equal_output_speed_only", tool_pin_agg
    assert tool_pin_agg["fair_contract"]["equal_output_speed_ok"] is True, tool_pin_agg
    assert tool_pin_agg["fair_contract"]["correctness_gate_ok"] is False, tool_pin_agg
    mixed_pin_records = [
        dict(pin_records[0], turn=1, tool_call_expected=True, tool_call_valid=True,
             out_tokens=12, out_tokens_match_requested=False),
        dict(pin_records[0], turn=2, tool_call_expected=False, tool_call_valid=False),
    ]
    mixed_pin_agg = aggregate_arm(aggregate_turns(mixed_pin_records, 2))
    assert mixed_pin_agg["pin_decode_ok"] is False, mixed_pin_agg
    assert mixed_pin_agg["pin_decode_tool_stop_conflict"] is True, mixed_pin_agg
    assert mixed_pin_agg["pin_decode_non_tool_ok"] is True, mixed_pin_agg
    assert mixed_pin_agg["pin_decode_claim_scope"] == "mixed_tool_and_non_tool", mixed_pin_agg
    natural_tool_records = [
        {"turn": 1, "repeat": 1, "wall_s": 1.0, "prompt_tokens": 100,
         "fresh_prefill": 100, "prefill_s": 0.5, "out_tokens": 5,
         "decode_s": 0.1, "tool_call_expected": True, "tool_call_valid": True},
        {"turn": 2, "repeat": 1, "wall_s": 1.0, "prompt_tokens": 120,
         "fresh_prefill": 20, "prefill_s": 0.1, "out_tokens": 6,
         "decode_s": 0.1, "tool_call_expected": False, "tool_call_valid": False},
    ]
    natural_tool_agg = aggregate_arm(aggregate_turns(natural_tool_records, 2))
    apply_fair_contract(natural_tool_agg, pin_decode_length=False)
    assert natural_tool_agg["fair_contract"]["claim_class"] == "natural_correctness", natural_tool_agg
    assert natural_tool_agg["fair_contract"]["correctness_gate_ok"] is True, natural_tool_agg
    bad_natural_records = [dict(natural_tool_records[0], tool_call_valid=False)]
    bad_natural_agg = aggregate_arm(aggregate_turns(bad_natural_records, 1))
    apply_fair_contract(bad_natural_agg, pin_decode_length=False)
    assert bad_natural_agg["fair_contract"]["correctness_gate_ok"] is False, bad_natural_agg
    print("  (b7a) pinned decode accounting: PASS")

    # (b8) dflash generation errors must censor a row even when HTTP succeeds.
    error_log = (
        "[server] chat CACHE msg_0 restore=false slot=-1 prefix_len=0 "
        "effective_prompt=1082 pflash=false\n"
        "[server] chat DONE msg_0 ok=false in=1082 effective_in=1082 out=2 "
        "10.7s 0.2 tok/s finish=error restore=false slot=-1 prefix_len=0 "
        "prefill=10.4s decode=0.0s(0.0tok/s) error=hybrid_spec_decode\n"
    )
    error_metrics, *_ = parse_log_for_request(error_log, 0, 0, 0, 0, 0, 0)
    assert error_metrics["chat_ok"] is False, f"expected chat_ok=false, got {error_metrics}"
    assert error_metrics["finish_reason"] == "error", f"finish_reason not parsed: {error_metrics}"
    assert error_metrics["chat_error"] == "hybrid_spec_decode", f"chat_error not parsed: {error_metrics}"
    failure = classify_request_failure("dflash", {}, error_metrics)
    assert failure == "dflash_generation_failed: hybrid_spec_decode", f"bad failure: {failure}"
    error_turns = aggregate_turns([
        {"turn": 1, "repeat": 1, "wall_s": 10.7, **error_metrics, "error": failure},
    ], 1)
    error_agg = aggregate_arm(error_turns)
    assert error_agg["ok"] is False and error_agg["error_count"] == 1, f"bad error agg: {error_agg}"
    print("  (b8) dflash generation failure censoring: PASS")

    # (b9) Luce prefix accounting: prefix_len is the physical restored
    # snapshot position; prefix_key_len is only the logical cache key.
    # This catches chunk-rounded snapshots such as key_len=1054/snapshot=1024.
    cache_accounting_log = (
        "[server] chat CACHE msg_1 restore=true slot=1 prefix_len=1024 "
        "effective_prompt=2287 prefix_key_len=1054 pflash=false\n"
        "[server] chat DONE msg_1 ok=true in=2287 effective_in=2287 out=192 "
        "3.8s 50.3 tok/s finish=stop restore=true slot=1 prefix_len=1024 "
        "prefill=2.3s decode=1.5s(127.9tok/s) error=-\n"
    )
    acct_metrics, *_ = parse_log_for_request(cache_accounting_log, 0, 0, 0, 0, 0, 0)
    assert acct_metrics["prefix_len"] == 1024, f"physical prefix wrong: {acct_metrics}"
    assert acct_metrics["prefix_key_len"] == 1054, f"logical key wrong: {acct_metrics}"
    assert acct_metrics["fresh_prefill"] == 1263, f"fresh prefill must use physical prefix: {acct_metrics}"
    print("  (b9) dflash physical prefix accounting: PASS")

    # (c) chat_simple.jsonl — 5 lines, valid JSON each
    trace = BENCH_DIR / "traces" / "chat_simple.jsonl"
    if not trace.exists():
        failures.append(f"chat_simple.jsonl missing: {trace}")
        print(f"  (c) chat_simple.jsonl: FAIL — file not found")
    else:
        lines = [l for l in trace.read_text().splitlines() if l.strip()]
        assert len(lines) == 5, f"expected 5 lines, got {len(lines)}"
        for i, ln in enumerate(lines):
            obj = json.loads(ln)
            assert "messages" in obj, f"line {i} missing messages"
        print(f"  (c) chat_simple.jsonl: PASS  ({len(lines)} lines)")

    if failures:
        print("\nSELFTEST FAILED:")
        for f in failures:
            print(f"  {f}")
        sys.exit(1)
    print("=== SELFTEST PASS ===")


def main() -> None:
    ap = argparse.ArgumentParser(description="ABC cache composition harness")
    ap.add_argument("--arm", required=False, choices=sorted(ARMS.keys()),
                    help="Which arm to run")
    ap.add_argument("--trace", default=str(BENCH_DIR / "traces" / "goldgate_fix.jsonl"),
                    help="Path to trace JSONL")
    ap.add_argument("--n", type=int, default=3, help="Number of full-trace repeats")
    ap.add_argument("--seed", type=int, default=42,
                    help="Sampling seed (noted in provenance; server may not support)")
    ap.add_argument("--send-seed", action="store_true",
                    help="Include --seed in each request that does not already carry a seed")
    ap.add_argument("--port", type=int, default=PORT, help="Server port (default 19099)")
    ap.add_argument("--max-ctx", type=int, default=MAX_CTX,
                    help=f"Context ceiling for server (default {MAX_CTX})")
    ap.add_argument("--binary", default=None,
                    help="Explicit dflash_server binary path (overrides default build-dir path)")
    ap.add_argument("--smoke", action="store_true",
                    help="Smoke gate: first 3 turns only, N=1")
    ap.add_argument("--turn-limit", type=int, default=None,
                    help="Run only the first N trace turns (diagnostic aid)")
    ap.add_argument("--selftest", action="store_true",
                    help="Run no-GPU self-check and exit")
    ap.add_argument("--restart-per-turn", action="store_true",
                    help=(
                        "Per-turn server restart mode: for each turn, launch a fresh "
                        "server, send that one turn's request, record metrics, then kill "
                        "before the next turn.  The --kv-cache-dir (if set in the arm) "
                        "persists across restarts within this run so the on-disk prefix "
                        "cache is the ONLY cross-turn reuse path.  Measures the "
                        "claude-code-reconnects-each-turn / cross-session disk-hit scenario."
                    ))
    ap.add_argument("--pin-decode-length", action="store_true",
                    help=(
                        "Force each turn to decode exactly its trace max_tokens when the "
                        "engine supports it. llama_cpp uses per-request ignore_eos+n_predict; "
                        "dflash uses process-wide DFLASH_MIN_TOKENS and therefore requires "
                        "a uniform max_tokens trace. Aggregate output-token equality is "
                        "reported as pin_decode_ok."
                    ))
    ap.add_argument("--power-sample-interval", type=float, default=0.0,
                    help=(
                        "Optional NVIDIA GPU power sample interval in seconds. "
                        "When >0, samples nvidia-smi during the benchmark envelope "
                        "and reports integrated energy_j plus average watts."
                    ))
    ap.add_argument("--power-gpu-index", type=int, default=0,
                    help="GPU index passed to nvidia-smi for power sampling")
    args = ap.parse_args()

    if args.selftest:
        run_selftest()
        return

    if not args.arm:
        ap.error("--arm is required (or use --selftest)")

    if args.port == FORBIDDEN_PORT:
        print(f"ERROR: port {FORBIDDEN_PORT} is the user's live server. Forbidden.", file=sys.stderr)
        sys.exit(1)

    # Override server binary if explicitly provided (preserved/external build).
    if args.binary:
        global SERVER_BIN
        SERVER_BIN = Path(args.binary)
        print(f"[binary] override: {SERVER_BIN}")

    arm_cfg = ARMS[args.arm]
    server_type = arm_cfg.get("server", "dflash")
    n_repeats = 1 if args.smoke else args.n
    port = args.port
    max_ctx = args.max_ctx
    restart_per_turn = args.restart_per_turn

    # Verify binary (dflash arms only; llama_cpp arms use arm_cfg["binary"])
    if is_dflash_server(server_type):
        if not SERVER_BIN.exists():
            print(f"ERROR: server binary not found: {SERVER_BIN}", file=sys.stderr)
            sys.exit(1)
        bin_sha = sha256_file(SERVER_BIN)
        KNOWN_SHAS = {"92ee2985", "eef74aa0", "47212487", "5c659610", "fc27fff4", "ab9af0a7", "bab0c1dd"}
        if not any(bin_sha.startswith(s) for s in KNOWN_SHAS):
            print(f"WARNING: binary sha {bin_sha[:16]}... not a known PR274 head SHA")
    else:
        # llama_cpp arm: verify the arm's binary exists
        llama_bin = Path(arm_cfg.get("binary", str(LLAMA_BIN)))
        if not llama_bin.exists():
            print(f"ERROR: llama-server binary not found: {llama_bin}", file=sys.stderr)
            sys.exit(1)
        bin_sha = sha256_file(llama_bin)
        print(f"[binary] llama_cpp binary sha: {bin_sha[:16]}...")

    # Load trace
    trace_path = Path(args.trace)
    turns = [json.loads(line) for line in trace_path.read_text().splitlines() if line.strip()]
    if args.turn_limit is not None:
        if args.turn_limit <= 0:
            ap.error("--turn-limit must be positive")
        turns = turns[:args.turn_limit]
    quality_probe = trace_path.name.startswith("charbench_") and not args.smoke
    print(f"Loaded {len(turns)} turns from {trace_path}")
    pin_decode_tokens: Optional[int] = None
    trace_max_tokens_unique: list[int] = []
    if args.pin_decode_length:
        trace_max_tokens = []
        for i, turn in enumerate(turns, start=1):
            mt = request_max_tokens(turn)
            if mt is None:
                ap.error(f"--pin-decode-length requires positive integer max_tokens on turn {i}")
            trace_max_tokens.append(mt)
        trace_max_tokens_unique = sorted(set(trace_max_tokens))
        if is_dflash_server(server_type) and len(trace_max_tokens_unique) != 1:
            ap.error(
                "--pin-decode-length on dflash requires uniform trace max_tokens "
                "because DFLASH_MIN_TOKENS is process-wide"
            )
        if len(trace_max_tokens_unique) == 1:
            pin_decode_tokens = trace_max_tokens_unique[0]
        print(f"[pin-decode] trace max_tokens unique={trace_max_tokens_unique}")

    # Provenance
    git = git_info(REPO_DIR)
    _arm_tgt  = arm_cfg.get("model_target",  TGT)
    _arm_dd   = arm_cfg.get("model_draft",   DD)
    _arm_temp = arm_cfg.get("temperature",   TEMP)
    seed_to_send = args.seed if args.send_seed else None

    provenance = {
        "binary": str(SERVER_BIN) if is_dflash_server(server_type) else arm_cfg.get("binary", str(LLAMA_BIN)),
        "binary_sha256": bin_sha,
        "git_branch": git["branch"],
        "git_commit": git["commit"],
        "arm": args.arm,
        "arm_description": arm_cfg["description"],
        "arm_extra_args": arm_cfg["extra_args"],
        "arm_env": arm_cfg.get("env", {}),
        "server_type": server_type,
        "model_target": str(arm_cfg.get("model", _arm_tgt)),
        "model_draft_decode": str(_arm_dd),
        "model_draft_prefill": str(PD),
        "chat_template": str(TMPL),
        "max_ctx": max_ctx,
        "cache_type_k": arm_cfg.get("kv_cache_types", ("tq3_0",))[0],
        "cache_type_v": arm_cfg.get("kv_cache_types", ("tq3_0", "tq3_0"))[1],
        "temperature": _arm_temp,
        "seed_requested": args.seed,
        "seed_sent": args.send_seed,
        "n_repeats": n_repeats,
        "smoke": args.smoke,
        "restart_per_turn": restart_per_turn,
        "quality_probe": quality_probe,
        "pin_decode_length": args.pin_decode_length,
        "trace_max_tokens_unique": trace_max_tokens_unique,
        "pin_decode_tokens_process": pin_decode_tokens,
        "pin_decode_mechanism": (
            "DFLASH_MIN_TOKENS+DFLASH_DEGENERATE_RUN_TOKENS=0" if args.pin_decode_length and is_dflash_server(server_type)
            else "ignore_eos+n_predict" if args.pin_decode_length and server_type == "llama_cpp"
            else None
        ),
        "power_sampling_enabled": args.power_sample_interval > 0,
        "power_sample_interval_s": args.power_sample_interval if args.power_sample_interval > 0 else None,
        "power_gpu_index": args.power_gpu_index if args.power_sample_interval > 0 else None,
        "port": port,
        "trace": str(trace_path),
        "n_turns_in_trace": len(turns),
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
    }

    print("=" * 72)
    print(f"ARM: {args.arm} — {arm_cfg['description']}  [{server_type}]")
    print(f"Binary SHA: {bin_sha[:16]}...")
    print(f"Branch: {git['branch']} @ {git['commit'][:12]}")
    if args.smoke:
        mode_str = "SMOKE (turns 1-3, N=1)"
    elif quality_probe:
        mode_str = f"QUALITY PROBE ({len(turns)} turns, N={n_repeats})"
    else:
        mode_str = f"FULL ({n_repeats} repeats)"
    if restart_per_turn:
        mode_str += " [restart-per-turn: disk-cache cross-session mode]"
    print(f"Mode: {mode_str}")
    print("=" * 72)

    # Check no server already on port
    try:
        with urllib.request.urlopen(f"http://{HOST}:{port}/health", timeout=2) as r:
            if r.status == 200:
                print(f"ERROR: Something already running on port {port}. Refusing to start.", file=sys.stderr)
                sys.exit(1)
    except Exception:
        pass  # expected: nothing running

    vram_pre, vram_total = nvidia_smi_vram()
    print(f"VRAM before load: {vram_pre}/{vram_total} MiB")

    kv_dir = arm_cfg.get("kv_cache_dir")
    if kv_dir:
        import shutil
        shutil.rmtree(kv_dir, ignore_errors=True)
        os.makedirs(kv_dir, exist_ok=True)
        print(f"[kv-cache-dir] fresh empty dir: {kv_dir}")
        provenance["kv_cache_dir"] = kv_dir

    slot_save_dir = arm_cfg.get("slot_save_dir")
    if slot_save_dir:
        import shutil
        shutil.rmtree(slot_save_dir, ignore_errors=True)
        os.makedirs(slot_save_dir, exist_ok=True)
        print(f"[slot-save-dir] fresh empty dir: {slot_save_dir}")
        provenance["slot_save_dir"] = slot_save_dir
        provenance["slot_save_file"] = arm_cfg.get("slot_save_file", f"{args.arm}.slot")

    ts_str = datetime.now().strftime("%Y%m%d_%H%M%S")
    all_records: list[dict] = []

    acquire_gpu_lock()
    power_monitor: Optional[GpuPowerMonitor] = None
    power_summary: Optional[dict] = None
    try:
        if args.power_sample_interval > 0:
            power_monitor = GpuPowerMonitor(args.power_sample_interval, args.power_gpu_index)
            power_monitor.start()
            print(
                f"[power] sampling GPU {args.power_gpu_index} every "
                f"{power_monitor.interval_s:.2f}s"
            )
        if not restart_per_turn:
            # ── single-session mode (default) ────────────────────────────────────
            log_path = BENCH_DIR / f"srv_{args.arm}_{ts_str}.log"
            proc, log_f = launch_server(
                args.arm,
                arm_cfg,
                log_path,
                port=port,
                max_ctx=max_ctx,
                pin_decode_tokens=pin_decode_tokens,
            )

            try:
                print(f"Waiting for server on port {port}...", end=" ", flush=True)
                ok, start_error = wait_for_server(
                    port, timeout=360, proc=proc, log_path=log_path
                )
                if not ok:
                    print("FAILED")
                    print(f"Server did not come up. Check {log_path}", file=sys.stderr)
                    if start_error:
                        print(start_error, file=sys.stderr)
                    sys.exit(1)
                print("READY")

                vram_post, _ = nvidia_smi_vram()
                print(f"VRAM after load: {vram_post}/{vram_total} MiB")

                for rep in range(n_repeats):
                    print(f"\n--- Repeat {rep+1}/{n_repeats} ---")
                    recs = run_trace_repeat(
                        turns, port, log_path, args.arm,
                        smoke=args.smoke, repeat_idx=rep,
                        arm_cfg=arm_cfg,
                        pin_decode_length=args.pin_decode_length,
                        seed_to_send=seed_to_send,
                    )
                    all_records.extend(recs)

            finally:
                print(f"\nKilling server PID={proc.pid}...")
                kill_server(proc, log_f, wait_s=15)
                time.sleep(5)
                vram_after, _ = nvidia_smi_vram()
                print(f"VRAM after kill: {vram_after}/{vram_total} MiB")

        else:
            # ── restart-per-turn mode ─────────────────────────────────────────────
            log_dir = BENCH_DIR / f"srv_{args.arm}_{ts_str}_turns"
            log_dir.mkdir(exist_ok=True)
            print(f"[restart-per-turn] Per-turn logs dir: {log_dir}")

            for rep in range(n_repeats):
                print(f"\n--- Repeat {rep+1}/{n_repeats} (restart-per-turn) ---")
                recs = run_trace_restart_per_turn(
                    turns, port, args.arm, arm_cfg, log_dir,
                    max_ctx=max_ctx, smoke=args.smoke, repeat_idx=rep,
                    pin_decode_length=args.pin_decode_length,
                    pin_decode_tokens=pin_decode_tokens,
                    seed_to_send=seed_to_send,
                )
                all_records.extend(recs)

            vram_after, _ = nvidia_smi_vram()
            print(f"VRAM after all per-turn servers killed: {vram_after}/{vram_total} MiB")

    finally:
        if power_monitor is not None:
            power_summary = power_monitor.stop()
            provenance["power"] = power_summary
            print(
                "[power] "
                f"energy_j={power_summary.get('energy_j')} "
                f"avg_power_w={power_summary.get('avg_power_w')} "
                f"samples={power_summary.get('valid_sample_count')}"
            )
        release_gpu_lock()

    # Aggregate
    n_turns_ran = min(3, len(turns)) if args.smoke else len(turns)
    per_turn = aggregate_turns(all_records, n_turns_ran)
    arm_agg  = aggregate_arm(per_turn)
    if not args.pin_decode_length:
        for key in (
            "pin_decode_ok",
            "pin_decode_turns",
            "pin_decode_tool_turns",
            "pin_decode_non_tool_turns",
            "pin_decode_non_tool_mismatch_count",
            "pin_decode_non_tool_ok",
            "pin_decode_tool_stop_conflict",
            "pin_decode_claim_scope",
        ):
            arm_agg[key] = None
    if power_summary is not None:
        arm_agg["energy_j"] = power_summary.get("energy_j")
        arm_agg["avg_power_w"] = power_summary.get("avg_power_w")
        arm_agg["max_power_w"] = power_summary.get("max_power_w")
        arm_agg["mean_gpu_util_pct"] = power_summary.get("mean_gpu_util_pct")
        arm_agg["max_memory_used_mib"] = power_summary.get("max_memory_used_mib")
    if args.pin_decode_length and arm_agg.get("pin_decode_ok") is False:
        arm_agg["ok"] = False
        arm_agg["censored"] = True
        arm_agg["invalid_reason"] = "pin_decode_length_violation"
    apply_fair_contract(arm_agg, args.pin_decode_length)

    # Write results. Censored runs are useful for diagnosis but must not sit
    # beside valid benchmark rows where they can be cited by accident.
    suffix = "smoke" if args.smoke else "quality" if quality_probe else "full"
    artifact_dir = BENCH_DIR / ("quarantine" if arm_agg.get("censored") else "results")
    artifact_dir.mkdir(exist_ok=True)
    artifact_suffix = f"{suffix}_invalid" if arm_agg.get("censored") else suffix
    report_path = artifact_dir / f"{args.arm}_{ts_str}_{artifact_suffix}.md"
    raw_path    = artifact_dir / f"{args.arm}_{ts_str}_{artifact_suffix}_raw.json"

    raw_path.write_text(json.dumps({
        "provenance": provenance,
        "per_turn_median": per_turn,
        "arm_aggregate": arm_agg,
        "all_records": all_records,
    }, indent=2, default=str))

    write_report(
        args.arm, per_turn, arm_agg, provenance, report_path,
        smoke=args.smoke, run_kind=suffix,
    )

    # Print summary
    print("\n" + "=" * 72)
    print(f"RESULTS: {args.arm}")
    print("=" * 72)
    print(f"ok                        : {arm_agg.get('ok')}")
    print(f"censored                  : {arm_agg.get('censored')}")
    print(f"valid_turns               : {arm_agg.get('valid_turns')}/{arm_agg.get('expected_turns')}")
    print(f"error_count               : {arm_agg.get('error_count')}")
    if arm_agg.get("fair_contract"):
        fc = arm_agg["fair_contract"]
        print(f"fair_claim_class          : {fc.get('claim_class')}")
        print(f"fair_row_usable           : {fc.get('row_usable')}")
        print(f"correctness_gate_ok       : {fc.get('correctness_gate_ok')}")
        print(f"equal_output_speed_ok     : {fc.get('equal_output_speed_ok')}")
    print(f"total_wall_s              : {arm_agg.get('total_wall_s')}")
    if arm_agg.get("energy_j") is not None:
        print(f"energy_j                 : {arm_agg.get('energy_j')}")
        print(f"avg_power_w              : {arm_agg.get('avg_power_w')}")
        print(f"max_power_w              : {arm_agg.get('max_power_w')}")
        print(f"mean_gpu_util_pct        : {arm_agg.get('mean_gpu_util_pct')}")
        print(f"max_memory_used_mib      : {arm_agg.get('max_memory_used_mib')}")
    print(f"total_prefill_s           : {arm_agg.get('total_prefill_s')}")
    print(f"total_decode_s            : {arm_agg.get('total_decode_s')}")
    print(f"sum_fresh_prefill_tokens  : {arm_agg.get('sum_fresh_prefill_tokens')}")
    print(f"sum_requested_out_tokens  : {arm_agg.get('sum_requested_out_tokens')}")
    print(f"sum_out_tokens            : {arm_agg.get('sum_out_tokens')}")
    print(f"out_token_mismatch_count  : {arm_agg.get('out_token_mismatch_count')}")
    print(f"pin_decode_ok             : {arm_agg.get('pin_decode_ok')}")
    if arm_agg.get("pin_decode_claim_scope") is not None:
        print(f"pin_decode_claim_scope    : {arm_agg.get('pin_decode_claim_scope')}")
        print(f"pin_decode_tool_turns     : {arm_agg.get('pin_decode_tool_turns')}")
        print(f"pin_decode_non_tool_turns : {arm_agg.get('pin_decode_non_tool_turns')}")
        print(f"pin_decode_tool_stop_conflict: {arm_agg.get('pin_decode_tool_stop_conflict')}")
        print(f"pin_decode_non_tool_ok    : {arm_agg.get('pin_decode_non_tool_ok')}")
    print(f"mean_cache_hit_ratio      : {arm_agg.get('mean_cache_hit_ratio')}")
    print(f"mean_decode_tps           : {arm_agg.get('mean_decode_tps')}")
    print(f"weighted_prompt_prefill_tps: {arm_agg.get('weighted_prompt_prefill_tps')}")
    print(f"weighted_fresh_prefill_tps : {arm_agg.get('weighted_fresh_prefill_tps')}")
    print(f"weighted_decode_tps       : {arm_agg.get('weighted_decode_tps')}")
    print(f"spec_engagement_rate      : {arm_agg.get('spec_engagement_rate')}")
    print(f"mean_accept_rate          : {arm_agg.get('mean_accept_rate')}")
    if arm_agg.get('mean_disk_hit_rate') is not None:
        print(f"mean_disk_hit_rate        : {arm_agg.get('mean_disk_hit_rate')}")
    if arm_agg.get('sum_tool_expected_turns') is not None:
        print(
            "tool_expected_valid       : "
            f"{arm_agg.get('sum_tool_expected_valid_turns')}/"
            f"{arm_agg.get('sum_tool_expected_turns')}"
        )
        print(f"unexpected_tool_call_rate : {arm_agg.get('unexpected_tool_call_rate')}")
    print(f"tool_call_valid_rate      : {arm_agg.get('tool_call_valid_rate')}")
    print(f"charbench_valid_rate      : {arm_agg.get('charbench_valid_rate')}")
    print(f"\nRaw:    {raw_path}")
    print(f"Report: {report_path}")

    # Smoke-specific check
    if args.smoke:
        print("\n=== SMOKE GATE CHECKS ===")
        t1 = next((t for t in per_turn if t["turn"] == 1), {})
        t2 = next((t for t in per_turn if t["turn"] == 2), {})
        pl1 = t1.get("prefix_len", 0)
        pl2 = t2.get("prefix_len")
        if server_type == "llama_cpp" and not restart_per_turn:
            # llama.cpp reports prompt_n as fresh prompt work after slot/cache reuse,
            # not as the full logical prompt length. Reuse is therefore visible as
            # reduced post-cold prompt eval and in checkpoint restore logs.
            p1 = t1.get("prompt_tokens")
            prompt_reuse = p1 is not None and any(
                t.get("prompt_tokens") is not None and t.get("prompt_tokens") < p1
                for t in per_turn
                if t.get("turn", 0) > 1
            )
            try:
                llama_log_text = log_path.read_text(errors="replace")
            except Exception:
                llama_log_text = ""
            log_reuse = "restored context checkpoint" in llama_log_text
            reuse_ok = bool(prompt_reuse or log_reuse)
            print(
                f"llama prompt-cache reuse after cold turn: "
                f"{'PASS' if reuse_ok else 'FAIL (cache reuse not visible)'}"
            )
        elif restart_per_turn:
            # Disk cache stores chunked checkpoints; the first reconnect hit can appear
            # after turn 2 on short traces.
            cold_ok = pl1 in (0, None)
            reuse_ok = cold_ok and any(
                (t.get("prefix_len") or 0) > 0 or (t.get("disk_hit_rate") or 0) > 0
                for t in per_turn
                if t.get("turn", 0) > 1
            )
            print(
                f"disk reconnect reuse after cold turn: "
                f"{'PASS' if reuse_ok else 'FAIL (no disk reuse visible)'}"
            )
        else:
            # Same-process pooled snapshots can become visible after turn 2 on
            # short traces; require cold turn 1 plus any later warm prefix hit.
            cold_ok = pl1 in (0, None)
            reuse_ok = cold_ok and any(
                (t.get("prefix_len") or 0) > 0
                for t in per_turn
                if t.get("turn", 0) > 1
            )
            print(
                f"prefix reuse after cold turn: "
                f"{'PASS' if reuse_ok else 'FAIL (cache reuse not visible)'}"
            )

        # Check decode_tps non-null
        dec_ok = any(t.get("decode_tps") is not None for t in per_turn)
        print(f"decode_tps non-null:     {'PASS' if dec_ok else 'FAIL (no decode metrics)'}")

        # Check parser not returning all nulls
        any_metrics = any(t.get("prompt_tokens") is not None for t in per_turn)
        print(f"parser extracting data:  {'PASS' if any_metrics else 'FAIL (all nulls)'}")
        pin_ok = (not args.pin_decode_length) or arm_agg.get("pin_decode_ok") is True
        if args.pin_decode_length:
            print(f"pin_decode_ok:           {'PASS' if pin_ok else 'FAIL (out-token mismatch)'}")

        if not reuse_ok or not dec_ok or not any_metrics or not pin_ok:
            print("\nSMOKE FAILED — do NOT proceed to full run.")
            print("Debug info:")
            for t in per_turn:
                print(f"  turn {t['turn']}: {json.dumps(t, default=str)}")
            if restart_per_turn:
                print(f"\nPer-turn server logs: {locals().get('log_dir', '?')}")
            else:
                print(f"\nLast 50 server log lines from {log_path}:")
                try:
                    lines = log_path.read_text().splitlines()
                    for ln in lines[-50:]:
                        print(f"  {ln}")
                except Exception:
                    pass
            sys.exit(2)
        else:
            print("\nSMOKE PASSED — safe to run full sweep.")

    if not args.smoke and not arm_agg.get("ok", False):
        print("\nRUN CENSORED — artifacts quarantined; this arm is not a valid speed row.")
        sys.exit(3)


if __name__ == "__main__":
    main()
