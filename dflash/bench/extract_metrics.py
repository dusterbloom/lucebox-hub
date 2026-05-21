#!/usr/bin/env python3
"""Extract metrics from a bench run directory."""
import sys
import re
import json
import os

server_log, timing_path, meta_path, label, stamp = sys.argv[1:]

# Parse accept rates from server log (MTP)
accept_rates = []
drafter_skips = 0
if os.path.exists(server_log):
    for line in open(server_log):
        m = re.search(r'accept_rate=([\d.]+)', line)
        if m:
            accept_rates.append(float(m.group(1)))
        if 'drafter-skip' in line or 'drafter_skip' in line or '[skip]' in line.lower():
            drafter_skips += 1

# Parse timing
turns = []
ok_done_count = 0
if os.path.exists(timing_path):
    turns = json.load(open(timing_path))
    ok_done_count = sum(1 for t in turns if t.get("ok_done"))

num_turns = len(turns)
mean_wall = round(sum(t.get("wall_s", 0) for t in turns) / num_turns, 3) if num_turns else None
mean_completion = round(sum(t.get("completion_tokens", 0) for t in turns) / num_turns, 1) if num_turns else None

meta = {
    "label": label,
    "stamp": stamp,
    "num_turns": num_turns,
    "ok_done_count": ok_done_count,
    "ok_done_seen": ok_done_count > 0,
    "mean_accept_rate": round(sum(accept_rates) / len(accept_rates), 3) if accept_rates else None,
    "accept_rates": accept_rates,
    "drafter_skips": drafter_skips,
    "mean_wall_per_turn_s": mean_wall,
    "mean_completion_tokens": mean_completion,
    "turns": turns,
}
with open(meta_path, "w") as f:
    json.dump(meta, f, indent=2)

print(json.dumps(meta, indent=2))
