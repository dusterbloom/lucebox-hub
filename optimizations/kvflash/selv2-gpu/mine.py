"""Read-only log mining. Run: python3 -B optimizations/kvflash/selv2-gpu/mine.py.

Writes only metrics.json beside this file. Uses the standard library; never runs
the harness, imports project modules, or initializes a GPU runtime.
"""
import hashlib
import json
import re
import statistics
import subprocess
from pathlib import Path

OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[2]
BASE = Path('/mnt/kimi-k3/split-work')
RETAINED_PATH = BASE / 'rlru4/retained.json'
RETAINED = set(json.loads(RETAINED_PATH.read_text()))
INPUTS = [RETAINED_PATH]
DIRS = {
    'OFF': BASE / 'rlru4k/v2/off',
    'ON6': BASE / 'rlru4k/v2/on6',
    'v2': BASE / 'selv2/e2e/v2',
}
SPLIT = re.compile(
    r'exact-scorer fast split: (?:n_q=(\d+) )?sealed=(\d+) resident=(\d+) '
    r'copy ([\d.]+) ms .*?, math ([\d.]+) ms')
SELECT = re.compile(
    r'mode=4 scorer=exact.*?: (\d+) page events; (\d+) in '
    r'\(([\d.]+) MiB H2D\).*? scorer ([\d.]+) ms')
RECENT = re.compile(r'recent-pool @\d+: (\d+) page events; (\d+) in \(([\d.]+) MiB H2D\)')
REPLAY = re.compile(r'R4=(\d+) \(end (\d+), re-prefill (\d+) tokens\)')


def summary(values):
    xs = sorted(x for x in values if x is not None)
    if not xs:
        return None
    return {
        'n': len(xs), 'median': statistics.median(xs),
        'mean': statistics.mean(xs), 'min': xs[0], 'max': xs[-1],
        'p90_order_statistic': xs[int(.9 * (len(xs) - 1))],
    }


def read_arm(directory):
    rowfile, logfile = (directory / ('QK16KND' + ext) for ext in ('.jsonl', '.server.log'))
    INPUTS.extend([rowfile, logfile])
    rows = [json.loads(line) for line in rowfile.read_text().splitlines()]
    blocks = logfile.read_text().split('] chat START ')[1:]
    assert len(rows) == len(blocks), (directory, len(rows), len(blocks))
    result = {}
    for row, block in zip(rows, blocks):
        t = row['timings']
        x = {
            'kind': row.get('kind'), 'retained': row['id'] in RETAINED,
            'prefill_ms': t['prefill_ms'], 'decode_ms': t['decode_ms'],
            'decode_tok_s': t['decode_tokens_per_sec'],
            'wall_ms': 1000 * row['client_wall_s'],
            'pt': t['prefilled_tokens'], 'cached_prefix': t['cached_prefix_tokens'],
            'completion_tokens': row['completion_tokens'],
            'prefill_timing_lines': block.count('[prefill-timing]'),
            'live_restore': int('[kvflash] live restore:' in block),
        }
        x['outside_ms'] = x['wall_ms'] - x['prefill_ms'] - x['decode_ms']
        m = SPLIT.search(block)
        if m:
            x.update(nq=int(m[1] or 1), sealed_chunks=int(m[2]), resident_chunks=int(m[3]),
                     scorer_copy_ms=float(m[4]), scorer_math_ms=float(m[5]))
        m = SELECT.search(block)
        x['reselected'] = bool(m)
        if m:
            x.update(page_events=int(m[1]), page_ins=int(m[2]), h2d_mib=float(m[3]), scorer_ms=float(m[4]))
        m = RECENT.search(block)
        if m:
            x.update(recent_events=int(m[1]), recent_ins=int(m[2]), recent_h2d_mib=float(m[3]))
        m = REPLAY.search(block)
        if m:
            x.update(r=int(m[1]), end=int(m[2]), replay_tokens=int(m[3]))
        m = re.search(r'selector v2: (\d+) of (\d+) queries', block)
        if m:
            x.update(captured_queries=int(m[1]), requested_queries=int(m[2]))
        x['scorer_free_prefill_ms'] = x['prefill_ms'] - x.get('scorer_ms', 0)
        x['scorer_free_wall_ms'] = x['wall_ms'] - x.get('scorer_ms', 0)
        assert row['id'] not in result
        result[row['id']] = x
    return result


def tab(rows):
    keys = set().union(*(x.keys() for x in rows)) if rows else set()
    return {k: summary(x.get(k) for x in rows)
            for k in sorted(keys - {'kind', 'retained', 'reselected'})}


arms = {a: read_arm(d) for a, d in DIRS.items()}
for a, rows in arms.items():
    assert RETAINED <= rows.keys(), a
    assert len(rows) == 1106
assert len(RETAINED) == 951
assert {i for i in RETAINED if arms['ON6'][i]['reselected']} == {
    i for i in RETAINED if arms['v2'][i]['reselected']}
selected = {i for i in RETAINED if arms['v2'][i]['reselected']}
sets = {'probes': RETAINED, 'reselected_probes': selected, 'fitting_probes': RETAINED - selected}
metrics = {'arms': {}, 'paired_differences': {}, 'gpu_only_projections': {}}
for a, rows in arms.items():
    metrics['arms'][a] = {
        'n_requests': len(rows), 'n_reselects': sum(x['reselected'] for x in rows.values()),
        **{name: tab([rows[i] for i in ids]) for name, ids in sets.items()},
        'all_reselects': tab([x for x in rows.values() if x['reselected']]),
    }
for a, b in [('ON6', 'OFF'), ('v2', 'OFF'), ('v2', 'ON6')]:
    metrics['paired_differences'][a + '-' + b] = {
        name: {k: summary(arms[a][i][k] - arms[b][i][k] for i in ids)
               for k in ['prefill_ms', 'wall_ms', 'decode_ms', 'outside_ms',
                         'scorer_free_prefill_ms', 'scorer_free_wall_ms']}
        for name, ids in sets.items()}
for a in ['v2', 'ON6']:
    metrics['gpu_only_projections'][a] = {
        str(g): {
            k: statistics.median(arms[a][i][k] - arms[a][i].get('scorer_ms', 0)
                                 + g * arms[a][i]['reselected'] for i in RETAINED)
            for k in ['prefill_ms', 'wall_ms']}
        for g in [0, 5, 20, 50]}

smoke = {a: read_arm(BASE / 'rlru4k/v2/smoke' / a) for a in ['off', 'noop', 'on6']}
smoke_ids = [i for i, x in smoke['on6'].items() if x['kind'] == 'probe' and x['reselected']]
smoke_diffs = []
for i in smoke_ids:
    off, noop, on = (smoke[a][i] for a in ['off', 'noop', 'on6'])
    # Noop omits the apply log's scorer time. Copy + math is its printed proxy.
    no_score = noop['prefill_ms'] - noop['scorer_copy_ms'] - noop['scorer_math_ms']
    on_no_score = on['prefill_ms'] - on['scorer_ms']
    smoke_diffs.append({
        'id': i, 'replay_tokens': noop['replay_tokens'],
        'noop_minus_off_no_scorer_ms': no_score - off['prefill_ms'],
        'on6_minus_off_no_scorer_ms': on_no_score - off['prefill_ms'],
        'on6_minus_noop_no_scorer_ms': on_no_score - no_score})
metrics['smoke'] = {'rows': smoke_diffs,
    'summary': {k: summary(x[k] for x in smoke_diffs) for k in smoke_diffs[0] if k != 'id'}}

# Byte counts from q4_0's 18 bytes per 32 values and engine dimensions.
T, L, Hk, Hq, D, Nq, ct = 52224, 16, 4, 24, 256, 64, 64
kbyte = L * Hk * (D // 32) * 18
metrics['derived'] = {
    'context_tokens': T, 'k_bytes_per_token': kbyte, 'kv_bytes_per_chunk': 2 * kbyte * ct,
    'all_K_bytes': T * kbyte, 'nonresident_K_bytes_at_full_pool': (T - 16384) * kbyte,
    'all_K_fp16_bytes': T * L * Hk * D * 2,
    'Q_fp32_bytes': Nq * L * Hq * D * 4,
    'LSE_fp32_bytes': (T // ct) * Nq * L * Hq * 4,
    'raw_logits_fp32_bytes': T * Nq * L * Hq * 4,
    'flops_full_context': 2 * T * Nq * L * Hq * D,
    'flops_at_selected_probe_median': 2 * 521 * ct * Nq * L * Hq * D,
    'meanK_fp16_per128_bytes': (T // 128) * L * Hk * D * 2,
    'minmax_fp16_per64_bytes': (T // 64) * L * Hk * D * 2 * 2,
    'dma_calls_per_chunk_per_direction': L * Hk * 2,
    'v2_reselects_with_64_queries': sum(x.get('captured_queries') == 64 for x in arms['v2'].values()),
}
metrics['provenance'] = {
    'source_ref': 'kvflash/2.0c-selector-v2',
    'resolved_commit': subprocess.check_output(['git', '-C', str(ROOT), 'rev-parse',
        'kvflash/2.0c-selector-v2'], text=True).strip(),
    'input_sha256': {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in INPUTS},
}
(OUT / 'metrics.json').write_text(json.dumps(metrics, indent=2) + '\n')
print(json.dumps({'output': str(OUT / 'metrics.json'), 'probes': len(RETAINED),
                  'reselected_probes': len(selected), 'smoke': metrics['smoke']['summary'],
                  'derived': metrics['derived']}, indent=2))
