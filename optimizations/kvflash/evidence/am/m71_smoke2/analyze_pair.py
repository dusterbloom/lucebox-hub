import json, re, statistics as st
# AMOFF2 (A) vs AMON0 (B): same binary, no think time. Checks 1-3, backlog, medians.
D = '/mnt/kimi-k3/kvflash-gate-m7/m71_smoke2'
L = lambda a: [json.loads(l) for l in open(f'{D}/{a}/{a}.jsonl')]
A, B = L('AMOFF2'), L('AMON0')
print('records', len(A), len(B), 'errors', [r['id'] for r in A + B if r.get('error')])
key = lambda r: (r.get('content'), json.dumps(r.get('tool_calls')), r.get('completion_tokens'), r.get('finish'))
tk = lambda r: (r['prompt_tokens'], (r.get('timings') or {}).get('cached_prefix_tokens'), (r.get('timings') or {}).get('prefilled_tokens'))
print('check1 prefilled/cached diffs:', sum(tk(a) != tk(b) for a, b in zip(A, B)))
print('check2 within-pool (<=16320) answer diffs:', [a['id'][-12:] for a, b in zip(A, B) if a['prompt_tokens'] <= 16320 and key(a) != key(b)],
      'of', sum(a['prompt_tokens'] <= 16320 for a in A), '; past-pool diffs (expected):', sum(a['prompt_tokens'] > 16320 and key(a) != key(b) for a, b in zip(A, B)))
for arm in ('AMOFF2', 'AMON0'):
    log = open(f'{D}/{arm}/{arm}.server.log', errors='replace').read()
    print(arm, 'FATAL', log.count('FATAL'), 'disk_hit=true', log.count('disk_hit=true'), 'disk_hit=false', log.count('disk_hit=false'),
          'fit_wait lines', len(re.findall('fit_wait_ms', log)))
log = open(f'{D}/AMON0/AMON0.server.log', errors='replace').read()
fits = re.findall(r'turn fit: fitted=(\d+) .*?no_rows=(\d+) backlog=(\d+) plane=block (\d+) fit_ms=([\d.]+)', log)
print('check3 turn fit lines', len(fits), 'fitted>0', sum(int(f[0]) > 0 for f in fits), 'total fitted', sum(int(f[0]) for f in fits),
      'no_rows', sum(int(f[1]) for f in fits), 'failed fits', len(re.findall(r'stays exact', log)))
if fits:
    print('backlog per pass:', [int(f[2]) for f in fits], 'final', fits[-1][2], 'max', max(int(f[2]) for f in fits))
    ms = [float(f[4]) for f in fits]
    print('fit_ms median %.1f max %.1f' % (st.median(ms), max(ms)))
print('fit_wait_ms:', re.findall(r'fit_wait_ms=([\d.]+)', log))
def ser(R):
    T = [r['timings'] for r in R if r.get('timings')]
    return {'wall_s': [r['client_wall_s'] for r in R], 'prefill_ms': [t['prefill_ms'] for t in T],
            'prefill_tok_s': [t['prefilled_tokens'] / (t['prefill_ms'] / 1000) for t in T if t['prefill_ms'] > 0 and t['prefilled_tokens'] > 200],
            'decode_tok_s': [t['decode_tokens_per_sec'] for t in T]}
SA, SB = ser(A), ser(B)
for m in SA:
    print('%-14s AMOFF2 med %.2f mean %.2f | AMON0 med %.2f mean %.2f' % (m, st.median(SA[m]), st.mean(SA[m]), st.median(SB[m]), st.mean(SB[m])))
