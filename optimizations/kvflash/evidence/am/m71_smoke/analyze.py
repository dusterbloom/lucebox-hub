import json, re, statistics as st, sys
D = '/mnt/kimi-k3/kvflash-gate-m7/m71_smoke'
def load(a):
    return [json.loads(l) for l in open(f'{D}/{a}/{a}.jsonl')]
A, B = load('AMOFF'), load('AMON')
print('records', len(A), len(B))
errs = [r['id'] for r in A + B if r.get('error')]
print('errors', errs)
# R1: prefilled and cached tokens per request, same order
first = None
diff_pf = []
diff_ans = []
for a, b in zip(A, B):
    assert a['id'] == b['id'], (a['id'], b['id'])
    ta, tb = a.get('timings') or {}, b.get('timings') or {}
    ka = (a['prompt_tokens'], ta.get('cached_prefix_tokens'), ta.get('prefilled_tokens'))
    kb = (b['prompt_tokens'], tb.get('cached_prefix_tokens'), tb.get('prefilled_tokens'))
    if ka != kb:
        diff_pf.append((a['id'], ka, kb))
    ans_a = (a.get('content'), json.dumps(a.get('tool_calls')), a.get('completion_tokens'), a.get('finish'))
    ans_b = (b.get('content'), json.dumps(b.get('tool_calls')), b.get('completion_tokens'), b.get('finish'))
    if ans_a != ans_b:
        diff_ans.append(a['id'])
print('R1 prefilled/cached token diffs:', len(diff_pf))
for d in diff_pf[:8]: print('  ', d)
print('answer diffs (all requests):', len(diff_ans), diff_ans[:6])
# logs
for arm in ('AMOFF', 'AMON'):
    log = open(f'{D}/{arm}/{arm}.server.log', errors='replace').read()
    print(arm, 'FATAL', len(re.findall('FATAL', log)), 'from-scratch/full-prefill lines',
          len(re.findall(r'(?i)fresh prefill|full re-?prefill|from.scratch|am_miss', log)),
          'disk_hit=true', len(re.findall('disk_hit=true', log)), 'disk_hit=false', len(re.findall('disk_hit=false', log)))
log = open(f'{D}/AMON/AMON.server.log', errors='replace').read()
fits = re.findall(r'turn fit: fitted=(\d+) ineligible=(\d+) already=(\d+) in_pool=(\d+) no_rows=(\d+) plane=block (\d+) fit_ms=([\d.]+) .*?host plane ([\d.]+) MiB, peak ([\d.]+) MiB', log)
print('turn fit lines', len(fits))
if fits:
    ms = [float(f[6]) for f in fits]
    ft = [int(f[0]) for f in fits]
    print('fit_ms median %.1f max %.1f ; fitted>0 turns %d ; total fitted %d ; no_rows total %d ; peak plane MiB %.1f' % (
        st.median(ms), max(ms), sum(1 for x in ft if x > 0), sum(ft), sum(int(f[4]) for f in fits), max(float(f[8]) for f in fits)))
print('block fit failures', len(re.findall(r'stays exact', log)), 'compacted lines', len(re.findall(r'compacted', log)))
print('first lines of turn fit:'); [print('  ', l) for l in re.findall(r'\[kvflash-am\] turn fit[^\n]*', log)[:3]]
# speeds
for name, R in (('AMOFF', A), ('AMON', B)):
    pf = [r['timings']['prefill_ms'] for r in R if r.get('timings')]
    dt = [r['timings']['decode_tokens_per_sec'] for r in R if r.get('timings')]
    pt = [r['timings']['prefilled_tokens'] for r in R if r.get('timings')]
    pps = [t / (m / 1000) for t, m in zip(pt, pf) if m > 0 and t > 200]
    wall = [r['client_wall_s'] for r in R]
    print(name, 'prefill_ms med %.1f mean %.1f | prefill tok/s (>200 tok) med %.0f | decode tok/s med %.1f | wall s med %.2f mean %.2f' % (
        st.median(pf), st.mean(pf), st.median(pps) if pps else 0, st.median(dt), st.median(wall), st.mean(wall)))
