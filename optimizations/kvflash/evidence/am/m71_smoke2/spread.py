import json, statistics as st, sys
# usage: python3 -I spread.py [dir]   A/A spread (AMOFF vs AMOFF2) next to the AM-on vs AM-off difference (AMON vs AMOFF).
D = sys.argv[1] if len(sys.argv) > 1 else '/mnt/kimi-k3/kvflash-gate-m7/m71_smoke2'
def load(a):
    return [json.loads(l) for l in open(f'{D}/{a}/{a}.jsonl')]
def series(R):
    T = [r['timings'] for r in R if r.get('timings')]
    pf = [t['prefill_ms'] for t in T]
    pps = [t['prefilled_tokens'] / (t['prefill_ms'] / 1000) for t in T if t['prefill_ms'] > 0 and t['prefilled_tokens'] > 200]
    return {'wall_s': [r['client_wall_s'] for r in R], 'prefill_ms': pf, 'prefill_tok_s': pps,
            'decode_tok_s': [t['decode_tokens_per_sec'] for t in T]}
S = {a: series(load(a)) for a in ('AMOFF', 'AMOFF2', 'AMON')}
print('%-14s %-8s %9s %9s %9s %9s' % ('metric', 'stat', 'AMOFF', 'AMOFF2', 'AMON', ''))
for m in S['AMOFF']:
    for name, f in (('median', st.median), ('mean', st.mean)):
        a, a2, b = (f(S[x][m]) for x in ('AMOFF', 'AMOFF2', 'AMON'))
        print('%-14s %-8s %9.2f %9.2f %9.2f   A/A spread %+.2f | ON-OFF %+.2f | ON-OFF2 %+.2f' % (m, name, a, a2, b, a2 - a, b - a, b - a2))
