import json, glob, math, sys
root='/home/peppi/Dev/lucebox/.claude/worktrees/kvflash-am/optimizations/kvflash/evidence/'
out=[]
for f in sorted(glob.glob(root+'**/*.jsonl',recursive=True)):
    try: rows=[json.loads(l) for l in open(f)]
    except Exception: continue
    rows=[r for r in rows if isinstance(r,dict) and r.get('timings') and r.get('client_wall_s') is not None and 'prefill_ms' in r['timings']]
    if len(rows)<20: continue
    ov=[r['client_wall_s']-(r['timings']['prefill_ms']+r['timings']['decode_ms'])/1000 for r in rows]
    w=[r['client_wall_s'] for r in rows]
    n=sum(o>1 for o in ov); tot=sum(w)
    t=0; ends=[]
    for o,x in zip(ov,w):
        t+=x
        if o>1 and x<6: ends.append(t)
    best=(0,0)
    if len(ends)>=6:
        for T in [x/10 for x in range(150,500)]:
            R=abs(sum(complex(math.cos(2*math.pi*e/T),math.sin(2*math.pi*e/T)) for e in ends))/len(ends)
            if R>best[0]: best=(R,T)
    print('%-70s n=%4d wall=%7.0fs spikes=%3d period_est=%5.1fs  (1 per %.0fs) shortspikes=%d R=%.2f (rand %.2f) T=%.1f'%(f.replace(root,''),len(rows),tot,n,tot/max(n,1),tot/max(n,1),len(ends),best[0],1/math.sqrt(max(len(ends),1)),best[1]))
