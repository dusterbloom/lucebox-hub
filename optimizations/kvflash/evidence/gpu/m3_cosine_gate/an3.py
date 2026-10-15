import numpy as np
b = '/mnt/kimi-k3/build/am_fixtures_v2/'
a = np.fromfile(b + 'traj1_12k_exactq/L03_q.f32', dtype=np.float32).reshape(24, 640, 256)
eng = np.fromfile('/mnt/kimi-k3/build/m3-diag/plane.f16', dtype=np.float16).astype(np.float32).reshape(16, 24, 640, 256)[0]
ra, re = a.reshape(-1, 256), eng.reshape(-1, 256)
print('abs profile corr', np.corrcoef(np.abs(ra).mean(0), np.abs(re).mean(0))[0, 1])
C = np.corrcoef(ra.T, re.T)[:256, 256:]
print('max |corr| dim x dim', np.abs(C).max(), 'diag mean', np.abs(np.diag(C)).mean())
# token-axis correlation per head, pooling dims: corr between heads' mean vectors
ha, he = a.mean(1), eng.mean(1)
print('mean-q per-head corr diag', np.mean([np.corrcoef(ha[i], he[i])[0, 1] for i in range(24)]))
Ch = np.corrcoef(ha, he)[:24, 24:]
print('head perm best', Ch.argmax(1))
# same across engine layers: does ref layer-3 match any engine layer?
for l in range(16):
    x = eng_l = np.fromfile('/mnt/kimi-k3/build/m3-diag/plane.f16', dtype=np.float16, count=24 * 640 * 256, offset=l * 24 * 640 * 256 * 2).astype(np.float32).reshape(-1, 256)
    cs = ((ra / np.linalg.norm(ra, axis=-1, keepdims=True)) * (x / np.linalg.norm(x, axis=-1, keepdims=True))).sum(-1).mean()
    print('eng layer idx', l, round(float(cs), 4))
