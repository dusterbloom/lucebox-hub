import numpy as np
b = '/mnt/kimi-k3/build/am_fixtures_v2/'
a = np.fromfile(b + 'traj1_12k_exactq/L03_q.f32', dtype=np.float32).reshape(24, 640, 256)
eng = np.fromfile('/mnt/kimi-k3/build/m3-diag/plane.f16', dtype=np.float16, count=24 * 640 * 256).astype(np.float32).reshape(24, 640, 256)
ra, re = a.reshape(-1, 256), eng.reshape(-1, 256)
C = np.corrcoef(ra.T, re.T)[:256, 256:]
am = np.abs(C).argmax(1)
mx = np.abs(C).max(1)
print('dims with |corr|>0.5:', int((mx > 0.5).sum()))
print([(i, int(am[i]), round(float(mx[i]), 2)) for i in range(0, 256, 1) if mx[i] > 0.5][:60])
print('rows 0..4 of dim argmax', am[:16], mx[:16].round(2))
