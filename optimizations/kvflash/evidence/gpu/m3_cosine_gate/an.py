import numpy as np
d = '/mnt/kimi-k3/build/am_fixtures_v2/traj1_12k_exactq/'
eng = np.fromfile('/mnt/kimi-k3/build/m3-diag/plane.f16', dtype=np.float16).astype(np.float32).reshape(16, 24, 640, 256)
ref = np.fromfile(d + 'L03_q.f32', dtype=np.float32).reshape(24, 640, 256)
e = eng[0]


def nrm(x):
    return x / np.linalg.norm(x, axis=-1, keepdims=True)


R = nrm(ref).reshape(-1, 256)
E = nrm(e).reshape(-1, 256)
C = R @ E.T
best = C.argmax(1)
print('best cos mean', C.max(1).mean())
bh, bt = best // 640, best % 640
rh, rt = np.arange(R.shape[0]) // 640, np.arange(R.shape[0]) % 640
print('head match frac', (bh == rh).mean(), 'token match', (bt == rt).mean())
print('sample', list(zip(rh[:3], rt[:3], bh[:3], bt[:3])), list(zip(rh[700:703], rt[700:703], bh[700:703], bt[700:703])))
for lo, hi in ((0, 64), (64, 256)):
    a = ref[..., lo:hi].reshape(-1, hi - lo)
    b = e[..., lo:hi].reshape(-1, hi - lo)
    print(lo, hi, 'cos', (nrm(a) * nrm(b)).sum(-1).mean())
ct = (nrm(ref) * nrm(e)).sum(-1).mean(0)
print(ct[:5], ct[-5:])
print('eng row0 head0', e[0, 0, :6], 'ref', ref[0, 0, :6])
