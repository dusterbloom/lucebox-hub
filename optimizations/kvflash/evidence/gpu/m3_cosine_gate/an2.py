import numpy as np
b = '/mnt/kimi-k3/build/am_fixtures_v2/'
a = np.fromfile(b + 'traj1_12k_exactq/L03_q.f32', dtype=np.float32).reshape(24, 640, 256)
c = np.fromfile(b + 'traj1_12k/L03_q.f32', dtype=np.float32).reshape(24, 640, 256)
eng = np.fromfile('/mnt/kimi-k3/build/m3-diag/plane.f16', dtype=np.float16).astype(np.float32).reshape(16, 24, 640, 256)[0]


def cs(x, y):
    return ((x / np.linalg.norm(x, axis=-1, keepdims=True)) * (y / np.linalg.norm(y, axis=-1, keepdims=True))).sum(-1).mean()


print('exactq vs ctxprefill dump', cs(a, c))
print('eng vs ctxprefill dump', cs(eng, c))
ta = np.fromfile(b + 'traj1_12k_exactq/tokens.i32', dtype=np.int32)
tc = np.fromfile(b + 'traj1_12k/tokens.i32', dtype=np.int32)
pt = np.fromfile(b + 'traj1_12k_exactq/prefix_tokens.i32', dtype=np.int32)
print('tokens equal', (ta == tc).all(), 'prefix tail equals tokens', (pt[-640:] == ta).all(), len(pt))
# shifted token compare: eng row t vs ref row t+s (same head)
for s in (-2, -1, 1, 2):
    if s > 0:
        print(s, cs(eng[:, :-s], a[:, s:]))
    else:
        print(s, cs(eng[:, -s:], a[:, :s]))
