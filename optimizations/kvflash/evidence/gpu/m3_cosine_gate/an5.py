import numpy as np
b = '/mnt/kimi-k3/build/am_fixtures_v2/'
D = '/mnt/kimi-k3/build/m3-diag/'


def nrm(x):
    return x / np.linalg.norm(x, axis=-1, keepdims=True)


def stats(name, x):
    x = np.asarray(x).ravel()
    print(f'{name}: min {x.min():.4f} median {np.median(x):.4f} max {x.max():.4f}')


# step 1: K and V of block 18 at layer 3, engine (F16 KV) vs fixture traj1_12k
for w in 'kv':
    ref = np.fromfile(b + f'traj1_12k/L03_{w}.f32', dtype=np.float32).reshape(4, 640, 256)
    eng = np.fromfile(D + f'kv_{w}.f16', dtype=np.float16).astype(np.float32).reshape(4, 640, 256)
    cos = (nrm(ref) * nrm(eng)).sum(-1).mean(1)  # per kv head, mean over tokens
    ratio = np.linalg.norm(eng, axis=(1, 2)) / np.linalg.norm(ref, axis=(1, 2))
    print(f'{w.upper()} per kv head cosine', cos.round(4), 'norm ratio', ratio.round(4))

# step 2: best-match values per engine Q row over all fixture rows of the layer
ref = np.fromfile(b + 'traj1_12k_exactq/L03_q.f32', dtype=np.float32).reshape(-1, 256)
for tag, f in (('default KV (Q4_0, FWHT-rotated Q)', 'plane.f16'), ('LUCE_KV_F16=1', 'plane_kvf16.f16')):
    eng = np.fromfile(D + f, dtype=np.float16, count=24 * 640 * 256).astype(np.float32).reshape(-1, 256)
    best = (nrm(eng) @ nrm(ref).T).max(1)
    stats(f'best-match cosine over {len(ref)} rows, {tag}', best)
