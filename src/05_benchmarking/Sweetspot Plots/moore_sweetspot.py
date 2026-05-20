import warnings
from collections import defaultdict
from pathlib import Path

import numpy as np
import scipy.sparse as sp

from sweetspot_framework import sweet_spot_experiment, plot_sweet_spot

_HERE    = Path(__file__).resolve().parent
BASE_DIR = _HERE.parents[2]
NPZ_PATH = (
    BASE_DIR / "data" / "graphs" / "adjacency_sparse"
    / "inter_to_ict_chb01_03_2980_3010_adjacency_sparse.npz"
)

N_CH = 23
N_TP = 7680

mat = sp.load_npz(NPZ_PATH).tocsr()
ch_temporal = np.zeros((N_CH, N_TP))
for ch in range(N_CH):
    s, e = ch * N_TP, (ch + 1) * N_TP
    ch_temporal[ch] = np.array(mat[s:e, s:e].sum(axis=1)).flatten()

with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    corr = np.corrcoef(ch_temporal)
corr = np.nan_to_num(corr, nan=0.0)
np.fill_diagonal(corr, 0.0)

W = np.where(corr > 0, corr, 0.0)

# build sorted edge list (heaviest first) — deterministic base
all_edges = sorted(
    [(i, j, W[i, j]) for i in range(N_CH) for j in range(i + 1, N_CH) if W[i, j] > 0],
    key=lambda x: -x[2]
)
N_EDGES = len(all_edges)


def stream_moore_partial(edges, min_dq=0.0):
    n       = N_CH
    degrees = W.sum(axis=1)
    m       = W.sum() / 2.0
    two_m   = 2.0 * m

    parent = list(range(n))
    size   = [1] * n
    a      = degrees.copy()
    e      = defaultdict(lambda: defaultdict(float))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(x, y):
        rx, ry = find(x), find(y)
        if rx == ry:
            return
        if size[rx] < size[ry]:
            rx, ry = ry, rx
        parent[ry] = rx
        size[rx]  += size[ry]
        a[rx]     += a[ry]
        a[ry]      = 0.0
        for other, cnt in list(e[ry].items()):
            if other == rx:
                e[rx].pop(ry, None)
                e[ry].pop(rx, None)
                continue
            e[rx][other] = e[rx].get(other, 0.0) + cnt
            e[other][rx] = e[other].get(rx, 0.0) + cnt
            e[other].pop(ry, None)
        e.pop(ry, None)

    for u, v, w in edges:
        cu, cv = find(u), find(v)
        if cu == cv:
            continue
        e_ab = e[cu][cv] + e[cv][cu] + w
        dq   = 2.0 * (e_ab / m - (a[cu] * a[cv]) / (two_m ** 2))
        if dq > min_dq:
            union(cu, cv)
        else:
            e[cu][cv] += w
            e[cv][cu] += w

    labels = np.array([find(i) for i in range(n)])
    unique = np.unique(labels)
    remap  = {v: i for i, v in enumerate(unique)}
    return np.array([remap[l] for l in labels])


def run_moore_wrapped(max_edges, seed):
    rng = np.random.default_rng(seed)
    shuffled = all_edges.copy()
    rng.shuffle(shuffled)
    return stream_moore_partial(shuffled[:max_edges])


terminations = [10, 25, 50, 100, 150, N_EDGES]
cpu, nmi = sweet_spot_experiment(run_fn=run_moore_wrapped, terminations=terminations, n_runs=5)
plot_sweet_spot(terminations, cpu, nmi,
                title="Sweet Spot — Moore Streaming",
                xlabel="Processed Edges",
                out=BASE_DIR / "reports" / "figures" / "moore_sweet_spot.png")
