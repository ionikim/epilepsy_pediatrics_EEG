import random
import time
import warnings
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import scipy.sparse as sp
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import squareform

_HERE    = Path(__file__).resolve().parent
BASE_DIR = _HERE.parents[2]
NPZ_PATH = (
    BASE_DIR / "data" / "graphs" / "adjacency_sparse"
    / "inter_to_ict_chb01_03_2980_3010_adjacency_sparse.npz"
)

N_CH      = 23
N_TP      = 7680
THRESHOLD = 0.3
N_RUNS    = 5

# load & compute correlation matrix once
print("Loading data ...")
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

dist_mat = squareform(np.clip(1.0 - corr, 0.0, None), checks=False)
W        = np.where(corr > THRESHOLD, corr, 0.0)
Z        = linkage(dist_mat, method="ward")

all_edges = sorted(
    [(i, j, W[i, j]) for i in range(N_CH) for j in range(i + 1, N_CH) if W[i, j] > 0],
    key=lambda x: -x[2]
)
N_EDGES = len(all_edges)


def compute_nmi(a, b):
    a, b = np.asarray(a), np.asarray(b)
    ca, cb = np.unique(a), np.unique(b)
    pa = np.array([(a == x).mean() for x in ca])
    pb = np.array([(b == x).mean() for x in cb])
    P  = np.zeros((len(ca), len(cb)))
    for i, x in enumerate(ca):
        for j, y in enumerate(cb):
            P[i, j] = ((a == x) & (b == y)).mean()
    mi = sum(P[i, j] * np.log(P[i, j] / (pa[i] * pb[j]))
             for i in range(len(ca)) for j in range(len(cb)) if P[i, j] > 0)
    ha = -sum(p * np.log(p) for p in pa if p > 0)
    hb = -sum(p * np.log(p) for p in pb if p > 0)
    return 2 * mi / (ha + hb) if ha + hb > 0 else 1.0


def run_experiment(run_fn, terminations):
    cpu_times, nmi_scores = [], []
    for t in terminations:
        runs = []
        t0 = time.perf_counter()
        for r in range(N_RUNS):
            runs.append(run_fn(t, seed=r))
        cpu_times.append(time.perf_counter() - t0)
        nmis = [compute_nmi(runs[0], x) for x in runs[1:]]
        nmi_scores.append(np.mean(nmis))
    return np.array(cpu_times), np.array(nmi_scores)


# Ward
def run_ward(k, seed=0):
    return fcluster(Z, k, criterion="maxclust")


# LPA
def run_lpa(max_iter, seed):
    random.seed(seed)
    n      = N_CH
    labels = list(range(n))
    for _ in range(max_iter):
        changes = 0
        order   = list(range(n))
        random.shuffle(order)
        for node in order:
            lw = defaultdict(float)
            for j in range(n):
                if j != node and W[node, j] > 0:
                    lw[labels[j]] += W[node, j]
            if not lw:
                continue
            new_label = max(lw, key=lw.get)
            if labels[node] != new_label:
                labels[node] = new_label
                changes += 1
        if changes == 0:
            break
    return np.array(labels)


# Spectral
def run_spectral(k, seed):
    A = corr.copy()
    A[A < THRESHOLD] = 0.0
    A[A < 0]         = 0.0
    np.fill_diagonal(A, 0.0)
    deg        = A.sum(axis=1)
    d_inv_sqrt = np.where(deg > 0, 1.0 / np.sqrt(np.maximum(deg, 1e-10)), 0.0)
    L          = np.eye(N_CH) - np.diag(d_inv_sqrt) @ A @ np.diag(d_inv_sqrt)
    _, vecs    = np.linalg.eigh(L)
    emb        = vecs[:, :k].copy()
    norms      = np.linalg.norm(emb, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    emb /= norms
    rng         = np.random.default_rng(seed)
    best_labels, best_inertia = None, np.inf
    for _ in range(10):
        centers = emb[rng.choice(N_CH, k, replace=False)]
        for _ in range(300):
            dists  = np.linalg.norm(emb[:, None, :] - centers[None, :, :], axis=2)
            assign = np.argmin(dists, axis=1)
            new_c  = np.array([emb[assign == ki].mean(0) if (assign == ki).any()
                               else emb[rng.integers(N_CH)] for ki in range(k)])
            if np.allclose(new_c, centers):
                break
            centers = new_c
        inertia = sum(np.sum((emb[assign == ki] - centers[ki]) ** 2)
                      for ki in range(k) if (assign == ki).any())
        if inertia < best_inertia:
            best_inertia, best_labels = inertia, assign.copy()
    return best_labels


# Moore
def stream_moore_partial(edges):
    m     = W.sum() / 2.0
    two_m = 2.0 * m
    parent = list(range(N_CH))
    size   = [1] * N_CH
    a      = W.sum(axis=1).copy()
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
        if dq > 0:
            union(cu, cv)
        else:
            e[cu][cv] += w
            e[cv][cu] += w

    labels = np.array([find(i) for i in range(N_CH)])
    unique = np.unique(labels)
    remap  = {v: i for i, v in enumerate(unique)}
    return np.array([remap[l] for l in labels])


def run_moore(max_edges, seed):
    rng      = np.random.default_rng(seed)
    shuffled = all_edges.copy()
    rng.shuffle(shuffled)
    return stream_moore_partial(shuffled[:max_edges])


# run all experiments
print("Running Ward ...")
ward_x   = [2, 3, 4, 6, 8, 10, 15]
ward_cpu, ward_nmi = run_experiment(run_ward, ward_x)

print("Running LPA ...")
lpa_x    = [1, 2, 5, 10, 20, 50, 100]
lpa_cpu, lpa_nmi = run_experiment(run_lpa, lpa_x)

print("Running Spectral ...")
spec_x   = [2, 3, 4, 5, 6, 8, 10]
spec_cpu, spec_nmi = run_experiment(run_spectral, spec_x)

print("Running Moore ...")
moore_x  = [5, 10, 25, 50, 100, N_EDGES]
moore_cpu, moore_nmi = run_experiment(run_moore, moore_x)

print("Plotting ...")


def plot_single(ax, x, cpu, nmi, title, xlabel):
    ax2 = ax.twinx()
    ax.plot(x, cpu, color="crimson", marker="o", label="CPU time")
    ax2.plot(x, nmi, color="seagreen", marker="s", label="NMI")
    ax.set_xlabel(xlabel)
    ax.set_ylabel("CPU time [s]", color="crimson")
    ax2.set_ylabel("NMI", color="seagreen")
    ax2.set_ylim(0, 1.05)
    if len(x) > 2:
        idx = max(range(1, len(x) - 1),
                  key=lambda i: (nmi[i + 1] - nmi[i - 1]) /
                                (cpu[i + 1] - cpu[i - 1] + 1e-9))
        ax.axvline(x[idx], linestyle="--", color="black", alpha=0.6)
    ax.set_title(title, fontsize=11)


fig, axes = plt.subplots(2, 2, figsize=(12, 8))

plot_single(axes[0, 0], ward_x,  ward_cpu,  ward_nmi,  "Ward Hierarchical Clustering",    "Dendrogram cut (k)")
plot_single(axes[0, 1], lpa_x,   lpa_cpu,   lpa_nmi,   "Label Propagation (LPA)",          "Max iterations")
plot_single(axes[1, 0], spec_x,  spec_cpu,  spec_nmi,  "Spectral Laplacian Clustering",    "Number of clusters (k)")
plot_single(axes[1, 1], moore_x, moore_cpu, moore_nmi, "Moore Streaming",                  "Processed edges")

fig.suptitle(
    "Sweet-Spot Analysis: CPU–Stability Trade-offs across Community Detection Algorithms",
    fontsize=13, y=0.98
)
plt.tight_layout()
plt.savefig(BASE_DIR / "reports" / "figures" / "sweet_spot_4algorithms.png", dpi=150)
plt.close()
print("Done. Saved sweet_spot_4algorithms.png")
