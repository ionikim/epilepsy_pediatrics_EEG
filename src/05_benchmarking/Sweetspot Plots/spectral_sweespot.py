import warnings
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

N_CH      = 23
N_TP      = 7680
THRESHOLD = 0.3

mat = sp.load_npz(NPZ_PATH).tocsr()
ch_temporal = np.zeros((N_CH, N_TP))
for ch in range(N_CH):
    s, e = ch * N_TP, (ch + 1) * N_TP
    ch_temporal[ch] = np.array(mat[s:e, s:e].sum(axis=1)).flatten()

with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    CORR = np.corrcoef(ch_temporal)
CORR = np.nan_to_num(CORR, nan=0.0)


def spectral_clustering(corr_matrix, k, threshold, seed=42):
    A = corr_matrix.copy()
    A[A < threshold] = 0.0
    A[A < 0]         = 0.0
    np.fill_diagonal(A, 0.0)
    deg        = A.sum(axis=1)
    d_inv_sqrt = np.where(deg > 0, 1.0 / np.sqrt(np.maximum(deg, 1e-10)), 0.0)
    L          = np.eye(len(A)) - np.diag(d_inv_sqrt) @ A @ np.diag(d_inv_sqrt)
    _, vecs    = np.linalg.eigh(L)
    emb        = vecs[:, :k].copy()
    norms      = np.linalg.norm(emb, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    emb /= norms

    rng         = np.random.default_rng(seed)
    n           = len(emb)
    best_labels, best_inertia = None, np.inf
    for _ in range(10):
        centers = emb[rng.choice(n, k, replace=False)]
        for _ in range(300):
            dists  = np.linalg.norm(emb[:, None, :] - centers[None, :, :], axis=2)
            assign = np.argmin(dists, axis=1)
            new_c  = np.array([emb[assign == ki].mean(0) if (assign == ki).any()
                               else emb[rng.integers(n)] for ki in range(k)])
            if np.allclose(new_c, centers):
                break
            centers = new_c
        inertia = sum(np.sum((emb[assign == ki] - centers[ki]) ** 2)
                      for ki in range(k) if (assign == ki).any())
        if inertia < best_inertia:
            best_inertia, best_labels = inertia, assign.copy()
    return best_labels


def run_spectral_wrapped(k, seed):
    return spectral_clustering(CORR, k, THRESHOLD, seed=seed)


ks = [2, 3, 4, 5, 6, 8, 10]
cpu, nmi = sweet_spot_experiment(run_fn=run_spectral_wrapped, terminations=ks, n_runs=5)
plot_sweet_spot(ks, cpu, nmi,
                title="Sweet Spot — Spectral Laplacian Clustering",
                xlabel="Number of Clusters (k)",
                out="spectral_sweet_spot.png")
