import warnings
from pathlib import Path

import numpy as np
import scipy.sparse as sp
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import squareform

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

DIST_MATRIX = squareform(np.clip(1.0 - corr, 0.0, None), checks=False)
Z = linkage(DIST_MATRIX, method="ward")


def run_ward_wrapped(k, seed=0):
    return fcluster(Z, k, criterion="maxclust")


ks = [2, 3, 4, 6, 8, 10, 15]
cpu, nmi = sweet_spot_experiment(run_fn=run_ward_wrapped, terminations=ks, n_runs=2)
plot_sweet_spot(ks, cpu, nmi,
                title="Sweet Spot — Ward Hierarchical Clustering",
                xlabel="Number of Clusters (Dendrogram Cut)",
                out="ward_sweet_spot.png")
