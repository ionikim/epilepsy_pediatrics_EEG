import random
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

W = np.where(corr > 0.3, corr, 0.0)


def run_lpa_wrapped(max_iter, seed):
    random.seed(seed)
    n      = W.shape[0]
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


terminations = [1, 2, 5, 10, 20, 50, 100]
cpu, nmi = sweet_spot_experiment(run_fn=run_lpa_wrapped, terminations=terminations, n_runs=5)
plot_sweet_spot(terminations, cpu, nmi,
                title="Sweet Spot — Label Propagation",
                xlabel="Max Iterations",
                out=BASE_DIR / "reports" / "figures" / "lpa_sweet_spot.png")
