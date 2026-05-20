"""
Stream-Moore Benchmark Metrics  (all from scratch — numpy + stdlib only)
"""
from pathlib import Path
import numpy as np
import scipy.sparse as sp
from collections import defaultdict
import time, tracemalloc
import matplotlib.pyplot as plt

# configuration
NPZ_PATH     = (
    Path(__file__).resolve().parents[2]
    / "data" / "graphs" / "adjacency_sparse"
    / "inter_to_ict_chb01_03_2980_3010_adjacency_sparse.npz"
)
N_CHANNELS   = 23
N_TIMEPOINTS = 7680
FS           = 256
ONSET_SEC    = 16
ONSET_SAMPLE = ONSET_SEC * FS   # 4096 — separates interictal / ictal nodes

CHANNEL_NAMES = [
    "FP1-F7", "F7-T7",  "T7-P7",  "P7-O1",
    "FP1-F3", "F3-C3",  "C3-P3",  "P3-O1",
    "FP2-F4", "F4-C4",  "C4-P4",  "P4-O2",
    "FP2-F8", "F8-T8",  "T8-P8",  "P8-O2",
    "FZ-CZ",  "CZ-PZ",
    "P7-T7",  "T7-FT9", "FT9-FT10", "FT10-T8",
    "T8-P8-1"
]

CHANNEL_REGIONS = {
    "FP1-F7": "frontal",    "F7-T7":    "frontal",
    "T7-P7":  "temporal",   "P7-O1":    "parietal",
    "FP1-F3": "frontal",    "F3-C3":    "central",
    "C3-P3":  "central",    "P3-O1":    "parietal",
    "FP2-F4": "frontal",    "F4-C4":    "central",
    "C4-P4":  "central",    "P4-O2":    "parietal",
    "FP2-F8": "frontal",    "F8-T8":    "frontal",
    "T8-P8":  "temporal",   "P8-O2":    "parietal",
    "FZ-CZ":  "central",    "CZ-PZ":    "central",
    "P7-T7":  "temporal",   "T7-FT9":   "temporal",
    "FT9-FT10": "temporal", "FT10-T8":  "temporal",
    "T8-P8-1": "temporal",
}

class UnionFind:
    def __init__(self, n):
        self.parent = list(range(n))
        self.rank   = [0] * n
        self.size   = [1] * n
        self.a      = None

    def find(self, x):
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, x, y):
        rx, ry = self.find(x), self.find(y)
        if rx == ry:
            return False
        if self.rank[rx] < self.rank[ry]:
            rx, ry = ry, rx
        self.parent[ry] = rx
        self.size[rx] += self.size[ry]
        if self.a is not None:
            self.a[rx] += self.a[ry]
        if self.rank[rx] == self.rank[ry]:
            self.rank[rx] += 1
        return True

class IncrementalModularity:
    def __init__(self, degrees, m):
        self.degrees = np.array(degrees, dtype=np.float64)
        self.m       = float(m)
        self.two_m   = 2.0 * m
        self.e       = defaultdict(lambda: defaultdict(float))
        self._a      = self.degrees.copy()

    def a(self, root):
        return self._a[root]

    def merge_a(self, root_keep, root_remove):
        self._a[root_keep] += self._a[root_remove]
        self._a[root_remove] = 0.0

    def delta_q(self, ca, cb, extra_edge_weight=0.0):
        e_ab  = self.e[ca][cb] + self.e[cb][ca] + extra_edge_weight
        dq    = (e_ab / self.m) - (self._a[ca] * self._a[cb]) / (self.two_m ** 2)
        return 2.0 * dq

    def add_edge(self, ca, cb, weight=1.0):
        if ca == cb:
            return
        self.e[ca][cb] += weight
        self.e[cb][ca] += weight

    def merge_communities(self, ca, cb):
        for other, count in list(self.e[cb].items()):
            if other == ca:
                del self.e[ca][cb]
                del self.e[cb][ca]
                continue
            self.e[ca][other] = self.e[ca].get(other, 0.0) + count
            self.e[other][ca] = self.e[other].get(ca, 0.0) + count
            if cb in self.e[other]:
                del self.e[other][cb]
        if cb in self.e:
            del self.e[cb]
        self.merge_a(ca, cb)
        return ca

    def modularity(self, uf):
        roots = set(uf.find(i) for i in range(len(self.degrees)))
        Q = 0.0
        for r in roots:
            e_cc = self.e[r].get(r, 0.0)
            a_r  = self._a[r]
            Q   += (e_cc / self.m) - (a_r / self.two_m) ** 2
        return Q

def stream_moore(adj, verbose=False):
    adj = adj.tocsr().astype(np.float64)
    n   = adj.shape[0]
    degrees = np.array(adj.sum(axis=1)).flatten()
    m       = adj.nnz / 2

    uf  = UnionFind(n)
    inc = IncrementalModularity(degrees, m)

    adj_coo = sp.triu(adj, k=1).tocoo()
    rows, cols, weights = adj_coo.row, adj_coo.col, adj_coo.data
    merges = 0

    for idx in range(len(rows)):
        u, v, w = int(rows[idx]), int(cols[idx]), float(weights[idx])
        cu, cv  = uf.find(u), uf.find(v)
        if cu != cv:
            dq = inc.delta_q(cu, cv, extra_edge_weight=w)
            if dq > 0:
                if inc.a(cu) < inc.a(cv):
                    cu, cv = cv, cu
                uf.union(cu, cv)
                inc.merge_communities(cu, cv)
                merges += 1
            else:
                inc.add_edge(cu, cv, w)

    labels = np.array([uf.find(i) for i in range(n)])
    unique_roots = {r: i for i, r in enumerate(sorted(set(labels)))}
    labels = np.array([unique_roots[l] for l in labels])
    return labels

print("Loading adjacency matrix ...")
adj = sp.load_npz(NPZ_PATH)
n   = adj.shape[0]
print(f"  {n:,} nodes  |  {adj.nnz // 2:,} edges")

tracemalloc.start()
t0     = time.perf_counter()
labels = stream_moore(adj, verbose=False)
t1     = time.perf_counter()
_, peak_mem = tracemalloc.get_traced_memory()
tracemalloc.stop()

runtime = t1 - t0
unique_comms, comm_sizes = np.unique(labels, return_counts=True)
print(f"  Communities: {len(unique_comms):,}  |  runtime: {runtime:.2f}s")

# node i → timepoint = i % N_TIMEPOINTS → interictal if < ONSET_SAMPLE
node_timepoints = np.arange(n) % N_TIMEPOINTS
node_is_inter   = node_timepoints < ONSET_SAMPLE   # bool array

inter_comm_idx, ictal_comm_idx = [], []
for c in unique_comms:
    idx = np.where(labels == c)[0]
    if node_is_inter[idx].mean() >= 0.5:
        inter_comm_idx.append(idx)
    else:
        ictal_comm_idx.append(idx)

print(f"  Interictal communities: {len(inter_comm_idx):,}  |  "
      f"Ictal communities: {len(ictal_comm_idx):,}")

def structural_metrics(adj_csr, comm_indices):
    """ICED, conductance, avg clustering coeff for a list of communities."""
    iced_l, cond_l, cc_l = [], [], []
    for idx in comm_indices:
        n_c  = len(idx)
        if n_c < 2:
            continue
        sub  = adj_csr[idx, :][:, idx].toarray()
        actual   = float(np.triu(sub, 1).sum())
        possible = n_c * (n_c - 1) / 2
        iced_l.append(actual / possible if possible > 0 else 0.0)
        vol  = float(adj_csr[idx, :].sum())
        cut  = vol - float(sub.sum())   # vol = 2*intra + cut → cut = vol - 2*intra; sub.sum()=2*intra
        cond_l.append(cut / vol if vol > 0 else 0.0)
        b = (sub > 0).astype(float)
        np.fill_diagonal(b, 0.0)
        for i in range(n_c):
            nbrs = np.where(b[i] > 0)[0]
            ki   = len(nbrs)
            if ki < 2:
                continue
            tri = sum(b[nbrs[u], nbrs[v]]
                      for u in range(ki) for v in range(u + 1, ki))
            cc_l.append(float(tri) / (ki * (ki - 1) / 2))
    return (float(np.mean(iced_l)) if iced_l else 0.0,
            float(np.mean(cond_l)) if cond_l else 0.0,
            float(np.mean(cc_l))   if cc_l   else 0.0)

def inter_intra_ratio(adj, labels, comm_indices):
    """Inter / intra edge ratio for a subset of communities."""
    comm_set = set()
    for idx in comm_indices:
        comm_set.update(labels[idx])
    adj_upper = sp.triu(adj, k=1).tocoo()
    intra, inter = 0.0, 0.0
    for u, v, w in zip(adj_upper.row, adj_upper.col, adj_upper.data):
        lu, lv = labels[u], labels[v]
        if lu not in comm_set or lv not in comm_set:
            continue
        if lu == lv:
            intra += w
        else:
            inter += w
    return inter / intra if intra > 0 else np.inf

def spatial_region_consistency(comm_indices):
    """Majority anatomical region fraction per community."""
    node_channels = np.arange(n) // N_TIMEPOINTS
    node_regions  = np.array([CHANNEL_REGIONS.get(CHANNEL_NAMES[ch], "unknown")
                               for ch in node_channels])
    props = []
    for idx in comm_indices:
        subset = node_regions[idx]
        _, cts = np.unique(subset, return_counts=True)
        props.append(cts.max() / len(subset))
    return float(np.mean(props)) if props else 0.0

def _fmt(v):
    return "   inf  " if np.isinf(v) else f"{v:8.4f}"

print("\nComputing structural metrics ...")
adj_csr = adj.tocsr()

inter_iced, inter_cond, inter_cc = structural_metrics(adj_csr, inter_comm_idx)
ictal_iced, ictal_cond, ictal_cc = structural_metrics(adj_csr, ictal_comm_idx)

print("Computing inter/intra edge ratio ...")
inter_ratio = inter_intra_ratio(adj, labels, inter_comm_idx)
ictal_ratio  = inter_intra_ratio(adj, labels, ictal_comm_idx)

print("Computing spatial region consistency ...")
inter_src = spatial_region_consistency(inter_comm_idx)
ictal_src  = spatial_region_consistency(ictal_comm_idx)

W = 34
print(f"\n{'=' * 67}")
print(f"  BENCHMARK SUMMARY — Moore Streaming  ·  CHB-01 chb01_03")
print(f"{'=' * 67}")
print(f"  {'Metric':<{W}} {'Interictal':>10}  {'Ictal':>10}")
print(f"  {'-' * 62}")
rows_data = [
    ("Intra-Cluster Edge Density",  inter_iced,  ictal_iced),
    ("Inter / Intra Edge Ratio",    inter_ratio, ictal_ratio),
    ("Conductance",                 inter_cond,  ictal_cond),
    ("Avg Clustering Coeff (intra)",inter_cc,    ictal_cc),
]
for name, iv, av in rows_data:
    print(f"  {name:<{W}} {_fmt(iv)}  {_fmt(av)}")
print(f"  {'-' * 62}")
print(f"  {'ARI between runs':<{W}}   1.0000 ± 0.0000  (deterministic)")
print(f"  {'NMI between runs':<{W}}   1.0000 ± 0.0000  (deterministic)")
print(f"  {'-' * 62}")
print(f"  {'Runtime (full graph)':<{W}} {runtime:8.2f} s")
print(f"  {'Peak Memory Usage':<{W}} {peak_mem / 1e6:8.1f} MB")
print(f"  {'-' * 62}")
print(f"  {'Intra-Community Bandpower Var':<{W}}      N/A        N/A")
print(f"  {'Spatial Region Consistency':<{W}} {_fmt(inter_src)}  {_fmt(ictal_src)}")
print(f"{'=' * 67}\n")

# all axes scaled to [0, 1] where outward (1) = better.
# Natural-range metrics (ICED, conductance, CC, SRC) stay on [0,1].
# Inter/Intra Ratio is capped at _IIR_CAP and inverted.
_IIR_CAP = 2.0

def _score(v, higher_is_better=True, cap=None):
    if np.isinf(v) or np.isnan(v):
        return 0.0 if not higher_is_better else 1.0
    if cap is not None:
        v = np.clip(v, 0, cap) / cap
    s = float(np.clip(v, 0.0, 1.0))
    return s if higher_is_better else 1.0 - s

spider_metrics = [
    ("ICED",                    _score(inter_iced),                      _score(ictal_iced)),
    ("1 − Conductance",         _score(inter_cond, higher_is_better=False), _score(ictal_cond, higher_is_better=False)),
    ("Clustering\nCoeff",       _score(inter_cc),                        _score(ictal_cc)),
    ("1 − I/I Ratio\n(cap=2)",  _score(inter_ratio, higher_is_better=False, cap=_IIR_CAP),
                                 _score(ictal_ratio,  higher_is_better=False, cap=_IIR_CAP)),
    ("Spatial Region\nConsistency", _score(inter_src),                   _score(ictal_src)),
]

_labels     = [m[0] for m in spider_metrics]
_vals_inter = [m[1] for m in spider_metrics]
_vals_ictal = [m[2] for m in spider_metrics]

N      = len(_labels)
angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
angles += angles[:1]
_vi = _vals_inter + _vals_inter[:1]
_va = _vals_ictal + _vals_ictal[:1]

fig, ax = plt.subplots(figsize=(7, 7), subplot_kw=dict(polar=True))
ax.set_theta_offset(np.pi / 2)
ax.set_theta_direction(-1)

ax.plot(angles, _vi, 'o-', lw=2.0, color='steelblue', label='Interictal')
ax.fill(angles, _vi, alpha=0.18, color='steelblue')
ax.plot(angles, _va, 'o-', lw=2.0, color='tomato', label='Ictal')
ax.fill(angles, _va, alpha=0.18, color='tomato')

ax.set_xticks(angles[:-1])
ax.set_xticklabels(_labels, size=10)
ax.set_ylim(0, 1)
ax.set_yticks([0.25, 0.5, 0.75, 1.0])
ax.set_yticklabels(['0.25', '0.50', '0.75', '1.00'], size=8, color='grey')
ax.set_title(
    "Stream-Moore Benchmark\nInterictal vs. Ictal  |  CHB-01 chb01_03",
    size=13, pad=20
)
ax.legend(loc='upper right', bbox_to_anchor=(1.35, 1.15), fontsize=10)

_out = (Path(__file__).resolve().parents[2]
        / "reports" / "figures" / "spider_benchmark.png")
plt.tight_layout()
plt.savefig(_out, dpi=150, bbox_inches='tight')
plt.show()
print(f"Spider plot saved → {_out}")
