"""
Spectral Clustering from Scratch — Graph Laplacian on EEG Adjacency Matrix
===========================================================================
Dataset : CHB-MIT · chb01_03 · 2980-3010 s
Onset   : +16 s into file (sample 4096)
 
No sklearn. No scipy clustering. Pure numpy throughout.
 
Pipeline:
  1. Load sparse adjacency matrix
  2. Extract 23x23 channel correlation matrix per sliding window
  3. Threshold correlation -> weighted adjacency graph
  4. Compute normalized graph Laplacian from scratch
  5. Eigendecompose Laplacian -> embed channels in eigenspace
  6. Assign clusters via simple k-means on eigenvectors (also from scratch)
  7. Track how cluster assignments change across the seizure transition
  8. Plot results
 
Run with:  python3 eeg_spectral_clustering.py
"""
 
import warnings
from pathlib import Path
import numpy as np
import scipy.sparse as sp
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

_HERE = Path(__file__).resolve().parent
BASE_DIR = _HERE.parents[1]
FIGURES_DIR = BASE_DIR / "reports" / "figures"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

# configuration
NPZ_PATH = (
    BASE_DIR
    / "data" / "graphs" / "adjacency_sparse"
    / "inter_to_ict_chb01_03_2980_3010_adjacency_sparse.npz"
)
N_CHANNELS   = 23
N_TIMEPOINTS = 7680
FS           = 256
WINDOW_SEC   = 5
STEP_SEC     = 1
ONSET_SEC    = 16
K_CLUSTERS   = 4      # number of clusters for spectral clustering
THRESHOLD    = 0.3    # correlation threshold to build the graph
                      # edges with r < threshold are set to 0
 
CHANNEL_NAMES = [
    "FP1-F7", "F7-T7",  "T7-P7",  "P7-O1",
    "FP1-F3", "F3-C3",  "C3-P3",  "P3-O1",
    "FP2-F4", "F4-C4",  "C4-P4",  "P4-O2",
    "FP2-F8", "F8-T8",  "T8-P8",  "P8-O2",
    "FZ-CZ",  "CZ-PZ",
    "P7-T7",  "T7-FT9", "FT9-FT10", "FT10-T8",
    "T8-P8-1"
]
 
BG             = "#0f1117"
PANEL_BG       = "#1a1d27"
TEXT           = "#e0e0e0"
ONSET_COLOR    = "#ff4f4f"
INTERICTAL_CLR = "#7c6af7"
ICTAL_CLR      = "#f7936a"
CLUSTER_COLORS = ["#7c6af7", "#f7936a", "#5ecfb1", "#e05c97", "#f5d547"]
 
 
print("Loading adjacency matrix ...")
mat = sp.load_npz(NPZ_PATH)
print(f"  Shape: {mat.shape}  |  Non-zeros: {mat.nnz:,}")
 
window_samples = WINDOW_SEC * FS
step_samples   = STEP_SEC * FS
t_starts       = list(range(0, N_TIMEPOINTS - window_samples + 1, step_samples))
t_centers_sec  = [(t + window_samples / 2) / FS for t in t_starts]
n_windows      = len(t_starts)
 
print(f"Precomputing {n_windows} correlation matrices ...")
all_corrs     = []
all_temporals = []
for t_start in t_starts:
    ch_temporal = np.zeros((N_CHANNELS, window_samples))
    for ch in range(N_CHANNELS):
        block = mat[
            ch * N_TIMEPOINTS + t_start : ch * N_TIMEPOINTS + t_start + window_samples,
            ch * N_TIMEPOINTS + t_start : ch * N_TIMEPOINTS + t_start + window_samples
        ]
        ch_temporal[ch] = np.array(block.sum(axis=1)).flatten()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        corr = np.corrcoef(ch_temporal)
    all_temporals.append(ch_temporal.copy())
    all_corrs.append(np.nan_to_num(corr, nan=0.0))

all_corrs     = np.array(all_corrs)      # (n_windows, 23, 23)
all_temporals = np.array(all_temporals)  # (n_windows, 23, window_samples)
print("Done.\n")
 
 
def build_adjacency(corr_matrix, threshold):
    A = corr_matrix.copy()
    A[A < threshold] = 0.0
    A[A < 0]         = 0.0
    np.fill_diagonal(A, 0.0)
    return A
 
 
def compute_normalized_laplacian(A):
    """L_sym = I - D^(-1/2) A D^(-1/2)"""
    N      = A.shape[0]
    degree = A.sum(axis=1)

    d_inv_sqrt = np.zeros(N)
    nonzero    = degree > 0
    d_inv_sqrt[nonzero] = 1.0 / np.sqrt(degree[nonzero])

    D_inv_sqrt = np.diag(d_inv_sqrt)
    return np.eye(N) - D_inv_sqrt @ A @ D_inv_sqrt
 
 
def spectral_embed(L, k):
    """Returns k smallest eigenvectors of L; smallest eigenvalue is always 0."""
    eigenvalues, eigenvectors = np.linalg.eigh(L)
    return eigenvectors[:, :k], eigenvalues
 
 
def normalize_rows(X):
    norms = np.linalg.norm(X, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return X / norms
 
 
def kmeans_scratch(X, k, n_init=10, max_iter=300, tol=1e-6, seed=42):
    rng          = np.random.default_rng(seed)
    best_labels  = None
    best_inertia = np.inf
 
    for init_run in range(n_init):
 
        # initialise centroids (k-means++ style)
        centroids = []
        first_idx = rng.integers(0, len(X))
        centroids.append(X[first_idx].copy())
 
        for _ in range(k - 1):
            dists = np.array([
                min(np.sum((x - c) ** 2) for c in centroids)
                for x in X
            ])
            probs    = dists / dists.sum()
            next_idx = rng.choice(len(X), p=probs)
            centroids.append(X[next_idx].copy())
 
        centroids = np.array(centroids)   # (k, d)
 
        labels = np.zeros(len(X), dtype=int)

        for iteration in range(max_iter):
            dists_all = np.array([
                np.sum((X - c) ** 2, axis=1) for c in centroids
            ])                                  # (k, N)
            new_labels = np.argmin(dists_all, axis=0)   # (N,)

            new_centroids = np.zeros_like(centroids)
            for cluster_id in range(k):
                mask = new_labels == cluster_id
                if mask.sum() > 0:
                    new_centroids[cluster_id] = X[mask].mean(axis=0)
                else:
                    # empty cluster — reinitialise to a random point
                    new_centroids[cluster_id] = X[rng.integers(0, len(X))]

            shift = np.linalg.norm(new_centroids - centroids)
            centroids = new_centroids
            labels    = new_labels

            if shift < tol:
                break

        inertia = sum(
            np.sum((X[labels == c] - centroids[c]) ** 2)
            for c in range(k)
            if (labels == c).sum() > 0
        )
 
        if inertia < best_inertia:
            best_inertia   = inertia
            best_labels    = labels.copy()
            best_centroids = centroids.copy()
 
    return best_labels, best_centroids, best_inertia
 
 
def spectral_clustering(corr_matrix, k, threshold):
    """
    Full spectral clustering pipeline for one correlation matrix.
 
    Steps:
      1. Build adjacency graph (threshold weak correlations)
      2. Compute normalized Laplacian
      3. Embed channels in k-dimensional eigenspace
      4. Row-normalize the embedding
      5. Run k-means from scratch on the embedding
 
    Parameters
    ----------
    corr_matrix : (N, N) correlation matrix
    k           : int, number of clusters
    threshold   : float, minimum correlation to keep edge
 
    Returns
    -------
    labels      : (N,) cluster assignment per channel
    eigenvalues : (N,) Laplacian eigenvalues (for eigengap inspection)
    embedding   : (N, k) spectral embedding
    """
    A          = build_adjacency(corr_matrix, threshold)
    L          = compute_normalized_laplacian(A)
    embedding, eigenvalues = spectral_embed(L, k)
    embedding  = normalize_rows(embedding)
    labels, _, _ = kmeans_scratch(embedding, k)
    return labels, eigenvalues, embedding
 
 
# STEP 2 — run spectral clustering on every window
print(f"Running spectral clustering (k={K_CLUSTERS}, threshold={THRESHOLD}) ...")
all_labels      = []   # cluster assignments per window
all_eigenvalues = []   # eigenvalue spectra per window
 
for i, corr in enumerate(all_corrs):
    labels, eigenvalues, _ = spectral_clustering(corr, K_CLUSTERS, THRESHOLD)
    all_labels.append(labels)
    all_eigenvalues.append(eigenvalues)
    t_s = t_starts[i] / FS
    print(f"  window {t_s:4.0f}s  clusters: {[int((labels==c).sum()) for c in range(K_CLUSTERS)]}")
 
all_labels      = np.array(all_labels)       # (n_windows, N_CHANNELS)
all_eigenvalues = np.array(all_eigenvalues)  # (n_windows, N_CHANNELS)
print("Done.\n")
 
 
# STEP 3 — align cluster labels across windows
# (k-means labels are arbitrary per run — align greedily to window 0)
def align_labels(ref_labels, new_labels, k):
    """
    Permute new_labels so they best match ref_labels.
    Uses greedy maximum overlap matching.
    """
    perm = {}
    used = set()
    for ref_c in range(k):
        ref_mask = ref_labels == ref_c
        best_overlap = -1
        best_new_c   = -1
        for new_c in range(k):
            if new_c in used:
                continue
            overlap = (new_labels[ref_mask] == new_c).sum()
            if overlap > best_overlap:
                best_overlap = overlap
                best_new_c   = new_c
        perm[best_new_c] = ref_c
        used.add(best_new_c)
    aligned = np.array([perm.get(l, l) for l in new_labels])
    return aligned
 
aligned_labels = [all_labels[0].copy()]
for i in range(1, n_windows):
    aligned = align_labels(aligned_labels[0], all_labels[i], K_CLUSTERS)
    aligned_labels.append(aligned)
aligned_labels = np.array(aligned_labels)
 
 
# PLOT 1 — cluster assignments over time (raster / carpet plot)
print("Saving plot 1: cluster assignment raster ...")
fig, ax = plt.subplots(figsize=(13, 5), facecolor=BG)
ax.set_facecolor(PANEL_BG)
for spine in ax.spines.values():
    spine.set_visible(False)
 
cmap_discrete = plt.cm.colors if hasattr(plt.cm, 'colors') else None
colors_4 = CLUSTER_COLORS[:K_CLUSTERS]
 
from matplotlib.colors import ListedColormap
cmap_disc = ListedColormap(colors_4)
 
img = ax.imshow(
    aligned_labels.T,           # (N_CHANNELS, n_windows)
    aspect="auto",
    cmap=cmap_disc,
    vmin=0, vmax=K_CLUSTERS - 1,
    interpolation="nearest"
)
 
# onset line
onset_frame = next(i for i, t in enumerate(t_centers_sec) if t > ONSET_SEC) - 0.5
ax.axvline(onset_frame, color=ONSET_COLOR, linewidth=2,
           linestyle="--", label="Seizure onset")
 
ax.set_yticks(range(N_CHANNELS))
ax.set_yticklabels(CHANNEL_NAMES, fontsize=7, color=TEXT)
ax.set_xticks(range(n_windows))
ax.set_xticklabels([f"{t:.0f}s" for t in t_centers_sec],
                   rotation=90, fontsize=7, color=TEXT)
ax.tick_params(colors=TEXT)
ax.set_xlabel("Window centre", color=TEXT, fontsize=10)
ax.set_title(
    f"Spectral Cluster Assignment per Channel over Time  .  k={K_CLUSTERS}  .  CHB-01 chb01_03",
    color=TEXT, fontsize=11, fontweight="bold"
)
 
from matplotlib.patches import Patch
legend_elements = [Patch(facecolor=colors_4[c], label=f"Cluster {c+1}")
                   for c in range(K_CLUSTERS)]
legend_elements.append(plt.Line2D([0], [0], color=ONSET_COLOR,
                                  linestyle="--", label="Seizure onset"))
ax.legend(handles=legend_elements, facecolor=PANEL_BG, edgecolor="#444",
          labelcolor=TEXT, fontsize=8, loc="upper left")
 
plt.tight_layout()
_out = FIGURES_DIR / "spectral_plot1_raster.png"
if not _out.exists():
    fig.savefig(_out, dpi=150, bbox_inches="tight", facecolor=BG)
    print(f"  Saved -> {_out.name}")
else:
    print(f"  Skipped (already exists) -> {_out.name}")
plt.close(fig)
 
 
# PLOT 2 — eigengap over time
# The eigengap (gap between k-th and (k+1)-th eigenvalue) tells you
# how well-separated the clusters are. A large gap = strong cluster structure.
print("Saving plot 2: eigengap over time ...")
eigengap = all_eigenvalues[:, K_CLUSTERS] - all_eigenvalues[:, K_CLUSTERS - 1]
 
fig, ax = plt.subplots(figsize=(11, 3.5), facecolor=BG)
ax.set_facecolor(PANEL_BG)
for spine in ax.spines.values():
    spine.set_edgecolor("#333")
 
ax.axvspan(0, ONSET_SEC, alpha=0.08, color=INTERICTAL_CLR, label="Interictal")
ax.axvspan(ONSET_SEC, t_centers_sec[-1] + WINDOW_SEC / 2,
           alpha=0.08, color=ICTAL_CLR, label="Ictal")
ax.axvline(ONSET_SEC, color=ONSET_COLOR, linewidth=2,
           linestyle="--", label="Seizure onset")
 
ax.plot(t_centers_sec, eigengap, color="white", linewidth=2, zorder=3)
ax.fill_between(t_centers_sec, eigengap, alpha=0.25, color="white")
 
ax.set_xlabel("Window centre (s)", color=TEXT, fontsize=10)
ax.set_ylabel(f"Eigengap (λ{K_CLUSTERS+1} - λ{K_CLUSTERS})", color=TEXT, fontsize=10)
ax.set_title(
    f"Laplacian Eigengap over Time  .  larger = stronger cluster structure",
    color=TEXT, fontsize=11, fontweight="bold"
)
ax.tick_params(colors=TEXT)
ax.legend(facecolor=PANEL_BG, edgecolor="#444", labelcolor=TEXT, fontsize=9)
 
plt.tight_layout()
_out = FIGURES_DIR / "spectral_plot2_eigengap.png"
if not _out.exists():
    fig.savefig(_out, dpi=150, bbox_inches="tight", facecolor=BG)
    print(f"  Saved -> {_out.name}")
else:
    print(f"  Skipped (already exists) -> {_out.name}")
plt.close(fig)
 
 
# PLOT 3 — interictal vs ictal cluster maps (which channels in which cluster)
print("Saving plot 3: interictal vs ictal cluster comparison ...")
 
interictal_idx  = [i for i, t in enumerate(t_centers_sec) if t <= ONSET_SEC]
ictal_idx       = [i for i, t in enumerate(t_centers_sec) if t >  ONSET_SEC]
 
# majority vote cluster per channel in each phase
def majority_cluster(labels_subset):
    """For each channel, return the most common cluster across windows."""
    result = np.zeros(N_CHANNELS, dtype=int)
    for ch in range(N_CHANNELS):
        counts = np.bincount(labels_subset[:, ch], minlength=K_CLUSTERS)
        result[ch] = np.argmax(counts)
    return result
 
inter_majority = majority_cluster(aligned_labels[interictal_idx])
ictal_majority = majority_cluster(aligned_labels[ictal_idx])
 
fig, axes = plt.subplots(1, 2, figsize=(14, 5), facecolor=BG)
fig.suptitle(
    "Dominant Cluster per Channel  .  Interictal vs Ictal  .  CHB-01 chb01_03",
    color=TEXT, fontsize=12, fontweight="bold", y=1.01
)
 
for ax, majority, title, accent in zip(
    axes,
    [inter_majority, ictal_majority],
    ["Interictal  (0-16 s)", "Ictal  (16-30 s)"],
    [INTERICTAL_CLR, ICTAL_CLR]
):
    ax.set_facecolor(PANEL_BG)
    for spine in ax.spines.values():
        spine.set_visible(False)
 
    # horizontal bar, one per channel, coloured by cluster
    for ch_idx, (ch_name, cluster) in enumerate(zip(CHANNEL_NAMES, majority)):
        ax.barh(ch_idx, 1, color=colors_4[cluster], alpha=0.85, height=0.8)
        ax.text(1.05, ch_idx, f"cluster {cluster + 1}",
                va="center", fontsize=7, color=colors_4[cluster])
 
    ax.set_yticks(range(N_CHANNELS))
    ax.set_yticklabels(CHANNEL_NAMES, fontsize=7.5, color=TEXT)
    ax.set_xlim(0, 1.5)
    ax.set_xticks([])
    ax.set_title(title, color=accent, fontsize=11, fontweight="bold", pad=8)
    ax.tick_params(colors=TEXT)
 
plt.tight_layout()
_out = FIGURES_DIR / "spectral_plot3_cluster_map.png"
if not _out.exists():
    fig.savefig(_out, dpi=150, bbox_inches="tight", facecolor=BG)
    print(f"  Saved -> {_out.name}")
else:
    print(f"  Skipped (already exists) -> {_out.name}")
plt.close(fig)
 
 
# PLOT 4 — eigenvalue spectrum for one interictal vs one ictal window
print("Saving plot 4: eigenvalue spectrum comparison ...")
 
mid_inter = interictal_idx[len(interictal_idx) // 2]
mid_ictal = ictal_idx[len(ictal_idx) // 2]
 
fig, axes = plt.subplots(1, 2, figsize=(12, 4), facecolor=BG)
fig.suptitle(
    "Laplacian Eigenvalue Spectrum  .  Interictal vs Ictal",
    color=TEXT, fontsize=12, fontweight="bold", y=1.01
)
 
for ax, win_idx, title, accent in zip(
    axes,
    [mid_inter, mid_ictal],
    [f"Interictal window  ({t_centers_sec[mid_inter]:.0f}s)",
     f"Ictal window  ({t_centers_sec[mid_ictal]:.0f}s)"],
    [INTERICTAL_CLR, ICTAL_CLR]
):
    ax.set_facecolor(PANEL_BG)
    for spine in ax.spines.values():
        spine.set_edgecolor("#333")
 
    evals = all_eigenvalues[win_idx]
    ax.bar(range(len(evals)), evals, color=accent, alpha=0.7, width=0.7)
 
    # mark the k-th eigengap
    ax.axvline(K_CLUSTERS - 0.5, color=ONSET_COLOR, linewidth=1.5,
               linestyle="--", label=f"k={K_CLUSTERS} cut")
 
    ax.set_xlabel("Eigenvalue index", color=TEXT, fontsize=9)
    ax.set_ylabel("Eigenvalue", color=TEXT, fontsize=9)
    ax.set_title(title, color=accent, fontsize=10, fontweight="bold")
    ax.tick_params(colors=TEXT, labelsize=8)
    ax.legend(facecolor=PANEL_BG, edgecolor="#444", labelcolor=TEXT, fontsize=8)
 
plt.tight_layout()
_out = FIGURES_DIR / "spectral_plot4_eigenspectrum.png"
if not _out.exists():
    fig.savefig(_out, dpi=150, bbox_inches="tight", facecolor=BG)
    print(f"  Saved -> {_out.name}")
else:
    print(f"  Skipped (already exists) -> {_out.name}")
plt.close(fig)

print(f"\nAll done. Figures in: {FIGURES_DIR}")
print("  spectral_plot1_raster.png        — cluster assignments over time")
print("  spectral_plot2_eigengap.png      — cluster separation strength over time")
print("  spectral_plot3_cluster_map.png   — interictal vs ictal cluster membership")
print("  spectral_plot4_eigenspectrum.png — eigenvalue spectrum comparison")
print("  spectral_plot4_eigenspectrum.png — Laplacian eigenvalue spectrum")

# BENCHMARK METRICS  (all from scratch — numpy + stdlib only)
import time, tracemalloc

CHANNEL_REGIONS = {
    "FP1-F7":    "frontal",   "F7-T7":    "frontal",
    "T7-P7":     "temporal",  "P7-O1":    "parietal",
    "FP1-F3":    "frontal",   "F3-C3":    "central",
    "C3-P3":     "central",   "P3-O1":    "parietal",
    "FP2-F4":    "frontal",   "F4-C4":    "central",
    "C4-P4":     "central",   "P4-O2":    "parietal",
    "FP2-F8":    "frontal",   "F8-T8":    "frontal",
    "T8-P8":     "temporal",  "P8-O2":    "parietal",
    "FZ-CZ":     "central",   "CZ-PZ":    "central",
    "P7-T7":     "temporal",  "T7-FT9":   "temporal",
    "FT9-FT10":  "temporal",  "FT10-T8":  "temporal",
    "T8-P8-1":   "temporal",
}

def _iced(A, labels):
    """Intra-cluster edge density: actual edges / possible edges within each cluster."""
    densities = []
    for c in np.unique(labels):
        idx = np.where(labels == c)[0]
        n_c = len(idx)
        if n_c < 2:
            continue
        sub      = A[np.ix_(idx, idx)]
        actual   = float(np.triu(sub, 1).sum())
        possible = n_c * (n_c - 1) / 2
        densities.append(actual / possible)
    return float(np.mean(densities)) if densities else 0.0

def _ratio(A, labels):
    """Inter / intra edge ratio."""
    intra, inter = 0.0, 0.0
    n = len(labels)
    for i in range(n):
        for j in range(i + 1, n):
            w = A[i, j]
            if w > 0:
                if labels[i] == labels[j]:
                    intra += w
                else:
                    inter += w
    return inter / intra if intra > 0 else np.inf

def _cond(A, labels):
    """Mean conductance: cut / vol per cluster."""
    vals = []
    for c in np.unique(labels):
        rows = np.where(labels == c)[0]
        cols = np.where(labels != c)[0]
        cut  = float(A[np.ix_(rows, cols)].sum())
        vol  = float(A[rows, :].sum())
        vals.append(cut / vol if vol > 0 else 0.0)
    return float(np.mean(vals))

def _cc(A, labels):
    """Mean local clustering coefficient within each cluster."""
    coeffs = []
    for c in np.unique(labels):
        idx = np.where(labels == c)[0]
        if len(idx) < 3:
            continue
        sub = (A[np.ix_(idx, idx)] > 0).astype(float)
        np.fill_diagonal(sub, 0.0)
        for i in range(len(idx)):
            nbrs = np.where(sub[i] > 0)[0]
            ki   = len(nbrs)
            if ki < 2:
                continue
            tri = sum(sub[nbrs[u], nbrs[v]]
                      for u in range(ki) for v in range(u + 1, ki))
            coeffs.append(float(tri) / (ki * (ki - 1) / 2))
    return float(np.mean(coeffs)) if coeffs else 0.0

def _c2(n):
    return n * (n - 1) // 2

def _ari(a, b):
    """Adjusted Rand Index from contingency table."""
    a, b = np.asarray(a), np.asarray(b)
    ca, cb = np.unique(a), np.unique(b)
    ma = {v: i for i, v in enumerate(ca)}
    mb = {v: i for i, v in enumerate(cb)}
    C  = np.zeros((len(ca), len(cb)), dtype=np.int64)
    for ai, bi in zip(a, b):
        C[ma[ai], mb[bi]] += 1
    sc  = sum(_c2(int(n)) for n in C.flatten())
    sa  = sum(_c2(int(n)) for n in C.sum(axis=1))
    sb  = sum(_c2(int(n)) for n in C.sum(axis=0))
    tot = _c2(len(a))
    exp = sa * sb / tot if tot > 0 else 0
    mx  = (sa + sb) / 2
    return float((sc - exp) / (mx - exp)) if (mx - exp) > 0 else 1.0

def _nmi(a, b):
    """Normalized Mutual Information."""
    n  = len(a)
    ca, cb = np.unique(a), np.unique(b)
    ma = {v: i for i, v in enumerate(ca)}
    mb = {v: i for i, v in enumerate(cb)}
    P  = np.zeros((len(ca), len(cb)))
    for ai, bi in zip(a, b):
        P[ma[ai], mb[bi]] += 1
    P  /= n
    pa, pb = P.sum(axis=1), P.sum(axis=0)
    mi = sum(P[i, j] * np.log(P[i, j] / (pa[i] * pb[j]))
             for i in range(len(ca)) for j in range(len(cb))
             if P[i, j] > 0 and pa[i] > 0 and pb[j] > 0)
    ha = -sum(p * np.log(p) for p in pa if p > 0)
    hb = -sum(p * np.log(p) for p in pb if p > 0)
    return float(2 * mi / (ha + hb)) if (ha + hb) > 0 else 1.0

def _bandpower(sig, fs, lo=1.0, hi=40.0):
    fft   = np.fft.rfft(sig)
    freqs = np.fft.rfftfreq(len(sig), d=1.0 / fs)
    psd   = (np.abs(fft) ** 2) / len(sig)
    idx   = (freqs >= lo) & (freqs <= hi)
    return float(psd[idx].mean()) if idx.sum() > 0 else 0.0

def _bpv(ch_temporal, labels, fs):
    """Intra-community bandpower variance."""
    bp = np.array([_bandpower(ch_temporal[ch], fs) for ch in range(len(ch_temporal))])
    vars_ = [float(bp[np.where(labels == c)[0]].var())
             for c in np.unique(labels) if (labels == c).sum() >= 2]
    return float(np.mean(vars_)) if vars_ else 0.0

def _src(labels, ch_names, regions):
    """Spatial region consistency: majority region fraction per cluster."""
    regs  = np.array([regions.get(ch, "unknown") for ch in ch_names])
    props = []
    for c in np.unique(labels):
        idx = np.where(labels == c)[0]
        if len(idx) == 0:
            continue
        _, cts = np.unique(regs[idx], return_counts=True)
        props.append(cts.max() / len(idx))
    return float(np.mean(props)) if props else 0.0

def _fmt(v):
    return "   inf  " if np.isinf(v) else f"{v:8.4f}"

def _pavg(lst, idx):
    return float(np.mean([lst[i] for i in idx]))

print("\nComputing benchmark metrics ...")

# structural quality per window
iced_l, ratio_l, cond_l, cc_l, bpv_l, src_l = [], [], [], [], [], []
for i, (corr, labs) in enumerate(zip(all_corrs, aligned_labels)):
    A_b = build_adjacency(corr, THRESHOLD)
    iced_l.append(_iced(A_b, labs))
    ratio_l.append(_ratio(A_b, labs))
    cond_l.append(_cond(A_b, labs))
    cc_l.append(_cc(A_b, labs))
    bpv_l.append(_bpv(all_temporals[i], labs, FS))
    src_l.append(_src(labs, CHANNEL_NAMES, CHANNEL_REGIONS))

# stability: 5 independent runs on a single interictal window
test_win = interictal_idx[len(interictal_idx) // 2]
run_labs = [spectral_clustering(all_corrs[test_win], K_CLUSTERS, THRESHOLD)[0]
            for _ in range(5)]
ari_vals = [_ari(run_labs[0], r) for r in run_labs[1:]]
nmi_vals = [_nmi(run_labs[0], r) for r in run_labs[1:]]

# runtime & peak memory
tracemalloc.start()
t0 = time.perf_counter()
for corr in all_corrs:
    spectral_clustering(corr, K_CLUSTERS, THRESHOLD)
t1 = time.perf_counter()
_, peak_mem = tracemalloc.get_traced_memory()
tracemalloc.stop()

# print summary
W = 34
print(f"\n{'=' * 67}")
print(f"  BENCHMARK SUMMARY — Spectral Laplacian  ·  CHB-01 chb01_03")
print(f"{'=' * 67}")
print(f"  {'Metric':<{W}} {'Interictal':>10}  {'Ictal':>10}")
print(f"  {'-' * 62}")
for name, lst in [("Intra-Cluster Edge Density",   iced_l),
                   ("Inter / Intra Edge Ratio",      ratio_l),
                   ("Conductance",                   cond_l),
                   ("Avg Clustering Coeff (intra)",  cc_l)]:
    print(f"  {name:<{W}} {_fmt(_pavg(lst, interictal_idx))}  {_fmt(_pavg(lst, ictal_idx))}")
print(f"  {'-' * 62}")
print(f"  {'ARI between runs':<{W}} {np.mean(ari_vals):8.4f} ± {np.std(ari_vals):.4f}")
print(f"  {'NMI between runs':<{W}} {np.mean(nmi_vals):8.4f} ± {np.std(nmi_vals):.4f}")
print(f"  {'-' * 62}")
print(f"  {'Runtime (all windows)':<{W}} {t1 - t0:8.2f} s")
print(f"  {'Peak Memory Usage':<{W}} {peak_mem / 1e6:8.1f} MB")
print(f"  {'-' * 62}")
for name, lst in [("Intra-Community Bandpower Var",  bpv_l),
                   ("Spatial Region Consistency",     src_l)]:
    print(f"  {name:<{W}} {_fmt(_pavg(lst, interictal_idx))}  {_fmt(_pavg(lst, ictal_idx))}")
print(f"{'=' * 67}\n")
 