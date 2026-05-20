import warnings
from pathlib import Path

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import scipy.cluster.hierarchy as sch
import scipy.sparse as sp
from matplotlib.colors import ListedColormap
from matplotlib.patches import Rectangle
from scipy.spatial.distance import squareform

_HERE    = Path(__file__).resolve().parent
BASE_DIR = _HERE.parents[1]
NPZ_PATH = (
    BASE_DIR / "data" / "graphs" / "adjacency_sparse"
    / "inter_to_ict_chb01_03_2980_3010_adjacency_sparse.npz"
)
FIG_DIR = BASE_DIR / "reports" / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

N_CH       = 23
N_TP       = 7680
FS         = 256
WINDOW_SEC = 5
STEP_SEC   = 1
ONSET_SEC  = 16
ONSET      = int(ONSET_SEC * FS)
K          = 4
THRESHOLD  = 0.3

CHANNEL_NAMES = [
    "FP1-F7", "F7-T7",  "T7-P7",  "P7-O1",
    "FP1-F3", "F3-C3",  "C3-P3",  "P3-O1",
    "FP2-F4", "F4-C4",  "C4-P4",  "P4-O2",
    "FP2-F8", "F8-T8",  "T8-P8",  "P8-O2",
    "FZ-CZ",  "CZ-PZ",
    "P7-T7",  "T7-FT9", "FT9-FT10", "FT10-T8",
    "T8-P8-1",
]

COLORS      = ["#B39DDB", "#81C3E8", "#88CFA4", "#F5D87A"]
ONSET_COLOR = "#C0392B"
INTER_COLOR = "#2980B9"
ICTAL_COLOR = "#E67E22"

plt.rcParams.update({
    "figure.facecolor":  "white",
    "axes.facecolor":    "white",
    "axes.edgecolor":    "#333333",
    "axes.linewidth":    0.8,
    "axes.grid":         False,
    "font.family":       "sans-serif",
    "font.size":         10,
    "axes.titlesize":    13,
    "axes.labelsize":    11,
    "xtick.labelsize":   9,
    "ytick.labelsize":   9,
    "legend.fontsize":   9,
    "legend.framealpha": 0.9,
    "legend.edgecolor":  "#cccccc",
    "savefig.dpi":       300,
    "savefig.bbox":      "tight",
})


def channel_corr(t_start, t_end):
    ch_tmp = np.zeros((N_CH, t_end - t_start), dtype=np.float64)
    for ch in range(N_CH):
        s = ch * N_TP + t_start
        e = ch * N_TP + t_end
        ch_tmp[ch] = np.array(mat[s:e, s:e].sum(axis=1)).flatten()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        c = np.corrcoef(ch_tmp)
    return np.nan_to_num(c, nan=0.0)


def ward(c):
    dist   = squareform(np.clip(1.0 - c, 0.0, None), checks=False)
    lnk    = sch.linkage(dist, method="ward")
    flat_c = sch.fcluster(lnk, t=K, criterion="maxclust") - 1
    ord_c  = sch.dendrogram(lnk, no_plot=True)["leaves"]
    return lnk, flat_c, ord_c


def spectral_one_window(corr_w):
    A = corr_w.copy()
    A[A < THRESHOLD] = 0.0
    A[A < 0]         = 0.0
    np.fill_diagonal(A, 0.0)
    deg        = A.sum(axis=1)
    d_inv_sqrt = np.where(deg > 0, 1.0 / np.sqrt(np.maximum(deg, 1e-10)), 0.0)
    L          = np.eye(N_CH) - np.diag(d_inv_sqrt) @ A @ np.diag(d_inv_sqrt)
    evals, vecs = np.linalg.eigh(L)
    emb        = vecs[:, :K].copy()
    norms      = np.linalg.norm(emb, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    emb /= norms

    rng = np.random.default_rng(42)
    best_labels, best_inertia = None, np.inf
    for _ in range(10):
        centers = emb[rng.choice(N_CH, K, replace=False)]
        for _ in range(200):
            dists  = np.linalg.norm(emb[:, None, :] - centers[None, :, :], axis=2)
            assign = np.argmin(dists, axis=1)
            new_c  = np.array([
                emb[assign == k].mean(0) if (assign == k).any()
                else emb[rng.integers(N_CH)]
                for k in range(K)
            ])
            if np.allclose(new_c, centers):
                break
            centers = new_c
        inertia = sum(np.sum((emb[assign == k] - centers[k])**2)
                      for k in range(K) if (assign == k).any())
        if inertia < best_inertia:
            best_inertia, best_labels = inertia, assign.copy()
    return best_labels, evals


def align_labels(ref, new):
    perm, used = {}, set()
    for rc in range(K):
        mask = ref == rc
        best_k, best_ov = -1, -1
        for nc in range(K):
            if nc in used:
                continue
            ov = (new[mask] == nc).sum()
            if ov > best_ov:
                best_ov, best_k = ov, nc
        perm[best_k] = rc
        used.add(best_k)
    return np.array([perm.get(l, l) for l in new])


def majority_cluster(subset):
    result = np.zeros(N_CH, dtype=int)
    for ch in range(N_CH):
        counts = np.bincount(subset[:, ch], minlength=K)
        result[ch] = np.argmax(counts)
    return result


def phase_spans(ax, t0, t1):
    ax.axvspan(t0, ONSET_SEC, alpha=0.08, color=INTER_COLOR, zorder=0)
    ax.axvspan(ONSET_SEC, t1, alpha=0.08, color=ICTAL_COLOR,  zorder=0)
    ax.axvline(ONSET_SEC, color=ONSET_COLOR, linewidth=2.0, linestyle="--", zorder=4)


print("Loading data ...")
mat = sp.load_npz(NPZ_PATH).tocsr()

corr_full  = channel_corr(0,     N_TP)
corr_inter = channel_corr(0,     ONSET)
corr_ictal = channel_corr(ONSET, N_TP)

linkage,       flat,       order       = ward(corr_full)
linkage_inter, flat_inter, order_inter = ward(corr_inter)
linkage_ictal, flat_ictal, order_ictal = ward(corr_ictal)

win_samples  = WINDOW_SEC * FS
step_samples = STEP_SEC * FS
t_starts     = list(range(0, N_TP - win_samples + 1, step_samples))
t_centers    = [(t + win_samples / 2) / FS for t in t_starts]
n_windows    = len(t_starts)

print("Running spectral clustering ...")
all_labels, all_eigenvalues, mean_corr = [], [], []
triu_idx = np.triu_indices(N_CH, k=1)

for t_start in t_starts:
    ch_win = np.zeros((N_CH, win_samples), dtype=np.float64)
    for ch in range(N_CH):
        block = mat[
            ch * N_TP + t_start : ch * N_TP + t_start + win_samples,
            ch * N_TP + t_start : ch * N_TP + t_start + win_samples
        ]
        ch_win[ch] = np.array(block.sum(axis=1)).flatten()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        corr_w = np.corrcoef(ch_win)
    corr_w = np.nan_to_num(corr_w, nan=0.0)
    mean_corr.append(corr_w[triu_idx].mean())
    labels, evals = spectral_one_window(corr_w)
    all_labels.append(labels)
    all_eigenvalues.append(evals)

all_labels      = np.array(all_labels)
all_eigenvalues = np.array(all_eigenvalues)

aligned = [all_labels[0].copy()]
for i in range(1, n_windows):
    aligned.append(align_labels(aligned[0], all_labels[i]))
aligned = np.array(aligned)

inter_idx = [i for i, t in enumerate(t_centers) if t <= ONSET_SEC]
ictal_idx = [i for i, t in enumerate(t_centers) if t >  ONSET_SEC]
inter_maj = majority_cluster(aligned[inter_idx])
ictal_maj = majority_cluster(aligned[ictal_idx])

print("Generating figures ...")


# Hierarchical: correlation heatmap
corr_ord   = corr_full[np.ix_(order, order)]
labels_ord = [CHANNEL_NAMES[i] for i in order]
flat_ord   = flat[order]

fig, ax = plt.subplots(figsize=(8.5, 7))
im = ax.imshow(corr_ord, cmap="RdYlBu_r", vmin=0, vmax=1,
               aspect="auto", interpolation="nearest")
for c in range(K):
    idx = [i for i, v in enumerate(flat_ord) if v == c]
    if idx:
        lo, hi = min(idx), max(idx)
        ax.add_patch(Rectangle(
            (lo - 0.5, lo - 0.5), hi - lo + 1, hi - lo + 1,
            linewidth=2.5, edgecolor=COLORS[c], facecolor="none", zorder=4
        ))
ax.set_xticks(range(N_CH))
ax.set_yticks(range(N_CH))
ax.set_xticklabels(labels_ord, rotation=90, fontsize=8)
ax.set_yticklabels(labels_ord, fontsize=8)
ax.set_title(
    "EEG Channel Correlation Matrix — Ward Hierarchical Clustering\n"
    "CHB-01 · chb01_03 · 2980–3010 s",
    fontweight="bold", pad=12
)
cbar = fig.colorbar(im, ax=ax, fraction=0.035, pad=0.03)
cbar.set_label("Pearson r", fontsize=10)
cbar.outline.set_linewidth(0.8)
ax.legend(
    handles=[mpatches.Patch(facecolor=COLORS[c], edgecolor="#aaaaaa",
             linewidth=0.5, label=f"Cluster {c+1}") for c in range(K)],
    loc="upper left", bbox_to_anchor=(1.18, 1.0), title="Clusters", title_fontsize=9
)
plt.tight_layout()
fig.savefig(FIG_DIR / "fig_hierarchical_heatmap.png", dpi=300, bbox_inches="tight")
plt.close(fig)


# Hierarchical: dendrograms (interictal & ictal)
sch.set_link_color_palette(COLORS)

def plot_dendrogram(lnk, title, accent_color, filename):
    cut = 0.55 * lnk[-1, 2]
    fig, ax = plt.subplots(figsize=(6, 8))
    sch.dendrogram(lnk, labels=CHANNEL_NAMES, orientation="left", ax=ax,
                   color_threshold=cut, above_threshold_color="#BBBBBB", leaf_font_size=10)
    ax.axvline(cut, color="#666666", linewidth=1.2, linestyle="--")
    ax.set_xlabel("Ward distance  (1 − r)", fontsize=11)
    ax.set_title(title, fontweight="bold", pad=12, color=accent_color)
    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)
    for line in ax.get_lines():
        line.set_linewidth(2.0)
    ax.legend(
        handles=[
            *[mpatches.Patch(facecolor=COLORS[c], edgecolor="#aaaaaa",
                             linewidth=0.5, label=f"Cluster {c+1}") for c in range(K)],
            mpatches.Patch(facecolor="#CCCCCC", edgecolor="#aaaaaa", linewidth=0.5, label="Above cut"),
            plt.Line2D([0], [0], color="#666666", linestyle="--", linewidth=1.2, label=f"Cut  (k={K})"),
        ],
        loc="lower right", fontsize=9, framealpha=0.9
    )
    plt.tight_layout()
    fig.savefig(FIG_DIR / filename, dpi=300, bbox_inches="tight")
    plt.close(fig)

plot_dendrogram(linkage_inter,
                "EEG Channel Dendrogram — Interictal  (0–16 s)\nCHB-01 · chb01_03 · 2980–2996 s",
                INTER_COLOR, "fig_hierarchical_dendrogram_interictal.png")
plot_dendrogram(linkage_ictal,
                "EEG Channel Dendrogram — Ictal  (16–30 s)\nCHB-01 · chb01_03 · 2996–3010 s",
                ICTAL_COLOR, "fig_hierarchical_dendrogram_ictal.png")


# Spectral: cluster raster
onset_frame = next(i for i, t in enumerate(t_centers) if t > ONSET_SEC) - 0.5

fig, ax = plt.subplots(figsize=(13, 5))
ax.imshow(aligned.T, aspect="auto", cmap=ListedColormap(COLORS[:K]),
          vmin=0, vmax=K - 1, interpolation="nearest")
ax.axvspan(-0.5, onset_frame, alpha=0.07, color=INTER_COLOR, zorder=0)
ax.axvspan(onset_frame, n_windows - 0.5, alpha=0.07, color=ICTAL_COLOR, zorder=0)
ax.axvline(onset_frame, color=ONSET_COLOR, linewidth=2.0, linestyle="--", zorder=5)

tick_frames = [min(range(n_windows), key=lambda i: abs(t_centers[i] - t)) for t in [5, 10, 15, 20, 25]]
ax.set_xticks(tick_frames)
ax.set_xticklabels(["5s", "10s", "15s", "20s", "25s"], fontsize=9)
ax.set_yticks(range(N_CH))
ax.set_yticklabels(CHANNEL_NAMES, fontsize=8.5)
ax.set_xlabel("Window centre (s)", fontsize=11)
ax.set_title("Spectral Cluster Assignment per Channel over Time  ·  k=4  ·  CHB-01 chb01_03",
             fontweight="bold", pad=10)
ax.annotate("Interictal", xy=(onset_frame * 0.5, 1), xycoords=("data", "axes fraction"),
            xytext=(0, -28), textcoords="offset points",
            ha="center", va="top", color=INTER_COLOR, fontsize=10, fontweight="bold")
ax.annotate("Ictal", xy=(onset_frame + (n_windows - onset_frame) * 0.5, 1),
            xycoords=("data", "axes fraction"), xytext=(0, -28), textcoords="offset points",
            ha="center", va="top", color=ICTAL_COLOR, fontsize=10, fontweight="bold")
for spine in ["top", "right"]:
    ax.spines[spine].set_visible(False)
ax.legend(
    handles=[
        *[mpatches.Patch(facecolor=COLORS[c], edgecolor="white", label=f"Cluster {c+1}") for c in range(K)],
        plt.Line2D([0], [0], color=ONSET_COLOR, linestyle="--", linewidth=2, label="Seizure onset"),
    ],
    loc="upper right", framealpha=0.9, fontsize=9
)
plt.tight_layout()
fig.savefig(FIG_DIR / "fig_spectral_raster.png", dpi=300, bbox_inches="tight")
plt.close(fig)


# Spectral: interictal vs ictal dominant cluster
sort_order   = np.argsort(inter_maj)
ch_sorted    = [CHANNEL_NAMES[i] for i in sort_order]
inter_sorted = inter_maj[sort_order]
ictal_sorted = ictal_maj[sort_order]
changed      = inter_sorted != ictal_sorted

fig, axes = plt.subplots(1, 2, figsize=(11, 7.5), sharey=True)
fig.suptitle("Dominant Spectral Cluster per Channel  ·  Interictal vs Ictal  ·  CHB-01 chb01_03",
             fontweight="bold", fontsize=13, y=1.01)
for ax, majority, title, accent in zip(
    axes,
    [inter_sorted, ictal_sorted],
    ["Interictal  (0 – 16 s)", "Ictal  (16 – 30 s)"],
    [INTER_COLOR, ICTAL_COLOR],
):
    for ch_i, (_, cluster) in enumerate(zip(ch_sorted, majority)):
        ax.barh(ch_i, 1.0, color=COLORS[cluster], alpha=0.85, height=0.72,
                edgecolor="white", linewidth=0.5)
        ax.text(1.04, ch_i, f"Cluster {cluster + 1}", va="center",
                fontsize=8.5, color=COLORS[cluster], fontweight="bold")
    ax.set_xlim(0, 1.45)
    ax.set_xticks([])
    ax.set_title(title, color=accent, fontsize=12, fontweight="bold", pad=10)
    for spine in ["top", "right", "bottom"]:
        ax.spines[spine].set_visible(False)
    ax.spines["left"].set_linewidth(0.8)
axes[0].set_yticks(range(N_CH))
axes[0].set_yticklabels(ch_sorted, fontsize=9)
for ch_i in range(N_CH):
    if changed[ch_i]:
        axes[1].annotate("◀ switched", xy=(0, ch_i), xytext=(-0.4, ch_i),
                         fontsize=7.5, color="#888888", va="center", ha="right")
fig.legend(
    handles=[mpatches.Patch(facecolor=COLORS[c], edgecolor="white", label=f"Cluster {c+1}")
             for c in range(K)],
    loc="lower center", ncol=4, bbox_to_anchor=(0.5, -0.03), framealpha=0.9, fontsize=9
)
plt.tight_layout()
fig.savefig(FIG_DIR / "fig_spectral_comparison.png", dpi=300, bbox_inches="tight")
plt.close(fig)


# Spectral: eigengap over time
eigengap = all_eigenvalues[:, K] - all_eigenvalues[:, K - 1]

fig, ax = plt.subplots(figsize=(11, 4))
phase_spans(ax, t_centers[0], t_centers[-1])
ax.plot(t_centers, eigengap, color="#2C3E50", linewidth=2.0, zorder=3)
ax.fill_between(t_centers, eigengap, alpha=0.15, color="#2C3E50", zorder=2)
ax.set_xlabel("Window centre (s)", fontsize=11)
ax.set_ylabel(f"Eigengap  (λ{K+1} − λ{K})", fontsize=11)
ax.set_title("Laplacian Eigengap over Time  ·  larger = stronger cluster structure  ·  CHB-01 chb01_03",
             fontweight="bold", pad=10)
ax.set_xlim(t_centers[0], t_centers[-1])
ax.set_ylim(bottom=0)
ax.text(ONSET_SEC / 2, ax.get_ylim()[1], "Interictal",
        ha="center", va="top", color=INTER_COLOR, fontsize=10, fontweight="bold")
ax.text(ONSET_SEC + (t_centers[-1] - ONSET_SEC) / 2, ax.get_ylim()[1], "Ictal",
        ha="center", va="top", color=ICTAL_COLOR, fontsize=10, fontweight="bold")
ax.legend(handles=[plt.Line2D([0], [0], color=ONSET_COLOR, linestyle="--",
                              linewidth=2, label="Seizure onset")],
          fontsize=9, framealpha=0.9)
for spine in ["top", "right"]:
    ax.spines[spine].set_visible(False)
plt.tight_layout()
fig.savefig(FIG_DIR / "fig_spectral_eigengap.png", dpi=300, bbox_inches="tight")
plt.close(fig)


# Global network synchrony
fig, ax = plt.subplots(figsize=(11, 3.5))
ax.axvspan(t_centers[0], ONSET_SEC,    alpha=0.12, color=INTER_COLOR, label="Interictal")
ax.axvspan(ONSET_SEC,    t_centers[-1], alpha=0.12, color=ICTAL_COLOR,  label="Ictal")
ax.axvline(ONSET_SEC, color=ONSET_COLOR, linewidth=2.0, linestyle="--",
           label=f"Seizure onset (+{ONSET_SEC}s)", zorder=4)
ax.plot(t_centers, mean_corr, color="#2C3E50", linewidth=2.0, zorder=3)
ax.fill_between(t_centers, mean_corr, alpha=0.18, color=INTER_COLOR, zorder=2)
ax.set_xlabel("Time within recording window (s)", fontsize=11)
ax.set_ylabel("Mean Pearson r", fontsize=11)
ax.set_title("Global Network Synchrony  ·  CHB-01  ·  chb01_03  ·  2980–3010 s",
             fontweight="bold", pad=10)
ax.set_xlim(t_centers[0], t_centers[-1])
ax.set_ylim(bottom=0)
ymax = ax.get_ylim()[1]
ax.text(ONSET_SEC / 2, ymax * 0.97, "Interictal",
        ha="center", va="top", color=INTER_COLOR, fontsize=10, fontweight="bold")
ax.text(ONSET_SEC + (t_centers[-1] - ONSET_SEC) / 2, ymax * 0.97, "Ictal",
        ha="center", va="top", color=ICTAL_COLOR, fontsize=10, fontweight="bold")
ax.legend(fontsize=9, framealpha=0.9, loc="lower right")
for spine in ["top", "right"]:
    ax.spines[spine].set_visible(False)
plt.tight_layout()
fig.savefig(FIG_DIR / "fig_global_synchrony.png", dpi=300, bbox_inches="tight")
plt.close(fig)

print("Done.")
