"""
Laplacian Spectral Clustering — Streamlit Live Demo
====================================================
Run with:  streamlit run streamlit_spectral.py
"""

from pathlib import Path
import time
import warnings
import numpy as np
import scipy.sparse as sp
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import streamlit as st

st.set_page_config(
    page_title="Spectral Clustering · CHB-01",
    page_icon="🧠",
    layout="wide"
)

# constants
NPZ_PATH     = (
    Path(__file__).resolve().parents[2]
    / "data" / "graphs" / "adjacency_sparse"
    / "inter_to_ict_chb01_03_2980_3010_adjacency_sparse.npz"
)
N_CHANNELS   = 23
N_TIMEPOINTS = 7680
FS           = 256
ONSET_SEC    = 16

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

# SPECTRAL CLUSTERING CORE  (pure numpy, no sklearn)

def build_adjacency(corr, threshold):
    A = corr.copy()
    A[A < threshold] = 0.0
    A[A < 0]         = 0.0
    np.fill_diagonal(A, 0.0)
    return A

def normalized_laplacian(A):
    degree     = A.sum(axis=1)
    d_inv_sqrt = np.zeros(N_CHANNELS)
    nz         = degree > 0
    d_inv_sqrt[nz] = 1.0 / np.sqrt(degree[nz])
    D          = np.diag(d_inv_sqrt)
    return np.eye(N_CHANNELS) - D @ A @ D

def spectral_embed(L, k):
    vals, vecs = np.linalg.eigh(L)
    return vecs[:, :k], vals

def normalize_rows(X):
    norms = np.linalg.norm(X, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return X / norms

def kmeans_scratch(X, k, n_init=8, max_iter=200, tol=1e-6, seed=42):
    rng          = np.random.default_rng(seed)
    best_labels  = None
    best_inertia = np.inf

    for run in range(n_init):
        centroids = [X[rng.integers(0, len(X))].copy()]
        for _ in range(k - 1):
            dists    = np.array([min(np.sum((x - c)**2) for c in centroids) for x in X])
            probs    = dists / (dists.sum() + 1e-12)
            centroids.append(X[rng.choice(len(X), p=probs)].copy())
        centroids = np.array(centroids)
        labels    = np.zeros(len(X), dtype=int)

        for _ in range(max_iter):
            d_all      = np.array([np.sum((X - c)**2, axis=1) for c in centroids])
            new_labels = np.argmin(d_all, axis=0)
            new_c      = np.array([
                X[new_labels == c].mean(axis=0) if (new_labels == c).any()
                else X[rng.integers(0, len(X))]
                for c in range(k)
            ])
            if np.linalg.norm(new_c - centroids) < tol:
                break
            centroids, labels = new_c, new_labels

        inertia = sum(
            np.sum((X[labels == c] - centroids[c])**2)
            for c in range(k) if (labels == c).any()
        )
        if inertia < best_inertia:
            best_inertia, best_labels = inertia, labels.copy()

    return best_labels

def spectral_clustering(corr, k, threshold):
    A          = build_adjacency(corr, threshold)
    L          = normalized_laplacian(A)
    emb, vals  = spectral_embed(L, k)
    emb        = normalize_rows(emb)
    labels     = kmeans_scratch(emb, k)
    return labels, vals, emb

def align_labels(ref, new, k):
    perm, used = {}, set()
    for rc in range(k):
        best_overlap, best_nc = -1, -1
        for nc in range(k):
            if nc in used:
                continue
            ov = (new[ref == rc] == nc).sum()
            if ov > best_overlap:
                best_overlap, best_nc = ov, nc
        perm[best_nc] = rc
        used.add(best_nc)
    return np.array([perm.get(l, l) for l in new])

# CACHED PRECOMPUTATION

@st.cache_data(show_spinner="Precomputing spectral clustering for all windows…")
def precompute(window_sec, step_sec, k, threshold):
    mat            = sp.load_npz(NPZ_PATH)
    window_samples = window_sec * FS
    step_samples   = step_sec   * FS
    t_starts       = list(range(0, N_TIMEPOINTS - window_samples + 1, step_samples))
    t_centers_sec  = [(t + window_samples / 2) / FS for t in t_starts]
    n_windows      = len(t_starts)

    all_labels     = []
    all_eigenvalues = []
    all_embeddings  = []
    all_corrs       = []

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
        corr = np.nan_to_num(corr, nan=0.0)

        labels, evals, emb = spectral_clustering(corr, k, threshold)
        all_labels.append(labels)
        all_eigenvalues.append(evals)
        all_embeddings.append(emb)
        all_corrs.append(corr)

    # align labels across windows to window 0
    aligned = [all_labels[0].copy()]
    for i in range(1, n_windows):
        aligned.append(align_labels(aligned[0], all_labels[i], k))

    all_labels      = np.array(aligned)
    all_eigenvalues = np.array(all_eigenvalues)
    all_embeddings  = np.array(all_embeddings)
    all_corrs       = np.array(all_corrs)

    eigengap = all_eigenvalues[:, k] - all_eigenvalues[:, k - 1]

    return all_labels, all_eigenvalues, all_embeddings, all_corrs, eigengap, t_starts, t_centers_sec

# SIDEBAR

st.sidebar.title("🧠 Spectral Clustering Demo")
st.sidebar.markdown(
    "**Dataset:** CHB-MIT · chb01_03  \n"
    "**Window:** 2980–3010 s  \n"
    "**Seizure onset:** +16 s"
)
st.sidebar.divider()

window_sec = st.sidebar.slider("Window size (s)",    2,   10,   5)
step_sec   = st.sidebar.slider("Step size (s)",      1,    5,   1)
k          = st.sidebar.slider("Clusters (k)",       2,    6,   4)
threshold  = st.sidebar.slider("Correlation threshold", 0.0, 0.8, 0.3, step=0.05)
speed      = st.sidebar.slider("Speed (s per frame)", 0.2, 3.0, 1.0, step=0.1)

st.sidebar.divider()
playing = st.sidebar.toggle("▶  Auto-play", value=True)

# LOAD DATA

all_labels, all_eigenvalues, all_embeddings, all_corrs, eigengap, t_starts, t_centers_sec = precompute(
    window_sec, step_sec, k, threshold
)
n_frames     = len(all_labels)
colors_k     = CLUSTER_COLORS[:k]

# session state
if "frame" not in st.session_state:
    st.session_state.frame = 0
if st.session_state.frame >= n_frames:
    st.session_state.frame = 0

# manual slider when paused
if not playing:
    st.session_state.frame = st.slider("Frame", 0, n_frames - 1, st.session_state.frame)

frame = st.session_state.frame
t_s   = t_starts[frame] / FS
t_e   = t_s + window_sec
t_c   = t_centers_sec[frame]
phase = "ICTAL" if t_c > ONSET_SEC else "Interictal"
color = ICTAL_CLR if t_c > ONSET_SEC else INTERICTAL_CLR

# HEADER

st.title("Laplacian Spectral Clustering · EEG Seizure Transition")
st.markdown(
    "Watch how EEG channels reorganise into spectral clusters as the seizure approaches. "
    "The **red dashed line** marks annotated seizure onset at **+16 s**."
)
st.markdown(
    f"**Window:** {t_s:.0f}–{t_e:.0f} s &nbsp;|&nbsp; "
    f"**Phase:** <span style='color:{color};font-weight:bold'>{phase}</span> &nbsp;|&nbsp; "
    f"**Eigengap:** {eigengap[frame]:.3f} &nbsp;|&nbsp; "
    f"**k = {k}** clusters",
    unsafe_allow_html=True
)

# PLOT 1 — cluster assignment per channel  (full width, viewport height)

labels_now = all_labels[frame]
emb_now    = all_embeddings[frame]
evals      = all_eigenvalues[frame]

st.subheader("1 · Cluster Assignment per Channel")

fig, ax = plt.subplots(figsize=(14, 9), facecolor=BG)
ax.set_facecolor(PANEL_BG)
for spine in ax.spines.values():
    spine.set_visible(False)

for ch_idx in range(N_CHANNELS):
    cl = labels_now[ch_idx]
    ax.barh(ch_idx, 1, color=colors_k[cl], alpha=0.85, height=0.75)
    ax.text(1.02, ch_idx, f"Cluster {cl + 1}", va="center", fontsize=11,
            color=colors_k[cl], fontweight="bold")

ax.set_yticks(range(N_CHANNELS))
ax.set_yticklabels(CHANNEL_NAMES, fontsize=12, color=TEXT)
ax.set_xlim(0, 1.5)
ax.set_xticks([])
ax.set_title(f"Cluster Assignment per Channel  ·  {t_s:.0f}–{t_e:.0f} s  ·  {phase}",
             color=color, fontsize=14, fontweight="bold", pad=16)
ax.tick_params(colors=TEXT)
legend_patches = [mpatches.Patch(color=colors_k[c], label=f"Cluster {c + 1}") for c in range(k)]
ax.legend(handles=legend_patches, facecolor=PANEL_BG, edgecolor="#444",
          labelcolor=TEXT, fontsize=11, loc="lower right")
plt.tight_layout()
st.pyplot(fig, width="stretch")
plt.close(fig)

# PLOT 2 — eigenspace scatter  (full width, viewport height)

st.divider()
st.subheader("2 · Eigenspace Embedding")

fig, ax = plt.subplots(figsize=(14, 9), facecolor=BG)
ax.set_facecolor(PANEL_BG)
for spine in ax.spines.values():
    spine.set_edgecolor("#333")
ax.grid(color="#333", linewidth=0.6, linestyle="--", alpha=0.5)

for cl in range(k):
    mask = labels_now == cl
    ax.scatter(
        emb_now[mask, 1], emb_now[mask, 2] if k > 2 else emb_now[mask, 0],
        color=colors_k[cl], s=280, zorder=3, alpha=0.9,
        edgecolors="white", linewidths=0.8, label=f"Cluster {cl + 1}"
    )
for ch_idx in range(N_CHANNELS):
    x = emb_now[ch_idx, 1]
    y = emb_now[ch_idx, 2] if k > 2 else emb_now[ch_idx, 0]
    ax.annotate(CHANNEL_NAMES[ch_idx], (x, y), fontsize=9, color=TEXT,
                alpha=0.9, xytext=(6, 6), textcoords="offset points")

ax.set_xlabel("Eigenvector 2  (λ₂)", color=TEXT, fontsize=12)
ax.set_ylabel("Eigenvector 3  (λ₃)", color=TEXT, fontsize=12)
ax.set_title(f"Eigenspace Embedding  ·  {t_s:.0f}–{t_e:.0f} s  ·  {phase}",
             color=color, fontsize=14, fontweight="bold", pad=16)
ax.tick_params(colors=TEXT, labelsize=10)
ax.legend(facecolor=PANEL_BG, edgecolor="#444", labelcolor=TEXT,
          fontsize=11, loc="upper right")
plt.tight_layout()
st.pyplot(fig, width="stretch")
plt.close(fig)

# PLOT 3 — eigenvalue spectrum  (full width, viewport height)

st.divider()
st.subheader("3 · Laplacian Eigenvalue Spectrum")

fig, ax = plt.subplots(figsize=(14, 9), facecolor=BG)
ax.set_facecolor(PANEL_BG)
for spine in ax.spines.values():
    spine.set_edgecolor("#333")

ax.bar(range(N_CHANNELS), evals, color=color, alpha=0.75, width=0.7)
ax.axvline(k - 0.5, color=ONSET_COLOR, linewidth=2.5, linestyle="--", label=f"k={k} cut")
ax.annotate(
    f"eigengap = {eigengap[frame]:.3f}",
    xy=(k - 0.5, evals[k]),
    xytext=(k + 1.5, evals[k] + 0.12),
    color=TEXT, fontsize=12,
    arrowprops=dict(arrowstyle="->", color=TEXT, lw=1.2)
)
ax.set_xlabel("Eigenvalue index", color=TEXT, fontsize=12)
ax.set_ylabel("Eigenvalue", color=TEXT, fontsize=12)
ax.set_title(f"Laplacian Eigenvalue Spectrum  ·  {t_s:.0f}–{t_e:.0f} s  ·  {phase}",
             color=color, fontsize=14, fontweight="bold", pad=16)
ax.tick_params(colors=TEXT, labelsize=10)
ax.legend(facecolor=PANEL_BG, edgecolor="#444", labelcolor=TEXT, fontsize=11)
plt.tight_layout()
st.pyplot(fig, width="stretch")
plt.close(fig)

# PLOT 4 — eigengap trace  (full width, viewport height)

st.divider()
st.subheader("4 · Cluster Separation Strength over Time")

fig, ax = plt.subplots(figsize=(14, 9), facecolor=BG)
ax.set_facecolor(PANEL_BG)
for spine in ax.spines.values():
    spine.set_edgecolor("#333")

ax.axvspan(0, ONSET_SEC, alpha=0.08, color=INTERICTAL_CLR, label="Interictal")
ax.axvspan(ONSET_SEC, t_centers_sec[-1] + window_sec / 2,
           alpha=0.08, color=ICTAL_CLR, label="Ictal")
ax.axvline(ONSET_SEC, color=ONSET_COLOR, linewidth=2.5, linestyle="--", label="Seizure onset")
ax.plot(t_centers_sec, eigengap, color="white", linewidth=3, alpha=0.7, zorder=2)
ax.fill_between(t_centers_sec, eigengap, alpha=0.18, color="white")
ax.axvline(t_c, color=color, linewidth=2.5, linestyle=":", zorder=3)
ax.plot(t_c, eigengap[frame], "o", color=color, markersize=16, zorder=4)

ax.set_xlabel("Window centre (s)", color=TEXT, fontsize=12)
ax.set_ylabel(f"Eigengap  (λ{k+1} − λ{k})", color=TEXT, fontsize=12)
ax.set_title("Cluster Separation Strength  ·  larger = more distinct clusters",
             color=TEXT, fontsize=14, fontweight="bold", pad=16)
ax.tick_params(colors=TEXT, labelsize=10)
ax.legend(facecolor=PANEL_BG, edgecolor="#444", labelcolor=TEXT,
          fontsize=11, loc="upper left", ncol=3)
plt.tight_layout()
st.pyplot(fig, width="stretch")
plt.close(fig)

# PLOT 5 — raster carpet  (full width, viewport height)

st.divider()
st.subheader("5 · All Cluster Assignments over Time")

from matplotlib.colors import ListedColormap
cmap_disc = ListedColormap(colors_k)

fig, ax = plt.subplots(figsize=(14, 9), facecolor=BG)
ax.set_facecolor(PANEL_BG)
for spine in ax.spines.values():
    spine.set_visible(False)

ax.imshow(all_labels.T, aspect="auto", cmap=cmap_disc,
          vmin=0, vmax=k - 1, interpolation="nearest")

onset_frame = next(
    (i for i, t in enumerate(t_centers_sec) if t > ONSET_SEC), len(t_centers_sec) - 1
) - 0.5
ax.axvline(onset_frame, color=ONSET_COLOR, linewidth=2.5, linestyle="--", label="Seizure onset")
ax.axvline(frame, color="white", linewidth=2.5, linestyle=":", alpha=0.9)

ax.set_yticks(range(N_CHANNELS))
ax.set_yticklabels(CHANNEL_NAMES, fontsize=10, color=TEXT)
ax.set_xticks(range(n_frames))
ax.set_xticklabels([f"{t:.0f}s" for t in t_centers_sec], rotation=90, fontsize=9, color=TEXT)
ax.tick_params(colors=TEXT)
ax.set_xlabel("Window centre", color=TEXT, fontsize=12)
ax.set_title(f"Cluster Assignment over Time  ·  k={k}  ·  CHB-01",
             color=TEXT, fontsize=14, fontweight="bold", pad=16)

legend_patches = [mpatches.Patch(color=colors_k[c], label=f"Cluster {c + 1}") for c in range(k)]
legend_patches.append(plt.Line2D([0], [0], color=ONSET_COLOR, linestyle="--", label="Seizure onset"))
ax.legend(handles=legend_patches, facecolor=PANEL_BG, edgecolor="#444",
          labelcolor=TEXT, fontsize=10, loc="upper left", ncol=k + 1)
plt.tight_layout()
st.pyplot(fig, width="stretch")
plt.close(fig)

# AUTO-PLAY LOOP

if playing:
    time.sleep(speed)
    st.session_state.frame = (frame + 1) % n_frames
    st.rerun()

st.divider()
st.caption(
    "CHB-MIT Scalp EEG Database · chb01_03.edf · "
    "Seizure onset annotated at 2996 s (PhysioNet). "
    "Spectral clustering via normalized graph Laplacian — pure numpy."
)
