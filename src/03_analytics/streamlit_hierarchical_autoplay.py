"""
EEG Network Transition — Streamlit Live Demo (auto-play)
=========================================================
Run with:  streamlit run streamlit_hierarchical_autoplay.py
"""
 
from pathlib import Path
import time
import warnings
import numpy as np
import scipy.sparse as sp
import scipy.cluster.hierarchy as sch
from scipy.spatial.distance import squareform
import matplotlib.pyplot as plt
import streamlit as st

st.set_page_config(
    page_title="EEG Network Transition · CHB-01",
    page_icon="🧠",
    layout="wide"
)

# constants
NPZ_PATH      = (
    Path(__file__).resolve().parents[2]
    / "data" / "graphs" / "adjacency_sparse"
    / "inter_to_ict_chb01_03_2980_3010_adjacency_sparse.npz"
)
N_CHANNELS    = 23
N_TIMEPOINTS  = 7680
FS            = 256
ONSET_SEC     = 16
 
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
 
 
# cached data loading
@st.cache_resource(show_spinner="Loading data and precomputing windows…")
def load_and_precompute(window_sec, step_sec):
    mat            = sp.load_npz(NPZ_PATH)
    window_samples = window_sec * FS
    step_samples   = step_sec * FS
    t_starts       = list(range(0, N_TIMEPOINTS - window_samples + 1, step_samples))
    t_centers_sec  = [(t + window_samples / 2) / FS for t in t_starts]
 
    all_corrs = []
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
        all_corrs.append(np.nan_to_num(corr, nan=0.0))
 
    all_corrs = np.array(all_corrs)
    idx       = np.triu_indices(N_CHANNELS, k=1)
    mean_corr = [c[idx].mean() for c in all_corrs]
 
    interictal_idx = [i for i, t in enumerate(t_centers_sec) if t <= ONSET_SEC]
    corr_inter     = all_corrs[interictal_idx].mean(axis=0)
    dist_inter     = squareform(1.0 - np.clip(corr_inter, -1, 1), checks=False)
    order          = sch.dendrogram(sch.linkage(dist_inter, method="ward"), no_plot=True)["leaves"]
 
    return all_corrs, mean_corr, t_starts, t_centers_sec, order
 
 
# sidebar
st.sidebar.title("🧠 EEG Network Demo")
st.sidebar.markdown(
    "**Dataset:** CHB-MIT · chb01_03  \n"
    "**Window:** 2980–3010 s  \n"
    "**Seizure onset:** +16 s"
)
st.sidebar.divider()
 
window_sec  = st.sidebar.slider("Window size (s)", 2, 10, 5)
step_sec    = st.sidebar.slider("Step size (s)", 1, 5, 1)
speed       = st.sidebar.slider("Speed (s per frame)", 0.2, 2.0, 0.8, step=0.1)
cmap        = st.sidebar.selectbox("Colour map", ["RdYlBu_r", "coolwarm", "viridis", "plasma"])
vmin        = st.sidebar.slider("Colour min", -0.5, 0.0, -0.2, step=0.05)
vmax        = st.sidebar.slider("Colour max",  0.1, 1.0,  0.5, step=0.05)
show_dendro = st.sidebar.checkbox("Show dendrogram", value=False)
 
st.sidebar.divider()
playing = st.sidebar.toggle("▶  Auto-play", value=True)
 
# load data
all_corrs, mean_corr, t_starts, t_centers_sec, order = load_and_precompute(window_sec, step_sec)
labels   = [CHANNEL_NAMES[i] for i in order]
n_frames = len(all_corrs)
 
def reorder(m):
    return m[np.ix_(order, order)]
 
# session state for current frame
if "frame" not in st.session_state:
    st.session_state.frame = 0
 
# header
st.title("EEG Network Transition: Interictal → Ictal")
st.markdown(
    "Watch the functional connectivity network reorganise as the seizure begins. "
    "The **red dashed line** marks seizure onset at **+16 s**."
)
 
# manual frame control (visible when paused)
if not playing:
    st.session_state.frame = st.slider(
        "Frame", 0, n_frames - 1, st.session_state.frame
    )
 
frame = st.session_state.frame
t_s   = t_starts[frame] / FS
t_e   = t_s + window_sec
t_c   = t_centers_sec[frame]
phase = "🔴 ICTAL" if t_c > ONSET_SEC else "🟣 interictal"
color = ICTAL_CLR if t_c > ONSET_SEC else INTERICTAL_CLR
 
st.markdown(
    f"**Window:** {t_s:.0f}–{t_e:.0f} s &nbsp;|&nbsp; "
    f"**Phase:** <span style='color:{color};font-weight:bold'>{phase}</span> &nbsp;|&nbsp; "
    f"**Mean r:** {mean_corr[frame]:.3f}",
    unsafe_allow_html=True
)
 
# plots
col1, col2 = st.columns([1.6, 1], gap="large")
 
with col1:
    fig, ax = plt.subplots(figsize=(7, 5.5), facecolor=BG)
    ax.set_facecolor(PANEL_BG)
    for spine in ax.spines.values(): spine.set_visible(False)
    im = ax.imshow(reorder(all_corrs[frame]), cmap=cmap,
                   vmin=vmin, vmax=vmax, aspect="auto", interpolation="nearest")
    ax.set_xticks(range(N_CHANNELS))
    ax.set_yticks(range(N_CHANNELS))
    ax.set_xticklabels(labels, rotation=90, fontsize=6.5, color=TEXT)
    ax.set_yticklabels(labels, fontsize=6.5, color=TEXT)
    ax.tick_params(colors=TEXT)
    ax.set_title(f"Channel Correlation  ·  {t_s:.0f}–{t_e:.0f} s",
                 color=color, fontsize=10, fontweight="bold")
    cbar = fig.colorbar(im, ax=ax, fraction=0.035, pad=0.02)
    cbar.set_label("Pearson r", color=TEXT, fontsize=9)
    cbar.ax.yaxis.set_tick_params(color=TEXT, labelcolor=TEXT)
    cbar.outline.set_edgecolor(PANEL_BG)
    plt.tight_layout()
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)
 
with col2:
    fig, ax = plt.subplots(figsize=(4.5, 5.5), facecolor=BG)
    ax.set_facecolor(PANEL_BG)
    for spine in ax.spines.values(): spine.set_edgecolor("#333")
    ax.axvspan(0, ONSET_SEC, alpha=0.10, color=INTERICTAL_CLR, label="Interictal")
    ax.axvspan(ONSET_SEC, t_centers_sec[-1] + window_sec / 2,
               alpha=0.10, color=ICTAL_CLR, label="Ictal")
    ax.axvline(ONSET_SEC, color=ONSET_COLOR, linewidth=2,
               linestyle="--", label="Seizure onset")
    ax.plot(t_centers_sec, mean_corr, color="white", linewidth=1.8, alpha=0.5)
    ax.fill_between(t_centers_sec, mean_corr, alpha=0.12, color="white")
    ax.axvline(t_c, color=color, linewidth=2, linestyle=":")
    ax.plot(t_c, mean_corr[frame], "o", color=color, markersize=10, zorder=5)
    ax.set_xlabel("Window centre (s)", color=TEXT, fontsize=9)
    ax.set_ylabel("Mean Pearson r", color=TEXT, fontsize=9)
    ax.set_title("Global Synchrony", color=TEXT, fontsize=10, fontweight="bold")
    ax.tick_params(colors=TEXT, labelsize=8)
    ax.legend(facecolor=PANEL_BG, edgecolor="#444", labelcolor=TEXT, fontsize=8)
    plt.tight_layout()
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)
 
# optional dendrogram
if show_dendro:
    st.divider()
    dist_w = squareform(1.0 - np.clip(all_corrs[frame], -1, 1), checks=False)
    link_w = sch.linkage(dist_w, method="ward")
    sch.set_link_color_palette(["#7c6af7", "#f7936a", "#5ecfb1", "#e05c97", "#f5d547"])
    fig, ax = plt.subplots(figsize=(11, 3.5), facecolor=BG)
    ax.set_facecolor(PANEL_BG)
    for spine in ax.spines.values(): spine.set_visible(False)
    sch.dendrogram(link_w, labels=CHANNEL_NAMES, orientation="left", ax=ax,
                   color_threshold=0.55 * link_w[-1, 2],
                   above_threshold_color=TEXT, leaf_font_size=9)
    ax.tick_params(colors=TEXT, labelsize=8)
    ax.set_xlabel("Ward distance  (1 − r)", color=TEXT, fontsize=9)
    ax.set_title(f"Cluster structure  ·  {t_s:.0f}–{t_e:.0f} s  ·  {phase}",
                 color=color, fontsize=10, fontweight="bold")
    for line in ax.get_lines(): line.set_linewidth(1.8)
    plt.tight_layout()
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)
 
# auto-play loop
if playing:
    time.sleep(speed)
    st.session_state.frame = (frame + 1) % n_frames
    st.rerun()
 
st.divider()
st.caption(
    "CHB-MIT Scalp EEG Database · chb01_03.edf · "
    "Seizure onset annotated at 2996 s (PhysioNet)."
)
 