"""
cosmograph_prep.py — HVG timepoint-level Cosmograph files

Nodes  = individual HVG timepoints (23 channels × phase_duration)
           interictal : 23 × 4096 = 94,208 nodes  (0–16 s)
           ictal      : 23 × 3584 = 82,432 nodes  (16–30 s)

Edges  = HVG adjacency edges within each phase (same file for all algorithms)

Community algorithms:
  Stream Moore  — community_labels.npy (per-node, from full-graph run)
  LPA           — synchronous weighted LPA, channel-initialised on HVG subgraph
  Hierarchical  — Ward clustering on phase-specific channel correlations;
                  each timepoint inherits its channel's cluster label
  Spectral      — spectral clustering on phase-specific channel correlations;
                  each timepoint inherits its channel's cluster label

Outputs (src/03_analytics/outputs/cosmograph/):
  interictal_edges.csv  /  ictal_edges.csv     ← shared per phase
  {algo}_{phase}_nodes.csv                      ← 8 node files
"""

import csv
import warnings
from pathlib import Path

import numpy as np
import scipy.cluster.hierarchy as sch
import scipy.sparse as sp
from scipy.spatial.distance import squareform

# ── paths ──────────────────────────────────────────────────────────────────
_HERE    = Path(__file__).resolve().parent
BASE_DIR = _HERE.parents[1]
NPZ_PATH = (
    BASE_DIR / "data" / "graphs" / "adjacency_sparse"
    / "inter_to_ict_chb01_03_2980_3010_adjacency_sparse.npz"
)
SM_LABELS_PATH = _HERE / "outputs" / "community_labels.npy"
OUT_DIR        = _HERE / "outputs" / "cosmograph"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ── config ─────────────────────────────────────────────────────────────────
N_CH         = 23
N_TP         = 7680        # total timepoints per channel (30 s × 256 Hz)
ONSET        = 4096        # seizure onset sample (16 s × 256 Hz)
FS           = 256
THRESHOLD    = 0.3         # correlation threshold for hierarchical / spectral
K_CLUSTERS   = 4

CHANNEL_NAMES = [
    "FP1-F7", "F7-T7",  "T7-P7",  "P7-O1",
    "FP1-F3", "F3-C3",  "C3-P3",  "P3-O1",
    "FP2-F4", "F4-C4",  "C4-P4",  "P4-O2",
    "FP2-F8", "F8-T8",  "T8-P8",  "P8-O2",
    "FZ-CZ",  "CZ-PZ",
    "P7-T7",  "T7-FT9", "FT9-FT10", "FT10-T8",
    "T8-P8-1",
]
CHANNEL_REGIONS = {
    "FP1-F7": "frontal",  "F7-T7": "frontal",   "T7-P7": "temporal",  "P7-O1": "parietal",
    "FP1-F3": "frontal",  "F3-C3": "central",   "C3-P3": "central",   "P3-O1": "parietal",
    "FP2-F4": "frontal",  "F4-C4": "central",   "C4-P4": "central",   "P4-O2": "parietal",
    "FP2-F8": "frontal",  "F8-T8": "frontal",   "T8-P8": "temporal",  "P8-O2": "parietal",
    "FZ-CZ":  "central",  "CZ-PZ": "central",
    "P7-T7":  "temporal", "T7-FT9": "temporal", "FT9-FT10": "temporal",
    "FT10-T8": "temporal", "T8-P8-1": "temporal",
}

PHASES = {
    "interictal": (0,     ONSET),   # 0–16 s
    "ictal":      (ONSET, N_TP),    # 16–30 s
}

# ── load data ──────────────────────────────────────────────────────────────
print("Loading adjacency matrix …")
mat       = sp.load_npz(NPZ_PATH).tocsr()
sm_labels = np.load(SM_LABELS_PATH)   # shape (176640,), per original node
print(f"  shape: {mat.shape}  |  {mat.nnz:,} non-zeros")
print(f"  SM labels: {len(np.unique(sm_labels))} communities\n")


# ── helper: remap label array to 0-indexed ints ────────────────────────────
def remap(labels):
    unique = np.unique(labels)
    m      = {v: i for i, v in enumerate(unique)}
    return np.array([m[l] for l in labels], dtype=np.int64)


# ── helper: channel-level hierarchical clustering ─────────────────────────
def hier_spectral_for_phase(t_start, t_end):
    """
    Compute phase-specific 23×23 channel correlation, then run Ward
    hierarchical clustering and spectral clustering.
    Returns (hier_labels, spec_labels) — one label per channel (0-22).
    """
    n_tp = t_end - t_start
    ch_temporal = np.zeros((N_CH, n_tp), dtype=np.float64)
    for ch in range(N_CH):
        s = ch * N_TP + t_start
        e = ch * N_TP + t_end
        ch_temporal[ch] = np.array(mat[s:e, s:e].sum(axis=1)).flatten()

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        corr = np.corrcoef(ch_temporal)
    corr = np.nan_to_num(corr, nan=0.0)

    # — hierarchical (Ward) —
    dist    = squareform(np.clip(1.0 - corr, 0.0, None), checks=False)
    linkage = sch.linkage(dist, method="ward")
    hier_ch = sch.fcluster(linkage, t=K_CLUSTERS, criterion="maxclust") - 1

    # — spectral —
    A = corr.copy()
    A[A < THRESHOLD] = 0.0
    A[A < 0]         = 0.0
    np.fill_diagonal(A, 0.0)
    deg        = A.sum(axis=1)
    d_inv_sqrt = np.where(deg > 0, 1.0 / np.sqrt(np.maximum(deg, 1e-10)), 0.0)
    L          = np.eye(N_CH) - np.diag(d_inv_sqrt) @ A @ np.diag(d_inv_sqrt)
    _, vecs    = np.linalg.eigh(L)
    emb        = vecs[:, :K_CLUSTERS].copy()
    norms      = np.linalg.norm(emb, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    emb /= norms

    rng         = np.random.default_rng(42)
    best_assign, best_inertia = None, np.inf
    for _ in range(20):
        centers = emb[rng.choice(N_CH, K_CLUSTERS, replace=False)]
        for _ in range(300):
            dists  = np.linalg.norm(emb[:, None, :] - centers[None, :, :], axis=2)
            assign = np.argmin(dists, axis=1)
            new_c  = np.array([
                emb[assign == k].mean(axis=0) if (assign == k).any()
                else emb[rng.integers(N_CH)]
                for k in range(K_CLUSTERS)
            ])
            if np.allclose(new_c, centers):
                break
            centers = new_c
        inertia = sum(
            np.sum((emb[assign == k] - centers[k]) ** 2)
            for k in range(K_CLUSTERS) if (assign == k).any()
        )
        if inertia < best_inertia:
            best_inertia, best_assign = inertia, assign.copy()

    return hier_ch, best_assign


# ── helper: synchronous weighted LPA ──────────────────────────────────────
def run_lpa(adj_csr, init_labels, max_iter=50):
    """
    Synchronous weighted LPA.
    init_labels: 1-D int array, length = n_nodes (channel index used as seed).
    Each iteration: every node adopts the label whose neighbours contribute
    the most total edge weight.  All nodes update simultaneously.
    """
    labels = init_labels.copy().astype(np.int64)

    for it in range(max_iter):
        unique_labs = np.unique(labels)
        k = len(unique_labs)
        # score[i, j] = total weight of edges from node i to nodes with label unique_labs[j]
        scores = np.zeros((adj_csr.shape[0], k), dtype=np.float32)
        for ki, lab in enumerate(unique_labs):
            mask          = (labels == lab).astype(np.float32)
            scores[:, ki] = adj_csr.dot(mask)

        new_labels = unique_labs[scores.argmax(axis=1)]

        if np.array_equal(new_labels, labels):
            print(f"      converged at iteration {it + 1}")
            break
        labels = new_labels

    return remap(labels)


# ── main: process each phase ───────────────────────────────────────────────
for phase, (t_start, t_end) in PHASES.items():
    n_tp_phase = t_end - t_start
    n_nodes    = N_CH * n_tp_phase

    print(f"{'='*60}")
    print(f"  PHASE: {phase.upper()}  "
          f"({t_start/FS:.0f}–{t_end/FS:.0f} s  |  {n_tp_phase} samples/channel)")
    print(f"  nodes: {n_nodes:,}")
    print(f"{'='*60}")

    # Node index array: original node IDs in the full graph
    node_ids = np.concatenate([
        np.arange(ch * N_TP + t_start, ch * N_TP + t_end)
        for ch in range(N_CH)
    ])                                            # shape (n_nodes,)

    # Subgraph (sparse, local indices 0..n_nodes-1)
    print("  Extracting HVG subgraph …")
    adj_sub = mat[node_ids][:, node_ids].tocsr()
    adj_coo = adj_sub.tocoo()
    n_edges = int((adj_coo.row < adj_coo.col).sum())
    print(f"  edges: {n_edges:,}")

    # ── community labels ──────────────────────────────────────────────────

    # 1. Stream Moore — look up per-node labels from saved array
    print("  [1/4] Stream Moore …")
    sm_sub = remap(sm_labels[node_ids])
    print(f"        communities: {len(np.unique(sm_sub))}")

    # 2. LPA — synchronous, initialised by channel index
    print("  [2/4] LPA …")
    ch_init  = np.array([int(nid) // N_TP for nid in node_ids], dtype=np.int64)
    lpa_sub  = run_lpa(adj_sub, ch_init)
    print(f"        communities: {len(np.unique(lpa_sub))}")

    # 3 & 4. Hierarchical + Spectral — channel-level, then extend to timepoints
    print("  [3/4] Hierarchical  &  [4/4] Spectral …")
    hier_ch, spec_ch = hier_spectral_for_phase(t_start, t_end)
    hier_sub = remap(np.array([hier_ch[int(nid) // N_TP] for nid in node_ids]))
    spec_sub = remap(np.array([spec_ch[int(nid) // N_TP] for nid in node_ids]))
    print(f"        hierarchical clusters : {len(np.unique(hier_sub))}")
    print(f"        spectral clusters     : {len(np.unique(spec_sub))}")

    # ── write shared edges CSV ────────────────────────────────────────────
    edges_path = OUT_DIR / f"{phase}_edges.csv"
    print(f"\n  Writing {edges_path.name} …")
    with open(edges_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["source", "target", "weight"])
        for r, c, v in zip(adj_coo.row, adj_coo.col, adj_coo.data):
            if r < c:
                w.writerow([int(node_ids[r]), int(node_ids[c]), round(float(v), 6)])

    # ── write per-algorithm node CSVs ─────────────────────────────────────
    algo_labels = {
        "stream_moore": sm_sub,
        "lpa":          lpa_sub,
        "hierarchical": hier_sub,
        "spectral":     spec_sub,
    }

    for algo, labs in algo_labels.items():
        nodes_path = OUT_DIR / f"{algo}_{phase}_nodes.csv"
        with open(nodes_path, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow([
                "id", "channel_idx", "channel_name", "region",
                "timepoint", "time_sec", "community",
            ])
            for local_i, orig_id in enumerate(node_ids):
                ch = int(orig_id) // N_TP
                tp = int(orig_id) % N_TP
                w.writerow([
                    int(orig_id),
                    ch,
                    CHANNEL_NAMES[ch],
                    CHANNEL_REGIONS.get(CHANNEL_NAMES[ch], "unknown"),
                    tp,
                    round(tp / FS, 4),
                    int(labs[local_i]),
                ])
        print(f"  {nodes_path.name}  ({len(np.unique(labs))} communities)")

    print()

# ── summary ────────────────────────────────────────────────────────────────
print(f"All files written to:\n  {OUT_DIR}\n")
print("Files per phase:")
print("  {phase}_edges.csv              ← shared edge file for that phase")
print("  stream_moore_{phase}_nodes.csv")
print("  lpa_{phase}_nodes.csv")
print("  hierarchical_{phase}_nodes.csv")
print("  spectral_{phase}_nodes.csv")
print()
print("Cosmograph upload (cosmograph.app):")
print("  1. New graph → Nodes: stream_moore_interictal_nodes.csv  (id: 'id')")
print("                 Links: interictal_edges.csv  (source / target)")
print("  2. Color nodes by 'community'")
print("  3. Optionally color by 'channel_name', 'region', or 'time_sec'")
print("  4. Repeat with ictal files to compare")
