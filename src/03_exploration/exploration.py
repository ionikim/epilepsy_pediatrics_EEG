"""
Network Exploration for EEG-Derived Functional Networks
-------------------------------------------------------
Descriptive, non-algorithmic exploration of weighted EEG graphs.
This file intentionally uses NetworkX only for feature extraction
and visualization (NOT for community detection).

import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from scipy import sparse

# ======================================================
# 1. Load adjacency matrix
# ======================================================
DATA_PATH = (
    "data/graphs/adjacency_sparse/"
    "inter_to_ict_chb01_03_2980_3010_adjacency_sparse.npz"
)

npz = np.load(DATA_PATH, allow_pickle=True)
print("Available keys:", npz.files)

A = npz[npz.files[0]]

# Handle stacked matrices (time windows)
if isinstance(A, np.ndarray) and A.dtype == object:
    A = A[0]

# ======================================================
# 2. Build weighted undirected graph
# ======================================================
G = nx.from_scipy_sparse_array(A, edge_attribute="weight")

print(nx.info(G))

# ======================================================
# 3. Basic structural statistics
# ======================================================
N = G.number_of_nodes()
M = G.number_of_edges()
density = nx.density(G)

weights = np.array([d["weight"] for _, _, d in G.edges(data=True)])
strengths = np.array([s for _, s in G.degree(weight="weight")])

print(f"Nodes: {N}")
print(f"Edges: {M}")
print(f"Density: {density:.4f}")

# ======================================================
# 4. Edge-weight distribution (Fig. 1)
# ======================================================
plt.figure(figsize=(6, 4))
plt.hist(weights, bins=60)
plt.xlabel("Edge weight")
plt.ylabel("Frequency")
plt.title("Edge weight distribution")
plt.tight_layout()
plt.show()

# ======================================================
# 5. Node strength distribution (Fig. 2)
# ======================================================
plt.figure(figsize=(6, 4))
plt.hist(strengths, bins=60)
plt.xlabel("Node strength (weighted degree)")
plt.ylabel("Frequency")
plt.title("Node strength distribution")
plt.tight_layout()
plt.show()

# ======================================================
# 6. Global clustering coefficient
# ======================================================
avg_clustering = nx.average_clustering(G, weight="weight")
print(f"Average weighted clustering coefficient: {avg_clustering:.4f}")

# ======================================================
# 7. Adjacency matrix heatmap (Fig. 3)
# ======================================================
plt.figure(figsize=(5, 5))
plt.imshow(A.todense(), cmap="viridis")
plt.colorbar(label="Edge weight")
plt.title("Adjacency matrix (functional connectivity)")
plt.xlabel("Node index")
plt.ylabel("Node index")
plt.tight_layout()
plt.show()

# ======================================================
# 8. Network visualization (sampled, Fig. 4)
# ======================================================
# Full EEG graphs are too dense → subsample nodes
np.random.seed(42)
sample_nodes = np.random.choice(G.nodes(), size=min(30, N), replace=False)
G_sub = G.subgraph(sample_nodes)

pos = nx.spring_layout(G_sub, seed=42, weight="weight")

plt.figure(figsize=(7, 7))
nx.draw_networkx_nodes(
    G_sub,
    pos,
    node_size=300,
    node_color="steelblue",
    alpha=0.9
)
nx.draw_networkx_edges(
    G_sub,
    pos,
    width=1.2,
    alpha=0.6
)
plt.title("Sampled EEG functional network (spring layout)")
plt.axis("off")
plt.tight_layout()
plt.show()

# ======================================================
# 9. Summary (for logging / report)
# ======================================================
summary = {
    "nodes": N,
    "edges": M,
    "density": density,
    "edge_weight_mean": float(weights.mean()),
    "edge_weight_std": float(weights.std()),
    "strength_mean": float(strengths.mean()),
    "clustering": avg_clustering,
}

print("\nSummary statistics:")
for k, v in summary.items():
    print(f"{k:25s}: {v}")

# ======================================================
# 10. Temporal network exploration (NEW – Figure 5)
# ======================================================
"""
Purpose:
Assess temporal non-stationarity of EEG-derived networks.
This motivates the use of adaptive / streaming community detection.
"""

# ------------------------------------------------------
# Assumption:
# NPZ file contains a sequence of adjacency matrices
# (sliding windows over time)
# ------------------------------------------------------

# Reload full object (may contain multiple windows)
npz = np.load(DATA_PATH, allow_pickle=True)
A_all = npz[npz.files[0]]

# Sanity check: only do temporal analysis if multiple windows exist
if isinstance(A_all, np.ndarray) and A_all.dtype == object and len(A_all) > 1:

    mean_strength_over_time = []
    mean_weight_over_time = []
    clustering_over_time = []

    for A_t in A_all:
        G_t = nx.from_scipy_sparse_array(A_t, edge_attribute="weight")

        # Mean node strength
        strengths_t = [s for _, s in G_t.degree(weight="weight")]
        mean_strength_over_time.append(np.mean(strengths_t))

        # Mean edge weight
        weights_t = [d["weight"] for _, _, d in G_t.edges(data=True)]
        mean_weight_over_time.append(np.mean(weights_t))

        # Global clustering
        clustering_over_time.append(
            nx.average_clustering(G_t, weight="weight")
        )

    # --------------------------------------------------
    # Plot temporal evolution (Figure 5)
    # --------------------------------------------------
    plt.figure(figsize=(7, 4))
    plt.plot(mean_strength_over_time, label="Mean node strength")
    plt.plot(mean_weight_over_time, label="Mean edge weight")
    plt.plot(clustering_over_time, label="Global clustering")

    plt.xlabel("Time window")
    plt.ylabel("Network-level metric value")
    plt.title("Temporal evolution of EEG network properties")
    plt.legend()
    plt.tight_layout()
    plt.show()

else:
    print(
        "Temporal analysis skipped: "
        "only a single adjacency matrix found."
    )
