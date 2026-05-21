"""
Network Exploration for EEG-Derived Functional Networks
-------------------------------------------------------
Extended network analysis with structural and statistical exploration.
"""

import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from scipy import sparse

# ======================================================
# SECTION HEADER
# ======================================================
print("\nSECTION 4 · Network Exploration")
print("──────────────────────────────────────────────────────────────────────")

# ======================================================
# 1. Load adjacency matrix
# ======================================================
DATA_PATH = (
    "data/graphs/adjacency_sparse/"
    "inter_to_ict_chb01_03_2980_3010_adjacency_sparse.npz"
)

print("Loading adjacency matrix …")

npz = np.load(DATA_PATH, allow_pickle=True)
print("Available keys:", npz.files)

A = npz[npz.files[0]]

if isinstance(A, np.ndarray) and A.dtype == object:
    A = A[0]

# ======================================================
# 2. Build weighted undirected graph
# ======================================================
G = nx.from_scipy_sparse_array(A, edge_attribute="weight")

# ======================================================
# 3. Basic structural statistics
# ======================================================
N = G.number_of_nodes()
M = G.number_of_edges()
density = nx.density(G)

weights = np.array([d["weight"] for _, _, d in G.edges(data=True)])
strengths = np.array([s for _, s in G.degree(weight="weight")])

print(f"Nodes {N:,} | Edges {M:,} | Density {density:.6f}")
print(f"Edge weight  mean={weights.mean():.4f}  std={weights.std():.6f}")
print(f"Node strength mean={strengths.mean():.4f}")

# ======================================================
# 4. Sampled clustering
# ======================================================
np.random.seed(42)
sample_nodes_5k = np.random.choice(list(G.nodes()), size=min(5000, N), replace=False)
G_sample_5k = G.subgraph(sample_nodes_5k)

clustering_sample = nx.average_clustering(G_sample_5k, weight="weight")
print(f"Sampled avg weighted clustering coeff (5,000 nodes): {clustering_sample:.4f}")

# ======================================================
# 5. Edge-weight distribution
# ======================================================
plt.figure(figsize=(6, 4))
plt.hist(weights, bins=60)
plt.xlabel("Edge weight")
plt.ylabel("Frequency")
plt.title("Edge weight distribution")
plt.tight_layout()
plt.savefig("exploration_fig1.png")
plt.close()

# ======================================================
# 6. Edge-weight distribution (zoomed)
# ======================================================
plt.figure(figsize=(6, 4))
plt.hist(weights, bins=100)
plt.xlim(0.9, 1.0)
plt.xlabel("Edge weight (zoomed)")
plt.ylabel("Frequency")
plt.title("Edge weight distribution (zoomed)")
plt.tight_layout()
plt.savefig("exploration_fig1_zoom.png")
plt.close()

# ======================================================
# 7. Node strength distribution
# ======================================================
plt.figure(figsize=(6, 4))
plt.hist(strengths, bins=60)
plt.xlabel("Node strength (weighted degree)")
plt.ylabel("Frequency")
plt.title("Node strength distribution")
plt.tight_layout()
plt.savefig("exploration_fig2.png")
plt.close()

# ======================================================
# 8. Ranked node strengths
# ======================================================
sorted_strengths = np.sort(strengths)[::-1]

plt.figure(figsize=(6, 4))
plt.plot(sorted_strengths[:1000])
plt.xlabel("Rank")
plt.ylabel("Node strength")
plt.title("Ranked node strengths (top 1000)")
plt.tight_layout()
plt.savefig("exploration_fig2_ranked.png")
plt.close()

# ======================================================
# 9. Global clustering coefficient
# ======================================================
avg_clustering = nx.average_clustering(G, weight="weight")

# ======================================================
# 10. Adjacency matrix heatmap
# ======================================================
plt.figure(figsize=(5, 5))
plt.imshow(A.todense(), cmap="viridis")
plt.colorbar(label="Edge weight")
plt.title("Adjacency matrix (functional connectivity)")
plt.xlabel("Node index")
plt.ylabel("Node index")
plt.tight_layout()
plt.savefig("exploration_fig3.png")
plt.close()

# ======================================================
# 11. High-strength node subgraph
# ======================================================
strength_dict = dict(G.degree(weight="weight"))
sorted_nodes = sorted(strength_dict, key=strength_dict.get, reverse=True)

top_nodes = sorted_nodes[:1000]
G_top = G.subgraph(top_nodes)

sample_nodes = np.random.choice(list(G_top.nodes()), size=min(50, len(G_top)), replace=False)
G_sub = G_top.subgraph(sample_nodes)

pos = nx.spring_layout(G_sub, seed=42, weight="weight")

plt.figure(figsize=(7, 7))
nx.draw_networkx_nodes(G_sub, pos, node_size=300, node_color="steelblue")
nx.draw_networkx_edges(G_sub, pos, alpha=0.6)
plt.title("Subgraph of high-strength EEG nodes")
plt.axis("off")
plt.tight_layout()
plt.savefig("exploration_fig4_high_strength.png")
plt.close()

# ======================================================
# 12. Random sample graph
# ======================================================
np.random.seed(42)
sample_nodes = np.random.choice(list(G.nodes()), size=min(30, N), replace=False)
G_sub = G.subgraph(sample_nodes)

pos = nx.spring_layout(G_sub, seed=42, weight="weight")

plt.figure(figsize=(7, 7))
nx.draw_networkx_nodes(G_sub, pos, node_size=300, node_color="steelblue")
nx.draw_networkx_edges(G_sub, pos, width=1.2, alpha=0.6)
plt.title("Sampled EEG functional network (random)")
plt.axis("off")
plt.tight_layout()
plt.savefig("exploration_fig4.png")
plt.close()

# ======================================================
# 13. Summary
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
# 14. Temporal network exploration
# ======================================================
npz = np.load(DATA_PATH, allow_pickle=True)
A_all = npz[npz.files[0]]

if isinstance(A_all, np.ndarray) and A_all.dtype == object and len(A_all) > 1:

    mean_strength_over_time = []
    mean_weight_over_time = []
    clustering_over_time = []

    for A_t in A_all:
        G_t = nx.from_scipy_sparse_array(A_t, edge_attribute="weight")

        strengths_t = [s for _, s in G_t.degree(weight="weight")]
        mean_strength_over_time.append(np.mean(strengths_t))

        weights_t = [d["weight"] for _, _, d in G_t.edges(data=True)]
        mean_weight_over_time.append(np.mean(weights_t))

        clustering_over_time.append(
            nx.average_clustering(G_t, weight="weight")
        )

    plt.figure(figsize=(7, 4))
    plt.plot(mean_strength_over_time, label="Mean node strength")
    plt.plot(mean_weight_over_time, label="Mean edge weight")
    plt.plot(clustering_over_time, label="Global clustering")

    plt.xlabel("Time window")
    plt.ylabel("Network-level metric value")
    plt.title("Temporal evolution of EEG network properties")
    plt.legend()
    plt.tight_layout()
    plt.savefig("exploration_fig5.png")
    plt.close()

else:
    print(
        "Temporal analysis skipped: only a single adjacency matrix found."
    )
