import numpy as np
import networkx as nx
from scipy import sparse
from collections import Counter, defaultdict
import random
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[2]
DATA_DIR = BASE_DIR / "data" / "graphs" / "adjacency_sparse"

npz_files = list(DATA_DIR.glob("*.npz"))

print("Gefundene NPZ-Dateien:")
for f in npz_files:
    print(" -", f)

if len(npz_files) == 0:
    raise FileNotFoundError(
        "Keine .npz Dateien im data/graphs/adjacency_sparse Ordner gefunden!"
    )

FILE = npz_files[0]
print("Verwende Datei:", FILE)

A = sparse.load_npz(FILE)

print("Adjacency shape:", A.shape)
print("Non-zero edges (roh):", A.nnz)

threshold = 0.2

A = A.tocsr()
A.data[A.data < threshold] = 0
A.eliminate_zeros()

print("Non-zero edges (nach Threshold):", A.nnz)

G = nx.from_scipy_sparse_array(A)

print(
    f"Graph: {G.number_of_nodes()} Knoten, "
    f"{G.number_of_edges()} Kanten"
)

def label_propagation(G, max_iter=100):
    labels = {node: node for node in G.nodes()}

    for it in range(max_iter):
        changes = 0
        nodes = list(G.nodes())
        random.shuffle(nodes)

        for node in nodes:
            neighbor_labels = [labels[n] for n in G.neighbors(node)]

            if not neighbor_labels:
                continue

            new_label = Counter(neighbor_labels).most_common(1)[0][0]

            if labels[node] != new_label:
                labels[node] = new_label
                changes += 1

        if changes == 0:
            print(f"LPA konvergiert nach {it + 1} Iterationen")
            break

    return labels

labels = label_propagation(G)

communities = defaultdict(list)
for node, label in labels.items():
    communities[label].append(node)

print(f"\nGefundene Communities: {len(communities)}")

if len(communities) == 0:
    raise RuntimeError(
        "LPA hat keine Communities erzeugt. "
        "Mögliche Ursachen: falsche Initialisierung, zu hoher Threshold "
        "oder isolierter Graph."
    )

sizes = sorted([len(v) for v in communities.values()], reverse=True)
print("Top Community-Größen:", sizes[:10])
print("Größte Community-Anteil:", sizes[0] / G.number_of_nodes())

coverage = len(labels) / G.number_of_nodes()
print("Label-Abdeckung:", coverage)

def run_lpa_once():
    return label_propagation(G)

def label_agreement(a, b):
    common = set(a) & set(b)
    if len(common) == 0:
        return 0.0
    return sum(a[n] == b[n] for n in common) / len(common)

runs = [run_lpa_once() for _ in range(10)]
agreements = [label_agreement(runs[0], r) for r in runs[1:]]

print("Label-Stabilitäten:", agreements)
print("Ø Stabilität:", np.mean(agreements))

from networkx.algorithms.community import modularity

communities_list = [set(v) for v in communities.values()]
Q_real = modularity(G, communities_list)

print("Modularity (real):", Q_real)

G_rand = nx.configuration_model(
    [d for _, d in G.degree()],
    create_using=nx.Graph()
)
G_rand.remove_edges_from(nx.selfloop_edges(G_rand))

labels_rand = label_propagation(G_rand)

communities_rand = defaultdict(list)
for n, l in labels_rand.items():
    communities_rand[l].append(n)

Q_rand = modularity(
    G_rand,
    [set(v) for v in communities_rand.values()]
)

print("Modularity (random):", Q_rand)
print("\nΔ Modularity (real − random):", Q_real - Q_rand)
