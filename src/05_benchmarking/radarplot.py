"""
Radar Diagram for Community Detection Benchmarking

This script generates a radar diagram comparing algorithm performance
across four dimensions:
- Runtime
- Stability
- Adaptability
- Interpretability

import numpy as np
import matplotlib.pyplot as plt

# -----------------------------
# Data (normalized 1–5 scores)
# -----------------------------
labels = ['Runtime', 'Stability', 'Adaptability', 'Interpretability']

ward = [4, 5, 1, 4]
lpa = [5, 3, 4, 2]
spectral = [3, 4, 3, 4]
streaming = [1, 4, 5, 2]

# -----------------------------
# Prepare radar angles
# -----------------------------
angles = np.linspace(0, 2 * np.pi, len(labels), endpoint=False)
angles = np.concatenate((angles, [angles[0]]))


def plot_algorithm(values, label, ax):
    """Plot a single algorithm on the radar chart"""
    values = values + [values[0]]
    ax.plot(angles, values, linewidth=2, label=label)


# -----------------------------
# Create figure
# -----------------------------
fig = plt.figure(figsize=(6, 7.5))
ax = plt.subplot(111, polar=True)

# Remove outer circle (clean look)
ax.spines['polar'].set_visible(False)
ax.set_frame_on(False)

# Subtle grid lines
for r in [1, 2, 3, 4, 5]:
    ax.plot(angles, [r] * len(angles), linewidth=0.4, linestyle='dotted')

# -----------------------------
# Plot algorithms (full names)
# -----------------------------
plot_algorithm(ward, 'Ward Hierarchical Clustering', ax)
plot_algorithm(lpa, 'Label Propagation Algorithm', ax)
plot_algorithm(spectral, 'Spectral Clustering', ax)
plot_algorithm(streaming, 'Moore Streaming Algorithm', ax)

# -----------------------------
# Axis formatting
# -----------------------------
ax.set_xticks(angles[:-1])
ax.set_xticklabels(labels, fontsize=12)
ax.set_yticks([])

# Title
plt.title('Radar Diagram', pad=25)

# Legend BELOW plot (no overlap)
plt.legend(loc='lower center', bbox_to_anchor=(0.5, -0.35), ncol=2)

# Layout & save
plt.tight_layout()
plt.savefig('radar_diagram.png', dpi=300, bbox_inches='tight')

plt.show()
