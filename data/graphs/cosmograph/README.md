# EEG Visibility Graph — Cosmograph Visualization Files

## Overview

This directory contains CSV files for visualizing the Visibility Graph (VG) network of CHB01 epileptic EEG data in [Cosmograph](https://cosmograph.app). Two scopes are provided: **single electrode (CZ)** for detailed streaming analysis, and **all 23 electrodes** for a full-brain head layout view.

---

## File Structure

├── cosmograph_nodes_CZ.csv      # CZ electrode nodes (timeline)

├── cosmograph_edges_CZ.csv      # CZ electrode edges (timeline)

├── cosmograph_nodes_all.csv     # All 23 electrodes nodes (timeline)

├── cosmograph_edges_all.csv     # All 23 electrodes edges (timeline)

├── cosmo_pre_nodes.csv          # Pre-ictal nodes, all electrodes (static)

├── cosmo_pre_edges.csv          # Pre-ictal edges, all electrodes (static)

├── cosmo_ict_nodes.csv          # Ictal nodes, all electrodes (static)

└── cosmo_ict_edges.csv          # Ictal edges, all electrodes (static)

---

## File Descriptions

### 1. `cosmograph_nodes_CZ.csv` + `cosmograph_edges_CZ.csv` (intra layer only)

**Purpose:** Streaming animation of the CZ electrode's Visibility Graph across all 30 one-second windows.

**What it shows:** How the VG structure of a single electrode evolves second by second — from sparse pre-ictal connections (windows 0–14) to a densely connected ictal network (windows 15–29).

**Node metadata:**

| Column | Type | Description |
|--------|------|-------------|
| `id` | string | Node ID (e.g. `w5_t128` = window 5, timepoint 128) |
| `x` | float | Circular layout X coordinate (radius = 500) |
| `y` | float | Circular layout Y coordinate |
| `degree` | float | VG degree of this timepoint in this window |
| `time` | int | Window index (0–29) → **Cosmograph timeline column** |
| `phase` | string | `pre-ictal` (windows 0–14) or `ictal` (windows 15–29) |
| `timepoint` | int | Position within the 1-second window (0–255) |

**Edge metadata:**

| Column | Type | Description |
|--------|------|-------------|
| `source` | string | Source node ID |
| `target` | string | Target node ID |
| `weight` | float | `1 / distance` — closer timepoints have higher weight |
| `time` | int | Window index → **Cosmograph timeline column** |

**Key statistics:**

| Metric | Value |
|--------|-------|
| Total nodes | 7,680 (256 timepoints × 30 windows) |
| Total edges | 1,458 |
| Pre-ictal edges | 64 (windows 0–14) |
| Ictal edges | 1,394 (windows 15–29) |
| Edge increase | **21.8×** |

---

### 2. `cosmograph_nodes_all.csv` + `cosmograph_edges_all.csv` (intra layer only)

**Purpose:** Streaming animation of all 23 EEG electrodes simultaneously, arranged in a 10-20 head layout.

**What it shows:** The full-brain VG network evolving second by second. Each electrode is rendered as a small circular VG (256 nodes in a ring) positioned at its anatomical scalp location. At window 15 (seizure onset), all electrodes activate simultaneously.

**Node metadata:**

| Column | Type | Description |
|--------|------|-------------|
| `id` | string | Node ID (e.g. `e9_w15_t128` = electrode 9, window 15, timepoint 128) |
| `x` | float | Head layout X coordinate (canvas 5000×5000) |
| `y` | float | Head layout Y coordinate |
| `degree` | float | VG degree of this timepoint in this window |
| `time` | int | Window index (0–29) → **Cosmograph timeline column** |
| `phase` | string | `pre-ictal` or `ictal` |
| `electrode` | string | Electrode label (e.g. FP1, CZ, P7) |
| `color` | string | Hex color code per electrode |
| `timepoint` | int | Position within the 1-second window (0–255) |

**Edge metadata:**

| Column | Type | Description |
|--------|------|-------------|
| `source` | string | Source node ID |
| `target` | string | Target node ID |
| `weight` | float | `1 / distance` |
| `time` | int | Window index → **Cosmograph timeline column** |
| `electrode` | string | Electrode label |

**Key statistics:**

| Metric | Value |
|--------|-------|
| Total nodes | 176,640 (23 electrodes × 30 windows × 256 timepoints) |
| Total edges | 33,419 |
| Pre-ictal edges | ~4,800 |
| Ictal edges | ~28,600 |

---

### 3. `cosmo_pre_nodes.csv` + `cosmo_pre_edges.csv` (intra+inter layer with threshold)

**Purpose:** Static snapshot of the 15-second pre-ictal VG network across all 23 electrodes.

**What it shows:** The baseline brain network before seizure onset. Node degree is averaged across all 15 pre-ictal windows. Edges include both intra-electrode VG connections and inter-electrode functional connectivity, filtered by threshold.

---

### 4. `cosmo_ict_nodes.csv` + `cosmo_ict_edges.csv` (intra+inter layer with threshold)

**Purpose:** Static snapshot of the 15-second ictal VG network across all 23 electrodes.

**What it shows:** The seizure-state brain network. Compared to pre-ictal, node degrees are dramatically higher, intra-electrode VG edges are 17× more numerous, and inter-electrode connections nearly quadruple — reflecting global brain synchronization during the seizure.

---

**Node metadata (pre/ictal static files):**

| Column | Type | Description |
|--------|------|-------------|
| `id` | string | Node ID (e.g. `e12_t128` = electrode 12, timepoint 128) |
| `x` | float | Head layout X coordinate |
| `y` | float | Head layout Y coordinate |
| `degree` | float | Mean VG degree averaged over 15 windows |
| `electrode` | string | Electrode label (FP1, FP2, ..., T9) |
| `electrode_id` | int | Electrode index (0–22) |
| `phase` | string | `pre-ictal` or `ictal` |
| `timepoint` | int | Position within the 1-second window (0–255) |

**Edge metadata (pre/ictal static files):**

| Column | Type | Description |
|--------|------|-------------|
| `source` | string | Source node ID |
| `target` | string | Target node ID |
| `weight` | float | Frequency 0–1 (intra) or Pearson correlation 0–1 (inter) |
| `type` | string | `intra` (within electrode) or `inter` (between electrodes) |
| `electrode` | string | Electrode label (intra) or `EL1-EL2` pair (inter) |

**Thresholds applied:**

| Edge type | Threshold | Rationale |
|-----------|-----------|-----------|
| Intra (VG frequency) | ≥ 0.2 | Edge must appear in at least 3 of 15 windows — filters transient noise-driven connections |
| Inter (degree correlation) | ≥ 0.5 | Only strongly co-activating electrode pairs — reflects genuine functional synchronization |

**Key statistics:**

| Metric | Pre-ictal | Ictal |
|--------|-----------|-------|
| Nodes | 5,888 | 5,888 |
| Intra edges (freq ≥ 0.2) | 243 | 4,122 |
| Inter edges (corr ≥ 0.5) | 59 | 220 |
| Total edges | **302** | **4,342** |
| Edge increase | — | **14.4×** |
| Mean inter-electrode correlation | 0.26 | 0.51 |
