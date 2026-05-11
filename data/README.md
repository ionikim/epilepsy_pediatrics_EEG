# Data Directory

This directory contains the raw EEG files, preprocessed data, and graph outputs used in this project.

---

## Structure

```text
data/
├── raw/
│   ├── chb01_03.edf
│   └── chb01_03.edf.seizures
│
├── preprocessed/
│   ├── chb01_01_preprocessed.pkl
│   └── chb01_15_full_labeled.pkl
│
└── graphs/
    ├── adjacency_sparse/
    ├── cosmograph/
    ├── edges/
    ├── gephi/
    ├── metadata/
    ├── nodes/
    └── windows/
```
---

## Raw Data

The original EEG recordings are from the CHB-MIT Scalp EEG Database.

Dataset source:

```text
https://physionet.org/content/chbmit/1.0.0/
```

Raw EEG files are **not included in this repository** because of file size and data distribution considerations.

Download the required `.edf` files from PhysioNet and place them in:

```text
data/raw/
```

Example:

```text
data/raw/chb01_03.edf
data/raw/chb01_03.edf.seizures
```

The `.edf.seizures` file contains seizure annotation information for the corresponding `.edf` recording.

## Preprocessed Data

The `preprocessed/` directory contains intermediate files generated from the raw EEG recordings.

These files may include filtered, segmented, or labeled EEG data saved as `.pkl` files.

Examples:

```text
data/preprocessed/chb01_01_preprocessed.pkl
data/preprocessed/chb01_15_full_labeled.pkl
```

## Graph Data

The `graphs/` directory contains graph outputs generated from the preprocessed EEG data.

These graph files are used for analysis and visualization of EEG-based brain networks.

### `graphs/adjacency_sparse/`

Contains sparse adjacency matrix outputs.

### `graphs/cosmograph/`

Contains node and edge CSV files for Cosmograph visualization.

Examples:

```text
data/graphs/cosmograph/cosmo_ict_nodes.csv
data/graphs/cosmograph/cosmo_ict_edges.csv
data/graphs/cosmograph/cosmo_pre_nodes.csv
data/graphs/cosmograph/cosmo_pre_edges.csv
data/graphs/cosmograph/cosmograph_nodes_all.csv
data/graphs/cosmograph/cosmograph_edges_all.csv
data/graphs/cosmograph/cosmograph_nodes_CZ.csv
data/graphs/cosmograph/cosmograph_edges_CZ.csv
```

### `graphs/gephi/`

Contains `.gexf` files for Gephi visualization.

Example:

```text
data/graphs/gephi/inter_to_ict_chb01_03_2980_3010_full.gexf
```

### `graphs/metadata/`

Contains metadata files used to interpret generated graphs.

Examples:

```text
data/graphs/metadata/inter_to_ict_chb01_03_2980_3010_metadata.json
data/graphs/metadata/inter_to_ict_chb01_03_2980_3010_layer_info.csv
data/graphs/metadata/inter_to_ict_chb01_03_2980_3010_node_index.json
data/graphs/metadata/inter_to_ict_chb01_03_2980_3010_node_labels.json
```

These files may include graph-level metadata, layer information, node indices, and node labels.

### `graphs/nodes/`

Contains node tables generated during graph construction.

### `graphs/edges/`

Contains edge tables generated during graph construction.

### `graphs/windows/`

Contains graph outputs organized by time window.

## Data Source & Reference
Guttag, J. (2010). CHB-MIT Scalp EEG Database (version 1.0.0). PhysioNet. RRID:SCR_007345. https://doi.org/10.13026/C2K01R
Ali Shoeb. Application of Machine Learning to Epileptic Seizure Onset Detection and Treatment. PhD Thesis, Massachusetts Institute of Technology, September 2009.
