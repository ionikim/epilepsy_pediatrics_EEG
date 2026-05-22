# Participants Information
| First Name | Last Name | GitHub Username |
| :--- | :--- | :--- |
| Antonia | SPOERK | @antoniaspoerk |
| Jione | KIM | @ionikim |
| Marina | KOEHLI | @marinakoe |
| Simon | KRUMMENACHER | @simonkrummenacher |

# Temporal and Spatial Network Evolution in Pediatric EEG during Ictal and Interictal Periods

## How to Run

The project can be run either locally or in Google Colab. The easiest way to reproduce the full workflow is to run `pipeline.py` from the project root. This executes the complete pipeline in one step, including data loading, preprocessing, graph construction, analytics, benchmarking, and figure generation.

Cloning the GitHub repository is the recommended method because it preserves the required folder structure and includes all scripts, dependencies, and configuration files. However, cloning is not strictly required; users may also download the repository as a ZIP file from GitHub and run the pipeline from the extracted folder.

---

## Quick Start — Run the Full Pipeline

The main pipeline is self-contained. For the core reproducible workflow, only `pipeline.py` and `requirements.txt` are required. The script creates the required output folders and downloads the raw EEG files if they are missing.

After setting up the environment and installing the dependencies, Place `pipeline.py` and `requirements.txt` in the same folder, then run:

```bash
pip install -r requirements.txt
python pipeline.py
```

The pipeline downloads the required raw EEG files if missing, creates the required output folders, builds and validates the MVG sparse adjacency matrix, and generates the main outputs.

---

## Option A — Local Setup (VS Code / Terminal)

### 1. Get the repository

Recommended method:

```bash
git clone https://github.com/ionikim/epilepsy_pediatrics_EEG.git
cd epilepsy_pediatrics_EEG
```

Alternatively, users can download the repository manually via **Code → Download ZIP**, extract it, and open a terminal in the extracted project folder.

### 2. Set up the environment

Create and activate a virtual environment, then install dependencies. Inside the virtual environment, `python` and `pip` point to the active environment.

**macOS / Linux**

```bash
python3 -m venv env
source env/bin/activate
pip install -r requirements.txt
```

**Windows**

```bash
python -m venv env
env\Scripts\activate
pip install -r requirements.txt
```

### 3. Run the full pipeline

Run from the project root:

```bash
python pipeline.py
```

This runs the full project workflow in one step.

---

### 4. Run individual Python scripts

Run from the **project root** (`epilepsy_pediatrics_EEG/`) with the venv active:

| Script | What it does | Command |
| :----- | :----------- | :------ |
| `analysis/main.py` | LPA community detection on the adjacency matrix | `python analysis/main.py` |
| `src/03_analytics/label_propagation.py` | Label Propagation Algorithm | `python src/03_analytics/label_propagation.py` |
| `src/03_analytics/hierarchical_transition_analysis.py` | Sliding window analysis — saves 4 PNG plots + 1 GIF | `python src/03_analytics/hierarchical_transition_analysis.py` |
| `src/03_analytics/laplacian_spectral_clustering.py` | Spectral clustering — saves 4 PNG plots | `python src/03_analytics/laplacian_spectral_clustering.py` |
| `src/03_analytics/stream_moore_benchmark.py` | Stream-Moore benchmark metrics | `python src/03_analytics/stream_moore_benchmark.py` |

### a. Interactive Streamlit apps

```bash
streamlit run src/03_analytics/streamlit_hierarchical.py          # manual slider
streamlit run src/03_analytics/streamlit_hierarchical_autoplay.py # auto-play animation
streamlit run src/03_analytics/streamlit_spectral.py              # spectral clustering live
```

### b. Jupyter notebooks

Launch JupyterLab from the project root and open notebooks in `src/`:

```bash
jupyter lab
```

| Folder | Contents |
| :----- | :------- |
| `src/01_loading/` | EEG preprocessing and visibility graph construction |
| `src/02_management/` | Data format conversion (NPZ → CSV) |
| `src/04_visualization/` | Gephi export, heatmaps, Cosmograph files |

> **Note:** The raw EEG data (`data/raw/chb01_03.edf`) and precomputed adjacency matrix (`data/graphs/adjacency_sparse/`) are included in the repo via Git LFS. Run `git lfs pull` if the files appear as pointers.

---

## Research Questions
1. How does the Visibility Graph structure (degree, edge density) change between interictal and ictal periods?
2. Do specific electrodes show consistently higher VG activity during seizure periods?
3. Can community structure in the functional connectivity network derived from VG degree sequences distinguish ictal from interictal brain states?
---
## Project Description

This project analyzes the temporal evolution of brain connectivity networks across epileptic seizure transitions using pediatric EEG recordings from the CHB-MIT Scalp EEG Database. We focused on a single patient (CHB01) and a 30-second segment centered on seizure onset — 15 seconds of interictal (normal) activity followed by 15 seconds of ictal (seizure) activity — recorded across 23 EEG channels at 256 Hz.

**Preprocessing.** Raw EDF recordings were re-referenced and bandpass filtered. Signals were segmented into time windows and organized into a structured DataFrame for downstream graph construction.

**Graph construction.** Rather than computing pairwise electrode correlations, we built a Horizontal Visibility Graph (HVG) separately for each electrode's time series. Two timepoints are connected if no intermediate amplitude value exceeds both of them, capturing the temporal regularity and periodicity of the signal. A continuous HVG was then constructed across the full 30-second interictal-to-ictal window, resulting in a sparse adjacency matrix of size 176,640 × 176,640 representing all connections across 23 electrodes and 7,680 timepoints.

**Community detection.** Three algorithms were implemented from scratch and applied to the HVG adjacency matrix. The **Stream-Moore algorithm** (streaming modularity maximisation) proved unsuitable due to near-uniform node degree (~22) and negative modularity (Q ≈ −0.0001). The **Label Propagation Algorithm (LPA)** was applied directly to the adjacency matrix. **Laplacian Spectral Clustering** constructed weighted adjacency graphs from thresholded channel correlation matrices, computed the normalised graph Laplacian, and clustered channels in eigenspace using a custom k-means++ implementation. Additionally, **Ward linkage hierarchical clustering** was applied across sliding 5-second windows to track how functional channel groupings evolve over time.

**Benchmarking.** All algorithms were evaluated using structural quality metrics computed without ground truth labels: intra-cluster edge density, inter/intra edge ratio, conductance, and average clustering coefficient, measured separately for interictal and ictal windows alongside runtime and memory usage.

**Visualization.** Network structure was visualized using Gephi (static MHVG and OPTN graphs), Cosmograph (frequency-weighted animated and static views across all 23 electrodes), HTML-based interactive circular visibility graphs, and three interactive Streamlit apps showing the sliding window correlation network, hierarchical clustering, and spectral clustering live as the seizure unfolds.

---
### 1) Network Loading & Data Management (Ji-one Kim, Marina Köhli): 

We use data from the CHB-MIT Scalp EEG Database, which includes EEG recordings from 5 male patients (ages 3–22) and 17 female patients (ages 1.5–19) with intractable seizures. The recordings contain 23 EEG channels sampled at 256 Hz with 16-bit resolution and are provided in EDF (European Data Format). For this project, we focus on one patient (CHB01) and analyze a 30-second segment centered on seizure onset, consisting of 15 seconds of pre-ictal activity and 15 seconds of ictal activity.

Instead of measuring pairwise correlations between electrodes, we build a Visibility Graph (VG) separately for each electrode’s EEG time series. In each 1-second window, sampled at 256 timepoints, two timepoints are connected if they can “see” each other without any intermediate signal value blocking the line between them. In this way, the graph captures the temporal structure of the signal. More regular and seizure-like signals tend to form denser graphs with more long-range connections and higher node degree, while irregular normal EEG activity tends to produce sparser graphs with mostly short-range connections.

The precomputed adjacency matrix is stored as a sparse matrix of size 176,640 × 176,640, representing all VG connections across 23 electrodes and 7,680 timepoints. Each node index corresponds to both an electrode and a specific timepoint within the full recording.

### 2) Network Exploration & Analytics (Antonia Spörk,Simon Krummenacher): 

We computed network measures to characterize brain connectivity across the seizure transition. Community detection was approached with three algorithms, all implemented from scratch: the **Label Propagation Algorithm (LPA)**, applied directly to the HVG adjacency matrix; the **Stream-Moore algorithm**, a streaming modularity-maximisation approach which proved unsuitable for this graph due to near-uniform node degree (~22) and negative modularity; and **Laplacian Spectral Clustering**, which embeds channels in eigenspace via the normalised graph Laplacian and clusters them with a custom k-means++ implementation. Additionally, **Ward linkage hierarchical clustering** was applied to sliding-window channel correlation matrices to reveal functional groupings over time. All algorithms were benchmarked using structural quality metrics (intra-cluster edge density, conductance, clustering coefficient, inter/intra edge ratio) computed without ground truth labels.

### 3) Network Visualization (Marina Köhli, Ji-One Kim, Antonia Spörk): 

We visualize the evolving VG network using two complementary approaches:

1. Cosmograph:

cosmograph_nodes_CZ.csv + cosmograph_edges_CZ.csv: CZ electrode VG with timeline (7,680 nodes, 30 windows), enabling second-by-second animation in Cosmograph.
cosmo_freq_nodes_preictal/ictal.csv + cosmo_freq_edges_preictal/ictal.csv: All 23 electrodes, frequency-weighted static comparison (5,888 nodes per file). With electrode-based clustering in Cosmograph, the ictal graph shows tight, well-separated electrode communities while the pre-ictal graph is diffuse — directly reflecting VG edge density differences.

2. Html-based interactive visualization:

eeg_vg_comparison_static.html — Compares pre-ictal and ictal circular visibility graphs for a selected electrode, highlighting sparse vs dense connectivity.
eeg_visibility_graph_online_mini_timepoint.html — Animates all 23 electrodes as circular visibility graphs to show seizure evolution over time.
eeg_vg_network_inter.html — Visualizes changing functional connectivity across 23 electrodes during seizure onset.

### 4) Analysis and Interpretation (Simon Krummenacher, Antonia Spörk): 

We will benchmark our solution against existing algorithms and evaluate the strengths and limitations of our project. If capacity allows, we might additionally predict other seizure states within/between subjects and/or conduct an influencer analysis of seizure source electrodes. 


---
## Documentation of work
| Step                                           | Explanation/Sub-steps                                            | Name              |
| :--------------------------------------------- | :--------------------------------------------------------------- | :---------------- |
| Selection of topic and database                |                                                                  | Everyone          |
| Preprocessing                                  | Re-reference and bandpass filter                                 | Marina            |
|                                                | Automate the process for other files, create DataFrame           | Marina and Ji-One |
| Multiplex horizontal visibility graph creation | Select windows (one ictal, one interictal)                       | Ji-One and Marina |
|                                                | Segment DataFrame into windows and averaging                     | Marina            |
|                                                | Create node list, inter- and intra-layer edges, adjacency matrix | Ji-One            |
|                                                | Visualize it in Gephi, compute first exploratory metrics         | Ji-One and Marina |
|                                                | Visualisations (Static visualization, MVG visualization(intra-layer and inter-layer) ) | Ji-One |
| Ordinal Pattern Transition Network creation    | Remove re-referencing, create OPTN, identify electrodes with stronger ictal-interictal differences, visualize it in Gephi                                                                                               | Ji-One            |
| Continuous Multiplex HVG creation              | Apply code from categorical HVG graphs to one continuous window (interictal to ictal) | Marina |
|                                                | Storage in LFS                                                   | Marina            |
| Applied Stream-Moore Algorithm                  | Applied the Stream-Moore community detection algorithm from scratch to HVG adjacency matrix.                                                                 | Antonia   
| Applied Hierarchical Clustering                | Designed a sliding window pipeline extracting 23×23 channel correlation matrices from intra-channel temporal connectivity profiles. Applied Ward linkage hierarchical clustering to reveal functional channel groupings                                                                 | Antonia   
| Build a Streamlit Live Demo                    | Built an interactive live demonstration of the sliding window network analysis using Streamlit, with auto-playing animation showing how the EEG functional connectivity network evolves second by second across the interictal-to-ictal transition                                                                 | Antonia   
| Applied Laplacian Spectral Clustering          | Implemented spectral clustering from scratch   | Antonia  
| Applied Label Propagation Algorithm (LPA)      | Applied the LPA from scratch to HVG adjacency matrix.  | Simon            |
| checked for benchmarking approaches  1   | Comparison between Label Propagation from-scratch and Network X version   | Antonia            |
| checked for benchmarking approaches  2   | Community Detection in Social Networks: An In-depth Benchmarking Study with a Procedure-Oriented Framework -> LPA works best but only with ground truth! Thus we need to evaluate without ground thruth  | Simon            |
| checked for benchmarking approaches  3   | work in progress: find a solution without groundtruth -> Evaluation using Clustering Quality Measures: Clustering quality measures, e.g., SSE (sum of squared errors) or inter- cluster distance  Quality measures used to evaluate community detection should be different from the ones used to find communities.  | Simon            |
| Creating Graphs in Cosmograph | Created Graphs after analysis to show interictal vs. ictal differences in communites and regional organisation                    | Antonia |
| Insights & Analysis |   Get the big picture and structure for report                    | Simon |
| Draft Final Report |                     | Simon |
| Writing Final Report |                 | Everyone |
| Presentation and Final Edit of Report |                 | Marina |
| Final Code Aggregation |                 | Jione |
