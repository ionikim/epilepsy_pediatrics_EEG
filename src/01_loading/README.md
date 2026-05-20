# EEG Preprocessing Pipeline

## Project Overview
This project analyzes pediatric EEG recordings from the **CHB-MIT Scalp EEG Database** to study differences between **interictal (normal brain activity)** and **ictal (seizure activity)**.

To prepare the data for network analysis, EEG recordings are preprocessed and segmented into short time windows.

---

## Dataset
: One recording (chb01_03.edf) from **patient chb01** was used which included a seizure annotation.

---

## Preprocessing Pipeline

The preprocessing steps transform raw EEG signals into structured window-based data.

1. Raw EEG loading (.edf)  
2. Bandpass filtering (1–45 Hz)  
3. Re-referencing (average reference)  
4. Convert signal to a 2D array  
   - rows = time points  
   - columns = electrodes  
5. 5-second window segmentation  
6. Seizure labeling (ictal / interictal)  
7. Feature extraction (window power)
8. Repeat for rest of the files
---

## Output

After preprocessing, each **5-second window** contains:

- signal features for all electrodes
- a label indicating **ictal** or **interictal**

This structured data will be used in the next stage to construct **brain network graphs** and apply **social network analysis metrics**.
