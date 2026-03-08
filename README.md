# 🧠 Overview
In this repository you can find the code of my Practical Project WiSe25/26, supervised by Leonardo Zaggia and Prof. Andrea Hildebrandt (Carl von Ossietzky Universität Oldenburg). Building on the paper Fang et al. (2025) https://doi.org/10.1038/s42003-025-08873-4, it explores the dynamic edge participation coefficient (dePC) in preterm and term-born neonates. 

## Data
We used the neonatal resting-state fMRI data from the Developing Human Connectome Project (Edwards et al., 2022 https://doi.org/10.3389/fnins.2022.886772).  
n = 796, preterm = 182 -> Preterm1; premenstrual age (PMA) at scan >= 37 = 133, Preterm2; PMA < 37  = 149, Full-term = 514  
We used Myers-Labonte parcellation ( https://doi.org/10.1093/cercor/bhae047, https://github.com/myersm0/myers-labonte_parcellation/)

### 📌 Acknowledgements
Data were provided by the developing Human Connectome Project, KCL-Imperial-Oxford Consortium funded by the European Research Council under the European Union Seventh Framework Programme (FP/2007-2013) / ERC Grant Agreement no. [319456]. We are grateful to the families who generously supported this trial.


## Prerequisites
- already preprocessed data with the atlas from Myers et al. (2024) (otherwise adaptations must be made in the code, including the hard coding of ROI number as well as the assignment of ROI to functional networks and their aggregation)
- 514GB RAM (if rs-fMRI data >= 283 ROI x 2300 time points) because of large data and heavy calculations 
### Data structure
```
project_path/
│
├── data/                         # not existing in this repository because of data security
│   ├── time_series/
│   │   └── sub*/
│   │       └── ses*
│   nnsi01.txt                    --> dhcp dataset with demographics
│   network_info.csv              --> what is included in which network, https://github.com/myersm0/myers-labonte_parcellation/ + added column of own clustering of these networks (see Clustering Table)
│   parcel_assignments            --> which parcel belongs to which network, https://github.com/myersm0/myers-labonte_parcellation/
│
├── scripts/
│   ├── create_labels.py
│   ├── depc_visualization.py
│   ├── epc_script_faster.py
│   ├── k10.py
│   └── splines.py
│
├── results_better/
│   └── figures/
│   
├── requirements_comet.txt
├── poster.pdf
├── .gitignore
├── README.md
└── LICENSE
```
# 📊 Methods
- usage of already preprocessed data with 283 ROI (Myers et al., 2024) https://doi.org/10.1093/cercor/bhae047) by Leonardo Zaggia (https://github.com/leonardozaggia/dHCP_NSP)
- k-means clustering to identify edge communities (script: k10.py)
- consensus clustering to aggregate different edge community solutions in one (create_labels.py)
- calculation of the dePC (epc_script_faster.py)
- heatmaps of the dePC per edge for each group (fullterm, preterm1, preterm2) (depc_visualization.py)
- Spline regressions to explore the associations of gestational age at scan on dePC (Model 1) and birth age on dePC (Model 2) for each network cluster (splines.py)

## 🚀 Work flow:
1. Preprocess your data in time-series for all ROI
2. k10.py:
  - content: clustering of edge timeseries to identify edge communities, 70 runs with 5 different centroid initializations
  - input: *nnsi01.txt* (demographic data of the participants), time_series/sub-*/ses-*/*.txt (time-series data)
  - output: k10_all_labels_ordered_10.npy (the cluster solutions for all runs of k-means), k10_sse_ordered_10.npy (sum of squared errors for each clustering solution), k10_co_ma_ordered_10.npy (co-occurence matrix)
3. create_labels.py:
  - content: aggregation of all 70 solution to one by consensus clustering, using hierarchical clustering
  - input: k10_co_ma_ordered_10.npy (co-occurence matrix)
  - output: k10_consensus_ordered_10.csv (final clustering solution)
4. epc_script_faster.py
  - content: calculates the ePC for each participant, for edge for different time windows (--> dePC) + ePC per edge (for replication of Fang et al. (2025))
  - input: k10_consensus_ordered_10.csv, *nnsi01.txt*
  - output: dePC*.csv (time-windows x edges)
5. depc_visualization.py
  - content: visualizes the mean std of dePC per edge for Preterm1, Preterm2 and Full-term as heat map, sorted after network clusters
  - input: dePC*.csv, *network_info.csv, parcel_assignment, nnsi01.txt*
  - output: *mean_depc_agg, *std_depc_agg, *mean_epc_agg, *std_epc_agg (heat maps for each group - std of dePC, mean of dePC)
6. splines.py
  - content: visualizes std dePC over gestational age and birth age at scan as spline regressions for each network cluster and over all networks + dePC over scan age - birth age over all networks 
  - input: dePC*.csv, *network_info.csv, parcel_assignment, nnsi01.txt*
  - output: 3_cluster_spline_within_between_6plots, cluster_spline_within_between_6plots_2, spline*

# 🎯Results
👉 **[Have a look at my poster (pdf):](poster.pdf)**

## Clustering Table
This table shows how we aggregated the networks from Myers et al. (2024) into functional similar clusters for visualization purposes
| Cluster | Networks |
|--------|----------|
| **0** | Unassigned |
| **1** | Posterior default mode, Anterior default mode, Superior precuneus, Inferior precuneus, Anterior precuneus |
| **2** | Posterior fronto-parietal, Anterior fronto-parietal, Inferior fronto-parietal, Precentral fronto-parietal, Lateral orbito-frontal |
| **3** | Ventral attention, Dorsal attention |
| **4** | Posterior premotor, Anterior premotor, Auditory, MotorHand, MotorMouth |
| **5** | Medial orbito-frontal, Salience, Cingulate, Insula |
| **6** | Medial visual, Lateral visual |


## 💡Replication
We did not aim to fully replicate Fang et al. (2025), and several methodological differences likely contributed to discrepancies in EPC values. This highlights that EPC may be sensitive to pipeline choices and motivates future work on the robustness of dynamic EPC.
### Methodological differences
1. Different parcellation of the cortex
2. k-means clustering of edge time series on subsamples of all data (including balanced preterm and full-term data) and different parameters
3. no individual adaptation of clustering centroids for each subject

### ePC Implementation
The ePC computation used in this project is based on a custom Python implementation.
The original code used in Fang et al. (2025) used a MATLAB function and existing Python implementations were too slow for large neonatal datasets.
To enable efficient computation of ePC and dynamic ePC, I translated the MATLAB logic into Python and optimized it for vectorized operations.
To ensure correctness, I verified that the Python implementation produces identical results to the MATLAB version on a test matrix.
The implementation is openly available in epc_script_faster.py (function: participation_coef) to support reuse.

This project is released under the MIT License to support open reuse, transparency, and reproducibility.


## ✉️ Contact 
I'm happy for any questions or suggestions on the topic! :brain::smile: 
- inga.henrike.zimmermann@uni-oldenburg.de
