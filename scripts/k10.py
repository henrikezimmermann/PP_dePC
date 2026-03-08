import os
os.environ["OPENBLAS_NUM_THREADS"] = "4"
os.environ["OMP_NUM_THREADS"] = "4"
os.environ["MKL_NUM_THREADS"] = "4"

import sys
from contextlib import contextmanager
# supresses print from comet function ets
@contextmanager
def suppress_stdout():
    with open(os.devnull, "w") as devnull:
        old_stdout = sys.stdout
        sys.stdout = devnull
        try:
            yield
        finally:
            sys.stdout = old_stdout

import numpy as np
import comet
from comet.connectivity import EdgeConnectivity
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_samples, silhouette_score
import glob
from scipy.stats import zscore
import pandas as pd
import time
from sklearn.metrics import adjusted_rand_score
from collections import defaultdict, Counter
from sklearn.cluster import AgglomerativeClustering
import random
from natsort import natsorted, ns

# # check if the data has the correct format (283 txt files for each participants, 2300 time points for each txt file)
# ses_dirs = glob.glob("/dss/work/supe4945/data/time_series/sub-*/ses-*")

# for ses in ses_dirs:
#     txt_files = glob.glob(os.path.join(ses, "*.txt"))
#     count = len(txt_files)

#     if count != 283:
#         print(f"{ses} hat {count} Dateien")

#     for f in txt_files: 
#         arr = np.loadtxt(f, delimiter="\t") 
#         if arr.shape[0] != 2300: 
#             print(f"FEHLER: {ses}, {f} hat Länge {arr.shape[0]}")


def calculate_cluster_stability(all_labels, n_items):
    start = time.time()
    """
    Calculate stability metrics for clustering across multiple runs.
    
    Returns:
        - co_occurrence_matrix: n_items x n_items matrix of co-clustering frequency
        - pairwise_ari: pairwise Adjusted Rand Index between runs
    """
    n_runs = len(all_labels)
    co_occurrence = np.zeros((n_items, n_items), dtype=np.float32)
    
    # Build co-occurrence matrix
    for labels in all_labels:
        for i in range(n_items):
            for j in range(i, n_items):
                # Items co-cluster if they have same cluster label and not noise (-1)
                if labels[i] >= 0 and labels[i] == labels[j]:
                    co_occurrence[i, j] += 1
                    if i != j:
                        co_occurrence[j, i] += 1
    
    # Normalize by number of runs
    co_occurrence /= n_runs
 
    # Calculate pairwise ARI between all runs
  #  ari_scores = []
  #  for i in range(n_runs):
  #      for j in range(i + 1, n_runs):
  #          ari = adjusted_rand_score(all_labels[i], all_labels[j])
  #          ari_scores.append(ari)
    end = time.time()
    print(f"calculate_cluster_stability without ari score calculation took {end-start} seconds", flush = True)
  #  return co_occurrence, ari_scores
    return co_occurrence



def create_consensus_clustering(co_occurrence, n_clusters_mode, linkage='average', 
                                min_stability=0.5):
    """
    Create a consensus clustering from the co-occurrence matrix.
    
    Uses hierarchical clustering on the co-occurrence matrix (treated as similarity)
    to create a stable clustering solution.
    
    Parameters:
    -----------
    co_occurrence : np.ndarray
        n_items x n_items matrix of co-clustering frequencies
    n_clusters_mode : int
        The most common number of clusters found across runs
    linkage : str
        Linkage method for hierarchical clustering ('average', 'complete', 'single')
    min_stability : float
        Minimum co-occurrence frequency to consider items as stably clustered together
        
    Returns:
    --------
    consensus_labels : np.ndarray
        Cluster labels for the consensus clustering
    stability_scores : np.ndarray
        Per-item stability scores (mean co-occurrence with cluster members)
    """
    start = time.time()
    n_items = co_occurrence.shape[0]
    
    # Convert co-occurrence (similarity) to distance
    # High co-occurrence = low distance
    distance_matrix = 1.0 - co_occurrence
    
    # Perform hierarchical clustering
    clustering = AgglomerativeClustering(
        n_clusters=n_clusters_mode,
        metric='precomputed',
        linkage=linkage
    )
    consensus_labels = clustering.fit_predict(distance_matrix)
    
    # Calculate stability scores for each item
    stability_scores = np.zeros(n_items)
    for i in range(n_items):
        # Items in the same cluster
        cluster_members = np.where(consensus_labels == consensus_labels[i])[0]
        # Mean co-occurrence with cluster members (excluding self)
        cluster_members = cluster_members[cluster_members != i]
        if len(cluster_members) > 0:
            stability_scores[i] = co_occurrence[i, cluster_members].mean()
        else:
            stability_scores[i] = 1.0  # Singleton cluster
    
    # Mark items with low stability as noise (-1)
    # These are items that don't consistently cluster with their assigned group
    low_stability_mask = stability_scores < min_stability
    consensus_labels_filtered = consensus_labels.copy()
    consensus_labels_filtered[low_stability_mask] = -1
    end = time.time()
    print(f"create_consensus_clustering took {end-start} seconds", flush = True)
    print(consensus_labels_filtered, flush = True)
    print(stability_scores, flush = True)
    return consensus_labels_filtered, stability_scores




start = time.time()
# load demographics
df = pd.read_csv("/dss/work/supe4945/data/nnsi01.txt", sep=None, engine="python")
# important columns
df_reg_var = df[['nscan_age_at_scan_days','fetal_age','src_subject_id','scan_validation','sex','nscan_ga_at_birth_weeks']]

# load all data paths
files = glob.glob("/dss/work/supe4945/data/time_series/sub-*/ses-*/*.txt")
# sort them after file path
files = natsorted(files, alg=ns.PATH)

# extract ses-* filenames
ses_dirs = [os.path.basename(os.path.dirname(f)) for f in files]
# delete ses- 
ses_nums = [s.replace("ses-", "") for s in ses_dirs]
# filter demographics of only those participants we use the data from
df_sel = df_reg_var[df_reg_var['scan_validation'].isin(ses_nums)]
# divide data in pre-and fullterm -> order in which data and eTS are saved later!
preterm = df_sel[pd.to_numeric(df_sel["nscan_ga_at_birth_weeks"]) < 37] 
fullterm = df_sel[pd.to_numeric(df_sel["nscan_ga_at_birth_weeks"]) >= 37] 
step1 = time.time()
print(f"The process of selecting data took {step1-start} seconds",flush=True)

# saves data for preterm - participant x file (ROI) x tineseries
all_pre = np.empty((len(preterm),283,2300),dtype=np.float32)
for i in range (len(preterm)):
    files1 = glob.glob(f"/dss/work/supe4945/data/time_series/{preterm.iloc[i]['src_subject_id']}/ses-{preterm.iloc[i]['scan_validation']}/*.txt")
    # sorts files of each participant after ROI number
    files1 = natsorted(files1, alg=ns.PATH)
    count = 0
    for f in files1:
        arr = np.loadtxt(f, delimiter="\t")
        all_pre[i,count,:] = arr
        count+=1

# saves data for fullterm - participant x file (ROI) x tineseries
all_full = np.empty((len(fullterm),283,2300),dtype=np.float32)
for j in range (len(fullterm)):
    files2 = glob.glob(f"/dss/work/supe4945/data/time_series/{fullterm.iloc[j]['src_subject_id']}/ses-{fullterm.iloc[j]['scan_validation']}/*.txt")
    # sorts files of each participant after ROI number
    files2 = natsorted(files2, alg=ns.PATH)
    count = 0
    for f in files2:
        arr = np.loadtxt(f, delimiter="\t")
        all_full[j,count,:] = arr
        count+=1

step2 = time.time()
print(f"The process of loading data and dividing it took {step2-step1} seconds",flush=True) 

# eTS for all preterm
ets_pre = np.empty(((len(preterm),39903,2300)),dtype=np.float32)
for each_ts in range(all_pre.shape[0]):
    with suppress_stdout():
        ets_pre[each_ts,:,:] = comet.connectivity.EdgeConnectivity(all_pre[each_ts,:,:].T,'eTS').estimate().T
# eTS for all fullterm
ets_full = np.empty(((len(fullterm),39903,2300)),dtype=np.float32)
for each_ts in range(all_full.shape[0]):
    with suppress_stdout():
        ets_full[each_ts,:,:] = comet.connectivity.EdgeConnectivity(all_full[each_ts,:,:].T,'eTS').estimate().T
step3 = time.time()
print(f"The process of eTS took {step3-step2} seconds",flush=True) 

# random sampling of 1/10 of the data for each group to perform k-means clustering on a subsample -> prevent overfitting + reduce computational load
all_labels = []
sse = []
for i in range(70):  # 70 runs of k-means
    sample_pre = ets_pre[random.sample(range(0,len(preterm)), int(np.round(len(preterm)/10))),:,:]
    sample_full = ets_full[random.sample(range(0,len(fullterm)), int(np.round(len(fullterm)/10))),:,:]

    all_ets = []
    # concatenate pre-and fullterm data - shape: (edges, participants*timpoints)
    all_ets = np.concatenate([sample_pre, sample_full], axis=0) 
    all_ets = all_ets.transpose(1, 0, 2) 
    all_ets = all_ets.reshape(39903, -1)
    print(f"eTS has shape {all_ets.shape}", flush = True)

    # k-means clustering with k=10 and random state = i (to have different initial centroids for each run)
    k_beg = time.time()
    kmeans = KMeans(n_clusters=10, random_state=i, n_init = 5).fit(all_ets)
    # save labels and sse for each run for later calculations
    all_labels.append(kmeans.labels_)
    sse.append(kmeans.inertia_)
    k_end = time.time()
    print(f"The process of kmeans (k=10), with a subsamples of 1/10 data and n_init = 5 took {k_end-k_beg} seconds",flush=True) 

np.save("/dss/work/supe4945/results_better/k10_all_labels_ordered_10.npy", all_labels)
np.save("/dss/work/supe4945/results_better/k10_sse_ordered_10.npy", np.array(sse))
# calculate co-occurrence matrix and pairwise ARI
# co_ma, ari = calculate_cluster_stability(all_labels,39903)
co_ma = calculate_cluster_stability(all_labels,39903)
# con_labels, stab_scores = create_consensus_clustering(co_ma, 10)

step4 = time.time()
print(f"The process of kmeans (k=10), with 70 runs of random subsamples of 1/10 data and n_init = 5 took {step4-step3} seconds",flush=True) 


# df_items = pd.DataFrame({
#     "consensus_label": con_labels,
#     "stability_score": stab_scores
# })
# df_items.to_csv(f"/dss/work/supe4945/results/k10_consensus.csv", index=False)
np.save(f"/dss/work/supe4945/results_better/k10_co_ma_ordered_10.npy", co_ma.astype(np.float16))

#df_ari = pd.DataFrame({"ari_scores": ari})
#df_ari.to_csv(f"/dss/work/supe4945/results/ari_k10_ordered_10.csv", index=False)

end = time.time()
print(f"The whole script took {end-start} seconds",flush=True) 
