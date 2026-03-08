import numpy as np
import pandas as pd
import time
from sklearn.cluster import AgglomerativeClustering

def create_consensus_clustering(co_occurrence, n_clusters_mode, linkage='average', 
                                min_stability=0.0):
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
    return consensus_labels_filtered, stability_scores

co_ma = np.load('/dss/work/supe4945/results_better/k10_co_ma_ordered_10.npy')
con_labels, stab_scores = create_consensus_clustering(co_ma, 10)
df_items = pd.DataFrame({
    "consensus_label": con_labels,
    "stability_score": stab_scores
})
df_items.to_csv(f"/dss/work/supe4945/results_better/k10_consensus_ordered_10.csv", index=False)

