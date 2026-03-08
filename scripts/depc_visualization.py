import numpy as np
import pandas as pd
import glob
import re
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle
import os
from scipy.stats import ttest_ind
from statsmodels.stats.multitest import multipletests
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap

# function to visualize mean dePC over time and participants + average std of timepoints over participants for every edge in a heatmap (nodes x nodes matrix)
# input: list of session IDs belonging to the group (ses_data), parcel assignment (parcels), group term for title of heatmap (term)
def sort_br(ses_data, idx):
    # perallocation of dePC mean and std matrices for all participants
    depc_std = np.zeros((39903, len(ses_data)))
    depc_mean = np.zeros((39903, len(ses_data)))
    epc = np.zeros((39903,len(ses_data)))
    
    count = 0
    # loading the data of the session IDs
    for ses in ses_data:
        files = glob.glob(f"/dss/work/supe4945/results_better/dePC*ses-{ses}.csv")
        files_ePC = glob.glob(f"/dss/work/supe4945/results_better/ePC*ses-{ses}.csv")
        depc = pd.read_csv(files[0], header=None)
        # calcukating std over timepoints for each edge
        depc_std[:,count] = np.array(np.std(depc, axis=1)) 
        # calculating mean over timepoints for each edge
        depc_mean[:,count] = np.array(np.mean(depc, axis=1))
        # add epc of participant
        epc[:,count] = pd.read_csv(files_ePC[0], header=None).squeeze()
        count += 1
    # taking the average of mean and std over participants
    combined_depc_std = np.mean(depc_std, axis=1) 
    combined_depc_mean = np.mean(depc_mean, axis=1)
    combined_epc_mean = np.mean(epc, axis=1)
    # preallocation of node x node matrices
    node_std = np.zeros((283, 283))
    node_mean = np.zeros((283, 283))
    epc_node_mean = np.zeros((283,283))

    # indicies of upper triangle of the matrix
    iu = np.triu_indices(283, k=1)
    # filling the upper triangle with the values from combined_depc (mapping edges to node x node matrix)
    node_std[iu] = combined_depc_std
    node_mean[iu] = combined_depc_mean
    epc_node_mean[iu] = combined_epc_mean

    # building the full symmetric matrix
    node_std = node_std + node_std.T
    node_mean = node_mean + node_mean.T
    epc_node_mean = epc_node_mean + epc_node_mean.T
    np.fill_diagonal(node_std, 0)
    np.fill_diagonal(node_mean, 0)
    np.fill_diagonal(epc_node_mean, 0)

    # sorting the matrices after brain regions
    m_mean_sorted = node_mean[idx][:, idx]
    m_std_sorted = node_std[idx][:, idx]
    epc_mean_sorted = epc_node_mean[idx][:, idx]

    return m_mean_sorted, m_std_sorted, epc_mean_sorted, depc_std

def plot_depc(m_mean_sorted, m_std_sorted, idx, term, min_mean, max_mean,min_std, max_std,parcels, typ):
    # plotting the mean dePC heatmap
    plt.figure(figsize=(8, 8))
    plt.imshow(m_mean_sorted, cmap=cmap_custom, vmin=min_mean, vmax=max_mean)
    if typ == 1:
        plt.colorbar(label="Mean SD of dePC")
        plt.title(f"{term}")
    elif typ == 2:
         plt.colorbar(label='p-value')
         plt.title(f"P-values {term}")
    else:
        plt.colorbar(label='mean ePC')
        plt.title(f"Mean ePC {term}")
    plt.xlabel("nodes - sorted after clusters")
    plt.ylabel("nodes - sorted after clusters")

    # calculating sizes of the groups for drawing squares
    group_sizes = np.bincount(parcels[idx].astype(int))
    boundaries = np.cumsum(group_sizes)

    # calculating start positions of the groups
    starts = np.concatenate(([0], boundaries[:-1]))

    # squares around the brain regions
    for start, size in zip(starts, group_sizes):
        rect = Rectangle(
            (start - 0.5, start - 0.5),   # (x, y)
            size,                        # width
            size,                        # height
            fill=False,
            edgecolor="black",
            linewidth=3.0
        )
        plt.gca().add_patch(rect)

    if typ == 1:
        mean_path = os.path.join("/dss/work/supe4945/results_better/figures", f"{term[:8]}_mean_depc_agg.pdf")
    elif typ == 2:
        mean_path = os.path.join("/dss/work/supe4945/results_better/figures", f"{term}_pvalues.pdf")
    else:
        mean_path = os.path.join("/dss/work/supe4945/results_better/figures", f"{term[:8]}_mean_epc_agg.pdf") 
    plt.savefig(mean_path, format="pdf", bbox_inches="tight")
    plt.close()

    # same for std heatmap
    if typ == 1:
        plt.figure(figsize=(8, 8))
        # changing thickness for colorbar axes
        im = plt.imshow(m_std_sorted, cmap=cmap_custom, vmin=min_std, vmax=max_std)
        cbar = plt.colorbar(im, ax=plt.gca(), label="SD of dePC")
        [cbar.ax.spines[s].set_linewidth(1.7) for s in cbar.ax.spines]   
        cbar.ax.tick_params(width=1.7)  
        plt.title(f"{term}")                  
        plt.xlabel("nodes - sorted after clusters")
        plt.ylabel("nodes - sorted after clusters")
        # squares around the brain regions
        for start, size in zip(starts, group_sizes):
            rect = Rectangle(
                (start - 0.5, start - 0.5),   # (x, y)
                size,                        # width
                size,                        # height
                fill=False,
                edgecolor="black",
                linewidth=3.0
            )
            plt.gca().add_patch(rect)

        mean_path = os.path.join("/dss/work/supe4945/results_better/figures", f"{term[:8]}_std_depc_agg.pdf")
        plt.savefig(mean_path, format="pdf", bbox_inches="tight")
        plt.close()

# setting of font size for all plots
plt.rcParams['font.size'] = 18          
plt.rcParams['axes.titlesize'] = 22     
plt.rcParams['axes.labelsize'] = 20     
plt.rcParams['xtick.labelsize'] = 18   
plt.rcParams['ytick.labelsize'] = 18    

# mm to pt
mm_to_pt = 72/25.4 
AXES_MM   = 0.6  
# changing thickness of axes
plt.rcParams.update({
    "axes.linewidth": AXES_MM * mm_to_pt
}) 

# create own colormap
cmap_custom = LinearSegmentedColormap.from_list(
    "my_cmap",
    ["#00786B", "#F5E9D3", "#D53D0E"]
)

# load parcel assignments and network info
parcels = np.loadtxt('/dss/work/supe4945/data/parcel_assignments.txt')
networks = pd.read_csv('/dss/work/supe4945/data/network_info.csv')
# load all depc file names
files = glob.glob("/dss/work/supe4945/results_better/dePC*.csv")
# selecting session IDs from file names
ids = [] 
for f in files: 
    m = re.search(r"ses-(\d+)", f) 
    if m: 
        ids.append(m.group(1))  
    
# load denographics
df = pd.read_csv("/dss/work/supe4945/data/nnsi01.txt", sep=None, engine="python")
# select only the rows with the session IDs we have depc data for
df_sel = df[df["scan_validation"].isin(ids)]   
# divide into preterm and fullterm groups
preterm = df_sel[pd.to_numeric(df_sel["nscan_ga_at_birth_weeks"]) < 37] 
fullterm = df_sel[pd.to_numeric(df_sel["nscan_ga_at_birth_weeks"]) >= 37]

# divide preterms by age at scan
pt2 = df_sel[(pd.to_numeric(df_sel["nscan_ga_at_birth_weeks"]) < 37) & (pd.to_numeric(df_sel['nscan_ga_at_scan_weeks']) < 37)] #df_sel['fetal_age'<259]]
pt1 = df_sel[(pd.to_numeric(df_sel["nscan_ga_at_birth_weeks"]) < 37) & (pd.to_numeric(df_sel['nscan_ga_at_scan_weeks']) >= 37)] #df_sel['fetal_age']>=259]

# map parcel assignments from meyers atlas to network assignments (e.g. visual network, default mode network, etc - similar to Yeo et al. 2011)
n_id = networks['network_id']
n_id_new = networks['my_assignment_1']

for i in range(len(n_id)):
    parcels[parcels == n_id[i]] = n_id_new[i]

# plotting the heatmaps for preterm and fullterm groups (node x node matrices)
idx = np.argsort(parcels)
iu = np.triu_indices(283, k=1)
mean_p, std_p, epc_mean_p, std_dePC_p = sort_br(preterm['scan_validation'], idx) 
mean_f, std_f, epc_mean_f, std_dePC_f = sort_br(fullterm["scan_validation"], idx)
min_mean = min(np.min(mean_p[iu]),np.min(mean_f[iu]))
max_mean = max(np.max(mean_p[iu]),np.max(mean_f[iu]))
min_std = min(np.min(std_p[iu]),np.min(std_f[iu]))
max_std = max(np.max(std_p[iu]),np.max(std_f[iu]))
epc_min_mean = min(np.min(epc_mean_p[iu]),np.min(epc_mean_f[iu]))
epc_max_mean = max(np.max(epc_mean_p[iu]),np.max(epc_mean_f[iu]))
plot_depc(mean_p,std_p,idx,"preterm", min_mean, max_mean,min_std, max_std,parcels,1)
plot_depc(mean_f,std_f,idx,"fullterm", min_mean, max_mean,min_std, max_std,parcels,1)


# plotting pt1 and pt2
mean_pt1, std_pt1, epc_mean_pt1, std_dePC_pt1 = sort_br(pt1['scan_validation'], idx) 
mean_pt2, std_pt2, epc_mean_pt2, std_dePC_pt2 = sort_br(pt2["scan_validation"], idx)
min_mean_pt = min(np.min(mean_pt1[iu]),np.min(mean_pt2[iu]),min_mean)
max_mean_pt = max(np.max(mean_pt1[iu]),np.max(mean_pt2[iu]),max_mean)
min_std_pt = min(np.min(std_pt1[iu]),np.min(std_pt2[iu]),min_std)
max_std_pt = max(np.max(std_pt1[iu]),np.max(std_pt2[iu]),max_std)
epc_min_mean_pt = min(np.min(epc_mean_pt1[iu]),np.min(epc_mean_pt2[iu]),epc_min_mean)
epc_max_mean_pt = max(np.max(epc_mean_pt1[iu]),np.max(epc_mean_pt2[iu]),epc_max_mean)
plot_depc(mean_pt1,std_pt1,idx,"Preterm1 (scanned PMA >= 37)", min_mean_pt, max_mean_pt,min_std_pt, max_std_pt,parcels,1)
plot_depc(mean_pt2,std_pt2,idx,"Preterm2 (scanned PMA < 37)", min_mean_pt, max_mean_pt,min_std_pt, max_std_pt,parcels,1)
plot_depc(mean_p,std_p,idx,"preterm_all", min_mean_pt, max_mean_pt,min_std_pt, max_std_pt,parcels,1)
plot_depc(mean_f,std_f,idx,"Full-term", min_mean_pt, max_mean_pt,min_std_pt, max_std_pt,parcels,1)

plot_depc(epc_mean_pt1,0,idx,"preterm_group1", epc_min_mean_pt, epc_max_mean_pt,0, 0,parcels,0)
plot_depc(epc_mean_pt2,0,idx,"preterm_group2", epc_min_mean_pt, epc_max_mean_pt,0, 0,parcels,0)
plot_depc(epc_mean_p,0,idx,"preterm_all", epc_min_mean_pt, epc_max_mean_pt,0, 0,parcels,0)
plot_depc(epc_mean_f,0,idx,"Full-term", epc_min_mean_pt, epc_max_mean_pt,0, 0,parcels,0)


print(f"Shape ft: {std_dePC_f.shape}, Shape pt1: {std_dePC_pt1.shape}, Shape pt2: {std_dePC_pt2.shape}", flush = True)

# ttest test between pt1, pt2 and ft for mean dePC
from scipy.stats import ttest_ind
n_edges = std_dePC_pt1.shape[0]
p_vals_f1 = np.zeros(n_edges)
p_vals_f2 = np.zeros(n_edges)
p_vals_12 = np.zeros(n_edges)

for i in range(n_edges):
    _, pf1 = ttest_ind(std_dePC_f[i], std_dePC_pt1[i], equal_var=False) 
    p_vals_f1[i] = pf1
    _, pf2 = ttest_ind(std_dePC_f[i], std_dePC_pt2[i], equal_var=False) 
    p_vals_f2[i] = pf2
    _, p12 = ttest_ind(std_dePC_pt1[i], std_dePC_pt2[i], equal_var=False) 
    p_vals_12[i] = p12

# fdr correction for multiple comparisons
from statsmodels.stats.multitest import multipletests
reject_f1, p_fdr_f1, _, _ = multipletests(p_vals_f1, alpha = 0.001,method='fdr_by')
reject_f2, p_fdr_f2, _, _ = multipletests(p_vals_f2, alpha = 0.001,method='fdr_by')
reject_12, p_fdr_12, _, _ = multipletests(p_vals_12, alpha = 0.01,method='fdr_by')

def build_node_matrix(p_fdr, idx):
    # preallocation of node x node matrix
    node_sig = np.zeros((283, 283))
    # indicies of upper triangle of the matrix
    iu = np.triu_indices(283, k=1)
    # filling the upper triangle with the values from combined_depc (mapping edges to node x node matrix)
    node_sig[iu] = p_fdr
    # building the full symmetric matrix
    node_sig = node_sig + node_sig.T
    # sorting the matrices after brain regions
    m_sig_sorted = node_sig[idx][:, idx]
    return m_sig_sorted
#min_p = min(np.min(p_fdr_f1),np.min(p_fdr_f2),np.min(p_fdr_12))
#max_p = max(np.max(p_fdr_f1),np.max(p_fdr_f2),np.max(p_fdr_12))
sig_matrix_f1 = build_node_matrix(reject_f1.astype(int), idx)
sig_matrix_f2 = build_node_matrix(reject_f2.astype(int), idx)
sig_matrix_12 = build_node_matrix(reject_12.astype(int), idx)
plot_depc(sig_matrix_f1, 0, idx, "Pt1 vs Ft (significant edges)", 0, 1, 0, 0, parcels, 2) 
plot_depc(sig_matrix_f2, 0, idx, "Pt2 vs Ft (significant edges)", 0, 1, 0, 0, parcels, 2) 
plot_depc(sig_matrix_12, 0, idx, "Pt1 vs Pt2 (significant edges)", 0, 1, 0, 0, parcels, 2)



min_mean = min(np.min(mean_p[iu]),np.min(mean_f[iu]))
max_mean = max(np.max(mean_p[iu]),np.max(mean_f[iu]))
min_std = min(np.min(std_p[iu]),np.min(std_f[iu]))
max_std = max(np.max(std_p[iu]),np.max(std_f[iu]))
epc_min_mean = min(np.min(epc_mean_p[iu]),np.min(epc_mean_f[iu]))
epc_max_mean = max(np.max(epc_mean_p[iu]),np.max(epc_mean_f[iu]))
plot_depc(mean_p,std_p,idx,"preterm", min_mean, max_mean,min_std, max_std,parcels,1)
plot_depc(mean_f,std_f,idx,"fullterm", min_mean, max_mean,min_std, max_std,parcels,1)

