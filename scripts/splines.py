import numpy as np
import pandas as pd
import glob
import re
import matplotlib.pyplot as plt
import seaborn as sns
import patsy
import statsmodels.api as sm
import time
import os

# Function to calculate within and between cluster means for a given subject's dePC standard deviation values
def within_between(pc, idx, parcels):
    # indicies of upper triangle of the matrix
    iu = np.triu_indices(283, k=1)
    node = np.zeros((283, 283))
    # filling the upper triangle with the values from combined_depc (mapping edges to node x node matrix)
    node[iu] = pc
    # building the full symmetric matrix
    node = node + node.T
    np.fill_diagonal(node, 0)
    # sorting the matrices after brain regions
    node_sorted = node[idx][:, idx]
    network_labels_sorted = parcels[idx]
    # differentiating between within and between cluster edges
    within_mask  = network_labels_sorted[:, None] == network_labels_sorted[None, :]
    between_mask = network_labels_sorted[:, None] != network_labels_sorted[None, :]
    within  = node_sorted[within_mask]
    between = node_sorted[between_mask]
    # calculating the mean of within and between cluster edges for this subject
    within_mean  = np.mean(within)
    between_mean  = np.mean(between)
    cluster_within = []
    cluster_between = []
    for c in np.unique(parcels): 
        # Within: both nodes in the same cluster c
        w_mask = (network_labels_sorted[:, None] == c) & (network_labels_sorted[None, :] == c) 
        # Between: one node in cluster c and the other node not in cluster c 
        b_mask = (network_labels_sorted[:, None] == c) & (network_labels_sorted[None, :] != c) | (network_labels_sorted[:, None] != c) & (network_labels_sorted[None, :] == c) 
        w_vals = node_sorted[w_mask] 
        b_vals = node_sorted[b_mask] 
        cluster_within.append(w_vals.mean()) 
        cluster_between.append(b_vals.mean())
    return within_mean, between_mean, cluster_within, cluster_between


start = time.time()
# load parcel assignments and network info
parcels = np.loadtxt('/dss/work/supe4945/data/parcel_assignments.txt')
networks = pd.read_csv('/dss/work/supe4945/data/network_info.csv')
# map parcel assignments from meyers atlas to network assignments (e.g. visual network, default mode network, etc - similar to Yeo et al. 2011)
n_id = networks['network_id']
n_id_new = networks['my_assignment_1'] 
for i in range(len(n_id)):
    parcels[parcels == n_id[i]] = n_id_new[i]
idx = np.argsort(parcels)


# load all depc file names
files = glob.glob("/dss/work/supe4945/results_better/dePC*.csv")
# selecting session IDs from file names
ids = [] 
for f in files: 
    m = re.search(r"ses-(\d+)", f) 
    if m: 
        ids.append(m.group(1))  
    
# load demographics
df = pd.read_csv("/dss/work/supe4945/data/nnsi01.txt", sep=None, engine="python")
# select only the rows with the session IDs we have depc data for
df_sel = df[df["scan_validation"].isin(ids)].copy()

df_sel['nscan_ga_at_scan_weeks'] = pd.to_numeric( df_sel['nscan_ga_at_scan_weeks'], errors='coerce' )
step1 = time.time()
print(f"Step 1 took {step1-start} seconds", flush = True)

mean_std = np.zeros(len(files))
mean_std_within = np.zeros(len(files))
mean_std_between = np.zeros(len(files))
mean_std_late_scan = []
mean_std_late_scan_within = []
mean_std_late_scan_between = []
cluster_within = np.zeros((len(files), 7))
cluster_between = np.zeros((len(files), 7))
count = 0
for ses in df_sel["scan_validation"]:
    depc_std = np.zeros(39903)
    depc_mean = np.zeros(39903)
    # loading the data of the session IDs
    file1 = glob.glob(f"/dss/work/supe4945/results_better/dePC*ses-{ses}.csv")
    depc = pd.read_csv(file1[0], header=None)
    # calculating std over timepoints for each edge
    depc_std[:] = np.array(np.std(depc, axis=1)) 
    # calculating mean over timepoints for each edge
    depc_mean[:] = np.array(np.mean(depc, axis=1))
    # calculating within and between cluster means for this subject
    mean_std_within[count], mean_std_between[count], cluster_within[count,:], cluster_between[count,:] = within_between(depc_std, idx, parcels)
    # calculating the mean of std over timepoints for each edge
    mean_std[count] = np.mean(np.array(np.std(depc, axis=1)) )
    count += 1

# selecting only the subjects with scans at fullterm equivalent age (after 37 weeks) for the second model    
mean_std_late_scan = mean_std[pd.to_numeric(df_sel['nscan_ga_at_scan_weeks']) >= 37]
mean_std_late_scan_within = mean_std_within[pd.to_numeric(df_sel['nscan_ga_at_scan_weeks']) >= 37]
mean_std_late_scan_between = mean_std_between[pd.to_numeric(df_sel['nscan_ga_at_scan_weeks']) >= 37]
cluster_mean_std_late_scan_within = cluster_within[pd.to_numeric(df_sel['nscan_ga_at_scan_weeks']) >= 37]
cluster_mean_std_late_scan_between = cluster_between[pd.to_numeric(df_sel['nscan_ga_at_scan_weeks']) >= 37]
df_not_prem = df_sel[pd.to_numeric(df_sel['nscan_ga_at_scan_weeks']) >= 37]
df_not_prem['nscan_ga_at_birth_weeks'] = pd.to_numeric(df_not_prem["nscan_ga_at_birth_weeks"])


# setting of font size for all plots
plt.rcParams['font.size'] = 16         
plt.rcParams['axes.titlesize'] = 20     
plt.rcParams['axes.labelsize'] = 18     
plt.rcParams['xtick.labelsize'] = 16    
plt.rcParams['ytick.labelsize'] = 16  

# pt to mm
mm_to_pt = 72/25.4
# linewidth settings
SPLINE_MM = 0.80   
AXES_MM   = 0.6  
TICK_MM   = 0.6  
GRID_MM   = 0.4 

plt.rcParams.update({
    "lines.linewidth": SPLINE_MM * mm_to_pt,
    "axes.linewidth": AXES_MM * mm_to_pt,
    "xtick.major.width": TICK_MM * mm_to_pt,
    "ytick.major.width": TICK_MM * mm_to_pt,
    "xtick.minor.width": TICK_MM * mm_to_pt,
    "ytick.minor.width": TICK_MM * mm_to_pt,
    "grid.linewidth": GRID_MM * mm_to_pt
}) 


step2 = time.time()

print(f"Step 2 took {step2-step1} seconds", flush = True)

# creating the design matrix for the first model (scan age at scan) with 4 knots at 29, 32, 37 and 40 weeks and degree of 1 (linear spline)
scan_age_t = patsy.dmatrix("bs(scan_age, knots=(29,32,37,40), degree=1)",
                           data={"scan_age":df_sel['nscan_ga_at_scan_weeks']},
                           return_type='dataframe')
# Fit the model
model = sm.OLS(mean_std,scan_age_t)
model_fit = model.fit()
print(model_fit.summary(),flush = True)
step3 = time.time()
print(f"Step 3 took {step3-step2} seconds", flush = True)

# Create evenly spaced values to plot the model predictions
xp = np.linspace(df_sel['nscan_ga_at_scan_weeks'].min(), df_sel['nscan_ga_at_scan_weeks'].max(), 100)
# Transform the evenly spaced values using the same spline basis as the model
xp_trans = patsy.dmatrix("bs(xp, knots=(29,32,37,40), degree=1)",
                         data={"xp": xp},
                         return_type='dataframe')
# Use model fitted before to predict values for generated age values
predictions = model_fit.predict(xp_trans)

# Plot the original data and the model fit
fig, ax = plt.subplots(figsize=(8,5))                         

sns.scatterplot(x=df_sel['nscan_ga_at_scan_weeks'], y=mean_std, alpha=0.4, ax=ax)  
ax.axvline(29, linestyle='--', alpha=0.4, color="black")        # Cut point 1
ax.axvline(32, linestyle='--', alpha=0.4, color="black")        # Cut point 2
ax.axvline(37, linestyle='--', alpha=0.4, color="black")        # Cut point 3
ax.axvline(40, linestyle='--', alpha=0.4, color="black")        # Cut point 4
ax.plot(xp, predictions, color='red')                           # Draw prediction
ax.set_ylabel("Mean (temporal) SD of dePC across edges")
ax.set_title("First order spline regression")

mean_path = os.path.join("/dss/work/supe4945/results_better/figures", "spline.pdf")
plt.savefig(mean_path, format="pdf", bbox_inches="tight")
plt.close()


# same for the second model: birth age (only for participants whos scan-age is at fullterm equivalent age):
# Create evenly spaced values to plot the model predictions
birth_age_t = patsy.dmatrix("bs(birth_age, knots=(29,32,37,40), degree=1)",
                           data={"birth_age":df_not_prem['nscan_ga_at_birth_weeks']},
                           return_type='dataframe')
# Fit the model
model = sm.OLS(mean_std_late_scan,birth_age_t)
model_fit = model.fit()
print(model_fit.summary(),flush = time)
xp = np.linspace(df_not_prem['nscan_ga_at_birth_weeks'].min(), df_not_prem['nscan_ga_at_birth_weeks'].max(), 100)
xp_trans = patsy.dmatrix("bs(xp, knots=(29,32,37,40), degree=1)",
                         data={"xp": xp},
                         return_type='dataframe')
predictions = model_fit.predict(xp_trans)

# Plot the original data and the model fit
fig, ax = plt.subplots(figsize=(8,5))                           

sns.scatterplot(x=df_not_prem['nscan_ga_at_birth_weeks'], y=mean_std_late_scan, alpha=0.4, ax=ax)   
ax.axvline(29, linestyle='--', alpha=0.4, color="black")        # Cut point 1
ax.axvline(32, linestyle='--', alpha=0.4, color="black")        # Cut point 2
ax.axvline(37, linestyle='--', alpha=0.4, color="black")        # Cut point 3
ax.axvline(40, linestyle='--', alpha=0.4, color="black")        # Cut point 4
ax.set_ylabel("Mean (temporal) SD of dePC across edges")
ax.plot(xp, predictions, color='red')                           # Draw prediction
ax.set_title("First order spline regression on scans at fulltime equivalent age")

mean_path = os.path.join("/dss/work/supe4945/results_better/figures", "spline2.pdf")
plt.savefig(mean_path, format="pdf", bbox_inches="tight")
plt.close()


# same for the third model (scan age - birth age -> exposure time exutero)
df_sel["scan-birth"] = df_sel["nscan_ga_at_scan_weeks"] - pd.to_numeric(df_sel["nscan_ga_at_birth_weeks"])
scan_birth_age_t = patsy.dmatrix("bs(scan_birth_age, knots=(0,4,8,12), degree=1)",
                           data={"scan_birth_age":df_sel['scan-birth']},
                           return_type='dataframe')
# Fit the model
model = sm.OLS(mean_std,scan_birth_age_t) 
model_fit = model.fit()
print(model_fit.summary(),flush = True)
step3 = time.time()
print(f"Step 3 took {step3-step2} seconds", flush = True)

# Create evenly spaced values to plot the model predictions
xp = np.linspace(df_sel['scan-birth'].min(), df_sel['scan-birth'].max(), 100)
xp_trans = patsy.dmatrix("bs(xp, knots=(0,4,8,12), degree=1)",
                         data={"xp": xp},
                         return_type='dataframe')
# Use model fitted before to predict values for generated age values
predictions = model_fit.predict(xp_trans)

# Plot the original data and the model fit
fig, ax = plt.subplots(figsize=(8,5))                           # Create figure object

sns.scatterplot(x=df_sel['scan-birth'], y=mean_std, alpha=0.4, ax=ax)   # Plot observations
ax.axvline(0, linestyle='--', alpha=0.4, color="black")        # Cut point 1
ax.axvline(4, linestyle='--', alpha=0.4, color="black")        # Cut point 2
ax.axvline(8, linestyle='--', alpha=0.4, color="black")        # Cut point 3
ax.axvline(12, linestyle='--', alpha=0.4, color="black")        # Cut point 4
ax.set_ylabel("Mean (temporal) SD of dePC across edges")
ax.plot(xp, predictions, color='red')                           # Draw prediction
ax.set_title("First order spline regression")

mean_path = os.path.join("/dss/work/supe4945/results_better/figures", "spline3.pdf")
plt.savefig(mean_path, format="pdf", bbox_inches="tight")
plt.close()


# same for the fourth model: scan age for within vs between cluster means
scan_age_t = patsy.dmatrix("bs(scan_age, knots=(29,32,37,40), degree=1)",
                           data={"scan_age":df_sel['nscan_ga_at_scan_weeks']},
                           return_type='dataframe')
# Fit the model
model_within = sm.OLS(mean_std_within, scan_age_t).fit() 
model_between = sm.OLS(mean_std_between, scan_age_t).fit()

print(model_within.summary(),flush = time)
print(model_between.summary(),flush = time)
step3 = time.time()

# Create evenly spaced values to plot the model predictions
xp = np.linspace(df_sel['nscan_ga_at_scan_weeks'].min(), df_sel['nscan_ga_at_scan_weeks'].max(), 100)
xp_trans = patsy.dmatrix("bs(xp, knots=(29,32,37,40), degree=1)",
                         data={"xp": xp},
                         return_type='dataframe')
# Use model fitted before to predict values for generated age values
pred_within = model_within.predict(xp_trans) 
pred_between = model_between.predict(xp_trans)

# Plot the original data and the model fit
fig, ax = plt.subplots(figsize=(8,5))  

# Scatter for both within and between cluster means
sns.scatterplot(x=df_sel['nscan_ga_at_scan_weeks'], y=mean_std_within, alpha=0.4, ax=ax, label="Within") 
sns.scatterplot(x=df_sel['nscan_ga_at_scan_weeks'], y=mean_std_between, alpha=0.4, ax=ax, label="Between") 
# Spline curves for both within and between cluster means
ax.plot(xp, pred_within, color='red', label="Within spline") 
ax.plot(xp, pred_between, color='blue', label="Between spline") 
# Knot lines
for k in [29, 32, 37, 40]: ax.axvline(k, linestyle='--', alpha=0.4, color="black") 
ax.set_ylabel("Mean (temporal) SD of dePC across edges") 
ax.set_title("First order spline regression: Within vs Between") 
ax.legend(loc="upper left") 
plt.savefig("/dss/work/supe4945/results_better/figures/spline_within_between.pdf", format="pdf", bbox_inches="tight") 
plt.close()


# fifth model: birth age for late scans within vs between
birth_age_t = patsy.dmatrix("bs(birth_age, knots=(29,32,37,40), degree=1)",
                           data={"birth_age":df_not_prem['nscan_ga_at_birth_weeks']},
                           return_type='dataframe')
# Fit the model
model_within = sm.OLS(mean_std_late_scan_within, birth_age_t).fit() 
model_between = sm.OLS(mean_std_late_scan_between, birth_age_t).fit()

print(model_within.summary(),flush = time)
print(model_between.summary(),flush = time)
step3 = time.time()

# Create evenly spaced values to plot the model predictions
xp = np.linspace(df_not_prem['nscan_ga_at_birth_weeks'].min(), df_not_prem['nscan_ga_at_birth_weeks'].max(), 100)
xp_trans = patsy.dmatrix("bs(xp, knots=(29,32,37,40), degree=1)",
                         data={"xp": xp},
                         return_type='dataframe')
# Use model fitted before to predict values for generated age values
pred_within = model_within.predict(xp_trans) 
pred_between = model_between.predict(xp_trans)

# Plot the original data and the model fit
fig, ax = plt.subplots(figsize=(8,5))  

# Scatter for both within and between cluster means
sns.scatterplot(x=df_not_prem['nscan_ga_at_birth_weeks'], y=mean_std_late_scan_within, alpha=0.4, ax=ax, label="Within") 
sns.scatterplot(x=df_not_prem['nscan_ga_at_birth_weeks'], y=mean_std_late_scan_between, alpha=0.4, ax=ax, label="Between") 
# Spline curves for both within and between cluster means
ax.plot(xp, pred_within, color='red', label="Within spline") 
ax.plot(xp, pred_between, color='blue', label="Between spline") 
# Knot lines
for k in [29, 32, 37, 40]: ax.axvline(k, linestyle='--', alpha=0.4, color="black") 
ax.set_ylabel("Mean (temporal) SD of dePC across edges") 
ax.set_title("First order spline regression: Within vs Between") 
ax.legend(loc="upper left") 
plt.savefig("/dss/work/supe4945/results_better/figures/spline2_within_between.pdf", format="pdf", bbox_inches="tight") 
plt.close()


# 6 Subplots for within vs between cluster means for each of the 6 clusters (excluding cluster 0 which is not assigned to any network) for the first model (scan age at scan):
fig, axes = plt.subplots(1, 6, figsize=(30, 5))
axes = axes.flatten()

parcels = parcels.astype(int)
clusters = np.unique(parcels)

for i, c in enumerate(clusters):
    if c == 0:
        continue
    ax = axes[i-1]

    # data for this cluster
    y_within  = cluster_within[:, c]      
    y_between = cluster_between[:, c]

    # design matrix for scan age at scan
    scan_age_t = patsy.dmatrix(
        "bs(scan_age, knots=(29,32,37,40), degree=1)",
        data={"scan_age": df_sel['nscan_ga_at_scan_weeks']},
        return_type='dataframe'
    )

    # Models
    model_w = sm.OLS(y_within, scan_age_t).fit()
    model_b = sm.OLS(y_between, scan_age_t).fit()

    # Predictions
    xp = np.linspace(df_sel['nscan_ga_at_scan_weeks'].min(),
                     df_sel['nscan_ga_at_scan_weeks'].max(), 100)

    xp_trans = patsy.dmatrix(
        "bs(xp, knots=(29,32,37,40), degree=1)",
        data={"xp": xp},
        return_type='dataframe'
    )

    pred_w = model_w.predict(xp_trans)
    pred_b = model_b.predict(xp_trans)

    # labels
    ax.plot(xp, pred_w, color="#00786B", label="Within")
    ax.plot(xp, pred_b, color="#D53D0E", label="Between")

    sns.scatterplot(x=df_sel['nscan_ga_at_scan_weeks'], y=y_within, alpha=0.2, ax=ax) 
    sns.scatterplot(x=df_sel['nscan_ga_at_scan_weeks'], y=y_between, alpha=0.2, ax=ax) 

    # Knot lines
    for k in [29, 32, 37, 40]:
        ax.axvline(k, linestyle='--', alpha=0.3, color="black")

    ax.set_title(f"Cluster {c}")
    ax.set_xlabel("Scan age (weeks)")
    ax.set_ylabel("Mean SD of dePC")

    y_min = min(cluster_within.min(), cluster_between.min()) * 0.95
    y_max = 0.03 
    ax.set_ylim(y_min, y_max)

    ax.legend(loc="upper left")

# optimize layout
plt.tight_layout()

plt.savefig("/dss/work/supe4945/results_better/figures/3_cluster_spline_within_between_6plots.pdf",
            format="pdf", bbox_inches="tight")

plt.close()


# Model 7: same as model 6 but for birth age at birth for late scans (>= 37 weeks)
# 6 Subplots: 
fig, axes = plt.subplots(1, 6, figsize=(30,5)) # 1,6  30,5
axes = axes.flatten()

parcels = parcels.astype(int)
clusters = np.unique(parcels)

for i, c in enumerate(clusters):
    if c == 0:
        continue
    ax = axes[i-1]

    # data for this cluster
    y_within  = cluster_mean_std_late_scan_within[:, c]      
    y_between = cluster_mean_std_late_scan_between[:, c]

    # Designmatrix
    birth_age_t = patsy.dmatrix(
        "bs(birth_age, knots=(29,32,37,40), degree=1)",
        data={"birth_age": df_not_prem['nscan_ga_at_birth_weeks']},
        return_type='dataframe'
    )

    # Models
    model_w = sm.OLS(y_within, birth_age_t).fit()
    model_b = sm.OLS(y_between, birth_age_t).fit()

    # Predictions
    xp = np.linspace(df_not_prem['nscan_ga_at_birth_weeks'].min(),
                     df_not_prem['nscan_ga_at_birth_weeks'].max(), 100)

    xp_trans = patsy.dmatrix(
        "bs(xp, knots=(29,32,37,40), degree=1)",
        data={"xp": xp},
        return_type='dataframe'
    )

    pred_w = model_w.predict(xp_trans)
    pred_b = model_b.predict(xp_trans)

    # Labels
    ax.plot(xp, pred_w, color="#00786B", label="Within")
    ax.plot(xp, pred_b, color="#D53D0E", label="Between")
    sns.scatterplot(x=df_not_prem['nscan_ga_at_birth_weeks'], y=y_within, alpha=0.2, ax=ax) 
    sns.scatterplot(x=df_not_prem['nscan_ga_at_birth_weeks'], y=y_between, alpha=0.2, ax=ax) 
    # Knot lines
    for k in [29, 32, 37, 40]:
        ax.axvline(k, linestyle='--', alpha=0.3, color="black")

    ax.set_title(f"Cluster {c}")
    ax.set_xlabel("Birth age (weeks)")
    ax.set_ylabel("Mean SD of dePC")

    y_min = min(cluster_mean_std_late_scan_within.min(), cluster_mean_std_late_scan_between.min()) * 0.95
    y_max = 0.030 
    ax.set_ylim(y_min, y_max)

    ax.legend(loc="upper left")

# optimize layout
plt.tight_layout()

plt.savefig("/dss/work/supe4945/results_better/figures/cluster_spline_within_between_6plots_2.pdf",
            format="pdf", bbox_inches="tight")

plt.close()