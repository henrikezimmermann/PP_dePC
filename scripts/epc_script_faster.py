import os
# set number of threads for numpy operations
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

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

import re
from natsort import natsorted, ns

# function to calculate ePC
# W: eFC matrix for one window, Ci: cluster labels
def participation_coef(W, Ci):
    start = time.time()
    # sets negative values to zero - ensures weights are non-negative and use float32 for speed / lower memory
    W = np.clip(W,0.0,None).astype(np.float64)
    # preallocation and conversion of cluster labels to int to lower memory usage
    Ci = np.array(Ci, dtype=int)
    n = W.shape[0]
    # sum of connection weights per edge
    ko = W.sum(axis=1)
    kc2 = np.zeros(n, dtype=np.float64)
    for m in np.unique(Ci):
        # selects edges belonging to cluster m
        mask = (Ci==m)
        # sum over squared community-wise strengths per edge
        kc2 += (W[:, mask].sum(axis=1))**2
    # Only compute participation where ko != 0 to avoid division by zero
    mask = ko>0
    p = np.zeros(n,dtype=np.float64)
    # compute participation coefficient by normalizing community-wise strengths with total edge strengths
    p[mask] = 1 - kc2[mask] / (ko[mask]**2)
    p = np.nan_to_num(p, nan=0.0, posinf=1.0, neginf=0.0)
    p = np.clip(p, 0.0, 1.0)
    end = time.time()
    #print(f"ePC done once for a window! {end-start} seconds", flush = True)
    return p

# function to caluclate eFC and then ePC
# W: eTS matrix for one window, klabels: cluster labels
def dynamic_ePC(eTS,klabels):
    start = time.time()
    E, T = eTS.shape
    # normalize the eTS matrix
    A = eTS.T.astype(dtype=np.float64)
    A-= A.mean(axis=0, keepdims = True)
    # compute norms for each column
    norms = np.linalg.norm(A, axis = 0)
    # handle zero norms safely
    norms_safe = np.where(norms == 0, 1.0, norms)
    # normalize the matrix
    A_norm = A/norms_safe
    # compute eFC as the dot product of normalized eTS
    eFC = A_norm.T @ A_norm
    # set diagonal to zero
    np.fill_diagonal(eFC, 0.0) # or 1.0?
    # compute ePC
    ePC = participation_coef(eFC,klabels)
    end = time.time()
    #print(f"dePC done for a window! {end-start} seconds", flush = True)
    return ePC

# extracts the session number from the file path
def extract_session(path):
    match = re.search(r"ses-[^/]+", path)
    return match.group(0) if match else "ses-unknown"

# load all data
start = time.time()
files = glob.glob("/dss/work/supe4945/data/time_series/sub-*/ses-*/*.txt")
# sort the files after file path
files = natsorted(files, alg=ns.PATH)
all_sessions = []
# create empty array to store the data (participants, regions, timepoints)
all_data = np.empty((int(len(files)/283),283,2300), dtype=np.float64)
count = 0
for fil in range(0,len(files),283):  
    sin_data = np.empty((283,2300),dtype=np.float64)
    # extract session number
    all_sessions.append(extract_session(files[fil]))
    count2 = 0
    # load data for each participant (283 regions)
    for f in files[fil:fil+283]:
        arr = np.loadtxt(f, delimiter="\t")
        sin_data[count2,:] = arr
        count2+=1
    all_data[count,:,:] = sin_data
    count+=1
print(f"The data that was loaded has the shape {all_data.shape}", flush = True)

# load cluster labels
klabels = pd.read_csv("/dss/work/supe4945/results_better/k10_consensus_ordered_10.csv", header=0).iloc[:,0].tolist()

# calculate dePC for each participant

window_size = 2300 # for dynamic ePC = 100, for static = 2300
step_size = 50
for i in range(all_data.shape[0]):
    dePC = [] 
    # suppress print from comet function ets
    with suppress_stdout():
            eTS = comet.connectivity.EdgeConnectivity(all_data[i,:,:].T, 'eTS' ).estimate().T
    # calculate eTS and dePC for each window
    for ws in range(0,all_data.shape[2]-window_size+1,step_size):
        # calculation of ePC for one window
        ePC = dynamic_ePC(eTS[:,ws:ws+window_size],klabels)
        dePC.append(ePC)

    dePC_all = np.stack(dePC, axis=1)
    np.savetxt(f"/dss/work/supe4945/results_better/ePC{i}_{all_sessions[i]}.csv", dePC_all, delimiter=",")
    mid_end = time.time()
    print(f"The ePC (float64) for one participant took {mid_end-start} seconds", flush = True)
end = time.time()
print(f"The whole epc_script_faster for all participants (float64 and new ets calculation) took {end-start} seconds", flush= True)