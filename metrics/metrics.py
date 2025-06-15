import numpy as np
import matplotlib.pyplot as plt
import math

from pyDRMetrics.pyDRMetrics import DRMetrics
from sklearn.neighbors import NearestNeighbors
from typing import Tuple


# -------------
# Basic metrics
# -------------

# NOTE: pyDRMetrics provides a ready implementation for most of basic DR metrics
# - https://github.com/zhangys11/pyDRMetrics

# NOTE: in all of the following functions, X represents original data, and Z represents reduced data (after applying dimensionality reduction)

# Co‑k-nearest neighbour size
# - Measures the proportion of shared k-nearest neighbors between the original high-dimensional data and its low-dimensional embedding
# - Returns metric values to each of k nearest neighbors (k values)
def metric_qnn(X, Z, k: int) -> np.ndarray:
    return DRMetrics(X, Z).QNN[:k]

# Co‑k-nearest neighbour AUC
# - Returns the area under the curve defined by the previous metric
def metric_qnn_auc(X, Z) -> float:
    return DRMetrics(X, Z).AUC

# Trustworthiness
# - Measures how well the k-nearest neighbor relationships in the low-dimensional embedding preserve the original high-dimensional neighborhoods 
#    by penalizing points that appear as neighbors in the embedding but are not neighbors in the original space
def metric_trustworthiness(X, Z, k: int) -> np.ndarray:
    return DRMetrics(X, Z).T[:k]

# Trustworthiness AUC
# - Returns the area under the curve defined by the previous metric
def metric_trustworthiness_auc(X, Z) -> float:
    return DRMetrics(X, Z).AUC_T

# Residual variance (Pearson)
# - Quantifies how much of the original global distance structure is lost after dimensionality reduction, 
#    with lower values indicating better preservation of overall data geometry.
def metric_residual_variance(X, Z) -> float:
    return DRMetrics(X, Z).Vr

# Residual variance (Spearman)
# - Similar to the one above, but uses Spearman correlation instead
def metric_residual_variance_spearman(X, Z) -> float:
    return DRMetrics(X, Z).Vrs


# ----------------
# Advanced metrics
# ----------------

# Class fidelity
# - Measures how well a dimensionality reduction method preserves class-based local neighborhood structure by evaluating the proportion of 
#    neighbors sharing the same class label in the reduced space.
def metric_cf(Z, labels, nmax: int = 1000) -> float:
    M = len(Z)
    nnmax = min(nnmax, M-1)
    
    # Initialize nearest neighbors model
    nn_model = NearestNeighbors(n_neighbors=nnmax+1, n_jobs=-1)
    nn_model.fit(Z)
    _, indices = nn_model.kneighbors(Z)
    
    labels = np.array(labels)
    cfnn_sum = 0
    
    # Calculate cfnn for each value of nn from 1 to nnmax
    for nn in range(1, nnmax+1):
        neighbor_labels = labels[indices[:, 1:nn+1]]   # shape (M, nn)
        matches = (neighbor_labels == labels[:, None])  # shape (M, nn), True/False
        cfnn = matches.sum() / (nn * M)
        cfnn_sum += cfnn
    
    return cfnn_sum / nnmax

# KNN-gain
# - Measures the improvement in preserving local neighborhood structure after dimensionality reduction by comparing the number of 
#    correctly retained k-nearest neighbors in the reduced space against the original space.
# - Returns both array of KNN-gain values as well as AUC as a second value
def knn_gain(X, Z, labels) -> Tuple[np.ndarray, float]:
    # Number of data points
    N = X.shape[0]
    N_1 = N - 1
    k_hd = np.zeros(shape=N_1, dtype=np.int64)
    k_ld = np.zeros(shape=N_1, dtype=np.int64)
    # For each data point
    for i in range(N):
        c_i = labels[i]
        di_hd = X[i, :].argsort(kind="mergesort")
        di_ld = Z[i, :].argsort(kind="mergesort")
        # Making sure that i is first in di_hd and di_ld
        for arr in [di_hd, di_ld]:
            for idj, j in enumerate(arr):
                if j == i:
                    idi = idj
                    break
            if idi != 0:
                arr[idi] = arr[0]
            arr = arr[1:]
        for k in range(N_1):
            if c_i == labels[di_hd[k]]:
                k_hd[k] += 1
            if c_i == labels[di_ld[k]]:
                k_ld[k] += 1

    # Computing the KNN gain
    gn = (k_ld.cumsum() - k_hd.cumsum()).astype(np.float64) / (
            (1.0 + np.arange(N_1)) * N
    )

    i_all_k = 1.0 / (np.arange(gn.size) + 1.0)
    auc = np.float64(gn.dot(i_all_k)) / (i_all_k.sum())

    # Returning the KNN gain and its AUC
    return gn, auc

# Rank-based Neighborhood Preservation Index (RNX)
# - Quantifies how well the relative ranking of neighbors is preserved after dimensionality reduction, reflecting the consistency of 
#    local neighborhood order between high- and low-dimensional spaces.
def metric_rnx(X, Z) -> Tuple[np.ndarray, float]:
    
    # A helper function
    def coranking(d_hd, d_ld):
        # Computing the permutations to sort the rows of the distance matrices in HDS and LDS.
        perm_hd = d_hd.argsort(axis=-1, kind="mergesort")
        perm_ld = d_ld.argsort(axis=-1, kind="mergesort")

        N = d_hd.shape[0]
        i = np.arange(N, dtype=np.int64)

        # Computing the ranks in the LDS
        R = np.empty(shape=(N, N), dtype=np.int64)
        for j in range(N):
            R[perm_ld[j, i], j] = i

        # Computing the co-ranking matrix
        Q = np.zeros(shape=(N, N), dtype=np.int64)
        for j in range(N):
            Q[i, R[perm_hd[j, i], j]] += 1

        # Returning
        return Q[1:, 1:]

    Q = coranking(X, Z)

    N_1 = Q.shape[0]
    N = N_1 + 1

    # Computing Q_NX
    qnxk = np.empty(shape=N_1, dtype=np.float64)
    acc_q = 0.0
    for K in range(N_1):
        acc_q += Q[K, K] + np.sum(Q[K, :K]) + np.sum(Q[:K, K])
        qnxk[K] = acc_q / ((K + 1) * N)

    # Computing R_NX
    arr_K = np.arange(N_1)[1:].astype(np.float64)
    rnxk = (N_1 * qnxk[: N_1 - 1] - arr_K) / (N_1 - arr_K)

    i_all_k = 1.0 / (np.arange(rnxk.size) + 1.0)
    auc = np.float64(rnxk.dot(i_all_k)) / (i_all_k.sum())

    return rnxk, auc