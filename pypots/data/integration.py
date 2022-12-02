"""
Integrate with data functions from other libraries.
"""

# Created by Wenjie Du <wenjay.du@gmail.com>
# License: GLP-v3


import numpy as np
import pycorruptor as corruptor


def cal_missing_rate(X):
    return corruptor.cal_missing_rate(X)


def masked_fill(X, mask, val):
    return corruptor.masked_fill(X, mask, val)


def mcar(X, rate, nan=0):
    return corruptor.mcar(X, rate, nan)


def mcar_sample_feature(X, feature_index, rate, nan=0):
    X_intact = np.copy(X)  # keep a copy of originally observed values in X_intact
    # select random indices for artificial mask
    samples = X.shape[0]
    steps = X.shape[1]
    sample_indices = np.random.choice(samples, int(samples * rate), replace=False)

    # create artificially-missing values by selected indices
    for i in sample_indices:
        step_indices = np.random.choice(steps, int(steps * rate), replace=False)
        for j in step_indices:
            X[i, j, feature_index] = np.nan  # mask values selected by indices

    indicating_mask = ((~np.isnan(X_intact)) ^ (~np.isnan(X))).astype(np.float32)
    missing_mask = (~np.isnan(X)).astype(np.float32)
    X_intact = np.nan_to_num(X_intact, nan=nan)
    X = np.nan_to_num(X, nan=nan)

    return X_intact, X, missing_mask, indicating_mask


def mcar_sample_all(X, feature_index, rate, nan=0):
    X_intact = np.copy(X)  # keep a copy of originally observed values in X_intact
    # select random indices for artificial mask
    samples = X.shape[0]
    steps = X.shape[1]
    sample_indices = np.random.choice(samples, int(samples * rate), replace=False)

    # create artificially-missing values by selected indices
    for i in sample_indices:
        step_indices = np.random.choice(steps, int(steps * rate), replace=False)
        for j in step_indices:
            X[i, j, :] = np.nan  # mask values selected by indices

    indicating_mask = ((~np.isnan(X_intact)) ^ (~np.isnan(X))).astype(np.float32)
    missing_mask = (~np.isnan(X)).astype(np.float32)
    X_intact = np.nan_to_num(X_intact, nan=nan)
    X = np.nan_to_num(X, nan=nan)

    return X_intact, X, missing_mask, indicating_mask
