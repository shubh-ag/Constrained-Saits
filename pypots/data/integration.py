"""
Integrate with data functions from other libraries.
"""

# Created by Wenjie Du <wenjay.du@gmail.com>
# License: GLP-v3


import numpy as np
import pycorruptor as corruptor
from tsdb import (
    pickle_load as _pickle_load,
    pickle_dump as _pickle_dump,
)

pickle_load = _pickle_load
pickle_dump = _pickle_dump


def cal_missing_rate(X):
    return corruptor.cal_missing_rate(X)


def masked_fill(X, mask, val):
    return corruptor.masked_fill(X, mask, val)


def mcar(X, rate, nan=0):
    return corruptor.mcar(X, rate, nan)


def mcar_feature(X, feature_index, rate, nan=0):
    original_shape = X.shape
    # X = X.flatten()
    # print(original_shape, X.shape)
    X_intact = np.copy(X)  # keep a copy of originally observed values in X_intact
    # select random indices for artificial mask
    indices = X.shape[1]
    # print(indices)
    indices = np.random.choice(indices, int(indices * rate), replace=False)
    # create artificially-missing values by selected indices
    X[:, indices, feature_index] = np.nan  # mask values selected by indices
    indicating_mask = ((~np.isnan(X_intact)) ^ (~np.isnan(X))).astype(np.float32)
    missing_mask = (~np.isnan(X)).astype(np.float32)
    X_intact = np.nan_to_num(X_intact, nan=nan)
    X = np.nan_to_num(X, nan=nan)
    # reshape into time-series data
    # X_intact = X_intact.reshape(original_shape)
    # X = X.reshape(original_shape)
    # missing_mask = missing_mask.reshape(original_shape)
    # indicating_mask = indicating_mask.reshape(original_shape)

    # print(X_intact.shape, missing_mask.shape, indicating_mask.shape)
    return X_intact, X, missing_mask, indicating_mask
