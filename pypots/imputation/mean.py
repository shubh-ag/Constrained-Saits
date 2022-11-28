"""
Implementation of the imputation method MEAN (Last Observed Carried Forward).
"""

# Created by Wenjie Du <wenjay.du@gmail.com>
# License: GLP-v3

import warnings

import numpy as np
import torch

from pypots.imputation.base import BaseImputer


class Mean(BaseImputer):
    """MEAN (Last Observed Carried Forward) imputation method.

    Attributes
    ----------
    nan : int/float
        Value used to impute data missing at the beginning of the sequence.
    """

    def __init__(self, nan=0):
        super().__init__("cpu")
        self.nan = nan

    def fit(self, train_X, val_X=None):
        warnings.warn(
            "MEAN (Last Observed Carried Forward) imputation class has no parameter to train. "
            "Please run func impute(X) directly."
        )

    def mean_numpy(self, X):
        """Numpy implementation of MEAN.

        Parameters
        ----------
        X : np.ndarray,
            Time series containing missing values (NaN) to be imputed.

        Returns
        -------
        X_imputed : array,
            Imputed time series.

        Notes
        -----
        This implementation gets inspired by the question on StackOverflow:
        https://stackoverflow.com/questions/41190852/most-efficient-way-to-forward-fill-nan-values-in-numpy-array
        """
        n_samples, n_steps, n_features = X.shape

        mean = np.nanmean(X, axis=1)
        std = np.nanstd(X, axis=1)

        for i in range(n_samples):
            for j in range(n_steps):
                for k in range(n_features):
                    if np.isnan(X[i][j][k]):
                        X[i][j][k] = mean[i][k] + 10*std[i][k]

        if np.isnan(X).any():
            X = np.nan_to_num(X, nan=self.nan)

        return X

    def impute(self, X):
        """Impute missing values

        Parameters
        ----------
        X : array-like,
            Time-series vectors containing missing values (NaN).

        Returns
        -------
        array-like,
            Imputed time series.
        """
        assert len(X.shape) == 3, (
            f"Input X should have 3 dimensions [n_samples, n_steps, n_features], "
            f"but the actual shape of X: {X.shape}"
        )
        if isinstance(X, list):
            X = np.asarray(X)

        if isinstance(X, np.ndarray):
            X_imputed = self.mean_numpy(X)
        elif isinstance(X, torch.Tensor):
            X_imputed = self.mean_numpy(X.detach().cpu().numpy())
        else:
            raise TypeError(
                "X must be type of list/np.ndarray/torch.Tensor, " f"but got {type(X)}"
            )
        return X_imputed