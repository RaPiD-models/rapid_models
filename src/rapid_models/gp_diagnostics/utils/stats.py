import itertools
from scipy.stats import norm
import numpy as np


def snorm_qq(x):
    """
    Function for calculating standard normal QQ plot data with 95% confidence.

    Based on extRemes.qqnorm in R. https://rdrr.io/cran/extRemes/man/qqnorm.html

    Args:
        x (array): data in 1D array

    Returns:
        q_sample (array): sample quantiles
        q_snorm (array): standard normal quantiles
        q_snorm_upper (array): 95% upper band
        q_snorm_lower (array): 95% lower band

    For plotting:
        x = q_snorm, q_snorm_upper, q_snorm_lower (Standard Normal Quantiles)
        y = q_sample (Sample Quantiles)

        Example:
        fig, ax = plt.subplots()
        ax.scatter(q_snorm, q_sample)
        ax.plot(q_snorm_upper, q_sample, 'k--')
        ax.plot(q_snorm_lower, q_sample, 'k--')
    """

    n = len(x)  # Number of data points

    # Sample quantiles
    q_sample = np.sort(x)

    # Cumulative probabilities used to extract quantiles
    p = (np.arange(n) + 0.5) / n

    # Theoretical quantiles
    q_snorm = norm.ppf(p)

    # Confidence intervals are calculated using +/- k, where
    k = 0.895 / (np.sqrt(n) * (1 - 0.01 / np.sqrt(n) + 0.85 / n))

    q_snorm_upper = norm.ppf(p + k)
    q_snorm_lower = norm.ppf(p - k)

    return q_sample, q_snorm, q_snorm_upper, q_snorm_lower


def split_test_train_fold(folds, X, i):
    """
    Split X into X_train, X_test where

    Args:
        folds (list of lists): The index subsets.
        X (array_like): The indexed object to split
        i (int): Split on the indices folds[i]

    Returns:
        X_test = X[folds[i]]
        X_train = the rest
    """

    idx_test = folds[i]
    idx_train = list(itertools.chain(*(folds[0:i] + folds[i + 1 :])))

    return X[idx_test], X[idx_train]
