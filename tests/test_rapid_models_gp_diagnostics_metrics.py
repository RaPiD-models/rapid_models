# import pytest
import numpy as np
from scipy import stats

from rapid_models.gp_diagnostics.metrics import (log_prob_normal,
                                                 log_prob_standard_normal)


def test_log_prob_normal():
    """
    test that log_prob_normal agrees with scipy.stats
    """

    # Generate some data
    np.random.seed(12)
    N = 17

    C = np.random.uniform(size=(N, N))
    C = C.dot(C.T)
    Y = np.random.multivariate_normal(mean=np.zeros(N), cov=C)

    L = np.linalg.cholesky(C)

    # Compute likelihood
    loglik = log_prob_normal(L, Y)
    loglik_scipy = stats.multivariate_normal.logpdf(Y, mean=np.zeros(N), cov=C)

    assert isinstance(loglik, float)
    assert np.allclose(loglik, loglik_scipy)


def test_log_prob_standard_normal():
    """
    test that log_prob_standard_normal agrees with scipy.stats
    """

    # Generate some data
    np.random.seed(12)
    N = 16

    Y = np.random.multivariate_normal(mean=np.zeros(N), cov=np.eye(N))

    # Compute likelihood
    loglik = log_prob_standard_normal(Y)
    loglik_scipy = stats.multivariate_normal.logpdf(
        Y, mean=np.zeros(N), cov=np.eye(N))  # type: ignore

    assert np.allclose(loglik, loglik_scipy)
