import numpy as np

from rapid_models.gp_diagnostics.cv import (
    multifold_cholesky,
    loo_cholesky,
    try_chol,
    check_numeric_array,
    check_folds_indices,
    check_lower_triangular,
)
from rapid_models.gp_diagnostics.utils.linalg import triang_solve


def evaluate_GP(K, Y_train, folds=None, noise_variance=0, check_args=True):
    """
    Compute a set of evaluation metrics for GP regression with noiseless
    (noise_variance = 0) or fixed variance iid Gaussian noise.

    Specify the list 'folds' of indices for multifold cross-validation (see
    documentation for cv.multifold), otherwise leave-one-out is assumed.

    Args:
        K (2d array): GP prior covariance matrix
        Y_train (array): training observations
        folds (list of lists): The index subsets for multifold cross-validation. Folds = None -> Leave-one-out
        noise_variance: variance of the observational noise. Set noise_variance = 0 for noiseless observations

        check_args (bool): Check (assert) that arguments are well-specified before computation

    Returns: a dict containing
        log_marginal_likelihood: The log probability of Y_train
        log_pseudo_likelihood: The log 'pseudo' likelihood is the sum of the
        log probabilities of each observation during cross-validation in the
        standard normal space
        RMSE: The root mean squared error obtained by using the GP posterior mean as a deterministic prediction

        (The residuals are also returned, for plotting and to check for normality)
        residuals_mean: Mean of CV residuals
        residuals_var: Variance of CV residuals
        residuals_transformed: The residuals transformed to the standard normal space
    """

    # Check arguments
    if check_args:
        check_numeric_array(
            Y_train, 1, "Y_train"
        )  # Check that Y_train is a 1d numeric array
        check_numeric_array(K, 2, "K")  # Check that K is a 2d array
        assert (
            K.shape[0] == Y_train.shape[0] and K.shape[1] == Y_train.shape[0]
        ), "The size of K {} is not compatible with Y_train {}".format(
            K.shape, Y_train.shape
        )  # Check that K has correct size
        assert (
            noise_variance >= 0
        ), "noise_variance must be non-negative"  # Check that the noise variance is non-negative

        if folds is not None:
            check_folds_indices(
                folds, Y_train.shape[0]
            )  # Check that the list of index subsets (list of lists) is valid

    # Try to compute the lower triangular cholesky factor
    L = try_chol(K, noise_variance, "evaluate_GP")
    if L is None:
        return None

    # Compute metrics and return
    return evaluate_GP_cholesky(L, Y_train, folds, check_args=False)


def evaluate_GP_cholesky(L, Y_train, folds=None, check_args=True):
    """
    This is called by evaluate_GP() with the appropriate Cholesky factor: LL^T = K + np.eye(K.shape[0])*noise_variance
    """

    # Check that arguments are ok
    if check_args:
        check_lower_triangular(L, "L")  # Check that L is a lower triangular matrix
        check_numeric_array(
            Y_train, 1, "Y_train"
        )  # Check that Y_train is a 1d numeric array
        if folds is not None:
            check_folds_indices(
                folds, Y_train.shape[0]
            )  # Check that the list of index subsets (list of lists) is valid

    res = {}

    # Compute CV residuals
    mean, cov, residuals_transformed = (
        multifold_cholesky(L, Y_train, folds, False)
        if folds is not None
        else loo_cholesky(L, Y_train, False)
    )

    # Compute log marginal likelihood
    res["log_marginal_likelihood"] = log_prob_normal(L, Y_train)

    # Compute log pseudo likelihood - This is the log probability of the residuals in the standard normal space
    # The sum of log probabilities of each observation in Y_train in the
    # posterior GP, assuming that the corresponding fold has been left out,
    # can be obtained by adding -0.5*log(det(cov)) = -log(chol(cov).diagonal()).sum() to res['log_pseudo_likelihood']
    res["log_pseudo_likelihood"] = log_prob_standard_normal(residuals_transformed)

    # Compute MSE
    res["MSE"] = np.linalg.norm(mean)

    # Append residuals
    res["residuals_mean"] = mean
    res["residuals_var"] = cov.diagonal()
    res["residuals_transformed"] = residuals_transformed

    return res


def log_prob_normal(L, Y):
    """
    Compute log probability of the data Y under an unbiased Gaussian with covariance L*L^T
    """
    a = triang_solve(L, Y)  # La = Y
    return (
        -(1 / 2) * np.linalg.norm(a) ** 2
        - np.log(L.diagonal()).sum()
        - (Y.shape[0] / 2) * np.log(2 * np.pi)
    )


def log_prob_standard_normal(Y):
    """
    Compute log probability of the data Y under an unbiased standard Gaussian
    """
    return -(1 / 2) * np.linalg.norm(Y) ** 2 - (Y.shape[0] / 2) * np.log(2 * np.pi)
