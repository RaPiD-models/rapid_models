import itertools
from typing import Any, List, Tuple, Union

import numpy as np
from nptyping import Float, Int, NDArray, Shape

import rapid_models.gp_diagnostics.utils.checks as checks
from rapid_models.gp_diagnostics.utils.linalg import (chol_inv, mulinv_solve,
                                                      triang_solve, try_chol)


def multifold(
    K: NDArray[Shape['N, N'], Float],  # noqa: F821
    Y_train: NDArray[Shape['N'], Float],  # noqa: F821
    folds: List[List[int]],
    noise_variance: float = 0.,
    check_args: bool = True,
) -> Union[Tuple[None, None, None], Tuple[
        NDArray[Shape['N'], Float],  # noqa: F821
        NDArray[Shape['N, N'], Float],  # noqa: F821
        NDArray[Shape['N'], Float]]]:  # noqa: F821
    """
    Compute multifold CV residuals for GP regression with noiseless
    (noise_variance = 0) or fixed variance iid Gaussian noise.
    (residual = observed - predicted)

    Args:
        K (2d array): GP prior covariance matrix
        Y_train (array): training observations
        folds (list of lists): The index subsets
        noise_variance: variance of the observational noise. Set noise_variance = 0 for noiseless observations

        check_args (bool): Check (assert) that arguments are well-specified before computation

    Returns:
        mean: Mean of CV residuals
        cov: Covariance of CV residuals
        residuals_transformed: The residuals transformed to the standard normal space

    This function just calls 'multifold_cholesky()' with the appropriate Cholesky factor.
    It is based on the formulation derived in:

    [D. Ginsbourger and C. Schaerer (2021). Fast calculation of Gaussian Process multiple-fold
    crossvalidation residuals and their covariances. arXiv:2101.03108]
    """

    # Check arguments
    if check_args:
        # Check that Y_train is a 1d numeric array
        check_numeric_array(Y_train, 1, "Y_train")
        # Check that K is a 2d array
        check_numeric_array(K, 2, "K")
        # Check that K has correct size
        assert K.shape[0] == Y_train.shape[0] and K.shape[1] == Y_train.shape[
            0], f"The size of K {K.shape} is not compatible with Y_train {Y_train.shape}"

        # Check that the noise variance is non-negative
        assert (noise_variance >= 0), "noise_variance must be non-negative"
        check_folds_indices(
            folds, Y_train.shape[0]
        )  # Check that the list of index subsets (list of lists) is valid

    # Try to compute the lower triangular cholesky factor
    L = try_chol(K, noise_variance, "multifold")
    if L is None:
        return None, None, None

    # Compute residuals
    return multifold_cholesky(L, Y_train, folds, False)


def multifold_cholesky(
    L: NDArray[Shape['N, N'], Float],  # noqa: F821
    Y_train: NDArray[Shape['N'], Float],  # noqa: F821
    folds: List[List[int]],
    check_args: bool = True,
) -> Tuple[NDArray[Shape['N'], Float],  # noqa: F821
           NDArray[Shape['N, N'], Float],  # noqa: F821
           NDArray[Shape['N'], Float]]:  # noqa: F821
    """
    Compute multifold CV residuals from the Cholesky factor L of the
    observation precision matrix and the training data Y_train
    (residual = observed - predicted)

    Args:
        L (2d array): lower triangular Cholesky factor of covariance matrix (L L.T = covariance matrix)
        Y_train (array): training observations
        folds (list of lists): The index subsets

        check_args (bool): Check (assert) that arguments are well-specified before computation

    Returns:
        mean: Mean of CV residuals
        cov: Covariance of CV residuals
        residuals_transformed: The residuals transformed to the standard normal space

    Note:
    * The matrix K = L L.T is the covariance matrix of the predicted observations Y_train
    * For observations including Gaussian noise with fixed variance (v), the matrix K is
    K = (K + v*I) where K[i, j] is the prior covariance of the latent GP between the i-th an j-th training location

    This implementation uses the Cholesky factor instead of the inverse precision matrix,
    but is otherwise equivalent to the formulas derived in

    [D. Ginsbourger and C. Schaerer (2021). Fast calculation of Gaussian Process multiple-fold
    crossvalidation residuals and their covariances. arXiv:2101.03108]
    """

    N_folds: int = len(folds)  # Number of folds
    N_training: int = Y_train.shape[0]  # Total number of training observations

    # Check that arguments are ok
    if check_args:
        # Check that L is a lower triangular matrix
        check_lower_triangular(L, "L")
        # Check that Y_train is a 1d numeric array
        check_numeric_array(Y_train, 1, "Y_train")
        # Check that the list of index subsets (list of lists) is valid
        check_folds_indices(folds, N_training)

    # Allocate
    D: NDArray[Shape['N, N'], Float]  # noqa: F821
    D = np.zeros(shape=(N_training, N_training))

    D_inv_mean: NDArray[Shape['N'], Float]  # noqa: F821
    D_inv_mean = np.zeros(N_training)

    mean: NDArray[Shape['N'], Float]  # noqa: F821
    mean = np.zeros(N_training)

    # We need some elements from the inverse covariance matrix
    K_inv: NDArray[Shape['N, N'], Float]  # noqa: F821
    K_inv = chol_inv(L)

    K_inv_Y: NDArray[Shape['N'], Float]  # noqa: F821
    K_inv_Y = mulinv_solve(L, Y_train)

    # Loop over each fold
    for i in range(N_folds):
        block_idx: Tuple[NDArray[Shape['Fold_length'], Int],  # noqa: F821
                         NDArray[Shape['Fold_length'], Int]]  # noqa: F821
        block_idx = np.ix_(folds[i], folds[i])

        # The cholesky factor of the i-th block
        block_chol = np.linalg.cholesky(K_inv[block_idx])
        # The inverse of the i-th block
        D[block_idx] = chol_inv(block_chol)

        # The residual mean
        mean[folds[i]] = mulinv_solve(block_chol, K_inv_Y[folds[i]])
        D_inv_mean[folds[i]] = K_inv[block_idx].dot(mean[folds[i]])

    # The covariance matrix
    alpha: NDArray[Shape['N, N'], Float]  # noqa: F821
    alpha = triang_solve(L, D)
    cov: NDArray[Shape['N, N'], Float]  # noqa: F821
    cov = alpha.T.dot(alpha)

    # The transformed residuals
    # @TODO: I feel we could a bit better harmonize the code in multifold_cholesky()
    #        with the code in loo_cholesky(), so that the comments, variable names
    #        and the flow of logic is a bit more similar / comparable.
    #        We can have a glimpse on it in a later round of refactoring :-)
    #        CLAROS, 2022-10-28
    residuals_transformed: NDArray[Shape['N'], Float]  # noqa: F821
    residuals_transformed = L.T.dot(D_inv_mean)

    return mean, cov, residuals_transformed


def loo(
    K: NDArray[Shape['N, N'], Float],  # noqa: F821
    Y_train: NDArray[Shape['N'], Float],  # noqa: F821
    noise_variance: float = 0.,
    check_args: bool = True,
) -> Union[Tuple[None, None, None], Tuple[
        NDArray[Shape['N'], Float],  # noqa: F821
        NDArray[Shape['N, N'], Float],  # noqa: F821
        NDArray[Shape['N'], Float]]]:  # noqa: F821
    """
    Compute Leave-One-Out (LOO) residuals for GP regression with noiseless
    (noise_variance = 0) or fixed variance iid Gaussian noise.
    (residual = observed - predicted)
    This function just calls 'loo_cholesky()' with the appropriate Cholesky factor.

    Args:
        K (2d array): GP prior covariance matrix
        Y_train (array): training observations
        noise_variance (float): variance of the observational noise. Set noise_variance = 0. for noiseless observations

        check_args (bool): Check (assert) that arguments are well-specified before computation

    Returns:
        mean: Mean of LOO residuals
        cov: Covariance of LOO residuals
        residuals_transformed: The residuals transformed to the standard normal space
    """

    if check_args:
        # Check that Y_train is a 1d numeric array
        check_numeric_array(Y_train, 1, "Y_train")
        # Check that K is a 2d array
        check_numeric_array(K, 2, "K")
        # Check that K has correct size
        assert K.shape[0] == Y_train.shape[0] and K.shape[1] == Y_train.shape[
            0], f"The size of K {K.shape} is not compatible with Y_train {Y_train.shape}"
        # Check that the noise variance is non-negative
        assert (noise_variance >= 0), "noise_variance must be non-negative"

    # Try to compute the lower triangular cholesky factor
    L = try_chol(K, noise_variance, "loo")

    return (None, None, None) if L is None else loo_cholesky(L, Y_train, False)


def loo_cholesky(
    L: NDArray[Shape['N, N'], Float],  # noqa: F821
    Y_train: NDArray[Shape['N'], Float],  # noqa: F821
    check_args: bool = True,
) -> Tuple[NDArray[Shape['N'], Float],  # noqa: F821
           NDArray[Shape['N, N'], Float],  # noqa: F821
           NDArray[Shape['N'], Float]]:  # noqa: F821
    """
    Compute Leave-One-Out (LOO) residuals from the Cholesky factor L of the
    observation precision matrix and the training data Y_train
    (residual = observed - predicted)

    Args:
        L (2d array): lower triangular Cholesky factor of covariance matrix (L L.T = covariance matrix)
        Y_train (array): training observations

        check_args (bool): Check (assert) that arguments are well-specified before computation

    Returns:
        mean: Mean of LOO residuals
        cov: Covariance of LOO residuals
        residuals_transformed: The residuals transformed to the standard normal space

    Note:
    * The matrix K = L L.T is the covariance matrix of the predicted observations Y_train
    * For observations including Gaussian noise with fixed variance (v), the matrix K is
    K = (K + v*I) where K[i, j] is the prior covariance of the latent GP between the i-th an j-th training location

    This implementation uses the Cholesky factor instead of the inverse
    precision matrix, but is otherwise equivalent to the formulas derived in

    [O. Dubrule. Cross validation of kriging in a unique neighborhood.
    Journal of the International Association for Mathematical Geology, 15 (6):687-699, 1983.]
    """

    # Check that arguments are ok
    if check_args:
        check_lower_triangular(
            L, "L")  # Check that L is a lower triangular matrix
        check_numeric_array(
            Y_train, 1, "Y_train")  # Check that Y_train is a 1d numeric array

    K_inv: NDArray[Shape['N, N'], Float]  # noqa: F821
    K_inv = chol_inv(L)

    var: NDArray[Shape['N'], Float]  # noqa: F821
    var = 1 / K_inv.diagonal()

    mean: NDArray[Shape['N'], Float]  # noqa: F821
    mean = mulinv_solve(L, Y_train) * var

    # Can be made a bit faster with einsum (as D is diagonal)
    D: NDArray[Shape['N, N'], Float]  # noqa: F821
    D = np.eye(var.shape[0]) * var

    # The covariance matrix
    alpha: NDArray[Shape['N, N'], Float]  # noqa: F821
    alpha = triang_solve(L, D)
    cov: NDArray[Shape['N, N'], Float]  # noqa: F821
    cov = alpha.T.dot(alpha)

    # The transformed residuals
    residuals_transformed: NDArray[Shape['N'], Float]  # noqa: F821
    residuals_transformed = L.T.dot(np.eye(var.shape[0]) * (1 / var)).dot(mean)

    return mean, cov, residuals_transformed


def check_folds_indices(
    folds: List[List[int]],
    n_max: int,
):
    """
    Check that the list of index subsets (list of lists) is valid

    Args:
        folds (list of lists): The index subsets.
        n_max (int): Total number of indices.

    Raises:
        AssertionError: if not 'folds' represents the range [0:n_max-1] of n_max indices split into
                        non overlapping subsets
    """

    assert isinstance(folds,
                      list), "'folds' has to be a list of lists of integers"
    assert all(isinstance(x, list)
               for x in folds), "'folds' has to be a list of lists of integers"
    assert [] not in folds, "'folds' has to be a list of lists of integers"

    all_elements_set = set(itertools.chain(*folds))
    assert all(
        np.issubdtype(type(x), np.integer) for x in
        all_elements_set), "'folds' has to be a list of lists of integers"
    assert all_elements_set == set(range(
        n_max)), "the indices in 'folds' has to be a partition of range(n_max)"


def check_lower_triangular(
    arr: Union[NDArray[Shape['N, N'], Float], Any],  # noqa: F821
    argname: str = "arr",
):
    """
    Check that the argument is a 2d numpy array which is lower triangular

    Args:
        arr (): object

    Raises:
        AssertionError: if not 'arr' represents a lower triangular matrix
    """
    assert checks.is_numeric_np_array(
        arr), f"{argname} must be a numpy array with numeric elements"

    assert checks.is_square(arr), f"{argname} must be a square numpy array"
    assert checks.is_lower_triang(arr), f"{argname} must be lower triangular"


def check_numeric_array(
    arr: Union[NDArray[Any, Float], Any],
    dim: int,
    argname: str = "arr",
):
    """
    Check that the argument is a numpy array of correct dimension

    Args:
        arr (): object

    Raises:
        AssertionError: if not 'arr' represents a 'dim'-dimensional numpy array
    """
    assert checks.is_numeric_np_array(
        arr), f"{argname} must be a numpy array with numeric elements"

    assert len(
        arr.shape) == dim, f"{argname} must be a {dim} dimensional array"


# @TODO: This function is nowhere used.
#        Delete?
#        CLAROS, 2022-10-28
def _multifold_inv(K, Y_train, folds):
    """
    Compute multifold cv residuals using matrix inverse (for testing)
    (residual = observed - predicted)

    Args:
        K (2d array): covariance matrix
        Y_train (array): training observations
        folds (list of lists): The index subsets.

    Returns:
        mean: Mean of CV residuals
        cov: Covariance of CV residuals
        residuals_transformed: The residuals transformed to the standard normal space

    [D. Ginsbourger and C. Schaerer (2021). Fast calculation of Gaussian Process multiple-fold
    crossvalidation residuals and their covariances. arXiv:2101.03108]
    """
    K_inv = np.linalg.inv(K)
    L = np.linalg.cholesky(K)

    N_training = Y_train.shape[0]  # Total number of training observations
    N_folds = len(folds)  # Number of folds

    K_inv_Y = K_inv.dot(Y_train)

    D = np.zeros(shape=(N_training, N_training))
    D_inv = np.zeros(shape=(N_training, N_training))
    mean = np.zeros(N_training)
    D_inv_mean = np.zeros(N_training)

    for i in range(N_folds):
        idx = np.ix_(folds[i], folds[i])

        block_inv = np.linalg.inv(K_inv[idx])
        D[idx] = block_inv
        D_inv[idx] = K_inv[idx]

        mean[folds[i]] = block_inv.dot(K_inv_Y[folds[i]])
        D_inv_mean[folds[i]] = D_inv[idx].dot(mean[folds[i]])

    cov = np.linalg.multi_dot([D, K_inv, D])

    residuals_transformed = L.T.dot(D_inv_mean)  # D_inv_mean = D_inv.dot(mean)

    return mean, cov, residuals_transformed
