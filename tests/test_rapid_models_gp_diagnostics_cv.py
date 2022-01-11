import pytest
import numpy as np
import random
import torch
import gpytorch

from rapid_models.gp_diagnostics.cv import (
    check_folds_indices,
    check_lower_triangular,
    check_numeric_array,
    loo,
    multifold,
)
from rapid_models.gp_diagnostics.utils.stats import split_test_train_fold

from rapid_models.gp_models.templates import ExactGPModel
import rapid_models.gp_models.utils as gputils


def test_check_folds_indices_correct():
    """
    Test that check_folds_indices throws assertion when needed

    - No assertion when input is correct
    """
    # Correct (should not do anything)
    check_folds_indices([[1, 3], [5, 6, 7], [0, 2], [4]], 8)


def test_check_folds_indices_nmax():
    """
    Test that check_folds_indices throws assertion when needed

    - Wrong n_max
    """
    with pytest.raises(Exception):
        check_folds_indices([[1, 3], [5, 6, 7], [0, 2], [4]], 6)  # Wrong n_max


def test_check_folds_indices_list():
    """
    Test that check_folds_indices throws assertion when needed

    - Not a list of lists
    """
    with pytest.raises(Exception):
        check_folds_indices([1, 3, 5, 6, 7, 0, 2, 4], 8)  # Not a list of lists


def test_check_folds_indices_int():
    """
    Test that check_folds_indices throws assertion when needed

    - Not a list of lists of ints
    """
    with pytest.raises(Exception):
        check_folds_indices([["22"]], 8)  # Wrong type
        check_folds_indices([[1, 0.5]], 8)  # Wrong type


def test_check_folds_indices_exh():
    """
    Test that check_folds_indices throws assertion when needed

    - Not exhaustive indices
    """
    with pytest.raises(Exception):
        check_folds_indices([[1, 3], [5, 7], [0, 2], [4]], 8)  # Not exhaustive indices
        check_folds_indices(
            [[1, 3], [5, 6, 7, 9], [0, 2], [4]], 8
        )  # Not exhaustive indices


def test_check_folds_indices_empty():
    """
    Test that check_folds_indices throws assertion when needed

    - Empty lists
    """
    with pytest.raises(Exception):
        check_folds_indices([[1, 3], [5, 6, 7], [0, 2], [], [4]], 8)


def test_check_lower_triangular():
    """
    Test that check_lower_triangular throws assertion when needed

    including multiple cases in same test as check_lower_triangular already uses tested functions
    """
    # These should be ok
    arr = np.array([[0.2, 0, 0], [3, 2.2, 0], [1, 2, 4]])

    check_lower_triangular(arr)

    arr = np.array([[1, 0], [2, 2.2]])
    check_lower_triangular(arr)

    arr = np.array([[1]])
    check_lower_triangular(arr)

    # These should raise error
    with pytest.raises(Exception):
        check_lower_triangular(np.array([[1, 2, 2.3], [0, 2.2, 3], [0.1, 0, 4]]))

    with pytest.raises(Exception):
        check_lower_triangular(np.ones(shape=(13, 14))) == False

    with pytest.raises(Exception):
        check_lower_triangular(np.array([[1, 2, 2.3], [0, 0.001, "a"]])) == False

    with pytest.raises(Exception):
        check_lower_triangular("a") == False


def test_check_numeric_array():
    """
    Test that check_numeric_array throws assertion when needed

    including multiple cases in same test as check_numeric_array already uses tested functions
    """
    # These should be ok
    check_numeric_array(np.ones(4), 1)
    check_numeric_array(np.ones(shape=(2, 4)), 2)
    check_numeric_array(np.ones(shape=(2, 4, 6)), 3)
    check_numeric_array(np.array(4), 0)

    # These should raise error
    with pytest.raises(Exception):
        check_numeric_array(np.array(4), 1)

    with pytest.raises(Exception):
        check_numeric_array(np.ones(shape=(2, 4)), 1)

    with pytest.raises(Exception):
        check_numeric_array("a", 1)

    with pytest.raises(Exception):
        check_numeric_array(np.array([1, "1"]), 1)


def test_loo_1d():
    """
    Test the 1d example from

    [D. Ginsbourger and C. Schaerer (2021). Fast calculation of Gaussian Process multiple-fold
    crossvalidation residuals and their covariances. arXiv:2101.03108]
    """

    # Covariance matrix and observations
    Y_train = np.array(
        [
            -0.6182,
            -0.3888,
            -0.3287,
            -0.2629,
            0.3614,
            0.1442,
            -0.0374,
            -0.0546,
            -0.0056,
            0.0529,
        ]
    )

    K = np.array(
        [
            [
                9.1475710e-02,
                5.4994639e-02,
                1.8560780e-02,
                4.8545646e-03,
                1.1045272e-03,
                2.3026903e-04,
                4.5216686e-05,
                8.5007923e-06,
                1.5461808e-06,
                2.7402149e-07,
            ],
            [
                5.4994639e-02,
                9.1475710e-02,
                5.4994628e-02,
                1.8560771e-02,
                4.8545646e-03,
                1.1045267e-03,
                2.3026903e-04,
                4.5216686e-05,
                8.5007923e-06,
                1.5461794e-06,
            ],
            [
                1.8560775e-02,
                5.4994617e-02,
                9.1475710e-02,
                5.4994628e-02,
                1.8560771e-02,
                4.8545622e-03,
                1.1045267e-03,
                2.3026903e-04,
                4.5216686e-05,
                8.5007923e-06,
            ],
            [
                4.8545646e-03,
                1.8560771e-02,
                5.4994628e-02,
                9.1475710e-02,
                5.4994628e-02,
                1.8560771e-02,
                4.8545646e-03,
                1.1045275e-03,
                2.3026903e-04,
                4.5216719e-05,
            ],
            [
                1.1045275e-03,
                4.8545646e-03,
                1.8560771e-02,
                5.4994628e-02,
                9.1475710e-02,
                5.4994617e-02,
                1.8560771e-02,
                4.8545646e-03,
                1.1045275e-03,
                2.3026903e-04,
            ],
            [
                2.3026903e-04,
                1.1045267e-03,
                4.8545622e-03,
                1.8560771e-02,
                5.4994617e-02,
                9.1475710e-02,
                5.4994635e-02,
                1.8560780e-02,
                4.8545660e-03,
                1.1045275e-03,
            ],
            [
                4.5216686e-05,
                2.3026884e-04,
                1.1045267e-03,
                4.8545646e-03,
                1.8560771e-02,
                5.4994635e-02,
                9.1475710e-02,
                5.4994639e-02,
                1.8560780e-02,
                4.8545660e-03,
            ],
            [
                8.5007923e-06,
                4.5216686e-05,
                2.3026903e-04,
                1.1045275e-03,
                4.8545646e-03,
                1.8560780e-02,
                5.4994639e-02,
                9.1475710e-02,
                5.4994617e-02,
                1.8560775e-02,
            ],
            [
                1.5461794e-06,
                8.5007923e-06,
                4.5216686e-05,
                2.3026903e-04,
                1.1045272e-03,
                4.8545660e-03,
                1.8560780e-02,
                5.4994628e-02,
                9.1475710e-02,
                5.4994639e-02,
            ],
            [
                2.7402149e-07,
                1.5461808e-06,
                8.5007923e-06,
                4.5216686e-05,
                2.3026903e-04,
                1.1045275e-03,
                4.8545660e-03,
                1.8560780e-02,
                5.4994639e-02,
                9.1475710e-02,
            ],
        ]
    )

    # From paper
    LOO_residuals_transformed_true = np.array(
        [
            -2.04393396,
            -0.07086865,
            -0.81325009,
            -0.33726709,
            2.07426555,
            -0.82414653,
            -0.05894939,
            -0.10534249,
            0.10176395,
            0.18906068,
        ]
    )

    LOO_mean_true = np.array(
        [
            -0.38365906,
            0.02787939,
            0.02736787,
            -0.29997396,
            0.36816096,
            -0.11112669,
            0.0047464,
            -0.01693309,
            -0.00649325,
            0.04409083,
        ]
    )

    LOO_cov_true = np.array(
        [
            [
                5.43868914e-02,
                -2.63221730e-02,
                1.02399085e-02,
                -3.40788392e-03,
                1.09471090e-03,
                -3.50791292e-04,
                1.12430032e-04,
                -3.61632192e-05,
                1.20125414e-05,
                -4.93005518e-06,
            ],
            [
                -2.63221748e-02,
                3.40218581e-02,
                -2.04288363e-02,
                8.01187754e-03,
                -2.66015902e-03,
                8.54554935e-04,
                -2.73940881e-04,
                8.81146188e-05,
                -2.92695640e-05,
                1.20124914e-05,
            ],
            [
                1.02399066e-02,
                -2.04288289e-02,
                3.19701731e-02,
                -1.97094288e-02,
                7.72963883e-03,
                -2.56570964e-03,
                8.24513379e-04,
                -2.65259878e-04,
                8.81142623e-05,
                -3.61629245e-05,
            ],
            [
                -3.40788229e-03,
                8.01187381e-03,
                -1.97094269e-02,
                3.17552164e-02,
                -1.96319763e-02,
                7.69940997e-03,
                -2.55653122e-03,
                8.24513205e-04,
                -2.73939862e-04,
                1.12429247e-04,
            ],
            [
                1.09471008e-03,
                -2.66015716e-03,
                7.72963790e-03,
                -1.96319763e-02,
                3.17334048e-02,
                -1.96248218e-02,
                7.69941136e-03,
                -2.56570987e-03,
                8.54552840e-04,
                -3.50789691e-04,
            ],
            [
                -3.50790826e-04,
                8.54554237e-04,
                -2.56570918e-03,
                7.69940997e-03,
                -1.96248218e-02,
                3.17334011e-02,
                -1.96319763e-02,
                7.72963837e-03,
                -2.66015413e-03,
                1.09470787e-03,
            ],
            [
                1.12429749e-04,
                -2.73940503e-04,
                8.24513205e-04,
                -2.55653122e-03,
                7.69941136e-03,
                -1.96319763e-02,
                3.17552052e-02,
                -1.97094250e-02,
                8.01186915e-03,
                -3.40787880e-03,
            ],
            [
                -3.61630882e-05,
                8.81144733e-05,
                -2.65259820e-04,
                8.24513321e-04,
                -2.56571034e-03,
                7.72963930e-03,
                -1.97094250e-02,
                3.19701731e-02,
                -2.04288270e-02,
                1.02399047e-02,
            ],
            [
                1.20125114e-05,
                -2.92695440e-05,
                8.81143787e-05,
                -2.73940270e-04,
                8.54553713e-04,
                -2.66015623e-03,
                8.01187288e-03,
                -2.04288345e-02,
                3.40218581e-02,
                -2.63221730e-02,
            ],
            [
                -4.93004836e-06,
                1.20124987e-05,
                -3.61630118e-05,
                1.12429487e-04,
                -3.50790157e-04,
                1.09470845e-03,
                -3.40788020e-03,
                1.02399066e-02,
                -2.63221730e-02,
                5.43868914e-02,
            ],
        ]
    )

    # Computed
    LOO_mean, LOO_cov, LOO_residuals_transformed = loo(K, Y_train)

    # Compare
    assert np.allclose(LOO_mean, LOO_mean_true, atol=1e-3)
    assert np.allclose(LOO_cov, LOO_cov_true)
    assert np.allclose(
        LOO_residuals_transformed, LOO_residuals_transformed_true, atol=1e-3
    )


def generate_cv_data(
    N_DIM=3, N_TRAIN=100, N_DUPLICATE_X=0, NUM_FOLDS=8, NOISE_VAR=0, SCRAMBLE=True
):
    """
    Generate some cross validation data manually for testing
    """

    # Set random seed
    random.seed(42)
    torch.manual_seed(42)
    np.random.seed(42)

    # Will generate data using a zero-mean Matern 5/2 GP with these parameters
    KER_SCALE_TRUE = 1.0
    KER_LENGTHSCALE_TRUE = torch.ones(N_DIM) * 0.5

    # Generate N_TRAIN training Xs, with N_DUPLICATE_X duplicates
    X_train = torch.rand(size=(N_TRAIN - N_DUPLICATE_X, N_DIM))

    if N_DUPLICATE_X > 0:
        X_train = torch.cat(
            [
                X_train,
                X_train[
                    np.random.choice(np.arange(N_TRAIN - N_DUPLICATE_X), N_DUPLICATE_X)
                ],
            ]
        )

    # Define kernel and sample training data
    ker = gputils.gpytorch_kernel_Matern(KER_SCALE_TRUE, KER_LENGTHSCALE_TRUE)
    K = ker(X_train)
    normal_rv = gpytorch.distributions.MultivariateNormal(
        mean=torch.zeros(N_TRAIN), covariance_matrix=K
    )

    if NOISE_VAR == 0:
        noise = 0
    else:
        noise_rv = gpytorch.distributions.MultivariateNormal(
            mean=torch.zeros(N_TRAIN), covariance_matrix=torch.eye(N_TRAIN) * NOISE_VAR
        )
        noise = noise_rv.sample()

    Y_train = normal_rv.sample() + noise

    # Create a list of index subsets
    if NUM_FOLDS == N_TRAIN:
        FOLDS_INDICES = [[i] for i in range(N_TRAIN)]

    else:
        # This sampling will not work if NUM_FOLDS is very big (wrt N_TRAIN), but we will only use it for some examples where NUM_FOLDS << N_TRAIN
        folds_end = np.random.multinomial(
            N_TRAIN, np.ones(NUM_FOLDS) / NUM_FOLDS, size=1
        )[
            0
        ].cumsum()  # last index of each fold
        folds_startstop = np.insert(folds_end, 0, 0, axis=0)
        FOLDS_INDICES = [
            list(range(folds_startstop[i], folds_startstop[i + 1]))
            for i in range(NUM_FOLDS)
        ]

    if SCRAMBLE:
        rnd_idx = np.random.permutation(N_TRAIN)  # Randomized indices
        FOLDS_INDICES = [list(rnd_idx[idx]) for idx in FOLDS_INDICES]

    check_folds_indices(FOLDS_INDICES, N_TRAIN)

    # Define GP model
    gp_lik_var = max(1e-6, NOISE_VAR)
    model = ExactGPModel(
        X_train,
        Y_train,  # Training data
        gputils.gpytorch_mean_constant(0.0, fixed=True),  # Mean function
        ker,  # Kernel
        gputils.gpytorch_likelihood_gaussian(
            variance=gp_lik_var, fixed=False
        ),  # Likelihood
        "",
        "",
    )  # Name and path for save/load

    # Run CV manually
    model.eval_mode()
    cv_residual_means = []  # Residual (observed - predicted) mean
    cv_residual_vars = []  # Residual (observed - predicted) variance

    for i in range(NUM_FOLDS):

        # Split on i-th fold
        fold_X_test, fold_X_train = split_test_train_fold(FOLDS_INDICES, X_train, i)
        fold_Y_test, fold_Y_train = split_test_train_fold(FOLDS_INDICES, Y_train, i)

        # Set training data
        model.set_train_data(inputs=fold_X_train, targets=fold_Y_train, strict=False)

        # Predict on test data
        m, v = model.predict(fold_X_test, latent=False)

        cv_residual_means.append((fold_Y_test - m).numpy())
        cv_residual_vars.append(v.numpy())

    # Concatenate and sort so that the residuals correspond to observation 1, 2, 3 etc.
    cv_residual_means = np.array(cv_residual_means).flatten()
    cv_residual_vars = np.array(cv_residual_vars).flatten()
    if NUM_FOLDS != N_TRAIN:
        cv_residual_means = np.concatenate(cv_residual_means)
        cv_residual_vars = np.concatenate(cv_residual_vars)

    folds_concat = sum(FOLDS_INDICES, [])
    idx_sort = list(np.argsort(folds_concat))
    cv_residual_means = cv_residual_means[idx_sort]
    cv_residual_vars = cv_residual_vars[idx_sort]

    return (
        cv_residual_means,
        cv_residual_vars,
        FOLDS_INDICES,
        K.evaluate().numpy(),
        X_train,
        Y_train,
    )


def multitest_loo(N_DIM, N_TRAIN, NOISE_VAR, N_DUPLICATE_X):
    """
    Used for running multiple tests of loo()

    Checks that mean and variance are the same as if the residuals were computed in a loop.
    This does NOT check covariance and transformed residuals
    """

    # Generate residuals
    cv_residual_means, cv_residual_vars, folds, K, X_train, Y_train = generate_cv_data(
        N_DIM, N_TRAIN, N_DUPLICATE_X, N_TRAIN, NOISE_VAR, False
    )
    cv_residual_means = np.array(cv_residual_means).flatten()
    cv_residual_vars = np.array(cv_residual_vars).flatten()

    # Compute residuals from cholesky factor incl jitter
    gp_lik_var = max(1e-6, NOISE_VAR)
    LOO_mean, LOO_cov, LOO_residuals_transformed = loo(K, Y_train.numpy(), gp_lik_var)
    LOO_var = LOO_cov.diagonal()

    # Check
    assert np.allclose(LOO_var, cv_residual_vars, atol=1e-3)
    assert np.allclose(LOO_mean, cv_residual_means, atol=1e-3)


def test_loo_noiseless():
    """
    Test that loo formula gives same result as computing the residuals in a loop

    No observational noise, no duplicates
    """
    multitest_loo(N_DIM=3, N_TRAIN=50, NOISE_VAR=0, N_DUPLICATE_X=0)


def test_loo_noise():
    """
    Test that loo formula gives same result as computing the residuals in a loop

    With observational noise, no duplicates
    """
    multitest_loo(N_DIM=3, N_TRAIN=50, NOISE_VAR=0.3, N_DUPLICATE_X=0)


def test_loo_noise_dupl():
    """
    Test that loo formula gives same result as computing the residuals in a loop

    With observational noise, with duplicates
    """
    multitest_loo(N_DIM=3, N_TRAIN=50, NOISE_VAR=0.3, N_DUPLICATE_X=30)


def multitest_multifold(N_DIM, N_TRAIN, NUM_FOLDS, NOISE_VAR, N_DUPLICATE_X, SCRABMLE):
    """
    Used for running multiple tests of multifold()

    Checks that mean and variance are the same as if the residuals were computed in a loop.
    This does NOT check covariance and transformed residuals
    """

    # Generate residuals
    cv_residual_means, cv_residual_vars, folds, K, X_train, Y_train = generate_cv_data(
        N_DIM, N_TRAIN, N_DUPLICATE_X, NUM_FOLDS, NOISE_VAR, SCRABMLE
    )

    # Compute residuals from cholesky factor incl jitter
    gp_lik_var = max(1e-6, NOISE_VAR)
    CV_mean, CV_cov, CV_residuals_transformed = multifold(
        K, Y_train.numpy(), folds, gp_lik_var
    )
    CV_var = CV_cov.diagonal()

    # Check
    assert np.allclose(CV_var, cv_residual_vars, atol=1e-4)
    assert np.allclose(CV_mean, cv_residual_means, atol=1e-3)


def test_multifold_noiseless():
    """
    Test that multifold formula gives same result as computing the residuals in a loop

    No observational noise, no duplicates
    """
    multitest_multifold(
        N_DIM=3, N_TRAIN=100, NUM_FOLDS=8, NOISE_VAR=0, N_DUPLICATE_X=0, SCRABMLE=True
    )


def test_multifold_noise():
    """
    Test that multifold formula gives same result as computing the residuals in a loop

    With observational noise, no duplicates
    """
    multitest_multifold(
        N_DIM=3, N_TRAIN=100, NUM_FOLDS=8, NOISE_VAR=0.3, N_DUPLICATE_X=0, SCRABMLE=True
    )


def test_multifold_noise_dupl():
    """
    Test that multifold formula gives same result as computing the residuals in a loop

    With observational noise, with duplicates
    """
    multitest_multifold(
        N_DIM=3,
        N_TRAIN=100,
        NUM_FOLDS=8,
        NOISE_VAR=0.3,
        N_DUPLICATE_X=30,
        SCRABMLE=True,
    )


def test_loo_multifold():
    """
    Test that LOO and multifold CV gives the same results when fold size = 1
    """

    # Generate some data
    N = 100
    NOISE_VAR = 0.23
    cv_residual_means, cv_residual_vars, folds, K, X_train, Y_train = generate_cv_data(
        N_DIM=2,
        N_TRAIN=N,
        N_DUPLICATE_X=20,
        NUM_FOLDS=N,
        NOISE_VAR=NOISE_VAR,
        SCRAMBLE=False,
    )

    gp_lik_var = max(1e-6, NOISE_VAR)
    CV_mean, CV_cov, CV_residuals_transformed = multifold(
        K, Y_train.numpy(), folds, gp_lik_var
    )
    LOO_mean, LOO_cov, LOO_residuals_transformed = loo(K, Y_train.numpy(), gp_lik_var)

    # Compute multifold CV and LOO
    assert np.allclose(CV_mean, LOO_mean)
    assert np.allclose(CV_cov, LOO_cov)
    assert np.allclose(CV_residuals_transformed, LOO_residuals_transformed)
