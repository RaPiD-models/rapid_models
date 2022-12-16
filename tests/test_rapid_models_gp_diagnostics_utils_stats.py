# import pytest
import numpy as np

from rapid_models.gp_diagnostics.utils.stats import (
    snorm_qq,
    split_test_train_fold,
)


def test_snorm_qq_equalR():
    """
    Check that the function returns the same as in R
    """

    # Inputs (generated from a standard normal variable)
    x = np.array([
        -0.325459352843027,
        -0.894532916553418,
        -0.651097296010105,
        0.431962456402265,
        1.11307123980658,
        0.16432330947537,
        -1.3009717579692,
        -1.20616922164046,
        -0.862389244884325,
        -0.783944517858679,
        -0.86289300469735,
        -0.452842255748436,
        -1.32372331616317,
        -1.21546884455723,
        -0.913805791530649,
        0.412898096437753,
        -0.044206507445586,
        -0.106133827148484,
        -1.21579545950289,
        -0.791685174905493,
    ])

    # Output
    q_sample, q_snorm, q_snorm_upper, q_snorm_lower = snorm_qq(x)

    # Output computed from extRemes.qqnorm in R
    R_lower = np.array([
        np.nan,
        np.nan,
        np.nan,
        np.nan,
        -1.8436377834902,
        -1.38767528310563,
        -1.11410079939012,
        -0.905433177563493,
        -0.730252354751578,
        -0.575081844939952,
        -0.432695573891231,
        -0.298612213033157,
        -0.169712878082784,
        -0.0435840752400704,
        0.0818524407565779,
        0.208595460977847,
        0.338795151235563,
        0.475032155363908,
        0.620750108775009,
        0.781065359836158,
    ])

    R_upper = np.array([
        -0.781065359836158,
        -0.620750108775009,
        -0.475032155363908,
        -0.338795151235563,
        -0.208595460977847,
        -0.0818524407565779,
        0.0435840752400707,
        0.169712878082784,
        0.298612213033157,
        0.432695573891231,
        0.575081844939952,
        0.730252354751577,
        0.905433177563493,
        1.11410079939012,
        1.38767528310563,
        1.8436377834902,
        np.nan,
        np.nan,
        np.nan,
        np.nan,
    ])

    R_snorm = np.array([
        -1.95996398454005,
        -1.43953147093846,
        -1.15034938037601,
        -0.93458929107348,
        -0.755415026360469,
        -0.597760126042478,
        -0.453762190169879,
        -0.318639363964375,
        -0.189118426272792,
        -0.0627067779432138,
        0.0627067779432138,
        0.189118426272792,
        0.318639363964375,
        0.45376219016988,
        0.597760126042478,
        0.755415026360469,
        0.93458929107348,
        1.15034938037601,
        1.43953147093846,
        1.95996398454005,
    ])

    R_data = np.array([
        -1.32372331616317,
        -1.3009717579692,
        -1.21579545950289,
        -1.21546884455723,
        -1.20616922164046,
        -0.913805791530649,
        -0.894532916553418,
        -0.86289300469735,
        -0.862389244884325,
        -0.791685174905493,
        -0.783944517858679,
        -0.651097296010105,
        -0.452842255748436,
        -0.325459352843027,
        -0.106133827148484,
        -0.044206507445586,
        0.16432330947537,
        0.412898096437753,
        0.431962456402265,
        1.11307123980658,
    ])

    # Check expected types
    assert isinstance(x, np.ndarray)
    assert isinstance(q_sample, np.ndarray)
    assert isinstance(q_snorm, np.ndarray)
    assert isinstance(q_snorm_upper, np.ndarray)
    assert isinstance(q_snorm_lower, np.ndarray)

    # Check expected shapes
    N: int = 20
    assert x.shape == (N, )
    assert q_sample.shape == (N, )
    assert q_snorm.shape == (N, )
    assert q_snorm_upper.shape == (N, )
    assert q_snorm_lower.shape == (N, )

    # Check that results match the reference output computed from extRemes.qqnorm in R
    assert np.allclose(q_snorm_lower, R_lower, equal_nan=True)
    assert np.allclose(q_snorm_upper, R_upper, equal_nan=True)
    assert np.allclose(q_snorm, R_snorm)
    assert np.allclose(q_sample, R_data)


def test_split_test_train_fold():
    """
    Check that split_test_train_fold() gives the expected results
    """
    folds = [[1, 4, 5], [0], [7], [2, 3, 6], [9, 8]]

    x = np.arange(10) * 1.1

    y1 = np.array([1.1, 4.4, 5.5, 0.0, 7.7, 2.2, 3.3, 6.6])
    y2 = np.array([9.9, 8.8])
    x_test, x_train = split_test_train_fold(folds, x, 4)
    assert np.allclose(y1, x_train)
    assert np.allclose(y2, x_test)

    y1 = np.array([1.1, 4.4, 5.5, 7.7, 2.2, 3.3, 6.6, 9.9, 8.8])
    y2 = np.array([0.0])
    x_test, x_train = split_test_train_fold(folds, x, 1)
    assert np.allclose(y1, x_train)
    assert np.allclose(y2, x_test)

    x_test, x_train = split_test_train_fold(folds, x.reshape(-1, 1), 1)
    assert np.allclose(y1.reshape(-1, 1), x_train)
    assert np.allclose(y2.reshape(-1, 1), x_test)
