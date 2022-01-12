# import pytest
import numpy as np

from rapid_models.gp_diagnostics.utils.linalg import (
    triang_solve,
    mulinv_solve,
    mulinv_solve_rev,
    symmetrify,
    chol_inv,
    traceprod,
    # try_chol,
)

# TODO: if we keep try_col, implement a test for it


def random_matrix(N, M, seed):
    """
    Return random NxM matrix
    """
    np.random.seed(seed)
    return np.random.uniform(size=(N, M))


def random_lower_triang_matrix(N, seed):
    """
    Return random NxN lower triangular matrix
    """
    tmp = random_matrix(N, N, seed)
    return np.linalg.cholesky(tmp.dot(tmp.T))


def test_symmetrify():
    N = 10

    A = random_lower_triang_matrix(N, 77)
    symmetrify(A, upper=False)
    
    assert np.allclose(A - A.T, np.zeros((N, N)))

    A = random_lower_triang_matrix(N, 77)
    symmetrify(A.T, upper=True)
    
    assert np.allclose(A - A.T, np.zeros((N, N)))


def test_triang_solve():
    N = 10
    M = 4

    # Generate matrices
    L = random_lower_triang_matrix(N, 42)
    B = random_matrix(N, M, 43)
    # A = L.dot(L.T)

    # Solve L*X = B and check
    X = triang_solve(L, B, lower=True, trans=False)
    assert np.allclose(B, L.dot(X))

    # Solve L.T*X = B and check
    X = triang_solve(L.T, B, lower=False, trans=False)
    assert np.allclose(B, L.T.dot(X))

    # Solve L.T*X = B and check
    X = triang_solve(L, B, lower=True, trans=True)
    assert np.allclose(B, L.T.dot(X))

    # Solve L*X = B and check
    X = triang_solve(L.T, B, lower=False, trans=True)
    assert np.allclose(B, L.dot(X))


def test_mulinv_solve():
    N = 10
    M = 4

    # Generate matrices
    L = random_lower_triang_matrix(N, 42)
    B = random_matrix(N, M, 43)
    A = L.dot(L.T)

    # Solve A*X = B and check
    X = mulinv_solve(L, B, lower=True)
    assert np.allclose(B, A.dot(X))

    # Solve A*X = B and check
    X = mulinv_solve(L.T, B, lower=False)
    assert np.allclose(B, L.T.dot(L).dot(X))


def test_mulinv_solve_rev():
    N = 10
    M = 4

    # Generate matrices
    L = random_lower_triang_matrix(N, 42)
    B = random_matrix(M, N, 43)
    A = L.dot(L.T)

    # Solve X*A = B and check
    X = mulinv_solve_rev(L, B, lower=True)
    assert np.allclose(B, X.dot(A))

    # Solve X*A = B and check
    X = mulinv_solve_rev(L.T, B, lower=False)
    assert np.allclose(B, X.dot(L.T.dot(L)))


def test_chol_inv():
    N = 10

    # Generate matrices
    L = random_lower_triang_matrix(N, 42)
    A = L.dot(L.T)

    # Invert and check
    A_inv_true = np.linalg.inv(A)
    A_inv = chol_inv(L)

    assert np.allclose(A_inv_true, A_inv)


def test_traceprod():
    N = 10
    M = 8

    # Generate matrices
    A = random_matrix(N, M, 42)
    B = random_matrix(M, N, 43)

    # Compute and check
    trace = traceprod(A, B)
    trace_true = A.dot(B).diagonal().sum()
    assert np.allclose(trace, trace_true)
