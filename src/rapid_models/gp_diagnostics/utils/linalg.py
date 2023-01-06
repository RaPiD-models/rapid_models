import contextlib
from typing import Union

import numpy as np
import scipy.linalg.lapack as lapack
from nptyping import Float, NDArray, Shape


def triang_solve(
    A: NDArray[Shape["N, N"], Float],  # noqa: F821
    B: Union[
        NDArray[Shape["N"], Float], NDArray[Shape["N, M"], Float]  # noqa: F821
    ],  # noqa: F821
    lower: bool = True,
    trans: bool = False,
) -> Union[
    NDArray[Shape["N"], Float], NDArray[Shape["N, M"], Float]  # noqa: F821
]:  # noqa: F821
    """
    Wrapper for lapack dtrtrs function
    DTRTRS solves a triangular system of the form
        A * X = B  or  A**T * X = B,
    where A is a triangular matrix of order N, and B is an N-by-NRHS
    matrix.  A check is made to verify that A is nonsingular.
    :param A: Matrix A(triangular)
    :param B: Matrix B
    :param lower: is matrix lower (true) or upper (false)
    :param trans: calculate A**T * X = B (true) or A * X = B (false)

    :returns: Solution to A * X = B or A**T * X = B
    """
    unitdiag = False

    lower_num = 1 if lower else 0
    trans_num = 1 if trans else 0
    unitdiag_num = 1 if unitdiag else 0

    A = np.asfortranarray(A)

    return lapack.dtrtrs(
        A, B, lower=lower_num, trans=trans_num, unitdiag=unitdiag_num
    )[0]


def mulinv_solve(
    F: NDArray[Shape["N, N"], Float],  # noqa: F821
    B: Union[
        NDArray[Shape["N"], Float], NDArray[Shape["N, M"], Float]  # noqa: F821
    ],  # noqa: F821
    lower: bool = True,
) -> Union[
    NDArray[Shape["N"], Float], NDArray[Shape["N, M"], Float]  # noqa: F821
]:  # noqa: F821
    """
    Solve A*X = B where A = F*F^{T}

    lower = True -> when F is LOWER triangular. This gives faster calculation
    """
    tmp = triang_solve(F, B, lower=lower, trans=False)
    return triang_solve(F, tmp, lower=lower, trans=True)


def mulinv_solve_rev(
    F: NDArray[Shape["N, N"], Float],  # noqa: F821
    B: Union[
        NDArray[Shape["N"], Float], NDArray[Shape["M, N"], Float]  # noqa: F821
    ],  # noqa: F821
    lower: bool = True,
) -> Union[
    NDArray[Shape["N"], Float], NDArray[Shape["M, N"], Float]  # noqa: F821
]:  # noqa: F821
    """
    Reversed version of mulinv_solve

    Solves X*A = B where A = F*F^{T}

    lower = True -> when F is LOWER triangular. This gives faster calculation

    """
    return mulinv_solve(F, B.T, lower).T


def symmetrify(
    A: NDArray[Shape["*, *"], Float],  # noqa: F722
    upper: bool = False,
):
    """Create symmetric matrix from triangular matrix"""
    triu = np.triu_indices_from(A, k=1)
    if upper:
        A.T[triu] = A[triu]
    else:
        A[triu] = A.T[triu]


def chol_inv(
    L: NDArray[Shape["N, N"], Float]  # noqa: F821
) -> NDArray[Shape["N, N"], Float]:  # noqa: F821
    """
    Return inverse of matrix A = L*L.T where L is lower triangular
    Uses LAPACK function dpotri
    """
    A_inv, _ = lapack.dpotri(L, lower=1)
    symmetrify(A_inv)
    return A_inv


def traceprod(
    A: NDArray[Shape["N, M"], Float],  # noqa: F821
    B: NDArray[Shape["M, N"], Float],  # noqa: F821
) -> float:
    """
    Calculate trace(A*B) for two matrices A and B
    """
    return np.einsum("ij,ji->", A, B)


def try_chol(
    K: NDArray[Shape["N, N"], Float],  # noqa: F821
    noise_variance: float = 0.0,
    func_name: str = "",
) -> Union[NDArray[Shape["N, N"], Float], None]:  # noqa: F821
    """
    Try to compute the Cholesky decomposition of (K + noise_variance*I), and
    print an error message if it does not work
    """
    A = K + np.eye(K.shape[0]) * noise_variance
    try:
        return np.linalg.cholesky(A)
    except Exception:
        print(
            "Could not compute \
                numpy.linalg.cholesky(K + np.eye(K.shape[0])*noise_variance)! \
                The matrix K is probably not positive definite.\n"
            f"Try using {func_name}_cholesky() with alternative Cholesky \
                factor, or add jitter by increasing noise_variance."
        )
        with contextlib.suppress(Exception):
            print(f"Smallest eigenvalue: {np.linalg.eig(A)[0].min()}")
    return None
