import pyDOE2  # ELD TODO: Is it ok to import all from pyDOE to use this doe as a wrapper?
import numpy as np


def fullfact_with_bounds(LBs, UBs, N_xi):
    """
    Return a ND array of corresponding (x_0, ..., x_i) values that span out the inputspace
    between lowerbound and upperbound in a structured (grid-like) way with n_x_i points in
    the i'th-dimension.

    Args:
        LBs (list-like, 1D): lower bounds of the input space. len(LBs) must equal len(UBs)
        UBs (list-like, 1D): upper bounds of the input space. len(UBs) must equal len(LBs)
        N_xi (list-like, 1D): number of equidistant samples for each xi-dimension

    Return:
        fullfact (ndarray): [[x_0,...,x_i],...,[x_0_n,...,x_i_n]]

    """
    LBs = np.array(LBs)
    UBs = np.array(UBs)
    N_xi = np.array(N_xi)

    if len(LBs.shape) > 1 or len(UBs.shape) > 1 or len(N_xi.shape) > 1:
        raise ValueError("LBs, UBs, N_xi must all be list-like 1D")
    if not LBs.shape == UBs.shape == N_xi.shape:
        raise ValueError("Shape of LBs, UBs, N_xi must be equal")
    if np.any(N_xi < 1):
        raise ValueError("All elements of N_xi must be greater than or equal to 1")
    if np.any(LBs > UBs):
        raise ValueError(
            "All elements of LBs must be less than the corresponding elements of UBs"
        )

    ffact = pyDOE2.fullfact(N_xi)

    return LBs + ffact / ffact.max(axis=0) * (UBs - LBs)


def lhs_with_bounds(nDim, nSamples, LBs, UBs, random_state=None):
    """
    Return a 2D array of corresponding (x, y) values that fill the input space
    between lowerbound and upperbound with n points using a Latin-hypercube design.

    Args:
        nDim (int): Number of dimensions
        nSamples (int): Number of total samples
        LBs (list-like): 1D, lower bounds of the input space. len(LBs) must equal len(UBs)
        UBs (list-like): 1D, upper bounds of the input space. len(UBs) must equal len(LBs)
        random_state (int, RandomState instance or None, default=None):
            Determines random number generation used to initialize the samples.
            Pass an int for reproducible results across multiple function calls.
    Returns:
        lhs (ndarray): Array of sample points with shape (nSamples, nDim)[[x_0,...,x_i],...,[x_0_n,...,x_i_n]]
    """

    LBs = np.array(LBs)
    UBs = np.array(UBs)

    if len(LBs.shape) > 1 or len(UBs.shape) > 1:
        raise ValueError("LBs, UBs must be list-like 1D")
    if not LBs.shape == UBs.shape == (nDim,):
        raise ValueError("Shape of LBs, UBs, must be equal and match the (nDim,)")

    lhs = pyDOE2.lhs(nDim, samples=nSamples, random_state=random_state)

    return LBs + lhs * (UBs - LBs)


def in_hull(p, hull):
    """
    Test if points in `p` are in `hull`

    `p` should be a `NxK` coordinates of `N` points in `K` dimensions
    `hull` is either a scipy.spatial.Delaunay object or the `MxK` array of the
    coordinates of `M` points in `K`dimensions for which Delaunay triangulation
    will be computed
    """
    from scipy.spatial import Delaunay

    if not isinstance(hull, Delaunay):
        hull = Delaunay(hull)

    return hull.find_simplex(p) >= 0
