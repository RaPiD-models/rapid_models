import pyDOE2  # ELD TODO: Is it ok to import all from pyDOE to use this doe as a wrapper?
import numpy as np

from scipy.spatial import Delaunay

from sklearn.cluster import KMeans
from scipy.spatial import distance_matrix


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
    will be computed.

    Args:
        p (array-like): NxK array. Set of `N` points in `K` dimensions for
            which to check if is inside convex hull.
        hull (array like / scipy.spatial.Delaunay): MxK array. Set of `M` points
            in `K` dimensions to calculate the Delaunay tessellation using the
            [Qhull library](http://www.qhull.org/), or an existing
            scipy.spatial.Delaunay object.
    Returns:
        b_in_hull (boolean ndarray): (N, ) boolean array of sample points where
            `True' indicate that the point of that index is inside the
            triangulation while `False` indicate that the point of
            corresponding index is outside the triangulation.

    """

    p = np.array(p)

    if not len(p.shape) == 2:
        raise ValueError(
            "p must be a NxK 2D array of `N`points in `K` dimensions. p.shape: {}".format(
                p
            )
        )

    if not isinstance(hull, Delaunay):
        hull = np.array(hull)
        if not len(hull.shape) == 2:
            raise ValueError(
                "hull must be a scipy.spatial.Delaunay object or a MxK 2D array of `M`points in `K` dimensions for which Delaunay triangulation will be computed"
            )
        if not p.shape[1] == hull.shape[1]:
            raise ValueError(
                "Size of second dimension of p and hull must be the same (i.e. p and hull must have the same 'K' dimensions."
            )

        hull = Delaunay(hull)

    return hull.find_simplex(p) >= 0


def kmeans_sample(points, N, values=None, type="center", random_state=42):
    """
    Based on
    https://stackoverflow.com/questions/69195903/

    Args:
        points (array-like): PxK array, set of ´P´ points in ´K´ dimensions
            from which to calculate N k-means clusters to use as distance
            maximizing samples in K-dimensions
        N (int): Number of clusters
        values (array-like, 1D): Array of corresponding values at each point in
            `points`. These values can be used to select e.g. minimum or maximum
            values inside each cluster by specifying `type`
        type (str, default="center"): default type select center of k-means
            cluster. "center_closest_point" select the point which are closest
            to the cluster center, "min" select the point with the corresponding
            minimum value, "max" select the point with the corresponding maximum
            value
    Returns:
        points (ndarray): NxK array of ´N´ points in ´K´ dimensions
            representing the `type` (default "center") in the N k-means
            clusters.
        values (ndarray): optional array of `N` values of the returned points.

    """

    ret_ixs = None

    kmeans = KMeans(n_clusters=N, random_state=random_state).fit(points)
    labels = kmeans.predict(points)
    cntr = kmeans.cluster_centers_

    if type.lower() == "center":
        return cntr
    elif type.lower() == "center_closest_point":
        # indices of nearest points to centres

        for q, c in enumerate(cntr):
            ixs = np.where(labels == q)[0]
            pts = points[ixs]
            d = distance_matrix(c[None, ...], pts)
            idx1 = np.argmin(d, axis=1) + 1
            idx2 = np.searchsorted(np.cumsum(labels == q), idx1)[0]
            if ret_ixs is None:
                ret_ixs = np.array(idx2)
            else:
                ret_ixs = np.append(ret_ixs, idx2)

        if values is not None:
            return points[ret_ixs], values[ret_ixs]
        else:
            return points[ret_ixs]

    elif type.lower() == "max":
        if values is None:
            print("values=None: type='max' is not possible, returning None")
            return None
        else:
            for q, c in enumerate(cntr):
                f_lab = labels == q
                val = values[f_lab].max()
                if ret_ixs is None:
                    ret_ixs = np.where(f_lab & (values == val))[0][:1]
                else:
                    ret_ixs = np.append(ret_ixs, np.where(f_lab & (values == val))[0])
            return points[ret_ixs], values[ret_ixs]

    elif type.lower() == "min":
        if values is None:
            print("values=None: type='min' is not possible, returning None")
            return None
        else:
            for q, c in enumerate(cntr):
                f_lab = labels == q
                val = values[f_lab].min()
                if ret_ixs is None:
                    ret_ixs = np.where(f_lab & (values == val))[0][:1]
                else:
                    ret_ixs = np.append(ret_ixs, np.where(f_lab & (values == val))[0])

            return points[ret_ixs], values[ret_ixs]
