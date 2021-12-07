"""Adaptive learning DOE package for rapid-models."""

import torch
import numpy as np

import warnings

# import scipy as sp


def AL_McKay92_idx(gp_std_at_lhs, nNew=1):
    """Active learning by McKay 1992
    Return index of nNew point with highest standard deviation

    Parameters:
      gp_std_at_lhs (list-like, 1D): List or array of standard deviation
        predictions from a Gaussian process (GP) model.The sample size should be
        a suitably large Latin-hypercube sample (LHS) from the entire valid
        input range (E.g. $> 100 x n$ where $n$ is the number of input
        dimensions)
      nNew (int, default=1): Number of largest values to return.
        ``nNew = 1`` will return the index of the largest value of the input,
        (decending) of the input values.

    Returns:
      (tuple): tuple containing:

        - **idxs** _(ndarray, 1D)_: Array of indexes of the nNew largest values in the input.
        - **Timp** _(ndarray, 1D)_: Array of improvement metrics (i.e. the nNew largest values sorted descending).

    """
    std = gp_std_at_lhs

    n_lhs = len(std)

    if n_lhs < 100 or n_lhs < 10 * std.shape[1] or n_lhs < 2 ** std.shape[1]:
        warnings.warn(
            "Size of X_lhs might not be sufficiently large. s={} samples are \
small compared to number of dimensions n={}".format(
                std.shape[0], std.shape[1]
            ),
            UserWarning,
        )

    idxs = np.argpartition(std, -nNew)[-nNew:]
    idxs = idxs[np.argsort(-std[idxs])]
    Timp = std[idxs]

    return idxs, Timp


def AL_Cohn96_idx(kernel_fn, X_train, X_lhs, nNew=1):
    """Active learning by Cohn 1996
    Return index of nNew points which gives the largest global variance reduction

    Parameters:
      kernel_fn (function): Gaussian process (GP) kernel function X_train
        (array-like, size n x d): The training features $\\mathbf{X}$. n is
        dimension size while d is number of training features.
      X_lhs (array-like, size n x s): Latin hypercube sample to estimate the
        improvement metric over. Number of samples s should be sufficiently large.
      nNew (int, default=1): Number of largest values to return. ``nNew = 1``
        will return the index of the largest improvement metric value, while
        ``nNew = len(X_lhs)`` will return a sorted list (decending) of the
        estimated improvement metric values.

    Returns:
      (tuple): tuple containing:

        - **idxs** _(ndarray, 1D)_: Array of indexes of the nNew largest values
          in the estimated improvement metric.
        - **Timp** _(ndarray, 1D)_: Array of the sorted (descending)
          improvement metric values of the nNew largest values.

    """

    if not X_train.shape[1] == X_lhs.shape[1]:
        raise ValueError("2nd dimension of X_train and X_lhs must be equal")

    n_train = len(X_train)
    n_lhs = len(X_lhs)

    if n_lhs < 100 or n_lhs < 10 * X_lhs.shape[1] or n_lhs < 2 ** X_lhs.shape[1]:
        warnings.warn(
            "Size of X_lhs might not be sufficiently large. s={} samples are \
small compared to number of dimensions n={}".format(
                X_lhs.shape[0], X_lhs.shape[1]
            ),
            UserWarning,
        )

    X_all = np.vstack([X_train, X_lhs])  # OK

    C_allx = kernel_fn(X_all)  # OK
    C_train = kernel_fn(X_train)  # OK
    C_lhs = kernel_fn(X_lhs)  # OK

    Cinv = np.linalg.inv(C_train)  # OK
    Cstar = C_allx[n_train:, :n_train]  # OK

    kstKkst = dotdot_a_b_aT_for_row_in_a(Cstar, C_train)  # OK

    tmpT = dotdot_a_b_aT(Cstar, Cinv) - C_lhs

    T_ALCs = np.einsum("ij,i->i", tmpT, 1 / (C_lhs.diagonal() - kstKkst))

    idx = np.argpartition(T_ALCs, -nNew)[-nNew:]
    idx = idx[np.argsort(-T_ALCs[idx])]
    Timp = [(T_ALCs[q]) / np.float(n_lhs) for q in idx]

    return idx, Timp  # , T_ALCs


def dotdot_a_b_aT_for_row_in_a(a, b):
    """Function for efficient calculation (using either numpy.einsum
    or torch.einsum) of row-wize dot(dot(a,b),aT) where aT==a.T as in:


    .. code:: python

      c=[]
      for q in range(len(a)):
          c.append( np.dot(np.dot(a[q, :], b), aT[:,q]))
      return np.array(c)


    Args:
      a (array-like, 2D)
      b (array-like, 2D)

    Returns:
      c (ndarray, 2D)

    """

    if torch.is_tensor(a) and torch.is_tensor(b):
        return torch.einsum("ij,ji->i", torch.einsum("ij,jk", a, b), a.T)
    else:
        return np.einsum("ij,ji->i", np.einsum("ij,jk", a, b), np.array(a).T)
    # return np.einsum('ij,jk,ik->i', a,b,a)


def dotdot_a_b_aT(a, b):
    """Function for efficient calculation (using either numpy.einsum
    or torch.einsum) of row-wize dot(dot(a,b),aT) for all combinations
    of rows in a and cols in aT as in:

    .. code:: python

      c=[]
      for q in range(len(a)):
          for qq in range(len(a))
              c.append( np.dot(np.dot(a[q, :], b), aT[:,qq]))
      return np.array(c)

    Args:
      a (array-like, 2D)
      b (array-like, 2D)

    Returns:
      c (ndarray, 2D)

    """
    if torch.is_tensor(a) and torch.is_tensor(b):
        return torch.einsum("ij,jk", torch.einsum("ij,jk", a, b), a.T)
    else:
        return np.einsum("ij,jk", np.einsum("ij,jk", a, b), np.array(a).T)
