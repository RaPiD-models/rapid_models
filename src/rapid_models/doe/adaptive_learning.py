"""Adaptive learning DOE package for rapid-models."""

import torch
import numpy as np

# import scipy as sp


def AL_McKay92_idx(gp_std_at_lhs, nNew=1):
    """Active learning by McKay 1992
    Return index of nNew point with highest standard deviation

    Args:
      gp_std_at_lhs (list-like, 1D): List or array of standard deviation predictions from a Gaussian process (GP) model. The sample size should be a suitably large Latin-hypercube sample (LHS) from the entire valid input range (E.g. $> 100 x n$ where $n$ is the number of input dimensions)
      nNew (int, default=1): Number of largest values to return. ``nNew = 1`` will return the index of the largest value of the input, while ``nNew = len(gp_std_at_lhs)`` will return a sorted list (decending) of the input values.

    Returns:
      idxs (ndarray, 1D): Array of indexes of the nNew largest values in the input.
      Timp (ndarray, 1D): Array of improvement metrics (i.e. the nNew largest values sorted descending).

    """
    std = gp_std_at_lhs

    # ELD TODO: need test on size of std/lhs to be a meaningful sample size?

    idxs = np.argpartition(std, -nNew)[-nNew:]
    idxs = idxs[np.argsort(-std[idxs])]
    Timp = std[idxs]

    return idxs, Timp


def AL_Cohn96_idx(kernel_fn, X_train, X_lhs, nNew=1):
    """Active learning by Cohn 1996
    Return index of nNew points which gives the largest global variance reduction

    Args:
      kernel_fn (function): Gaussian process (GP) kernel function
      X_train (array-like, size n x d): The training features $\mathbf{X}$. n is dimension size while d is number of training features.
      X_lhs (array-like, size n x s): Latin hypercube sample to estimate the improvement metric over.
      nNew (int, default=1): Number of largest values to return. ``nNew = 1`` will return the index of the largest improvement metric value, while ``nNew = len(X_lhs)`` will return a sorted list (decending) of the estimated improvement metric values.

    Returns:
      idxs (ndarray, 1D): Array of indexes of the nNew largest values in the estimated improvement metric.
      Timp (ndarray, 1D): Array of the sorted (descending) improvement metric values of the nNew largest values.

    """

    n_train = len(X_train)
    n_lhs = len(X_lhs)

    # ELD TODO: need test on size of std/lhs to be a meaningful sample size?

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
    """function to calculate row wize dot(dot(a,b),aT) where aT==a.T as in
    tmpval=[]
    for q in range(len(a)):
        tmpval.append( np.dot(np.dot(a[q, :], b), aT[:,q]))
    return np.array(tmpval)

    """
    if torch.is_tensor(a) and torch.is_tensor(b):
        return torch.einsum("ij,ji->i", torch.einsum("ij,jk", a, b), a.T)
    else:
        return np.einsum("ij,ji->i", np.einsum("ij,jk", a, b), np.array(a).T)
    # return np.einsum('ij,jk,ik->i', a,b,a)


def dotdot_a_b_aT(a, b):
    """function to calculate row wize dot(dot(a,b),aT) for all combinations
    of rows in a and cols in aT as in:

    tmpval=[]
    for q in range(len(a)):
        for qq in range(len(a))
            tmpval.append( np.dot(np.dot(a[q, :], b), aT[:,qq]))
    return np.array(tmpval)

    """
    if torch.is_tensor(a) and torch.is_tensor(b):
        return torch.einsum("ij,jk", torch.einsum("ij,jk", a, b), a.T)
    else:
        return np.einsum("ij,jk", np.einsum("ij,jk", a, b), np.array(a).T)
