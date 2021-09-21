import pyDOE2
import numpy as np

def fullfact_with_bounds(LBs, UBs, N_xi):
    """
    Return a ND array of corresponding (x_0, ..., x_i) values that span out the inputspace
    between lowerbound and upperbound in a structured (grid-like) way with n_x_i points in 
    the i'th-dimension.
    
    **Parameters:**
    LBs: list-like 1D, lower bounds of the input space. len(LBs) must equal len(UBs)
    UBs: list-like 1D, upper bounds of the input space. len(UBs) must equal len(LBs)
    N_xi: list-like, 
    
    **Return:**
    array like [[x_0,...,x_i],...,[x_0_n,...,x_i_n]]
    
    """
    LBs=np.array(LBs)
    UBs=np.array(UBs)
    N_xi=np.array(N_xi)
    
    if not LBs.shape==UBs.shape==N_xi.shape:
        raise ValueError("Shape of LBs, UBs, N_xi must be equal")
    if np.any(N_xi < 1):
        raise ValueError("All elements of N_xi must be greater than or equal to 1")
    if np.any(LBs>UBs):
        raise ValueError("All elements of LBs must be less than the corresponding elements of UBs")
    
    
    ffact = pyDOE2.fullfact(N_xi)
    
    return LBs + ffact/ffact.max(axis=0)*(UBs-LBs)


def lhs(nDim, nSamples, LBs, UBs, random_state=None):
    """
    Return a 2D array of corresponding (x, y) values that fill the inputspace
    between lowerbound and upperbound with n points using a Latin-hypercube design.
    Use random seed 42.
    
    """
    ### BEGIN SOLUTION
    LBs=np.array(LBs)
    UBs=np.array(UBs)
    
    lhs = pyDOE2.lhs(nDim, samples=nSamples, random_state=random_state)
    return LBs + lhs*(UBs-LBs)
    ### END SOLUTION