import numpy as np
import scipy.sparse as spa

def dense_strongly_convex_qp(n, n_eq, n_in, seed=None):
    if seed is not None:
        np.random.seed(seed)
    
    # Generate strongly convex Hessian H
    H = np.random.randn(n, n)
    H = H.T @ H + 1e-2 * np.eye(n)
    
    # Generate gradient g
    g = np.random.randn(n)
    
    # Generate equality constraints
    if n_eq > 0:
        A = np.random.randn(n_eq, n)
        b = np.random.randn(n_eq)
    else:
        A = None
        b = None
        
    # Generate inequality constraints
    if n_in > 0:
        C = np.random.randn(n_in, n)
        u = np.random.randn(n_in)
        # we need to make sure the problem is feasible.
        # let's generate a feasible point x0
        x0 = np.random.randn(n)
        if A is not None:
            # project x0 onto Ax = b
            # x0 = x0 - A^T (A A^T)^-1 (A x0 - b)
            # a simple way is to re-assign b = A @ x0
            b = A @ x0
            
        u = C @ x0 + np.random.rand(n_in) # make it strictly feasible
        l = np.full(n_in, -np.inf)
    else:
        C = None
        u = None
        l = None
        
    return H, g, A, b, C, l, u

