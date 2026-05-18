#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
from pathlib import Path
import numpy as np
import qpbenchmark
import pyarrow.parquet as pq

# Try to import proxsuite generator if available, else use custom numpy one
try:
    from proxsuite.proxqp.utils.random_qp import dense_strongly_convex_qp as px_dense_strongly_convex_qp
    HAS_PROXSUITE_GEN = True
except ImportError:
    try:
        from proxsuite.proxqp.dense.utils import generate_dense_qp as px_dense_strongly_convex_qp
        HAS_PROXSUITE_GEN = True
    except ImportError:
        HAS_PROXSUITE_GEN = False

def custom_dense_strongly_convex_qp(n, n_eq=0, n_in=0, seed=None):
    if seed is not None:
        np.random.seed(seed)
    
    # Generate strongly convex Hessian H
    H = np.random.randn(n, n)
    H = H.T @ H + 1e-2 * np.eye(n)
    
    # Generate gradient q
    q = np.random.randn(n)
    
    # Generate equality constraints
    if n_eq > 0:
        A = np.random.randn(n_eq, n)
        x0 = np.random.randn(n)
        b = A @ x0
    else:
        A = None
        b = None
        
    # Generate inequality constraints (G x <= h)
    if n_in > 0:
        G = np.random.randn(n_in, n)
        if n_eq > 0:
            pass # use x0
        else:
            x0 = np.random.randn(n)
        h = G @ x0 + np.random.rand(n_in) # make it strictly feasible
    else:
        G = None
        h = None
        
    return H, q, A, b, G, h

def generate_problems():
    problems = qpbenchmark.ProblemList()
    
    sizes = [2, 4, 10, 20, 50, 100, 200, 500, 1000]
    seeds_per_size = 5 # 5 problems per size
    
    for n in sizes:
        for seed in range(seeds_per_size):
            actual_seed = n * 100 + seed
            print(f"Generating problem size={n}, seed={actual_seed}")
            
            if HAS_PROXSUITE_GEN and False: # Fallback to custom since px structure is tricky
                pass
            else:
                H, q, A, b, G, h = custom_dense_strongly_convex_qp(n, n_eq=0, n_in=0, seed=actual_seed)
            
            # We add random box constraints
            x0 = np.random.randn(n)
            # Ensure feasibility
            lb = x0 - np.random.rand(n) * 10
            ub = x0 + np.random.rand(n) * 10
            
            problem = qpbenchmark.Problem(
                P=H,
                q=q,
                G=G,
                h=h,
                A=A,
                b=b,
                lb=lb,
                ub=ub,
                name=f"random_dense_box_{n}_{seed}"
            )
            problems.append(problem)
            
    return problems

if __name__ == "__main__":
    problems = generate_problems()
    output_dir = Path(__file__).resolve().parent.parent / "data"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "random_qpbenchmark.parquet"
    problems.to_parquet(output_path)
    print(f"Saved problems to {output_path}")

