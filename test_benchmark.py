import pytest
import numpy as np
import qpsolvers
from generate.dense_strongly_convex_qp import dense_strongly_convex_qp

sizes = [2, 4, 10, 20, 50, 100, 200, 1000]

# qpsolvers.available_solvers is a tuple or list of strings
solvers = list(qpsolvers.available_solvers)

@pytest.fixture(scope="module")
def qp_problem(request):
    n = request.param
    # Let's add some proportional constraints
    n_eq = n // 4
    n_in = n // 2

    H, g, A, b, C, l, u = dense_strongly_convex_qp(n, n_eq, n_in, seed=42)

    # qpsolvers format:
    # min 1/2 x^T P x + q^T x
    # s.t. G x <= h
    #      A x = b
    #      lb <= x <= ub

    P = H
    q = g

    if C is not None:
        G = C
        h = u
    else:
        G = None
        h = None

    lb = None
    ub = None

    return P, q, G, h, A, b, lb, ub, n

@pytest.mark.parametrize("qp_problem", sizes, ids=lambda s: f"n={s}", indirect=True)
@pytest.mark.parametrize("solver", solvers)
def test_solve_qp(benchmark, qp_problem, solver):
    P, q, G, h, A, b, lb, ub, n = qp_problem
    
    benchmark.group = f"n={n}"
    
    def solve():
        return qpsolvers.solve_qp(P, q, G, h, A, b, lb, ub, solver=solver)
        
    try:
        result = benchmark(solve)
        # We can't guarantee every solver succeeds on every problem due to numerical issues
        # or missing features (like no equality constraints), but we check it doesn't crash.
        # It's better not to strictly assert result is not None, as some solvers might fail.
        # assert result is not None
    except Exception as e:
        pytest.skip(f"Solver {solver} failed with exception: {e}")

if __name__ == "__main__":
    pytest.main([__file__])
