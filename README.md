# Random QP Benchmark

This is a new benchmark for `qpbenchmark` containing randomly generated strongly convex Quadratic Programming (QP) problems with box constraints. It is inspired by `ik_qpbenchmark` and `maros_meszaros_qpbenchmark`.

The problems are generated according to the same methodology as ProxSuite's `dense_strongly_convex_qp` utility.

## Setup

This repository uses `pixi` for environment management. To set up the environment, run:
```bash
pixi install
```

## Generation

The dataset of randomly generated QPs is saved as a Parquet file in `data/random_qpbenchmark.parquet`. 
To regenerate the dataset, run:
```bash
pixi run generate
```

This will generate strongly convex QP problems with box constraints across a variety of sizes (2, 4, 10, 20, 50, 100, 200, 500, 1000).

## Usage

To run the benchmark against all available solvers:
```bash
pixi run python random_qpbenchmark.py run
```

To list all problems in the test set:
```bash
pixi run python random_qpbenchmark.py list_problems
```

## Problem Format

The QPs in this benchmark follow the standard form used by `qpbenchmark` (via `qpsolvers`):
```
min 1/2 x^T P x + q^T x
s.t. 
A x = b
G x <= h
lb <= x <= ub
```
In this particular benchmark, `P` is always dense and strictly positive-definite. Constraint matrices `A` and `G` are empty, and only `lb` and `ub` are active (box constraints).
