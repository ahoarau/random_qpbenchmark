#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import qpbenchmark
from qpbenchmark.benchmark import main

class RandomQpbenchmark(qpbenchmark.ParquetTestSet):
    """Random test set with dense strongly convex QPs and box constraints."""

    @property
    def description(self) -> str:
        return (
            "Randomly generated dense strongly convex QP problems "
            "with box constraints."
        )

    @property
    def title(self) -> str:
        return "Random QP test set"

    @property
    def sparse_only(self) -> bool:
        return False

    def __init__(self):
        script_dir = Path(__file__).resolve().parent
        self.parquet_path = script_dir / "data" / "random_qpbenchmark.parquet"
        qpbenchmark.TestSet.__init__(self)

    def __iter__(self):
        limit = getattr(self, "limit", 0)
        count = 0
        parquet_file = pq.ParquetFile(self.parquet_path)
        for batch in parquet_file.iter_batches():
            df = batch.to_pandas()
            for _, row in df.iterrows():
                if limit and count >= limit:
                    return
                n = row["q"].size
                pb_data = {}
                for key in qpbenchmark.ProblemList.KEYS:
                    if isinstance(row[key], np.ndarray):
                        # Copy before reshape (reshape requires a writeable array)
                        pb_data[key] = row[key].copy()
                        if key in ("P", "G", "A", "H"):
                            # Parquet unrolls matrices to 1D, so reshape if needed
                            # The shape might depend on n and m
                            # For H, P it is usually (n, n)
                            if key in ("H", "P") and pb_data[key].size > 0:
                                pb_data[key] = pb_data[key].reshape((n, n))
                            elif key == "G" and pb_data[key].size > 0:
                                m = pb_data[key].size // n
                                pb_data[key] = pb_data[key].reshape((m, n))
                            elif key == "A" and pb_data[key].size > 0:
                                m = pb_data[key].size // n
                                pb_data[key] = pb_data[key].reshape((m, n))
                    else:  # string or None
                        pb_data[key] = row[key]
                yield qpbenchmark.Problem(**pb_data)
                count += 1

    def count_problems(self) -> int:
        parquet_file = pq.ParquetFile(self.parquet_path)
        return parquet_file.metadata.num_rows

if __name__ == "__main__":
    test_set_path = Path(__file__).resolve()
    results_path = (
        test_set_path.parent / "results" / "qpbenchmark_results.parquet"
    )
    results_path.parent.mkdir(parents=True, exist_ok=True)
    main(test_set_path=test_set_path, results_path=results_path)
