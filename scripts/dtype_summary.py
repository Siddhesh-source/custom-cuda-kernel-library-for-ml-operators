"""Median over shapes of (best vendor / best ccml) per dtype, overall and by
shape class, from a ccml_bench_matmul CSV (warm L2). > 1: ccml faster.

    python scripts/dtype_summary.py results/sm_75/tensor_cores/matmul.csv
"""
import statistics as st
import sys

import pandas as pd

d = pd.read_csv(sys.argv[1], comment="#")
d = d[(d.cache == "warm") & (d.gate == "PASS")]
for dt, x in d.groupby("dtype"):
    by = {}
    for tag, y in x.groupby("tag"):
        o = y[y.impl.isin(["ccml", "ccml_tiles"])]
        v = y[y.impl.isin(["cublas", "cublaslt"])]
        if o.empty or v.empty:
            continue
        m = int(y.M.iloc[0])
        k = "decode" if m <= 16 else ("mid" if m <= 128 else "large")
        r = v.median_us.min() / o.median_us.min()
        by.setdefault(k, []).append(r)
        by.setdefault("all", []).append(r)
    print(dt, "  ".join(f"{k}: {st.median(v):.2f}x (n={len(v)})" for k, v in by.items()))
