"""T4/sm_89 summary of ccml_bench_matmul_epilogue: ccml fused (best of MMA_SYNC /
SPLIT_K) against ccml + separate kernel, cuBLAS + separate kernel and the
cuBLASLt epilogue; median ratio (competitor / fused) by shape class, and the
shapes. Ratios > 1 mean the fused ccml kernel is faster.

    python scripts/epilogue_summary.py results/sm_75/tensor_cores/matmul_epilogue.csv
"""
import statistics as st
import sys

import pandas as pd

d = pd.read_csv(sys.argv[1], comment="#")
d = d[(d.cache == "warm")]
comp = {"ours+separate": ["mma_sync+separate", "split_k+separate"],
        "cublas+separate": ["gemmex+separate"], "cublaslt_epilogue": ["lt_epilogue"]}
by, rows = {}, []
for tag, x in d.groupby("tag", sort=False):
    p = x[x.gate == "PASS"]
    f = p[p.algo.isin(["fused_mma_sync", "fused_split_k"])]
    if f.empty:
        continue
    fu = f.median_us.min()
    m = int(x.M.iloc[0])
    k = "decode" if m <= 16 else ("mid" if m <= 128 else "large")
    cells = []
    for name, algos in comp.items():
        c = p[p.algo.isin(algos)]
        if c.empty:
            cells.append("FAIL")
            continue
        r = c.median_us.min() / fu
        by.setdefault((k, name), []).append(r)
        cells.append(f"{r:.2f}x")
    rows.append(f"| {tag} | {fu:.1f} | " + " | ".join(cells) + " |")
for k in ("large", "mid", "decode"):
    print(k, "  ".join(f"{n}: median {st.median(v):.2f}x (n={len(v)})"
                       for (kk, n), v in by.items() if kk == k))
print("| shape | ccml fused us | " + " | ".join(comp) + " |")
print("|---|---|---|---|---|")
print("\n".join(rows))
