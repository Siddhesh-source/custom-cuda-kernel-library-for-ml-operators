"""Shape-by-shape table: our best kernel against the best cuBLAS kernel
(reference.md section 9), from a ccml_bench_matmul or ccml_bench_matmul_epilogue CSV.

    python scripts/vendor_table.py results/sm_75/tensor_cores/matmul.csv --dtype f16
        [--cache warm] [--ours ccml,ccml_tiles] [--vendor cublas,cublaslt] [--summary]

A row is a WIN or LOSS only when the bootstrap 95% CIs of the two medians do
not overlap; otherwise it is a TIE. A shape where every vendor row failed the
fp64 gate is VENDOR FAIL (no timing exists to compare). Rows are compared
only within one CSV, i.e. one session on one GPU.
"""
import argparse

import pandas as pd


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("csv")
    ap.add_argument("--dtype", default="f16")
    ap.add_argument("--cache", default="warm")
    ap.add_argument("--ours", default="ccml,ccml_tiles")
    ap.add_argument("--vendor", default="cublas,cublaslt")
    ap.add_argument("--summary", action="store_true", help="counts only")
    a = ap.parse_args()
    d = pd.read_csv(a.csv, comment="#")
    d = d[(d.dtype == a.dtype) & (d.cache == a.cache)]
    ours_impl, vendor_impl = a.ours.split(","), a.vendor.split(",")
    rows, counts = [], {"WIN": 0, "LOSS": 0, "TIE": 0, "VENDOR FAIL": 0}
    for tag, x in d.groupby("tag", sort=False):
        ours = x[x.impl.isin(ours_impl) & (x.gate == "PASS")]
        vend_all = x[x.impl.isin(vendor_impl)]
        vend = vend_all[vend_all.gate == "PASS"]
        if ours.empty or vend_all.empty:
            continue
        o = ours.loc[ours.median_us.idxmin()]
        shape = f"{int(o.M)}x{int(o.N)}x{int(o.K)}"
        if vend.empty:
            counts["VENDOR FAIL"] += 1
            rows.append(f"| {tag} | {shape} | {o.algo} | {o.median_us:.1f} | FAIL | - | - | "
                        f"{o.pct_peak_theory:.1f}% | VENDOR FAIL |")
            continue
        v = vend.loc[vend.median_us.idxmin()]
        if o.ci95_hi_us < v.ci95_lo_us:
            verdict = "WIN"
        elif v.ci95_hi_us < o.ci95_lo_us:
            verdict = "LOSS"
        else:
            verdict = "TIE"
        counts[verdict] += 1
        rows.append(f"| {tag} | {shape} | {o.algo} | {o.median_us:.1f} | {v.algo} | "
                    f"{v.median_us:.1f} | {v.median_us / o.median_us:.2f}x | "
                    f"{o.pct_peak_theory:.1f}% | {verdict} |")
    if not a.summary:
        print("| shape | M x N x K | best ccml | us | best vendor | us | vendor/ccml | "
              "ccml %peak | verdict (95% CI) |")
        print("|---|---|---|---|---|---|---|---|---|")
        print("\n".join(rows))
    print(f"\n{a.dtype} {a.cache}: " + ", ".join(f"{k} {v}" for k, v in counts.items()))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
