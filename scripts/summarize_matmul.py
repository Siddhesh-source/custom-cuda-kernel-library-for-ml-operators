"""Per-shape summary of a matmul CSV (ccml_bench_matmul output): previous rung,
new rung, best ccml row, cuBLAS, %peak and clocks, for the optimization log.

    python scripts/summarize_matmul.py --prev double_buffered --new wmma \
        --dtype f16 results/sm_89/wmma_*.csv
"""
import argparse
import glob

import pandas as pd


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("files", nargs="+")
    ap.add_argument("--prev", required=True, help="algo of the previous rung")
    ap.add_argument("--new", required=True, help="algo of the new rung")
    ap.add_argument("--dtype", default="f16")
    ap.add_argument("--cache", default="warm")
    a = ap.parse_args()
    paths = sorted(p for f in a.files for p in glob.glob(f))
    d = pd.concat([pd.read_csv(p, comment="#") for p in paths])
    d = d[(d.gate == "PASS") & (d.dtype == a.dtype) & (d.cache == a.cache)]
    print(f"| shape | {a.prev} us | {a.new} us | best ccml (row) | cuBLAS us | {a.new} %peak "
          f"(theory / measured) | new vs prev | new vs cuBLAS | SM MHz |")
    print("|---|---|---|---|---|---|---|---|---|")
    for tag, x in d.groupby("tag", sort=False):
        prev = x[(x.impl == "ccml") & (x.algo == a.prev)]
        new = x[(x.impl == "ccml") & (x.algo == a.new)]
        cub = x[x.impl == "cublas"]
        ours = x[x.impl.isin(["ccml", "ccml_tiles"])]
        if prev.empty or new.empty or cub.empty:
            continue
        p, n, c = prev.iloc[0], new.iloc[0], cub.iloc[0]
        b = ours.loc[ours.median_us.idxmin()]
        clocks = sorted(set(x.sm_clock_mhz.dropna().astype(int)))
        print(f"| {tag} {int(n.M)}x{int(n.N)}x{int(n.K)} | {p.median_us:.1f} | {n.median_us:.1f} | "
              f"{b.median_us:.1f} ({b.algo}) | {c.median_us:.1f} | "
              f"{n.pct_peak_theory:.1f}% / {n.pct_peak_measured:.1f}% | "
              f"{p.median_us / n.median_us:.2f}x | {c.median_us / n.median_us:.2f}x | "
              f"{clocks[0]}-{clocks[-1]} |")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
