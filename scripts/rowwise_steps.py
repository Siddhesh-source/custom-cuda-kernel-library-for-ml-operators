"""Per-step summary for a row-wise operator CSV (ccml_bench_layernorm /
ccml_bench_softmax and their fused benches): for each rung pair prev:new, the
median ratio prev/new over shapes, by shape class, per dtype; optionally only
shapes matching a column / row predicate.

    python scripts/rowwise_steps.py results/sm_75/softmax_variants/softmax.csv \
        --pairs block_per_row:online,online:warp_per_row,online:split_row \
        [--cache warm] [--max-cols 1024] [--max-rows 16] [--min-cols 4097]

Classes: small (rows <= 16), mid (rows <= 1024), large (rows > 1024).
"""
import argparse
import statistics as st

import pandas as pd


def cls(rows: int) -> str:
    return "small" if rows <= 16 else ("mid" if rows <= 1024 else "large")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("csv")
    ap.add_argument("--pairs", required=True)
    ap.add_argument("--cache", default="warm")
    ap.add_argument("--max-cols", type=int, default=1 << 30)
    ap.add_argument("--min-cols", type=int, default=0)
    ap.add_argument("--max-rows", type=int, default=1 << 30)
    a = ap.parse_args()
    d = pd.read_csv(a.csv, comment="#")
    d = d[(d.gate == "PASS") & (d.cache == a.cache) & (d.cols <= a.max_cols) &
          (d.cols >= a.min_cols) & (d.rows <= a.max_rows)]
    for pair in a.pairs.split(","):
        prev, new = pair.split(":")
        print(f"## {prev} -> {new} ({a.cache})")
        for dt, x in d.groupby("dtype"):
            by = {}
            for _, y in x.groupby(["tag"]):
                p, n = y[y.algo == prev], y[y.algo == new]
                if p.empty or n.empty:
                    continue
                by.setdefault(cls(int(n.rows.iloc[0])), []).append(
                    (p.median_us.iloc[0] / n.median_us.iloc[0], n.gbps.iloc[0]))
            parts = [f"{k} n={len(v)} {st.median(r for r, _ in v):.2f}x "
                     f"(max {max(r for r, _ in v):.2f}x, new median {st.median(g for _, g in v):.0f} GB/s)"
                     for k, v in sorted(by.items())]
            print(f"  {dt}: " + "; ".join(parts))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
