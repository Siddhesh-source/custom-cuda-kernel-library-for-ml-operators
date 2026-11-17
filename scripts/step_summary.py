"""Per-step summary of a ccml_bench_matmul CSV: for each (prev, new) rung pair,
the median speedup new/prev and the median new-vs-best-vendor ratio, by shape
class (decode M <= 16, mid M <= 128, large M >= 512), plus the 5-line table
for a few representative shapes.

    python scripts/step_summary.py results/sm_75/tensor_cores/matmul.csv --dtype f16 \
        --pairs double_buffered:wmma,wmma:mma_sync,mma_sync:split_k,mma_sync:mma_sync_swizzled
"""
import argparse
import statistics as st

import pandas as pd

SHOW = ["gpt2m.ffn_up.m2048", "gpt2s.ffn_up.m512", "gpt2s.qkv.m128", "gpt2s.qkv.m16",
        "gpt2s.ffn_down.m1", "llama1b.ffn_up.m2048"]


def cls(m: int) -> str:
    return "decode" if m <= 16 else ("mid" if m <= 128 else "large")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("csv")
    ap.add_argument("--dtype", default="f16")
    ap.add_argument("--cache", default="warm")
    ap.add_argument("--pairs", required=True)
    a = ap.parse_args()
    d = pd.read_csv(a.csv, comment="#")
    d = d[(d.dtype == a.dtype) & (d.cache == a.cache) & (d.gate == "PASS")]
    for pair in a.pairs.split(","):
        prev, new = pair.split(":")
        print(f"\n## {prev} -> {new} ({a.dtype}, {a.cache})")
        by = {}
        table = []
        for tag, x in d.groupby("tag", sort=False):
            p = x[(x.impl == "ccml") & (x.algo == prev)]
            n = x[(x.impl == "ccml") & (x.algo == new)]
            v = x[x.impl.isin(["cublas", "cublaslt"])]
            if p.empty or n.empty:
                continue
            p, n = p.iloc[0], n.iloc[0]
            vend = v.median_us.min() if not v.empty else float("nan")
            by.setdefault(cls(int(n.M)), []).append(
                (p.median_us / n.median_us, vend / n.median_us))
            if tag in SHOW:
                table.append(f"| {tag} {int(n.M)}x{int(n.N)}x{int(n.K)} | {p.median_us:.1f} | "
                             f"{n.median_us:.1f} | "
                             + (f"{vend:.1f}" if vend == vend else "FAIL")
                             + f" | {n.pct_peak_theory:.1f}% | {int(n.sm_clock_mhz)} |")
        for k in ("large", "mid", "decode"):
            if k in by:
                sp = [s for s, _ in by[k]]
                vr = [r for _, r in by[k] if r == r]
                print(f"  {k:7s} n={len(sp):3d}  median {new}/{prev} = {st.median(sp):.2f}x"
                      + (f"  median vendor/{new} = {st.median(vr):.2f}x" if vr else ""))
        print(f"| shape | {prev} us | {new} us | best cuBLAS us | {new} %peak | SM MHz |")
        print("|---|---|---|---|---|---|")
        print("\n".join(table))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
