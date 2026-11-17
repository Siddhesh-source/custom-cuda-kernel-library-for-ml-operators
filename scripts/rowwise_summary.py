"""Row-wise operator summary (layernorm, softmax) from a C++ bench CSV and the
matching bench_torch.py CSV of the same session: for each shape and dtype, the
best ccml variant, the best baseline (PyTorch eager, torch.compile, Triton),
their ratio with a CI-based verdict, and the GB/s and % of measured peak
bandwidth; then the best variant per column bucket (for AUTO, enhancement S4).

    python scripts/rowwise_summary.py results/sm_75/layernorm_variants/layernorm.csv \
        results/sm_75/layernorm_variants/torch_layernorm.csv --dtype f16 [--cache warm]
        [--table] [--buckets]
"""
import argparse
import statistics as st

import pandas as pd


def bucket(cols: int) -> str:
    for edge in (32, 256, 1024, 4096):
        if cols <= edge:
            return f"<={edge}"
    return ">4096"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("ccml_csv")
    ap.add_argument("torch_csv")
    ap.add_argument("--dtype", default="f32")
    ap.add_argument("--cache", default="warm")
    ap.add_argument("--table", action="store_true")
    ap.add_argument("--buckets", action="store_true")
    a = ap.parse_args()
    d = pd.concat([pd.read_csv(a.ccml_csv, comment="#"), pd.read_csv(a.torch_csv, comment="#")])
    d = d[(d.dtype == a.dtype) & (d.cache == a.cache)]
    passed = d[d.gate == "PASS"]
    # CLAUDE.md: comparisons use the same clocks. The T4 drops SM clocks under
    # its 70 W cap, so a CI win whose baseline ran at a clock more than 5%
    # below ccml's counts only if it survives clock normalization (baseline
    # time x baseline clock / ccml clock; conservative for memory-bound
    # kernels, as in phase-11-tensor-cores.md); otherwise it is CLOCK.
    # Losses are never excused by clocks: they are reported raw.
    counts = {"WIN": 0, "LOSS": 0, "TIE": 0, "CLOCK": 0}
    ratios, rows, best_by_bucket = [], [], {}
    for tag, x in passed.groupby("tag", sort=False):
        o = x[x.impl == "ccml"]
        b = x[x.impl != "ccml"]
        if o.empty or b.empty:
            continue
        o = o.loc[o.median_us.idxmin()]
        b = b.loc[b.median_us.idxmin()]
        verdict = ("WIN" if o.ci95_hi_us < b.ci95_lo_us else
                   "LOSS" if b.ci95_hi_us < o.ci95_lo_us else "TIE")
        oc, bc = float(o.sm_clock_mhz), float(b.sm_clock_mhz)
        if verdict == "WIN" and bc > 0 and oc / bc > 1.05:
            verdict = "WIN" if b.median_us * bc / oc > o.median_us else "CLOCK"
        counts[verdict] += 1
        r = b.median_us / o.median_us
        ratios.append(r)
        best_by_bucket.setdefault((bucket(int(o.cols)), int(o.rows) <= 16), []).append(o.algo)
        rows.append(f"| {tag} | {int(o.rows)} x {int(o.cols)} | {o.algo} | {o.median_us:.1f} | "
                    f"{o.gbps:.0f} | {o.pct_peak_measured:.0f}% | {b.impl} {b.algo} | "
                    f"{b.median_us:.1f} | {r:.2f}x | {verdict} | {int(o.sm_clock_mhz)}/{int(b.sm_clock_mhz)} |")
    failed = d[(d.gate == "FAIL")]
    if a.table:
        print("| shape | rows x cols | best ccml | us | GB/s | % measured peak | best baseline | "
              "us | baseline/ccml | verdict (95% CI) | SM MHz ccml/base |")
        print("|---|---|---|---|---|---|---|---|---|---|---|")
        print("\n".join(rows))
    print(f"{a.dtype} {a.cache}: " + ", ".join(f"{k} {v}" for k, v in counts.items())
          + (f"; median baseline/ccml {st.median(ratios):.2f}x" if ratios else "")
          + f"; FAIL rows: {len(failed)} ({', '.join(sorted(set(failed.impl + ':' + failed.algo)))})")
    if a.buckets:
        for (bk, small), algos in sorted(best_by_bucket.items()):
            winner = max(set(algos), key=algos.count)
            print(f"  cols {bk:6s} rows {'<=16' if small else '>16 '}: best {winner} "
                  f"({algos.count(winner)} of {len(algos)} shapes)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
