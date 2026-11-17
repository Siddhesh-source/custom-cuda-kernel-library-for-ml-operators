"""Print the metrics of scripts/ncu_matmul.ps1's *_metrics.csv files side by side.

    python scripts/ncu_table.py profiles/sm_89/matmul/wmma_vs_mma_sync_*_metrics.csv
"""
import csv
import glob
import sys


def main() -> int:
    for f in sorted(p for a in sys.argv[1:] for p in glob.glob(a)):
        rows = list(csv.reader(open(f, encoding="utf-8-sig")))
        start = next(i for i, r in enumerate(rows) if r and r[0] == "ID")
        hdr, data = rows[start], rows[start + 2:]
        print("==", f)
        k = hdr.index("Kernel Name")
        names = [r[k].split("<")[0].split()[-1] for r in data]
        print(f"  {'metric':76s}" + "".join(f"{n[:16]:>18s}" for n in names))
        for i, h in enumerate(hdr):
            if i <= k + 6:
                continue
            print(f"  {h[:76]:76s}" + "".join(f"{r[i][:16]:>18s}" for r in data))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
