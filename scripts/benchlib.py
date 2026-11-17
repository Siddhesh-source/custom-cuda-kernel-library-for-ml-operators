"""Measurement helpers shared by the Python baselines (scripts/bench_torch.py).

Mirrors bench/support/ in C++ so both sides produce the same CSV schema with
the same statistics, the same correctness gate and the same telemetry
(reference.md, "Benchmarking method"; docs/benchmarking.md). Kept in step with
tests/test_bench_stats.cpp by scripts/tests/test_benchlib.py.
"""

from __future__ import annotations

import json
import math
import os
from dataclasses import dataclass, field
from typing import Callable, Optional, Sequence

CSV_HEADER = (
    "op,impl,algo,dtype,tag,M,N,K,rows,cols,cache,iters,median_us,p10_us,"
    "p90_us,iqr_us,min_us,ci95_lo_us,ci95_hi_us,batched_mean_us,tflops,gbps,"
    "pct_peak_theory,pct_peak_measured,max_abs_err,max_rel_err,gate,"
    "sm_clock_mhz,mem_clock_mhz,temp_c,clock_reasons,throttled,gpu,cc,"
    "driver,cuda,commit"
)

FLOAT32_EPS = 1.1920928955078125e-07


# ------------------------------------------------------------------ stats --

def percentile_sorted(values: Sequence[float], q: float) -> float:
    """Linear interpolation, numpy's default definition."""
    if not values:
        return 0.0
    q = min(1.0, max(0.0, q))
    pos = q * (len(values) - 1)
    lo = int(math.floor(pos))
    hi = min(lo + 1, len(values) - 1)
    return values[lo] + (values[hi] - values[lo]) * (pos - lo)


def median(values: Sequence[float]) -> float:
    return percentile_sorted(sorted(values), 0.5)


def _splitmix64(state: int):
    mask = (1 << 64) - 1
    while True:
        state = (state + 0x9E3779B97F4A7C15) & mask
        z = state
        z = ((z ^ (z >> 30)) * 0xBF58476D1CE4E5B9) & mask
        z = ((z ^ (z >> 27)) * 0x94D049BB133111EB) & mask
        yield z ^ (z >> 31)


def bootstrap_median_ci(samples: Sequence[float], resamples: int = 2000,
                        seed: int = 0x5EED) -> tuple[float, float]:
    """Percentile bootstrap of the median, same generator as the C++ side."""
    if not samples or resamples <= 0:
        return 0.0, 0.0
    n = len(samples)
    rng = _splitmix64(seed)
    medians = sorted(median([samples[next(rng) % n] for _ in range(n)])
                     for _ in range(resamples))
    return percentile_sorted(medians, 0.025), percentile_sorted(medians, 0.975)


@dataclass
class Summary:
    n: int = 0
    median: float = 0.0
    p10: float = 0.0
    p90: float = 0.0
    iqr: float = 0.0
    min: float = 0.0
    max: float = 0.0
    mean: float = 0.0
    ci_lo: float = 0.0
    ci_hi: float = 0.0
    valid: bool = False


def summarize(samples: Sequence[float], resamples: int = 2000) -> Summary:
    if not samples:
        return Summary()
    s = sorted(samples)
    lo, hi = bootstrap_median_ci(list(samples), resamples)
    return Summary(n=len(s), median=percentile_sorted(s, 0.5),
                   p10=percentile_sorted(s, 0.1), p90=percentile_sorted(s, 0.9),
                   iqr=percentile_sorted(s, 0.75) - percentile_sorted(s, 0.25),
                   min=s[0], max=s[-1], mean=sum(s) / len(s), ci_lo=lo, ci_hi=hi,
                   valid=True)


def choose_iterations(estimate_ms: float, min_iters: int = 100,
                      max_iters: int = 200, budget_ms: float = 3000.0) -> int:
    if not estimate_ms > 0.0:
        return max_iters
    fit = budget_ms / estimate_ms
    n = max_iters if fit >= max_iters else int(fit)
    return max(min_iters, min(max_iters, n))


# -------------------------------------------------------------- metrics --

def matmul_flops(m: int, n: int, k: int) -> float:
    return 2.0 * m * n * k


def matmul_bytes(m: int, n: int, k: int, elem: int) -> float:
    return float(m * k + k * n + m * n) * elem


def layernorm_bytes(rows: int, cols: int, elem: int, affine: bool = True) -> float:
    return float(2 * rows * cols + (2 * cols if affine else 0)) * elem


def softmax_bytes(rows: int, cols: int, elem: int) -> float:
    return float(2 * rows * cols) * elem


def is_throttled(reasons: int) -> bool:
    return bool(reasons & (0x08 | 0x20 | 0x40 | 0x80))


# ----------------------------------------------------------------- gate --

def tolerance_reduction(terms: int) -> tuple[float, float]:
    g = math.sqrt(max(terms, 1))
    t = 8.0 * FLOAT32_EPS * g
    return t, t


def tolerance_softmax(cols: int) -> tuple[float, float]:
    """Relative, with an absolute floor at FLT_MIN (support/compare.h)."""
    return 1.1754943508222875e-38, tolerance_reduction(cols)[1]


def tolerance_dot(terms: int, scale: float) -> tuple[float, float]:
    abs_t, rel_t = tolerance_reduction(terms)
    return abs_t * (scale if scale > 0 else 1.0), rel_t


def compare(got, ref, abs_tol: float, rel_tol: float):
    """got, ref: torch tensors (ref float64). Returns (pass, max_abs, max_rel)."""
    import torch
    g = got.double()
    if not torch.isfinite(g).all():
        return False, float("inf"), float("inf")
    diff = (g - ref).abs()
    denom = ref.abs()
    max_abs = diff.max().item()
    rel = torch.where(denom > 0, diff / denom, diff)
    max_rel = rel.max().item()
    normalized = (diff / (abs_tol + rel_tol * denom)).max().item()
    return normalized <= 1.0, max_abs, max_rel


# ------------------------------------------------------------ telemetry --

class Telemetry:
    def __init__(self) -> None:
        self.ok = False
        try:
            import pynvml
            pynvml.nvmlInit()
            self.nvml = pynvml
            self.handle = pynvml.nvmlDeviceGetHandleByIndex(0)
            self.ok = True
        except Exception:
            self.nvml = None

    def driver(self) -> str:
        if not self.ok:
            return "unknown"
        v = self.nvml.nvmlSystemGetDriverVersion()
        return v.decode() if isinstance(v, bytes) else str(v)

    def other_processes(self) -> int:
        if not self.ok:
            return -1
        try:
            procs = self.nvml.nvmlDeviceGetComputeRunningProcesses(self.handle)
            return sum(1 for p in procs if p.pid != os.getpid())
        except Exception:
            return -1

    def sample(self) -> dict:
        if not self.ok:
            return {}
        n = self.nvml
        out = {"sm": n.nvmlDeviceGetClockInfo(self.handle, n.NVML_CLOCK_SM),
               "mem": n.nvmlDeviceGetClockInfo(self.handle, n.NVML_CLOCK_MEM),
               "temp": n.nvmlDeviceGetTemperature(self.handle, n.NVML_TEMPERATURE_GPU)}
        getter = (getattr(n, "nvmlDeviceGetCurrentClocksEventReasons", None)
                  or getattr(n, "nvmlDeviceGetCurrentClocksThrottleReasons"))
        out["reasons"] = int(getter(self.handle))
        return out


# --------------------------------------------------------------- timing --

@dataclass
class Candidate:
    impl: str
    algo: str
    fn: Callable[[], object]


@dataclass
class Measured:
    summary: Summary = field(default_factory=Summary)
    batched_mean_us: float = 0.0
    telemetry: dict = field(default_factory=dict)  # sampled mid-run
    valid: bool = False
    error: str = ""


def _spin_cycles(guard_us: float, tel: "Telemetry | None") -> int:
    """Cycles for the launch-gap guard (bench/support/spin.h explains why).
    Uses the GPU's max SM clock when known, so the guard is never shorter
    than intended."""
    mhz = 1600
    if tel is not None and tel.ok:
        try:
            mhz = tel.nvml.nvmlDeviceGetMaxClockInfo(tel.handle, tel.nvml.NVML_CLOCK_SM)
        except Exception:  # noqa: BLE001
            pass
    return int(guard_us * mhz)


def time_interleaved(cands: Sequence[Candidate], cold: bool, flush_buf,
                     warmup: int = 20, min_iters: int = 100, max_iters: int = 200,
                     budget_ms: float = 3000.0, rounds: int = 4,
                     guard_us: float = 200.0, tel: "Telemetry | None" = None) -> list[Measured]:
    """Same method as bench/support/timer.cpp: warmup, per-iteration CUDA
    events each preceded by a launch-gap guard, interleaved rounds, optional
    L2 flush outside the timed region, telemetry sampled mid-run, plus the
    spec's batched pattern as a cross-check (warm only)."""
    import torch
    spin = _spin_cycles(guard_us, tel)
    out = [Measured() for _ in cands]
    iters = [max_iters] * len(cands)
    samples: list[list[float]] = [[] for _ in cands]
    alive = [True] * len(cands)

    for i, c in enumerate(cands):
        try:
            for _ in range(warmup):
                c.fn()
            torch.cuda.synchronize()
            a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            a.record()
            for _ in range(5):
                c.fn()
            b.record()
            b.synchronize()
            iters[i] = choose_iterations(a.elapsed_time(b) / 5, min_iters, max_iters, budget_ms)
        except Exception as e:  # noqa: BLE001 -- any failure means "not timed"
            alive[i], out[i].error = False, f"{type(e).__name__}: {e}"

    for r in range(max(1, rounds)):
        for i, c in enumerate(cands):
            if not alive[i]:
                continue
            per_round = -(-iters[i] // rounds)
            try:
                starts = [torch.cuda.Event(enable_timing=True) for _ in range(per_round)]
                stops = [torch.cuda.Event(enable_timing=True) for _ in range(per_round)]
                for k in range(per_round):
                    if cold:
                        flush_buf.zero_()
                    if spin > 0:
                        torch.cuda._sleep(spin)
                    starts[k].record()
                    c.fn()
                    stops[k].record()
                if tel is not None and r == rounds - 1:
                    out[i].telemetry = tel.sample()
                stops[-1].synchronize()
                samples[i].extend(1e3 * s.elapsed_time(e) for s, e in zip(starts, stops))
            except Exception as e:  # noqa: BLE001
                alive[i], out[i].error = False, f"{type(e).__name__}: {e}"

    for i, c in enumerate(cands):
        if not alive[i]:
            continue
        batched = 0.0
        if not cold:
            a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            a.record()
            for _ in range(iters[i]):
                c.fn()
            b.record()
            b.synchronize()
            batched = 1e3 * a.elapsed_time(b) / iters[i]
        out[i].summary = summarize(samples[i])
        out[i].batched_mean_us = batched
        out[i].valid = out[i].summary.valid and out[i].summary.n >= min_iters
    return out


# ------------------------------------------------------------------ CSV --

def _fmt(v: float, digits: int = 6) -> str:
    return f"{v:.{digits}g}"


def csv_row(*, op: str, impl: str, algo: str, dtype: str, shape: dict, cold: bool,
            gate: str, max_abs: Optional[float], max_rel: Optional[float],
            flops: float, nbytes: float, measured: Optional[Measured], tel: dict,
            ctx: dict) -> str:
    timed = gate == "PASS" and measured is not None and measured.valid
    s = measured.summary if timed else Summary()
    sec = s.median * 1e-6
    is_mm = flops > 0
    tflops = flops / sec / 1e12 if timed and is_mm else 0.0
    gbps = nbytes / sec / 1e9 if timed else 0.0
    achieved = tflops if is_mm else gbps
    theory = ctx["fp32_tflops_theory"] if is_mm else ctx["bandwidth_gbps_theory"]
    meas = ctx["fp32_tflops_measured"] if is_mm else ctx["bandwidth_gbps_measured"]
    peak_ok = timed and (not is_mm or dtype == "f32")

    def opt(v, ok, d=6):
        return _fmt(v, d) if ok else ""

    cols = [op, impl, algo, dtype, shape.get("tag", ""),
            str(shape.get("M", 0)), str(shape.get("N", 0)), str(shape.get("K", 0)),
            str(shape.get("rows", 0)), str(shape.get("cols", 0)),
            "cold" if cold else "warm", str(s.n) if timed else "",
            opt(s.median, timed), opt(s.p10, timed), opt(s.p90, timed),
            opt(s.iqr, timed), opt(s.min, timed), opt(s.ci_lo, timed),
            opt(s.ci_hi, timed),
            opt(measured.batched_mean_us if measured else 0.0, timed and not cold),
            opt(tflops, timed and is_mm, 5), opt(gbps, timed, 5),
            opt(100 * achieved / theory if theory > 0 else 0.0, peak_ok and theory > 0, 4),
            opt(100 * achieved / meas if meas > 0 else 0.0, peak_ok and meas > 0, 4),
            opt(max_abs or 0.0, max_abs is not None, 4),
            opt(max_rel or 0.0, max_rel is not None, 4), gate]
    if tel:
        cols += [str(tel["sm"]), str(tel["mem"]), str(tel["temp"]),
                 hex(tel["reasons"]), "1" if is_throttled(tel["reasons"]) else "0"]
    else:
        cols += [""] * 5
    cols += [f'"{ctx["gpu"]}"', ctx["cc"], ctx["driver"], ctx["cuda"], ctx["commit"]]
    return ",".join(cols)


def load_peaks(path: Optional[str]) -> dict:
    if not path or not os.path.exists(path):
        return {}
    with open(path, encoding="utf-8") as f:
        return json.load(f)
