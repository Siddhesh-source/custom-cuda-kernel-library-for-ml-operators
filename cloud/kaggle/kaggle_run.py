"""Run ccml's GPU work on Kaggle: one or several parallel sessions.

Kaggle gives roughly 30 GPU-hours a week at no cost and needs no payment method.
It is the GPU backend for all GPU work in this project (see docs/local.md for
the exceptions).

Jobs (scripts/kaggle_job.sh defines what each one does)
-------------------------------------------------------
    sanitize                    ctest, Python helper tests, compute-sanitizer
    matmul, layernorm, softmax  every implementation of one operator, C++ and
                                Python, in one session on one GPU
    all                         everything in a single session

Each job is its own Kaggle kernel (slug ccml-<job>), so jobs run in parallel.
Kaggle allows 2 concurrent GPU sessions on this account and rejects a push over
the cap, so this script pushes as slots free up. Jobs are split by operator, never by
implementation, so every comparison table comes from one physical GPU.

Usage
-----
    python cloud/kaggle/kaggle_run.py                       # all four jobs in parallel, wait, collect
    python cloud/kaggle/kaggle_run.py --jobs matmul         # one job
    python cloud/kaggle/kaggle_run.py --jobs matmul --impls ccml   # re-time only ccml
    python cloud/kaggle/kaggle_run.py --push                # push and return
    python cloud/kaggle/kaggle_run.py --fetch               # fetch and collect the jobs' outputs
    python cloud/kaggle/kaggle_run.py --no-gpu --jobs build # CPU kernel: compile check only
    python cloud/kaggle/kaggle_run.py --any-gpu             # do not pin the T4

How the source gets there
-------------------------
The whole repository is tarred, gzipped, base64-encoded and embedded in the
generated kernel script, with the commit id and the job name. No GitHub push,
Dataset or internet access is needed inside the kernel.

Credentials: ~/.kaggle/access_token (or kaggle.json). See README.md here.
"""

from __future__ import annotations

import argparse
import base64
import io
import json
import os
import pathlib
import shutil
import subprocess
import sys
import tarfile
import time

REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
HERE = pathlib.Path(__file__).resolve().parent
PUSH_ROOT = HERE / "_build"
OUTPUT_ROOT = HERE / "_output"

DEFAULT_JOBS = ["sanitize", "matmul", "layernorm", "softmax"]
KNOWN_JOBS = set(DEFAULT_JOBS) | {"all", "build"}
EXCLUDE_DIRS = {".git", "build", "out", "__pycache__", ".pytest_cache", ".vs",
                ".venv", "_build", "_output"}
# Kaggle rejects a very large script; fail loudly rather than mysteriously.
MAX_PAYLOAD_BYTES = 800_000


def slug(job: str) -> str:
    return "ccml-" + job


def config_dir() -> pathlib.Path:
    override = os.environ.get("KAGGLE_CONFIG_DIR")
    return pathlib.Path(override) if override else pathlib.Path.home() / ".kaggle"


def kaggle_username() -> str:
    """Resolve the Kaggle username, whichever auth style is configured.

    Kaggle supports two: a legacy kaggle.json holding username and key, and a
    newer single access token (KGAT_...) in access_token or KAGGLE_API_TOKEN,
    which carries no username at all. Asking the CLI works for both, so that is
    the primary path and the file is only a fallback.
    """
    if os.environ.get("KAGGLE_USERNAME"):
        return os.environ["KAGGLE_USERNAME"]

    completed = subprocess.run(
        [sys.executable, "-m", "kaggle", "config", "view"],
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    for line in completed.stdout.splitlines():
        stripped = line.strip().lstrip("- ")
        if stripped.startswith("username:"):
            username = stripped.split(":", 1)[1].strip()
            if username and username.lower() != "none":
                return username

    legacy = config_dir() / "kaggle.json"
    if legacy.is_file():
        with open(legacy, encoding="utf-8") as handle:
            return json.load(handle)["username"]

    raise SystemExit(
        "Could not determine the Kaggle username.\n\n"
        "Authenticate with either:\n"
        "  - an API token at " + str(config_dir() / "access_token")
        + " (or the\n    KAGGLE_API_TOKEN environment variable), or\n"
        "  - a legacy kaggle.json at " + str(legacy) + "\n\n"
        "Both come from https://www.kaggle.com/settings -> API.\n"
        "See cloud/kaggle/README.md.\n\n"
        "kaggle config view said:\n" + completed.stdout
    )


def commit_id() -> str:
    def git(*args: str) -> str:
        return subprocess.run(["git", *args], cwd=REPO_ROOT, stdout=subprocess.PIPE,
                              stderr=subprocess.DEVNULL, text=True).stdout.strip()
    head = git("rev-parse", "--short=12", "HEAD") or "unknown"
    return head + ("-dirty" if git("status", "--porcelain") else "")


def _add_bytes(tar: tarfile.TarFile, name: str, data: bytes) -> None:
    info = tarfile.TarInfo(name)
    info.size = len(data)
    tar.addfile(info, io.BytesIO(data))


def build_payload(job: str, impls: str) -> str:
    """tar.gz the repository and base64 it for embedding in the kernel script."""
    buffer = io.BytesIO()
    file_count = 0

    with tarfile.open(fileobj=buffer, mode="w:gz") as tar:
        # What this session is: the commit being measured (the kernel has no
        # .git; "-dirty" marks uncommitted changes), the job and its scope.
        _add_bytes(tar, ".ccml_commit", commit_id().encode("utf-8"))
        _add_bytes(tar, ".ccml_job", (job + " " + impls).encode("utf-8"))
        for root, dirs, files in os.walk(REPO_ROOT):
            dirs[:] = [d for d in dirs if d not in EXCLUDE_DIRS]
            for name in files:
                full = pathlib.Path(root) / name
                relative = full.relative_to(REPO_ROOT).as_posix()
                if set(relative.split("/")) & EXCLUDE_DIRS:
                    continue
                if full.suffix == ".sh":
                    # Shell scripts run under bash on Linux: strip any CRLF a
                    # Windows editor introduced, whatever the working copy has.
                    _add_bytes(tar, relative,
                               full.read_bytes().replace(bytes([13, 10]), bytes([10])))
                else:
                    tar.add(full, arcname=relative)
                file_count += 1

    encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
    print("packed " + str(file_count) + " files into "
          + str(round(len(encoded) / 1024, 1)) + " KB of base64")
    if len(encoded) > MAX_PAYLOAD_BYTES:
        raise SystemExit("payload is " + str(len(encoded)) + " bytes, over the "
                         + str(MAX_PAYLOAD_BYTES) + " byte guard. Add large files to "
                         "EXCLUDE_DIRS or switch to a Kaggle Dataset.")
    return encoded


def render_kernel_script(payload: str) -> str:
    template = (HERE / "kernel_template.py").read_text(encoding="utf-8")
    if "__CCML_PAYLOAD__" not in template:
        raise SystemExit("kernel_template.py is missing the __CCML_PAYLOAD__ marker")
    return template.replace("__CCML_PAYLOAD__", payload)


def write_push_dir(username: str, job: str, use_gpu: bool, payload: str,
                   pin_t4: bool) -> pathlib.Path:
    push_dir = PUSH_ROOT / job
    shutil.rmtree(push_dir, ignore_errors=True)
    push_dir.mkdir(parents=True)
    (push_dir / "ccml_kaggle.py").write_text(render_kernel_script(payload),
                                             encoding="utf-8")
    metadata = {
        "id": username + "/" + slug(job),
        "title": "ccml " + job,
        "code_file": "ccml_kaggle.py",
        "language": "python",
        "kernel_type": "script",
        "is_private": True,
        "enable_gpu": use_gpu,
        # Always a T4 (sm_75): the project's benchmark target. Never a P100.
        **({"machine_shape": "NvidiaTeslaT4"} if use_gpu and pin_t4 else {}),
        # The image already has CUDA, CMake, Ninja, PyTorch, Triton and pynvml.
        "enable_internet": False,
        "dataset_sources": [],
        "competition_sources": [],
        "kernel_sources": [],
    }
    (push_dir / "kernel-metadata.json").write_text(json.dumps(metadata, indent=2) + "\n",
                                                   encoding="utf-8")
    return push_dir


def kaggle(*args: str, check: bool = True, quiet: bool = False) -> subprocess.CompletedProcess:
    command = [sys.executable, "-m", "kaggle"] + list(args)
    completed = subprocess.run(command, stdout=subprocess.PIPE,
                               stderr=subprocess.STDOUT, text=True)
    if not quiet:
        print("\n".join(line for line in completed.stdout.splitlines()
                        if "outdated" not in line))
    if check and completed.returncode != 0:
        raise SystemExit("kaggle " + " ".join(args) + " failed")
    return completed


def status_of(kernel_id: str) -> str:
    out = kaggle("kernels", "status", kernel_id, check=False, quiet=True).stdout
    lines = [l for l in out.strip().splitlines() if "outdated" not in l]
    return lines[-1] if lines else "unknown"


def is_finished(text: str) -> bool:
    lowered = text.lower()
    return any(word in lowered for word in ("complete", "error", "cancel"))


def collect_results(output_dir: pathlib.Path) -> None:
    """Copy out/results/<arch>/ from a session's output into the repository's
    results/<arch>/, so its CSVs, peaks and sanitizer logs can be committed."""
    fetched = output_dir / "out" / "results"
    if not fetched.is_dir():
        return
    for arch_dir in sorted(p for p in fetched.iterdir() if p.is_dir()):
        target = REPO_ROOT / "results" / arch_dir.name
        shutil.copytree(arch_dir, target, dirs_exist_ok=True)
        print("collected " + str(sum(1 for _ in arch_dir.rglob("*") if _.is_file()))
              + " files into " + str(target.relative_to(REPO_ROOT)))


def fetch_output(job: str, kernel_id: str, show_log: bool) -> bool:
    """Fetch one session's output, collect its results. Returns False on failure."""
    output_dir = OUTPUT_ROOT / job
    shutil.rmtree(output_dir, ignore_errors=True)
    output_dir.mkdir(parents=True)
    kaggle("kernels", "output", kernel_id, "-p", str(output_dir), check=False, quiet=True)
    collect_results(output_dir)

    log = output_dir / "ccml_build_log.txt"
    if not log.is_file():
        print(job + ": no log retrieved; see https://www.kaggle.com/code/" + kernel_id)
        return False
    text = log.read_text(encoding="utf-8", errors="replace")
    if show_log:
        print("\n" + "#" * 72 + "\n# " + job + "\n" + "#" * 72)
        print(text)
    failed = "CCML_RUN_FAILED" in text
    print(job + ": " + ("FAILED (see " + str(log) + ")" if failed else "log at " + str(log)))
    return not failed


SESSION_CAP_MESSAGE = "Maximum batch GPU session count"


def push_with_slots(username: str, jobs: list[str], ids: dict, args) -> list[str]:
    """Push each job; Kaggle caps concurrent GPU sessions (2 on this account)
    and REJECTS a push over the cap rather than queueing it. Returns the jobs
    that were rejected for that reason, to be pushed when a slot frees."""
    rejected = []
    for job in jobs:
        push_dir = write_push_dir(username, job, args.gpu,
                                  build_payload(job, args.impls), args.pin_t4)
        out = kaggle("kernels", "push", "-p", str(push_dir), check=False, quiet=True).stdout
        if SESSION_CAP_MESSAGE in out:
            print(job + ": GPU session cap reached; will push when a slot frees")
            rejected.append(job)
        elif "successfully pushed" in out:
            print(job + ": pushed " + ids[job])
        else:
            raise SystemExit(job + ": push failed:\n" + out)
    return rejected


def schedule(username: str, pending: list[str], ids: dict, args,
             timeout_seconds: int = 8 * 3600) -> None:
    """Wait for running jobs; push pending ones as slots free; return when all
    jobs have finished."""
    start = time.time()
    active = [j for j in ids if j not in pending]
    last: dict = {}
    while time.time() - start < timeout_seconds:
        for job in active:
            text = status_of(ids[job])
            if text != last.get(job):
                print("  [" + str(int(time.time() - start)) + "s] " + job + ": "
                      + text.split(".")[-1].strip('"'))
                last[job] = text
        finished = [j for j in active if is_finished(last.get(j, ""))]
        if pending and len(active) - len(finished) < 2:
            pushed_now = [j for j in pending
                          if j not in push_with_slots(username, [j], ids, args)]
            for j in pushed_now:
                pending.remove(j)
                active.append(j)
        if not pending and all(is_finished(last.get(j, "")) for j in active):
            return
        time.sleep(30)
    print("timed out waiting; pending: " + ", ".join(pending))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--jobs", default=",".join(DEFAULT_JOBS),
                        help="comma list from: " + ", ".join(sorted(KNOWN_JOBS)))
    parser.add_argument("--impls", default="all",
                        help="limit re-measured implementations, e.g. ccml")
    parser.add_argument("--push", action="store_true", help="push and exit without waiting")
    parser.add_argument("--fetch", action="store_true", help="fetch outputs and exit")
    parser.add_argument("--show-log", action="store_true", help="print each full log")
    parser.add_argument("--resume", action="store_true",
                        help="attach to jobs already running instead of re-pushing them")
    parser.add_argument("--no-gpu", dest="gpu", action="store_false",
                        help="run on a CPU kernel: compile check only")
    parser.add_argument("--any-gpu", dest="pin_t4", action="store_false",
                        help="do not pin the T4; use if the T4 queue stalls")
    args = parser.parse_args()

    jobs = [j.strip() for j in args.jobs.split(",") if j.strip()]
    unknown = [j for j in jobs if j not in KNOWN_JOBS]
    if unknown:
        raise SystemExit("unknown job(s): " + ", ".join(unknown))

    username = kaggle_username()
    ids = {job: username + "/" + slug(job) for job in jobs}

    if not args.fetch:
        pending = []
        for job in jobs:
            # A job already running or queued (e.g. after a restart of this
            # script) is waited on, not pushed a second time.
            current = status_of(ids[job]).lower()
            if args.resume and ("running" in current or "queued" in current):
                print(job + ": already " + current.split(".")[-1].strip('"') + ", attaching")
                continue
            pending.append(job)
        pending = push_with_slots(username, pending, ids, args)
        if args.push:
            if pending:
                print("not pushed (GPU session cap): " + ", ".join(pending))
            print("\nFetch later with:\n    python cloud/kaggle/kaggle_run.py "
                  "--fetch --jobs " + ",".join(jobs))
            return
        schedule(username, pending, ids, args)

    ok = all([fetch_output(job, ids[job], args.show_log) for job in jobs])
    if not ok:
        raise SystemExit("one or more sessions failed; see the logs above")


if __name__ == "__main__":
    main()
