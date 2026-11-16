"""Build and test ccml on a free Kaggle GPU kernel.

Kaggle gives roughly 30 GPU-hours a week at no cost and needs no payment method.
It is the GPU backend for all GPU work in this project (see docs/local.md for
the exceptions).

How the source gets there
-------------------------
The whole repository is tarred, gzipped, base64-encoded and embedded directly in
the generated kernel script -- about 60 KB. That avoids needing a GitHub push, a
Kaggle Dataset upload, or internet access inside the kernel. The kernel unpacks
itself and builds.

Usage
-----
    python cloud/kaggle/kaggle_run.py            # push, wait, fetch the log
    python cloud/kaggle/kaggle_run.py --push     # push and return immediately
    python cloud/kaggle/kaggle_run.py --fetch    # fetch the log of the last run
    python cloud/kaggle/kaggle_run.py --no-gpu   # CPU kernel: compile check only

Credentials
-----------
Needs ~/.kaggle/kaggle.json from https://www.kaggle.com/settings ("API" section,
"Create New Token"). See README.md in this directory.
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
PUSH_DIR = HERE / "_build"
OUTPUT_DIR = HERE / "_output"

SLUG = "ccml-build"
EXCLUDE_DIRS = {".git", "build", "out", "__pycache__", ".pytest_cache", ".vs",
                # This script's own generated push dir and fetched output.
                # Without these the payload packs a previous run's results and
                # roughly doubles in size each time.
                "_build", "_output"}
# Kaggle rejects a push whose script is very large; 60 KB is nowhere near the
# limit, but fail loudly rather than mysteriously if the repo grows.
MAX_PAYLOAD_BYTES = 800_000


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


def build_payload() -> str:
    """tar.gz the repository and base64 it for embedding in the kernel script."""
    buffer = io.BytesIO()
    file_count = 0

    with tarfile.open(fileobj=buffer, mode="w:gz") as tar:
        # The commit being measured, so every CSV row names it (the kernel has
        # no .git). "-dirty" marks uncommitted changes in the working tree.
        commit = commit_id().encode("utf-8")
        info = tarfile.TarInfo(".ccml_commit")
        info.size = len(commit)
        tar.addfile(info, io.BytesIO(commit))
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
                    data = full.read_bytes().replace(bytes([13, 10]), bytes([10]))
                    info = tar.gettarinfo(str(full), arcname=relative)
                    info.size = len(data)
                    tar.addfile(info, io.BytesIO(data))
                else:
                    tar.add(full, arcname=relative)
                file_count += 1

    encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
    print("packed " + str(file_count) + " files into "
          + str(round(len(encoded) / 1024, 1)) + " KB of base64")

    if len(encoded) > MAX_PAYLOAD_BYTES:
        raise SystemExit(
            "payload is " + str(len(encoded)) + " bytes, over the "
            + str(MAX_PAYLOAD_BYTES) + " byte guard. Add large files to "
            "EXCLUDE_DIRS or switch to a Kaggle Dataset."
        )
    return encoded


def render_kernel_script(payload: str) -> str:
    template = (HERE / "kernel_template.py").read_text(encoding="utf-8")
    if "__CCML_PAYLOAD__" not in template:
        raise SystemExit("kernel_template.py is missing the __CCML_PAYLOAD__ marker")
    return template.replace("__CCML_PAYLOAD__", payload)


def write_push_dir(username: str, use_gpu: bool, payload: str,
                   pin_t4: bool = True) -> None:
    shutil.rmtree(PUSH_DIR, ignore_errors=True)
    PUSH_DIR.mkdir(parents=True)

    (PUSH_DIR / "ccml_kaggle.py").write_text(
        render_kernel_script(payload), encoding="utf-8")

    metadata = {
        "id": username + "/" + SLUG,
        "title": "ccml build",
        "code_file": "ccml_kaggle.py",
        "language": "python",
        "kernel_type": "script",
        "is_private": True,
        "enable_gpu": use_gpu,
        # Always a T4 (sm_75): the project's benchmark target. Never a P100.
        **({"machine_shape": "NvidiaTeslaT4"} if use_gpu and pin_t4 else {}),
        # No internet needed: the source is embedded and Kaggle's image already
        # has the CUDA toolkit, numpy, pytest and torch.
        "enable_internet": False,
        "dataset_sources": [],
        "competition_sources": [],
        "kernel_sources": [],
    }
    (PUSH_DIR / "kernel-metadata.json").write_text(
        json.dumps(metadata, indent=2) + "\n", encoding="utf-8")


def kaggle(*args: str, check: bool = True) -> subprocess.CompletedProcess:
    command = [sys.executable, "-m", "kaggle"] + list(args)
    completed = subprocess.run(
        command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    print(completed.stdout, end="")
    if check and completed.returncode != 0:
        raise SystemExit("kaggle " + " ".join(args) + " failed")
    return completed


def wait_for_completion(kernel_id: str, timeout_seconds: int = 6 * 3600) -> str:
    """Poll until the kernel stops running. Returns the final status text."""
    print("\nwaiting for the kernel to finish (Ctrl-C to stop waiting; the run "
          "continues on Kaggle)")
    deadline = time.time() + timeout_seconds
    last = ""

    while time.time() < deadline:
        completed = kaggle("kernels", "status", kernel_id, check=False)
        text = completed.stdout.strip()
        if text != last:
            print("  status: " + text)
            last = text
        lowered = text.lower()
        if "complete" in lowered or "error" in lowered or "cancel" in lowered:
            return text
        time.sleep(20)

    return "timed out after " + str(timeout_seconds) + "s: " + last


def fetch_output(kernel_id: str) -> None:
    shutil.rmtree(OUTPUT_DIR, ignore_errors=True)
    OUTPUT_DIR.mkdir(parents=True)
    kaggle("kernels", "output", kernel_id, "-p", str(OUTPUT_DIR), check=False)

    print("\n" + "#" * 72)
    print("# KERNEL OUTPUT")
    print("#" * 72)

    collect_results()

    log = OUTPUT_DIR / "ccml_build_log.txt"
    if log.is_file():
        text = log.read_text(encoding="utf-8", errors="replace")
        print(text)
        if "CCML_RUN_FAILED" in text:
            raise SystemExit("the remote run failed; see the log above")
        return

    # Fall back to whatever Kaggle returned, usually <slug>.log as JSON records.
    found = sorted(OUTPUT_DIR.glob("*.log")) + sorted(OUTPUT_DIR.glob("*.txt"))
    if not found:
        print("no output files retrieved; check "
              "https://www.kaggle.com/code/" + kernel_id.replace("/", "/") )
        return
    for path in found:
        print("--- " + path.name + " ---")
        print(path.read_text(encoding="utf-8", errors="replace"))


def collect_results() -> None:
    """Copy out/results/<arch>/ from the fetched output into the repository's
    results/<arch>/, so a run's CSVs, peaks and sanitizer logs can be committed."""
    fetched = OUTPUT_DIR / "out" / "results"
    if not fetched.is_dir():
        return
    for arch_dir in sorted(p for p in fetched.iterdir() if p.is_dir()):
        target = REPO_ROOT / "results" / arch_dir.name
        shutil.copytree(arch_dir, target, dirs_exist_ok=True)
        print("collected " + str(sum(1 for _ in arch_dir.rglob("*"))) + " files into "
              + str(target.relative_to(REPO_ROOT)))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--push", action="store_true",
                        help="push and exit without waiting")
    parser.add_argument("--fetch", action="store_true",
                        help="fetch the last run's output and exit")
    parser.add_argument("--no-gpu", dest="gpu", action="store_false",
                        help="run on a CPU kernel: compile check only")
    parser.add_argument("--any-gpu", dest="pin_t4", action="store_false",
                        help="do not pin the T4; use if the T4 queue stalls")
    args = parser.parse_args()

    username = kaggle_username()
    kernel_id = username + "/" + SLUG
    print("kernel: " + kernel_id)

    if args.fetch:
        fetch_output(kernel_id)
        return

    write_push_dir(username, args.gpu, build_payload(), args.pin_t4)
    print("accelerator: " + ("GPU" if args.gpu else "none (CPU)"))
    kaggle("kernels", "push", "-p", str(PUSH_DIR))

    if args.push:
        print("\npushed. Fetch results later with:\n"
              "    python cloud/kaggle/kaggle_run.py --fetch")
        return

    status = wait_for_completion(kernel_id)
    print("\nfinal status: " + status)
    fetch_output(kernel_id)


if __name__ == "__main__":
    main()
