#!/usr/bin/env bash
# Environment check and Python environment (reference.md, "Common setup").
#
#   bash scripts/setup_env.sh [VENV_DIR] > env.txt
#
# Verifies nvidia-smi, nvcc, ncu, nsys and compute-sanitizer, then creates a
# venv that reuses the CUDA build of PyTorch (--system-site-packages) and
# ensures triton, numpy, pandas, matplotlib, pytest and nvidia-ml-py.
# Prints everything it finds; exits non-zero only if a required tool is missing.
set -u
VENV="${1:-.venv}"
# The interpreter to build the venv from: on Kaggle, the image's CUDA-PyTorch
# Python (passed by the kernel template as CCML_PYTHON); elsewhere python3.
PYBIN="${CCML_PYTHON:-python3}"
missing=0

section() { printf '\n== %s\n' "$1"; }
need() {
  if command -v "$1" >/dev/null 2>&1; then
    printf '%s: %s\n' "$1" "$(command -v "$1")"
  else
    printf '%s: MISSING\n' "$1"
    missing=1
  fi
}

export PATH="/usr/local/cuda/bin:$PATH"

section "tools"
for t in nvidia-smi nvcc ncu nsys compute-sanitizer cmake ninja; do need "$t"; done
printf 'python: %s\n' "$PYBIN"

section "nvidia-smi"
nvidia-smi --query-gpu=name,compute_cap,driver_version,memory.total,clocks.max.sm,clocks.max.mem,power.limit --format=csv
section "nvcc";              nvcc --version | tail -2
section "ncu";               ncu --version | tail -1
section "nsys";              nsys --version
section "compute-sanitizer"; compute-sanitizer --version | tail -1
section "cmake";             cmake --version | head -1

section "venv ($VENV from $PYBIN, system site packages: reuses the CUDA PyTorch)"
if "$PYBIN" -m venv --system-site-packages "$VENV" >/dev/null 2>&1 \
   && "$VENV/bin/python" -c "import torch" >/dev/null 2>&1; then
  PY="$VENV/bin/python"
  "$PY" -m pip install -q --disable-pip-version-check \
    triton numpy pandas matplotlib pytest nvidia-ml-py 2>&1 | tail -3
else
  # Some images (Kaggle) cannot bootstrap pip into a venv. Their interpreter
  # already carries every package, so it is used directly and reported as such.
  echo "venv unusable here; using $PYBIN directly"
  PY="$PYBIN"
fi
echo "python in use: $PY"

"$PY" - <<'EOF'
import importlib
import torch
print("torch", torch.__version__, "cuda", torch.version.cuda,
      "available", torch.cuda.is_available(),
      torch.cuda.get_device_name(0) if torch.cuda.is_available() else "")
for m in ("triton", "numpy", "pandas", "matplotlib", "pytest", "pynvml"):
    try:
        mod = importlib.import_module(m)
        print(m, getattr(mod, "__version__", "ok"))
    except Exception as e:  # noqa: BLE001
        print(m, "MISSING", e)
EOF

exit $missing
