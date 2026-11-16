#!/usr/bin/env bash
# The work one Kaggle GPU session performs, after cloud/kaggle/kernel_template.py
# has unpacked and built the repository. Batched into a single session to save
# GPU quota (enhancement G5).
#
#   bash scripts/kaggle_job.sh SOURCE_DIR BUILD_DIR OUT_DIR
#
# Everything under OUT_DIR/results/<arch>/ is copied back into the repository
# by `python cloud/kaggle/kaggle_run.py --fetch`.
set -u
SRC="${1:?source dir}"
BUILD="${2:?build dir}"
OUT="${3:?output dir}"
BIN="$BUILD/bin"
export PATH="/usr/local/cuda/bin:$PATH"

cc=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader | head -1 | tr -d ' .')
ARCH="sm_${cc:-unknown}"
RES="$OUT/results/$ARCH"
mkdir -p "$RES" "$OUT/logs"
step() { printf '\n######## %s\n' "$1"; }

step "environment"
VENV=/tmp/ccml-venv   # outside /kaggle/working: a venv is not a result
bash "$SRC/scripts/setup_env.sh" "$VENV" > "$RES/env.txt" 2>&1
echo "setup_env exit $?"; cat "$RES/env.txt"
PY="$VENV/bin/python"; [ -x "$PY" ] || PY=python3

step "correctness tests (ctest)"
ctest --test-dir "$BUILD" --output-on-failure 2>&1 | tee "$RES/ctest.txt"

step "python measurement-helper tests"
"$PY" -m pytest "$SRC/scripts/tests" -q 2>&1 | tee "$RES/pytest_benchlib.txt"

step "ctypes binding tests (interim binding, until the cpp_extension replaces it)"
CCML_LIBRARY_PATH="$BUILD/lib/libccml.so" "$PY" -m pytest "$SRC/python/tests" -q 2>&1 | tail -3 | tee "$RES/pytest_binding.txt"

step "compute-sanitizer"
bash "$SRC/scripts/sanitize.sh" "$BIN" "$RES/sanitizer"
echo "sanitize exit $?"

step "device peaks"
"$BIN/ccml_bench_peaks" --out "$RES/peaks.json"

for op in matmul layernorm softmax; do
  step "C++ bench: $op"
  "$BIN/ccml_bench_$op" --peaks "$RES/peaks.json" --out "$RES/$op.csv" \
    2> "$OUT/logs/bench_$op.log"
  echo "exit $?"; tail -3 "$OUT/logs/bench_$op.log"
done

step "Python baselines: eager, torch.compile, Triton"
"$PY" "$SRC/scripts/bench_torch.py" --out-dir "$RES" --peaks "$RES/peaks.json" \
  2> "$OUT/logs/bench_torch.log"
echo "exit $?"; tail -5 "$OUT/logs/bench_torch.log"

step "done"
ls -la "$RES"
