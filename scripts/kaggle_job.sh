#!/usr/bin/env bash
# The work one Kaggle GPU session performs, after cloud/kaggle/kernel_template.py
# has unpacked and built the repository.
#
#   bash scripts/kaggle_job.sh SOURCE_DIR BUILD_DIR OUT_DIR JOB [IMPLS]
#
# JOB is one of:
#   sanitize                    ctest, Python helper tests, compute-sanitizer
#   matmul | layernorm | softmax  every implementation of ONE operator, C++ and
#                               Python, in this one session
#   all                         everything, sequentially (one session)
#
# Splitting by operator, never by implementation, keeps every comparison in a
# results table on one physical GPU in one session (reference.md §11: "always
# compare on the same GPU, same dtype and same clocks"). Sessions run in
# parallel; each records its own measured peaks.
#
# IMPLS limits what is re-measured in later phases (default: all), e.g.
# "ccml" re-times only our kernels against the committed baselines.
#
# Everything under OUT_DIR/results/<arch>/ is copied back into the repository by
# `python cloud/kaggle/kaggle_run.py --fetch`.
set -u
SRC="${1:?source dir}"
BUILD="${2:?build dir}"
OUT="${3:?output dir}"
JOB="${4:-all}"
IMPLS="${5:-all}"
BIN="$BUILD/bin"
export PATH="/usr/local/cuda/bin:$PATH"

cc=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader | head -1 | tr -d ' .')
ARCH="sm_${cc:-unknown}"
RES="$OUT/results/$ARCH"
mkdir -p "$RES" "$OUT/logs"
step() { printf '\n######## [%s] %s  (%s)\n' "$JOB" "$1" "$(date -u +%H:%M:%S)"; }
has_impl() { [ "$IMPLS" = "all" ] || [[ ",$IMPLS," == *",$1,"* ]]; }

step "environment"
VENV=/tmp/ccml-venv   # outside /kaggle/working: a venv is not a result
bash "$SRC/scripts/setup_env.sh" "$VENV" > "$RES/env_$JOB.txt" 2>&1
echo "setup_env exit $?"; cat "$RES/env_$JOB.txt"
# Use the venv only if torch imports from it; otherwise the image interpreter.
PY="$VENV/bin/python"
"$PY" -c "import torch" 2>/dev/null || PY="${CCML_PYTHON:-python3}"
echo "python for this job: $PY"

run_sanitize() {
  step "correctness tests (ctest)"
  ctest --test-dir "$BUILD" --output-on-failure 2>&1 | tee "$RES/ctest.txt"
  step "python measurement-helper tests"
  "$PY" -m pytest "$SRC/scripts/tests" -q 2>&1 | tail -3 | tee "$RES/pytest_benchlib.txt"
  step "ctypes binding tests (interim binding, until the cpp_extension replaces it)"
  CCML_LIBRARY_PATH="$BUILD/lib/libccml.so" "$PY" -m pytest "$SRC/python/tests" -q 2>&1 \
    | tail -3 | tee "$RES/pytest_binding.txt"
  step "compute-sanitizer"
  bash "$SRC/scripts/sanitize.sh" "$BIN" "$RES/sanitizer"
  echo "sanitize exit $?"
}

run_op() {
  local op="$1" peaks="$RES/peaks_$1.json"
  step "device peaks (this session's GPU)"
  "$BIN/ccml_bench_peaks" --out "$peaks"

  if has_impl ccml || has_impl ccml_tiles || has_impl cublas || has_impl cublaslt || has_impl cudnn; then
    step "C++ bench: $op"
    local extra=()
    # The naive matmul is ~20x slower than cuBLAS; above 5 GFLOP per call it
    # adds tens of minutes and no information (rows say NOT_RUN).
    # Matmul also times every tile configuration of the tiled family
    # (reference.md section 5: tile and block size sweep) and every storage
    # type (section 5: dtype comparison); CCML_MATMUL_DTYPES narrows the list.
    # The CUDA-core sweep was measured in Phases 9-10; CCML_MATMUL_SWEEP=both
    # repeats it.
    [ "$op" = "matmul" ] && extra=(--naive-max-gflop 5 --tile-sweep
                                   --sweep "${CCML_MATMUL_SWEEP:-tc}"
                                   --dtype "${CCML_MATMUL_DTYPES:-f32,f16,bf16}")
    [ "$op" = "layernorm" ] && extra=(--dtype "${CCML_ROWWISE_DTYPES:-f32,f16,bf16}")
    [ "$IMPLS" != "all" ] && extra+=(--impls "$IMPLS")
    "$BIN/ccml_bench_$op" --peaks "$peaks" --out "$RES/$op.csv" "${extra[@]}" \
      2> "$OUT/logs/bench_$op.log"
    echo "exit $?"; tail -3 "$OUT/logs/bench_$op.log"
    if [ "$op" = "layernorm" ] && [ -x "$BIN/ccml_bench_layernorm_residual" ]; then
      step "C++ bench: fused residual + layer norm (F2 / F4)"
      local rimpls=()
      [ "$IMPLS" != "all" ] && rimpls=(--impls "$IMPLS")
      "$BIN/ccml_bench_layernorm_residual" --peaks "$peaks" --out "$RES/layernorm_residual.csv" \
        --dtype "${CCML_ROWWISE_DTYPES:-f32,f16,bf16}" "${rimpls[@]}" \
        2> "$OUT/logs/bench_layernorm_residual.log"
      echo "exit $?"; tail -3 "$OUT/logs/bench_layernorm_residual.log"
    fi
    if [ "$op" = "matmul" ] && [ -x "$BIN/ccml_bench_matmul_epilogue" ]; then
      step "C++ bench: fused matmul epilogue (F1 / B2)"
      local eimpls=()
      [ "$IMPLS" != "all" ] && eimpls=(--impls "$IMPLS")
      "$BIN/ccml_bench_matmul_epilogue" --peaks "$peaks" --out "$RES/matmul_epilogue.csv"         --dtype f16 "${eimpls[@]}" 2> "$OUT/logs/bench_matmul_epilogue.log"
      echo "exit $?"; tail -3 "$OUT/logs/bench_matmul_epilogue.log"
    fi
  fi

  local py_impls=()
  has_impl eager && py_impls+=(eager)
  has_impl compile && py_impls+=(compile)
  has_impl triton && py_impls+=(triton)
  if [ ${#py_impls[@]} -gt 0 ]; then
    # Matmul under max-autotune spends ~1 min per shape autotuning Triton
    # templates; Inductor's default mode lowers mm to cuBLAS. Row-wise ops keep
    # max-autotune, where generated kernels are the point of the baseline.
    local mode=max-autotune
    [ "$op" = "matmul" ] && mode=default
    step "Python baselines: $op (${py_impls[*]}, compile mode $mode)"
    local ops="$op"
    [ "$op" = "layernorm" ] && ops="layernorm,layernorm_residual"
    "$PY" "$SRC/scripts/bench_torch.py" --out-dir "$RES" --peaks "$peaks" --ops "$ops" \
      --impls "$(IFS=,; echo "${py_impls[*]}")" --compile-mode "$mode" \
      --matmul-dtypes "${CCML_MATMUL_DTYPES:-f32,f16,bf16}" \
      --layernorm-dtypes "${CCML_ROWWISE_DTYPES:-f32,f16,bf16}" \
      2> "$OUT/logs/bench_torch_$op.log"
    echo "exit $?"; tail -5 "$OUT/logs/bench_torch_$op.log"
  fi
}

case "$JOB" in
  sanitize) run_sanitize ;;
  matmul|layernorm|softmax) run_op "$JOB" ;;
  all) run_sanitize; for op in matmul layernorm softmax; do run_op "$op"; done ;;
  *) echo "unknown job: $JOB"; exit 2 ;;
esac

step "done"
ls -la "$RES"
