#!/usr/bin/env bash
# compute-sanitizer over every correctness suite (reference.md, Method:
# "use compute-sanitizer (memcheck, racecheck) on kernels"; enhancement C1 adds
# initcheck and synccheck).
#
#   bash scripts/sanitize.sh BUILD_BIN_DIR LOG_DIR
#
# Writes one log per (suite, tool) and a summary table. Exits non-zero if any
# tool reports an error, so a dirty result cannot pass silently.
set -u
BIN="${1:?build bin dir}"
LOGS="${2:?log dir}"
mkdir -p "$LOGS"
export PATH="/usr/local/cuda/bin:$PATH"

suites=(infrastructure harness matmul matmul_tc layernorm softmax dtypes)
tools=(memcheck racecheck initcheck synccheck)
status=0
summary="$LOGS/summary.txt"
printf '%-16s %-10s %-8s %s\n' suite tool exit result > "$summary"

for s in "${suites[@]}"; do
  exe="$BIN/ccml_test_$s"
  [ -x "$exe" ] || { echo "missing $exe" | tee -a "$summary"; status=1; continue; }
  for t in "${tools[@]}"; do
    log="$LOGS/${s}_${t}.log"
    # --error-exitcode makes a finding fail the step; the suite's own exit code
    # (a failed correctness test) is kept separately in the log.
    timeout 3600 compute-sanitizer --tool "$t" --error-exitcode 99 \
      --print-limit 50 "$exe" > "$log" 2>&1
    rc=$?
    result=$(grep -E "ERROR SUMMARY|RACECHECK SUMMARY" "$log" | tail -1)
    # One known, deliberate API error: infrastructure.cuda_failures_carry_the_runtime_message
    # requests an impossible allocation to check that CUDA's message is preserved,
    # and memcheck reports every failed API call. It is accepted ONLY if it is the
    # sole error and comes from that test; anything else still fails the run.
    if [ "$rc" -ne 0 ] && [ "$s" = infrastructure ] && [ "$t" = memcheck ] \
       && grep -q "ERROR SUMMARY: 1 error" "$log" \
       && grep -q "cudaErrorMemoryAllocation" "$log" \
       && grep -q "cuda_failures_carry_the_runtime_message" "$log"; then
      rc=0
      result="$result (expected: deliberate OOM in cuda_failures_carry_the_runtime_message)"
    fi
    printf '%-16s %-10s %-8s %s\n' "$s" "$t" "$rc" "${result:-no summary line}" >> "$summary"
    [ "$rc" -eq 0 ] || status=1
  done
done

cat "$summary"
exit $status
