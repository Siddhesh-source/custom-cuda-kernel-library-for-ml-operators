# Nsight Compute evidence for one matmul optimization step (docs/local.md L3;
# reference.md section 9). Profiles the first gate launch of each kernel that
# matches -Kernel, on each shape, inside ccml_bench_matmul, and writes
#   profiles/sm_89/matmul/<Name>_<shape>.ncu-rep        (section-limited, small)
#   profiles/sm_89/matmul/<Name>_<shape>_metrics.csv    (the metrics the log cites)
#
# Needs counter access: an elevated prompt, or a reboot after
# RmProfilingAdminOnly=0 (docs/local.md). ncu.bat runs under cmd, so -Kernel
# must not contain "|": use a character class (e.g. "matmul_[dw][om]").
#
#   powershell -File scripts/ncu_matmul.ps1 -Name double_buffered_vs_wmma `
#     -Kernel "matmul_[dw][om]" -Count 2 -Dtype f16 `
#     -Shapes gpt2s.ffn_up.m512,gpt2s.qkv.m16 [-Impls ccml]
param(
  [Parameter(Mandatory = $true)][string]$Name,
  [Parameter(Mandatory = $true)][string]$Kernel,
  [Parameter(Mandatory = $true)][string[]]$Shapes,
  [int]$Count = 2,
  [string]$Dtype = "f16",
  [string]$Impls = "ccml",
  [string]$Extra = ""
)
$ErrorActionPreference = "Continue"
$root = Split-Path $PSScriptRoot -Parent
$ncu = Get-ChildItem "C:\Program Files\NVIDIA Corporation\Nsight Compute*\ncu.bat" |
  Sort-Object FullName | Select-Object -Last 1 -ExpandProperty FullName
$bench = Join-Path $root "build\local\bin\ccml_bench_matmul.exe"
$out = Join-Path $root "profiles\sm_89\matmul"
New-Item -ItemType Directory -Force $out | Out-Null
$log = Join-Path $out "ncu_$Name.log"
"ncu: $ncu`nbench: $bench`n$(Get-Date -Format o)" | Out-File -Encoding utf8 $log

$metrics = @(
  "gpu__time_duration.sum",
  "launch__registers_per_thread",
  "launch__occupancy_limit_registers",
  "launch__occupancy_limit_shared_mem",
  "sm__warps_active.avg.pct_of_peak_sustained_active",
  "sm__throughput.avg.pct_of_peak_sustained_elapsed",
  "dram__throughput.avg.pct_of_peak_sustained_elapsed",
  "sm__pipe_fma_cycles_active.avg.pct_of_peak_sustained_active",
  "sm__pipe_tensor_cycles_active.avg.pct_of_peak_sustained_active",
  "sm__inst_executed_pipe_tensor.sum",
  "smsp__inst_executed.sum",
  "l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_ld.sum",
  "l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_st.sum",
  "smsp__average_warps_issue_stalled_barrier_per_issue_active.ratio",
  "smsp__average_warps_issue_stalled_long_scoreboard_per_issue_active.ratio",
  "smsp__average_warps_issue_stalled_short_scoreboard_per_issue_active.ratio",
  "smsp__average_warps_issue_stalled_mio_throttle_per_issue_active.ratio",
  "smsp__average_warps_issue_stalled_lg_throttle_per_issue_active.ratio",
  "smsp__average_warps_issue_stalled_math_pipe_throttle_per_issue_active.ratio",
  "smsp__average_warps_issue_stalled_not_selected_per_issue_active.ratio",
  "smsp__average_warps_issue_stalled_wait_per_issue_active.ratio"
) -join ","

foreach ($shape in ($Shapes -join "," -split ",")) {
  $rep = Join-Path $out "${Name}_$shape"
  $nargs = @("--section", "SpeedOfLight", "--section", "Occupancy", "--section", "LaunchStats",
            "--section", "WarpStateStats", "--section", "SchedulerStats",
            "--section", "ComputeWorkloadAnalysis", "--section", "MemoryWorkloadAnalysis",
            "--metrics", $metrics,
            "--kernel-name", "regex:$Kernel", "--launch-count", "$Count",
            "-f", "-o", $rep,
            $bench, "--filter", $shape, "--dtype", $Dtype, "--impls", $Impls,
            "--cache", "warm", "--max-iters", "100")
  if ($Extra) { $nargs += $Extra -split " " }
  "`n#### $shape" | Out-File -Append -Encoding utf8 $log
  & $ncu @nargs 2>&1 | Out-File -Append -Encoding utf8 $log
  & $ncu -i "$rep.ncu-rep" --csv --page raw --metrics $metrics 2>&1 |
    Out-File -Encoding utf8 "${rep}_metrics.csv"
}
"done $(Get-Date -Format o)" | Out-File -Append -Encoding utf8 $log
