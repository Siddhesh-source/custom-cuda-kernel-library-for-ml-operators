# Local sm_89 build for the docs/local.md items only (BF16 Tensor Cores, cp.async, Nsight Compute).
# Everything else runs on Kaggle. Usage:  powershell -File scripts/local_build.ps1 [-Preset local]
param([string]$Preset = "local")
$ErrorActionPreference = "Stop"
$vswhere = "${env:ProgramFiles(x86)}\Microsoft Visual Studio\Installer\vswhere.exe"
$vs = & $vswhere -latest -products * -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -property installationPath
if (-not $vs) { throw "MSVC not found. See docs/local.md step 1." }
# Import the MSVC + Windows SDK environment into this PowerShell session.
cmd /c "`"$vs\VC\Auxiliary\Build\vcvars64.bat`" >nul && set" | ForEach-Object {
  if ($_ -match '^([^=]+)=(.*)$') { Set-Item -Path "env:$($matches[1])" -Value $matches[2] }
}
$ninjaDir = "$env:APPDATA\Python\Python311\Scripts"
if (Test-Path "$ninjaDir\ninja.exe") { $env:PATH = "$ninjaDir;$env:PATH" }
Set-Location (Split-Path $PSScriptRoot -Parent)
cmake --workflow --preset $Preset
