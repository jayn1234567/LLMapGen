param(
    [string]$WorkRoot = "D:\data\fulldata_local512",
    [string]$PythonExe = "python",
    [string]$StagingRoot = "",
    [string]$AuditJsonl = "",
    [string]$OutputRoot = "",
    [string]$PackagePath = "",
    [switch]$NoResume,
    [switch]$OverwritePhaseB,
    [switch]$SkipPackage
)

$ErrorActionPreference = "Stop"

$ScriptPath = $MyInvocation.MyCommand.Path
$ScriptDir = Split-Path -Parent $ScriptPath
$RepoRoot = Resolve-Path (Join-Path $ScriptDir "..\..")
Set-Location $RepoRoot

$Builder = "scripts\tools\build_rc_dataset_v3_local512_550k_stageab_windows.py"
$ResolvedOutputRoot = if ($OutputRoot.Trim().Length -gt 0) { $OutputRoot } else { Join-Path $WorkRoot "output_local512v3_550k_stageab" }
$ResolvedPackagePath = if ($PackagePath.Trim().Length -gt 0) { $PackagePath } else { Join-Path (Join-Path $WorkRoot "packages_v3") "local512v3_550k_stageab.tar" }
$ResolvedDatasetRoot = Join-Path $ResolvedOutputRoot "local512v3_550k_stageab"
$ArgsList = @(
    $Builder,
    "--work-root", $WorkRoot
)

if (-not $NoResume) {
    $ArgsList += "--resume"
}
if ($OverwritePhaseB) {
    $ArgsList += "--overwrite-phase-b"
}
if ($SkipPackage) {
    $ArgsList += "--skip-package"
}
if ($StagingRoot.Trim().Length -gt 0) {
    $ArgsList += @("--staging-root", $StagingRoot)
}
if ($AuditJsonl.Trim().Length -gt 0) {
    $ArgsList += @("--audit-jsonl", $AuditJsonl)
}
if ($OutputRoot.Trim().Length -gt 0) {
    $ArgsList += @("--output-root", $OutputRoot)
}
if ($PackagePath.Trim().Length -gt 0) {
    $ArgsList += @("--package-path", $PackagePath)
}

Write-Host "============================================================"
Write-Host "[local512v3-550k-stageab] repo:        $RepoRoot"
Write-Host "[local512v3-550k-stageab] work root:   $WorkRoot"
Write-Host "[local512v3-550k-stageab] python:      $PythonExe"
Write-Host "[local512v3-550k-stageab] builder:     $Builder"
Write-Host "[local512v3-550k-stageab] phase_b:     3 incoming trace points, 50px spacing; full neighbor intersection polygons"
Write-Host "============================================================"

& $PythonExe @ArgsList
if ($LASTEXITCODE -ne 0) {
    throw "local512v3_550k_stageab build failed with exit code $LASTEXITCODE"
}

Write-Host "============================================================"
Write-Host "[local512v3-550k-stageab] done"
Write-Host "[local512v3-550k-stageab] dataset: $ResolvedDatasetRoot"
Write-Host "[local512v3-550k-stageab] package: $ResolvedPackagePath"
Write-Host "============================================================"
