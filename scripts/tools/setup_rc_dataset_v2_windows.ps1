param(
    [string]$EnvName = "rc-dataset-v2",
    [string]$CloneFrom = "py311",
    [string]$PythonVersion = "3.11",
    [string]$PipIndexUrl = ""
)

$ErrorActionPreference = "Stop"

if (-not (Get-Command conda -ErrorAction SilentlyContinue)) {
    throw "conda was not found on PATH. Open Anaconda Prompt and run this script again."
}

$pipPackages = @(
    "numpy==1.26.4",
    "pillow>=10,<12",
    "tqdm>=4.66,<5",
    "shapely>=2.0,<3",
    "rasterio>=1.3,<2",
    "geopandas>=0.14,<2",
    "pyproj>=3.6,<4",
    "setuptools<81"
)

$envList = conda env list --json | ConvertFrom-Json
$exists = $false
$cloneSourceExists = $false
foreach ($prefix in $envList.envs) {
    $leaf = Split-Path $prefix -Leaf
    if ($leaf -eq $EnvName) {
        $exists = $true
    }
    if ($leaf -eq $CloneFrom -or $prefix -eq $CloneFrom) {
        $cloneSourceExists = $true
    }
}

if (-not $exists) {
    if (-not $cloneSourceExists) {
        throw "Source conda environment '$CloneFrom' was not found. Create it first or pass -CloneFrom with an existing environment name/path."
    }
    Write-Host "[dataset-v2-env] cloning conda environment: $CloneFrom -> $EnvName"
    & conda create -n $EnvName --clone $CloneFrom -y
    if ($LASTEXITCODE -ne 0) {
        throw "conda environment clone failed with exit code $LASTEXITCODE"
    }
} else {
    Write-Host "[dataset-v2-env] reusing conda environment: $EnvName"
}

$actualPythonVersion = (& conda run -n $EnvName python -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')").Trim()
if ($LASTEXITCODE -ne 0) {
    throw "failed to inspect Python in environment '$EnvName'"
}
if ($actualPythonVersion -ne $PythonVersion) {
    throw "Environment '$EnvName' uses Python $actualPythonVersion; expected $PythonVersion. Check the -CloneFrom environment."
}
Write-Host "[dataset-v2-env] Python version: $actualPythonVersion"

$pipArgs = @("run", "-n", $EnvName, "python", "-m", "pip", "install")
if ($PipIndexUrl) {
    $pipArgs += @("--index-url", $PipIndexUrl)
}
$pipArgs += $pipPackages
Write-Host "[dataset-v2-env] installing data preparation packages with pip"
& conda @pipArgs
if ($LASTEXITCODE -ne 0) {
    throw "dataset dependency installation failed with exit code $LASTEXITCODE"
}

& conda run -n $EnvName python -c "import geopandas, numpy, PIL, pyproj, rasterio, shapely; print('dataset-v2 environment ok')"
if ($LASTEXITCODE -ne 0) {
    throw "dataset-v2 environment preflight failed with exit code $LASTEXITCODE"
}

Write-Host "[dataset-v2-env] ready"
Write-Host "[dataset-v2-env] activate with: conda activate $EnvName"
Write-Host "[dataset-v2-env] OBS access uses obsutil.exe; install and configure it separately."
