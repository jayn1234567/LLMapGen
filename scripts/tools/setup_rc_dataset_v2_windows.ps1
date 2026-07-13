param(
    [string]$EnvName = "rc-dataset-v2",
    [string]$PythonVersion = "3.11",
    [string]$Channel = "conda-forge"
)

$ErrorActionPreference = "Stop"

if (-not (Get-Command conda -ErrorAction SilentlyContinue)) {
    throw "conda was not found on PATH. Open Anaconda Prompt and run this script again."
}

$packages = @(
    "python=$PythonVersion",
    "numpy=1.26",
    "pillow",
    "tqdm",
    "shapely>=2.0,<3",
    "rasterio>=1.3,<2",
    "geopandas>=0.14,<2",
    "pyproj>=3.6,<4"
)

$envList = conda env list --json | ConvertFrom-Json
$exists = $false
foreach ($prefix in $envList.envs) {
    if ((Split-Path $prefix -Leaf) -eq $EnvName) {
        $exists = $true
        break
    }
}

if ($exists) {
    Write-Host "[dataset-v2-env] updating conda environment: $EnvName"
    & conda install -n $EnvName -c $Channel @packages -y
} else {
    Write-Host "[dataset-v2-env] creating conda environment: $EnvName"
    & conda create -n $EnvName -c $Channel @packages -y
}
if ($LASTEXITCODE -ne 0) {
    throw "conda environment installation failed with exit code $LASTEXITCODE"
}

& conda run -n $EnvName python -m pip install "setuptools<81"
if ($LASTEXITCODE -ne 0) {
    throw "setuptools installation failed with exit code $LASTEXITCODE"
}

& conda run -n $EnvName python -c "import geopandas, numpy, PIL, pyproj, rasterio, shapely; print('dataset-v2 environment ok')"
if ($LASTEXITCODE -ne 0) {
    throw "dataset-v2 environment preflight failed with exit code $LASTEXITCODE"
}

Write-Host "[dataset-v2-env] ready"
Write-Host "[dataset-v2-env] activate with: conda activate $EnvName"
Write-Host "[dataset-v2-env] OBS access uses obsutil.exe; install and configure it separately."
