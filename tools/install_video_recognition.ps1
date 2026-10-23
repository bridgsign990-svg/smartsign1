$ErrorActionPreference = "Stop"

if (-not $env:LOCALAPPDATA) {
    throw "LOCALAPPDATA is not set. This installer currently supports Windows."
}

$projectRoot = Split-Path -Parent $PSScriptRoot
$runtimeRoot = Join-Path $env:LOCALAPPDATA "BridgeSign\OpenHandsRuntime"
$venvPath = Join-Path $runtimeRoot "venv"
$sourcePath = Join-Path $runtimeRoot "source"
$pythonPath = Join-Path $venvPath "Scripts\python.exe"
$commit = "b9ccf9eaf2a71301fc4601a64a6af06fdef04c9b"

$bootstrapPython = Join-Path $projectRoot ".venv-2\Scripts\python.exe"
if (-not (Test-Path $bootstrapPython)) {
    $pythonCommand = Get-Command python -ErrorAction SilentlyContinue
    if (-not $pythonCommand) {
        throw "Python 3.12 was not found. Install Python 3.12 or create the project .venv-2 first."
    }
    $bootstrapPython = $pythonCommand.Source
}
$bootstrapVersion = & $bootstrapPython -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')"
if ($LASTEXITCODE -ne 0 -or $bootstrapVersion.Trim() -ne "3.12") {
    throw "The isolated OpenHands environment requires Python 3.12."
}

New-Item -ItemType Directory -Force -Path $runtimeRoot | Out-Null
if (-not (Test-Path $pythonPath)) {
    & $bootstrapPython -m venv $venvPath
    if ($LASTEXITCODE -ne 0) { throw "Could not create the isolated OpenHands environment." }
}

New-Item -ItemType Directory -Force -Path $sourcePath | Out-Null
if (-not (Test-Path (Join-Path $sourcePath ".git"))) {
    git -C $sourcePath init
    if ($LASTEXITCODE -ne 0) { throw "Could not initialize the OpenHands source directory." }
    git -C $sourcePath remote add origin https://github.com/AI4Bharat/OpenHands.git
} else {
    git -C $sourcePath remote set-url origin https://github.com/AI4Bharat/OpenHands.git
}
if ($LASTEXITCODE -ne 0) { throw "Could not configure the OpenHands source repository." }

git -C $sourcePath fetch --depth 1 origin $commit
if ($LASTEXITCODE -ne 0) { throw "Could not fetch the pinned OpenHands source revision." }
git -C $sourcePath checkout --detach FETCH_HEAD
if ($LASTEXITCODE -ne 0) { throw "Could not select the pinned OpenHands source revision." }

& $pythonPath -m pip install --index-url https://download.pytorch.org/whl/cpu torch==2.14.1 torchvision==0.29.1
if ($LASTEXITCODE -ne 0) { throw "Could not install the CPU-only PyTorch runtime." }
& $pythonPath -m pip install -r (Join-Path $projectRoot "video_recognition_requirements.txt")
if ($LASTEXITCODE -ne 0) { throw "Could not install the isolated video-recognition dependencies." }

[Environment]::SetEnvironmentVariable("BRIDGESIGN_VIDEO_RECOGNITION_PYTHON", $pythonPath, "User")
[Environment]::SetEnvironmentVariable("BRIDGESIGN_OPENHANDS_SOURCE_DIR", $sourcePath, "User")
$env:BRIDGESIGN_VIDEO_RECOGNITION_PYTHON = $pythonPath
$env:BRIDGESIGN_OPENHANDS_SOURCE_DIR = $sourcePath

Write-Host "OpenHands WLASL inference is installed in its own environment."
Write-Host "Restart BridgeSign. To launch locally: .\.venv-2\Scripts\python.exe app.py"
