param(
    [Parameter(Mandatory = $true)]
    [string]$DagsHubUsername,

    [Parameter(Mandatory = $true)]
    [string]$DagsHubRepo,

    [Parameter(Mandatory = $true)]
    [string]$DagsHubToken,

    [string]$DagsHubUserForAuth = $DagsHubUsername,
    [string]$ExperimentName = "spotify-recommender",
    [string]$RegisteredModelName = "spotify-kmeans-recommender"
)

$ErrorActionPreference = "Stop"

$projectRoot = Split-Path -Parent $PSScriptRoot
$pythonPath = Join-Path $projectRoot ".venv\Scripts\python.exe"

if (-not (Test-Path $pythonPath)) {
    $pythonPath = "python"
}

$env:MLFLOW_TRACKING_URI = "https://dagshub.com/$DagsHubUsername/$DagsHubRepo.mlflow"
$env:MLFLOW_TRACKING_USERNAME = $DagsHubUserForAuth
$env:MLFLOW_TRACKING_PASSWORD = $DagsHubToken
$env:MLFLOW_EXPERIMENT_NAME = $ExperimentName
$env:MLFLOW_REGISTERED_MODEL_NAME = $RegisteredModelName
$env:PYTHONIOENCODING = "utf-8"

Write-Host "Registering model '$RegisteredModelName' to $env:MLFLOW_TRACKING_URI"
& $pythonPath (Join-Path $projectRoot "src\clustering.py")
if ($LASTEXITCODE -ne 0) {
    throw "Model registration failed with exit code $LASTEXITCODE"
}

Write-Host ""
Write-Host "Model registry URL:"
Write-Host "https://dagshub.com/$DagsHubUsername/$DagsHubRepo.mlflow/#/models/$RegisteredModelName"
