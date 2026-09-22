# AERIS-3D — Clean Environment E2E Test (PowerShell version)
$ErrorActionPreference = "Stop"

Write-Host "===========================================" -ForegroundColor Cyan
Write-Host "AERIS-3D End-to-End Release Candidate Test " -ForegroundColor Cyan
Write-Host "===========================================" -ForegroundColor Cyan

$pythonExe = "python"
if (Test-Path "venv/Scripts/python.exe") {
    $pythonExe = "venv/Scripts/python.exe"
}

Write-Host "1. Checking dependencies..."
& $pythonExe -c "import torch, numpy, core, backend, fastapi, trimesh; print('All dependencies satisfied.')"

Write-Host "2. Running Unit Tests (Phases 0-12)..."
& $pythonExe -m pytest tests/ -q --tb=short

Write-Host "3. Running CLI Demo Integration Test..."
& $pythonExe scripts/demo.py --no-cache

Write-Host "4. Testing FastAPI Backend (Phases 13-17)..."
$uvicornProcess = Start-Process -FilePath $pythonExe -ArgumentList "-m", "uvicorn", "backend.main:app", "--host", "127.0.0.1", "--port", "8001" -PassThru
Start-Sleep -Seconds 4

try {
    Write-Host "   Submitting API job..."
    $curlOut = curl.exe -s -X POST http://127.0.0.1:8001/api/process -F "file=@data/input/synthetic_test.jpg"
    $res = $curlOut | ConvertFrom-Json
    $jobId = $res.job_id

    if (-not $jobId) {
        throw "API failed to queue job. Response: $curlOut"
    }
    Write-Host "   Job queued: $jobId"

    $maxPoll = 30
    for ($i = 1; $i -le $maxPoll; $i++) {
        $statusOut = curl.exe -s http://127.0.0.1:8001/api/job/$jobId
        $statusJson = $statusOut | ConvertFrom-Json
        $status = $statusJson.status
        if ($status -eq "done") {
            Write-Host "   Job completed successfully."
            break
        }
        if ($status -eq "error") {
            throw "Job failed: $statusOut"
        }
        Start-Sleep -Seconds 1
    }

    Write-Host "   Verifying GLB output..."
    $httpCode = curl.exe -s -o NUL -w "%{http_code}" http://127.0.0.1:8001/api/file/$jobId/mesh.glb
    if ($httpCode -ne "200") {
        throw "Failed to serve mesh.glb (HTTP $httpCode)"
    }
    Write-Host "   Mesh served correctly (HTTP 200)."

    Write-Host "===========================================" -ForegroundColor Green
    Write-Host "E2E TEST PASSED. Release Candidate ready. " -ForegroundColor Green
    Write-Host "===========================================" -ForegroundColor Green
} finally {
    if ($uvicornProcess -and -not $uvicornProcess.HasExited) {
        Stop-Process -Id $uvicornProcess.Id -Force -ErrorAction SilentlyContinue
    }
}
