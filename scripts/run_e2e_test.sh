#!/usr/bin/env bash
# AERIS-3D — Clean Environment E2E Test (Phase 41)

set -e

echo "==========================================="
echo "AERIS-3D End-to-End Release Candidate Test "
echo "==========================================="

echo "1. Checking dependencies..."
python3 -c "import torch, numpy, cv2, fastapi, trimesh, skimage" || {
  echo "Missing dependencies. Run: pip install -r requirements.txt"
  exit 1
}

echo "2. Running Unit Tests (Phases 0-12)..."
pytest tests/ -q --tb=short

echo "3. Running CLI Demo Integration Test..."
python3 scripts/demo.py --no-cache

echo "4. Testing FastAPI Backend (Phases 13-17)..."
# Start backend in background
uvicorn backend.main:app --host 127.0.0.1 --port 8001 &
UVICORN_PID=$!
sleep 3

# Submit synthetic test
echo "   Submitting API job..."
RES=$(curl -s -X POST http://127.0.0.1:8001/api/process -F "file=@data/input/synthetic_test.jpg")
JOB_ID=$(echo $RES | grep -o '"job_id":"[^"]*' | cut -d'"' -f4)

if [ -z "$JOB_ID" ]; then
  echo "   API failed to queue job. Response: $RES"
  kill $UVICORN_PID
  exit 1
fi
echo "   Job queued: $JOB_ID"

# Poll until done
MAX_POLL=30
for i in $(seq 1 $MAX_POLL); do
  STATUS=$(curl -s http://127.0.0.1:8001/api/job/$JOB_ID | grep -o '"status":"[^"]*' | cut -d'"' -f4)
  if [ "$STATUS" = "done" ]; then
    echo "   Job completed successfully."
    break
  fi
  if [ "$STATUS" = "error" ]; then
    echo "   Job failed."
    kill $UVICORN_PID
    exit 1
  fi
  sleep 1
done

# Check if mesh generated
echo "   Verifying GLB output..."
HTTP_CODE=$(curl -s -o /dev/null -w "%{http_code}" http://127.0.0.1:8001/api/file/$JOB_ID/mesh.glb)
if [ "$HTTP_CODE" != "200" ]; then
  echo "   Failed to serve mesh.glb (HTTP $HTTP_CODE)"
  kill $UVICORN_PID
  exit 1
fi
echo "   Mesh served correctly."

# Cleanup
kill $UVICORN_PID
echo "==========================================="
echo "E2E TEST PASSED. Release Candidate ready. "
echo "==========================================="
