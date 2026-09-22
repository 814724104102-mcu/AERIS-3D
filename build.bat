@echo off
echo Building AERIS-3D Application...
echo.

echo Checking Python environment...
if not exist "venv\Scripts\python.exe" (
    echo Virtual environment not found. Creating one...
    python -m venv venv
)

echo Installing dependencies...
venv\Scripts\python.exe -m pip install -r requirements.txt
venv\Scripts\python.exe -m pip install transformers rasterio pandas h5py

echo.
echo Build complete! The frontend is already bundled into the FastAPI backend as a single deployable.
echo Run the app with: venv\Scripts\python.exe backend\main.py
echo Or use 'docker-compose up --build'
pause
