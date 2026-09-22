import os
import sys
from pathlib import Path
import h5py
import cv2
import numpy as np
import rasterio
from rasterio.transform import from_bounds
import requests

# Add root to sys path so we can import backend
sys.path.insert(0, str(Path(r"n:\AERIS-3D-main")))

from backend.main import _process_image_job

def download_arcgis_tile(minx, miny, maxx, maxy, out_path, is_georef=True):
    print(f"Downloading ArcGIS tile for bounds {minx},{miny},{maxx},{maxy}...")
    url = f"https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/export?bbox={minx},{miny},{maxx},{maxy}&bboxSR=4326&imageSR=4326&size=1024,1024&f=image"
    r = requests.get(url)
    if r.status_code != 200:
        print(f"Failed to download from ArcGIS: {r.status_code}")
        return False
        
    img_array = np.asarray(bytearray(r.content), dtype=np.uint8)
    img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
    
    if not is_georef:
        cv2.imwrite(str(out_path), img)
    else:
        # Convert BGR to RGB
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        transform = from_bounds(minx, miny, maxx, maxy, 1024, 1024)
        with rasterio.open(
            out_path, 'w', driver='GTiff',
            height=1024, width=1024, count=3,
            dtype=img_rgb.dtype, crs='+proj=latlong',
            transform=transform
        ) as dst:
            dst.write(img_rgb[:, :, 0], 1)
            dst.write(img_rgb[:, :, 1], 2)
            dst.write(img_rgb[:, :, 2], 3)
    return True

def prepare_inputs():
    test_dir = Path(r"n:\AERIS-3D-main\data\e2e_tests")
    test_dir.mkdir(parents=True, exist_ok=True)
    
    inputs = []
    
    # 1. Urban (non-georeferenced) - from GAMUS DC
    urban_path = test_dir / "urban_nongeoref.jpg"
    gamus_h5 = Path(r"n:\AERIS-3D-main\data\GAMUS\images\test\DC_47_43_RGB.h5")
    if not urban_path.exists() and gamus_h5.exists():
        with h5py.File(gamus_h5, 'r') as f:
            img = f['image'][:]
            # Convert RGB to BGR for cv2
            img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            cv2.imwrite(str(urban_path), img_bgr)
    if urban_path.exists():
        inputs.append(("urban", urban_path, "non-georeferenced"))
        
    # 2. Hilly (georeferenced) - Mussoorie
    hilly_path = Path(r"n:\AERIS-3D-main\data\hilly_test\mussoorie_rgb.tif")
    if hilly_path.exists():
        inputs.append(("hilly", hilly_path, "georeferenced"))
        
    # 3. Sparse (georeferenced) - Sahara Desert
    sparse_path = test_dir / "sparse_georef.tif"
    if not sparse_path.exists():
        # Lon: 10, Lat: 25 -> a 0.05 degree box
        download_arcgis_tile(10.0, 25.0, 10.05, 25.05, sparse_path, is_georef=True)
    if sparse_path.exists():
        inputs.append(("sparse", sparse_path, "georeferenced"))
        
    # 4. Forested (non-georeferenced) - Amazon
    forest_path = test_dir / "forest_nongeoref.jpg"
    if not forest_path.exists():
        # Lon: -60, Lat: -3
        download_arcgis_tile(-60.05, -3.05, -60.0, -3.0, forest_path, is_georef=False)
    if forest_path.exists():
        inputs.append(("forested", forest_path, "non-georeferenced"))
        
    return inputs

def run_tests():
    inputs = prepare_inputs()
    jobs_dir = Path(r"n:\AERIS-3D-main\data\jobs")
    
    logs = []
    
    for category, path, ref_type in inputs:
        print(f"\n--- Running E2E Test: {category.upper()} ({ref_type}) ---")
        job_dir = jobs_dir / f"e2e_{category}"
        job_dir.mkdir(parents=True, exist_ok=True)
        
        try:
            summary = _process_image_job(path, job_dir)
            logs.append(f"SUCCESS: {category} ({ref_type}) -> Output GLB size: {(job_dir / 'mesh.glb').stat().st_size} bytes")
            print(f"Success! Job ID: {summary['job_id']}")
            print(f"Mesh stats: Vertices={summary['mesh']['vertices']}, Scale Mode={summary['depth']['relative_only']}")
        except Exception as e:
            logs.append(f"FAILED: {category} ({ref_type}) -> {str(e)}")
            print(f"Failed! {str(e)}")
            
    print("\n\n=== E2E TEST SUMMARY ===")
    for log in logs:
        print(log)

if __name__ == "__main__":
    run_tests()
