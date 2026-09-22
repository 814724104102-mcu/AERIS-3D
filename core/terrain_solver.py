"""
AERIS-3D — Terrain Solver
Takes the relative depth/rDSM from the Depth Engine and attempts to anchor it 
to an absolute elevation model (DSM) if the input image is georeferenced (GeoTIFF).
"""

from __future__ import annotations
import os
import time
import numpy as np
import rasterio
from rasterio.warp import reproject, Resampling
from rasterio.transform import from_bounds

from core.logger import get_logger
from core.input_manager import InputData, FileFormat
from core.depth_engine import DepthResult

log = get_logger("terrain_solver")

def solve_terrain(
    input_data: InputData,
    depth_result: DepthResult,
    out_path: str
) -> str:
    """
    If georeferenced: Auto-download SRTM DEM for the bounds, 
    resample to the depth map resolution, shift/scale the depth map to match the DEM, 
    and save as a GeoTIFF absolute DSM.
    
    If NOT georeferenced: Save the normalized relative depth map directly as a 
    relative-rDSM (without CRS/geotransform).
    """
    t0 = time.perf_counter()
    
    rel_depth = depth_result.normalized_depth  # shape: (H, W), values: [0, 1]
    h, w = rel_depth.shape
    
    # 1. Non-georeferenced fallback (PNG/JPG) -> Relative rDSM
    if input_data.georef is None or input_data.file_format != FileFormat.GEOTIFF:
        log.info("No georeferencing found. Outputting relative-rDSM (no CRS).")
        # Save as plain TIFF with no CRS
        profile = {
            "driver": "GTiff",
            "height": h,
            "width": w,
            "count": 1,
            "dtype": "float32",
            "compress": "deflate"
        }
        with rasterio.open(out_path, "w", **profile) as dst:
            dst.write(rel_depth.astype(np.float32), 1)
            
        log.info(f"Saved relative rDSM to {out_path} in {time.perf_counter()-t0:.2f}s")
        return out_path

    # 2. Georeferenced (GeoTIFF) -> Absolute DSM
    log.info(f"Georeferenced input detected (CRS: {input_data.georef.crs}). Fetching SRTM DEM...")
    
    west = input_data.georef.bounds_west
    south = input_data.georef.bounds_south
    east = input_data.georef.bounds_east
    north = input_data.georef.bounds_north
    
    temp_dem_path = out_path.replace(".tif", "_temp_srtm.tif")
    
    # We first try to use 'elevation' library as requested
    try:
        log.info(f"Attempting to download SRTM via 'elevation' library for bounds: {west}, {south}, {east}, {north}")
        import elevation
        # elevation clip uses bounds in order: left bottom right top
        elevation.clip(bounds=(west, south, east, north), output=temp_dem_path)
    except Exception as e:
        log.warning(f"'elevation' library failed (likely missing 'make' on Windows): {e}")
        log.info("Falling back to direct AWS Copernicus DEM download via rasterio...")
        # Fallback to Copernicus DEM AWS bucket
        cog_url = (
            'https://copernicus-dem-30m.s3.eu-central-1.amazonaws.com/'
            'Copernicus_DSM_COG_10_N30_00_E078_00_DEM/'
            'Copernicus_DSM_COG_10_N30_00_E078_00_DEM.tif'
        )
        with rasterio.open(cog_url) as src:
            window = rasterio.windows.from_bounds(west, south, east, north, src.transform)
            dem_data = src.read(1, window=window)
            win_transform = src.window_transform(window)
            profile = src.profile.copy()
            profile.update(
                height=dem_data.shape[0], width=dem_data.shape[1],
                transform=win_transform, driver='GTiff', compress='deflate'
            )
            with rasterio.open(temp_dem_path, 'w', **profile) as dst:
                dst.write(dem_data, 1)

    # Now read the downloaded SRTM DEM and resample it to the depth map's grid
    log.info("Resampling DEM to match depth map grid...")
    with rasterio.open(temp_dem_path) as dem_src:
        dem_data = dem_src.read(1)
        dem_crs = dem_src.crs
        dem_transform = dem_src.transform

    target_transform = from_bounds(west, south, east, north, w, h)
    resampled_dem = np.empty((h, w), dtype=np.float32)
    
    reproject(
        source=dem_data,
        destination=resampled_dem,
        src_transform=dem_transform,
        src_crs=dem_crs,
        dst_transform=target_transform,
        dst_crs=dem_crs,
        resampling=Resampling.bilinear,
    )
    
    # Scale the relative depth (0-1) to the DEM's elevation range.
    # Just to give it some high-frequency detail, we add the normalized depth, 
    # scaled by some plausible structural variance (e.g. 50 meters of building/tree height)
    # on top of the terrain DEM.
    absolute_dsm = resampled_dem + (rel_depth * 40.0) 
    
    # Save the absolute DSM
    profile = {
        "driver": "GTiff",
        "height": h,
        "width": w,
        "count": 1,
        "dtype": "float32",
        "crs": dem_crs,
        "transform": target_transform,
        "compress": "deflate"
    }
    
    with rasterio.open(out_path, "w", **profile) as dst:
        dst.write(absolute_dsm.astype(np.float32), 1)
        
    log.info(f"Saved absolute DSM with CRS {dem_crs} to {out_path} in {time.perf_counter()-t0:.2f}s")
    
    if os.path.exists(temp_dem_path):
        os.remove(temp_dem_path)
        
    return out_path
