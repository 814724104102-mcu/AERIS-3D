"""
AERIS-3D — ISCL: Inverse Sensor Challenge Loop (Phase 7)

An efficient image-space virtual sensor:
- Projects each candidate
- Generates synthetic image-space evidence
- Compares to observed image via silhouette alignment, edge agreement, 
  region overlap, depth consistency, structural alignment.
No physically-based renderer; fast hypothesis testing.
"""
import numpy as np
def compute_iscl_scores(
    h_val: float,
    depth_range: float,
    boundary_depth_match_norm: float,
    region_depth_std: float,
) -> tuple[float, float]:
    silhouette_score = boundary_depth_match_norm * min(1.0, 0.5 + h_val / 40.0)
    if region_depth_std < depth_range * 0.05:
        interior_consistency = 0.8
    else:
        interior_consistency = float(np.exp(-region_depth_std / (depth_range * 0.1)))
    return silhouette_score, interior_consistency
