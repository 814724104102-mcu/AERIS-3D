"""
AERIS-3D — Disaster Management Layer (Phase 16)

Provides backend logic to compute disaster risk assessment:
- Flood risk per structure (height vs flood level)
- Slope risk overlay (steep slope zones)
- Auto-generated Disaster Risk Report
"""

from dataclasses import dataclass
import numpy as np

@dataclass
class FloodRiskResult:
    flood_level: float
    structures_at_risk: int
    total_structures: int
    risk_by_object: dict[str, str] # object_id -> 'red', 'amber', 'green'
    slope_risk_area_m2: float # approximate
    report_text: str

def compute_slope(depth_map: np.ndarray, gsd: float = 1.0) -> np.ndarray:
    """Compute slope map in degrees from depth/elevation map."""
    dy, dx = np.gradient(depth_map, gsd, gsd)
    slope = np.degrees(np.arctan(np.sqrt(dx**2 + dy**2)))
    return slope

def run_flood_simulation(
    flood_level: float,
    survivors: dict,
    terrain_result,
    depth_map: np.ndarray,
    slope_threshold_deg: float = 30.0,
    gsd: float = 1.0,
) -> FloodRiskResult:
    total_structures = len(survivors)
    structures_at_risk = 0
    risk_by_object = {}
    
    for obj_id, score in survivors.items():
        obj_height = score.height_value
        if obj_height < flood_level:
            risk_by_object[obj_id] = "red"
            structures_at_risk += 1
        elif obj_height < flood_level + 2.0:
            risk_by_object[obj_id] = "amber"
        else:
            risk_by_object[obj_id] = "green"
            
    slope_map = compute_slope(depth_map, gsd)
    steep_mask = slope_map > slope_threshold_deg
    slope_risk_area_m2 = float(np.sum(steep_mask)) * (gsd ** 2)
    
    disclaimer = "SCENARIO SIMULATION — DECISION-SUPPORT PROTOTYPE, NOT AN OPERATIONAL HAZARD FORECAST."
    
    report_text = (
        f"{disclaimer}\n\n"
        f"Disaster Risk Report (Flood Level: {flood_level:.1f} m)\n"
        f"---------------------------------------------------\n"
        f"{structures_at_risk} of {total_structures} structures below +{flood_level:.1f} m.\n"
        f"Steep-slope zones (> {slope_threshold_deg:.1f}°): {slope_risk_area_m2:.1f} m².\n"
    )
    
    return FloodRiskResult(
        flood_level=flood_level,
        structures_at_risk=structures_at_risk,
        total_structures=total_structures,
        risk_by_object=risk_by_object,
        slope_risk_area_m2=slope_risk_area_m2,
        report_text=report_text
    )

def generate_simulation_metadata(survivors: dict, depth_map: np.ndarray, config: dict) -> dict:
    z_min = float(depth_map.min())
    z_max = float(depth_map.max())
    return {
        "disclaimer": "SCENARIO SIMULATION — DECISION-SUPPORT PROTOTYPE, NOT AN OPERATIONAL HAZARD FORECAST.",
        "true_relief": {
            "z_min": z_min,
            "z_max": z_max,
        },
        "structures": {
            obj_id: {
                "height_value": score.height_value,
                "height_unit": score.height_unit
            }
            for obj_id, score in survivors.items()
        }
    }
