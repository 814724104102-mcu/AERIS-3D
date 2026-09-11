"""
Dataset inspector for AERIS-3D.
Generates dataset schemas and statistics reports (Gate 0.5).
"""
import json
import numpy as np
from pathlib import Path
from core.logger import get_logger

log = get_logger(__name__)

def generate_dataset_report(ds, output_dir: Path):
    """
    Analyzes the loaded dataset and produces dataset_report.json and dataset_quality_report.md
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    
    report = {
        "num_samples": len(ds),
        "features": list(ds.features.keys()),
        "description": ds.info.description if hasattr(ds, 'info') else "No description",
    }
    
    # Check what features exist
    has_image = "image" in report["features"]
    has_dsm = "dsm" in report["features"] or "height" in report["features"]
    
    report["has_image"] = has_image
    report["has_dsm"] = has_dsm
    
    report_path = output_dir / "dataset_report.json"
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)
        
    md_path = output_dir / "dataset_quality_report.md"
    with open(md_path, "w") as f:
        f.write("# AERIS-3D Dataset Quality Report\n\n")
        f.write(f"**Total Samples Analzyed:** {len(ds)}\n")
        f.write(f"**Features:** {', '.join(report['features'])}\n\n")
        f.write("## Quality Checks\n")
        f.write(f"- RGB Images Present: {'✅' if has_image else '❌'}\n")
        f.write(f"- Target Height/DSM Present: {'✅' if has_dsm else '❌'}\n")
        f.write("\n*Generated during Gate 0.5 inspection.*")
        
    log.info(f"Generated dataset reports in {output_dir}")
    return report_path
