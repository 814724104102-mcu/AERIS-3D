"""
AERIS-3D — Tile Terrain Labeller

Scans the downloaded GAMUS dataset and any custom test tiles, tags each
tile with a terrain category based on its source city / region, and
writes a CSV that the evaluator can group results by.

Usage:
    python scripts/label_tiles.py                 # default paths
    python scripts/label_tiles.py --out labels.csv # custom output

City → Terrain mapping (GAMUS):
    DC  → government_urban   (Washington DC: monuments, parks, mid-rise)
    PHL → industrial_urban   (Philadelphia: dense row-houses, industrial waterfront)
    NYC → dense_urban        (New York City: high-rise canyons, very dense)

Custom tiles:
    mussoorie → hilly         (Himalayan foothills, 1000–2300 m elevation)
"""

from __future__ import annotations

import argparse
import csv
import os
from pathlib import Path

# ── City prefix → terrain category ────────────────────────────────
CITY_TERRAIN = {
    "DC":  "government_urban",
    "PHL": "industrial_urban",
    "NYC": "dense_urban",
}

# ── Custom (non-GAMUS) tiles ─────────────────────────────────────
CUSTOM_TILES = {
    "mussoorie": "hilly",
}


def scan_gamus(gamus_root: Path) -> list[dict]:
    """Scan GAMUS directory and return labelled tile records."""
    rows = []
    for data_type in ("images", "heights"):
        for split in ("train", "val", "test"):
            split_dir = gamus_root / data_type / split
            if not split_dir.is_dir():
                continue
            for fname in sorted(os.listdir(split_dir)):
                if not fname.endswith(".h5"):
                    continue
                prefix = fname.split("_")[0]
                terrain = CITY_TERRAIN.get(prefix, "unknown")
                # Parse grid coords from filename: CITY_ROW_COL_TYPE.h5
                parts = fname.replace(".h5", "").split("_")
                row_col = f"{parts[1]}_{parts[2]}" if len(parts) >= 3 else ""
                rows.append({
                    "filename": fname,
                    "path": str(split_dir / fname),
                    "source": "GAMUS",
                    "city": prefix,
                    "grid": row_col,
                    "split": split,
                    "data_type": data_type,
                    "terrain": terrain,
                })
    return rows


def scan_custom(hilly_root: Path) -> list[dict]:
    """Scan custom test tiles (e.g. Mussoorie DEM + RGB)."""
    rows = []
    if not hilly_root.is_dir():
        return rows
    for fname in sorted(os.listdir(hilly_root)):
        if not (fname.endswith(".tif") or fname.endswith(".png")):
            continue
        # Derive region name from filename (e.g. "mussoorie_dem.tif" → "mussoorie")
        region = fname.split("_")[0].lower()
        terrain = CUSTOM_TILES.get(region, "unknown")
        data_type = "heights" if "dem" in fname.lower() else "images"
        rows.append({
            "filename": fname,
            "path": str(hilly_root / fname),
            "source": "custom",
            "city": region,
            "grid": "",
            "split": "test",
            "data_type": data_type,
            "terrain": terrain,
        })
    return rows


def main():
    parser = argparse.ArgumentParser(description="Label AERIS-3D tiles by terrain category")
    parser.add_argument(
        "--gamus", type=Path,
        default=Path("data/GAMUS"),
        help="Path to GAMUS dataset root",
    )
    parser.add_argument(
        "--custom", type=Path,
        default=Path("data/hilly_test"),
        help="Path to custom test tiles",
    )
    parser.add_argument(
        "--out", type=Path,
        default=Path("data/tile_labels.csv"),
        help="Output CSV path",
    )
    args = parser.parse_args()

    rows = scan_gamus(args.gamus) + scan_custom(args.custom)

    # Write CSV
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["filename", "path", "source", "city", "grid", "split", "data_type", "terrain"]
    with open(args.out, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    # Summary
    from collections import Counter
    terrain_counts = Counter(r["terrain"] for r in rows)
    split_counts = Counter(r["split"] for r in rows)
    dtype_counts = Counter(r["data_type"] for r in rows)

    print(f"Labelled {len(rows)} tiles -> {args.out}")
    print()
    print("By terrain:")
    for t, c in terrain_counts.most_common():
        print(f"  {t:25s} {c:>5}")
    print()
    print("By split:")
    for s, c in split_counts.most_common():
        print(f"  {s:25s} {c:>5}")
    print()
    print("By data type:")
    for d, c in dtype_counts.most_common():
        print(f"  {d:25s} {c:>5}")


if __name__ == "__main__":
    main()
