# AERIS-3D Dataset Quality Report

Generated: 2026-09-11 02:37:50 UTC

## Dataset Status

**Path**: `/Users/mugil/Downloads/Pause/ARIES/data/dataset/raw`  
**Status**: `DATASET_NOT_FOUND`

GAMUS (earthflow/GAMUS) — RGB + normalized-DSM tiles across 5 US cities. 6 land-cover classes: ground, low-vegetation, building, water, road, tree. Pre-split by city (geography-clean). Suitable for depth-backbone adaptation.

## Inspection Summary

| Field | Value |
|---|---|
| Sample count | None |
| Image dimensions | None |
| Channels | None |
| Height targets available | None |
| Native split available | None |

## Notes

- Dataset directory not found: /Users/mugil/Downloads/Pause/ARIES/data/dataset/raw. To download GAMUS: `pip install datasets` then `python -c "from datasets import load_dataset; ds = load_dataset('earthflow/GAMUS'); ds.save_to_disk('data/dataset/raw')"`

## §8.3 Baseline Metrics

**Status**: `SKIPPED_BY_FLAG`  
**Model**: `N/A`

_No metrics computed — reference data unavailable._

## §8.4 Fine-tuning Decision

**Decision**: `SKIPPED_INSUFFICIENT_GPU`  
Apple M2 MPS detected. MPS is suitable for inference but not recommended for fine-tuning a ViT-based depth backbone (insufficient VRAM, slow backward pass). Fine-tuning SKIPPED explicitly. The pretrained-only baseline is the MVP as per §8.4.

## §8.5 Comparison Table

See `comparison_table.json` for full details.

| Mode | Status |
|---|---|
| A — Pretrained depth only (no AERIS) | SKIPPED_NO_REFERENCE_DATA |
| B — Pretrained depth + full AERIS-3D | REQUIRES_DATASET_PRESENT_FOR_BATCH_EVALUATION |
| C — Fine-tuned depth + full AERIS-3D | SKIPPED |

> ⚠ All metrics are scale-invariant (prediction and GT normalized to [0,1]).
> Absolute metric accuracy requires GSD or reference DEM anchoring.
