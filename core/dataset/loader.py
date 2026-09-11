"""
Dataset loader for AERIS-3D.
Handles downloading and caching the GAMUS remote-sensing dataset.
"""
from datasets import load_dataset, Dataset
from core.config_loader import load_config
from core.logger import get_logger

log = get_logger(__name__)

def load_gamus_subset() -> Dataset:
    """
    Loads a small subset of the GAMUS dataset for baseline metrics generation.
    Uses streaming to avoid downloading the entire massive dataset during prototyping.
    """
    config = load_config()
    ds_path = config.get("dataset", {}).get("path", "earthflow/GAMUS")
    subset_size = config.get("dataset", {}).get("subset_size", 100)
    
    log.info(f"Loading dataset from {ds_path} (subset: {subset_size} samples)...")
    try:
        # Load in streaming mode to just take the first N samples
        ds = load_dataset(ds_path, split="train", streaming=True)
        samples = list(ds.take(subset_size))
        
        # Convert to a regular Dataset object
        small_ds = Dataset.from_list(samples)
        
        log.info(f"Successfully loaded {len(small_ds)} samples from {ds_path}.")
        return small_ds
    except Exception as e:
        log.error(f"Failed to load dataset {ds_path}: {e}")
        return None
