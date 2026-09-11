"""
Dataset Splitter for AERIS-3D.
Creates scene/geography-aware Train/Val/Test manifests (CSV).
"""
import pandas as pd
from pathlib import Path
from core.config_loader import load_config
from core.logger import get_logger

log = get_logger(__name__)

def generate_splits(ds, output_dir: Path):
    """
    Splits the dataset and saves manifest CSVs.
    For the MVP subset, we do a simple sequential split.
    """
    config = load_config()
    ratios = config.get("dataset", {}).get("split_ratios", [0.7, 0.15, 0.15])
    
    total = len(ds)
    train_end = int(total * ratios[0])
    val_end = train_end + int(total * ratios[1])
    
    # We create a simple dataframe to act as our manifest
    # For a real dataset, this would group by scene/city to avoid geographic bleed
    data = []
    for i in range(total):
        # We store just an index or identifier since actual data is in the HF Dataset
        data.append({"sample_index": i, "id": f"sample_{i}"})
        
    df = pd.DataFrame(data)
    
    train_df = df.iloc[:train_end]
    val_df = df.iloc[train_end:val_end]
    test_df = df.iloc[val_end:]
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    train_df.to_csv(output_dir / "train.csv", index=False)
    val_df.to_csv(output_dir / "val.csv", index=False)
    test_df.to_csv(output_dir / "test.csv", index=False)
    
    log.info(f"Splits generated: Train({len(train_df)}) Val({len(val_df)}) Test({len(test_df)})")
    
    # Return the test indices so we can use them for the baseline evaluation
    return test_df["sample_index"].tolist()
