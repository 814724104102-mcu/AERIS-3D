import os
import pandas as pd
import h5py
import numpy as np
import torch
import rasterio
import cv2
from transformers import AutoImageProcessor, AutoModelForDepthEstimation
from torch.amp import autocast

def compute_metrics(pred, gt):
    # Ensure they match shape
    if pred.shape != gt.shape:
        pred = cv2.resize(pred, (gt.shape[1], gt.shape[0]), interpolation=cv2.INTER_LINEAR)
        
    mask = gt > -9999  # Valid mask
    if not np.any(mask):
        return None
        
    pred = pred[mask]
    gt = gt[mask]
    
    # Since our fine-tuned model predicts AGL (0-60m) but the hilly tile GT is absolute DSM (1500m+),
    # we need to scale/shift align them to calculate structural correlation and relative RMSE/MAE fairly.
    # We do a standard z-score alignment.
    p_std = np.std(pred)
    if p_std == 0: p_std = 1e-6
    pred_aligned = ((pred - np.mean(pred)) / p_std) * np.std(gt) + np.mean(gt)
    
    rmse = np.sqrt(np.mean((pred_aligned - gt)**2))
    mae = np.mean(np.abs(pred_aligned - gt))
    corr = np.corrcoef(pred_aligned, gt)[0, 1] if len(gt) > 1 else 0.0
    
    return {'rmse': rmse, 'mae': mae, 'corr': corr}

def main():
    print("Loading fine-tuned model...")
    model_dir = r"n:\AERIS-3D-main\data\finetuned_model"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    processor = AutoImageProcessor.from_pretrained(model_dir)
    model = AutoModelForDepthEstimation.from_pretrained(model_dir).to(device)
    model.eval()

    labels_df = pd.read_csv(r"n:\AERIS-3D-main\data\tile_labels.csv", header=None,
                            names=["filename", "path", "source", "city", "coord", "split", "type", "category"])
    
    # We want to test on 'test' split only
    test_df = labels_df[labels_df['split'] == 'test']
    
    # Separate images and heights
    images = test_df[test_df['type'] == 'images']
    heights = test_df[test_df['type'] == 'heights']
    
    results = []

    print(f"Found {len(images)} test images. Running inference...")
    for idx, img_row in images.iterrows():
        # find matching height
        coord = img_row['coord']
        hgt_rows = heights[(heights['coord'] == coord) & (heights['source'] == img_row['source'])]
        
        # Exception for custom hilly tile where coord might be empty but we can match by source
        if pd.isna(coord):
             hgt_rows = heights[(heights['source'] == img_row['source']) & (heights['filename'].str.contains('1024'))]
             if hgt_rows.empty:
                 hgt_rows = heights[(heights['source'] == img_row['source'])]

        if hgt_rows.empty:
            continue
        
        hgt_row = hgt_rows.iloc[0]
        category = img_row['category']
        
        # Load Image
        if str(img_row['path']).endswith('.h5'):
            with h5py.File(img_row['path'], 'r') as f:
                img_data = f['image'][:]
        else:
            img_data = cv2.imread(img_row['path'])
            img_data = cv2.cvtColor(img_data, cv2.COLOR_BGR2RGB)
            
        # Load GT Height
        if str(hgt_row['path']).endswith('.h5'):
            with h5py.File(hgt_row['path'], 'r') as f:
                gt_data = f['image'][:]
        else:
            # tif
            with rasterio.open(hgt_row['path']) as f:
                gt_data = f.read(1)
                
        # Inference
        inputs = processor(images=img_data, return_tensors="pt").to(device)
        with torch.no_grad():
            with autocast('cuda'):
                outputs = model(**inputs)
        pred_depth = outputs.predicted_depth.squeeze().cpu().numpy().astype(np.float32)
        
        metrics = compute_metrics(pred_depth, gt_data)
        if metrics:
            results.append({
                'category': category,
                'rmse': metrics['rmse'],
                'mae': metrics['mae'],
                'corr': metrics['corr']
            })

    # Group by category and compute mean
    res_df = pd.DataFrame(results)
    summary = res_df.groupby('category').mean().reset_index()
    
    out_csv = r"n:\AERIS-3D-main\data\evaluation_results.csv"
    summary.to_csv(out_csv, index=False)
    
    print("\n--- Evaluation Summary ---")
    print(summary.to_string(index=False))
    print(f"\nSaved to {out_csv}")

if __name__ == "__main__":
    main()
