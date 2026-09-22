import os
import glob
import h5py
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoModelForDepthEstimation, AutoImageProcessor
from torch.amp import GradScaler, autocast
import matplotlib.pyplot as plt
import cv2

class GAMUSDataset(Dataset):
    def __init__(self, images_dir, heights_dir, processor, max_samples=None):
        self.image_files = sorted(glob.glob(os.path.join(images_dir, "*.h5")))
        self.height_files = sorted(glob.glob(os.path.join(heights_dir, "*.h5")))
        if max_samples:
            self.image_files = self.image_files[:max_samples]
            self.height_files = self.height_files[:max_samples]
        self.processor = processor
        
    def __len__(self):
        return len(self.image_files)
        
    def __getitem__(self, idx):
        img_path = self.image_files[idx]
        # Heights file is named _AGL instead of _RGB
        hgt_path = self.height_files[idx]
        
        with h5py.File(img_path, 'r') as f:
            image = f['image'][:]
            
        with h5py.File(hgt_path, 'r') as f:
            height = f['image'][:]
            
        # process image
        inputs = self.processor(images=image, return_tensors="pt")
        pixel_values = inputs.pixel_values.squeeze(0) # (3, H, W)
        
        # height resize to match pixel_values H,W
        h, w = pixel_values.shape[1], pixel_values.shape[2]
        height_resized = cv2.resize(height, (w, h), interpolation=cv2.INTER_LINEAR)
        labels = torch.tensor(height_resized, dtype=torch.float32)
        
        return {"pixel_values": pixel_values, "labels": labels}

def train():
    model_name = "depth-anything/Depth-Anything-V2-Small-hf"
    print("Loading processor and model...")
    processor = AutoImageProcessor.from_pretrained(model_name)
    model = AutoModelForDepthEstimation.from_pretrained(model_name)
    model.train()
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    
    train_dataset = GAMUSDataset(
        r"n:\AERIS-3D-main\data\GAMUS\images\train",
        r"n:\AERIS-3D-main\data\GAMUS\heights\train",
        processor,
        max_samples=200 # Train on subset to fit in time constraints
    )
    
    dataloader = DataLoader(train_dataset, batch_size=4, shuffle=True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    scaler = torch.amp.GradScaler('cuda')
    
    print("Starting training...")
    epochs = 1
    for epoch in range(epochs):
        for i, batch in enumerate(dataloader):
            pixel_values = batch["pixel_values"].to(device)
            labels = batch["labels"].to(device)
            
            optimizer.zero_grad()
            
            with autocast('cuda'):
                outputs = model(pixel_values=pixel_values)
                # Some models calculate loss internally, some don't. We compute it manually just in case.
                loss = torch.nn.functional.l1_loss(outputs.predicted_depth, labels)
            
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
            if i % 10 == 0:
                print(f"Epoch {epoch} Step {i}/{len(dataloader)}: Loss = {loss.item():.4f}")
                
    # Save
    out_dir = r"n:\AERIS-3D-main\data\finetuned_model"
    os.makedirs(out_dir, exist_ok=True)
    model.save_pretrained(out_dir)
    processor.save_pretrained(out_dir)
    print(f"Model saved to {out_dir}")

def test():
    print("Testing model...")
    out_dir = r"n:\AERIS-3D-main\data\finetuned_model"
    model = AutoModelForDepthEstimation.from_pretrained(out_dir)
    processor = AutoImageProcessor.from_pretrained(out_dir)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    model.eval()
    
    # We will pick a held-out image from the test set
    test_img_path = r"n:\AERIS-3D-main\data\GAMUS\images\test\DC_03_26_RGB.h5"
    with h5py.File(test_img_path, 'r') as f:
        image = f['image'][:]
        
    inputs = processor(images=image, return_tensors="pt").to(device)
    with torch.no_grad():
        with autocast('cuda'):
            outputs = model(**inputs)
            
    depth_pred = outputs.predicted_depth.squeeze().cpu().numpy()
    
    # Save the plot
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.imshow(image)
    plt.title("RGB")
    plt.subplot(1, 2, 2)
    plt.imshow(depth_pred, cmap="inferno")
    plt.title("Predicted Height")
    plt.savefig(r"n:\AERIS-3D-main\data\test_output.png")
    print("Test output saved to data/test_output.png")

if __name__ == "__main__":
    train()
    test()
