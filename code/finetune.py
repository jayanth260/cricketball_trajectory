import cv2
import pandas as pd
import numpy as np
from ultralytics import YOLO
import os
import argparse
from tqdm import tqdm

def train_model(model_name = "yolov8n.pt", modell = None, data_path ="/content/dataset/data.yaml"  ):
    if modell is None:
        model = YOLO(model_name)
    else:
        model = modell
        if 'model' not in model.overrides:
            model.overrides['model'] = model_name


    # Train the model
    results = model.train(
        data=data_path,
        epochs=100,            
        imgsz=1280,
        batch=16,
        project = "/content/drive/MyDrive/Edgefleet_Assessment/models",
        name="cricket_ball_v26",

        # --- Overfitting Controls ---
        patience=10,           # Stop if no improvement for 10 epochs (Prevents Overfitting)
        dropout=0.05,           # Use 0.1 or 0.2 if your dataset is very small (<500 images)
        optimizer='AdamW',     # Generally more stable than SGD for small datasets
        lr0 = 0.001,
        lrf=0.01,
        # --- Augmentation (YOLO defaults are usually good, but these help) ---
        augment=True,          # Ensure augmentation is on
        mosaic=0.0,
        # blur=0.1,
        # box=7.5,
        # degrees=10.0,          # Rotate ball slightly (helps if camera angle changes)
        # fliplr=0.5,            # Flip left-right (cricket pitch is symmetrical)
        device=0,
    )

    return model

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train YOLO model for cricket ball detection')
    parser.add_argument('--model', type=str, default='yolov8n.pt', 
                        help='Path to YOLO model (default: yolov8n.pt)')
    parser.add_argument('--data', type=str, default='/content/dataset/data.yaml',
                        help='Path to data.yaml file (default: /content/dataset/data.yaml)')
    parser.add_argument('--resume', type=str, default=None,
                        help='Path to checkpoint to resume training from')
    
    args = parser.parse_args()
    
    # Load model if resuming
    model_to_use = None
    if args.resume:
        model_to_use = YOLO(args.resume)
    
    trained_model = train_model(
        model_name=args.model,
        modell=model_to_use,
        data_path=args.data
    )
