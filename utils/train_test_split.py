import os
import random
from pathlib import Path

"""
Делю на train и val с учетом баланса RECT и SQUARE
"""

def create_yolo_dataset_files(base_path, output_path, train_ratio=0.8, seed=42):
    random.seed(seed)
    
    train_files = []
    val_files = []
    
    for data_type in ['rect', 'square']:
        images_path = f"{base_path}/{data_type}/images"
        images = [f for f in os.listdir(images_path) if f.endswith(('.jpg', '.jpeg', '.png'))]
        random.shuffle(images)
        
        split_idx = int(len(images) * train_ratio)
        
        for i, img_name in enumerate(images):
            img_path = f"../{images_path}/{img_name}"
            if i < split_idx:
                train_files.append(img_path)
            else:
                val_files.append(img_path)
    
    with open(f"{output_path}/train.txt", 'w') as f:
        f.write('\n'.join(train_files))
    
    with open(f"{output_path}/val.txt", 'w') as f:
        f.write('\n'.join(val_files))
    
    print(len(train_files))
    print(len(val_files))

create_yolo_dataset_files('data', 'data')