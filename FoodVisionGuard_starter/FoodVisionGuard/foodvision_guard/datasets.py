import os, csv, glob, random
from typing import List, Tuple, Dict, Optional
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import albumentations as A
from albumentations.pytorch import ToTensorV2

def build_transforms(img_size: int, is_train: bool):
    if is_train:
        return A.Compose([
            A.LongestMaxSize(max_size=img_size),
            A.PadIfNeeded(min_height=img_size, min_width=img_size, border_mode=0, value=(0,0,0)),
            A.HorizontalFlip(p=0.5),
            A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.05, rotate_limit=10, p=0.7),
            A.RandomBrightnessContrast(p=0.7),
            A.ColorJitter(p=0.5),
            A.MotionBlur(p=0.2),
            A.GaussNoise(p=0.2),
            A.Normalize(),
            ToTensorV2(),
        ])
    else:
        return A.Compose([
            A.LongestMaxSize(max_size=img_size),
            A.PadIfNeeded(min_height=img_size, min_width=img_size, border_mode=0, value=(0,0,0)),
            A.Normalize(),
            ToTensorV2(),
        ])

class FolderClassificationDataset(Dataset):
    def __init__(self, root_dir: str, classes: List[str], img_size: int, is_train: bool):
        self.root_dir = root_dir
        self.classes = classes
        self.class_to_idx = {c:i for i,c in enumerate(classes)}
        self.samples = []
        for c in classes:
            for p in glob.glob(os.path.join(root_dir, c, "**", "*.*"), recursive=True):
                if p.lower().endswith((".jpg",".jpeg",".png",".bmp",".webp")):
                    self.samples.append( (p, self.class_to_idx[c]) )
        self.transform = build_transforms(img_size, is_train)

    def __len__(self): return len(self.samples)
    def __getitem__(self, idx):
        path, label = self.samples[idx]
        img = Image.open(path).convert("RGB")
        img = self.transform(image=np.array(img))["image"]
        return img, torch.tensor(label, dtype=torch.long), path

import numpy as np

class FishMultiHeadDataset(Dataset):
    def __init__(self, csv_path: str, img_root: str, labels: Dict[str, List[str]], img_size: int, is_train: bool):
        self.records = []
        with open(csv_path, newline='', encoding="utf-8") as f:
            rdr = csv.DictReader(f)
            for row in rdr:
                self.records.append(row)
        self.img_root = img_root
        self.labels = labels
        self.label_to_idx = {k: {name:i for i,name in enumerate(v)} for k,v in labels.items()}
        self.transform = build_transforms(img_size, is_train)

    def __len__(self): return len(self.records)
    def __getitem__(self, idx):
        r = self.records[idx]
        path = os.path.join(self.img_root, r["filepath"])
        img = Image.open(path).convert("RGB")
        img = self.transform(image=np.array(img))["image"]
        out = {}
        for head in ["eye","gill","skin"]:
            out[head] = torch.tensor(self.label_to_idx[head][r[f"{head}_label"]], dtype=torch.long)
        return img, out, path
