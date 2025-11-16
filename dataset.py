import os
import kagglehub
import cv2
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, ConcatDataset
from torchvision import models, datasets, transforms
from PIL import Image
from tqdm import tqdm
from sklearn.metrics import f1_score
from torch.utils.data import Dataset
import numpy as np
import pandas as pd
import random

class CustomDataset(Dataset):
    def __init__(self, base_dataset, transform=None, apply_compression=False, quality=50):
        self.base = base_dataset
        self.transform = transform
        self.apply_compression = apply_compression
        self.quality = quality

    def __len__(self):
        return len(self.base)

    def __getitem__(self, idx):
        x, y = self.base[idx]               # PIL image, label
        
        # ---- apply compression ----
        if self.apply_compression:
            x = self._compress_pil_image(x, quality=self.quality)

        # ---- apply transforms ----
        if self.transform:
            x = self.transform(x)

        return x, y

    def _compress_pil_image(self, img_pil, quality=50):
        img_np = np.array(img_pil)[..., ::-1]  # RGB→BGR

        fmt = random.choice(["jpeg", "webp"])
        flag = cv2.IMWRITE_WEBP_QUALITY if fmt == "webp" else cv2.IMWRITE_JPEG_QUALITY

        ok, encoded = cv2.imencode(f".{fmt}", img_np, [flag, quality])
        if not ok:
            return img_pil

        decoded = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
        decoded = cv2.cvtColor(decoded, cv2.COLOR_BGR2RGB)
        return Image.fromarray(decoded)

def get_transforms(is_grayscale=False):
    if is_grayscale:
        normalize = transforms.Normalize([0.5], [0.25])
    else:
        normalize = transforms.Normalize([0.485, 0.456, 0.406],
                                         [0.229, 0.224, 0.225])
    train_tfms = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.ToTensor(),
        normalize
    ])
    val_tfms = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        normalize
    ])
    return train_tfms, val_tfms


#For OxfordIIITPet
root = "/Users/princelu/Desktop/ALL/ML Learning/Final"

train_base_pet = datasets.OxfordIIITPet(
    root=root,
    split="trainval",
    target_types="category",
    download=False
)

val_base = datasets.OxfordIIITPet(
        root=root, split="test",
        target_types="category", download=True)

#Apply compression 
train_clean = CustomDataset(train_base_pet, transform=get_transforms(), apply_compression=False)
train_comp  = CustomDataset(train_base_pet, transform=get_transforms(), apply_compression=True)
train_set = ConcatDataset([train_clean, train_comp])
print(len(train_clean))
print(len(train_set))
