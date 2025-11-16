import os, random, cv2, torch, numpy as np, pandas as pd
from PIL import Image
from tqdm import tqdm
from sklearn.metrics import f1_score
from torch.utils.data import Dataset, DataLoader, ConcatDataset
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms, models

class CustomDataset(Dataset):
    def __init__(self, base_dataset, transform=None, apply_compression=False):
        self.base = base_dataset
        self.transform = transform
        self.apply_compression = apply_compression

    def __len__(self):
        return len(self.base)

    def __getitem__(self, idx):
        x, y = self.base[idx]
        
        if self.apply_compression:
            img_cv = np.array(x)[..., ::-1]  # RGB->BGR
            fmt = random.choice(["jpeg", "webp"])
            q = 50
            flag = cv2.IMWRITE_WEBP_QUALITY if fmt == "webp" else cv2.IMWRITE_JPEG_QUALITY
            ok, buf = cv2.imencode(f".{fmt}", img_cv, [flag, q])
            if ok:
                img_cv = cv2.imdecode(buf, cv2.IMREAD_COLOR)
                x = Image.fromarray(cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB))

        if self.transform:
            x = self.transform(x)
        return x, y

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