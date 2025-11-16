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
        """
        base_dataset can be:
        - A torchvision dataset that returns (PIL, label)
        - A list of file paths (Food-101)
        """
        self.base = base_dataset
        self.transform = transform
        self.apply_compression = apply_compression
        self.quality = quality

        # ---- detect if base_dataset is list of paths ----
        self.is_path_list = isinstance(base_dataset, list)

        if self.is_path_list:
            # Build class indexing automatically from folder names
            class_names = sorted(list(set(p.split("/")[-2] for p in base_dataset)))
            self.class_to_idx = {c: i for i, c in enumerate(class_names)}
        else:
            self.class_to_idx = None  # dataset provides its own label

    def __len__(self):
        return len(self.base)

    def __getitem__(self, idx):
        
        # ========= CASE 1: base_dataset returns (PIL, label) =========
        if not self.is_path_list:
            x, y = self.base[idx]   # Oxford-IIIT Pet, ImageFolder, Stanford Cars

        # ========= CASE 2: base_dataset is a list of image paths =========
        else:
            path = self.base[idx]

            # Load PIL image
            x = Image.open(path).convert("RGB")

            # Label from folder name
            class_name = path.split("/")[-2]
            y = self.class_to_idx[class_name]

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

val_base_pet = datasets.OxfordIIITPet(
    root=root,
    split="test",
    target_types="category",
    download=False
)

train_tfms, val_tfms = get_transforms()

train_clean_pet = CustomDataset(train_base_pet, transform=train_tfms, apply_compression=False)
train_comp_pet  = CustomDataset(train_base_pet, transform=train_tfms, apply_compression=True)
train_pet = ConcatDataset([train_clean_pet, train_comp_pet])
print(len(train_pet))


#For Food 101
PATH="/Users/princelu/Desktop/ALL/ML Learning/Final/food-101/food-101"

def load_food101_split(root, split="train"):
    images_root = os.path.join(root, "images")
    split_file = os.path.join(root, "meta", f"{split}.txt")

    with open(split_file, "r") as f:
        items = [line.strip() for line in f]

    paths = [os.path.join(images_root, p + ".jpg") for p in items]
    return paths

food_root = "/Users/princelu/Desktop/ALL/ML Learning/Final/food-101/food-101"
food_train_paths = load_food101_split(food_root, "train")

train_clean_food = CustomDataset(food_train_paths, transform=train_tfms, apply_compression=False)
train_comp_food  = CustomDataset(food_train_paths, transform=train_tfms, apply_compression=True)

train_food = ConcatDataset([train_clean_food, train_comp_food])
print(len(train_food))

