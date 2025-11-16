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

import os
import cv2
import random
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset, ConcatDataset
from torchvision import datasets, transforms


# ======================================================
#  Custom Dataset
# ======================================================
class CustomDataset(Dataset):
    def __init__(self, base_dataset, transform=None, apply_compression=False, quality=50):
        """
        base_dataset can be:
        - torchvision dataset (returns PIL, label)
        - list of image paths (Food-101)
        """
        self.base = base_dataset
        self.transform = transform
        self.apply_compression = apply_compression
        self.quality = quality

        self.is_path_list = isinstance(base_dataset, list)

        if self.is_path_list:
            # Extract class names from folder structure
            classes = sorted({p.split("/")[-2] for p in base_dataset})
            self.class_to_idx = {c: i for i, c in enumerate(classes)}
        else:
            self.class_to_idx = None

    def __len__(self):
        return len(self.base)

    def __getitem__(self, idx):

        # Case 1: torchvision dataset
        if not self.is_path_list:
            x, y = self.base[idx]

        # Case 2: path list (Food-101)
        else:
            path = self.base[idx]
            x = Image.open(path).convert("RGB")
            class_name = path.split("/")[-2]
            y = self.class_to_idx[class_name]

        # Compression
        if self.apply_compression:
            x = self._compress(x)

        if self.transform:
            x = self.transform(x)

        return x, y

    def _compress(self, img_pil):
        img_cv = np.array(img_pil)[..., ::-1]  # RGB → BGR

        fmt = random.choice(["jpeg", "webp"])
        flag = cv2.IMWRITE_WEBP_QUALITY if fmt == "webp" else cv2.IMWRITE_JPEG_QUALITY
        ok, buf = cv2.imencode("." + fmt, img_cv, [flag, self.quality])

        if not ok:
            return img_pil

        img_cv = cv2.imdecode(buf, cv2.IMREAD_COLOR)
        img_rgb = cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB)
        return Image.fromarray(img_rgb)


# ======================================================
#  Transform builder
# ======================================================
def get_transforms(dataset_type):
    """
    dataset_type = "pet" | "food" | "xray"
    """
    if dataset_type == "xray":
        # True grayscale → 3 channels
        to_gray = [transforms.Grayscale(num_output_channels=3)]
        normalize = transforms.Normalize(
            [0.485, 0.456, 0.406],
            [0.229, 0.224, 0.225]
        )
    else:
        to_gray = []  # keep RGB
        normalize = transforms.Normalize(
            [0.485, 0.456, 0.406],
            [0.229, 0.224, 0.225]
        )

    train_tfms = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ColorJitter(0.2, 0.2, 0.2),
        *to_gray,
        transforms.ToTensor(),
        normalize
    ])

    val_tfms = transforms.Compose([
        transforms.Resize((224, 224)),
        *to_gray,
        transforms.ToTensor(),
        normalize
    ])

    return train_tfms, val_tfms

#For OxfordIIITPet
PATH_OXFORD = "/Users/princelu/Desktop/ALL/ML Learning/Final"

train_base_pet = datasets.OxfordIIITPet(
    root=PATH_OXFORD,
    split="trainval",
    target_types="category",
    download=False
)

val_base_pet = datasets.OxfordIIITPet(
    root=PATH_OXFORD,
    split="test",
    target_types="category",
    download=False
)

train_tfms, val_tfms = get_transforms("other")

train_clean_pet = CustomDataset(train_base_pet, transform=train_tfms, apply_compression=False)
train_comp_pet  = CustomDataset(train_base_pet, transform=train_tfms, apply_compression=True)
train_pet = ConcatDataset([train_clean_pet, train_comp_pet])
print(len(train_pet))


#For Food 101
def load_food101_split(root, split="train"):
    images_root = os.path.join(root, "images")
    split_file = os.path.join(root, "meta", f"{split}.txt")

    with open(split_file, "r") as f:
        items = [line.strip() for line in f]

    paths = [os.path.join(images_root, p + ".jpg") for p in items]
    return paths

PATH_FOOD = "/Users/princelu/Desktop/ALL/ML Learning/Final/food-101/food-101"
food_train_paths = load_food101_split(PATH_FOOD, "train")

train_clean_food = CustomDataset(food_train_paths, transform=train_tfms, apply_compression=False)
train_comp_food  = CustomDataset(food_train_paths, transform=train_tfms, apply_compression=True)

train_food = ConcatDataset([train_clean_food, train_comp_food])
print(len(train_food))

#For chest x-ray
train_tfms_x_ray, val_tfms_x_ray = get_transforms('xray')

PATH_X_RAY = "/Users/princelu/Desktop/ALL/ML Learning/Final/chest_xray"
train_base_x_ray = datasets.ImageFolder(os.path.join(PATH_X_RAY , "train"))
val_base_x_ray   = datasets.ImageFolder(os.path.join(PATH_X_RAY , "test"))
train_clean_x_ray = CustomDataset(train_base_x_ray, transform=train_tfms_x_ray, apply_compression=False)
train_comp_x_ray  = CustomDataset(train_base_x_ray, transform=train_tfms_x_ray, apply_compression=True)
train_x_ray = ConcatDataset([train_clean_x_ray,train_comp_x_ray ])
print(len(train_x_ray))
