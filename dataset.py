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
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset, ConcatDataset
from sklearn.model_selection import train_test_split
import random

# ======================================================
#  Custom Dataset
# ======================================================
class CustomDataset(Dataset):
    def __init__(self, base_dataset, transform=None, apply_compression=False, quality=50, is_nih=False):
        self.transform = transform
        self.apply_compression = apply_compression
        self.quality = quality
        self.is_nih = is_nih

        # Case A: NIH CSV (DataFrame)
        if is_nih:
            self.df = base_dataset     # base_dataset is DataFrame
            self.base = None
            self.is_path_list = False

        else:
            self.base = base_dataset
            self.is_path_list = isinstance(base_dataset, list)

            if self.is_path_list:
                classes = sorted({p.split("/")[-2] for p in base_dataset})
                self.class_to_idx = {c: i for i, c in enumerate(classes)}
            else:
                self.class_to_idx = None

    def __len__(self):
        if self.is_nih:
            return len(self.df)
        return len(self.base)

    def __getitem__(self, idx):

        # ========== NIH CSV case ==========
        if self.is_nih:
            row = self.df.iloc[idx]
            path = row["path"]
            label = int(row["label"])
            img = Image.open(path).convert("RGB")

        # ========== Torchvision dataset ==========
        elif not self.is_path_list:
            img, label = self.base[idx]

        # ========== Path list case (Food-101 / Osteosarcoma) ==========
        else:
            path = self.base[idx]
            img = Image.open(path).convert("RGB")
            class_name = path.split("/")[-2]
            label = self.class_to_idx[class_name]

        # ---- Compression ----
        if self.apply_compression:
            img = self._compress(img)

        # ---- Transform ----
        if self.transform:
            img = self.transform(img)

        return img, label

    def _compress(self, img_pil):
        img_cv = np.array(img_pil)[..., ::-1]

        fmt = random.choice(["jpeg", "webp"])
        flag = cv2.IMWRITE_WEBP_QUALITY if fmt == "webp" else cv2.IMWRITE_JPEG_QUALITY

        ok, buf = cv2.imencode("." + fmt, img_cv, [flag, self.quality])
        if not ok:
            return img_pil

        img_cv = cv2.imdecode(buf, cv2.IMREAD_COLOR)
        img_rgb = cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB)
        return Image.fromarray(img_rgb)


# ======================================================
#  Transforms
# ======================================================
def get_transforms(dataset_type):
    if dataset_type == "xray" or dataset_type == "nih":
        to_gray = [transforms.Grayscale(num_output_channels=3)]
        normalize = transforms.Normalize([0.485, 0.456, 0.406],
                                         [0.229, 0.224, 0.225])
    else:
        to_gray = []
        normalize = transforms.Normalize([0.485, 0.456, 0.406],
                                         [0.229, 0.224, 0.225])

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

#For Tumor
PATH_OS = "/Users/princelu/Desktop/ALL/ML Learning/Final/PKG - Osteosarcoma Tumor Assessment"

def collect_osteosarcoma(PATH_OS):
    roots = [
        os.path.join(PATH_OS, "Training-Set-1"),
        os.path.join(PATH_OS, "Training-Set-2"),
        PATH_OS
    ]

    all_paths = []

    for base in roots:
        if not os.path.exists(base):
            continue
        for root, _, files in os.walk(base):
            for f in files:
                if f.lower().endswith((".jpg", ".jpeg", ".png", ".webp")):
                    full = os.path.join(root, f)
                    all_paths.append(full)

    return sorted(all_paths)

PATH_OS_ALL=collect_osteosarcoma(PATH_OS)

train_paths, val_paths = train_test_split(
    PATH_OS_ALL, 
    test_size=0.2, 
    random_state=42,
    shuffle=True
)
train_clean_os = CustomDataset(train_paths, transform=train_tfms, apply_compression=False)
train_comp_os  = CustomDataset(train_paths, transform=train_tfms, apply_compression=True)
train_os = ConcatDataset([train_clean_os,train_comp_os ])

val_clean_os = CustomDataset(val_paths, transform=val_tfms, apply_compression=False)
val_comp_os  = CustomDataset(val_paths, transform=val_tfms, apply_compression=True)

print(len(train_os))

#For standford
PATH_STANFORD="/Users/princelu/Desktop/ALL/ML Learning/Final/stanford-cars/car_data/car_data"
train_dir_st = os.path.join(PATH_STANFORD, "train")
test_dir_st  = os.path.join(PATH_STANFORD, "test")
train_base_st = datasets.ImageFolder(train_dir_st)
val_base_st   = datasets.ImageFolder(test_dir_st)
train_clean_st = CustomDataset(train_base_st, transform=train_tfms, apply_compression=False)
train_comp_x_st  = CustomDataset(train_base_st, transform=train_tfms, apply_compression=True)
train_x_ray = ConcatDataset([train_clean_st,train_comp_x_st])
print(len(train_clean_st))

#For Nih
PATH_NIH = "/Users/princelu/Desktop/ALL/ML Learning/Final/CXR8"

CSV_TRAIN = f"{PATH_NIH}/csv/train_clean.csv"
CSV_VAL   = f"{PATH_NIH}/csv/val_clean.csv"

# IMPORTANT: the images are inside images/images/
IMG_ROOT  = f"{PATH_NIH}/images/images"

def build_nih_df(csv_path, img_root):
    df = pd.read_csv(csv_path)

    # Make full path to each image
    df["path"] = df["image"].apply(lambda x: os.path.join(img_root, x))

    # Filter only existing files
    df = df[df["path"].apply(os.path.exists)]
    return df.reset_index(drop=True)

# Load NIH CSVs
train_df_nih = build_nih_df(CSV_TRAIN, IMG_ROOT)
val_df_nih   = build_nih_df(CSV_VAL, IMG_ROOT)

# Build transforms
train_tfms_nih, val_tfms_nih = get_transforms("nih")

# Clean + Compressed
train_clean_nih = CustomDataset(train_df_nih, transform=train_tfms_nih,
                                apply_compression=False, is_nih=True)
train_comp_nih  = CustomDataset(train_df_nih, transform=train_tfms_nih,
                                apply_compression=True,  is_nih=True)
train_nih = ConcatDataset([train_clean_nih, train_comp_nih])

# Validation sets
val_clean_nih = CustomDataset(val_df_nih, transform=val_tfms_nih,
                              apply_compression=False, is_nih=True)
val_comp_nih  = CustomDataset(val_df_nih, transform=val_tfms_nih,
                              apply_compression=True,  is_nih=True)

print(len(train_clean_nih))



