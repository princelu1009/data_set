import os
import cv2
import torch
import random
import pandas as pd
import numpy as np

from PIL import Image
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, ConcatDataset
from torchvision import datasets, transforms

# ======================================================
#  PORTABLE ROOT PATH (WORKS ON ANY COMPUTER)
# ======================================================
DATA_ROOT = os.getenv("DATA_ROOT")
if DATA_ROOT is None:
    raise RuntimeError(
        "❌ DATA_ROOT environment variable not set.\n"
        "Set it before running:\n\n"
        "   export DATA_ROOT=\"/Users/you/datasets\"\n"
        "or\n"
        "   setx DATA_ROOT \"D:\\\\datasets\" (Windows)\n"
    )

# ======================================================
#  DATASET PATHS (RELATIVE TO DATA_ROOT)
# ======================================================
PATH_OXFORD   = DATA_ROOT                                     # contains oxford-iiit-pet/
PATH_FOOD     = os.path.join(DATA_ROOT, "food-101", "food-101")
PATH_X_RAY    = os.path.join(DATA_ROOT, "chest_xray")
PATH_OS       = os.path.join(DATA_ROOT, "PKG - Osteosarcoma Tumor Assessment")
PATH_STANFORD = os.path.join(DATA_ROOT, "stanford-cars", "car_data", "car_data")
PATH_NIH      = os.path.join(DATA_ROOT, "CXR8")


# ======================================================
#  Custom Dataset
# ======================================================
class CustomDataset(Dataset):
    def __init__(self, base_dataset, transform=None, apply_compression=False,
                 quality=50, is_nih=False):
        self.transform = transform
        self.apply_compression = apply_compression
        self.quality = quality
        self.is_nih = is_nih

        if is_nih:
            self.df = base_dataset
            self.base = None
            self.is_path_list = False
        else:
            self.df = None
            self.base = base_dataset
            self.is_path_list = isinstance(base_dataset, list)

            if self.is_path_list:
                classes = sorted({p.split("/")[-2] for p in base_dataset})
                self.class_to_idx = {c: i for i, c in enumerate(classes)}
            else:
                self.class_to_idx = None

    def __len__(self):
        return len(self.df) if self.is_nih else len(self.base)

    def __getitem__(self, idx):
        if self.is_nih:
            row = self.df.iloc[idx]
            path = row["path"]
            label = int(row["label"])
            img = Image.open(path).convert("RGB")

        elif not self.is_path_list:
            img, label = self.base[idx]

        else:
            path = self.base[idx]
            img = Image.open(path).convert("RGB")
            class_name = path.split("/")[-2]
            label = self.class_to_idx[class_name]

        if self.apply_compression:
            img = self._compress(img)

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
    if dataset_type in ["xray", "nih"]:
        to_gray = [transforms.Grayscale(num_output_channels=3)]
    else:
        to_gray = []

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


# ======================================================
#  Food-101 Loader
# ======================================================
def load_food101_split(root, split="train"):
    images_root = os.path.join(root, "images")
    split_file = os.path.join(root, "meta", f"{split}.txt")

    with open(split_file, "r") as f:
        items = [line.strip() for line in f]

    return [os.path.join(images_root, p + ".jpg") for p in items]


# ======================================================
#  Osteosarcoma Loader
# ======================================================
def collect_osteosarcoma(root):
    roots = [
        os.path.join(root, "Training-Set-1"),
        os.path.join(root, "Training-Set-2"),
        root
    ]

    all_paths = []
    for base in roots:
        if not os.path.exists(base):
            continue
        for r, _, files in os.walk(base):
            for f in files:
                if f.lower().endswith((".jpg", ".jpeg", ".png", ".webp")):
                    all_paths.append(os.path.join(r, f))

    return sorted(all_paths)


# ======================================================
#  NIH Loader
# ======================================================
def build_nih_df(csv_path, img_root):
    df = pd.read_csv(csv_path)
    df["path"] = df["image"].apply(lambda x: os.path.join(img_root, x))
    df = df[df["path"].apply(os.path.exists)]
    return df.reset_index(drop=True)


# ======================================================
#  MAIN
# ======================================================
if __name__ == "__main__":
    print("Using DATA_ROOT:", DATA_ROOT)

    train_tfms, val_tfms = get_transforms("other")

    # -------------------------------
    # Oxford-IIIT Pet
    # -------------------------------
    print("\nBuilding Oxford-IIIT Pet...")

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

    train_clean_pet = CustomDataset(train_base_pet, transform=train_tfms)
    train_comp_pet = CustomDataset(train_base_pet, transform=train_tfms, apply_compression=True)
    train_pet = ConcatDataset([train_clean_pet, train_comp_pet])

    print("Oxford Pet train:", len(train_pet))

    # -------------------------------
    # Food-101
    # -------------------------------
    print("\nBuilding Food-101...")

    food_train_paths = load_food101_split(PATH_FOOD, "train")

    train_clean_food = CustomDataset(food_train_paths, transform=train_tfms)
    train_comp_food = CustomDataset(food_train_paths, transform=train_tfms, apply_compression=True)
    train_food = ConcatDataset([train_clean_food, train_comp_food])

    print("Food-101 train:", len(train_food))

    # -------------------------------
    # Chest X-ray
    # -------------------------------
    print("\nBuilding Chest X-ray...")

    train_tfms_xray, val_tfms_xray = get_transforms("xray")

    train_base_xray = datasets.ImageFolder(os.path.join(PATH_X_RAY, "train"))
    val_base_xray = datasets.ImageFolder(os.path.join(PATH_X_RAY, "test"))

    train_clean_xray = CustomDataset(train_base_xray, transform=train_tfms_xray)
    train_comp_xray = CustomDataset(train_base_xray, transform=train_tfms_xray, apply_compression=True)
    train_x_ray = ConcatDataset([train_clean_xray, train_comp_xray])

    print("Chest X-ray train:", len(train_x_ray))

    # -------------------------------
    # Osteosarcoma
    # -------------------------------
    print("\nBuilding Osteosarcoma...")

    OS_ALL = collect_osteosarcoma(PATH_OS)
    train_paths, val_paths = train_test_split(OS_ALL, test_size=0.2, shuffle=True, random_state=42)

    train_clean_os = CustomDataset(train_paths, transform=train_tfms)
    train_comp_os = CustomDataset(train_paths, transform=train_tfms, apply_compression=True)
    train_os = ConcatDataset([train_clean_os, train_comp_os])

    print("Osteosarcoma train:", len(train_os))

    # -------------------------------
    # Stanford Cars
    # -------------------------------
    print("\nBuilding Stanford Cars...")

    train_base_st = datasets.ImageFolder(os.path.join(PATH_STANFORD, "train"))
    val_base_st = datasets.ImageFolder(os.path.join(PATH_STANFORD, "test"))

    train_clean_st = CustomDataset(train_base_st, transform=train_tfms)
    train_comp_st = CustomDataset(train_base_st, transform=train_tfms, apply_compression=True)
    train_stanford = ConcatDataset([train_clean_st, train_comp_st])

    print("Stanford Cars train:", len(train_stanford))

    # -------------------------------
    # NIH
    # -------------------------------
    print("\nBuilding NIH CXR8...")

    CSV_TRAIN = os.path.join(PATH_NIH, "csv", "train_clean.csv")
    CSV_VAL   = os.path.join(PATH_NIH, "csv", "val_clean.csv")
    IMG_ROOT  = os.path.join(PATH_NIH, "images", "images")

    train_df_nih = build_nih_df(CSV_TRAIN, IMG_ROOT)

    train_tfms_nih, _ = get_transforms("nih")

    train_clean_nih = CustomDataset(train_df_nih, transform=train_tfms_nih, is_nih=True)
    train_comp_nih  = CustomDataset(train_df_nih, transform=train_tfms_nih, apply_compression=True, is_nih=True)
    train_nih = ConcatDataset([train_clean_nih, train_comp_nih])

    print("NIH train:", len(train_nih))

    print("\n🎉 All datasets built successfully!")
