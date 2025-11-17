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
#  DATASET PATHS
# ======================================================
PATH_OXFORD   = DATA_ROOT
PATH_FOOD     = os.path.join(DATA_ROOT, "food-101", "food-101")
PATH_X_RAY    = os.path.join(DATA_ROOT, "chest_xray")
PATH_OS       = os.path.join(DATA_ROOT, "PKG - Osteosarcoma Tumor Assessment")
PATH_STANFORD = os.path.join(DATA_ROOT, "stanford-cars", "car_data", "car_data")
PATH_NIH      = os.path.join(DATA_ROOT, "CXR8")


# ======================================================
#  Custom Dataset
# ======================================================
class CustomDataset(Dataset):
    def __init__(self, base_dataset, transform=None, mode="clean", quality=50, is_nih=False):
        self.transform = transform
        self.mode = mode
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
            img = Image.open(row["path"]).convert("RGB")
            label = int(row["label"])
        elif not self.is_path_list:
            img, label = self.base[idx]
        else:
            path = self.base[idx]
            img = Image.open(path).convert("RGB")
            label = self.class_to_idx[path.split("/")[-2]]

        if self.mode in ["jpeg", "webp"]:
            img = self._compress(img, ".jpg" if self.mode=="jpeg" else ".webp")

        if self.transform:
            img = self.transform(img)

        return img, label

    def _compress(self, img_pil, ext):
        img_cv = np.array(img_pil)[..., ::-1]
        flag = cv2.IMWRITE_JPEG_QUALITY if ext == ".jpg" else cv2.IMWRITE_WEBP_QUALITY
        ok, buf = cv2.imencode(ext, img_cv, [flag, self.quality])
        if not ok:
            return img_pil
        img_cv = cv2.imdecode(buf, cv2.IMREAD_COLOR)
        return Image.fromarray(cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB))


# ======================================================
#  Transforms
# ======================================================
def get_transforms(dataset_type):
    to_gray = [transforms.Grayscale(num_output_channels=3)] if dataset_type in ["xray","nih"] else []

    normalize = transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])

    train_tfms = transforms.Compose([
        transforms.Resize((224,224)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ColorJitter(0.2,0.2,0.2),
        *to_gray,
        transforms.ToTensor(),
        normalize
    ])

    val_tfms = transforms.Compose([
        transforms.Resize((224,224)),
        *to_gray,
        transforms.ToTensor(),
        normalize
    ])
    return train_tfms, val_tfms


# ======================================================
#  Loaders
# ======================================================
def load_food101_split(root, split="train"):
    images_root = os.path.join(root, "images")
    split_file = os.path.join(root, "meta", f"{split}.txt")
    with open(split_file, "r") as f:
        return [os.path.join(images_root, line.strip()+".jpg") for line in f]

def collect_osteosarcoma(root):
    all_paths=[]
    for base in [
        os.path.join(root,"Training-Set-1"),
        os.path.join(root,"Training-Set-2"),
        root]:
        if not os.path.exists(base): continue
        for r,_,files in os.walk(base):
            for f in files:
                if f.lower().endswith((".jpg",".jpeg",".png",".webp")):
                    all_paths.append(os.path.join(r,f))
    return sorted(all_paths)

def build_nih_df(csv_path, img_root):
    df = pd.read_csv(csv_path)
    df["path"] = df["image"].apply(lambda x: os.path.join(img_root, x))
    return df[df["path"].apply(os.path.exists)].reset_index(drop=True)


# ======================================================
#  UNIVERSAL BUILDER
# ======================================================
def build_train_val_sets(base_train, base_val, train_tfms, val_tfms,
                         is_nih=False, paths_train=None, paths_val=None):

    # ---- TRAIN ----
    if paths_train is None:
        train_clean = CustomDataset(base_train, transform=train_tfms, mode="clean", is_nih=is_nih)
        train_jpeg  = CustomDataset(base_train, transform=train_tfms, mode="jpeg", is_nih=is_nih)
        train_webp  = CustomDataset(base_train, transform=train_tfms, mode="webp", is_nih=is_nih)
    else:
        train_clean = CustomDataset(paths_train, transform=train_tfms, mode="clean")
        train_jpeg  = CustomDataset(paths_train, transform=train_tfms, mode="jpeg")
        train_webp  = CustomDataset(paths_train, transform=train_tfms, mode="webp")

    train_comb_jpeg = ConcatDataset([train_clean, train_jpeg])
    train_comb_webp = ConcatDataset([train_clean, train_webp])

    # ---- VALID ----
    if paths_val is None:
        val_clean = CustomDataset(base_val, transform=val_tfms, mode="clean", is_nih=is_nih)
        val_jpeg  = CustomDataset(base_val, transform=val_tfms, mode="jpeg", is_nih=is_nih)
        val_webp  = CustomDataset(base_val, transform=val_tfms, mode="webp", is_nih=is_nih)
    else:
        val_clean = CustomDataset(paths_val, transform=val_tfms, mode="clean")
        val_jpeg  = CustomDataset(paths_val, transform=val_tfms, mode="jpeg")
        val_webp  = CustomDataset(paths_val, transform=val_tfms, mode="webp")

    val_comb_jpeg = ConcatDataset([val_clean, val_jpeg])
    val_comb_webp = ConcatDataset([val_clean, val_webp])

    return {
        "train_clean_jpeg":train_clean,
        "train_comb_jpeg": train_comb_jpeg,
        "train_comb_webp": train_comb_webp,
        "val_clean":val_clean,
        "val_comb_jpeg": val_comb_jpeg,
        "val_comb_webp": val_comb_webp,
    }


# ======================================================
#  MAIN
# ======================================================
if __name__ == "__main__":

    train_tfms, val_tfms = get_transforms("other")

    # ------------------ Oxford Pet --------------------
    train_base_pet = datasets.OxfordIIITPet(PATH_OXFORD, split="trainval", target_types="category")
    val_base_pet   = datasets.OxfordIIITPet(PATH_OXFORD, split="test", target_types="category")
    pet_sets = build_train_val_sets(train_base_pet, val_base_pet, train_tfms, val_tfms)
    print("Oxford Pet JPEG:", len(pet_sets["train_comb_jpeg"]))
    print("Oxford Pet WEBP:", len(pet_sets["train_comb_webp"]))

    # ------------------ Food 101 ----------------------
    food_train = load_food101_split(PATH_FOOD, "train")
    food_val   = load_food101_split(PATH_FOOD, "test")
    food_sets = build_train_val_sets(None, None, train_tfms, val_tfms,
                                     paths_train=food_train, paths_val=food_val)
    print("Food101 JPEG:", len(food_sets["train_comb_jpeg"]))
    print("Food101 WEBP:", len(food_sets["train_comb_webp"]))


    # ------------------ Chest X-ray --------------------
    train_tfms_x, val_tfms_x = get_transforms("xray")
    x_train = datasets.ImageFolder(os.path.join(PATH_X_RAY,"train"))
    x_val   = datasets.ImageFolder(os.path.join(PATH_X_RAY,"test"))
    x_sets = build_train_val_sets(x_train, x_val, train_tfms_x, val_tfms_x)
    print("X-ray JPEG:", len(x_sets["train_comb_jpeg"]))

    # ------------------ Osteosarcoma -------------------
    os_all = collect_osteosarcoma(PATH_OS)
    os_train, os_val = train_test_split(os_all, test_size=0.2, random_state=42)
    os_sets = build_train_val_sets(None, None, train_tfms, val_tfms,
                                   paths_train=os_train, paths_val=os_val)
    print("Osteosarcoma JPEG:", len(os_sets["train_comb_jpeg"]))

    # ------------------ Stanford Cars ------------------
    st_train = datasets.ImageFolder(os.path.join(PATH_STANFORD,"train"))
    st_val   = datasets.ImageFolder(os.path.join(PATH_STANFORD,"test"))
    st_sets = build_train_val_sets(st_train, st_val, train_tfms, val_tfms)
    print("Stanford Cars JPEG:", len(st_sets["train_comb_jpeg"]))

    # ------------------ NIH CXR8 -----------------------
    CSV_TRAIN = os.path.join(PATH_NIH, "csv", "train_clean.csv")
    CSV_VAL   = os.path.join(PATH_NIH, "csv", "val_clean.csv")
    IMG_ROOT  = os.path.join(PATH_NIH, "images", "images")

    df_train = build_nih_df(CSV_TRAIN, IMG_ROOT)
    df_val   = build_nih_df(CSV_VAL, IMG_ROOT)
    train_tfms_nih, val_tfms_nih = get_transforms("nih")

    nih_sets = build_train_val_sets(None, None, train_tfms_nih, val_tfms_nih,
                                    is_nih=True, paths_train=df_train, paths_val=df_val)
    print("NIH JPEG:", len(nih_sets["train_comb_jpeg"]))

    print("\n🎉 ALL datasets built successfully!")
