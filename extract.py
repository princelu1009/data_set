import os
from functools import partial
from multiprocessing import Pool, cpu_count
from PIL import Image
from tqdm import tqdm
import torch
from torchvision import transforms


from dataset import (
    build_train_val_sets,
    get_transforms,
    load_food101_split,
    collect_osteosarcoma,
    build_nih_df,
    PATH_FOOD,
    PATH_OXFORD,
    PATH_X_RAY,
    PATH_OS,
    PATH_STANFORD,
    PATH_NIH,
    CustomDataset,
    datasets,
    train_test_split
)

# ======================================================
# Directory Tools
# ======================================================
def ensure_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)


# ======================================================
# Save ONE ITEM of dataset (for multiprocessing)
# ======================================================
def save_one_item(args):
    img, label, fpath = args

    # convert tensor → PIL
    if not isinstance(img, Image.Image):
        img = transforms.ToPILImage()(img)

    ensure_dir(os.path.dirname(fpath))

    # Save based on extension
    ext = os.path.splitext(fpath)[-1].lower()
    if ext == ".webp":
        img.save(fpath, "WEBP", quality=95)
    else:
        img.save(fpath, "JPEG", quality=95)


# ======================================================
# Export ANY dataset variant
# Multiprocess version
# ======================================================
def export_dataset_variant(dataset, out_root, variant_name):
    ensure_dir(out_root)

    print(f"\n📦 Exporting {variant_name} → {out_root}")

    tasks = []

    # Prepare all file-writing tasks
    for idx in range(len(dataset)):
        img, label = dataset[idx]

        class_folder = os.path.join(out_root, str(label))
        ensure_dir(class_folder)

        # Choose extension: .jpg for JPEG, .webp for webp
        if "webp" in variant_name.lower():
            ext = ".webp"
        else:
            ext = ".jpg"

        fpath = os.path.join(class_folder, f"{idx}{ext}")
        tasks.append((img, label, fpath))

    # Multiprocessing write
    with Pool(cpu_count()) as p:
        list(tqdm(p.imap(save_one_item, tasks), total=len(tasks)))


# ======================================================
# Export ALL variants (clean / jpeg / webp)
# ======================================================
def export_full_dataset(set_dict, dataset_name, output_root="data"):
    root = os.path.join(output_root, dataset_name)

    # ---------------- JPEG ----------------
    jpeg_root = os.path.join(root, "jpeg")

    export_dataset_variant(set_dict["train_clean_jpeg"], os.path.join(jpeg_root, "train-clean"),     "clean-jpeg")
    export_dataset_variant(set_dict["train_comb_jpeg"],  os.path.join(jpeg_root, "train-combine"),   "combine-jpeg")
    export_dataset_variant(set_dict["val_clean"],        os.path.join(jpeg_root, "validate-clean"),  "clean-jpeg")
    export_dataset_variant(set_dict["val_comb_jpeg"],    os.path.join(jpeg_root, "validate-combine"),"combine-jpeg")

    # ---------------- WEBP ----------------
    webp_root = os.path.join(root, "webp")

    export_dataset_variant(set_dict["train_clean_jpeg"], os.path.join(webp_root, "train-clean"),     "clean-webp")
    export_dataset_variant(set_dict["train_comb_webp"],  os.path.join(webp_root, "train-combine"),   "combine-webp")
    export_dataset_variant(set_dict["val_clean"],        os.path.join(webp_root, "validate-clean"),  "clean-webp")
    export_dataset_variant(set_dict["val_comb_webp"],    os.path.join(webp_root, "validate-combine"),"combine-webp")

    print(f"\n🎉 Finished exporting {dataset_name}!\n")


# ======================================================
# MAIN PIPELINE
# ======================================================
if __name__ == "__main__":
    print("\n🚀 Starting optimized dataset extraction (multiprocessing)...\n")

    # default transforms
    train_tfms, val_tfms = get_transforms("other")

    # =====================================================
    # Oxford Pet
    # =====================================================
    train_pet = datasets.OxfordIIITPet(PATH_OXFORD, split="trainval", target_types="category")
    val_pet   = datasets.OxfordIIITPet(PATH_OXFORD, split="test",     target_types="category")

    pet_sets = build_train_val_sets(train_pet, val_pet, train_tfms, val_tfms)
    export_full_dataset(pet_sets, "oxford-pet")

    # =====================================================
    # Food-101
    # =====================================================
    train_food = load_food101_split(PATH_FOOD, "train")
    val_food   = load_food101_split(PATH_FOOD, "test")

    food_sets = build_train_val_sets(None, None, train_tfms, val_tfms,
                                     paths_train=train_food, paths_val=val_food)
    export_full_dataset(food_sets, "food-101")

    # =====================================================
    # Chest X-ray
    # =====================================================
    train_tfms_x, val_tfms_x = get_transforms("xray")

    x_train = datasets.ImageFolder(os.path.join(PATH_X_RAY, "train"))
    x_val   = datasets.ImageFolder(os.path.join(PATH_X_RAY, "test"))

    x_sets = build_train_val_sets(x_train, x_val, train_tfms_x, val_tfms_x)
    export_full_dataset(x_sets, "chest-xray")

    # =====================================================
    # Osteosarcoma
    # =====================================================
    os_all = collect_osteosarcoma(PATH_OS)
    os_train, os_val = train_test_split(os_all, test_size=0.2, random_state=42)

    os_sets = build_train_val_sets(None, None, train_tfms, val_tfms,
                                   paths_train=os_train, paths_val=os_val)
    export_full_dataset(os_sets, "osteosarcoma")

    # =====================================================
    # Stanford Cars
    # =====================================================
    st_train = datasets.ImageFolder(os.path.join(PATH_STANFORD, "train"))
    st_val   = datasets.ImageFolder(os.path.join(PATH_STANFORD, "test"))

    st_sets = build_train_val_sets(st_train, st_val, train_tfms, val_tfms)
    export_full_dataset(st_sets, "stanford-cars")

    # =====================================================
    # NIH CXR-8
    # =====================================================
    CSV_TRAIN = os.path.join(PATH_NIH, "csv", "train_clean.csv")
    CSV_VAL   = os.path.join(PATH_NIH, "csv", "val_clean.csv")
    IMG_ROOT  = os.path.join(PATH_NIH, "images", "images")

    df_train = build_nih_df(CSV_TRAIN, IMG_ROOT)
    df_val   = build_nih_df(CSV_VAL, IMG_ROOT)
    train_tfms_nih, val_tfms_nih = get_transforms("nih")

    nih_sets = build_train_val_sets(None, None, train_tfms_nih, val_tfms_nih,
                                    is_nih=True, paths_train=df_train, paths_val=df_val)
    export_full_dataset(nih_sets, "nih-cxr8")

    print("\n🎉 ALL DATASETS EXPORTED WITH MULTIPROCESSING SUCCESSFULLY!\n")
