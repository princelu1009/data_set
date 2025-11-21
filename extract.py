import os
from multiprocessing import Pool, cpu_count
from PIL import Image
from tqdm import tqdm
import torch
from torchvision import transforms

from dataset import (
    build_train_val_sets,
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
# Directory maker
# ======================================================
def ensure_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)


# ======================================================
# Save ONE item (worker)
# ======================================================
def save_one_item(args):
    img, label, fpath, is_clean = args

    # If tensor → unnormalize safely
    if isinstance(img, torch.Tensor):
        if img.min() < 0 or img.max() > 1:
            mean = torch.tensor([0.485, 0.456, 0.406]).view(3,1,1)
            std  = torch.tensor([0.229, 0.224, 0.225]).view(3,1,1)
            img = img * std + mean

        img = img.clamp(0, 1)
        img = transforms.ToPILImage()(img)

    ensure_dir(os.path.dirname(fpath))

    ext = os.path.splitext(fpath)[1].lower()

    # -----------------------------
    # clean = 95, compressed = 50
    # -----------------------------
    q = 95 if is_clean else 50

    if ext == ".webp":
        img.save(fpath, "WEBP", quality=q)
    else:
        img.save(fpath, "JPEG", quality=q)


# ======================================================
# Export a single variant (clean / combine)
# ======================================================
# ======================================================
# Export a single variant (clean / combine)
# ======================================================
def export_dataset_variant(dataset, out_root, variant_name, is_clean):
    ensure_dir(out_root)
    print(f"\n📦 Exporting {variant_name} → {out_root}")

    tasks = []
    for idx in range(len(dataset)):
        img, label = dataset[idx]

        class_folder = os.path.join(out_root, str(label))
        ensure_dir(class_folder)

        ext = ".webp" if "webp" in variant_name else ".jpg"
        fpath = os.path.join(class_folder, f"{idx}{ext}")

        tasks.append((img, label, fpath, is_clean))
    # ----------------------------------------------------
    NUM_WORKERS = 2   # <= adjust based on your RAM
    with Pool(NUM_WORKERS) as p:
        list(tqdm(p.imap(save_one_item, tasks), total=len(tasks)))



# ======================================================
# Export ALL subsets of a dataset
# ======================================================
def export_full_dataset(sets, name, output_root="data"):
    root = os.path.join(output_root, name)

    # ---------------- JPEG ----------------
    jpeg_root = os.path.join(root, "jpeg")

    # clean → quality=95
    export_dataset_variant(
        sets["train_clean"],
        os.path.join(jpeg_root, "train-clean"),
        "clean-jpeg",
        is_clean=True
    )

    # compressed → quality=50
    export_dataset_variant(
        sets["train_comb_jpeg"],
        os.path.join(jpeg_root, "train-combine"),
        "combine-jpeg",
        is_clean=False
    )

    export_dataset_variant(
        sets["val_clean"],
        os.path.join(jpeg_root, "validate-clean"),
        "clean-jpeg",
        is_clean=True
    )

    export_dataset_variant(
        sets["val_comb_jpeg"],
        os.path.join(jpeg_root, "validate-combine"),
        "combine-jpeg",
        is_clean=False
    )

    # ---------------- WEBP ----------------
    webp_root = os.path.join(root, "webp")

    export_dataset_variant(
        sets["train_clean"],
        os.path.join(webp_root, "train-clean"),
        "clean-webp",
        is_clean=True
    )

    export_dataset_variant(
        sets["train_comb_webp"],
        os.path.join(webp_root, "train-combine"),
        "combine-webp",
        is_clean=False
    )

    export_dataset_variant(
        sets["val_clean"],
        os.path.join(webp_root, "validate-clean"),
        "clean-webp",
        is_clean=True
    )

    export_dataset_variant(
        sets["val_comb_webp"],
        os.path.join(webp_root, "validate-combine"),
        "combine-webp",
        is_clean=False
    )

    print(f"\n🎉 Finished exporting {name}!\n")



# ======================================================
# Export transforms (NO normalize, no aug)
# ======================================================
def get_export_transforms(dataset_type):
    to_gray = [transforms.Grayscale(num_output_channels=3)] if dataset_type in ["xray", "nih"] else []

    tfm = transforms.Compose([
        transforms.Resize((224, 224)),
        *to_gray,
        transforms.ToTensor()
    ])
    return tfm, tfm


# ======================================================
# Main
# ======================================================
if __name__ == "__main__":
    print("\n🚀 Starting dataset extraction...\n")

    # -----------------------------------------------
    # Oxford Pet
    # -----------------------------------------------
    train_pet = datasets.OxfordIIITPet(PATH_OXFORD, split="trainval", target_types="category")
    val_pet   = datasets.OxfordIIITPet(PATH_OXFORD, split="test",     target_types="category")

    tfm_pet_train, tfm_pet_val = get_export_transforms("other")

    pet_sets = build_train_val_sets(train_pet, val_pet, tfm_pet_train, tfm_pet_val)
    export_full_dataset(pet_sets, "oxford-pet")

    # -----------------------------------------------
    # Food-101
    # -----------------------------------------------
    food_train = load_food101_split(PATH_FOOD, "train")
    food_val   = load_food101_split(PATH_FOOD, "test")

    food_train = food_train[:int(len(food_train)*0.1)]
    food_val   = food_val[:int(len(food_val)*0.1)]
    tfm_food_train, tfm_food_val = get_export_transforms("other")

    food_sets = build_train_val_sets(
        None, None,
        tfm_food_train, tfm_food_val,
        paths_train=food_train,
        paths_val=food_val
    )
    export_full_dataset(food_sets, "food-101")

    # -----------------------------------------------
    # Chest X-ray (IMPORTANT: needs grayscale→RGB)
    # -----------------------------------------------
    tfm_x_train, tfm_x_val = get_export_transforms("xray")

    x_train = datasets.ImageFolder(os.path.join(PATH_X_RAY, "train"))
    x_val   = datasets.ImageFolder(os.path.join(PATH_X_RAY, "test"))

    x_sets = build_train_val_sets(x_train, x_val, tfm_x_train, tfm_x_val)
    export_full_dataset(x_sets, "chest-xray")

    # # -----------------------------------------------
    # Osteosarcoma
    # -----------------------------------------------
    os_all = collect_osteosarcoma(PATH_OS)
    os_train, os_val = train_test_split(os_all, test_size=0.2, random_state=42)

    tfm_os_train, tfm_os_val = get_export_transforms("other")

    os_sets = build_train_val_sets(
        None, None,
        tfm_os_train, tfm_os_val,
        paths_train=os_train,
        paths_val=os_val
    )
    export_full_dataset(os_sets, "osteosarcoma")

    # -----------------------------------------------
    # Stanford Cars
    # -----------------------------------------------
    st_train = datasets.ImageFolder(os.path.join(PATH_STANFORD, "train"))
    st_val   = datasets.ImageFolder(os.path.join(PATH_STANFORD, "test"))

    tfm_st_train, tfm_st_val = get_export_transforms("other")

    st_sets = build_train_val_sets(st_train, st_val, tfm_st_train, tfm_st_val)
    export_full_dataset(st_sets, "stanford-cars")

    # # -----------------------------------------------
    # NIH CXR-8
    # -----------------------------------------------
    CSV_TRAIN = os.path.join(PATH_NIH, "csv", "train_clean.csv")
    CSV_VAL   = os.path.join(PATH_NIH, "csv", "val_clean.csv")
    IMG_ROOT  = os.path.join(PATH_NIH, "images")

    df_train = build_nih_df(CSV_TRAIN, IMG_ROOT)
    df_val   = build_nih_df(CSV_VAL, IMG_ROOT)

    df_train = df_train.sample(frac=0.1).reset_index(drop=True)
    df_val   = df_val.sample(frac=0.1).reset_index(drop=True)

    tfm_nih_train, tfm_nih_val = get_export_transforms("nih")

    nih_sets = build_train_val_sets(
        None, None,
        tfm_nih_train, tfm_nih_val,
        is_nih=True,
        paths_train=df_train,
        paths_val=df_val
    )
    export_full_dataset(nih_sets, "nih-cxr8")

    # print("\n🎉 ALL DATASETS EXPORTED SUCCESSFULLY! 🚀\n")
