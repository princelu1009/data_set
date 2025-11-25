import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms, datasets, models
from torchvision.models import vit_b_16, ViT_B_16_Weights, MobileNet_V2_Weights
from sklearn.metrics import f1_score
from tqdm import tqdm
import pandas as pd

if torch.backends.mps.is_available():
    device = "mps"
elif torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"

print(f"🧠 Using device: {device}")


def get_transforms():
    train_tfms = transforms.Compose([
        transforms.ToTensor()
    ])
    val_tfms = transforms.Compose([
        transforms.ToTensor()
    ])
    return train_tfms, val_tfms


def get_model(model_name: str, num_classes: int) -> nn.Module:
    model_name = model_name.lower()
    if model_name == "mobilenet":
        model = models.mobilenet_v2(weights=MobileNet_V2_Weights.IMAGENET1K_V1)
        in_features = model.classifier[1].in_features
        model.classifier[1] = nn.Linear(in_features, num_classes)
        return model

    elif model_name == "vit":
        weights = ViT_B_16_Weights.IMAGENET1K_V1
        model = vit_b_16(weights=weights)
        in_features = model.heads.head.in_features
        model.heads.head = nn.Linear(in_features, num_classes)
        return model

    else:
        raise ValueError(f"Unknown model name: {model_name}")

def epoch_pass(model: nn.Module,
               loader: DataLoader,
               criterion: nn.Module,
               optimizer: optim.Optimizer | None = None):
    train_mode = optimizer is not None
    model.train(train_mode)

    total_loss = 0.0
    total_correct = 0
    total = 0
    all_y = []
    all_pred = []

    for x, y in tqdm(loader, leave=False, desc="Train" if train_mode else "Val"):
        x, y = x.to(device), y.to(device)

        if train_mode:
            optimizer.zero_grad()

        with torch.set_grad_enabled(train_mode):
            out = model(x)
            loss = criterion(out, y)

        if train_mode:
            loss.backward()
            optimizer.step()

        bs = x.size(0)
        total_loss += loss.item() * bs

        pred = out.argmax(1)
        total_correct += (pred == y).sum().item()
        total += bs

        all_y.extend(y.cpu().tolist())
        all_pred.extend(pred.cpu().tolist())

    avg_loss = total_loss / max(1, total)
    acc = total_correct / max(1, total)
    f1 = f1_score(all_y, all_pred, average="macro")

    return avg_loss, acc, f1

def load_dataset_split(data_root: str,
                       dataset_name: str,
                       fmt: str,
                       train_type: str,
                       train_tfms,
                       val_tfms):

    base = os.path.join(data_root, dataset_name, fmt)

    train_path = os.path.join(base, train_type)
    val_clean_path = os.path.join(base, "validate-clean")
    val_comp_path = os.path.join(base, "validate-combine")

    if not os.path.isdir(train_path):
        raise FileNotFoundError(f"Train path not found: {train_path}")
    if not os.path.isdir(val_clean_path):
        raise FileNotFoundError(f"Validate-clean path not found: {val_clean_path}")
    if not os.path.isdir(val_comp_path):
        raise FileNotFoundError(f"Validate-comp path not found: {val_comp_path}")

    train_set = datasets.ImageFolder(train_path, transform=train_tfms)
    val_clean = datasets.ImageFolder(val_clean_path, transform=val_tfms)
    val_comp = datasets.ImageFolder(val_comp_path, transform=val_tfms)

    num_classes = len(train_set.classes)
    return train_set, val_clean, val_comp, num_classes

def run_experiment(data_root: str,
                   dataset_name: str,
                   fmt: str,
                   train_type: str,
                   model_name: str,
                   num_classes: int,
                   train_tfms,
                   val_tfms,
                   epochs: int = 5,
                   batch_size: int = 32):


    tag = f"{dataset_name} | {fmt} | {train_type} | {model_name}"

    train_set, val_clean_set, val_comp_set, _ = load_dataset_split(
        data_root, dataset_name, fmt, train_type, train_tfms, val_tfms
    )

    num_workers = 0 if device == "mps" else 4

    train_loader = DataLoader(train_set, batch_size=batch_size,
                              shuffle=True, num_workers=num_workers)
    val_loader_clean = DataLoader(val_clean_set, batch_size=batch_size,
                                  shuffle=False, num_workers=num_workers)
    val_loader_comp = DataLoader(val_comp_set, batch_size=batch_size,
                                 shuffle=False, num_workers=num_workers)

    model = get_model(model_name, num_classes)
    model.to(device)

    criterion = nn.CrossEntropyLoss()
    # Slightly smaller LR for ViT
    lr = 3e-5 if model_name.lower() == "vit" else 1e-4
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)

    best_f1_comp = 0.0
    last_clean_f1 = 0.0
    last_comp_f1 = 0.0
    last_pi = 0.0

    for ep in range(1, epochs + 1):
        tr_loss, tr_acc, tr_f1 = epoch_pass(model, train_loader, criterion, optimizer)
        va_loss_c, va_acc_c, va_f1_c = epoch_pass(model, val_loader_clean, criterion)
        va_loss_x, va_acc_x, va_f1_x = epoch_pass(model, val_loader_comp, criterion)

        denom = 0.5 * (va_f1_c + va_f1_x) + 1e-8
        pi = 1.0 - abs(va_f1_c - va_f1_x) / denom

        print(
            f"[{tag}] "
            f"Epoch {ep:02d} | "
            f"TrainF1={tr_f1:.3f} | "
            f"ValCleanF1={va_f1_c:.3f} | "
            f"ValCompF1={va_f1_x:.3f} | "
            f"PI={pi:.3f}"
        )

        last_clean_f1 = va_f1_c
        last_comp_f1 = va_f1_x
        last_pi = pi

        if va_f1_x > best_f1_comp:
            best_f1_comp = va_f1_x
            ckpt_name = f"best_{model_name}_{dataset_name}_{fmt}_{train_type}.pt"
            torch.save(model.state_dict(), ckpt_name)

    return {
        "dataset": dataset_name,
        "format": fmt,
        "train_type": train_type,
        "model": model_name,
        "clean_f1": last_clean_f1,
        "comp_f1": last_comp_f1,
        "pi": last_pi,
    }


def main():
    SCRIPT_DIR_P = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    DATA_ROOT = os.path.join(SCRIPT_DIR_P,"data")
    ALL_DATASETS = [
        "chest-xray",
        "food-101",
        "nih-cxr8",
        'osteosarcoma',
        "stanford-cars",
        "oxford-pet",
    ]

    FORMATS = ["jpeg", "webp"]
    TRAIN_TYPES = ["train-clean", "train-combine"]
    MODELS = ["mobilenet", "vit"]   # run both

    train_tfms, val_tfms = get_transforms()
    results = []

    for dataset_name in ALL_DATASETS:
        print("\n" + "=" * 40)
        print(f"DATASET: {dataset_name}")
        print("=" * 40)

        # Detect num_classes from jpeg/train-clean
        _, _, _, num_classes = load_dataset_split(
            DATA_ROOT, dataset_name, "jpeg", "train-clean",
            train_tfms, val_tfms
        )
        for fmt in FORMATS:
            for tt in TRAIN_TYPES:
                for model_name in MODELS:
                    res = run_experiment(
                        DATA_ROOT, dataset_name, fmt, tt,
                        model_name, num_classes,
                        train_tfms, val_tfms,
                        epochs=5,    # adjust if you want longer/shorter training
                        batch_size=32
                    )
                    results.append(res)

    # Save all results to CSV
    df = pd.DataFrame(results)
    out_csv = "all_results_mobilenet_vit_pi_scores.csv"
    df.to_csv(out_csv, index=False)
    print(df)


if __name__ == "__main__":
    main()
