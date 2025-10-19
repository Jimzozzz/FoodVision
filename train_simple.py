# train_simple.py  (improved)
import argparse, os, time, random
from pathlib import Path
import numpy as np

import torch, torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models

# --------- Utils (no sklearn) ----------
def set_seed(seed: int = 42):
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)

def f1_macro_numpy(y_true, y_pred):
    y_true = np.asarray(y_true); y_pred = np.asarray(y_pred)
    classes = np.unique(np.concatenate([y_true, y_pred]))
    f1s = []
    for c in classes:
        tp = np.sum((y_true == c) & (y_pred == c))
        fp = np.sum((y_true != c) & (y_pred == c))
        fn = np.sum((y_true == c) & (y_pred != c))
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec  = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1   = 0.0 if (prec + rec) == 0 else 2 * prec * rec / (prec + rec)
        f1s.append(f1)
    return float(np.mean(f1s))

def accuracy_numpy(y_true, y_pred):
    y_true = np.asarray(y_true); y_pred = np.asarray(y_pred)
    return float(np.mean(y_true == y_pred))

# --------- Data ----------
def get_data(train_dir, val_dir, img_size, batch, num_workers=2):
    # stronger augmentation for train
    tfm_train = transforms.Compose([
        transforms.RandomResizedCrop((img_size, img_size), scale=(0.8, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(0.3, 0.3, 0.3, 0.15),
        transforms.RandomRotation(12),
        transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 1.5)),
        transforms.ToTensor(),
        transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225]),
    ])
    tfm_val = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225]),
    ])
    train_ds = datasets.ImageFolder(train_dir, tfm_train)
    val_ds   = datasets.ImageFolder(val_dir,   tfm_val)
    train_dl = DataLoader(train_ds, batch_size=batch, shuffle=True,  num_workers=num_workers)
    val_dl   = DataLoader(val_ds,   batch_size=batch, shuffle=False, num_workers=num_workers)
    return train_ds, val_ds, train_dl, val_dl

# --------- Model ----------
def build_model(num_classes: int, backbone: str = "resnet18"):
    backbone = backbone.lower()
    if backbone == "efficientnet_b0":
        try:
            weights = models.EfficientNet_B0_Weights.IMAGENET1K_V1
            model = models.efficientnet_b0(weights=weights)
        except Exception:
            model = models.efficientnet_b0(pretrained=True)
        in_features = model.classifier[1].in_features
        model.classifier[1] = nn.Linear(in_features, num_classes)
    else:  # resnet18 (default)
        try:
            weights = models.ResNet18_Weights.IMAGENET1K_V1
            model = models.resnet18(weights=weights)
        except Exception:
            model = models.resnet18(pretrained=True)
        in_features = model.fc.in_features
        model.fc = nn.Linear(in_features, num_classes)
    return model

# --------- Train / Eval ----------
def train_one_epoch(model, dl, device, opt, loss_fn):
    model.train()
    total, running_loss = 0, 0.0
    for xb, yb in dl:
        xb, yb = xb.to(device), yb.to(device)
        logits = model(xb)
        loss = loss_fn(logits, yb)
        opt.zero_grad(); loss.backward(); opt.step()
        running_loss += loss.item() * xb.size(0)
        total += xb.size(0)
    return running_loss / max(total, 1)

@torch.no_grad()
def evaluate(model, dl, device):
    model.eval()
    ys, ps = [], []
    for xb, yb in dl:
        xb = xb.to(device)
        logits = model(xb)
        pred = logits.argmax(1).cpu().numpy()
        ys.extend(yb.numpy()); ps.extend(pred)
    acc = accuracy_numpy(np.array(ys), np.array(ps))
    f1  = f1_macro_numpy(np.array(ys), np.array(ps))
    return acc, f1

# --------- Main ----------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train_dir", default="data/bread/train")
    ap.add_argument("--val_dir",   default="data/bread/val")
    ap.add_argument("--img_size",  type=int, default=384)
    ap.add_argument("--batch",     type=int, default=32)
    ap.add_argument("--epochs",    type=int, default=20)
    ap.add_argument("--lr",        type=float, default=3e-4)
    ap.add_argument("--weight_decay", type=float, default=1e-2)
    ap.add_argument("--backbone",  choices=["resnet18","efficientnet_b0"], default="resnet18")
    ap.add_argument("--label_smoothing", type=float, default=0.05)
    ap.add_argument("--early_stop_patience", type=int, default=5)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out",       default="best_bread_mold.pt")
    args = ap.parse_args()

    set_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    train_ds, val_ds, train_dl, val_dl = get_data(
        args.train_dir, args.val_dir, args.img_size, args.batch
    )
    classes = train_ds.classes
    num_classes = len(classes)
    print("Classes:", classes)

    model = build_model(num_classes, backbone=args.backbone).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    loss_fn = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)

    best_f1, best_path = -1.0, args.out
    patience_left = args.early_stop_patience

    for epoch in range(1, args.epochs + 1):
        tr_loss = train_one_epoch(model, train_dl, device, opt, loss_fn)
        acc, f1 = evaluate(model, val_dl, device)
        print(f"Epoch {epoch:02d} | train_loss={tr_loss:.4f} | val_acc={acc:.3f} | val_f1={f1:.3f}")

        improved = f1 > best_f1 + 1e-4
        if improved:
            best_f1 = f1
            torch.save(model.state_dict(), best_path)
            patience_left = args.early_stop_patience
        else:
            patience_left -= 1
            if patience_left <= 0:
                print(f"Early stopping at epoch {epoch} (best F1={best_f1:.3f})")
                break

    print(f"Best F1={best_f1:.3f}  saved: {best_path}")

if __name__ == "__main__":
    main()
