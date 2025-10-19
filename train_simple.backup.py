import argparse, os, time
from pathlib import Path

import torch, torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models
from sklearn.metrics import f1_score, accuracy_score

def get_data(train_dir, val_dir, img_size, batch):
    tfm_train = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ColorJitter(0.2,0.2,0.2,0.1),
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
    train_dl = DataLoader(train_ds, batch_size=batch, shuffle=True,  num_workers=2)
    val_dl   = DataLoader(val_ds,   batch_size=batch, shuffle=False, num_workers=2)
    return train_ds, val_ds, train_dl, val_dl

def build_model(num_classes):
    # ใช้ ResNet18 (pretrained) เพื่อให้ติดตั้งง่าย
    try:
        weights = models.ResNet18_Weights.IMAGENET1K_V1
        model = models.resnet18(weights=weights)
    except Exception:
        model = models.resnet18(pretrained=True)
    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, num_classes)
    return model

def train_one_epoch(model, dl, device, opt, loss_fn):
    model.train()
    total = 0; running_loss = 0.0
    for xb, yb in dl:
        xb, yb = xb.to(device), yb.to(device)
        logits = model(xb)
        loss = loss_fn(logits, yb)
        opt.zero_grad(); loss.backward(); opt.step()
        running_loss += loss.item() * xb.size(0)
        total += xb.size(0)
    return running_loss/total

@torch.no_grad()
def evaluate(model, dl, device):
    model.eval()
    ys, ps = [], []
    for xb, yb in dl:
        xb = xb.to(device)
        logits = model(xb)
        pred = logits.argmax(1).cpu()
        ys.extend(yb.numpy().tolist())
        ps.extend(pred.numpy().tolist())
    acc = accuracy_score(ys, ps)
    f1  = f1_score(ys, ps, average="macro")
    return acc, f1

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train_dir", default="data/bread/train")
    ap.add_argument("--val_dir",   default="data/bread/val")
    ap.add_argument("--img_size",  type=int, default=384)
    ap.add_argument("--batch",     type=int, default=32)
    ap.add_argument("--epochs",    type=int, default=20)
    ap.add_argument("--lr",        type=float, default=3e-4)
    ap.add_argument("--out",       default="best_bread_mold.pt")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    train_ds, val_ds, train_dl, val_dl = get_data(args.train_dir, args.val_dir, args.img_size, args.batch)
    num_classes = len(train_ds.classes)
    print("Classes:", train_ds.classes)

    model = build_model(num_classes).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-2)
    loss_fn = nn.CrossEntropyLoss()

    best_f1, best_path = -1.0, args.out
    for epoch in range(1, args.epochs+1):
        tr_loss = train_one_epoch(model, train_dl, device, opt, loss_fn)
        acc, f1 = evaluate(model, val_dl, device)
        print(f"Epoch {epoch:02d} | train_loss={tr_loss:.4f} | val_acc={acc:.3f} | val_f1={f1:.3f}")
        if f1 > best_f1:
            best_f1 = f1
            torch.save(model.state_dict(), best_path)
    print(f"Best F1={best_f1:.3f}  saved: {best_path}")

if __name__ == "__main__":
    main()
