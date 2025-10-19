import os, yaml, time, argparse
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report
from .datasets import FolderClassificationDataset, FishMultiHeadDataset
from .models import Classifier, MultiHeadFish
from .utils import set_seed, compute_metrics_cls

def train_cls(cfg):
    set_seed(cfg.get("seed", 42))
    device = "cuda" if torch.cuda.is_available() else "cpu"
    classes = cfg["data"]["classes"]
    img_size = cfg["data"]["img_size"]
    bs = cfg["train"]["batch_size"]
    nw = cfg["data"]["num_workers"]
    train_ds = FolderClassificationDataset(cfg["data"]["train_dir"], classes, img_size, True)
    val_ds   = FolderClassificationDataset(cfg["data"]["val_dir"],   classes, img_size, False)
    model = Classifier(cfg["model"]["backbone"], len(classes), cfg["model"]["pretrained"], cfg["model"]["dropout"]).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg["train"]["lr"], weight_decay=cfg["train"]["weight_decay"])
    loss_fn = nn.CrossEntropyLoss()
    best_f1, best_path = -1, f'best_{cfg["project"]}.pt'

    for epoch in range(cfg["train"]["epochs"]):
        model.train()
        tr_loader = DataLoader(train_ds, batch_size=bs, shuffle=True, num_workers=nw)
        for imgs, labels, _ in tr_loader:
            imgs, labels = imgs.to(device), labels.to(device)
            logits = model(imgs)
            loss = loss_fn(logits, labels)
            opt.zero_grad(); loss.backward(); opt.step()

        model.eval()
        va_loader = DataLoader(val_ds, batch_size=bs, shuffle=False, num_workers=nw)
        preds, gts = [], []
        with torch.no_grad():
            for imgs, labels, _ in va_loader:
                imgs = imgs.to(device)
                logits = model(imgs)
                pred = logits.argmax(1).cpu().numpy().tolist()
                gt = labels.numpy().tolist()
                preds += pred; gts += gt
        mets = compute_metrics_cls(gts, preds)
        print(f"Epoch {epoch+1}: Acc={mets['accuracy']:.3f} F1={mets['f1_macro']:.3f}")
        if mets['f1_macro'] > best_f1:
            best_f1 = mets['f1_macro']
            torch.save(model.state_dict(), best_path)
    print("Best F1:", best_f1, " saved:", best_path)

def train_fish(cfg):
    set_seed(cfg.get("seed", 42))
    device = "cuda" if torch.cuda.is_available() else "cpu"
    labels = cfg["labels"]
    img_size = cfg["data"]["img_size"]
    bs = cfg["train"]["batch_size"]
    nw = cfg["data"]["num_workers"]
    ds_tr = FishMultiHeadDataset(cfg["data"]["csv"], cfg["data"]["img_root"], labels, img_size, True)
    ds_va = FishMultiHeadDataset(cfg["data"]["csv"].replace('.csv','_val.csv'), cfg["data"]["img_root"], labels, img_size, False)
    model = MultiHeadFish(cfg["model"]["backbone"], len(labels["eye"]), len(labels["gill"]), len(labels["skin"]),
                          cfg["model"]["pretrained"], cfg["model"]["dropout"]).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg["train"]["lr"], weight_decay=cfg["train"]["weight_decay"])
    loss_fn = torch.nn.CrossEntropyLoss()
    best_f1, best_path = -1, f'best_{cfg["project"]}.pt'

    for epoch in range(cfg["train"]["epochs"]):
        model.train()
        tr_loader = DataLoader(ds_tr, batch_size=bs, shuffle=True, num_workers=nw)
        for imgs, targets, _ in tr_loader:
            imgs = imgs.to(device)
            eye = targets["eye"].to(device); gill = targets["gill"].to(device); skin = targets["skin"].to(device)
            out = model(imgs)
            loss = loss_fn(out["eye"], eye) + loss_fn(out["gill"], gill) + loss_fn(out["skin"], skin)
            opt.zero_grad(); loss.backward(); opt.step()

        model.eval()
        va_loader = DataLoader(ds_va, batch_size=bs, shuffle=False, num_workers=nw)
        preds, gts = [], []
        with torch.no_grad():
            for imgs, targets, _ in va_loader:
                imgs = imgs.to(device)
                out = model(imgs)
                pred_eye = out["eye"].argmax(1).cpu().numpy()
                pred_gill = out["gill"].argmax(1).cpu().numpy()
                pred_skin = out["skin"].argmax(1).cpu().numpy()
                # aggregate a simple majority for validation metric (placeholder)
                pred = (pred_eye + pred_gill + pred_skin) >= 2
                gt = (targets["eye"].numpy()==0) + (targets["gill"].numpy()==0) + (targets["skin"].numpy()==0)
                gt = (gt >= 2).astype(int)
                preds += pred.tolist(); gts += gt.tolist()
        from sklearn.metrics import f1_score, accuracy_score
        f1 = f1_score(gts, preds)
        acc = accuracy_score(gts, preds)
        print(f"Epoch {epoch+1}: Acc={acc:.3f} F1={f1:.3f}")
        if f1 > best_f1:
            best_f1 = f1
            torch.save(model.state_dict(), best_path)
    print("Best F1:", best_f1, " saved:", best_path)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True, type=str)
    args = ap.parse_args()
    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    if cfg["project"] in ("bread_mold","fruit_bruise"):
        train_cls(cfg)
    elif cfg["project"]=="fish_freshness":
        train_fish(cfg)
    else:
        raise SystemExit("Unknown project in config")

if __name__ == "__main__":
    main()
