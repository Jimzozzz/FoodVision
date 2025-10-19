import os, time, json, math, random
import numpy as np
import torch
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, roc_auc_score

def set_seed(seed: int=42):
    import random, os, numpy as np, torch
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)

def to_device(batch, device):
    if isinstance(batch, (list, tuple)):
        return [to_device(x, device) for x in batch]
    if isinstance(batch, dict):
        return {k: to_device(v, device) for k,v in batch.items()}
    if hasattr(batch, "to"): return batch.to(device)
    return batch

def compute_metrics_cls(y_true, y_pred, average="macro"):
    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, average=average)
    cm = confusion_matrix(y_true, y_pred).tolist()
    return {"accuracy": acc, "f1_macro": f1, "confusion_matrix": cm}
