import argparse, numpy as np, torch
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader

def f1_macro_numpy(y_true, y_pred):
    y_true=np.asarray(y_true); y_pred=np.asarray(y_pred)
    classes=np.unique(np.concatenate([y_true,y_pred])); f1s=[]
    for c in classes:
        tp=((y_true==c)&(y_pred==c)).sum()
        fp=((y_true!=c)&(y_pred==c)).sum()
        fn=((y_true==c)&(y_pred!=c)).sum()
        prec=tp/(tp+fp) if (tp+fp)>0 else 0.0
        rec =tp/(tp+fn) if (tp+fn)>0 else 0.0
        f1 =0.0 if (prec+rec)==0 else 2*prec*rec/(prec+rec)
        f1s.append(f1)
    return float(np.mean(f1s))

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--test_dir", default="data/bread/test")
    ap.add_argument("--train_dir", default="data/bread/train")
    ap.add_argument("--weights",  default="best_bread_mold.pt")
    ap.add_argument("--img_size", type=int, default=384)
    args=ap.parse_args()

    tfm=transforms.Compose([
        transforms.Resize((args.img_size,args.img_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225]),
    ])

    train_ds=datasets.ImageFolder(args.train_dir); classes=train_ds.classes
    test_ds =datasets.ImageFolder(args.test_dir, tfm)
    dl=DataLoader(test_ds, batch_size=32, shuffle=False, num_workers=2)

    # ใช้สถาปัตยกรรมเดียวกับที่เทรนล่าสุด (EfficientNet-B0)
    try:
        w=models.EfficientNet_B0_Weights.IMAGENET1K_V1
        model=models.efficientnet_b0(weights=w)
    except Exception:
        model=models.efficientnet_b0(pretrained=True)
    model.classifier[1]=torch.nn.Linear(model.classifier[1].in_features, len(classes))
    model.load_state_dict(torch.load(args.weights, map_location="cpu")); model.eval()

    ys,ps=[],[]
    with torch.no_grad():
        for xb,yb in dl:
            pred=model(xb).argmax(1).numpy()
            ys.extend(yb.numpy()); ps.extend(pred)

    ys=np.array(ys); ps=np.array(ps)
    acc=float((ys==ps).mean()); f1=f1_macro_numpy(ys,ps)
    print("Classes:", classes)
    print(f"Test Acc={acc:.3f}  F1(macro)={f1:.3f}")

if __name__=="__main__":
    main()
