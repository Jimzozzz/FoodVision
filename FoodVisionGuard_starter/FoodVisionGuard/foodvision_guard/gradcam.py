import argparse, yaml, torch, cv2, numpy as np
from PIL import Image
from .models import Classifier
import timm

# Simple Grad-CAM for classifier's last conv feature map
def grad_cam(img_tensor, model, target_class=None):
    model.eval()
    feats = None
    grads = None
    def fwd_hook(module, inp, out):
        nonlocal feats; feats = out.detach()
    def bwd_hook(module, grad_in, grad_out):
        nonlocal grads; grads = grad_out[0].detach()

    # find last conv layer
    last_conv = None
    for n,m in model.backbone.named_modules():
        if isinstance(m, torch.nn.Conv2d):
            last_conv = m
    if last_conv is None:
        raise RuntimeError("No Conv2d found for Grad-CAM")

    h1 = last_conv.register_forward_hook(fwd_hook)
    h2 = last_conv.register_backward_hook(bwd_hook)

    logits = model(img_tensor)
    if target_class is None:
        target_class = logits.argmax(1).item()
    loss = logits[0, target_class]
    model.zero_grad()
    loss.backward()

    weights = grads.mean(dim=(2,3), keepdim=True)  # GAP over H,W
    cam = (weights * feats).sum(dim=1, keepdim=True)
    cam = torch.relu(cam)
    cam = cam.squeeze().cpu().numpy()
    cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
    h1.remove(); h2.remove()
    return cam

def overlay_heatmap(img_bgr, cam, alpha=0.35):
    cam_resized = cv2.resize(cam, (img_bgr.shape[1], img_bgr.shape[0]))
    heat = cv2.applyColorMap((cam_resized*255).astype(np.uint8), cv2.COLORMAP_JET)
    out = (alpha*heat + (1-alpha)*img_bgr).astype(np.uint8)
    return out

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--weights", default=None)
    ap.add_argument("--img", required=True)
    args = ap.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    classes = cfg["data"].get("classes", [])
    model = Classifier(cfg["model"]["backbone"], len(classes), pretrained=True, dropout=cfg["model"]["dropout"])
    if args.weights:
        model.load_state_dict(torch.load(args.weights, map_location="cpu"))

    # preprocess
    img = Image.open(args.img).convert("RGB")
    img_np = np.array(img)
    img_resized = cv2.resize(img_np, (cfg["data"]["img_size"], cfg["data"]["img_size"]))
    x = img_resized.astype(np.float32)/255.0
    x = (x - np.array([0.485,0.456,0.406]))/np.array([0.229,0.224,0.225])
    x = torch.from_numpy(x.transpose(2,0,1)).unsqueeze(0)

    cam = grad_cam(x, model)
    bgr = cv2.cvtColor(img_resized, cv2.COLOR_RGB2BGR)
    out = overlay_heatmap(bgr, cam, alpha=0.4)
    out_path = "gradcam_output.jpg"
    cv2.imwrite(out_path, out)
    print("Saved:", out_path)

if __name__ == "__main__":
    main()
